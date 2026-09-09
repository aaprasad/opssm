# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mesh-free continuous-time Zakai PINN loss + SNIS collocation."""

import math

import torch


def kl_target_pred(target, log_pred, eps=1e-12):
    """KL(target || pred), target a mass (T,B,Nz), log_pred the predicted log-mass."""
    t = target.clamp_min(eps)
    return (t * (t.log() - log_pred)).sum(dim=-1).mean()


def sample_collocation(xs, mask, n_colloc, near_std, broad_std, center=None):
    """MESH-FREE collocation: draw state points from a data-following proposal q (no grid,
    no zmax). Half are near the observation (the peaked observed posterior), half broad
    (covering the wells / the gap's bimodality where there is no observation). Returns
    z (T,B,K,d) sampled points and log_q (T,B,K) the ISOTROPIC-Gaussian-mixture proposal log-density
    (the SNIS weights). In high-D-LATENT this q is the importance-sampling lever that replaces the O(N^d)
    grid; ESS decays with d, so n_colloc must grow with d. `center` (T,B,d): the LATENT location for the
    near component; default = xs (direct obs, D=d). For a high-D linear sensor y pass a z-estimate (the
    pseudo-inverse (C^T C)^-1 C^T (y-d)) since the obs is not in latent space."""
    T, B = xs.shape[0], xs.shape[1]
    dev = xs.device
    ctr = xs if center is None else center                      # (T,B,d)
    d = ctr.shape[-1]
    obs = mask[..., 0]                                          # (T,B) observed indicator
    ctr = ctr * obs.unsqueeze(-1)                               # latent estimate where observed (T,B,d)
    # "near" component: tight around the obs where observed. In the GAP there is no obs and the posterior
    # is BIMODAL (mass at the wells, not at 0), so widen the near component to broad there.
    near_sd = (near_std * obs + broad_std * (1.0 - obs))        # (T,B) per step
    Kn = n_colloc // 2
    Kb = n_colloc - Kn
    near = ctr.unsqueeze(2) + near_sd[..., None, None] * torch.randn(T, B, Kn, d, device=dev)
    broad = broad_std * torch.randn(T, B, Kb, d, device=dev)
    z = torch.cat([near, broad], dim=2)                        # (T,B,K,d)
    off = math.log(2.0) + 0.5 * d * math.log(2 * math.pi)      # d-dim mixture normalizer
    lqn = (-0.5 * ((z - ctr.unsqueeze(2)) / near_sd[..., None, None]).pow(2).sum(-1)
           - d * near_sd[..., None].log() - off)               # (T,B,K)
    lqb = -0.5 * (z / broad_std).pow(2).sum(-1) - d * math.log(broad_std) - off
    log_q = torch.logaddexp(lqn, lqb)                          # log(0.5 q_near + 0.5 q_broad), (T,B,K)
    return z, log_q                                            # (T,B,K,d), (T,B,K)


def pinn_zakai_loss(model, xs, mask, z_col, log_q, s_coll, drift, sigma, log_prior,
                    noise_std, dt, n_tcoll=None, res_post=0.0, res_mode="l2", decode=None):
    """MESH-FREE continuous-time Zakai PINN -- no grid, no time-stepping, no Euler.
    Collocation points z_col (T,B,K) are SAMPLED from the proposal (log_q its log-density);
    the normalizer Z and evidence c are self-normalized importance-sampling (SNIS) estimates
    over them; the drift f, f' are evaluated ANALYTICALLY at the samples (drift(z)->f,df).
        predict (PINN):  d_tau ell = -(f' + f d_z ell) + 1/2 g^2 ((d_z ell)^2 + d2_z ell)
        update  (jump):  pi_{i+1}(.,0) = normalize( lik_{i+1} * pi_i(.,dt) )   [SNIS]
        anchor  (IC):    pi_0(.,0)     = normalize( lik_0 * prior )            [SNIS]
    ell_i(z,s) = b(c_i,s).state_basis(z); d_z ell = b.d_z tau, d_s ell = (d_s b).tau, all exact
    autodiff (jvp). pi(.,s) is SNIS weights W = softmax(ell - log_q). d_tau = d_s/dt.
    The recursion/SNIS terms use only the cheap basis tau (a forward, no autodiff) at ALL
    steps; the costly per-sample Jacobians (d_z, d2_z, d_s) for the FP residual are taken
    on a random subsample of n_tcoll steps (stochastic time collocation), so the residual's
    cost is decoupled from T. Returns (residual_loss, jump_loss, ic_loss, data_nll)."""
    T, B, K, d = z_col.shape
    ctx = model.context(xs, mask)                                  # (T,B,C)

    # ---- cheap path (no autodiff): basis at all steps; coeffs at the interval ends s=0,1
    tau = model.state_basis(z_col.reshape(-1, d)).reshape(T, B, K, -1)   # (T,B,K,p)
    b_ends = model.coeffs(ctx, torch.tensor([0.0, 1.0], device=z_col.device))   # (T,B,2,p)
    ell0 = torch.einsum("tbp,tbkp->tbk", b_ends[:, :, 0], tau) + model.bias   # post-update (s=0)

    if decode is None:                                                # direct obs h(z) = z (D = d)
        loglik = -0.5 * ((xs.unsqueeze(2) - z_col) ** 2).sum(-1) / noise_std ** 2   # (T,B,K)
    else:                                                             # high-D obs: lik = N(y; h(z), sigma^2 I)
        loglik = -0.5 * ((xs.unsqueeze(2) - decode(z_col)) ** 2).sum(-1) / noise_std ** 2   # (T,B,K)
    m = mask[..., 0]                                                   # (T,B)
    logZ0 = torch.logsumexp(ell0 - log_q, dim=-1, keepdim=True) - math.log(K)
    logpi0 = ell0 - logZ0                                              # normalized log-density

    # predict pi_i(.,dt) as SNIS weights, EVALUATED on step (i+1)'s samples (shared support
    # for the bootstrap): ell of step i (s=1) at z_col[i+1] = b_i(.,1) . trunk(z_col[i+1]).
    ellT_next = torch.einsum("tbp,tbkp->tbk", b_ends[:-1, :, 1], tau[1:]) + model.bias
    logW_pred = torch.log_softmax(ellT_next - log_q[1:], dim=-1).detach()
    logW_tgt = torch.log_softmax(logW_pred + loglik[1:] * m[1:].unsqueeze(-1), dim=-1)
    jump_loss = -(logW_tgt.exp() * logpi0[1:]).sum(-1).mean()
    logW_ic = torch.log_softmax(
        log_prior(z_col[0]) + loglik[0] * m[0].unsqueeze(-1) - log_q[0], dim=-1)
    ic_loss = -(logW_ic.exp() * logpi0[0]).sum(-1).mean()
    logc = torch.logsumexp(logW_pred + loglik[1:], dim=-1)            # evidence c_{i+1}
    nll = -(logc * m[1:]).sum(0).mean()

    # ---- residual path (autodiff) on a random time subsample
    ti = (torch.randperm(T, device=z_col.device)[:n_tcoll] if n_tcoll and n_tcoll < T
          else torch.arange(T, device=z_col.device))
    b_s, ds_b = model.coeffs_dtime(ctx[ti], s_coll)                   # (Ts,B,Ns,p)
    tau_s, grad_tau, lap_tau = model.trunk_zderivs(z_col[ti])         # (Ts,B,K,p),(...,K,d,p),(...,K,p)
    grad_ell = torch.einsum("tbsp,tbkdp->tbskd", b_s, grad_tau)       # (Ts,B,Ns,K,d)  grad ell
    grad_ell_sq = grad_ell.pow(2).sum(-1)                             # (Ts,B,Ns,K)    |grad ell|^2
    lap_ell = torch.einsum("tbsp,tbkp->tbsk", b_s, lap_tau)           # (Ts,B,Ns,K)    Laplacian ell
    ds_ell = torch.einsum("tbsp,tbkp->tbsk", ds_b, tau_s)
    f, div_f = drift(z_col[ti])                                       # f (Ts,B,K,d), div f (Ts,B,K)
    f_dot = torch.einsum("tbskd,tbkd->tbsk", grad_ell, f)            # f . grad ell  (Ts,B,Ns,K)
    divf = div_f.unsqueeze(2)                                        # (Ts,B,1,K)
    if callable(sigma):                                             # state-dependent g^2(z) -- 1-D only (g_net)
        g2, dg2, d2g2 = sigma(z_col[ti])                            # (Ts,B,K) each (DiffusionNet.diffusion is 1-D)
        g2 = g2.unsqueeze(2); dg2 = dg2.unsqueeze(2); d2g2 = d2g2.unsqueeze(2)
        dz_ell = grad_ell[..., 0]                                   # d==1 gradient component
        # 1/2 d^2_z(g^2 rho) in log-space: 1/2 (g^2)'' + (g^2)' ell' + 1/2 g^2 (ell'^2 + ell'')
        rhs = (-(divf + f_dot)
               + 0.5 * d2g2 + dg2 * dz_ell + 0.5 * g2 * (grad_ell_sq + lap_ell))
    else:                                                          # constant scalar g (isotropic D = g^2 I)
        # d-D FP in log-space: -(div f + f . grad ell) + 1/2 g^2 (|grad ell|^2 + Laplacian ell)
        rhs = -(divf + f_dot) + 0.5 * sigma ** 2 * (grad_ell_sq + lap_ell)
    res2 = (ds_ell / dt - rhs) ** 2                                   # (Ts,B,Ns,K)
    # SCALE-INVARIANT residual (over-dispersion fix): in LOG space the FP terms scale as ~1/sigma^2,
    # so for a SHARP density the L2 residual EXPLODES at collocation samples far from the mode
    # (d_z ell ~ z/sigma^2). Minimizing that absolute magnitude biases the operator WIDE. Measuring
    # the residual RELATIVE to the local FP magnitude (or log-compressing it) removes the tail blow-up
    # while keeping a broad proposal for coverage. "l2" = original absolute residual.
    if res_mode == "rel":                                             # relative to local FP scale
        scale = rhs.detach() ** 2 + (ds_ell / dt).detach() ** 2 + 1.0
        res2 = res2 / scale
    elif res_mode == "log1p":                                         # log-compress large residuals
        res2 = torch.log1p(res2)
    if res_post > 0:
        # weight the FP residual by the posterior mass: enforce the physics WHERE THE DENSITY
        # IS (the modes) rather than uniformly over the broad proposal -- targets the gap
        # over-dispersion. w = (1-res_post)/K + res_post * softmax(ell - log_q), sums to 1.
        W = torch.softmax(ell0[ti] - log_q[ti], dim=-1).detach()      # (Ts,B,K) posterior weights
        w = (1.0 - res_post) / K + res_post * W
        res_loss = (w.unsqueeze(2) * res2).sum(-1).mean()
    else:
        res_loss = res2.mean()
    return res_loss, jump_loss, ic_loss, nll


def accumulate_pinn_grads(model, xs, mask, s_coll, drift, sigma, log_prior, noise_std, dt,
                          n_colloc, near_std, broad_std, n_tcoll, chunk_size,
                          w_nll=0.0, num_steps=1, res_post=0.0, res_mode="l2", w_res=1.0,
                          decode=None, center=None):
    """MEMORY-CAPPED forward+backward of the mesh-free Zakai PINN loss. The recursion is
    per-trajectory and the residual is per-point, both INDEPENDENT across the batch, so we
    split the batch into chunks of `chunk_size`, run pinn_zakai_loss + backward per chunk
    (freeing each chunk's autodiff graph before the next), and accumulate gradients. Peak
    memory scales with chunk_size rather than the full batch, and the result is EXACT (a
    batch-mean is the size-weighted mean of chunk-means). Set w_nll>0 to add the data-NLL
    term w_nll * NLL/num_steps (Stage 3, learn dynamics). Returns the batch-averaged
    (res, jump, ic, nll) scalars for logging; gradients are left accumulated on the params
    (caller does zero_grad before / step after)."""
    B = xs.shape[1]
    agg = [0.0, 0.0, 0.0, 0.0]
    for st in range(0, B, chunk_size):
        sl = slice(st, min(st + chunk_size, B))
        bw = (sl.stop - sl.start) / B                       # chunk's batch fraction
        ctr = None if center is None else center[:, sl]
        z_col, log_q = sample_collocation(xs[:, sl], mask[:, sl], n_colloc, near_std, broad_std,
                                          center=ctr)
        res, jump, ic, nll = pinn_zakai_loss(
            model, xs[:, sl], mask[:, sl], z_col, log_q, s_coll, drift, sigma, log_prior,
            noise_std, dt, n_tcoll=n_tcoll, res_post=res_post, res_mode=res_mode, decode=decode)
        # w_res down-weights the FP RESIDUAL relative to the jump/ic recursion: the residual SPREADS
        # and the likelihood update (jump) SHARPENS, so w_res<1 sharpens the balance toward the exact
        # filter (w_res=0 collapses to a spike -- keep it > 0 to still enforce the FP dynamics).
        (bw * (w_res * res + jump + ic + w_nll * nll / num_steps)).backward()
        for i, v in enumerate((res, jump, ic, nll)):
            agg[i] += bw * v.item()
    return agg


def pinn_adjoint_loss(model_b, xs, mask, z_col, log_q, s_coll, drift, sigma,
                      noise_std, dt, n_tcoll=None, res_mode="rel", decode=None):
    """BACKWARD adjoint-Zakai PINN for the smoother's log post-update backward MESSAGE
    lmsg = log msg_t(z), msg_t(z) = p(y_{t:T} | z_t). Time-reversed mirror of pinn_zakai_loss:
        adjoint residual (s=0 at obs t, s=1 backward toward obs t-1 -- the backward Kolmogorov
        GENERATOR on log msg):
            d_s lmsg / dt = f * d_z lmsg + 0.5 g^2 ((d_z lmsg)^2 + d2_z lmsg)
            [vs the forward rhs -(f' + f d_z ell) + 0.5 g^2(...): DROP the f' term, FLIP -f d_z -> +f d_z,
             diffusion unchanged (+); no global time-sign flip. State-dependent g DROPS the forward's
             d2g2/dg2 terms (generator, not the FP adjoint).]
        backward jump (i -> i-1):  msg_{i-1}(s=0) = normalize( backward-predict_i(s=1) * lik_{i-1} )
        terminal (i = T-1):        msg_{T-1} = normalize( lik_{T-1} )   [beta_{T-1}=1, no future]
    Returns (res_b, jump_b, tc_b). Reuses sample_collocation's z_col/log_q, res_mode, and (via
    accumulate_adjoint_grads) w_res. The smoothed readout is gamma_t proportional to
    alpha_t * msg_t / lik_t (see mstep.log_smoothed)."""
    T, B, K, d = z_col.shape
    ctx = model_b.context(xs, mask)                                # (T,B,C) anti-causal
    tau = model_b.state_basis(z_col.reshape(-1, d)).reshape(T, B, K, -1)  # (T,B,K,p)
    b_ends = model_b.coeffs(ctx, torch.tensor([0.0, 1.0], device=z_col.device))   # (T,B,2,p)
    lmsg0 = torch.einsum("tbp,tbkp->tbk", b_ends[:, :, 0], tau) + model_b.bias     # (T,B,K) msg, s=0

    if decode is None:                                             # direct obs h(z) = z (D = d)
        loglik = -0.5 * ((xs.unsqueeze(2) - z_col) ** 2).sum(-1) / noise_std ** 2   # (T,B,K)
    else:                                                          # high-D: lik = N(y; C z + d, sigma^2 I)
        loglik = -0.5 * ((xs.unsqueeze(2) - decode(z_col)) ** 2).sum(-1) / noise_std ** 2
    m = mask[..., 0]                                               # (T,B)
    logZ0 = torch.logsumexp(lmsg0 - log_q, dim=-1, keepdim=True) - math.log(K)
    lmsgpi0 = lmsg0 - logZ0                                        # normalized (mirror of logpi0)

    # ---- backward jump: step i's s=1 (backward-transported) message on step (i-1)'s nodes, x lik_{i-1}
    bpred = torch.einsum("tbp,tbkp->tbk", b_ends[1:, :, 1], tau[:-1]) + model_b.bias   # (T-1,B,K)
    logW_bpred = torch.log_softmax(bpred - log_q[:-1], dim=-1).detach()
    logW_btgt = torch.log_softmax(logW_bpred + loglik[:-1] * m[:-1].unsqueeze(-1), dim=-1)
    jump_b = -(logW_btgt.exp() * lmsgpi0[:-1]).sum(-1).mean()      # teach step (i-1)'s s=0

    # ---- terminal condition: msg_{T-1} = normalize(lik_{T-1}) (no future beyond the last obs)
    logW_tc = torch.log_softmax(loglik[-1] * m[-1].unsqueeze(-1) - log_q[-1], dim=-1)
    tc_b = -(logW_tc.exp() * lmsgpi0[-1]).sum(-1).mean()

    # ---- adjoint residual (autodiff on a random time subsample), same machinery as the forward
    ti = (torch.randperm(T, device=z_col.device)[:n_tcoll] if n_tcoll and n_tcoll < T
          else torch.arange(T, device=z_col.device))
    b_s, ds_b = model_b.coeffs_dtime(ctx[ti], s_coll)             # (Ts,B,Ns,p)
    tau_s, grad_tau, lap_tau = model_b.trunk_zderivs(z_col[ti])   # (Ts,B,K,p),(...,K,d,p),(...,K,p)
    grad_lm = torch.einsum("tbsp,tbkdp->tbskd", b_s, grad_tau)   # (Ts,B,Ns,K,d)
    grad_lm_sq = grad_lm.pow(2).sum(-1)                          # (Ts,B,Ns,K)
    lap_lm = torch.einsum("tbsp,tbkp->tbsk", b_s, lap_tau)       # (Ts,B,Ns,K)
    ds_lm = torch.einsum("tbsp,tbkp->tbsk", ds_b, tau_s)
    f, _ = drift(z_col[ti])                                       # backward GENERATOR uses f (not div f)
    f_dot = torch.einsum("tbskd,tbkd->tbsk", grad_lm, f)         # f . grad lmsg  (Ts,B,Ns,K)
    if callable(sigma):                                          # state-dependent g^2(z) -- 1-D only (g_net)
        g2 = sigma(z_col[ti])[0].unsqueeze(2)
        rhs = f_dot + 0.5 * g2 * (grad_lm_sq + lap_lm)           # NO d2g2/dg2 (generator, not FP adjoint)
    else:                                                        # constant scalar g (isotropic)
        # d-D backward generator: f . grad lmsg + 1/2 g^2 (|grad lmsg|^2 + Laplacian lmsg)
        rhs = f_dot + 0.5 * sigma ** 2 * (grad_lm_sq + lap_lm)
    res2 = (ds_lm / dt - rhs) ** 2                                # (Ts,B,Ns,K)
    if res_mode == "rel":                                         # scale-invariant residual (as forward)
        scale = rhs.detach() ** 2 + (ds_lm / dt).detach() ** 2 + 1.0
        res2 = res2 / scale
    elif res_mode == "log1p":
        res2 = torch.log1p(res2)
    res_b = res2.mean()
    return res_b, jump_b, tc_b


def accumulate_adjoint_grads(model_b, xs, mask, s_coll, drift, sigma, noise_std, dt,
                             n_colloc, near_std, broad_std, n_tcoll, chunk_size,
                             res_mode="rel", w_res=1.0, decode=None, center=None):
    """Memory-capped chunked forward+backward of the backward adjoint-Zakai loss (sibling of
    accumulate_pinn_grads). Returns the batch-averaged (res_b, jump_b, tc_b); gradients left on
    model_b's params (caller does zero_grad before / step after)."""
    B = xs.shape[1]
    agg = [0.0, 0.0, 0.0]
    for st in range(0, B, chunk_size):
        sl = slice(st, min(st + chunk_size, B))
        bw = (sl.stop - sl.start) / B
        ctr = None if center is None else center[:, sl]
        z_col, log_q = sample_collocation(xs[:, sl], mask[:, sl], n_colloc, near_std, broad_std,
                                          center=ctr)
        res_b, jump_b, tc_b = pinn_adjoint_loss(
            model_b, xs[:, sl], mask[:, sl], z_col, log_q, s_coll, drift, sigma,
            noise_std, dt, n_tcoll=n_tcoll, res_mode=res_mode, decode=decode)
        (bw * (w_res * res_b + jump_b + tc_b)).backward()
        for i, v in enumerate((res_b, jump_b, tc_b)):
            agg[i] += bw * v.item()
    return agg
