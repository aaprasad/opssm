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
    z (T,B,K) sampled points and log_q (T,B,K) the proposal log-density (the importance
    weights for the SNIS normalizer). In high-D this q is the importance-sampling lever
    that replaces the O(N^d) grid. `center` (T,B): the LATENT location to draw the near
    component around; default xs[...,0] (1D direct obs). For high-D obs y, pass a z-estimate
    (e.g. the pseudo-inverse C^+ (y-d)) since the obs itself is not in latent space."""
    T, B = xs.shape[0], xs.shape[1]
    dev = xs.device
    obs = mask[..., 0]                                          # (T,B) observed indicator
    center = (xs[..., 0] if center is None else center) * obs   # latent estimate where observed
    # "near" component: tight around the obs where observed. In the GAP there is no obs and
    # the posterior is BIMODAL (mass at the wells, not at 0), so widen the near component to
    # broad there -- otherwise half the samples pile up at z=0, exactly where the gap
    # posterior is empty, and the modes stay under-sampled.
    near_sd = (near_std * obs + broad_std * (1.0 - obs)).unsqueeze(-1)   # (T,B,1) per step
    Kn = n_colloc // 2
    Kb = n_colloc - Kn
    near = center.unsqueeze(-1) + near_sd * torch.randn(T, B, Kn, device=dev)
    broad = broad_std * torch.randn(T, B, Kb, device=dev)
    z = torch.cat([near, broad], dim=-1)                        # (T,B,K)
    c2 = math.log(2.0) + 0.5 * math.log(2 * math.pi)
    lqn = -0.5 * ((z - center.unsqueeze(-1)) / near_sd) ** 2 - near_sd.log() - c2
    lqb = -0.5 * (z / broad_std) ** 2 - math.log(broad_std) - c2
    log_q = torch.logaddexp(lqn, lqb)                           # log(0.5 q_near + 0.5 q_broad)
    return z, log_q


def pinn_zakai_loss(model, xs, mask, z_col, log_q, s_coll, drift, sigma, log_prior,
                    noise_std, dt, n_tcoll=None, res_post=0.0, decode=None):
    """MESH-FREE continuous-time Zakai PINN -- no grid, no time-stepping, no Euler.
    Collocation points z_col (T,B,K) are SAMPLED from the proposal (log_q its log-density);
    the normalizer Z and evidence c are self-normalized importance-sampling (SNIS) estimates
    over them; the drift f, f' are evaluated ANALYTICALLY at the samples (drift(z)->f,df).
        predict (PINN):  d_tau ell = -(f' + f d_z ell) + 1/2 g^2 ((d_z ell)^2 + d2_z ell)
        update  (jump):  pi_{i+1}(.,0) = normalize( lik_{i+1} * pi_i(.,dt) )   [SNIS]
        anchor  (IC):    pi_0(.,0)     = normalize( lik_0 * prior )            [SNIS]
    ell_i(z,s) = b(c_i,s).trunk(z); d_z ell = b.d_z tau, d_s ell = (d_s b).tau, all exact
    autodiff (jvp). pi(.,s) is SNIS weights W = softmax(ell - log_q). d_tau = d_s/dt.
    The recursion/SNIS terms use only the cheap basis tau (a forward, no autodiff) at ALL
    steps; the costly per-sample Jacobians (d_z, d2_z, d_s) for the FP residual are taken
    on a random subsample of n_tcoll steps (stochastic time collocation), so the residual's
    cost is decoupled from T. Returns (residual_loss, jump_loss, ic_loss, data_nll)."""
    T, B, K = z_col.shape
    ctx = model.context(xs, mask)                                  # (T,B,C)

    # ---- cheap path (no autodiff): basis at all steps; coeffs at the interval ends s=0,1
    tau = model.trunk(z_col.reshape(-1, 1)).reshape(T, B, K, -1)   # (T,B,K,p)
    b_ends = model.coeffs(ctx, torch.tensor([0.0, 1.0], device=z_col.device))   # (T,B,2,p)
    ell0 = torch.einsum("tbp,tbkp->tbk", b_ends[:, :, 0], tau) + model.bias   # post-update (s=0)

    if decode is None:                                                # 1D direct obs: h(z) = z
        loglik = -0.5 * (xs[..., 0].unsqueeze(-1) - z_col) ** 2 / noise_std ** 2   # (T,B,K)
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
    tau_s, dz_tau, d2z_tau = model.trunk_zderivs(z_col[ti])           # (Ts,B,K,p)
    dz_ell = torch.einsum("tbsp,tbkp->tbsk", b_s, dz_tau)
    d2z_ell = torch.einsum("tbsp,tbkp->tbsk", b_s, d2z_tau)
    ds_ell = torch.einsum("tbsp,tbkp->tbsk", ds_b, tau_s)
    f, df = drift(z_col[ti])                                          # analytic at samples
    f = f.unsqueeze(2); df = df.unsqueeze(2)                          # (Ts,B,1,K)
    if callable(sigma):                                              # state-dependent g^2(z)
        g2, dg2, d2g2 = sigma(z_col[ti])                             # (Ts,B,K) each
        g2 = g2.unsqueeze(2); dg2 = dg2.unsqueeze(2); d2g2 = d2g2.unsqueeze(2)
        # 1/2 d^2_z(g^2 rho) in log-space: 1/2 (g^2)'' + (g^2)' ell' + 1/2 g^2 (ell'^2 + ell'')
        rhs = (-(df + f * dz_ell)
               + 0.5 * d2g2 + dg2 * dz_ell + 0.5 * g2 * (dz_ell ** 2 + d2z_ell))
    else:                                                            # constant scalar g
        rhs = -(df + f * dz_ell) + 0.5 * sigma ** 2 * (dz_ell ** 2 + d2z_ell)
    res2 = (ds_ell / dt - rhs) ** 2                                   # (Ts,B,Ns,K)
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
                          w_nll=0.0, num_steps=1, res_post=0.0, decode=None, center=None):
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
            noise_std, dt, n_tcoll=n_tcoll, res_post=res_post, decode=decode)
        (bw * (res + jump + ic + w_nll * nll / num_steps)).backward()
        for i, v in enumerate((res, jump, ic, nll)):
            agg[i] += bw * v.item()
    return agg
