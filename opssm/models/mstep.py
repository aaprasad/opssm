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

"""EM M-step: fit the SDE coefficients (drift f, diffusion g) and the high-D observation map
(C, d) from the operator's inferred latent path. Deduplicated across the 1-D and high-D cases:
1-D is the degenerate `cstab == 0` instance of the high-D sensor-before-dynamics curriculum, so
the drift gate is written once. The fits are the strong "complete-data" channel:
    f(z)   = E[Delta z | z] / dt           (conditional mean of increments)
    g^2(z) = Var[Delta z | z] / dt          (conditional variance, trapezoidal-drift residual)
    C      = unit direction of cross-cov(y, z_hat)   (orthogonal Procrustes; scale s fixed)
"""

import math

import torch

from opssm.models.obs import zhat_from_obs


@torch.no_grad()
def posterior_mean(model, x, mask, z_grid):
    """E[z_t | obs_{0:t}] from the operator's grid log-posterior -> (T, B). Grid quadrature
    (O(Nz) per step); see posterior_mean_fixed for the mesh-free replacement."""
    log_pi = model.log_posterior(x, mask, z_grid)
    return (log_pi.exp() * z_grid).sum(-1)


def _fixed_nodes(center, mask, n_samples, near_std, broad_std):
    """FIXED-seed CRN proposal nodes shared across t -- the deterministic-readout building block reused
    by posterior_mean_fixed / smoother_mean_fixed / smoother_pair_fixed. Half near the data-following
    center (near_std), half broad (broad_std); nodes are per-t via the near center but the seed is fixed
    so they are reused every M-step (no fresh randomness). Returns (z (T,B,K), log_q (T,B,K))."""
    T, B = center.shape
    dev = center.device
    gen = torch.Generator(device=dev).manual_seed(0)                 # FIXED nodes -> deterministic readout
    obs = mask[..., 0]
    c = center * obs
    near_sd = (near_std * obs + broad_std * (1.0 - obs)).unsqueeze(-1)
    Kn = n_samples // 2
    Kb = n_samples - Kn
    eps_n = torch.randn(1, B, Kn, device=dev, generator=gen)         # shared across t (CRN) + fixed seed
    eps_b = torch.randn(1, B, Kb, device=dev, generator=gen)
    near = c.unsqueeze(-1) + near_sd * eps_n
    broad = (broad_std * eps_b).expand(T, B, Kb)
    z = torch.cat([near, broad], dim=-1)                            # (T,B,K)
    c2 = math.log(2.0) + 0.5 * math.log(2 * math.pi)
    lqn = -0.5 * ((z - c.unsqueeze(-1)) / near_sd) ** 2 - near_sd.log() - c2
    lqb = -0.5 * (z / broad_std) ** 2 - math.log(broad_std) - c2
    log_q = torch.logaddexp(lqn, lqb)
    return z, log_q


@torch.no_grad()
def posterior_mean_fixed(model, x, mask, center, n_samples, near_std, broad_std):
    """DETERMINISTIC mesh-free FILTER MEAN via FIXED-NODE importance sampling: the proposal nodes are
    drawn ONCE (fixed seed) and reused every M-step. So z_hat = sum_k w_k z_k is a smooth deterministic
    function of the operator -- the grid's deterministic mean, but with data-following nodes instead of a
    uniform grid. No fresh per-M-step randomness => no readout noise (the SNIS blowup was the fresh
    per-step sampling noise; fixing the nodes removes it while keeping the MEAN, unlike the mode).
    Returns (z_hat (T,B), ess_frac)."""
    z, log_q = _fixed_nodes(center, mask, n_samples, near_std, broad_std)
    ctx = model.context(x, mask)
    b0 = model.coeffs(ctx, torch.zeros(1, device=z.device))[:, :, 0]  # (T,B,p) at s=0
    tau = model.trunk(z.unsqueeze(-1))                              # (T,B,K,p)
    ell = torch.einsum("tbp,tbkp->tbk", b0, tau) + model.bias       # (T,B,K)
    w = torch.softmax(ell - log_q, dim=-1)                          # SNIS weights
    ess = 1.0 / (w.pow(2).sum(-1) * z.shape[-1])                    # (T,B) fraction
    return (w * z).sum(-1), float(ess.mean())                       # (T,B), scalar


@torch.no_grad()
def smoother_mean_fixed(model, model_b, x, mask, center, n_samples, near_std, broad_std):
    """DETERMINISTIC mesh-free SMOOTHER MEAN E[z_t | y_{0:T}] on the same fixed-node CRN proposal, with
    the numerically STABLE predict*msg smoother weights
        W_k = softmax_k( log predict_t(z_k) + log msg_t(z_k) - log q_k ),
      log predict_t = forward operator s=1 of step t-1 evaluated on node_t (= log p(z_t|y_{0:t-1}), smooth);
      log msg_t     = backward operator s=0 (= log p(y_{t:T}|z_t)).
    Unlike the FILTER mean, the smoother mean is a proper state estimate, so the midpoint/trapezoidal drift
    correction becomes valid on its increments (v2 M-step). Returns (z_hat (T,B), ess_frac)."""
    z, log_q = _fixed_nodes(center, mask, n_samples, near_std, broad_std)
    dev = z.device
    ctx_f = model.context(x, mask)
    ctx_b = model_b.context(x, mask)
    tau_f = model.trunk(z.unsqueeze(-1))                            # (T,B,K,p) forward basis on nodes
    tau_b = model_b.trunk(z.unsqueeze(-1))                          # (T,B,K,p) backward basis (own weights)
    b1_f = model.coeffs(ctx_f, torch.ones(1, device=dev))[:, :, 0]  # (T,B,p) forward s=1 coeffs
    log_pred = torch.empty_like(z)                                  # (T,B,K)
    log_pred[0] = -0.5 * z[0] ** 2                                  # predict_0 = prior N(0,1) on node_0
    log_pred[1:] = torch.einsum("tbp,tbkp->tbk", b1_f[:-1], tau_f[1:]) + model.bias  # predict_t on node_t
    b0_b = model_b.coeffs(ctx_b, torch.zeros(1, device=dev))[:, :, 0]   # (T,B,p) backward s=0
    lmsg = torch.einsum("tbp,tbkp->tbk", b0_b, tau_b) + model_b.bias    # (T,B,K) log msg_t on node_t
    w = torch.softmax(log_pred + lmsg - log_q, dim=-1)
    ess = 1.0 / (w.pow(2).sum(-1) * z.shape[-1])
    return (w * z).sum(-1), float(ess.mean())


@torch.no_grad()
def smoother_pair_fixed(model, model_b, x, mask, center, drift_net, g_cur, dt,
                        n_samples, near_std, broad_std):
    """Deterministic fixed-node CRN LAG-ONE joint p(z_t, z_{t+1} | y_{0:T}) -> the cross-covariance that
    fixes the diffusion g. Proposal: z_t^k ~ q_t (fixed nodes); z_{t+1}^k = z_t^k + f(z_t^k) dt +
    sqrt(g^2 dt) eps^k (fixed eps^k = the transition K, which CANCELS in the importance weight). The
    two-filter lag-one joint is proportional to alpha_t(z_t) K(z_{t+1}|z_t) msg_{t+1}(z_{t+1}), so
        W_k = softmax_k( log alpha_t(z_t^k) - log q_t^k + log msg_{t+1}(z_{t+1}^k) )
    (alpha_t = forward s=0; msg_{t+1} = backward s=0 -- INCLUDES lik_{t+1}, Convention B). Data-local nodes,
    so no tail blow-up. Returns (zt, zt1, W) each (T-1,B,K): joint samples + weights for fit_diffusion's
    square-then-average g^2 = (1/dt) mean_pairs sum_k W_k (Δz^k - 1/2(f_k+f'_k) dt)^2. CRN (fixed nodes +
    fixed eps) makes this g deterministic across M-steps (no stochastic-FFBS runaway)."""
    z, log_q = _fixed_nodes(center, mask, n_samples, near_std, broad_std)   # (T,B,K)
    dev = z.device
    gen = torch.Generator(device=dev).manual_seed(1)                       # FIXED transition noise (CRN)
    eps = torch.randn(1, z.shape[1], z.shape[2], device=dev, generator=gen)  # shared across t
    f_z = drift_net.drift(z)[0]                                            # (T,B,K) drift on nodes
    z_next = z + f_z * dt + math.sqrt(max(float(g_cur), 1e-6) ** 2 * dt) * eps  # (T,B,K) ~ K(.|z)
    zt, zt1 = z[:-1], z_next[:-1]                                          # pair (t, t+1) samples
    ctx_f = model.context(x, mask); ctx_b = model_b.context(x, mask)
    b0_f = model.coeffs(ctx_f, torch.zeros(1, device=dev))[:, :, 0]        # (T,B,p) forward s=0
    b0_b = model_b.coeffs(ctx_b, torch.zeros(1, device=dev))[:, :, 0]      # (T,B,p) backward s=0
    l_alpha = torch.einsum("tbp,tbkp->tbk", b0_f[:-1], model.trunk(zt.unsqueeze(-1))) + model.bias
    l_msg1 = torch.einsum("tbp,tbkp->tbk", b0_b[1:], model_b.trunk(zt1.unsqueeze(-1))) + model_b.bias
    W = torch.softmax((l_alpha - log_q[:-1]) + l_msg1, dim=-1)             # (T-1,B,K)
    return zt, zt1, W


@torch.no_grad()
def log_smoothed(model, model_b, x, mask, z_grid):
    """Neural SMOOTHER marginal log p(z_t | y_{0:T}) on a grid, via the numerically STABLE predict*msg
    readout:  log gamma_t = log predict_t + log msg_t, normalized over z.
      predict_t = the FORWARD operator's FP-predicted density (s=1) carried from step t-1 -- i.e.
                  log p(z_t | y_{0:t-1}), SMOOTH in z (no observation applied), = the s=1 slice of step t-1.
      msg_t     = the BACKWARD operator's post-update message (s=0) = log p(y_{t:T} | z_t) (INCLUDES lik_t).
    Since alpha_t = predict_t * lik_t, predict*msg = predict*lik*beta = alpha*beta = gamma -- but the
    equivalent alpha*msg/lik form is a numerical trap (the -loglik reaches ~+1/(2 noise_std^2) in the tails
    and overwhelms any finite floor on log alpha, flushing all smoothed mass to the tails). predict is
    smooth, so log predict + log msg has no such cancellation. t=0 uses the Gaussian prior for predict_0."""
    ctx_f = model.context(x, mask)
    ctx_b = model_b.context(x, mask)
    ell_pred = model.log_density(ctx_f, z_grid, s=1.0)             # (T,B,Nz) forward s=1 = log predict_{t+1}
    log_pred = torch.empty_like(ell_pred)
    log_pred[0] = -0.5 * z_grid ** 2                              # predict_0 = prior N(0,1) (const cancels)
    log_pred[1:] = ell_pred[:-1]                                  # predict_t = step (t-1) s=1
    lmsg = model_b.log_density(ctx_b, z_grid, s=0.0)             # (T,B,Nz) log msg_t
    log_g = log_pred + lmsg
    return log_g - torch.logsumexp(log_g, dim=-1, keepdim=True)


def fit_drift(drift_net, dr_opt, zc, dz, z_reg, hr, reg_lambda, m_inner,
              zc_next=None, method="euler"):
    """Regress f_theta ~ dz (the mean increment / dt) with an H2 (curvature) smoothness penalty.
    `method` sets the quadrature relating f to the increment:
      'euler'       left-endpoint  f(z_t) ~ dz               -- the ONLY valid rule on the FILTER mean
                    (whose increment is filter propagation, not an SDE integral: 2nd-order rules failed).
      'trapezoidal' 1/2(f(z_t)+f(z_{t+1})) ~ dz              -- 2nd-order; valid on the SMOOTHER mean
      'midpoint'    f((z_t+z_{t+1})/2)      ~ dz             -- 2nd-order; valid on the SMOOTHER mean
    The 2nd-order rules cancel the left-endpoint bias IFF the increment is a genuine state increment,
    i.e. on the smoother mean (a proper state estimate) -- the hypothesis this knob tests. Updates in place."""
    drift_net.requires_grad_(True)
    zc_next = zc if zc_next is None else zc_next
    zmid = 0.5 * (zc + zc_next)
    for _ in range(m_inner):
        dr_opt.zero_grad()
        if method == "trapezoidal":
            f = 0.5 * (drift_net.net(zc.unsqueeze(-1)).squeeze(-1)
                       + drift_net.net(zc_next.unsqueeze(-1)).squeeze(-1))
        elif method == "midpoint":
            f = drift_net.net(zmid.unsqueeze(-1)).squeeze(-1)
        else:                                                # euler (left-endpoint)
            f = drift_net.net(zc.unsqueeze(-1)).squeeze(-1)
        fit = ((f - dz) ** 2).mean()
        fr = drift_net.net(z_reg.unsqueeze(-1)).squeeze(-1)
        f_pp = (fr[2:] - 2 * fr[1:-1] + fr[:-2]) / hr ** 2
        (fit + reg_lambda * (f_pp ** 2).mean()).backward()
        dr_opt.step()
    drift_net.requires_grad_(False)


def fit_diffusion(diff_net, dg_opt, drift_net, zc, zc_next, dz, z_reg, hr, dt,
                  g_net, reg_lambda_g, m_inner, pair=None):
    """Diffusion from the increment residual after the trapezoidal drift
    (r = dz - 1/2(f(z_t)+f(z_t+1))). g_net=True fits a g^2(z) network on the per-sample target
    dt*r^2 with an H2 penalty; else a constant scalar. Returns the scalar g summary.

    `pair` (zt, zt1, W) from smoother_pair_fixed enables the CORRECT square-then-average estimator
    (v2): g^2 = (1/dt) mean_pairs sum_k W_k (z_{t+1}^k - z_t^k - 1/2(f_k+f'_k) dt)^2. Unlike the
    mean-increment r above -- which uses (E[z_{t+1}] - E[z_t]) and so drops the increment VARIANCE,
    under-reading g -- this averages the SQUARED per-sample residual over the lag-one joint, recovering
    E[(Δz)^2|y] = (Δmean)^2 + Var_t + Var_{t+1} - 2 Cov. Scalar g only (the high-D / em_smooth case)."""
    if pair is not None:                                              # v2 cross-cov square-then-average g
        zt, zt1, W = pair
        with torch.no_grad():
            f_t = drift_net.net(zt.unsqueeze(-1)).squeeze(-1)
            f_t1 = drift_net.net(zt1.unsqueeze(-1)).squeeze(-1)
        res = zt1 - zt - 0.5 * (f_t + f_t1) * dt                     # (T-1,B,K) trapezoidal residual
        g2 = (W * res ** 2).sum(-1).mean() / dt                     # weighted mean over joint samples
        return float(g2.sqrt().clamp(min=0.05))
    with torch.no_grad():
        f_trap = 0.5 * (drift_net.net(zc.unsqueeze(-1)).squeeze(-1)
                        + drift_net.net(zc_next.unsqueeze(-1)).squeeze(-1))
    r = dz - f_trap
    if g_net:
        target_g2 = dt * r ** 2
        diff_net.requires_grad_(True)
        for _ in range(m_inner):
            dg_opt.zero_grad()
            g2p = diff_net._g2(zc.unsqueeze(-1)).squeeze(-1)
            g2r = diff_net._g2(z_reg.unsqueeze(-1)).squeeze(-1)
            g2_pp = (g2r[2:] - 2 * g2r[1:-1] + g2r[:-2]) / hr ** 2
            (((g2p - target_g2) ** 2).mean()
             + reg_lambda_g * (g2_pp ** 2).mean()).backward()
            dg_opt.step()
        diff_net.requires_grad_(False)
        with torch.no_grad():
            return float(diff_net._g2(z_reg.unsqueeze(-1)).clamp(min=1e-6).sqrt().mean())
    return float((r.pow(2).mean() * dt).sqrt().clamp(min=0.05))


@torch.no_grad()
def fit_obs_map_stiefel(z_hat, y, s_scale, C_cur):
    """High-D Stiefel observation map (orthogonal Procrustes): C = unit direction of the
    cross-covariance of y and the inferred latent; the scale s is FIXED (no runaway) and d is the
    intercept. Returns (C_new, d_new, cstab) with cstab = 1 - |cos(C_new, C_cur)| (the gate signal)."""
    zf = z_hat.reshape(-1)
    yf = y.reshape(-1, y.shape[-1])
    M = ((yf - yf.mean(0)) * (zf - zf.mean())[:, None]).sum(0)          # cross-cov (D,)
    C_new = M / M.norm().clamp_min(1e-8)
    cstab = 1.0 - abs(float(C_new @ C_cur))
    d_new = yf.mean(0) - s_scale * C_new * zf.mean()
    return C_new, d_new, cstab


def mstep(model, x, mask, z_grid, dt, drift_net, dr_opt, diff_net, dg_opt, z_reg, hr, *,
          learn_g, g_net, reg_lambda, reg_lambda_g, m_inner,
          learn_obs=False, c_stable_tol=0.05, C_cur=None, d_cur=None, s_scale=1.0,
          meshfree_mean=False, n_mean=256, near_std=0.3, broad_std=1.6,
          model_b=None, smoother_mstep=False, g_cur_in=None, drift_method="euler"):
    """One EM M-step. Order: posterior-mean increments -> (high-D) Stiefel obs-map + cstab ->
    drift GATED on `cstab < c_stable_tol` -> diffusion. In 1-D (learn_obs=False) cstab==0, so the
    gate is always open and this reduces to the plain f,g M-step. `meshfree_mean` replaces the grid
    E[z|y] with the SNIS estimate (no z_grid).

    `smoother_mstep` (v2, requires model_b) reads the M-step statistics from the SMOOTHER instead of
    the filter: the drift regresses on the SMOOTHER-mean increments (a proper state estimate, less laggy
    than the filter mean, esp. through gaps), and g uses the deterministic lag-one cross-cov
    (smoother_pair_fixed) so the square-then-average estimator recovers the increment variance the
    filter-mean g drops (fixes the g under-read). `g_cur_in` is the current scalar g for the pair proposal
    (benign RTS-EM circularity; CRN nodes+eps keep it from running away). Returns {g_cur,C_cur,d_cur,cstab}."""
    ess = None
    center = (zhat_from_obs(x, C_cur, d_cur) / s_scale) if learn_obs else x[..., 0]
    pair = None
    if smoother_mstep:                                               # v2: statistics from the smoother
        z_hat, ess = smoother_mean_fixed(model, model_b, x, mask, center, n_mean, near_std, broad_std)
        if learn_g:
            pair = smoother_pair_fixed(model, model_b, x, mask, center, drift_net,
                                       g_cur_in if g_cur_in is not None else 0.5, dt,
                                       n_mean, near_std, broad_std)
    elif meshfree_mean:                                              # grid-free E[z|y] via deterministic fixed-node mean
        z_hat, ess = posterior_mean_fixed(model, x, mask, center, n_mean, near_std, broad_std)
    else:
        z_hat = posterior_mean(model, x, mask, z_grid)                # (T, B)
    m = mask[..., 0]
    valid = (m[:-1] * m[1:]).bool()                                   # data-anchored increments
    zc = z_hat[:-1][valid]
    zc_next = z_hat[1:][valid]
    dz = ((z_hat[1:] - z_hat[:-1]) / dt)[valid]

    cstab = 0.0
    if learn_obs:
        C_cur, d_cur, cstab = fit_obs_map_stiefel(z_hat, x, s_scale, C_cur)
    if cstab < c_stable_tol:                                          # sensor-before-dynamics gate
        fit_drift(drift_net, dr_opt, zc, dz, z_reg, hr, reg_lambda, m_inner,
                  zc_next=zc_next, method=drift_method)
    g_cur = None
    if learn_g:
        g_cur = fit_diffusion(diff_net, dg_opt, drift_net, zc, zc_next, dz, z_reg, hr, dt,
                              g_net, reg_lambda_g, m_inner, pair=pair)
    return dict(g_cur=g_cur, C_cur=C_cur, d_cur=d_cur, cstab=cstab, ess=ess)
