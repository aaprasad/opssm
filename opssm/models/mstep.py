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


@torch.no_grad()
def posterior_mean_fixed(model, x, mask, center, n_samples, near_std, broad_std):
    """DETERMINISTIC mesh-free MEAN via FIXED-NODE importance sampling: the proposal nodes are drawn
    ONCE (fixed seed) and reused every M-step, shared across t. So z_hat = sum_k w_k z_k is a smooth
    deterministic function of the operator -- the grid's deterministic mean, but with data-following
    nodes instead of a uniform grid. No fresh per-M-step randomness => no readout noise => the diffusion
    estimator g^2 = Var[dz]/dt stays stable (the SNIS blowup was the fresh per-step sampling noise;
    fixing the nodes removes it while keeping the MEAN, unlike the mode). Returns (z_hat (T,B), ess_frac)."""
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
    ctx = model.context(x, mask)
    b0 = model.coeffs(ctx, torch.zeros(1, device=dev))[:, :, 0]      # (T,B,p) at s=0
    tau = model.trunk(z.unsqueeze(-1))                              # (T,B,K,p)
    ell = torch.einsum("tbp,tbkp->tbk", b0, tau) + model.bias       # (T,B,K)
    w = torch.softmax(ell - log_q, dim=-1)                          # SNIS weights
    ess = 1.0 / (w.pow(2).sum(-1) * z.shape[-1])                    # (T,B) fraction
    return (w * z).sum(-1), float(ess.mean())                       # (T,B), scalar


def fit_drift(drift_net, dr_opt, zc, dz, z_reg, hr, reg_lambda, m_inner):
    """Regress f_theta(z) ~ dz with an H2 (curvature) smoothness penalty. Updates drift_net in place."""
    drift_net.requires_grad_(True)
    for _ in range(m_inner):
        dr_opt.zero_grad()
        f = drift_net.net(zc.unsqueeze(-1)).squeeze(-1)
        fit = ((f - dz) ** 2).mean()
        fr = drift_net.net(z_reg.unsqueeze(-1)).squeeze(-1)
        f_pp = (fr[2:] - 2 * fr[1:-1] + fr[:-2]) / hr ** 2
        (fit + reg_lambda * (f_pp ** 2).mean()).backward()
        dr_opt.step()
    drift_net.requires_grad_(False)


def fit_diffusion(diff_net, dg_opt, drift_net, zc, zc_next, dz, z_reg, hr, dt,
                  g_net, reg_lambda_g, m_inner):
    """Diffusion from the increment residual after the trapezoidal drift
    (r = dz - 1/2(f(z_t)+f(z_t+1))). g_net=True fits a g^2(z) network on the per-sample target
    dt*r^2 with an H2 penalty; else a constant scalar. Returns the scalar g summary."""
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
          meshfree_mean=False, n_mean=256, near_std=0.3, broad_std=1.6):
    """One EM M-step. Order: posterior-mean increments -> (high-D) Stiefel obs-map + cstab ->
    drift GATED on `cstab < c_stable_tol` -> diffusion. In 1-D (learn_obs=False) cstab==0, so the
    gate is always open and this reduces to the plain f,g M-step. `meshfree_mean` replaces the grid
    E[z|y] with the SNIS estimate (no z_grid). Returns updated {g_cur, C_cur, d_cur, cstab}."""
    ess = None
    if meshfree_mean:                                                 # grid-free E[z|y] via deterministic fixed-node mean
        center = (zhat_from_obs(x, C_cur, d_cur) / s_scale) if learn_obs else x[..., 0]
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
        fit_drift(drift_net, dr_opt, zc, dz, z_reg, hr, reg_lambda, m_inner)
    g_cur = None
    if learn_g:
        g_cur = fit_diffusion(diff_net, dg_opt, drift_net, zc, zc_next, dz, z_reg, hr, dt,
                              g_net, reg_lambda_g, m_inner)
    return dict(g_cur=g_cur, C_cur=C_cur, d_cur=d_cur, cstab=cstab, ess=ess)
