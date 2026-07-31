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

from opssm.models.obs import make_decode, zhat_from_obs
from opssm.models.losses import sample_collocation


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
    so they are reused every M-step (no fresh randomness). Returns (z (T,B,K,d), log_q (T,B,K))."""
    T, B, d = center.shape
    dev = center.device
    gen = torch.Generator(device=dev).manual_seed(0)                 # FIXED nodes -> deterministic readout
    obs = mask[..., 0]
    c = center * obs.unsqueeze(-1)                                   # (T,B,d)
    near_sd = (near_std * obs + broad_std * (1.0 - obs))             # (T,B)
    Kn = n_samples // 2
    Kb = n_samples - Kn
    eps_n = torch.randn(1, B, Kn, d, device=dev, generator=gen)      # shared across t (CRN) + fixed seed
    eps_b = torch.randn(1, B, Kb, d, device=dev, generator=gen)
    near = c.unsqueeze(2) + near_sd[..., None, None] * eps_n         # (T,B,Kn,d)
    broad = (broad_std * eps_b).expand(T, B, Kb, d)
    z = torch.cat([near, broad], dim=2)                            # (T,B,K,d)
    off = math.log(2.0) + 0.5 * d * math.log(2 * math.pi)
    lqn = -0.5 * ((z - c.unsqueeze(2)) / near_sd[..., None, None]).pow(2).sum(-1) - d * near_sd[..., None].log() - off
    lqb = -0.5 * (z / broad_std).pow(2).sum(-1) - d * math.log(broad_std) - off
    log_q = torch.logaddexp(lqn, lqb)                              # (T,B,K)
    return z, log_q


@torch.no_grad()
def posterior_mean_fixed(model, x, mask, center, n_samples, near_std, broad_std):
    """DETERMINISTIC mesh-free FILTER MEAN via FIXED-NODE importance sampling: the proposal nodes are
    drawn ONCE (fixed seed) and reused every M-step. So z_hat = sum_k w_k z_k is a smooth deterministic
    function of the operator -- the grid's deterministic mean, but with data-following nodes instead of a
    uniform grid. No fresh per-M-step randomness => no readout noise (the SNIS blowup was the fresh
    per-step sampling noise; fixing the nodes removes it while keeping the MEAN, unlike the mode).
    Returns (z_hat (T,B), ess_frac)."""
    z, log_q = _fixed_nodes(center, mask, n_samples, near_std, broad_std)   # (T,B,K,d)
    ctx = model.context(x, mask)
    b0 = model.coeffs(ctx, torch.zeros(1, device=z.device))[:, :, 0]  # (T,B,p) at s=0
    tau = model.trunk(z)                                           # (T,B,K,p)  (z already (...,d))
    ell = torch.einsum("tbp,tbkp->tbk", b0, tau) + model.bias       # (T,B,K)
    w = torch.softmax(ell - log_q, dim=-1)                          # SNIS weights
    ess = 1.0 / (w.pow(2).sum(-1) * z.shape[2])                     # (T,B) fraction (K = z.shape[2])
    return (w.unsqueeze(-1) * z).sum(2), float(ess.mean())          # (T,B,d), scalar


@torch.no_grad()
def _mala_chains(model, x, mask, center, n_chains, n_steps, broad_std, rng="stochastic",
                 step_size=0.1, burn_in=None, init_std=0.0, adapt=True):
    """Core sampler: T*B INDEPENDENT d-dim MALA chains on the per-time filter marginal pi_t ~ exp(ell_t) (the
    marginals are independent across t -> fully parallel, no time recurrence). ell_t from b0 = coeffs(ctx,s=0);
    grad ell = einsum(b0, grad_tau) via trunk_grad. Chains init AT the pseudo-inverse `center` (init_std=0 ->
    the first Langevin step decorrelates them), BROAD (broad_std) only at unobserved sites; per-site step size
    adapts toward the 0.574 MALA optimum during burn_in (default n_steps//3). `rng`: 'stochastic' = fresh each
    call; 'crn' = fixed seed (deterministic); 'qmc' deferred. Returns (z_final (T,B,K,d) the final chain states
    ~ pi_t, z_mean (T,B,d) post-burn-in chain mean, accept_frac). The mean readout uses z_mean; the lag-one
    joint g uses z_final as samples from alpha_t."""
    dev = center.device
    T, B, d = center.shape
    if burn_in is None:
        burn_in = max(1, n_steps // 3)                              # auto: a third of the sweeps
    if rng == "crn":
        gen = torch.Generator(device=dev).manual_seed(0)            # fixed draws -> deterministic across M-steps
    elif rng == "stochastic":
        gen = None                                                  # default RNG -> fresh each M-step
    else:
        raise NotImplementedError(f"rng={rng!r}: qmc = Sobol across the chain ensemble; implement only if "
                                  "stochastic diverges (plan Phase 1 fallback order stochastic->qmc->crn)")
    randn = lambda *s: torch.randn(*s, device=dev, generator=gen)   # noqa: E731
    rand = lambda *s: torch.rand(*s, device=dev, generator=gen)     # noqa: E731
    ctx = model.context(x, mask)
    b0 = model.coeffs(ctx, torch.zeros(1, device=dev))[:, :, 0]     # (T,B,p) log-density coeffs at s=0

    def ell_grad(z):                                               # z (T,B,K,d) -> ell (T,B,K), grad (T,B,K,d)
        tau, gtau = model.trunk_grad(z)                            # tau (T,B,K,p), gtau (T,B,K,d,p)
        ell = torch.einsum("tbp,tbkp->tbk", b0, tau) + model.bias
        return ell, torch.einsum("tbp,tbkdp->tbkd", b0, gtau)

    obs = mask[..., 0]                                             # (T,B) observed indicator
    c = (center * obs.unsqueeze(-1)).unsqueeze(2)                  # (T,B,1,d) zero the meaningless gap center
    sd = (init_std * obs + broad_std * (1.0 - obs))[..., None, None]   # (T,B,1,1) BROAD init at gaps
    z = c + sd * randn(T, B, n_chains, d)                         # (T,B,K,d) chain init
    ell, grad = ell_grad(z)
    eps = torch.full((T, B, 1, 1), float(step_size), device=dev)  # per-site step size
    acc_sum, zsum, ncol = 0.0, torch.zeros_like(center), 0
    for k in range(n_steps):
        noise = randn(T, B, n_chains, d)
        z_p = z + eps * grad + (2 * eps).sqrt() * noise           # Langevin proposal
        ell_p, grad_p = ell_grad(z_p)
        e4 = 4 * eps.squeeze(-1)                                   # (T,B,1)
        log_q_fwd = -(z_p - z - eps * grad).pow(2).sum(-1) / e4    # log q(z'|z)  (T,B,K)
        log_q_bwd = -(z - z_p - eps * grad_p).pow(2).sum(-1) / e4  # log q(z|z')
        log_alpha = (ell_p - ell) + (log_q_bwd - log_q_fwd)       # (T,B,K) Metropolis ratio
        acc = rand(T, B, n_chains).log() < log_alpha              # (T,B,K) accept
        ae = acc.unsqueeze(-1)
        z = torch.where(ae, z_p, z); ell = torch.where(acc, ell_p, ell); grad = torch.where(ae, grad_p, grad)
        acc_sum += float(acc.float().mean())
        if adapt and k < burn_in:                                 # per-site step adaptation toward 0.574 (deterministic)
            ap = acc.float().mean(-1)[..., None, None]            # (T,B,1,1)
            eps = (eps * (0.75 * (ap - 0.574)).exp()).clamp(1e-4, 2.0)
        if k >= burn_in:
            zsum = zsum + z.mean(2); ncol += 1                    # accumulate post-burn-in chain mean
    return z, zsum / max(ncol, 1), acc_sum / max(n_steps, 1)      # (T,B,K,d) states, (T,B,d) mean, accept


@torch.no_grad()
def posterior_mean_mala(model, x, mask, center, n_chains, n_steps, broad_std, rng="stochastic",
                        step_size=0.1, burn_in=None, init_std=0.0, adapt=True):
    """MALA readout of the per-time filter mean E[z_t|y_{0:t}] (see _mala_chains). Gradient MCMC replaces SNIS
    as the latent dim d grows (IS weights degenerate ~exp(-c d); MALA cost/eff-sample ~d^{1/3}). Only n_chains
    / n_steps (compute budget) and rng are user-facing -- the rest self-tune. Returns (z_hat (T,B,d), accept)."""
    _, z_mean, acc = _mala_chains(model, x, mask, center, n_chains, n_steps, broad_std, rng,
                                  step_size, burn_in, init_std, adapt)
    return z_mean, acc


@torch.no_grad()
def filter_mean(model, x, mask, center, *, method, n_mean, near_std, broad_std, mala=None):
    """Selector for the mesh-free FILTER MEAN E[z_t|y_{0:t}] -> (z_hat (T,B,d), diag). method='fixed' =
    deterministic fixed-node importance sampling (posterior_mean_fixed, diag={'ess':..}); method='mala' =
    gradient MCMC (posterior_mean_mala, knobs in the `mala` dict, diag={'accept':..}). One seam for both the
    M-step readout and the d>1 validation mean."""
    if method == "mala":
        z_hat, acc = posterior_mean_mala(model, x, mask, center, broad_std=broad_std, **(mala or {}))
        return z_hat, {"accept": acc}
    z_hat, ess = posterior_mean_fixed(model, x, mask, center, n_mean, near_std, broad_std)
    return z_hat, {"ess": ess}


@torch.no_grad()
def smoother_mean_fixed(model, model_b, x, mask, center, n_samples, near_std, broad_std):
    """DETERMINISTIC mesh-free SMOOTHER MEAN E[z_t | y_{0:T}] on the same fixed-node CRN proposal, with
    the numerically STABLE predict*msg smoother weights
        W_k = softmax_k( log predict_t(z_k) + log msg_t(z_k) - log q_k ),
      log predict_t = forward operator s=1 of step t-1 evaluated on node_t (= log p(z_t|y_{0:t-1}), smooth);
      log msg_t     = backward operator s=0 (= log p(y_{t:T}|z_t)).
    Unlike the FILTER mean, the smoother mean is a proper state estimate, so the midpoint/trapezoidal drift
    correction becomes valid on its increments (v2 M-step). Returns (z_hat (T,B), ess_frac)."""
    z, log_q = _fixed_nodes(center, mask, n_samples, near_std, broad_std)   # (T,B,K,d)
    dev = z.device
    ctx_f = model.context(x, mask)
    ctx_b = model_b.context(x, mask)
    tau_f = model.trunk(z)                                          # (T,B,K,p) forward basis on nodes
    tau_b = model_b.trunk(z)                                        # (T,B,K,p) backward basis (own weights)
    b1_f = model.coeffs(ctx_f, torch.ones(1, device=dev))[:, :, 0]  # (T,B,p) forward s=1 coeffs
    log_pred = torch.empty(z.shape[:3], device=dev)                # (T,B,K)
    log_pred[0] = -0.5 * z[0].pow(2).sum(-1)                       # predict_0 = prior N(0,I) on node_0
    log_pred[1:] = torch.einsum("tbp,tbkp->tbk", b1_f[:-1], tau_f[1:]) + model.bias  # predict_t on node_t
    b0_b = model_b.coeffs(ctx_b, torch.zeros(1, device=dev))[:, :, 0]   # (T,B,p) backward s=0
    lmsg = torch.einsum("tbp,tbkp->tbk", b0_b, tau_b) + model_b.bias    # (T,B,K) log msg_t on node_t
    w = torch.softmax(log_pred + lmsg - log_q, dim=-1)
    ess = 1.0 / (w.pow(2).sum(-1) * z.shape[2])
    return (w.unsqueeze(-1) * z).sum(2), float(ess.mean())


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
    z, log_q = _fixed_nodes(center, mask, n_samples, near_std, broad_std)   # (T,B,K,d)
    dev = z.device
    gen = torch.Generator(device=dev).manual_seed(1)                       # FIXED transition noise (CRN)
    eps = torch.randn(1, z.shape[1], z.shape[2], z.shape[3], device=dev, generator=gen)  # (1,B,K,d)
    f_z = drift_net.drift(z)[0]                                            # (T,B,K,d) drift on nodes
    z_next = z + f_z * dt + math.sqrt(max(float(g_cur), 1e-6) ** 2 * dt) * eps  # (T,B,K,d) ~ K(.|z)
    zt, zt1 = z[:-1], z_next[:-1]                                          # pair (t, t+1) samples
    ctx_f = model.context(x, mask); ctx_b = model_b.context(x, mask)
    b0_f = model.coeffs(ctx_f, torch.zeros(1, device=dev))[:, :, 0]        # (T,B,p) forward s=0
    b0_b = model_b.coeffs(ctx_b, torch.zeros(1, device=dev))[:, :, 0]      # (T,B,p) backward s=0
    l_alpha = torch.einsum("tbp,tbkp->tbk", b0_f[:-1], model.trunk(zt)) + model.bias
    l_msg1 = torch.einsum("tbp,tbkp->tbk", b0_b[1:], model_b.trunk(zt1)) + model_b.bias
    W = torch.softmax((l_alpha - log_q[:-1]) + l_msg1, dim=-1)             # (T-1,B,K)
    return zt, zt1, W


def filter_pair_fixed(model, x, mask, center, drift_net, g_cur, dt,
                      n_samples, near_std, broad_std, noise_std, decode=None):
    """Deterministic fixed-node CRN FILTER lag-one joint p(z_t, z_{t+1} | y_{0:t+1}) -> the cross-covariance
    that fixes the diffusion g, using ONLY the forward operator (no backward/smoother -- so it is CONSISTENT
    with the filter-mean drift: one posterior feeds both moments). Proposal: z_t^k ~ q_t (fixed nodes);
    z_{t+1}^k = z_t^k + f(z_t^k) dt + sqrt(g^2 dt) eps^k (fixed eps^k), so the proposal transition equals the
    model K and CANCELS in the importance weight. The forward filter joint is proportional to
    alpha_t(z_t) K(z_{t+1}|z_t) lik_{t+1}(z_{t+1}), hence
        W_k = softmax_k( log alpha_t(z_t^k) - log q_t^k + loglik_{t+1}(z_{t+1}^k) )
    (alpha_t = forward s=0; lik_{t+1} = the Gaussian obs likelihood at y_{t+1}, computed DIRECTLY from the
    obs model -- stable, unlike the alpha/predict ratio, whose -loglik blows up in the tails). Returns
    (zt, zt1, W) each (T-1,B,K) for fit_diffusion's square-then-average g."""
    z, log_q = _fixed_nodes(center, mask, n_samples, near_std, broad_std)   # (T,B,K,d)
    dev = z.device
    gen = torch.Generator(device=dev).manual_seed(1)                       # FIXED transition noise (CRN)
    eps = torch.randn(1, z.shape[1], z.shape[2], z.shape[3], device=dev, generator=gen)  # (1,B,K,d)
    f_z = drift_net.drift(z)[0]                                            # (T,B,K,d) drift on nodes
    z_next = z + f_z * dt + math.sqrt(max(float(g_cur), 1e-6) ** 2 * dt) * eps  # (T,B,K,d) ~ K(.|z)
    zt, zt1 = z[:-1], z_next[:-1]                                          # pair (t, t+1) samples
    ctx_f = model.context(x, mask)
    b0_f = model.coeffs(ctx_f, torch.zeros(1, device=dev))[:, :, 0]        # (T,B,p) forward s=0
    l_alpha = torch.einsum("tbp,tbkp->tbk", b0_f[:-1], model.trunk(zt)) + model.bias
    xnext = x[1:]                                                          # obs at t+1 (T-1,B,D)
    if decode is None:                                                    # direct obs h(z)=z (D=d)
        loglik = -0.5 * ((xnext.unsqueeze(2) - zt1) ** 2).sum(-1) / noise_std ** 2   # (T-1,B,K)
    else:                                                                 # high-D: h(z) = s C z + d
        loglik = -0.5 * ((xnext.unsqueeze(2) - decode(zt1)) ** 2).sum(-1) / noise_std ** 2
    W = torch.softmax((l_alpha - log_q[:-1]) + loglik, dim=-1)            # (T-1,B,K)
    return zt, zt1, W


@torch.no_grad()
def filter_pair_mala(model, x, mask, center, drift_net, g_cur, dt, noise_std, C_cur, d_cur,
                     n_chains, n_steps, broad_std, rng="stochastic"):
    """MCMC estimate of the FILTER lag-one joint p(z_t, z_{t+1} | y_{0:t+1}) for the diffusion g, replacing the
    SNIS filter_pair_fixed. The SINGLE change vs filter_pair_fixed: sample z_t ~ alpha_t by MALA (EXACT) instead
    of from a fixed-node proposal q_t. That removes the SNIS weight's `alpha_t/q_t` factor -- the piece that
    COLLAPSES as d grows (ess ~exp(-c d)) and biases g at d=3 -- leaving only the MILD obs-likelihood weight.
    Everything else matches filter_pair_fixed: z_{t+1}^k = z_t^k + f(z_t^k) dt + sqrt(g^2 dt) eps^k is the model
    PREDICT (so the residual is the FRESH process noise -> square-then-average recovers g^2 dt, NOT the
    obs-contracted filtered increment), and W_k = softmax_k loglik_{t+1}(z_{t+1}^k). Returns (zt, zt1, W) each
    (T-1,B,K,.) for fit_diffusion. C_cur=None => direct obs (h(z)=z)."""
    dev = center.device
    d = center.shape[-1]
    zt_all, _, _ = _mala_chains(model, x, mask, center, n_chains, n_steps, broad_std, rng)  # (T,B,K,d) ~ alpha_t EXACT
    f_z = drift_net.drift(zt_all)[0]                             # (T,B,K,d) drift on the samples
    gen = None if rng == "stochastic" else torch.Generator(device=dev).manual_seed(2)
    eps = torch.randn(*zt_all.shape, device=dev, generator=gen)  # (T,B,K,d) fresh process noise
    z_next = zt_all + f_z * dt + math.sqrt(max(float(g_cur), 1e-6) ** 2 * dt) * eps   # PREDICT (T,B,K,d)
    zt, zt1 = zt_all[:-1], z_next[:-1]                          # pairs (t, t+1)  each (T-1,B,K,d)
    xnext = x[1:]                                               # obs at t+1 (T-1,B,D), standardized
    if C_cur is None:                                          # direct obs h(z)=z (D=d)
        loglik = -0.5 * ((xnext.unsqueeze(2) - zt1) ** 2).sum(-1) / noise_std ** 2
    else:                                                     # high-D: h(z) = C z + d
        loglik = -0.5 * ((xnext.unsqueeze(2) - (zt1 @ C_cur.t() + d_cur)) ** 2).sum(-1) / noise_std ** 2
    return zt, zt1, torch.softmax(loglik, dim=-1)             # (T-1,B,K,d), (T-1,B,K,d), (T-1,B,K)


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


def fit_dynamics(drift_net, dr_opt, zc, dz, g_cur, dt, reg_lambda, m_inner, *,
                 w_em=1.0, w_inv=0.0, inv=None, g_lr=1e-2, w_g=0.0):
    """JOINTLY fit the drift f (DriftNet) AND the diffusion g^2 (a scalar) by gradient descent on a WEIGHTED
    loss -- the two co-adapt to the physics each step, not alternated or switched:
        loss = w_em  * ||f(z_t) - dz||^2                                  (EM: filter-MEAN increment; f only)
             + w_inv * (ds_ell/dt + div f + f.grad ell - g^2 B)^2         (FP residual for f AND g^2; the Zakai
             + reg_lambda * ||f||^2                                        PDE coefficient the density implies)
    with B = 1/2(|grad ell|^2 + Laplacian ell) the diffusion signature. The FP residual is BILINEAR in f and
    g^2, so f and g^2 step TOGETHER. The increment pins the ROTATIONAL (divergence-free) drift the residual
    leaves free at d>=2; the residual is free of the 1/dt increment amplification and supplies g from the
    LAPLACIAN signature (not the drift-limited increment variance g_est^2 = g^2 + drift_rmse^2*dt). `inv =
    (z_col, grad_ell, ds_ell, lap_ell)` the precomputed operator derivatives (detached, from _fp_residual_terms)
    or None to skip the FP term. Returns the fitted scalar g if learned here (w_inv>0 with inv), else None.
    Updates drift_net in place. zc, dz: (N,d)."""
    dev = zc.device
    learn_g_here = w_inv > 0 and inv is not None
    g_anchor = max(float(g_cur), 1e-3)                                # increment-g estimate: g init AND data anchor
    g = torch.tensor(g_anchor, device=dev, requires_grad=learn_g_here)   # the FACTOR g (not g^2)
    g_opt = torch.optim.Adam([g], lr=g_lr) if learn_g_here else None
    drift_net.requires_grad_(True)
    for _ in range(m_inner):
        dr_opt.zero_grad()
        if g_opt is not None:
            g_opt.zero_grad()
        loss = reg_lambda * sum(p.pow(2).sum() for p in drift_net.net.parameters())   # l_reg
        if w_em > 0:                                                # l_em: filter-mean increment regression (f)
            loss = loss + w_em * ((drift_net.net(zc) - dz) ** 2).sum(-1).mean()
        if learn_g_here:                                           # l_inv: FP residual (f AND g jointly)
            z_col, grad_ell, ds_ell, lap_ell = inv
            f, div_f = drift_net.drift(z_col)                       # (T,B,K,d), (T,B,K)  WITH grad (div f too)
            f_dot = torch.einsum("tbskd,tbkd->tbsk", grad_ell, f)  # (T,B,Ns,K)  f . grad ell
            Bsig = 0.5 * (grad_ell.pow(2).sum(-1) + lap_ell)       # (T,B,Ns,K)  diffusion signature
            res = ds_ell / dt + div_f.unsqueeze(2) + f_dot - g ** 2 * Bsig   # g^2 in the FP term; g is the param
            loss = loss + w_inv * res.pow(2).mean()
            if w_g > 0:                                            # DATA ANCHOR for g: pull toward the increment-g
                loss = loss + w_g * (g - g_anchor) ** 2            #   (our stand-in for the paper's observed WIDTH;
                #   points don't pin the posterior width the way densities do, so g-from-FP alone runs away).
                #   NB own weight w_g -- the anchor & FP-residual live at very different scales.
        loss.backward()
        dr_opt.step()
        if g_opt is not None:
            g_opt.step()
    drift_net.requires_grad_(False)
    return float(g.detach().abs().clamp_min(0.05)) if learn_g_here else None   # report |g| (sign-symmetric)


def fit_diffusion(diff_net, dg_opt, drift_net, zc, zc_next, dz, z_reg, hr, dt,
                  g_net, reg_lambda_g, m_inner, pair=None):
    """Diffusion from the increment residual after the trapezoidal drift
    (r = dz - 1/2(f(z_t)+f(z_t+1))). g_net=True fits a g^2(z) network on the per-sample target
    dt*r^2 with an H2 penalty; else a constant scalar. Returns the scalar g summary.

    `pair` (zt, zt1, W) from filter_pair_fixed switches to the CORRECT square-then-average estimator:
        g^2 = (1/dt) mean_pairs sum_k W_k (z_{t+1}^k - z_t^k - 1/2(f_k+f'_k) dt)^2.
    The default path below uses the increment of the filter MEAN (E[z_{t+1}]-E[z_t]), which drops the
    increment VARIANCE and so under-reads g (~0.54 vs 0.60 high-D). The pair path averages the SQUARED
    per-sample residual over the lag-one joint, recovering E[(Δz)^2|y] = (Δmean)^2 + Var_t + Var_{t+1}
    - 2 Cov -- the hidden-state form of the NKE covariance-increment diffusion loss. Scalar g."""
    if pair is not None:                                            # lag-one joint square-then-average g
        zt, zt1, W = pair                                          # zt,zt1 (T-1,B,K,d); W (T-1,B,K)
        with torch.no_grad():
            f_t = drift_net.net(zt)                                # (T-1,B,K,d)
            f_t1 = drift_net.net(zt1)
        res = zt1 - zt - 0.5 * (f_t + f_t1) * dt                   # (T-1,B,K,d) trapezoidal residual
        dcnt = res.shape[-1]
        g2 = (W * res.pow(2).sum(-1)).sum(-1).mean() / (dt * dcnt) # isotropic: E[|res|^2]/(d dt)
        return float(g2.detach().sqrt().clamp(min=0.05))
    with torch.no_grad():
        f_trap = 0.5 * (drift_net.net(zc) + drift_net.net(zc_next))  # (N,d)
    r = dz - f_trap                                               # (N,d)
    dcnt = r.shape[-1]
    if g_net:                                                     # state-dependent g^2(z) -- 1-D only for now
        target_g2 = dt * r.pow(2).sum(-1, keepdim=True)           # (N,1)  (d==1)
        diff_net.requires_grad_(True)
        for _ in range(m_inner):
            dg_opt.zero_grad()
            g2p = diff_net._g2(zc)                                # (N,1)
            g2r = diff_net._g2(z_reg.unsqueeze(-1)).squeeze(-1)   # 1-D reg grid
            g2_pp = (g2r[2:] - 2 * g2r[1:-1] + g2r[:-2]) / hr ** 2
            (((g2p - target_g2) ** 2).mean()
             + reg_lambda_g * (g2_pp ** 2).mean()).backward()
            dg_opt.step()
        diff_net.requires_grad_(False)
        with torch.no_grad():
            return float(diff_net._g2(z_reg.unsqueeze(-1)).clamp(min=1e-6).sqrt().mean())
    return float((r.pow(2).sum(-1).mean() * dt / dcnt).sqrt().clamp(min=0.05))


@torch.no_grad()
def _fp_residual_terms(model, x, mask, n_colloc, near_std, broad_std, center=None, n_scoll=4):
    """Shared FP-residual pieces for the inverse-problem dynamics fit: sample collocation and return the
    operator's DENSITY DERIVATIVES at those points, all DETACHED (the density is fixed for the M-step). Computed
    ONCE per M-step and handed to fit_dynamics as `inv`. Returns (z_col, grad_ell, ds_ell, lap_ell) with shapes
    (T,B,K,d), (T,B,Ns,K,d), (T,B,Ns,K), (T,B,Ns,K). Mirrors the FP-residual derivative block of pinn_zakai_loss."""
    dev = x.device
    s_coll = torch.linspace(0.0, 1.0, n_scoll, device=dev)
    z_col, _ = sample_collocation(x, mask, n_colloc, near_std, broad_std, center)   # (T,B,K,d)
    ctx = model.context(x, mask)
    b_s, ds_b = model.coeffs_dtime(ctx, s_coll)                       # (T,B,Ns,p)
    tau_s, grad_tau, lap_tau = model.trunk_zderivs(z_col)             # (T,B,K,p),(...,K,d,p),(...,K,p)
    grad_ell = torch.einsum("tbsp,tbkdp->tbskd", b_s, grad_tau)      # (T,B,Ns,K,d)  grad ell
    ds_ell = torch.einsum("tbsp,tbkp->tbsk", ds_b, tau_s)           # (T,B,Ns,K)  d_s ell
    lap_ell = torch.einsum("tbsp,tbkp->tbsk", b_s, lap_tau)         # (T,B,Ns,K)  Laplacian ell
    return z_col, grad_ell, ds_ell, lap_ell


@torch.no_grad()
def fit_obs_map_stiefel(z_hat, y, C_cur):
    """High-D Stiefel observation map (orthogonal Procrustes): C = unit direction of the cross-covariance
    of y and the inferred latent; d is the intercept. Obs are standardized upstream so the decode has no
    scale factor (h(z) = C z + d). z_hat (T,B,d), y (T,B,D), C_cur (D,d). C = nearest orthonormal-columns
    matrix to the D x d cross-cov via SVD (C = U V^T); cstab = 1 - mean principal-angle cosine."""
    d = z_hat.shape[-1]
    Z = z_hat.reshape(-1, d)
    Y = y.reshape(-1, y.shape[-1])
    M = (Y - Y.mean(0)).t() @ (Z - Z.mean(0))                          # (D,d) cross-cov
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)               # U (D,d), Vt (d,d)
    C_new = U @ Vt                                                    # (D,d) Stiefel (orthonormal cols)
    cos = torch.linalg.svdvals(C_new.t() @ C_cur).clamp(max=1.0)      # principal-angle cosines (d,)
    cstab = 1.0 - float(cos.mean())
    d_new = Y.mean(0) - C_new @ Z.mean(0)                            # (D,) intercept
    return C_new, d_new, cstab


def mstep(model, x, mask, z_grid, dt, drift_net, dr_opt, diff_net, dg_opt, z_reg, hr, *,
          learn_g, g_net, reg_lambda, reg_lambda_g, m_inner,
          learn_obs=False, c_stable_tol=0.05, C_cur=None, d_cur=None,
          meshfree_mean=False, n_mean=256, near_std=0.3, broad_std=1.6, mean_method="fixed", mala=None,
          joint_g=False, noise_std=None, g_cur_in=None, w_em=1.0, w_inv=0.0, g_lr=1e-2, w_g=0.0, n_colloc=None):
    """One EM M-step. Order: posterior-mean increments -> (high-D) Stiefel obs-map + cstab ->
    drift GATED on `cstab < c_stable_tol` -> diffusion. In 1-D (learn_obs=False) cstab==0, so the
    gate is always open and this reduces to the plain f,g M-step. `meshfree_mean` replaces the grid
    E[z|y] with the SNIS estimate (no z_grid).

    `joint_g` estimates g from the FILTER lag-one joint (filter_pair_fixed) instead of the filter-MEAN
    increment -- the mean drops the increment variance and under-reads g (~0.54 vs 0.60 high-D); the
    joint's square-then-average residual recovers it. DRIFT still uses the filter mean (the smoother mean
    is the wrong, over-smoothed drift target), so ONE posterior (the forward filter) feeds both moments.
    Returns updated {g_cur, C_cur, d_cur, cstab}."""
    ess = accept = None
    center = zhat_from_obs(x, C_cur, d_cur) if learn_obs else x       # (T,B,d) (obs standardized upstream)
    if meshfree_mean:                                                 # grid-free E[z|y]: fixed-node IS or MALA
        z_hat, diag = filter_mean(model, x, mask, center, method=mean_method, n_mean=n_mean,
                                  near_std=near_std, broad_std=broad_std, mala=mala)
        ess, accept = diag.get("ess"), diag.get("accept")
    else:
        z_hat = posterior_mean(model, x, mask, z_grid)                # (T, B) -- 1-D grid only
    pair = None                                                       # filter lag-one joint for g (drift stays on the mean)
    if joint_g and learn_g and w_inv == 0:                            # only the increment g-path uses the pair
        g_j = g_cur_in if g_cur_in is not None else 0.5
        if mean_method == "mala":                                     # MCMC joint (Rao-Blackwellized; no SNIS collapse)
            mk = {k: mala[k] for k in ("n_chains", "n_steps", "rng")} if mala else {}
            pair = filter_pair_mala(model, x, mask, center, drift_net, g_j, dt, noise_std,
                                    C_cur if learn_obs else None, d_cur if learn_obs else None,
                                    broad_std=broad_std, **mk)
        else:                                                         # fixed-node SNIS joint
            decode = make_decode(C_cur, d_cur) if learn_obs else None
            pair = filter_pair_fixed(model, x, mask, center, drift_net, g_j, dt,
                                     n_mean, near_std, broad_std, noise_std, decode)
    m = mask[..., 0]
    valid = (m[:-1] * m[1:]).bool()                                   # data-anchored increments
    zc = z_hat[:-1][valid]                                            # (N,d)
    zc_next = z_hat[1:][valid]
    dz = ((z_hat[1:] - z_hat[:-1]) / dt)[valid]                       # (N,d)

    cstab = 0.0
    if learn_obs:
        C_cur, d_cur, cstab = fit_obs_map_stiefel(z_hat, x, C_cur)
    g_joint = None
    if cstab < c_stable_tol:                                          # sensor-before-dynamics gate: fit f (+ g jointly)
        inv = _fp_residual_terms(model, x, mask, n_colloc if n_colloc is not None else n_mean,
                                 near_std, broad_std, center) if w_inv > 0 else None
        g_init = g_cur_in if g_cur_in is not None else 0.5
        if w_inv > 0 and learn_g:                                    # warm-start g from the classic increment M-step
            with torch.no_grad():                                    #   estimate, then let the FP residual refine its bias
                r = dz - 0.5 * (drift_net.net(zc) + drift_net.net(zc_next))   # trapezoidal increment residual
                g_init = float((r.pow(2).sum(-1).mean() * dt / r.shape[-1]).sqrt().clamp(min=0.05))
        g_joint = fit_dynamics(drift_net, dr_opt, zc, dz, g_init, dt,
                               reg_lambda, m_inner, w_em=w_em, w_inv=w_inv, inv=inv, g_lr=g_lr, w_g=w_g)
    g_cur = None
    if learn_g:                                                      # g_joint set iff w_inv>0 (fit WITH f); else increment
        g_cur = g_joint if g_joint is not None else fit_diffusion(
            diff_net, dg_opt, drift_net, zc, zc_next, dz, z_reg, hr, dt, g_net, reg_lambda_g, m_inner, pair=pair)
    return dict(g_cur=g_cur, C_cur=C_cur, d_cur=d_cur, cstab=cstab, ess=ess, accept=accept)
