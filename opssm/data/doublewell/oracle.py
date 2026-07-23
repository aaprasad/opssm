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

"""Exact grid Zakai filter oracle + transition operators (validation ground truth).

The Euler-Maruyama transition matrix on a z-grid discretizes the Fokker-Planck / Zakai dynamics;
forward filtering + backward smoothing give the exact p(z_t | x_{0:t}) / p(z_t | x_{0:T}) the neural
operator is validated against. `grid_filter_target` (1-D direct obs) and `grid_filter_highd`
(high-D linear sensor) build the held-out reference posterior."""

import torch


def analytic_stationary_density(z, sigma):
    """Analytic equilibrium density p(z) ~ exp(-2 V/sigma^2), V = z^4/4 - z^2/2."""
    V = z ** 4 / 4.0 - z ** 2 / 2.0
    p = torch.exp(-2.0 * V / sigma ** 2)
    Z = torch.trapezoid(p, x=z)
    return p / Z


def build_transition(f_grid, z, dt_sub, sigma):
    """Column-stochastic Euler-Maruyama transition matrix on the grid.

    T[k, j] = N(z_k; z_j + f(z_j) dt_sub, sigma^2 dt_sub), normalized over k so
    each source column conserves mass. Differentiable w.r.t. f_grid.
    """
    mean = z + f_grid * dt_sub                       # (Nz,) target mean for each source j
    var = sigma ** 2 * dt_sub
    diff = z.unsqueeze(1) - mean.unsqueeze(0)        # (Nz_k, Nz_j)
    logT = -0.5 * diff ** 2 / var
    T = torch.softmax(logT, dim=0)                   # column-normalize (mass-conserving), stable
    return T


def transition_power(T, n):
    Tn = T
    for _ in range(n - 1):
        Tn = T @ Tn
    return Tn


def stationary_density(T, n_iter=4000):
    """Stationary mass distribution of the (single-substep) transition by iteration."""
    Nz = T.shape[0]
    pi = torch.full((Nz,), 1.0 / Nz, device=T.device)
    for _ in range(n_iter):
        pi = T @ pi
    return pi


def gauss_loglik_fn(xs, z, rho):
    """1-D direct-observation log-likelihood closure loglik_fn(i) -> (B,Nz): log N(x_i; z, rho^2)."""
    inv2var = 0.5 / rho ** 2

    def loglik_fn(i):
        return -inv2var * (xs[i, :, 0].unsqueeze(1) - z.unsqueeze(0)) ** 2      # (B, Nz)
    return loglik_fn


def highd_loglik_fn(y, z, C, d, rho):
    """High-D linear-sensor log-likelihood closure loglik_fn(i) -> (B,Nz):
    log N(y_i; C z + d, rho^2 I). hz = C z + d precomputed on the grid."""
    hz = C[None, :] * z[:, None] + d[None, :]                                   # (Nz, D)
    inv2var = 0.5 / rho ** 2

    def loglik_fn(i):
        return -inv2var * ((y[i][:, None, :] - hz[None, :, :]) ** 2).sum(-1)    # (B, Nz)
    return loglik_fn


@torch.no_grad()
def forward_backward(xs, z, T_full, loglik_fn, mask=None):
    """Forward filter + backward smoother (discrete HMM on the grid) -- EXACT
    latent-state inference, no variational approximation.

    loglik_fn(i) -> (B, Nz): per-step LOG-likelihood log p(obs_i | z) on the grid (Gaussian for 1-D
    via gauss_loglik_fn; high-D linear sensor via highd_loglik_fn) -- the single hook that lets 1-D
    and high-D share this forward+backward+gamma code. Stabilized by per-step max-subtraction (the
    constant cancels in every per-step normalization).

    mask: optional (T_steps,) bool; False = no observation at that step (the
    update is skipped, so the state is propagated by the dynamics alone). Lets us
    infer through gaps -- where the smoother visibly beats the filter.

    Returns (filtered, smoothed), each (T_steps, B, Nz) mass distributions:
      filtered[t] = p(z_t | observed x_{0:t})   (causal / online)
      smoothed[t] = p(z_t | observed x_{0:T})   (all observations)
    """
    T_steps, B = xs.shape[0], xs.shape[1]
    Nz = z.shape[0]

    def obs(i):
        return mask is None or bool(mask[i])

    def lik(i):
        ll = loglik_fn(i)
        return (ll - ll.max(dim=1, keepdim=True).values).exp()                 # (B, Nz), stabilized

    # Forward filter (skip the update where unobserved).
    prior = torch.softmax(-0.5 * z ** 2, dim=0)
    pi_pred = prior.unsqueeze(0).expand(B, Nz).clone()
    filtered = []
    for i in range(T_steps):
        w = lik(i) * pi_pred if obs(i) else pi_pred
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
        filtered.append(w)
        pi_pred = w @ T_full.t()
    filtered = torch.stack(filtered)                            # (T, B, Nz)

    # Backward smoother:  beta_t = p(obs x_{t+1:T} | z_t);  gamma_t ∝ filtered_t * beta_t.
    smoothed = [None] * T_steps
    smoothed[-1] = filtered[-1]
    beta = torch.ones(B, Nz, device=xs.device)
    for i in range(T_steps - 2, -1, -1):
        msg = (lik(i + 1) * beta) if obs(i + 1) else beta
        beta = msg @ T_full                                     # (B, Nz)
        beta = beta / (beta.sum(dim=1, keepdim=True) + 1e-12)
        g = filtered[i] * beta
        smoothed[i] = g / (g.sum(dim=1, keepdim=True) + 1e-12)
    return filtered, torch.stack(smoothed)


@torch.no_grad()
def ffbs_sample(z, T_full, filtered, n_samp):
    """Forward-filtering backward-sampling -> JOINT posterior trajectory samples
    z_{0:T} ~ p(z_{0:T} | x_{0:T}). filtered: (T,B,Nz). Returns (T, B, n_samp) z-values.
    """
    T_steps, B, Nz = filtered.shape
    idx = torch.zeros(T_steps, B, n_samp, dtype=torch.long, device=z.device)
    idx[-1] = torch.multinomial(filtered[-1], n_samp, replacement=True)      # z_{T-1} ~ filtered
    for i in range(T_steps - 2, -1, -1):
        k = idx[i + 1]                                          # (B, n_samp) next-state index
        Tk = T_full[k]                                          # (B, n_samp, Nz): P(z_{i+1}=k | z_i=j)
        w = filtered[i].unsqueeze(1) * Tk                       # backward-sampling weights over z_i
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-12)
        idx[i] = torch.multinomial(w.reshape(B * n_samp, Nz), 1).reshape(B, n_samp)
    return z[idx]                                               # (T, B, n_samp)


@torch.no_grad()
def grid_filter_target(xs, z_grid, a, sigma, noise_std, dt_obs, n_sub, mask=None):
    """Exact p(z_t | x_{0:t}) (filtered) and p(z_t | x_{0:T}) (smoothed) on the grid, using the TRUE
    drift f = a(z - z^3). `mask` (T,) bool: False = no observation (predict-only). Returns (T,B,Nz)."""
    f_true = a * (z_grid - z_grid ** 3)
    K = build_transition(f_true, z_grid, dt_obs / n_sub, sigma)
    K_full = transition_power(K, n_sub)
    filtered, smoothed = forward_backward(xs, z_grid, K_full, gauss_loglik_fn(xs, z_grid, noise_std),
                                          mask=mask)
    return filtered, smoothed


@torch.no_grad()
def grid_filter_highd(y, z_grid, a, sigma, noise_std, dt, n_sub, C, d, mask=None):
    """Exact p(z_t | y_{0:t}) (filtered) and p(z_t | y_{0:T}) (smoothed) on the grid: true f/g
    transition + high-D Gaussian likelihood N(y; C z + d, sigma^2 I), via the shared forward_backward.
    Returns (filtered, smoothed), each (T,B,Nz). The high-D validation oracle (uses TRUE C,d)."""
    f_true = a * (z_grid - z_grid ** 3)
    K = transition_power(build_transition(f_true, z_grid, dt / n_sub, sigma), n_sub)   # (Nz,Nz)
    return forward_backward(y, z_grid, K, highd_loglik_fn(y, z_grid, C, d, noise_std), mask=mask)


@torch.no_grad()
def grid_nll_masked(xs, z, T_full, noise_std, mask):
    """Exact grid-filter data NLL = -sum_i log c_i over OBSERVED steps, using the same
    (unnormalized-Gaussian) likelihood as the operator's SNIS evidence so the two are directly
    comparable -- the optimal the learned NLL should approach. mask (T,) bool."""
    T_steps, B = xs.shape[0], xs.shape[1]
    inv2var = 0.5 / noise_std ** 2
    pi_pred = torch.softmax(-0.5 * z ** 2, dim=0).unsqueeze(0).expand(B, z.numel()).clone()
    nll = torch.zeros(B, device=xs.device)
    for i in range(T_steps):
        if bool(mask[i]):
            w = torch.exp(-inv2var * (xs[i, :, 0:1] - z.unsqueeze(0)) ** 2) * pi_pred
            c = w.sum(1)
            nll = nll - torch.log(c + 1e-12)
            pi_pred = w / (c.unsqueeze(1) + 1e-12)
        pi_pred = pi_pred @ T_full.t()
    return nll.mean()
