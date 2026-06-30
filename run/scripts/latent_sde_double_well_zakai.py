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

"""Learn the double-well drift via a differentiable Bayesian filter (discrete Zakai).

Instead of a Fokker-Planck PRIOR (data-blind) we use the Zakai / filtering
equation, whose predict-update recursion is the discretized Zakai and whose data
marginal likelihood is the principled objective for learning the drift.

Signal:        dz = f(z) dt + sigma dW           (f learnable, sigma known)
Observations:  x_i = z_i + eps_i,  eps_i ~ N(0, rho^2)   (discrete, noisy)

Filter (probability mass pi on a z-grid, batched over trajectories):
  predict:  pi^- = T_full @ pi          (T = EM transition, the FP / Zakai drift term)
  update:   c = sum_k N(x_i; z_k, rho^2) pi^-_k ;  pi = (likelihood * pi^-)/c   (data term)
  loss:     -sum_i log c_i  =  -log p(x_{0:T})       (maximize data likelihood)

The transition matrix is autonomous, so it's ONE shared matrix built each step
from the learnable drift f_net (substepped for the stiff cubic). Only f is
learned (sigma, rho known). No encoder, no variational inference -- this is EXACT
inference (only error = grid/dt discretization), not an ELBO.

Once the drift is learned, the same transition T gives EXACT latent-state
inference by forward-backward on the grid (no VI):
  filtered  p(z_t | x_{0:t})   -- forward pass
  smoothed  p(z_t | x_{0:T})   -- backward pass (forward_backward)
  trajectories  z_{0:T} ~ p(z_{0:T} | x_{0:T})  -- FFBS (ffbs_sample)
and the recovered prior is the stationary distribution of T (the FP equilibrium
of the learned f), the bimodal p(z) ~ exp(-2 V/sigma^2), V = z^4/4 - z^2/2.

To run:
python -m run.scripts.latent_sde_double_well_zakai
"""

import logging
import math
import os

import fire
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import tqdm
from torch import nn
from torch import optim

import torchsde


class DoubleWell(object):
    """Stochastic 1D double-well system: dX = a (X - X^3) dt + sigma dW."""

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, a: float = 1.0, sigma: float = 0.6):
        self.a = a
        self.sigma = sigma

    def f(self, t, y):
        return self.a * (y - y ** 3)

    def g(self, t, y):
        return torch.full_like(y, self.sigma)

    @torch.no_grad()
    def sample(self, x0, ts, noise_std):
        xs = torchsde.sdeint(self, x0, ts, dt=1e-2)
        if noise_std > 0:
            xs = xs + torch.randn_like(xs) * noise_std
        return xs


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


def filter_nll(xs, z, T_full, rho, return_filters=False):
    """Run the batched grid filter; return per-trajectory NLL = -sum_i log c_i.

    xs: (T_steps, B, 1) observations. z: (Nz,) grid (mass representation).
    """
    T_steps, B = xs.shape[0], xs.shape[1]
    Nz = z.shape[0]
    inv2var = 0.5 / rho ** 2
    norm = 1.0 / (math.sqrt(2 * math.pi) * rho)

    # Prior mass N(0,1) on the grid (prediction at t_0 before any observation).
    prior = torch.softmax(-0.5 * z ** 2, dim=0)      # (Nz,) sums to 1
    pi_pred = prior.unsqueeze(0).expand(B, Nz)        # (B, Nz)

    nll = torch.zeros(B, device=xs.device)
    filters = [] if return_filters else None
    for i in range(T_steps):
        x = xs[i, :, 0]                               # (B,)
        ell = norm * torch.exp(-inv2var * (x.unsqueeze(1) - z.unsqueeze(0)) ** 2)  # (B, Nz)
        w = ell * pi_pred                             # unnormalized posterior mass
        c = w.sum(dim=1)                              # (B,) one-step evidence
        nll = nll - torch.log(c + 1e-12)
        pi = w / (c.unsqueeze(1) + 1e-12)             # (B, Nz) posterior mass
        if return_filters:
            filters.append(pi.detach())
        pi_pred = pi @ T_full.t()                     # predict to next step
    if return_filters:
        return nll, filters
    return nll


def stationary_density(T, n_iter=4000):
    """Stationary mass distribution of the (single-substep) transition by iteration."""
    Nz = T.shape[0]
    pi = torch.full((Nz,), 1.0 / Nz, device=T.device)
    for _ in range(n_iter):
        pi = T @ pi
    return pi


@torch.no_grad()
def forward_backward(xs, z, T_full, rho, mask=None):
    """Forward filter + backward smoother (discrete HMM on the grid) -- EXACT
    latent-state inference, no variational approximation.

    mask: optional (T_steps,) bool; False = no observation at that step (the
    update is skipped, so the state is propagated by the dynamics alone). Lets us
    infer through gaps -- where the smoother visibly beats the filter.

    Returns (filtered, smoothed), each (T_steps, B, Nz) mass distributions:
      filtered[t] = p(z_t | observed x_{0:t})   (causal / online)
      smoothed[t] = p(z_t | observed x_{0:T})   (all observations)
    """
    T_steps, B = xs.shape[0], xs.shape[1]
    Nz = z.shape[0]
    inv2var = 0.5 / rho ** 2

    def obs(i):
        return mask is None or bool(mask[i])

    def lik(i):
        x = xs[i, :, 0]
        return torch.exp(-inv2var * (x.unsqueeze(1) - z.unsqueeze(0)) ** 2)   # (B, Nz)

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


class DriftNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Softplus(),
            nn.Linear(hidden, hidden), nn.Softplus(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        return self.net(z.unsqueeze(-1)).squeeze(-1)


def make_dataset(t0, t1, batch_size, noise_std, num_steps, sigma, a, train_dir, device):
    data_path = os.path.join(train_dir, f"double_well_raw_sigma{sigma}.pth")
    if os.path.exists(data_path):
        data_dict = torch.load(data_path)
        xs, ts = data_dict["xs"], data_dict["ts"]
        logging.warning(f"Loaded toy data at: {data_path}")
        if xs.shape[1] != batch_size:
            raise ValueError("Batch size has changed; delete and regenerate the data.")
        if ts[0] != t0 or ts[-1] != t1:
            raise ValueError("Time interval changed; delete and regenerate the data.")
    else:
        _y0 = torch.randn(batch_size, 1, device=device)
        ts = torch.linspace(t0, t1, steps=num_steps, device=device)
        xs = DoubleWell(a=a, sigma=sigma).sample(_y0, ts, noise_std)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        torch.save({"xs": xs, "ts": ts}, data_path)
        logging.warning(f"Stored toy data at: {data_path}")
    return xs, ts


def vis(xs, ts, z, drift_net, T_single, T_full, sigma, a, rho, img_path, n_traj=3, n_samp=6):
    z_np = z.cpu().numpy()
    ts_np = ts.cpu().numpy()
    hz = (z[1] - z[0])
    with torch.no_grad():
        f_learned = drift_net(z).cpu().numpy()
        f_true = (a * (z - z ** 3)).cpu().numpy()
        pi_eq = stationary_density(T_single)
        p_eq = (pi_eq / hz).cpu().numpy()
        p_ana = analytic_stationary_density(z, sigma).cpu().numpy()
        # exact inference on the first n_traj trajectories, THROUGH A MASKED GAP
        T_steps = xs.shape[0]
        g0, g1 = int(0.35 * T_steps), int(0.65 * T_steps)        # mask the middle ~30%
        mask = torch.ones(T_steps, dtype=torch.bool, device=z.device)
        mask[g0:g1] = False
        xsub = xs[:, :n_traj]
        filtered, smoothed = forward_backward(xsub, z, T_full, rho, mask=mask)  # (T, n_traj, Nz)
        samp = ffbs_sample(z, T_full, filtered, n_samp).cpu().numpy()           # (T, n_traj, n_samp)
        sm_mean = (smoothed * z).sum(-1)                                     # (T, n_traj)
        sm_std = ((smoothed * z ** 2).sum(-1) - sm_mean ** 2).clamp_min(0).sqrt()
        sm_mean = sm_mean.cpu().numpy(); sm_std = sm_std.cpu().numpy()
        mask_np = mask.cpu().numpy()

    data = xs.cpu().numpy()[..., 0]
    lo, hi = float(data.min()), float(data.max())
    rng = (lo - 0.3, hi + 0.3)

    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(2, 3)

    # learned vs true drift
    ax = fig.add_subplot(gs[0, 0])
    ax.axhline(0, color="gray", lw=0.5)
    ax.plot(z_np, f_true, "k--", lw=2, label="true $f(z)$")
    ax.plot(z_np, f_learned, "C3", lw=2, label="learned $f(z)$")
    ax.set_xlim(rng); ax.set_ylim(-3, 3)
    ax.set_xlabel("$z$"); ax.set_ylabel("drift"); ax.set_title("learned vs true drift")
    ax.legend(fontsize=10)

    # recovered prior (stationary of T) vs analytic vs data marginal
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(data.reshape(-1), bins=80, range=rng, density=True, color="C0", alpha=0.3, label="data marginal")
    ax.plot(z_np, p_ana, "k--", lw=2, label="analytic $p(z)$")
    ax.plot(z_np, p_eq, "C3", lw=2, label="stationary of $T$")
    ax.set_xlim(rng); ax.set_xlabel("$z$"); ax.set_ylabel("density")
    ax.set_title("recovered prior (FP equilibrium of learned $f$)")
    ax.legend(fontsize=9)

    # filtered vs smoothed marginal at the gap center (traj 0) -- smoothing visibly tightens
    mid = (g0 + g1) // 2
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(z_np, (filtered[mid, 0] / hz).cpu().numpy(), "C1", lw=1.5, label="filtered (past + dynamics)")
    ax.plot(z_np, (smoothed[mid, 0] / hz).cpu().numpy(), "C2", lw=1.5, label="smoothed (all obs)")
    ax.set_xlim(rng); ax.set_xlabel("$z$")
    ax.set_title(f"filtered vs smoothed at $t_{{{mid}}}$ (inside gap)"); ax.legend(fontsize=8)

    # exact latent trajectory inference THROUGH THE GAP for the first n_traj trajectories
    for j in range(n_traj):
        ax = fig.add_subplot(gs[1, j])
        ax.axvspan(ts_np[g0], ts_np[g1 - 1], color="gray", alpha=0.15, label="no obs (gap)")
        ax.plot(ts_np[mask_np], data[mask_np, j], "C0.", ms=3, label="obs $x$")
        for s in range(n_samp):
            ax.plot(ts_np, samp[:, j, s], color="C2", lw=0.6, alpha=0.45)
        ax.plot(ts_np, sm_mean[:, j], "C3", lw=1.8, label="smoothed mean")
        ax.fill_between(ts_np, sm_mean[:, j] - 2 * sm_std[:, j], sm_mean[:, j] + 2 * sm_std[:, j],
                        color="C3", alpha=0.2, label="$\\pm 2\\sigma$")
        ax.set_xlabel("$t$"); ax.set_ylabel("$z$")
        ax.set_title(f"latent inference through gap — traj {j}", fontsize=11)
        if j == 0:
            ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()


def main(
    batch_size=512,
    hidden=64,
    lr_init=3e-3,
    lr_gamma=0.999,
    t0=0.0,
    t1=10.0,
    num_steps=100,
    a=1.0,
    sigma=0.6,
    noise_std=0.05,
    Nz=400,
    zmax=3.0,
    n_sub=5,
    num_iters=3000,
    pause_every=50,
    grad_clip=1.0,
    train_dir="./dump/double_well_zakai/",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xs, ts = make_dataset(
        t0=t0, t1=t1, batch_size=batch_size, noise_std=noise_std, num_steps=num_steps,
        sigma=sigma, a=a, train_dir=train_dir, device=device,
    )
    dt = float((ts[1] - ts[0]).item())
    dt_sub = dt / n_sub
    z = torch.linspace(-zmax, zmax, Nz, device=device)
    p_ana = analytic_stationary_density(z, sigma)
    f_true = a * (z - z ** 3)

    drift_net = DriftNet(hidden=hidden).to(device)
    optimizer = optim.Adam(drift_net.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

    for global_step in tqdm.tqdm(range(1, num_iters + 1)):
        optimizer.zero_grad()
        f_grid = drift_net(z)                                   # (Nz,)
        T = build_transition(f_grid, z, dt_sub, sigma)          # single substep transition
        T_full = transition_power(T, n_sub)                     # over one observation interval
        nll = filter_nll(xs, z, T_full, noise_std)              # (B,)
        loss = nll.mean()
        if not torch.isfinite(loss):
            logging.warning(f"global_step: {global_step:06d}, non-finite loss; skipping")
            continue
        loss.backward()
        for p in drift_net.parameters():
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(drift_net.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        if global_step % pause_every == 0:
            with torch.no_grad():
                l2_f = torch.sqrt(torch.trapezoid((f_grid - f_true) ** 2, x=z)).item()
                pi_eq = stationary_density(T)
                p_eq = pi_eq / (z[1] - z[0])
                l2_p = torch.sqrt(torch.trapezoid((p_eq - p_ana) ** 2, x=z)).item()
            lr_now = optimizer.param_groups[0]["lr"]
            logging.warning(
                f"global_step: {global_step:06d}, lr: {lr_now:.5f}, nll: {loss.item():.4f}, "
                f"L2(f-true): {l2_f:.4f}, L2(p_eq-ana): {l2_p:.4f}"
            )
            img_path = os.path.join(train_dir, f"global_step_{global_step:06d}.pdf")
            vis(xs, ts, z, drift_net, T.detach(), T_full.detach(), sigma, a, noise_std, img_path)


if __name__ == "__main__":
    fire.Fire(main)
