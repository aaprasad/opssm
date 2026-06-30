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

"""STAGE 3, integrated EM -- one model that learns the dynamics AND filters.

The neural-operator Zakai filter is BOTH the model and the E-step. We learn the drift NOT
through the (weak) marginal filtering likelihood -- which fails (see neural_zakai_filter_learn.py,
drift L2 stuck ~1.2-1.4) -- but through the EM complete-data channel: regress f from the
operator's own inferred latent path. Online EM in a single training loop:

    operator update (every step):  train the mesh-free Zakai operator (FP residual + recursion)
                                    for the CURRENT f -- this is the E-step (infer the posterior).
    M-step (every m_every steps):   regress f_theta(z) ~ E[ (z_{t+1}-z_t)/dt | z_t ] from the
                                    operator's posterior-MEAN path, over OBSERVED steps only
                                    (where the mean is anchored to the data, not to f -- so the
                                    regression recovers the TRUE drift, not the current estimate).

A better f sharpens the operator, whose mean sharpens f. f never sees the weak gradient channel.
Validated against the exact grid filter (true f): drift L2 -> ~0.6 (recovered on the support)
and the operator filters as well as with the true drift.

To run:
python -m run.scripts.neural_zakai_filter_em
"""

import logging
import os

import fire
import torch
import tqdm
from torch import optim

from run.scripts.latent_sde_double_well_zakai import make_dataset
from run.scripts.neural_zakai_filter import (
    OperatorFilter, accumulate_pinn_grads, grid_filter_target, kl_target_pred)
from run.scripts.neural_zakai_filter_learn import DriftNet, DiffusionNet, vis


@torch.no_grad()
def posterior_mean(model, xs, mask, z_grid):
    """Operator filtering posterior mean E_pi[z] at each step -> (T,B). The E-step latent
    estimate; a grid sum here for the 1D readout (the model itself stays mesh-free)."""
    pi = model.log_posterior(xs, mask, z_grid).exp()
    return (pi * z_grid).sum(-1)


def main(
    batch_size=256, n_val=32, t0=0.0, t1=10.0, num_steps=100, a=1.0, sigma=0.6,
    noise_std=0.1, Nz=200, zmax=3.0, n_sub=5, gap_lo=0.4, gap_hi=0.7,
    n_scoll=4, n_tcoll=24, n_colloc=128, chunk_size=16, near_std=0.3, broad_std=1.6,
    gru_hidden=64, ctx_dim=64, p=64, drift_hidden=64, drift_lr=2e-3, lr=2e-3,
    num_iters=14000, warmup=2000, m_every=500, m_inner=400, reg_lambda=2e-3,
    learn_g=True, g_init=1.0,   # learn the diffusion g from the increment variance
    g_net=False, reg_lambda_g=3e-3,   # g_net: state-dependent g^2(z) network (M-step), else scalar
    pause_every=2000, train_dir="./dump/neural_zakai_em/",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    os.makedirs(train_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(train_dir, "train.log"), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(fh)

    xs, ts = make_dataset(t0=t0, t1=t1, batch_size=batch_size, noise_std=noise_std,
                          num_steps=num_steps, sigma=sigma, a=a, train_dir=train_dir, device=device)
    dt = float((ts[1] - ts[0]).item())
    z_grid = torch.linspace(-zmax, zmax, Nz, device=device)
    s_coll = torch.linspace(0.0, 1.0, n_scoll, device=device)

    mask_t = torch.ones(num_steps, dtype=torch.bool, device=device)
    mask_t[int(gap_lo * num_steps):int(gap_hi * num_steps)] = False
    mask = mask_t.view(num_steps, 1, 1).float().expand(num_steps, batch_size, 1).contiguous()
    filtered, _ = grid_filter_target(xs, z_grid, a, sigma, noise_std, dt, n_sub, mask=mask_t)

    xs_tr, xs_val = xs[:, n_val:], xs[:, :n_val]
    filt_val = filtered[:, :n_val]
    mask_tr, mask_val = mask[:, n_val:], mask[:, :n_val]
    m_tr = mask_tr[..., 0]                                  # (T,Btr) observed indicator

    f_true_grid = a * (z_grid - z_grid ** 3)
    supp = z_grid.abs() <= 2.0
    z_reg = torch.linspace(-2.0, 2.0, 80, device=device)   # grid for the H2 smoothness penalty
    hr = (z_reg[1] - z_reg[0]).item()

    def log_prior(z):
        return -0.5 * z ** 2

    model = OperatorFilter(1, gru_hidden, ctx_dim, p).to(device)
    drift_net = DriftNet(drift_hidden).to(device)
    drift_net.requires_grad_(False)                        # f is updated ONLY by the M-step
    op_opt = optim.Adam(model.parameters(), lr=lr)
    op_sched = optim.lr_scheduler.ExponentialLR(op_opt, gamma=0.9998)
    dr_opt = optim.Adam(drift_net.parameters(), lr=drift_lr)
    g_cur = g_init if learn_g else sigma           # learnable scalar diffusion (M-step), else true
    diff_net = dg_opt = None
    if learn_g and g_net:                          # state-dependent g^2(z) instead of a scalar
        diff_net = DiffusionNet(drift_hidden, g_init=g_init).to(device)
        diff_net.requires_grad_(False)             # g^2(z) updated ONLY by the M-step
        dg_opt = optim.Adam(diff_net.parameters(), lr=drift_lr)

    for step in tqdm.tqdm(range(1, num_iters + 1)):
        # ---- E-step: operator update for the current f (f frozen; only the operator moves)
        op_opt.zero_grad()
        drift = (lambda z: (torch.zeros_like(z), torch.zeros_like(z))) if step <= warmup \
            else drift_net.drift
        diffusion = diff_net.diffusion if (g_net and step > warmup) else g_cur
        res, jump, ic, _ = accumulate_pinn_grads(
            model, xs_tr, mask_tr, s_coll, drift, diffusion, log_prior, noise_std, dt,
            n_colloc, near_std, broad_std, n_tcoll, chunk_size, w_nll=0.0)
        op_opt.step(); op_sched.step()

        # ---- M-step: regress f from the operator's posterior-mean path (observed steps only)
        if step > warmup and step % m_every == 0:
            z_hat = posterior_mean(model, xs_tr, mask_tr, z_grid)          # (T,Btr), no grad
            valid = (m_tr[:-1] * m_tr[1:]).bool()                          # data-anchored increments
            zc = z_hat[:-1][valid]
            zc_next = z_hat[1:][valid]
            dz = ((z_hat[1:] - z_hat[:-1]) / dt)[valid]
            drift_net.requires_grad_(True)
            for _ in range(m_inner):
                dr_opt.zero_grad()
                f = drift_net.net(zc.unsqueeze(-1)).squeeze(-1)
                fit = ((f - dz) ** 2).mean()
                fr = drift_net.net(z_reg.unsqueeze(-1)).squeeze(-1)
                f_pp = (fr[2:] - 2 * fr[1:-1] + fr[:-2]) / hr ** 2          # f''(z)
                curv = (f_pp ** 2).mean()                                  # H2 smoothness penalty
                (fit + reg_lambda * curv).backward()
                dr_opt.step()
            drift_net.requires_grad_(False)
            # M-step for g: the increments' residual after the drift has variance g^2/dt, so
            # E[(dz - f(z))^2] * dt = g^2(z) is the conditional variance.  Scalar: average it.
            # Network: regress g^2(z) on the per-sample target dt*r^2 (its conditional mean is
            # g^2(z)) with an H2 smoothness penalty -- same structure as the drift fit.
            if learn_g:
                # trapezoidal drift 1/2(f(z_t)+f(z_t+1)) removes the deterministic excursion over
                # dt -- the finite-dt bias that inflates the increment variance near the unstable
                # barrier (diagnosed to recover flat g^2 on the true path).
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
                    with torch.no_grad():        # scalar summary of g(z) for the log
                        g_cur = float(diff_net._g2(z_reg.unsqueeze(-1)).clamp(min=1e-6).sqrt().mean())
                else:
                    g_cur = float((r.pow(2).mean() * dt).sqrt().clamp(min=0.05))

        if step % pause_every == 0:
            with torch.no_grad():
                kl_val = kl_target_pred(
                    filt_val, model.log_posterior(xs_val, mask_val, z_grid)).item()
                f_err = (drift_net.drift(z_grid)[0][supp] - f_true_grid[supp]
                         ).pow(2).mean().sqrt().item()
            logging.warning(
                f"step {step:05d}, res: {res:.4f}, jump: {jump:.4f}, ic: {ic:.4f}, "
                f"drift L2 err: {f_err:.4f}, g: {g_cur:.4f} (true {sigma}), KL(post): {kl_val:.4f}")
            vis(model, drift_net, xs_val, mask_val, filt_val, z_grid, ts, a,
                os.path.join(train_dir, f"step_{step:05d}.pdf"), diff_net=diff_net, sigma=sigma)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    fire.Fire(main)
