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

"""Neural mesh-free Zakai filter -- HIGH-DIMENSIONAL OBSERVATIONS (Duncker-style baseline).

A low-D latent double-well z_t is observed through a high-D LINEAR sensor
    y_t = C z_t + d + noise        (y in R^D, z in R^1),
and from y alone we recover: the filtering posterior p(z_t | y_{0:t}) (the operator, which
filters the LOW-D latent -- where mesh-free shines), the latent drift f and diffusion g
(EM M-step on the inferred latent path), and -- with learn_obs=True -- the observation map
(C, d) (regression of y on the inferred latent). This is the latent-SDE-from-high-D-observations
setting of Duncker et al. (2019); we match that baseline first. Figure: (A) obs + reconstruction,
(B) latent posterior vs true, (C) learned drift, (D) learned C,d.

The operator's encoder ingests D-dim y; the Zakai likelihood becomes N(y; C z + d, sigma^2 I)
via the `decode` hook; the collocation centers on a pseudo-inverse latent estimate C^+(y-d).

To run:
python -m run.scripts.neural_zakai_filter_highd
"""

import logging
import math
import os

import fire
import matplotlib.pyplot as plt
import torch
import tqdm
from torch import optim

from run.scripts.latent_sde_double_well_zakai import build_transition, transition_power
from run.scripts.neural_zakai_filter import (
    OperatorFilter, accumulate_pinn_grads, kl_target_pred)
from run.scripts.neural_zakai_filter_learn import DriftNet, DiffusionNet


def make_dataset_highd(t0, t1, batch_size, num_steps, sigma, a, obs_dim, noise_std,
                       c_scale, n_sub, device, seed=0):
    """Latent double-well SDE z (T,B) + high-D linear obs y = C z + d + noise (T,B,D).
    Returns z, y, ts, C (D,), d (D,)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    dt = (t1 - t0) / num_steps
    sub = dt / n_sub
    ts = torch.linspace(t0, t1, num_steps, device=device)
    z = torch.zeros(num_steps, batch_size, device=device)
    z[0] = torch.randn(batch_size, device=device, generator=gen)
    for i in range(1, num_steps):
        zz = z[i - 1].clone()
        for _ in range(n_sub):
            zz = zz + a * (zz - zz ** 3) * sub + sigma * math.sqrt(sub) * \
                torch.randn(batch_size, device=device, generator=gen)
        z[i] = zz
    C = c_scale * torch.randn(obs_dim, device=device, generator=gen)
    d = torch.randn(obs_dim, device=device, generator=gen)
    y = C * z[..., None] + d + noise_std * torch.randn(num_steps, batch_size, obs_dim,
                                                       device=device, generator=gen)
    return z, y, ts, C, d


@torch.no_grad()
def grid_filter_highd(y, z_grid, a, sigma, noise_std, dt, n_sub, C, d):
    """Exact p(z_t | y_{0:t}) on the grid: true f/g transition + high-D Gaussian likelihood
    N(y; C z + d, sigma^2 I). Returns (T,B,Nz). The validation oracle (uses the TRUE C,d)."""
    f_true = a * (z_grid - z_grid ** 3)
    K = transition_power(build_transition(f_true, z_grid, dt / n_sub, sigma), n_sub)   # (Nz,Nz)
    T, B, _ = y.shape
    hz = C[None, :] * z_grid[:, None] + d[None, :]               # (Nz, D)
    pi = torch.softmax(-0.5 * z_grid ** 2, dim=0)[None, :].expand(B, z_grid.numel()).clone()
    filt = torch.zeros(T, B, z_grid.numel(), device=y.device)
    for t in range(T):
        ll = -0.5 * ((y[t][:, None, :] - hz[None, :, :]) ** 2).sum(-1) / noise_std ** 2   # (B,Nz)
        w = pi * (ll - ll.max(-1, keepdim=True).values).exp()
        pi = w / w.sum(-1, keepdim=True).clamp_min(1e-30)
        filt[t] = pi
        pi = pi @ K.t()                                         # predict
    return filt


@torch.no_grad()
def posterior_mean(model, y, mask, z_grid):
    """Operator posterior mean E[z_t | y_{0:t}] on the grid -> (T,B)."""
    log_pi = model.log_posterior(y, mask, z_grid)
    return (log_pi.exp() * z_grid).sum(-1)


def zhat_from_obs(y, C, d):
    """Pseudo-inverse latent estimate z_hat = C^+ (y - d) per (T,B) -- the collocation center."""
    return ((y - d) * C).sum(-1) / (C * C).sum().clamp_min(1e-8)


@torch.no_grad()
def vis_highd(model, drift_net, diff_net, y_val, mask_val, z_val_true, filt_val, z_grid, ts,
              a, sigma, C_cur, d_cur, C_true, d_true, learn_obs, img_path, n_traj=2, s_scale=1.0):
    zg = z_grid.cpu().numpy(); ts_np = ts.cpu().numpy()
    log_pi = model.log_posterior(y_val, mask_val, z_grid)
    pi = log_pi.exp()
    m_op = (pi * z_grid).sum(-1)                                 # (T,B)
    s_op = (pi * z_grid ** 2).sum(-1).sub(m_op ** 2).clamp_min(0).sqrt()
    ex_m = (filt_val * z_grid).sum(-1)
    recon = s_scale * C_cur * m_op[..., None] + d_cur            # (T,B,D)
    nd = min(2, y_val.shape[-1])

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for j in range(n_traj):                                     # A: obs + reconstruction
        ax = axes[0, j]
        for dim in range(nd):
            ax.plot(ts_np, y_val[:, j, dim].cpu().numpy(), ".", ms=3, alpha=0.35, color=f"C{dim}")
            ax.plot(ts_np, recon[:, j, dim].cpu().numpy(), "-", lw=2, color=f"C{dim}",
                    label=f"$y_{dim}$" if j == 0 else None)
        ax.set_title(f"obs + recon, traj {j}"); ax.set_xlabel("$t$")
        if j == 0:
            ax.legend(fontsize=8)
    for j in range(n_traj):                                     # B: latent posterior vs true
        ax = axes[1, j]
        ax.plot(ts_np, z_val_true[:, j].cpu().numpy(), "k-", lw=2, label="true $z$")
        ax.plot(ts_np, ex_m[:, j].cpu().numpy(), "C7-", lw=1.2, label="exact mean")
        ax.plot(ts_np, m_op[:, j].cpu().numpy(), "r--", lw=2, label="operator mean")
        ax.fill_between(ts_np, (m_op[:, j] - 2 * s_op[:, j]).cpu().numpy(),
                        (m_op[:, j] + 2 * s_op[:, j]).cpu().numpy(), color="r", alpha=0.2)
        ax.set_title(f"latent $z$, traj {j}"); ax.set_xlabel("$t$")
        if j == 0:
            ax.legend(fontsize=8)
    ax = axes[0, 2]                                            # C: drift
    ax.plot(zg, (a * (z_grid - z_grid ** 3)).cpu().numpy(), "k-", lw=2, label="true $a(z-z^3)$")
    ax.plot(zg, drift_net.drift(z_grid)[0].cpu().numpy(), "C2--", lw=2, label=r"learned $f_\theta$")
    ax.set_ylim(-4, 4); ax.set_xlim(-2.5, 2.5); ax.set_xlabel("$z$"); ax.set_ylabel("$f(z)$")
    ax.set_title("learned drift"); ax.legend(fontsize=8)
    ax = axes[1, 2]                                            # diffusion
    if diff_net is not None:
        ax.plot(zg, diff_net._g2(z_grid.unsqueeze(-1)).squeeze(-1).cpu().numpy(),
                "C0-", lw=2, label=r"learned $g^2(z)$")
    ax.axhline(sigma ** 2, ls="--", c="k", lw=2, label=fr"true $\sigma^2={sigma ** 2:.3f}$")
    ax.set_ylim(0, max(0.6, sigma ** 2 * 2)); ax.set_xlabel("$z$"); ax.set_ylabel("$g^2(z)$")
    ax.set_title("learned diffusion"); ax.legend(fontsize=8)
    ax = axes[0, 3]                                            # D: obs map (unit C direction)
    if learn_obs:
        C_true_u = C_true / C_true.norm()
        cos = float(C_cur @ C_true_u)
        flip = 1.0 if cos >= 0 else -1.0                       # align the sign gauge
        ax.plot(C_true_u.cpu().numpy(), (flip * C_cur).cpu().numpy(), "C0o", label="$C$ dir")
        lim = float(C_true_u.abs().max()) * 1.2
        ax.plot([-lim, lim], [-lim, lim], "k:", lw=1)
        ax.set_xlabel("true unit $C$"); ax.set_ylabel("learned (sign-aligned)")
        ax.set_title(fr"obs map $C$:  $|\cos|$={abs(cos):.3f}"); ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "C, d fixed (known)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("observation map $C, d$")
    axes[1, 3].axis("off")
    plt.tight_layout(); plt.savefig(img_path); plt.close()


def main(
    batch_size=256, n_val=32, t0=0.0, t1=10.0, num_steps=100, a=1.0, sigma=0.6,
    obs_dim=10, noise_std=0.3, c_scale=1.0, n_sub=5, Nz=200, zmax=3.0,
    n_scoll=4, n_tcoll=24, n_colloc=128, chunk_size=16, near_std=0.3, broad_std=1.6,
    gru_hidden=64, ctx_dim=64, p=64, drift_hidden=64, drift_lr=2e-3, lr=2e-3,
    num_iters=14000, warmup=2000, m_every=2000, m_inner=400, reg_lambda=3e-4,
    learn_g=True, g_net=False, reg_lambda_g=3e-3, g_init=1.0,
    learn_obs=False,            # stage 1: fixed known C,d.  stage 2: learn C,d by regression
    pca_init=True,              # init the Stiefel C to the top PCA direction; False = random
                                # direction (tests whether the EM, not PCA, recovers the sensor)
    c_stable_tol=0.05,          # learn the drift only once the sensor direction is stable to this
                                # (1 - |cos| between successive C); blocks fitting f on a bad latent
    pause_every=2000, train_dir="./dump/neural_zakai_highd/",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    os.makedirs(train_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(train_dir, "train.log"), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(fh)

    z_true, y, ts, C_true, d_true = make_dataset_highd(
        t0, t1, batch_size, num_steps, sigma, a, obs_dim, noise_std, c_scale, n_sub, device)
    dt = float((ts[1] - ts[0]).item())
    z_grid = torch.linspace(-zmax, zmax, Nz, device=device)
    s_coll = torch.linspace(0.0, 1.0, n_scoll, device=device)
    mask = torch.ones(num_steps, batch_size, 1, device=device)   # dense obs (no gap)

    y_tr, y_val = y[:, n_val:], y[:, :n_val]
    z_val_true = z_true[:, :n_val]
    mask_tr, mask_val = mask[:, n_val:], mask[:, :n_val]
    m_tr = mask_tr[..., 0]
    filt_val = grid_filter_highd(y_val, z_grid, a, sigma, noise_std, dt, n_sub, C_true, d_true)

    f_true_grid = a * (z_grid - z_grid ** 3)
    supp = z_grid.abs() <= 2.0
    z_reg = torch.linspace(-2.0, 2.0, 80, device=device)
    hr = (z_reg[1] - z_reg[0]).item()

    def log_prior(z):
        return -0.5 * z ** 2

    # observation map. Stage 1: fixed-true. Stage 2 (learn_obs): a STIEFEL decoder -- C has unit
    # (orthonormal) columns, the latent->obs SCALE s is fixed from the data (s = sigma_1/sqrt(N),
    # the principal spread), so neither C nor the scale can run away (the free-scale OLS version
    # collapses the latent). C is PCA-initialized to the top principal direction (~ the true sensor
    # direction up to sign) so the operator starts in the right subspace, not an orthogonal one.
    s_scale = 1.0
    if learn_obs:
        Yc = y_tr.reshape(-1, obs_dim)
        ybar = Yc.mean(0)
        _, Sv, Vt = torch.linalg.svd(Yc - ybar, full_matrices=False)
        s_scale = float(Sv[0] / math.sqrt(Yc.shape[0]))          # fixed latent->obs scale (data magnitude)
        if pca_init:
            C_cur = Vt[0].clone()                                # PCA: top principal direction
        else:
            C_cur = torch.randn(obs_dim, device=device)          # random unit direction (no PCA shortcut)
            C_cur = C_cur / C_cur.norm()
        d_cur = ybar.clone()
    else:
        C_cur, d_cur = C_true.clone(), d_true.clone()

    def decode(z):                                               # y = s * C * z + d ; C orthonormal
        return s_scale * C_cur * z[..., None] + d_cur

    model = OperatorFilter(obs_dim, gru_hidden, ctx_dim, p).to(device)
    drift_net = DriftNet(drift_hidden).to(device)
    drift_net.requires_grad_(False)
    op_opt = optim.Adam(model.parameters(), lr=lr)
    op_sched = optim.lr_scheduler.ExponentialLR(op_opt, gamma=0.9998)
    dr_opt = optim.Adam(drift_net.parameters(), lr=drift_lr)
    g_cur = g_init if learn_g else sigma
    diff_net = dg_opt = None
    if learn_g and g_net:
        diff_net = DiffusionNet(drift_hidden, g_init=g_init).to(device)
        diff_net.requires_grad_(False)
        dg_opt = optim.Adam(diff_net.parameters(), lr=drift_lr)

    for step in tqdm.tqdm(range(1, num_iters + 1)):
        op_opt.zero_grad()
        drift = (lambda z: (torch.zeros_like(z), torch.zeros_like(z))) if step <= warmup \
            else drift_net.drift
        diffusion = diff_net.diffusion if (g_net and step > warmup) else g_cur
        center = zhat_from_obs(y_tr, C_cur, d_cur) / s_scale     # latent estimate for collocation
        res, jump, ic, _ = accumulate_pinn_grads(
            model, y_tr, mask_tr, s_coll, drift, diffusion, log_prior, noise_std, dt,
            n_colloc, near_std, broad_std, n_tcoll, chunk_size, decode=decode, center=center)
        op_opt.step(); op_sched.step()

        if step > warmup and step % m_every == 0:
            z_hat = posterior_mean(model, y_tr, mask_tr, z_grid)            # (T,Btr)
            valid = (m_tr[:-1] * m_tr[1:]).bool()
            zc = z_hat[:-1][valid]
            zc_next = z_hat[1:][valid]
            dz = ((z_hat[1:] - z_hat[:-1]) / dt)[valid]
            # ---- M-step: Stiefel obs map FIRST (Procrustes: C = unit direction of the cross-
            # covariance; scale fixed -> no runaway). cstab = how far the sensor direction moved
            # since the last M-step (gauge-free), used to gate the drift below.
            cstab = 0.0
            if learn_obs:
                with torch.no_grad():
                    zf = z_hat.reshape(-1)
                    yf = y_tr.reshape(-1, obs_dim)
                    M = ((yf - yf.mean(0)) * (zf - zf.mean())[:, None]).sum(0)   # cross-cov (D,)
                    C_new = M / M.norm().clamp_min(1e-8)
                    cstab = 1.0 - abs(float(C_new @ C_cur))       # 0 = sensor direction unchanged
                    C_cur, d_cur = C_new, yf.mean(0) - s_scale * C_new * zf.mean()
            # ---- M-step: drift -- ONLY once the sensor has stabilized; otherwise f is fit on a
            # misaligned latent and the warm-started net stays stuck there (sensor-before-dynamics)
            if cstab < c_stable_tol:
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
            if learn_g:                                                    # ---- M-step: diffusion
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
                        g_cur = float(diff_net._g2(z_reg.unsqueeze(-1)).clamp(min=1e-6).sqrt().mean())
                else:
                    g_cur = float((r.pow(2).mean() * dt).sqrt().clamp(min=0.05))

        if step % pause_every == 0:
            with torch.no_grad():
                kl_val = kl_target_pred(
                    filt_val, model.log_posterior(y_val, mask_val, z_grid)).item()
                f_err = (drift_net.drift(z_grid)[0][supp] - f_true_grid[supp]
                         ).pow(2).mean().sqrt().item()
                # obs-map recovery: |cos| between learned unit C and the true direction (->1 = right
                # subspace; sign-free since the latent carries a sign gauge)
                ccos = float((C_cur @ (C_true / C_true.norm())).abs())
            logging.warning(
                f"step {step:05d}, res: {res:.4f}, jump: {jump:.4f}, ic: {ic:.4f}, "
                f"drift L2: {f_err:.4f}, g: {g_cur:.4f} (true {sigma}), "
                f"C cos: {ccos:.4f}, KL(post): {kl_val:.4f}")
            vis_highd(model, drift_net, diff_net, y_val, mask_val, z_val_true, filt_val, z_grid,
                      ts, a, sigma, C_cur, d_cur, C_true, d_true, learn_obs,
                      os.path.join(train_dir, f"step_{step:05d}.pdf"), s_scale=s_scale)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    fire.Fire(main)
