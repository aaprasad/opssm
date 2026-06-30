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

"""Plotting: operator-vs-exact-filter, drift/diffusion learning, high-D Duncker panels."""

import matplotlib.pyplot as plt
import torch


# ---------------------------------------------------------------------------
# Visualization: operator posterior vs exact grid filter on HELD-OUT sequences.
# ---------------------------------------------------------------------------
@torch.no_grad()
def vis_operator(model, xs_val, mask_val, filt_val, z_grid, ts, img_path, n_traj=3):
    z = z_grid.cpu().numpy()
    ts_np = ts.cpu().numpy()
    log_pi = model.log_posterior(xs_val, mask_val, z_grid)  # (T,Bv,Nz)
    pi = log_pi.exp().cpu().numpy()
    filt = filt_val.cpu().numpy()
    data = xs_val.cpu().numpy()[..., 0]                     # (T,Bv)
    obs = mask_val[:, 0, 0].cpu().numpy().astype(bool)      # (T,) observed
    gap = ~obs

    # pick, per trajectory, the time of MAX spread in the exact filter (= where the
    # posterior goes bimodal, typically inside the observation gap).
    mean = (filt_val * z_grid).sum(-1)                     # (T,Bv)
    var = (filt_val * z_grid ** 2).sum(-1) - mean ** 2
    t_star = var[2:-2].argmax(dim=0).cpu().numpy() + 2     # (Bv,)

    fig, axes = plt.subplots(2, n_traj, figsize=(6 * n_traj, 9))
    for j in range(n_traj):
        tj = int(t_star[j])
        ax = axes[0, j]
        ax.plot(z, filt[tj, j], "k-", lw=2, label="exact filter")
        ax.plot(z, pi[tj, j], "C3--", lw=2, label="operator")
        if obs[tj]:
            ax.axvline(data[tj, j], color="C0", lw=1, ls=":", label="obs $x_t$")
        ax.set_xlabel("$z$"); ax.set_title(
            f"held-out traj {j}, $t={tj}$ ({'in gap' if gap[tj] else 'observed'})")
        if j == 0:
            ax.legend(fontsize=9)

        m_op = (log_pi[:, j].exp() * z_grid).sum(-1).cpu().numpy()
        s_op = ((log_pi[:, j].exp() * z_grid ** 2).sum(-1).cpu().numpy() - m_op ** 2).clip(0) ** 0.5
        m_ex = mean[:, j].cpu().numpy()
        ax = axes[1, j]
        if gap.any():
            ax.axvspan(ts_np[gap][0], ts_np[gap][-1], color="gray", alpha=0.15, label="no obs")
        ax.plot(ts_np[obs], data[obs, j], "C0.", ms=3, label="obs")
        ax.plot(ts_np, m_ex, "k-", lw=2, label="exact mean")
        ax.plot(ts_np, m_op, "C3--", lw=2, label="operator mean")
        ax.fill_between(ts_np, m_op - 2 * s_op, m_op + 2 * s_op, color="C3", alpha=0.2)
        ax.set_xlabel("$t$"); ax.set_ylabel("$z$")
        if j == 0:
            ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(img_path); plt.close()


@torch.no_grad()
def vis_learn(model, drift_net, xs_val, mask_val, filt_val, z_grid, ts, a, img_path, n_traj=2,
        diff_net=None, sigma=None):
    z = z_grid.cpu().numpy()
    ts_np = ts.cpu().numpy()
    f_learned = drift_net.drift(z_grid)[0].cpu().numpy()
    f_true = (a * (z_grid - z_grid ** 3)).cpu().numpy()
    log_pi = model.log_posterior(xs_val, mask_val, z_grid)
    pi = log_pi.exp().cpu().numpy()
    filt = filt_val.cpu().numpy()
    mean = (filt_val * z_grid).sum(-1)
    var = (filt_val * z_grid ** 2).sum(-1) - mean ** 2
    t_star = var[2:-2].argmax(dim=0).cpu().numpy() + 2
    obs = mask_val[:, 0, 0].cpu().numpy().astype(bool)
    gap = ~obs

    has_g = diff_net is not None
    n_cols = n_traj + 2 + (1 if has_g else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4.5))
    # learned drift vs truth
    ax = axes[0]
    ax.plot(z, f_true, "k-", lw=2, label="true $a(z-z^3)$")
    ax.plot(z, f_learned, "C2--", lw=2, label=r"learned $f_\theta$")
    ax.axvspan(-2, 2, color="C2", alpha=0.06, label="support")
    ax.set_xlabel("$z$"); ax.set_ylabel("$f(z)$"); ax.set_ylim(-4, 4)
    ax.set_title("learned drift"); ax.legend(fontsize=9)
    off = 1
    # learned diffusion g^2(z) vs the true constant (when a DiffusionNet is given)
    if has_g:
        ax = axes[1]
        g2 = diff_net._g2(z_grid.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        ax.plot(z, g2, "C0-", lw=2, label=r"learned $g^2(z)$")
        if sigma is not None:
            ax.axhline(sigma ** 2, ls="--", c="k", lw=2, label=fr"true $\sigma^2={sigma ** 2:.3f}$")
        ax.axvspan(-2, 2, color="C2", alpha=0.06)
        ax.set_xlabel("$z$"); ax.set_ylabel("$g^2(z)$")
        ax.set_ylim(0, max(0.6, float(g2.max()) * 1.2))
        ax.set_title("learned diffusion"); ax.legend(fontsize=9)
        off = 2
    # posterior snapshots in the gap
    for j in range(n_traj):
        tj = int(t_star[j]); ax = axes[off + j]
        ax.plot(z, filt[tj, j], "k-", lw=2, label="exact filter")
        ax.plot(z, pi[tj, j], "C3--", lw=2, label="operator")
        ax.set_xlabel("$z$"); ax.set_title(f"posterior traj {j}, $t={tj}$ (gap)")
        if j == 0:
            ax.legend(fontsize=9)
    # mean +/- 2 std over time, traj 0
    j = 0; ax = axes[-1]
    m_op = (log_pi[:, j].exp() * z_grid).sum(-1).cpu().numpy()
    s_op = ((log_pi[:, j].exp() * z_grid ** 2).sum(-1).cpu().numpy() - m_op ** 2).clip(0) ** 0.5
    if gap.any():
        ax.axvspan(ts_np[gap][0], ts_np[gap][-1], color="gray", alpha=0.15, label="no obs")
    ax.plot(ts_np, mean[:, j].cpu().numpy(), "k-", lw=2, label="exact mean")
    ax.plot(ts_np, m_op, "C3--", lw=2, label="operator mean")
    ax.fill_between(ts_np, m_op - 2 * s_op, m_op + 2 * s_op, color="C3", alpha=0.2)
    ax.set_xlabel("$t$"); ax.set_ylabel("$z$"); ax.set_title("estimate traj 0"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(img_path); plt.close()


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
