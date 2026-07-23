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
def _posterior3d(ax, z, ts, filt_j, pi_j, lo, hi, title, color="C3", op_label="operator"):
    """3D posterior evolution p(z, t) as a WATERFALL of per-time density profiles: the EXACT reference
    as light dashed curves, the OPERATOR as solid curves, over the data-regime z. Over-dispersion
    reads as the solid operator curve sitting lower/broader than the dashed exact one. `ax` is a
    pre-made 3D axis; filt_j / pi_j are (T, Nz) numpy densities for one traj. `color`/`op_label` let
    the caller distinguish the filter panel (red) from the smoother panel (blue)."""
    import numpy as np
    T = len(ts)
    ti = np.arange(0, T, max(1, T // 28))                   # subsample time for legibility
    inr = (z >= lo - 0.3) & (z <= hi + 0.3)                 # focus z on where there is mass
    zc = z[inr]
    for t in ti:
        tt = np.full_like(zc, ts[t])
        ax.plot(zc, tt, filt_j[t, inr], color="0.62", ls="--", lw=0.8)   # exact: light dashed
        ax.plot(zc, tt, pi_j[t, inr], color=color, ls="-", lw=1.1)       # operator: solid
    ax.plot([], [], [], color=color, ls="-", label=op_label)
    ax.plot([], [], [], color="0.62", ls="--", label="exact")
    ax.set_xlabel("$z$"); ax.set_ylabel("$t$"); ax.set_zlabel("$p$")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8, loc="upper left")
    ax.view_init(elev=32, azim=-58)
    return ax


def vis_learn(model, drift_net, xs_val, mask_val, filt_val, z_grid, ts, a, img_path, n_traj=2,
        diff_net=None, sigma=None, model_b=None, smoothed_val=None):
    from opssm.models.mstep import log_smoothed
    z = z_grid.cpu().numpy()
    ts_np = ts.cpu().numpy()
    f_learned = drift_net.drift(z_grid)[0].cpu().numpy()
    f_true = (a * (z_grid - z_grid ** 3)).cpu().numpy()
    log_pi = model.log_posterior(xs_val, mask_val, z_grid)
    pi = log_pi.exp().cpu().numpy()
    # SMOOTHER marginal (when the backward operator is present): p(z_t|y_{0:T}) vs the oracle smoothed.
    show_sm = model_b is not None and smoothed_val is not None
    if show_sm:
        log_sm = log_smoothed(model, model_b, xs_val, mask_val, z_grid)
        pi_sm = log_sm.exp()
        m_sm = (pi_sm * z_grid).sum(-1)
        s_sm = ((pi_sm * z_grid ** 2).sum(-1) - m_sm ** 2).clamp_min(0).sqrt()
        sm_ex = smoothed_val.cpu().numpy(); pi_sm = pi_sm.cpu().numpy()
        m_sm_ex = (smoothed_val * z_grid).sum(-1)
    # DATA REGIME: the range the inferred latent actually visits (operator posterior mean's 1-99
    # percentile) -- the SAME support the drift_l2 metric scores on. Outside it there is no data,
    # so the drift is unconstrained and naturally diverges; shading/limiting to it keeps the figure
    # honest about where the fit is being judged.
    m_op_all = (log_pi.exp() * z_grid).sum(-1)
    lo, hi = float(m_op_all.quantile(0.01)), float(m_op_all.quantile(0.99))
    filt = filt_val.cpu().numpy()
    mean = (filt_val * z_grid).sum(-1)
    var = (filt_val * z_grid ** 2).sum(-1) - mean ** 2
    t_star = var[2:-2].argmax(dim=0).cpu().numpy() + 2
    obs = mask_val[:, 0, 0].cpu().numpy().astype(bool)
    gap = ~obs

    has_g = diff_net is not None
    off = 2 if has_g else 1
    n_cols = n_traj + 2 + (1 if has_g else 0)
    post_idx = set(range(off, off + n_traj))                # posterior panels rendered in 3D
    fig = plt.figure(figsize=(6 * n_cols, 4.8))
    gs = fig.add_gridspec(1, n_cols)
    axes = [fig.add_subplot(gs[0, k], projection="3d") if k in post_idx else fig.add_subplot(gs[0, k])
            for k in range(n_cols)]
    # learned drift vs truth
    ax = axes[0]
    ax.plot(z, f_true, "k-", lw=2, label="true $a(z-z^3)$")
    ax.plot(z, f_learned, "C2--", lw=2, label=r"learned $f_\theta$")
    ax.axvspan(lo, hi, color="C2", alpha=0.12, label="data regime")
    mrg = 0.3
    ax.set_xlim(lo - mrg, hi + mrg)                          # focus on where there is data
    ax.set_xlabel("$z$"); ax.set_ylabel("$f(z)$"); ax.set_ylim(-4, 4)
    ax.set_title("learned drift"); ax.legend(fontsize=9)
    # learned diffusion g^2(z) vs the true constant (when a DiffusionNet is given)
    if has_g:
        ax = axes[1]
        g2 = diff_net._g2(z_grid.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        ax.plot(z, g2, "C0-", lw=2, label=r"learned $g^2(z)$")
        if sigma is not None:
            ax.axhline(sigma ** 2, ls="--", c="k", lw=2, label=fr"true $\sigma^2={sigma ** 2:.3f}$")
        ax.axvspan(lo, hi, color="C2", alpha=0.12)
        ax.set_xlabel("$z$"); ax.set_ylabel("$g^2(z)$")
        ax.set_ylim(0, max(0.6, float(g2.max()) * 1.2)); ax.set_xlim(lo - 0.3, hi + 0.3)
        ax.set_title("learned diffusion"); ax.legend(fontsize=9)
    # posterior EVOLUTION p(z,t) (3D). With a smoother: FILTER traj 0 + SMOOTHER traj 0 side by side
    # (each vs its own oracle) so the smoother's narrower gap density is visible. Else filter, per traj.
    if show_sm:
        _posterior3d(axes[off], z, ts_np, filt[:, 0], pi[:, 0], lo, hi, "filter $p(z,t)$ traj 0")
        _posterior3d(axes[off + 1], z, ts_np, sm_ex[:, 0], pi_sm[:, 0], lo, hi,
                     "smoother $p(z,t)$ traj 0", color="C0", op_label="smoother")
    else:
        for j in range(n_traj):
            _posterior3d(axes[off + j], z, ts_np, filt[:, j], pi[:, j], lo, hi,
                         f"posterior $p(z,t)$ traj {j}")
    # mean +/- 2 std over time, traj 0: filter (red) and, when present, smoother (blue, tighter in the gap)
    j = 0; ax = axes[-1]
    m_op = (log_pi[:, j].exp() * z_grid).sum(-1).cpu().numpy()
    s_op = ((log_pi[:, j].exp() * z_grid ** 2).sum(-1).cpu().numpy() - m_op ** 2).clip(0) ** 0.5
    if gap.any():
        ax.axvspan(ts_np[gap][0], ts_np[gap][-1], color="gray", alpha=0.15, label="no obs")
    ax.plot(ts_np, mean[:, j].cpu().numpy(), "k-", lw=2, label="exact filter mean")
    ax.plot(ts_np, m_op, "C3--", lw=2, label="filter mean")
    ax.fill_between(ts_np, m_op - 2 * s_op, m_op + 2 * s_op, color="C3", alpha=0.2)
    if show_sm:
        msm = m_sm[:, j].cpu().numpy(); ssm = s_sm[:, j].cpu().numpy()
        ax.plot(ts_np, m_sm_ex[:, j].cpu().numpy(), color="0.4", ls="-", lw=1.5, label="exact smoother mean")
        ax.plot(ts_np, msm, "C0--", lw=2, label="smoother mean")
        ax.fill_between(ts_np, msm - 2 * ssm, msm + 2 * ssm, color="C0", alpha=0.2)
    ax.set_xlabel("$t$"); ax.set_ylabel("$z$"); ax.set_title("estimate traj 0"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(img_path); plt.close()


@torch.no_grad()
def vis_highd(model, drift_net, diff_net, y_val, mask_val, z_val_true, filt_val, z_grid, ts,
              a, sigma, C_cur, d_cur, C_true, d_true, learn_obs, img_path, n_traj=2, s_scale=1.0,
              g_scalar=None, model_b=None, smoothed_val=None):
    """High-D Duncker panels with UNCERTAINTY BANDS everywhere and the drift/diffusion shown over the
    DATA REGIME only (the range the inferred latent actually visits). The drift/diffusion bands are the
    empirical +/-2 SE per z-bin -- wide where the latent rarely goes (cf. Duncker's GP uncertainty);
    the latent/recon panels carry the posterior +/-2 std; a data-density panel shows where the estimates
    are actually constrained."""
    import numpy as np
    zg = z_grid.cpu().numpy(); ts_np = ts.cpu().numpy(); dt = float(ts[1] - ts[0])
    log_pi = model.log_posterior(y_val, mask_val, z_grid)
    pi = log_pi.exp()
    m_op = (pi * z_grid).sum(-1)                                          # (T,B)
    s_op = (pi * z_grid ** 2).sum(-1).sub(m_op ** 2).clamp_min(0).sqrt()
    ex_m = (filt_val * z_grid).sum(-1)
    ex_s = (filt_val * z_grid ** 2).sum(-1).sub(ex_m ** 2).clamp_min(0).sqrt()
    recon = s_scale * C_cur * m_op[..., None] + d_cur                     # (T,B,D)
    nd = min(2, y_val.shape[-1])
    # SMOOTHER marginal (when the backward operator is present): p(z_t|y_{0:T}) mean/std for the latent panel.
    show_sm = model_b is not None and smoothed_val is not None
    if show_sm:
        from opssm.models.mstep import log_smoothed
        pi_sm = log_smoothed(model, model_b, y_val, mask_val, z_grid).exp()
        m_sm = (pi_sm * z_grid).sum(-1)                                   # (T,B)
        s_sm = (pi_sm * z_grid ** 2).sum(-1).sub(m_sm ** 2).clamp_min(0).sqrt()

    # ---- empirical drift / diffusion with data-density (+/-2 SE) uncertainty ----
    zc = m_op[:-1].reshape(-1)
    dz = ((m_op[1:] - m_op[:-1]) / dt).reshape(-1)
    ft = 0.5 * (drift_net.net(m_op[:-1].reshape(-1, 1)) + drift_net.net(m_op[1:].reshape(-1, 1))).squeeze(-1)
    r2dt = (dz - ft) ** 2 * dt                                            # per-sample g^2 target
    zc = zc.cpu().numpy(); dz = dz.cpu().numpy(); r2dt = r2dt.cpu().numpy()
    lo, hi = -1.5, 1.5                                                    # fixed data-regime window (Duncker axes)
    nb = 22; bins = np.linspace(lo, hi, nb + 1); ctr = 0.5 * (bins[:-1] + bins[1:])
    idx = np.clip(np.digitize(zc, bins) - 1, 0, nb - 1)
    f_emp = np.full(nb, np.nan); f_se = np.full(nb, np.nan)
    g2_emp = np.full(nb, np.nan); g2_se = np.full(nb, np.nan); dens = np.zeros(nb)
    for b in range(nb):
        msk = idx == b; n = int(msk.sum()); dens[b] = n
        if n >= 5:
            f_emp[b] = dz[msk].mean(); f_se[b] = dz[msk].std() / np.sqrt(n)
            g2_emp[b] = r2dt[msk].mean(); g2_se[b] = g2_emp[b] * np.sqrt(2.0 / n)
    inr = (zg >= lo) & (zg <= hi)

    fig = plt.figure(figsize=(25, 8))
    gsp = fig.add_gridspec(2, 5)
    axes = np.empty((2, 4), dtype=object)
    for r in range(2):
        for c in range(4):
            axes[r, c] = fig.add_subplot(gsp[r, c])
    if show_sm:                                                          # stack FILTER + SMOOTHER waterfalls
        ax3d = fig.add_subplot(gsp[0, 4], projection="3d")
        ax3d_s = fig.add_subplot(gsp[1, 4], projection="3d")
    else:
        ax3d = fig.add_subplot(gsp[:, 4], projection="3d")              # tall filter-only p(z,t) panel
    for j in range(n_traj):                                              # A: obs + recon + predictive band
        ax = axes[0, j]
        for dim in range(nd):
            rc = recon[:, j, dim].cpu().numpy()
            rc_sd = (s_scale * abs(float(C_cur[dim])) * s_op[:, j]).cpu().numpy()   # latent unc. -> obs space
            ax.plot(ts_np, y_val[:, j, dim].cpu().numpy(), ".", ms=3, alpha=0.3, color=f"C{dim}")
            ax.plot(ts_np, rc, "-", lw=2, color=f"C{dim}", label=f"$y_{dim}$" if j == 0 else None)
            ax.fill_between(ts_np, rc - 2 * rc_sd, rc + 2 * rc_sd, color=f"C{dim}", alpha=0.15)
        ax.set_title(f"obs + recon, traj {j}"); ax.set_xlabel("$t$")
        if j == 0:
            ax.legend(fontsize=8)
    for j in range(n_traj):                                              # B: latent + operator & exact bands
        ax = axes[1, j]
        ax.plot(ts_np, z_val_true[:, j].cpu().numpy(), "k-", lw=2, label="true $z$")
        exm = ex_m[:, j].cpu().numpy(); exs = ex_s[:, j].cpu().numpy()
        ax.plot(ts_np, exm, "C7-", lw=1.2, label="exact mean")
        ax.fill_between(ts_np, exm - 2 * exs, exm + 2 * exs, color="C7", alpha=0.18)
        mo = m_op[:, j].cpu().numpy(); so = s_op[:, j].cpu().numpy()
        ax.plot(ts_np, mo, "r--", lw=2, label="filter mean" if show_sm else "operator mean")
        ax.fill_between(ts_np, mo - 2 * so, mo + 2 * so, color="r", alpha=0.2)
        if show_sm:                                                       # smoother: tighter band, less lag
            msm = m_sm[:, j].cpu().numpy(); ssm = s_sm[:, j].cpu().numpy()
            ax.plot(ts_np, msm, "C0--", lw=2, label="smoother mean")
            ax.fill_between(ts_np, msm - 2 * ssm, msm + 2 * ssm, color="C0", alpha=0.2)
        ax.set_ylim(-1.7, 1.7); ax.set_title(f"latent $z$, traj {j}"); ax.set_xlabel("$t$")
        if j == 0:
            ax.legend(fontsize=8)
    ax = axes[0, 2]                                                      # C: drift over data regime + band
    ax.plot(zg[inr], (a * (z_grid - z_grid ** 3)).cpu().numpy()[inr], "k-", lw=2, label="true $a(z-z^3)$")
    ax.plot(zg[inr], drift_net.drift(z_grid)[0].cpu().numpy()[inr], "C2--", lw=2, label=r"learned $f_\theta$")
    ax.fill_between(ctr, f_emp - 2 * f_se, f_emp + 2 * f_se, color="C1", alpha=0.25, label=r"empirical $\pm2$SE")
    ax.plot(ctr, f_emp, "C1.", ms=4)
    ax.set_xlim(lo, hi); ax.set_ylim(-1.5, 1.5); ax.set_xlabel("$z$"); ax.set_ylabel("$f(z)$")
    ax.set_title("learned drift (data regime)"); ax.legend(fontsize=8)
    ax = axes[1, 2]                                                      # diffusion in g (not g^2): natural scale
    g_emp = np.sqrt(np.clip(g2_emp, 0.0, None))                         # g = sqrt(g^2)
    g_emp_se = g2_se / (2.0 * np.clip(g_emp, 1e-3, None))               # delta method: SE(g)=SE(g^2)/(2g)
    if diff_net is not None:
        gc = np.sqrt(np.clip(diff_net._g2(z_grid.unsqueeze(-1)).squeeze(-1).cpu().numpy(), 0.0, None))
        ax.plot(zg[inr], gc[inr], "C0-", lw=2, label=r"learned $g(z)$")
    elif g_scalar is not None:
        ax.axhline(g_scalar, color="C0", lw=2, label=fr"learned $g={g_scalar:.3f}$")
    ax.fill_between(ctr, g_emp - 2 * g_emp_se, g_emp + 2 * g_emp_se, color="C1", alpha=0.25, label=r"empirical $\pm2$SE")
    ax.plot(ctr, g_emp, "C1.", ms=4)
    ax.axhline(sigma, ls="--", c="k", lw=2, label=fr"true $\sigma={sigma:.3f}$")
    ax.set_xlim(lo, hi); ax.set_ylim(0, 1.0); ax.set_xlabel("$z$"); ax.set_ylabel("$g(z)$")
    ax.set_title("learned diffusion (data regime)"); ax.legend(fontsize=8)
    ax = axes[0, 3]                                                      # D: obs map (unit C direction)
    if learn_obs:
        C_true_u = C_true / C_true.norm()
        cos = float(C_cur @ C_true_u)
        flip = 1.0 if cos >= 0 else -1.0                                 # align the sign gauge
        ax.plot(C_true_u.cpu().numpy(), (flip * C_cur).cpu().numpy(), "C0o", label="$C$ dir")
        lim = float(C_true_u.abs().max()) * 1.2
        ax.plot([-lim, lim], [-lim, lim], "k:", lw=1)
        ax.set_xlabel("true unit $C$"); ax.set_ylabel("learned (sign-aligned)")
        ax.set_title(fr"obs map $C$:  $|\cos|$={abs(cos):.3f}"); ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "C, d fixed (known)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("observation map $C, d$")
    ax = axes[1, 3]                                                      # data density -> where estimates are constrained
    ax.bar(ctr, dens, width=(hi - lo) / nb, color="gray", alpha=0.5)
    ax.set_xlim(lo, hi); ax.set_xlabel("$z$"); ax.set_ylabel("count")
    ax.set_title("latent occupancy (data density)")
    # E: 3D posterior evolution p(z,t). Filter (red); with a smoother, a stacked SMOOTHER panel (blue,
    # narrower through the dynamics), each vs its own oracle.
    _posterior3d(ax3d, zg, ts_np, filt_val[:, 0].cpu().numpy(), pi[:, 0].cpu().numpy(),
                 lo, hi, ("filter $p(z,t)$ traj 0" if show_sm else "posterior $p(z,t)$ traj 0"))
    if show_sm:
        _posterior3d(ax3d_s, zg, ts_np, smoothed_val[:, 0].cpu().numpy(), pi_sm[:, 0].cpu().numpy(),
                     lo, hi, "smoother $p(z,t)$ traj 0", color="C0", op_label="smoother")
    plt.tight_layout(); plt.savefig(img_path); plt.close()
