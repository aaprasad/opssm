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
    f_learned = drift_net.drift(z_grid.unsqueeze(-1))[0].squeeze(-1).cpu().numpy()   # generalized drift is (...,d)
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
    ax.plot(z, f_learned, "C2-", lw=2, label=r"learned $f_\theta$")             # MODEL = solid
    ax.plot(z, f_true, "k--", lw=2, label="true $a(z-z^3)$")                     # GROUND TRUTH = dashed
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
    ax.plot(ts_np, m_op, "C3-", lw=2, label="filter mean")                          # MODEL = solid
    ax.fill_between(ts_np, m_op - 2 * s_op, m_op + 2 * s_op, color="C3", alpha=0.2)
    ax.plot(ts_np, mean[:, j].cpu().numpy(), "k--", lw=2, label="exact filter mean")  # exact-filter reference = dashed
    if show_sm:
        msm = m_sm[:, j].cpu().numpy(); ssm = s_sm[:, j].cpu().numpy()
        ax.plot(ts_np, msm, "C0-", lw=2, label="smoother mean")                      # MODEL = solid
        ax.plot(ts_np, m_sm_ex[:, j].cpu().numpy(), color="0.4", ls="--", lw=1.5, label="exact smoother mean")
        ax.fill_between(ts_np, msm - 2 * ssm, msm + 2 * ssm, color="C0", alpha=0.2)
    ax.set_xlabel("$t$"); ax.set_ylabel("$z$"); ax.set_title("estimate traj 0"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(img_path); plt.close()


@torch.no_grad()
def vis_highd(model, drift_net, diff_net, y_val, mask_val, z_val_true, filt_val, z_grid, ts,
              a, sigma, C_cur, d_cur, C_true, d_true, learn_obs, img_path, n_traj=2, s_scale=1.0,
              g_scalar=None, model_b=None, smoothed_val=None, s_fit=None, aligned=False,
              obs_mean=None, obs_scale=1.0):
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
    # RAW (s=1) vs PROCRUSTES-ALIGNED (s=s_fit) view: operator-z lives on the identifiable-up-to-scale gauge,
    # so scale every operator-derived quantity (latent, drift, diffusion, p(z,t)) to the true frame. Called
    # once aligned=False (raw) and once True (aligned) -> two side-by-side figures (report both, hide nothing).
    s = float(s_fit) if (aligned and s_fit is not None) else 1.0
    md = s * m_op                                                        # displayed latent mean (raw if s=1)

    def _push_density(p):                                                # p_op(m) -> p_z(z)=p_op(z/s)/s, traj 0
        p0 = p[:, 0].cpu().numpy()
        if s == 1.0:
            return p0
        pa = np.stack([np.interp(zg / s, zg, p0[t]) for t in range(p0.shape[0])]) / abs(s)
        return pa / pa.sum(-1, keepdims=True).clip(1e-12)

    # ---- empirical drift / diffusion with data-density (+/-2 SE) uncertainty (on the displayed scale) ----
    zc = md[:-1].reshape(-1)
    dz = ((md[1:] - md[:-1]) / dt).reshape(-1)
    ft = s * 0.5 * (drift_net.net(m_op[:-1].reshape(-1, 1)) + drift_net.net(m_op[1:].reshape(-1, 1))).squeeze(-1)
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
    # A: noisy obs (dots) + model RECON (dashed) vs the TRUE noiseless signal C_true z + d (solid), in
    # PHYSICAL units (un-standardized). Obs reconstruction is GAUGE-INVARIANT (the z->A z freedom cancels
    # in y=Cz+d), so recon and the true signal are directly comparable with NO alignment: solid == dashed
    # means accurate reconstruction. Judging recon against the noisy dots alone is not meaningful.
    om = obs_mean.cpu().numpy() if obs_mean is not None else None
    for j in range(n_traj):
        ax = axes[0, j]
        for dim in range(nd):
            off = float(om[dim]) if om is not None else 0.0
            rc = recon[:, j, dim].cpu().numpy() * obs_scale + off               # model recon (physical)
            rc_sd = (abs(float(C_cur[dim])) * s_op[:, j]).cpu().numpy() * obs_scale
            yobs = y_val[:, j, dim].cpu().numpy() * obs_scale + off             # noisy obs (physical)
            y_sig = (C_true[dim] * z_val_true[:, j] + d_true[dim]).cpu().numpy()  # TRUE noiseless obs
            ax.plot(ts_np, yobs, ".", ms=3, alpha=0.22, color=f"C{dim}")               # noisy obs
            ax.plot(ts_np, rc, "-", lw=2, color=f"C{dim}", label=f"recon $y_{dim}$" if j == 0 else None)
            ax.fill_between(ts_np, rc - 2 * rc_sd, rc + 2 * rc_sd, color=f"C{dim}", alpha=0.15)   # +/-2 std
            ax.plot(ts_np, y_sig, "--", lw=1.5, color=f"C{dim}", alpha=0.9,             # TRUE noiseless obs
                    label=f"true $y_{dim}$" if j == 0 else None)
        ax.set_title(f"obs recon vs true signal, traj {j}"); ax.set_xlabel("$t$")
        if j == 0:
            ax.legend(fontsize=8)
    for j in range(n_traj):                                              # B: latent + operator & exact bands
        ax = axes[1, j]
        ax.plot(ts_np, z_val_true[:, j].cpu().numpy(), "k--", lw=2, label="true $z$")   # GROUND TRUTH = dashed
        exm = ex_m[:, j].cpu().numpy(); exs = ex_s[:, j].cpu().numpy()
        ax.plot(ts_np, exm, "C7--", lw=1.2, label="exact mean")                         # exact-filter reference = dashed
        ax.fill_between(ts_np, exm - 2 * exs, exm + 2 * exs, color="C7", alpha=0.18)
        mo = md[:, j].cpu().numpy(); so = (abs(s) * s_op[:, j]).cpu().numpy()
        ax.plot(ts_np, mo, "r-", lw=2, label="filter mean" if show_sm else "operator mean")   # MODEL = solid
        ax.fill_between(ts_np, mo - 2 * so, mo + 2 * so, color="r", alpha=0.2)
        if show_sm:                                                       # smoother: tighter band, less lag
            msm = (s * m_sm[:, j]).cpu().numpy(); ssm = (abs(s) * s_sm[:, j]).cpu().numpy()
            ax.plot(ts_np, msm, "C0-", lw=2, label="smoother mean")                     # MODEL = solid
            ax.fill_between(ts_np, msm - 2 * ssm, msm + 2 * ssm, color="C0", alpha=0.2)
        ax.set_ylim(-1.7, 1.7); ax.set_title(f"latent $z$, traj {j}"); ax.set_xlabel("$t$")
        if j == 0:
            ax.legend(fontsize=8)
    ax = axes[0, 2]                                                      # C: drift over data regime + band
    ax.plot(zg[inr], (a * (z_grid - z_grid ** 3)).cpu().numpy()[inr], "k--", lw=2, label="true $a(z-z^3)$")  # GT dashed
    f_op = drift_net.drift((z_grid / s).unsqueeze(-1))[0].squeeze(-1)     # (Nz,): generalized drift is (...,d)
    ax.plot(zg[inr], (s * f_op).cpu().numpy()[inr], "C2-", lw=2,
            label=r"learned $f_\theta$")            # MODEL = solid; s*f_op(z/s) is true-scale drift when aligned
    ax.fill_between(ctr, f_emp - 2 * f_se, f_emp + 2 * f_se, color="C1", alpha=0.25, label=r"empirical $\pm2$SE")
    ax.plot(ctr, f_emp, "C1.", ms=4)
    ax.set_xlim(lo, hi); ax.set_ylim(-1.5, 1.5); ax.set_xlabel("$z$"); ax.set_ylabel("$f(z)$")
    ax.set_title("learned drift (data regime)"); ax.legend(fontsize=8)
    ax = axes[1, 2]                                                      # diffusion in g (not g^2): natural scale
    g_emp = np.sqrt(np.clip(g2_emp, 0.0, None))                         # g = sqrt(g^2)
    g_emp_se = g2_se / (2.0 * np.clip(g_emp, 1e-3, None))               # delta method: SE(g)=SE(g^2)/(2g)
    if diff_net is not None:                                             # g is a MAGNITUDE -> scale by |s|
        gc = abs(s) * np.sqrt(np.clip(diff_net._g2((z_grid / s).unsqueeze(-1)).squeeze(-1).cpu().numpy(), 0.0, None))
        ax.plot(zg[inr], gc[inr], "C0-", lw=2, label=r"learned $g(z)$")
    elif g_scalar is not None:
        ax.axhline(abs(s) * g_scalar, color="C0", lw=2, label=fr"learned $g={abs(s) * g_scalar:.3f}$")
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
    # E: 3D posterior evolution p(z,t) -- OPERATOR density pushed to the frame (raw or aligned), vs its oracle.
    _posterior3d(ax3d, zg, ts_np, filt_val[:, 0].cpu().numpy(), _push_density(pi),
                 lo, hi, ("filter $p(z,t)$ traj 0" if show_sm else "posterior $p(z,t)$ traj 0"))
    if show_sm:
        _posterior3d(ax3d_s, zg, ts_np, smoothed_val[:, 0].cpu().numpy(), _push_density(pi_sm),
                     lo, hi, "smoother $p(z,t)$ traj 0", color="C0", op_label="smoother")
    fig.suptitle("gauge-ALIGNED (operator -> true scale)" if s != 1.0 else "RAW (operator gauge)",
                 fontsize=13, y=1.0)
    plt.tight_layout(); plt.savefig(img_path); plt.close()


def _affine_gauge(m_op, z_true):
    """Affine Procrustes gauge z_true ~ A m_op + b. The offset b is essential for OFFSET latents (Lorenz z3
    mean ~25); a linear A alone cannot represent a mean shift. Returns A (d,d) linear part, b (d,) offset,
    A_inv (d,d). Map true->operator frame as (z - b) @ A_inv; the drift transforms under A only (an offset
    doesn't change velocities)."""
    d = z_true.shape[-1]
    M = m_op.reshape(-1, d); Z = z_true.reshape(-1, d)
    Mh = torch.cat([M, torch.ones_like(M[:, :1])], dim=-1)          # (N,d+1) design matrix [m, 1]
    sol = torch.linalg.lstsq(Mh, Z).solution                       # (d+1,d): Z ~ [M,1] @ sol
    return sol[:d], sol[d], torch.linalg.pinv(sol[:d])


@torch.no_grad()
def vis_latent2d(drift_net, m_op, z_true, ts, true_drift, g_scalar, img_path, n_traj=4, ng=32):
    """Phase-plane visualization for a 2-D latent (Van der Pol). The DRIFT field is shown as a STREAMPLOT
    over the (z1,z2) plane. Because the latent SDE is identifiable only up to a linear-map gauge
    (z_true ~ A m_op), the LEARNED drift is gauge-mapped into the TRUE frame -- f_true_frame(z) =
    f_op(z A^-1) A -- so its streamplot is directly comparable to the benchmark field f_op(z) A on the
    same axes. Three panels: (a) true drift + true trajectories, (b) learned drift (aligned) + inferred
    trajectories, (c) latent recovery (true vs aligned-inferred). m_op / z_true are (T,B,2)."""
    import numpy as np
    dev = m_op.device
    A, b, A_inv = _affine_gauge(m_op, z_true)                        # AFFINE Procrustes z_true ~ A m_op + b
    Z = z_true.reshape(-1, 2)                                        # true latent points (N,2)
    z_al = m_op @ A + b                                             # aligned inferred latent (T,B,2), true frame

    # DATA-REGIME extent (1-99 percentile of the true latent, robust to transients) + small margin -- restrict
    # to where the latent actually lives (cf. Duncker Fig 2, [-2,2]). Off-data corners are pure MLP
    # extrapolation (the true VdP field blows up cubically there) and would dominate the color scale.
    lo = Z.quantile(0.01, dim=0); hi = Z.quantile(0.99, dim=0)
    pad = 0.1 * (hi - lo).clamp_min(1e-3)
    lo = (lo - pad).cpu().numpy(); hi = (hi + pad).cpu().numpy()
    xs = torch.linspace(float(lo[0]), float(hi[0]), ng, device=dev)
    ys = torch.linspace(float(lo[1]), float(hi[1]), ng, device=dev)
    GX, GY = torch.meshgrid(xs, ys, indexing="xy")                  # (ng,ng); streamplot wants U,V as (Ny,Nx)
    P = torch.stack([GX.reshape(-1), GY.reshape(-1)], dim=-1)       # (ng*ng, 2) true-frame grid points
    FT = true_drift(P)                                             # true drift on the grid (ng*ng, 2)
    FL = drift_net.drift((P - b) @ A_inv)[0] @ A                    # learned drift -> true frame (offset-corrected)
    xs_np, ys_np = xs.cpu().numpy(), ys.cpu().numpy()

    def _stream(ax, F, title):
        u = F[:, 0].reshape(ng, ng).cpu().numpy(); v = F[:, 1].reshape(ng, ng).cpu().numpy()
        spd = np.hypot(u, v)
        strm = ax.streamplot(xs_np, ys_np, u, v, color=spd, cmap="viridis",
                             density=1.2, linewidth=1.0, arrowsize=0.8)
        ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$"); ax.set_title(title)
        ax.set_xlim(xs_np[0], xs_np[-1]); ax.set_ylim(ys_np[0], ys_np[-1])
        return strm

    zt = z_true.cpu().numpy(); za = z_al.cpu().numpy()
    nj = min(n_traj, zt.shape[1])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    s0 = _stream(axes[0], FT, "true drift $f(z)$")
    for j in range(nj):
        axes[0].plot(zt[:, j, 0], zt[:, j, 1], "k-", lw=0.8, alpha=0.5)
    fig.colorbar(s0.lines, ax=axes[0], label="$|f|$", fraction=0.046)
    s1 = _stream(axes[1], FL, "learned drift (aligned to true frame)")
    for j in range(nj):
        axes[1].plot(za[:, j, 0], za[:, j, 1], "C3-", lw=0.8, alpha=0.6)
    fig.colorbar(s1.lines, ax=axes[1], label="$|f|$", fraction=0.046)
    ax = axes[2]
    for j in range(nj):
        ax.plot(za[:, j, 0], za[:, j, 1], "C3-", lw=1.0, alpha=0.8,               # MODEL (inferred) = solid
                label="inferred (aligned)" if j == 0 else None)
        ax.plot(zt[:, j, 0], zt[:, j, 1], "k--", lw=1.0, alpha=0.6,               # GROUND TRUTH = dashed
                label="true" if j == 0 else None)
    ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$"); ax.set_title("latent recovery"); ax.legend(fontsize=9)
    fig.suptitle(fr"Van der Pol 2-D latent -- learned $g$={float(g_scalar):.3f}", fontsize=13, y=1.01)
    plt.tight_layout(); plt.savefig(img_path); plt.close()


@torch.no_grad()
def vis_latent3d(drift_net, m_op, z_true, ts, true_drift, g_scalar, img_path, n_traj=3, n_arrows=160):
    """Phase-space visualization for a 3-D latent (Lorenz). Panel 0: the 3-D attractor -- true (black)
    vs Procrustes-aligned inferred (red) -- the latent-recovery headline. Panels 1-3: the three 2-D
    coordinate projections showing DRIFT-field agreement as overlaid unit-direction quivers, true (gray)
    vs learned (red), at sampled attractor points. As in the 2-D viz the learned drift is gauge-mapped to
    the true frame (f(z) = f_op(z A^-1) A, A the gauge z_true ~ A m_op) so it is directly comparable; the
    arrows are normalized to DIRECTION (magnitude agreement is the drift_l2_aln metric). m_op/z_true (T,B,3)."""
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
    A, b, A_inv = _affine_gauge(m_op, z_true)                      # AFFINE gauge z_true ~ A m_op + b (offset b)
    Z = z_true.reshape(-1, 3)
    z_al = m_op @ A + b                                            # aligned inferred latent (T,B,3)
    lo = Z.quantile(0.01, dim=0).cpu().numpy(); hi = Z.quantile(0.99, dim=0).cpu().numpy()   # data-REGIME bounds
    pad = 0.1 * (hi - lo); lo, hi = lo - pad, hi + pad             # restrict plotting to where the latent lives
    sub = max(1, Z.shape[0] // n_arrows)
    P = Z[::sub]                                                   # sampled true-frame attractor points
    FT = true_drift(P)                                            # true drift there
    FL = drift_net.drift((P - b) @ A_inv)[0] @ A                   # learned drift -> true frame (offset-corrected)
    p = P.cpu().numpy()
    ftn = (FT / FT.norm(dim=-1, keepdim=True).clamp_min(1e-9)).cpu().numpy()   # unit directions
    fln = (FL / FL.norm(dim=-1, keepdim=True).clamp_min(1e-9)).cpu().numpy()
    zt = z_true.cpu().numpy(); za = z_al.cpu().numpy(); nj = min(n_traj, zt.shape[1])

    fig = plt.figure(figsize=(21, 5.2))
    ax0 = fig.add_subplot(1, 4, 1, projection="3d")               # 3-D attractor: latent recovery
    for j in range(nj):
        ax0.plot(za[:, j, 0], za[:, j, 1], za[:, j, 2], "C3-", lw=0.7, alpha=0.8,      # MODEL (inferred) = solid
                 label="inferred (aligned)" if j == 0 else None)
        ax0.plot(zt[:, j, 0], zt[:, j, 1], zt[:, j, 2], "k--", lw=0.6, alpha=0.6,      # GROUND TRUTH = dashed
                 label="true" if j == 0 else None)
    ax0.set_xlabel("$z_1$"); ax0.set_ylabel("$z_2$"); ax0.set_zlabel("$z_3$")
    ax0.set_title("latent recovery (3-D attractor)"); ax0.legend(fontsize=8)
    ax0.set_xlim(lo[0], hi[0]); ax0.set_ylim(lo[1], hi[1]); ax0.set_zlim(lo[2], hi[2])   # clip inferred spikes
    for ax, (i, k), name in zip([fig.add_subplot(1, 4, c) for c in (2, 3, 4)],
                                [(0, 1), (0, 2), (1, 2)], ["z1-z2", "z1-z3", "z2-z3"]):
        ax.quiver(p[:, i], p[:, k], ftn[:, i], ftn[:, k], color="0.5", alpha=0.7,
                  angles="xy", scale=28, width=0.004, label="true")
        ax.quiver(p[:, i], p[:, k], fln[:, i], fln[:, k], color="C3", alpha=0.6,
                  angles="xy", scale=28, width=0.004, label="learned")
        ax.set_xlabel(fr"$z_{{{i + 1}}}$"); ax.set_ylabel(fr"$z_{{{k + 1}}}$")
        ax.set_title(f"drift dirs ({name})"); ax.legend(fontsize=8)
        ax.set_xlim(lo[i], hi[i]); ax.set_ylim(lo[k], hi[k])       # data-regime extent (match panel 0)
    fig.suptitle(fr"Lorenz 3-D latent -- learned $g$={float(g_scalar):.3f}", fontsize=13, y=1.01)
    plt.tight_layout(); plt.savefig(img_path); plt.close()


def _posterior_grid(model, x, mask, m_op, z_true, traj, ng):
    """Shared setup for the posterior animations: gauge A (z_true ~ A m_op), a DATA-REGIME grid over the
    latent (true frame), and the normalized filtering posterior p(z|y_{0:t}) on it for one trajectory.
    Returns (grid_pts_true (ng^d, d), p (T, ng^d), lo, hi, true_path (T,d), d). Density is evaluated by
    mapping the true-frame grid to the operator frame via A^-1 and calling model.log_posterior."""
    dev = x.device
    d = z_true.shape[-1]
    A, b, A_inv = _affine_gauge(m_op, z_true)                       # AFFINE gauge z_true ~ A m_op + b
    Z = z_true.reshape(-1, d)
    lo = Z.quantile(0.01, 0); hi = Z.quantile(0.99, 0)             # data-regime bounds (where the latent lives)
    pad = 0.1 * (hi - lo).clamp_min(1e-3); lo, hi = lo - pad, hi + pad
    axes = [torch.linspace(float(lo[i]), float(hi[i]), ng, device=dev) for i in range(d)]
    G = torch.meshgrid(*axes, indexing="ij")
    grid = torch.stack([g.reshape(-1) for g in G], dim=-1)         # (ng^d, d) true-frame grid
    log_pi = model.log_posterior(x[:, traj:traj + 1], mask[:, traj:traj + 1], (grid - b) @ A_inv)[:, 0]  # (T, ng^d)
    return grid, log_pi.exp(), lo, hi, z_true[:, traj], d          # p normalized over the grid


@torch.no_grad()
def anim_posterior2d(model, x, mask, m_op, z_true, img_path, traj=0, ng=50, n_frames=60, fps=12):
    """Animate the VdP (d=2) filtering posterior p(z|y_{0:t}) as a 3-D SURFACE over the latent plane
    (x,y = z1,z2; height = p), evolving through time, with the true latent point overlaid."""
    import numpy as np
    from matplotlib import animation
    grid, p, lo, hi, zt, _ = _posterior_grid(model, x, mask, m_op, z_true, traj, ng)
    GX = grid[:, 0].reshape(ng, ng).cpu().numpy(); GY = grid[:, 1].reshape(ng, ng).cpu().numpy()
    p = p.reshape(-1, ng, ng).cpu().numpy(); zt = zt.cpu().numpy()
    T = p.shape[0]; frames = np.linspace(0, T - 1, min(n_frames, T)).astype(int)
    fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")

    def draw(fi):
        t = frames[fi]; ax.clear()
        s = p[t] / max(float(p[t].max()), 1e-12)                  # PER-FRAME normalized height (a peaked posterior
        ax.plot_surface(GX, GY, s, cmap="viridis", linewidth=0, antialiased=True, vmin=0, vmax=1)   # else flattens)
        ax.scatter([zt[t, 0]], [zt[t, 1]], [1.02], color="r", s=30, label="true $z$")
        ax.set_xlim(float(lo[0]), float(hi[0])); ax.set_ylim(float(lo[1]), float(hi[1])); ax.set_zlim(0, 1.05)
        ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$"); ax.set_zlabel(r"$p(z\,|\,y_{0:t})\,/\,\max$")
        ax.set_title(f"VdP posterior, frame {t}/{T}"); ax.legend(fontsize=8, loc="upper right")

    animation.FuncAnimation(fig, draw, frames=len(frames), blit=False).save(img_path, writer="ffmpeg", fps=fps, dpi=90)
    plt.close(fig)


@torch.no_grad()
def anim_posterior3d(model, x, mask, m_op, z_true, img_path, traj=0, ng=40, n_frames=50, fps=12):
    """Animate the Lorenz (d=3) posterior p(z|y_{0:t}) as a 3-D density cloud over the latent volume (x,y,z =
    z1,z2,z3), evolving through time. The filtering posterior is a PEAKED blob, so points are drawn with
    per-point ALPHA proportional to the PER-FRAME-normalized density (p/max at each t) -- the blob glows and
    the tails fade, robust whether the posterior is sharp or diffuse (a global fixed threshold hides a sharp
    peak). Fine grid to resolve the peak. True latent path-so-far + current point overlaid."""
    import numpy as np
    from matplotlib import animation
    grid, p, lo, hi, zt, _ = _posterior_grid(model, x, mask, m_op, z_true, traj, ng)
    pts = grid.cpu().numpy(); p = p.cpu().numpy(); zt = zt.cpu().numpy()
    lo, hi = lo.cpu().numpy(), hi.cpu().numpy()
    T = p.shape[0]; frames = np.linspace(0, T - 1, min(n_frames, T)).astype(int)
    fig = plt.figure(figsize=(7.5, 6)); ax = fig.add_subplot(111, projection="3d")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1)); sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.6, label=r"$p(z\,|\,y_{0:t})\,/\,\max$ (per frame)")

    def draw(fi):
        t = frames[fi]; ax.clear()
        a = p[t] / max(float(p[t].max()), 1e-12)                  # PER-FRAME normalized density in [0,1]
        sel = a > 0.03                                            # drop near-zero grid points (speed)
        rgba = plt.cm.viridis(a[sel]); rgba[:, 3] = a[sel] ** 0.6  # alpha ~ density -> glowing blob
        ax.scatter(pts[sel, 0], pts[sel, 1], pts[sel, 2], color=rgba, s=18, edgecolors="none")
        ax.plot(zt[:t + 1, 0], zt[:t + 1, 1], zt[:t + 1, 2], color="0.5", lw=0.4, alpha=0.3)   # true path so far
        ax.scatter([zt[t, 0]], [zt[t, 1]], [zt[t, 2]], color="r", s=30, label="true $z$")
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$"); ax.set_zlabel("$z_3$")
        ax.set_title(f"Lorenz posterior, frame {t}/{T}"); ax.legend(fontsize=8, loc="upper left")

    animation.FuncAnimation(fig, draw, frames=len(frames), blit=False).save(img_path, writer="ffmpeg", fps=fps, dpi=90)
    plt.close(fig)
