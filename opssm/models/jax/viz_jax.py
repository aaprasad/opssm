"""Validation figures for the JAX backend (self-contained numpy/matplotlib, fed by the JAX operator's arrays
-- decoupled from the torch-bound opssm.analysis.viz, but mirroring its layouts). d==1: gauge-aligned
filtering posterior p(z|y_{0:t}) vs the EXACT filter + latent + drift (like vis_highd). d==2: phase-plane
drift streamplots + latent recovery (like vis_latent2d). d==3: 3-D attractor + drift-direction quivers
(like vis_latent3d). `plot_validation` dispatches on latent_dim."""
import os

import numpy as np
import jax.numpy as jnp


def plot_validation(op, drift_net, g_cur, C_cur, d_cur, refs, hp, step, fig_dir, metrics, m_op):
    """Dispatch the validation figure on latent_dim. m_op (T,B,d) = the (MALA/grid) filter mean."""
    d = hp["latent_dim"]
    if refs.get("z_val_true") is None:                            # GT-free (Kato): recon + behavior manifold
        return plot_kato(m_op, refs, C_cur, d_cur, step, fig_dir, metrics)
    if d == 1:
        return plot_em_highd_d1(op, drift_net, g_cur, C_cur, d_cur, refs, step, fig_dir, metrics)
    zt = np.asarray(refs["z_val_true"]); mo = np.asarray(m_op); ts = np.asarray(refs["ts"])
    if d == 2:
        return plot_latent2d(drift_net, mo, zt, ts, refs["true_drift"], g_cur, step, fig_dir, metrics)
    if d == 3:
        return plot_latent3d(drift_net, mo, zt, ts, refs["true_drift"], g_cur, step, fig_dir, metrics)
    return None


def _affine_gauge_np(m_op, z_true):
    d = z_true.shape[-1]
    M = m_op.reshape(-1, d); Z = z_true.reshape(-1, d)
    sol, *_ = np.linalg.lstsq(np.concatenate([M, np.ones_like(M[:, :1])], -1), Z, rcond=None)
    A, b = sol[:d], sol[d]
    return A, b, np.linalg.pinv(A)


def _aligned_posterior(pi_op_traj, zg, s, b):
    """pi_op_traj (T,Nz) op-frame density on grid zg -> aligned true-frame density on zg (T,Nz)."""
    q = (zg - b) / s                                              # true z -> op-frame coordinate
    out = np.stack([np.interp(q, zg, pi_op_traj[t]) for t in range(pi_op_traj.shape[0])], 0)
    out = np.clip(out, 0, None) / abs(s)
    return out / np.clip(out.sum(-1, keepdims=True), 1e-12, None)


def plot_em_highd_d1(op, drift_net, g_cur, C_cur, d_cur, refs, step, fig_dir, metrics, n_traj=2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xv, mv, filt = refs["x_val"], refs["mask_val"], refs["filt_val"]
    zg, z_true = refs["z_grid"], refs["z_val_true"]
    a, sigma, ts = refs["a"], refs["sigma"], np.asarray(refs["ts"])
    log_pi = np.asarray(op.log_posterior(xv, mv, zg))            # (T,B,Nz)
    pi_op = np.exp(log_pi)
    m_op = (pi_op * np.asarray(zg)).sum(-1)                       # (T,B)
    zg_np = np.asarray(zg); filt_np = np.asarray(filt); zt_np = np.asarray(z_true)[..., 0]

    # Procrustes gauge z_true ~ s*m_op + b (global, all sites)
    Mm = m_op.reshape(-1, 1); Z = zt_np.reshape(-1, 1)
    sol, *_ = np.linalg.lstsq(np.concatenate([Mm, np.ones_like(Mm)], -1), Z, rcond=None)
    s, b = float(sol[0, 0]), float(sol[1, 0])

    T, B = m_op.shape
    ext = [ts[0], ts[-1], zg_np[0], zg_np[-1]]
    fig, axes = plt.subplots(n_traj, 3, figsize=(15, 3.2 * n_traj), squeeze=False)
    for r in range(n_traj):
        pa = _aligned_posterior(pi_op[:, r], zg_np, s, b)        # (T,Nz) aligned JAX posterior
        fa = filt_np[:, r]                                       # (T,Nz) exact filter
        for c, (dens, title) in enumerate([(pa, "JAX filter (aligned)"), (fa, "exact filter")]):
            ax = axes[r][c]
            ax.imshow(dens.T, origin="lower", aspect="auto", extent=ext, cmap="magma")
            ax.plot(ts, zt_np[:, r], color="cyan", lw=1.2, label="true z")
            ax.set_ylim(-2.2, 2.2); ax.set_title(f"traj {r}: {title}")
            if c == 0:
                ax.set_ylabel("z")
        ax = axes[r][2]                                          # aligned latent mean vs true
        ax.plot(ts, zt_np[:, r], color="k", lw=1.5, label="true")
        ax.plot(ts, s * m_op[:, r] + b, color="C1", lw=1.2, ls="--", label="JAX E[z] (aligned)")
        ax.set_ylim(-2.2, 2.2); ax.set_title(f"traj {r}: latent mean"); ax.legend(fontsize=8)
    m = metrics
    fig.suptitle(f"em_highd JAX  step {step}   kl_aln={m.get('kl_aln',float('nan')):.3f}  "
                 f"drift_rel={m.get('drift_rel',float('nan')):.3f}  g_rel={m.get('g_rel',float('nan')):.3f}  "
                 f"c_cos={m.get('c_cos',float('nan')):.3f}  g={float(g_cur):.3f} (σ={sigma})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(fig_dir, exist_ok=True)
    p1 = os.path.join(fig_dir, f"jax_step_{step:05d}.png")
    fig.savefig(p1, dpi=110); plt.close(fig)

    # drift panel (true frame): aligned f(z) vs a(z - z^3)
    zop = zg_np
    f_op = np.asarray(drift_net.net(jnp.asarray(zop)[:, None]))[:, 0]
    z_tf = s * zop + b                                           # op grid -> true frame
    f_al = s * f_op                                             # drift maps under the linear part
    order = np.argsort(z_tf); z_tf, f_al = z_tf[order], f_al[order]
    reg = np.abs(z_tf) <= 2.0
    fig2, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(z_tf[reg], f_al[reg], color="C0", lw=1.6, label="JAX drift (aligned)")
    ax.plot(z_tf[reg], a * (z_tf[reg] - z_tf[reg] ** 3), color="k", lw=1.4, ls="--", label="true a(z-z³)")
    ax.axhline(0, color="gray", lw=0.5); ax.set_xlabel("z"); ax.set_ylabel("f(z)")
    ax.set_title(f"drift  step {step}  drift_rel={m.get('drift_rel',float('nan')):.3f}"); ax.legend()
    fig2.tight_layout()
    p2 = os.path.join(fig_dir, f"jax_step_{step:05d}_drift.png")
    fig2.savefig(p2, dpi=110); plt.close(fig2)
    return p1, p2


def plot_latent2d(drift_net, m_op, z_true, ts, true_drift, g_cur, step, fig_dir, metrics, n_traj=4, ng=32):
    """d=2 phase-plane: true drift streamplot + true traj / learned drift (aligned) + inferred traj /
    latent recovery. Mirrors opssm.analysis.viz.vis_latent2d. m_op, z_true (T,B,2) numpy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A, b, A_inv = _affine_gauge_np(m_op, z_true)
    Z = z_true.reshape(-1, 2)
    z_al = m_op @ A + b
    lo = np.quantile(Z, 0.01, 0); hi = np.quantile(Z, 0.99, 0)
    pad = 0.1 * np.clip(hi - lo, 1e-3, None); lo, hi = lo - pad, hi + pad
    xs = np.linspace(lo[0], hi[0], ng); ys = np.linspace(lo[1], hi[1], ng)
    GX, GY = np.meshgrid(xs, ys)                                  # (ng,ng) indexing 'xy'
    P = np.stack([GX.reshape(-1), GY.reshape(-1)], -1)          # (ng^2,2) true-frame grid
    FT = np.asarray(true_drift(jnp.asarray(P)))
    FL = np.asarray(drift_net.net(jnp.asarray((P - b) @ A_inv))) @ A   # learned drift -> true frame

    def stream(ax, F, title):
        u = F[:, 0].reshape(ng, ng); v = F[:, 1].reshape(ng, ng)
        s = ax.streamplot(xs, ys, u, v, color=np.hypot(u, v), cmap="viridis",
                          density=1.2, linewidth=1.0, arrowsize=0.8)
        ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$"); ax.set_title(title)
        ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1]); return s

    nj = min(n_traj, z_true.shape[1])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    s0 = stream(axes[0], FT, "true drift f(z)")
    for j in range(nj):
        axes[0].plot(z_true[:, j, 0], z_true[:, j, 1], "k-", lw=0.8, alpha=0.5)
    fig.colorbar(s0.lines, ax=axes[0], fraction=0.046)
    s1 = stream(axes[1], FL, "learned drift (aligned)")
    for j in range(nj):
        axes[1].plot(z_al[:, j, 0], z_al[:, j, 1], "C3-", lw=0.8, alpha=0.6)
    fig.colorbar(s1.lines, ax=axes[1], fraction=0.046)
    ax = axes[2]
    for j in range(nj):
        ax.plot(z_al[:, j, 0], z_al[:, j, 1], "C3-", lw=1.0, alpha=0.8, label="inferred (aligned)" if j == 0 else None)
        ax.plot(z_true[:, j, 0], z_true[:, j, 1], "k--", lw=1.0, alpha=0.6, label="true" if j == 0 else None)
    ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$"); ax.set_title("latent recovery"); ax.legend(fontsize=9)
    m = metrics
    fig.suptitle(f"VdP 2-D  step {step}  lat_rel={m.get('lat_rel',float('nan')):.3f} "
                 f"drift_rel={m.get('drift_rel',float('nan')):.3f} g_rel={m.get('g_rel',float('nan')):.3f} "
                 f"c_cos={m.get('c_cos',float('nan')):.3f} recon_r2={m.get('recon_r2',float('nan')):.3f} "
                 f"g={float(g_cur):.3f}", fontsize=12, y=1.02)
    fig.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    p = os.path.join(fig_dir, f"jax_step_{step:05d}.png")
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    return (p,)


def plot_latent3d(drift_net, m_op, z_true, ts, true_drift, g_cur, step, fig_dir, metrics, n_traj=3, n_arrows=160):
    """d=3 phase-space: 3-D attractor (true vs aligned) + 3 coordinate-projection drift-direction quivers.
    Mirrors opssm.analysis.viz.vis_latent3d. m_op, z_true (T,B,3) numpy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
    A, b, A_inv = _affine_gauge_np(m_op, z_true)
    Z = z_true.reshape(-1, 3)
    z_al = m_op @ A + b
    lo = np.quantile(Z, 0.01, 0); hi = np.quantile(Z, 0.99, 0)
    pad = 0.1 * (hi - lo); lo, hi = lo - pad, hi + pad
    sub = max(1, Z.shape[0] // n_arrows); P = Z[::sub]
    FT = np.asarray(true_drift(jnp.asarray(P)))
    FL = np.asarray(drift_net.net(jnp.asarray((P - b) @ A_inv))) @ A
    ftn = FT / np.clip(np.linalg.norm(FT, axis=-1, keepdims=True), 1e-9, None)
    fln = FL / np.clip(np.linalg.norm(FL, axis=-1, keepdims=True), 1e-9, None)
    nj = min(n_traj, z_true.shape[1])
    fig = plt.figure(figsize=(21, 5.2))
    ax0 = fig.add_subplot(1, 4, 1, projection="3d")
    for j in range(nj):
        ax0.plot(z_al[:, j, 0], z_al[:, j, 1], z_al[:, j, 2], "C3-", lw=0.7, alpha=0.8, label="inferred (aligned)" if j == 0 else None)
        ax0.plot(z_true[:, j, 0], z_true[:, j, 1], z_true[:, j, 2], "k--", lw=0.6, alpha=0.6, label="true" if j == 0 else None)
    ax0.set_xlabel("$z_1$"); ax0.set_ylabel("$z_2$"); ax0.set_zlabel("$z_3$")
    ax0.set_title("latent recovery (3-D)"); ax0.legend(fontsize=8)
    ax0.set_xlim(lo[0], hi[0]); ax0.set_ylim(lo[1], hi[1]); ax0.set_zlim(lo[2], hi[2])
    for ax, (i, k), name in zip([fig.add_subplot(1, 4, c) for c in (2, 3, 4)],
                                [(0, 1), (0, 2), (1, 2)], ["z1-z2", "z1-z3", "z2-z3"]):
        ax.quiver(P[:, i], P[:, k], ftn[:, i], ftn[:, k], color="0.5", alpha=0.7, angles="xy", scale=28, width=0.004, label="true")
        ax.quiver(P[:, i], P[:, k], fln[:, i], fln[:, k], color="C3", alpha=0.6, angles="xy", scale=28, width=0.004, label="learned")
        ax.set_xlabel(f"$z_{{{i+1}}}$"); ax.set_ylabel(f"$z_{{{k+1}}}$"); ax.set_title(f"drift dirs ({name})"); ax.legend(fontsize=8)
        ax.set_xlim(lo[i], hi[i]); ax.set_ylim(lo[k], hi[k])
    m = metrics
    fig.suptitle(f"Lorenz 3-D  step {step}  lat_rel={m.get('lat_rel',float('nan')):.3f} "
                 f"drift_rel={m.get('drift_rel',float('nan')):.3f} g_rel={m.get('g_rel',float('nan')):.3f} "
                 f"c_cos={m.get('c_cos',float('nan')):.3f} recon_r2={m.get('recon_r2',float('nan')):.3f} "
                 f"g={float(g_cur):.3f}", fontsize=12, y=1.02)
    fig.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    p = os.path.join(fig_dir, f"jax_step_{step:05d}.png")
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    return (p,)


def plot_kato(m_op, refs, C_cur, d_cur, step, fig_dir, metrics, n_neurons=6):
    """GT-free Kato figure: (a) reconstruction (a few neurons: obs vs decode over one val window),
    (b) latent manifold = PCA-2D of the val latent means colored by behavior state, (c) leading latent
    dims over time. m_op (win,B,d) val filter means; obs from refs['x_val'] (win,B,N). No ground truth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xv = np.asarray(refs["x_val"])                               # (win,B,N) standardized obs
    C = np.asarray(C_cur); d_off = np.asarray(d_cur)
    z = np.asarray(m_op)                                         # (win,B,d)
    win, B, dlat = z.shape
    y_hat = z @ C.T + d_off                                      # (win,B,N) reconstruction
    states = refs.get("states_val")                             # (win,B) int or None
    names = refs.get("state_names")

    fig = plt.figure(figsize=(18, 5.0))
    # (a) reconstruction: a few neurons, window 0
    axr = fig.add_subplot(1, 3, 1)
    t = np.arange(win)
    sel = np.linspace(0, xv.shape[-1] - 1, n_neurons).astype(int)
    for i, n in enumerate(sel):
        axr.plot(t, xv[:, 0, n] + i * 3, color="k", lw=0.9, alpha=0.7)
        axr.plot(t, y_hat[:, 0, n] + i * 3, color="C3", lw=0.9, alpha=0.8)
    axr.set_title(f"reconstruction (window 0, {n_neurons} neurons)\nblack=obs  red=decode  "
                  f"recon_r2={metrics.get('recon_r2', float('nan')):.3f}")
    axr.set_xlabel("t (window)"); axr.set_yticks([])

    # (b) latent manifold: PCA-2D of all val latent means, colored by behavior
    Z = z.reshape(-1, dlat)
    Zc = Z - Z.mean(0)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    P = Zc @ Vt[:2].T                                           # (N,2) top-2 PCs
    axm = fig.add_subplot(1, 3, 2)
    if states is not None:
        sv = np.asarray(states).reshape(-1)
        uq = np.unique(sv)
        cmap = plt.get_cmap("tab10")
        for j, s in enumerate(uq):
            msk = sv == s
            lbl = names[int(s)] if (names is not None and 0 <= int(s) < len(names)) else f"state {int(s)}"
            axm.scatter(P[msk, 0], P[msk, 1], s=4, alpha=0.5, color=cmap(j % 10), label=lbl)
        axm.legend(fontsize=7, markerscale=2, ncol=2)
    else:
        axm.scatter(P[:, 0], P[:, 1], s=4, alpha=0.5, c=np.tile(np.arange(win), B), cmap="viridis")
    axm.set_xlabel("PC1"); axm.set_ylabel("PC2"); axm.set_title("latent manifold (PCA-2D of E[z|y])")

    # (c) leading latent dims over time (window 0)
    axl = fig.add_subplot(1, 3, 3)
    for k in range(min(4, dlat)):
        axl.plot(t, z[:, 0, k], lw=1.0, label=f"z{k+1}")
    axl.set_xlabel("t (window)"); axl.set_ylabel("E[z|y]"); axl.set_title("leading latent dims (window 0)")
    axl.legend(fontsize=8)

    fig.suptitle(f"Kato (GT-free)  step {step}  recon_r2={metrics.get('recon_r2', float('nan')):.3f}  "
                 f"d={dlat}  g={metrics.get('g', float('nan')):.3f}", fontsize=12, y=1.02)
    fig.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    p = os.path.join(fig_dir, f"jax_step_{step:05d}.png")
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    return (p,)
