"""Minimal d=1 em_highd validation figure for the JAX backend (self-contained numpy/matplotlib, fed by the
JAX operator's arrays -- decoupled from the torch-bound opssm.analysis.viz). Per validation trajectory:
JAX gauge-aligned filtering posterior p(z|y_{0:t}) vs the EXACT filter (side-by-side heatmaps, true latent
overlaid), the aligned latent mean vs truth, and the learned drift f(z) vs the true double-well a(z-z^3)."""
import os

import numpy as np
import jax.numpy as jnp


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
