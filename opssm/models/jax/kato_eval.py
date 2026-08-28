"""Kato eval figures for the JAX backend -- eval_recon / eval_manifold / eval_dynamics, reproducing
opssm.eval.core.kato_report's figures. The figure functions below are copied VERBATIM from eval-core
(pure numpy/sklearn/matplotlib, model-neutral); kato_report_jax feeds them the JAX whole-trace filter
outputs directly (no Result/EvalContext / torch). Self-contained so it rebases cleanly onto baselines-
harness later, where the proper JaxOpssmAdapter -> kato_report seam replaces it.
"""
import numpy as np


def recon_metrics(y, y_hat):
    """Reconstruction R2 (global + per-neuron). y, y_hat: (T,N) in the SAME units (R2 is affine-invariant,
    so standardized and physical give identical R2)."""
    y = np.asarray(y); y_hat = np.asarray(y_hat)
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean(0, keepdims=True)) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot)
    per_neuron = 1 - ((y - y_hat) ** 2).sum(0) / (((y - y.mean(0)) ** 2).sum(0) + 1e-8)
    return r2, per_neuron


def decode(Z, y, tag, verbose=True):
    """BLOCKED (contiguous-time) 5-fold CV accuracy + confusion for SVM(rbf) and LDA. Blocked, not random:
    random folds leak through temporal autocorrelation (adjacent frames in train+test) and inflate accuracy
    ~10 pts on this data -- contiguous time blocks are the honest estimate for a time series."""
    from sklearn.svm import SVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.metrics import confusion_matrix, accuracy_score
    cv = KFold(5, shuffle=False)
    out = {}
    for name, clf in [("SVM", SVC(C=2.0)), ("LDA", LDA())]:
        pred = cross_val_predict(clf, Z, y, cv=cv)
        out[name] = (accuracy_score(y, pred), confusion_matrix(y, pred, labels=np.unique(y)), pred)
    base = np.bincount(y).max() / len(y)
    if verbose:
        print(f"  [{tag:16}] base={base:.3f} | " + " | ".join(f"{k}={v[0]:.3f}" for k, v in out.items()))
    return out, base


def data_flow(P2, nb=24, bw=0.10):
    """Empirical dP/dt (per-step velocity), Nadaraya-Watson kernel-smoothed onto a grid; sparse cells -> NaN.
    P2 (T,2) -> (GX, GY, FU, FV) for a streamplot of how the DATA actually flows. Model-agnostic."""
    P2 = np.asarray(P2)
    V = np.gradient(P2, axis=0)
    gx = np.linspace(P2[:, 0].min(), P2[:, 0].max(), nb)
    gy = np.linspace(P2[:, 1].min(), P2[:, 1].max(), nb)
    GX, GY = np.meshgrid(gx, gy)
    G = np.stack([GX.ravel(), GY.ravel()], 1)
    h = bw * np.hypot(np.ptp(gx), np.ptp(gy))
    d2 = ((G[:, None, :] - P2[None, :, :]) ** 2).sum(-1)
    W = np.exp(-d2 / (2 * h * h)); Wsum = W.sum(1)
    FU = (W @ V[:, 0]) / (Wsum + 1e-8); FV = (W @ V[:, 1]) / (Wsum + 1e-8)
    sparse = Wsum < 0.02 * Wsum.max()
    FU[sparse] = np.nan; FV[sparse] = np.nan
    return GX, GY, FU.reshape(GX.shape), FV.reshape(GX.shape)


# ------------------------------------------------------------------ figures (ported from scripts/eval_kato.py)
# All model-neutral: a Result's z_hat/y_hat/drift_fn + an EvalContext's states/neuron_ids drive them, so opssm
# and every baseline render identical figure types. Kept byte-faithful to the original eval_kato plots.

def _cmap():
    import matplotlib.pyplot as plt
    return plt.get_cmap("tab10")


def _quiv(ax, P2, C, L, cmap, ok=None, xc=None):
    """per-point arrows tangent to the local direction of travel, colored by state C; x = misclassified (by TRUE)."""
    d = np.gradient(P2, axis=0)
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9) * L
    q = ax.quiver(P2[:, 0], P2[:, 1], d[:, 0], d[:, 1], C, cmap=cmap, angles="xy", scale_units="xy",
                  scale=1, width=0.004, headwidth=4, headlength=5, alpha=0.8, zorder=2)
    q.set_clim(0, 9)
    if ok is not None:
        ax.scatter(P2[~ok, 0], P2[~ok, 1], c=xc[~ok], cmap=cmap, vmin=0, vmax=9, s=22, alpha=0.95,
                   marker="x", linewidths=1.1, zorder=3)


def _pcl(ev, i):
    return f"PC{i + 1} ({ev[i] * 100:.0f}%)"


def _finish(ax, P2, ttl, ev):
    ax.plot(P2[:, 0], P2[:, 1], color="0.5", lw=0.25, alpha=0.3, zorder=1)
    ax.scatter(*P2[0], c="lime", s=95, edgecolor="k", lw=1, marker="o", zorder=5)
    ax.scatter(*P2[-1], c="red", s=95, edgecolor="k", lw=1, marker="s", zorder=5)
    ax.annotate("start", P2[0], fontsize=8, weight="bold", zorder=6)
    ax.annotate("end", P2[-1], fontsize=8, weight="bold", zorder=6)
    ax.set_title(ttl, fontsize=8); ax.set_xlabel(_pcl(ev, 0), fontsize=7); ax.set_ylabel(_pcl(ev, 1), fontsize=7)
    ax.tick_params(labelsize=6)


def _arrow_len(P2):
    return 0.035 * float(np.mean(P2[:, :2].max(0) - P2[:, :2].min(0)))


def _beh2d(ax, P, c, ttl, ev, cmap):
    _quiv(ax, P[:, :2], c, _arrow_len(P), cmap); _finish(ax, P[:, :2], ttl, ev)


def _pred2d(ax, P, pred, true, ttl, ev, cmap):
    _quiv(ax, P[:, :2], pred, _arrow_len(P), cmap, ok=(pred == true), xc=true); _finish(ax, P[:, :2], ttl, ev)


def _traj3d(ax, P, c, ttl, ev, cmap):
    ax.plot(P[:, 0], P[:, 1], P[:, 2], color="k", lw=0.4, alpha=0.5)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=c, cmap=cmap, vmin=0, vmax=9, s=3, alpha=0.6)
    ax.scatter(*P[0], c="lime", s=70, edgecolor="k", marker="o")
    ax.scatter(*P[-1], c="red", s=70, edgecolor="k", marker="s")
    ax.set_title(ttl, fontsize=8); ax.tick_params(labelsize=5)
    ax.set_xlabel(_pcl(ev, 0), fontsize=6); ax.set_ylabel(_pcl(ev, 1), fontsize=6); ax.set_zlabel(_pcl(ev, 2), fontsize=6)


def fig_recon(out_path, name, y, y_hat, per_neuron_r2, neuron_ids, tvec, r2):
    """Measured vs reconstructed traces for the top-6 best-reconstructed labeled neurons (physical units)."""
    import matplotlib.pyplot as plt
    labeled = [(i, s) for i, s in enumerate(neuron_ids) if s]
    labeled.sort(key=lambda t: -per_neuron_r2[t[0]])
    show = labeled[:6] or [(i, f"n{i}") for i in np.argsort(-per_neuron_r2)[:6]]
    fig, ax = plt.subplots(len(show), 1, figsize=(12, 1.4 * len(show)), sharex=True)
    for k, (i, nm) in enumerate(show):
        ax[k].plot(tvec, y[:, i], lw=0.7, label="measured")
        ax[k].plot(tvec, y_hat[:, i], lw=0.9, label="reconstructed")
        ax[k].set_ylabel(f"{nm}\nR2={per_neuron_r2[i]:.2f}", fontsize=7, rotation=0, ha="right", va="center")
        ax[k].set_yticks([])
    ax[0].legend(fontsize=8, ncol=2, loc="upper right"); ax[-1].set_xlabel("time (s)")
    fig.suptitle(f"{name}  reconstruction  (overall R2={r2:.3f})")
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def fig_overview(out_path, name, tvec, z_hat, y_phys, states, state_names, pred_full, pred_pca_full,
                 dec_z, dec_pca, keep):
    """Behavior strip (model/true/PCA) + latents offset + model & raw-data PCA manifolds + dual confusion."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from sklearn.decomposition import PCA
    cmap = _cmap(); ns = len(state_names)
    zbar = z_hat.mean(0)
    zp3 = PCA(3).fit(z_hat - zbar); Zp = zp3.transform(z_hat - zbar)
    Yc = (y_phys - y_phys.mean(0)) / (y_phys.std(0) + 1e-6)
    yp3 = PCA(3).fit(Yc); Yp = yp3.transform(Yc)
    zev, yev = zp3.explained_variance_ratio_, yp3.explained_variance_ratio_

    fig = plt.figure(figsize=(15, 13)); gs = fig.add_gridspec(4, 3, height_ratios=[0.5, 1.0, 1.4, 1.4])
    axL = fig.add_subplot(gs[0, :])
    axL.imshow(np.stack([pred_full, states, pred_pca_full]), aspect="auto", cmap=cmap, vmin=0, vmax=9,
               extent=[tvec[0], tvec[-1], 0, 3], interpolation="nearest")
    axL.axhline(1.0, color="w", lw=1.5); axL.axhline(2.0, color="w", lw=1.5)
    axL.set_yticks([0.5, 1.5, 2.5]); axL.set_yticklabels(["PCA", "true", "model"], fontsize=8)
    axL.set_title(f"{name}  behavior (SVM blocked-CV): model (top) / true (mid) / PCA-of-data (bottom)   "
                  f"model acc={dec_z['SVM'][0]:.2f}  PCA acc={dec_pca['SVM'][0]:.2f}")
    axL.legend(handles=[Patch(color=cmap(i), label=nm) for i, nm in enumerate(state_names)],
               ncol=ns, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.5))
    axT = fig.add_subplot(gs[1, :])
    for j in range(z_hat.shape[1]):
        axT.plot(tvec, z_hat[:, j] + j * 3, lw=0.6)
    axT.set_yticks([]); axT.set_xlabel("time (s)"); axT.set_title("filtered latents z_hat (offset)")
    _pred2d(fig.add_subplot(gs[2, 0]), Zp, pred_full, states, "model z_hat PCA1-2 (PREDICTED; x=misclassified)", zev, cmap)
    _traj3d(fig.add_subplot(gs[2, 1], projection="3d"), Zp, pred_full, "model z_hat PCA1-2-3 (predicted)", zev, cmap)
    _beh2d(fig.add_subplot(gs[3, 0]), Yp, states, "RAW-data PCA1-2 (behavior) [Kato]", yev, cmap)
    _traj3d(fig.add_subplot(gs[3, 1], projection="3d"), Yp, states, "RAW-data PCA1-2-3 [Kato]", yev, cmap)
    lbl = [state_names[i] for i in np.unique(states[keep])]

    def _confus(ax, cm, ttl):
        ax.imshow(cm / cm.sum(1, keepdims=True).clip(1), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(lbl))); ax.set_xticklabels(lbl, fontsize=6, rotation=45)
        ax.set_yticks(range(len(lbl))); ax.set_yticklabels(lbl, fontsize=6)
        ax.set_box_aspect(1); ax.set_title(ttl, fontsize=8)

    _confus(fig.add_subplot(gs[2, 2]), dec_z["SVM"][1], f"MODEL SVM confusion (acc {dec_z['SVM'][0]:.2f})")
    _confus(fig.add_subplot(gs[3, 2]), dec_pca["SVM"][1], f"PCA-of-data SVM confusion (acc {dec_pca['SVM'][0]:.2f})")
    fig.suptitle(f"{name}: latents / manifold / decoding   arrows = direction of travel (start=o end=square); "
                 f"x = misclassified   var-expl z={zev.round(2)} raw={yev.round(2)}", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def fig_dynamics(out_path, name, z_hat, y_phys, states, drift_fn):
    """MODEL's learned drift field (latent-PCA plane) vs DATA's empirical flow (raw-PCA plane).
    drift_fn: z (numpy (G,d)) -> f (numpy (G,d)); if None only the data-flow panel is drawn."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    cmap = _cmap()
    zbar = z_hat.mean(0)
    Yc = (y_phys - y_phys.mean(0)) / (y_phys.std(0) + 1e-6)
    spca = PCA(2).fit(z_hat); ypca = PCA(2).fit(Yc)

    def _model_flow(basis, P2):
        gx = np.linspace(P2[:, 0].min(), P2[:, 0].max(), 24)
        gy = np.linspace(P2[:, 1].min(), P2[:, 1].max(), 24)
        GX, GY = np.meshgrid(gx, gy)
        grid_z = zbar + GX.ravel()[:, None] * basis[0] + GY.ravel()[:, None] * basis[1]
        fz = np.asarray(drift_fn(grid_z))
        return GX, GY, (fz @ basis[0]).reshape(GX.shape), (fz @ basis[1]).reshape(GX.shape)

    def _draw(ax, GX, GY, FU, FV, P2, ttl, prefix, ev):
        strm = ax.streamplot(GX, GY, FU, FV, color=np.hypot(FU, FV), cmap="viridis",
                             density=1.1, linewidth=0.6, arrowsize=0.9)
        ax.scatter(P2[:, 0], P2[:, 1], c=states, cmap=cmap, vmin=0, vmax=9, s=4, alpha=0.3, zorder=2)
        ax.plot(P2[:, 0], P2[:, 1], color="k", lw=0.4, alpha=0.35, zorder=3)
        ax.scatter(*P2[0], c="lime", s=70, edgecolor="k", marker="o", zorder=5)
        ax.scatter(*P2[-1], c="red", s=70, edgecolor="k", marker="s", zorder=5)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel(f"{prefix} {_pcl(ev, 0)}", fontsize=8); ax.set_ylabel(f"{prefix} {_pcl(ev, 1)}", fontsize=8)
        fig.colorbar(strm.lines, ax=ax, fraction=0.045, pad=0.02, label="speed")

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    Zs, Yp2 = spca.transform(z_hat), ypca.transform(Yc)
    if drift_fn is not None:
        _draw(axs[0], *_model_flow(spca.components_, Zs), Zs,
              "MODEL learned drift  (latent-PCA plane)", "latent", spca.explained_variance_ratio_)
    else:
        axs[0].set_title("MODEL drift  (n/a for this model)", fontsize=9); axs[0].axis("off")
    _draw(axs[1], *data_flow(Yp2), Yp2,
          "DATA empirical flow  (raw-PCA plane)", "raw", ypca.explained_variance_ratio_)
    fig.suptitle(f"{name}  dynamics: model's learned drift vs data's empirical flow  "
                 f"(streamlines colored by speed; black = trajectory, start=o end=square)")
    fig.savefig(out_path, dpi=115, bbox_inches="tight"); plt.close(fig)



def kato_report_jax(out_dir, name, z_hat, y_hat_std, y_phys, obs_std_full, obs_mean, obs_scale,
                    states, state_names, neuron_ids, dt, drift_fn):
    """JAX entry: reproduce eval_recon/manifold/dynamics from whole-trace filter outputs. z_hat (T,d),
    y_hat_std (T,N) standardized recon, y_phys (T,N) physical, obs_std_full (T,N) standardized full trace,
    states (T,), drift_fn: numpy z (G,d)->f (G,d) (wraps the JAX drift). Mirrors eval-core kato_report."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    z_hat = np.asarray(z_hat); y_hat_std = np.asarray(y_hat_std); y_phys = np.asarray(y_phys)
    y_hat_phys = y_hat_std * obs_scale + obs_mean
    r2, per_neuron = recon_metrics(y_phys, y_hat_phys)
    states = np.asarray(states).astype(int); names = list(state_names)
    keep = states < len(names)
    if "NOSTATE" in names:
        keep = keep & (states != names.index("NOSTATE"))
    dz = np.gradient(z_hat, axis=0)
    print(f"=== {name} | T={z_hat.shape[0]} N={y_phys.shape[1]} d={z_hat.shape[1]} ===")
    print(f"(A) reconstruction R2 = {r2:.3f}  (per-neuron median {np.median(per_neuron):.3f})")
    print("(B) behavior decoding (BLOCKED 5-fold CV):")
    dec_z, base = decode(z_hat[keep], states[keep], "model z_hat")
    dec_dz, _ = decode(dz[keep], states[keep], "model dz/dt")
    from sklearn.decomposition import PCA
    pca_base = PCA(z_hat.shape[1]).fit_transform(np.asarray(obs_std_full))
    dec_pca, _ = decode(pca_base[keep], states[keep], "PCA-of-data")
    pred_full = states.copy(); pred_full[keep] = dec_z["SVM"][2]
    pred_pca_full = states.copy(); pred_pca_full[keep] = dec_pca["SVM"][2]
    tvec = np.arange(z_hat.shape[0]) * dt
    fig_recon(f"{out_dir}/eval_recon.png", name, y_phys, y_hat_phys, per_neuron, neuron_ids, tvec, r2)
    fig_overview(f"{out_dir}/eval_manifold.png", name, tvec, z_hat, y_phys, states, names,
                 pred_full, pred_pca_full, dec_z, dec_pca, keep)
    fig_dynamics(f"{out_dir}/eval_dynamics.png", name, z_hat, y_phys, states, drift_fn)
    print(f"saved eval_recon/manifold/dynamics.png -> {out_dir}  recon_r2={r2:.3f} "
          f"decode_svm={dec_z['SVM'][0]:.3f} pca_svm={dec_pca['SVM'][0]:.3f} base={base:.3f}")
    return {"recon_r2": r2, "decode_svm": dec_z["SVM"][0], "decode_pca_svm": dec_pca["SVM"][0], "base": base}
