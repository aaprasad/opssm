"""Post-hoc eval of a trained opssm model on Kato 2015 C. elegans data.

Loads <train_dir>/model.pt, filters the FULL worm trace (windowed + stitched) to latents z_hat, then:
  (A) reconstruction  -- R2 (physical dF/F) + measured-vs-reconstructed individual neuron traces
  (B) behavior decode -- SVM + LDA on z_hat (and dz/dt) -> Kato behavior states: CV accuracy + confusion
  (C) latents/dynamics -- z_hat over time, PCA/t-SNE colored by behavior, learned drift field in the PC plane
Usage: python scripts/eval_kato.py --model dump/kato_stim0/model.pt
"""
import argparse
import inspect
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection="3d")
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

from opssm.models.filter_module import ZakaiFilterModule
from opssm.models.mstep import filter_mean
from opssm.models.obs import zhat_from_obs
from opssm.data.kato.load import load_worm

dev = "cuda" if torch.cuda.is_available() else "cpu"


def load_kato(model_pt):
    d = torch.load(model_pt, map_location="cpu", weights_only=False)
    hp = d["hparams"]
    args = set(inspect.signature(ZakaiFilterModule.__init__).parameters)
    model = ZakaiFilterModule(**{k: v for k, v in hp.items() if k in args})
    model.load_state_dict(d["state_dict"], strict=False)
    model.C_cur, model.d_cur, model.g_cur = d["C_cur"].to(dev), d["d_cur"].to(dev), d["g_cur"]
    return model.to(dev).eval(), d, hp


@torch.no_grad()
def filter_full_trace(model, y_std, window, stride):
    """y_std (T,N) standardized -> z_hat (T,d) stitched from windowed filter means (overlaps averaged)."""
    T, N = y_std.shape
    d = model.model.latent_dim
    starts = list(range(0, T - window + 1, stride)) or [0]
    x = torch.stack([y_std[s:s + window] for s in starts], dim=1).to(dev)   # (window,B,N)
    mask = torch.ones(x.shape[0], x.shape[1], 1, device=dev)
    center = zhat_from_obs(x, model.C_cur, model.d_cur)
    h = model.hparams
    zc, _ = filter_mean(model.model, x, mask, center, method="mala", n_mean=h.n_mean,
                        near_std=h.near_std, broad_std=h.broad_std, mala=model._mala_cfg())   # (window,B,d)
    acc = torch.zeros(T, d, device=dev); cnt = torch.zeros(T, 1, device=dev)
    for b, s in enumerate(starts):
        L = min(window, T - s)
        acc[s:s + L] += zc[:L, b]; cnt[s:s + L] += 1
    return (acc / cnt.clamp_min(1)).cpu().numpy()                            # (T,d)


def decode(Z, y, tag):
    """BLOCKED (contiguous-time) 5-fold CV accuracy + confusion for SVM(rbf) and LDA. Blocked, not random:
    random folds leak through temporal autocorrelation (adjacent frames in train+test) and inflate accuracy
    ~10 pts on this data -- contiguous time blocks are the honest estimate for a time series."""
    cv = KFold(5, shuffle=False)
    out = {}
    for name, clf in [("SVM", SVC(C=2.0)), ("LDA", LDA())]:
        pred = cross_val_predict(clf, Z, y, cv=cv)
        out[name] = (accuracy_score(y, pred), confusion_matrix(y, pred, labels=np.unique(y)), pred)
    base = np.bincount(y).max() / len(y)
    print(f"  [{tag:16}] base={base:.3f} | " + " | ".join(f"{k}={v[0]:.3f}" for k, v in out.items()))
    return out, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to <train_dir>/model.pt")
    ap.add_argument("--out", default=None, help="output dir (default: alongside model.pt)")
    a = ap.parse_args()
    outdir = a.out or os.path.dirname(a.model)
    model, ckpt, hp = load_kato(a.model)
    dm_h = ckpt["data_hparams"]                          # mat_path, worm, window, stride, clip_negative, subsample

    # --- data (full trace) + standardize with the SAVED scaler ---
    w = load_worm(dm_h["mat_path"], dm_h["worm"])
    y = w["traces"]
    if dm_h["clip_negative"]:
        y = np.clip(y, 0, None)
    if dm_h["subsample"] > 1:
        y = y[::dm_h["subsample"]]
    states = w["states"][::dm_h["subsample"]] if dm_h["subsample"] > 1 else w["states"]
    names = w["state_names"]
    T, N = y.shape
    obs_mean, obs_scale = ckpt["obs_mean"].numpy(), ckpt["obs_scale"]
    y_std = torch.from_numpy((y - obs_mean) / obs_scale).float()

    # --- filter to latents ---
    z_hat = filter_full_trace(model, y_std, dm_h["window"], dm_h["stride"])  # (T,d)

    # --- (A) reconstruction ---
    C, d_off = model.C_cur.cpu().numpy(), model.d_cur.cpu().numpy()
    y_hat = ((z_hat @ C.T + d_off) * obs_scale + obs_mean)                   # (T,N) physical dF/F
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean(0, keepdims=True)) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    per_neuron_r2 = 1 - ((y - y_hat) ** 2).sum(0) / (((y - y.mean(0)) ** 2).sum(0) + 1e-8)
    print(f"\n=== {w['name']} | T={T} N={N} d={z_hat.shape[1]} ===")
    print(f"(A) reconstruction R2 = {r2:.3f}  (per-neuron median {np.median(per_neuron_r2):.3f})")

    # --- (B) behavior decoding (drop NOSTATE if present) ---
    keep = states < len(names)
    if "NOSTATE" in names:
        keep &= states != names.index("NOSTATE")
    print("(B) behavior decoding (BLOCKED 5-fold CV):")
    dz = np.gradient(z_hat, axis=0)                                          # latent derivative
    dec_z, base = decode(z_hat[keep], states[keep], "model z_hat")
    dec_dz, _ = decode(dz[keep], states[keep], "model dz/dt")
    # baseline: PCA of the raw neural data (same dim) -- does the model's latent beat a linear projection?
    pca_base = PCA(z_hat.shape[1]).fit_transform(y_std.numpy())
    dec_pca, _ = decode(pca_base[keep], states[keep], "PCA-of-data")
    pred_full = states.copy(); pred_full[keep] = dec_z["SVM"][2]            # model SVM blocked-CV preds per timepoint
    pred_pca_full = states.copy(); pred_pca_full[keep] = dec_pca["SVM"][2]  # PCA-baseline SVM preds (same, dropped=true)

    # =================== figures ===================
    tvec = np.arange(T) * ckpt["dt"]
    cmap = plt.get_cmap("tab10")
    ns = len(names)
    zbar = z_hat.mean(0)

    # shared embeddings
    zp3 = PCA(3).fit(z_hat - zbar); Zp = zp3.transform(z_hat - zbar)              # model latents -> 3D PCA
    Yc = (y - y.mean(0)) / (y.std(0) + 1e-6)
    yp3 = PCA(3).fit(Yc); Yp = yp3.transform(Yc)                                  # raw data -> 3D PCA (Kato)

    def _quiv(ax, P2, C, L, ok=None, xc=None):
        """per-point arrows tangent to the local direction of travel, colored by state C (ALL points, so the
        flow stays continuous). If `ok` given, overlay 'x' markers at misclassified points -- colored by the
        TRUE label `xc` (arrow = predicted state, x = what it should have been)."""
        d = np.gradient(P2, axis=0)                                          # local tangent (centered diff)
        d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9) * L        # unit direction * fixed length
        q = ax.quiver(P2[:, 0], P2[:, 1], d[:, 0], d[:, 1], C, cmap=cmap,
                      angles="xy", scale_units="xy", scale=1, width=0.004, headwidth=4, headlength=5,
                      alpha=0.8, zorder=2)
        q.set_clim(0, 9)
        if ok is not None:                                                   # x = misclassified, colored by TRUE label
            ax.scatter(P2[~ok, 0], P2[~ok, 1], c=xc[~ok], cmap=cmap, vmin=0, vmax=9, s=22, alpha=0.95,
                       marker="x", linewidths=1.1, zorder=3)

    def _pcl(ev, i):                                                        # axis label "PC{i} (xx% var)"
        return f"PC{i + 1} ({ev[i] * 100:.0f}%)"

    def _finish(ax, P2, ttl, ev):                                          # faint path + start/end + labels
        ax.plot(P2[:, 0], P2[:, 1], color="0.5", lw=0.25, alpha=0.3, zorder=1)
        ax.scatter(*P2[0], c="lime", s=95, edgecolor="k", lw=1, marker="o", zorder=5)
        ax.scatter(*P2[-1], c="red", s=95, edgecolor="k", lw=1, marker="s", zorder=5)
        ax.annotate("start", P2[0], fontsize=8, weight="bold", zorder=6)
        ax.annotate("end", P2[-1], fontsize=8, weight="bold", zorder=6)
        ax.set_title(ttl, fontsize=8); ax.set_xlabel(_pcl(ev, 0), fontsize=7); ax.set_ylabel(_pcl(ev, 1), fontsize=7)
        ax.tick_params(labelsize=6)

    def _arrow_len(P2):
        return 0.035 * float(np.mean(P2[:, :2].max(0) - P2[:, :2].min(0)))

    def _beh2d(ax, P, c, ttl, ev):                                         # arrows colored by behavior
        _quiv(ax, P[:, :2], c, _arrow_len(P)); _finish(ax, P[:, :2], ttl, ev)

    def _pred2d(ax, P, pred, true, ttl, ev):                               # arrows=predicted; x=misclassified (by TRUE)
        _quiv(ax, P[:, :2], pred, _arrow_len(P), ok=(pred == true), xc=true); _finish(ax, P[:, :2], ttl, ev)

    def _traj3d(ax, P, c, ttl, ev):
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color="k", lw=0.4, alpha=0.5)
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=c, cmap=cmap, vmin=0, vmax=9, s=3, alpha=0.6)
        ax.scatter(*P[0], c="lime", s=70, edgecolor="k", marker="o")
        ax.scatter(*P[-1], c="red", s=70, edgecolor="k", marker="s")
        ax.set_title(ttl, fontsize=8); ax.tick_params(labelsize=5)
        ax.set_xlabel(_pcl(ev, 0), fontsize=6); ax.set_ylabel(_pcl(ev, 1), fontsize=6)
        ax.set_zlabel(_pcl(ev, 2), fontsize=6)

    # Fig 1: reconstruction -- best-reconstructed labeled neurons, measured vs reconstructed
    labeled = [(i, s) for i, s in enumerate(w["neuron_ids"]) if s]
    labeled.sort(key=lambda t: -per_neuron_r2[t[0]])
    show = labeled[:6] or [(i, f"n{i}") for i in np.argsort(-per_neuron_r2)[:6]]
    fig, ax = plt.subplots(len(show), 1, figsize=(12, 1.4 * len(show)), sharex=True)
    for k, (i, nm) in enumerate(show):
        ax[k].plot(tvec, y[:, i], lw=0.7, label="measured")
        ax[k].plot(tvec, y_hat[:, i], lw=0.9, label="reconstructed")
        ax[k].set_ylabel(f"{nm}\nR2={per_neuron_r2[i]:.2f}", fontsize=7, rotation=0, ha="right", va="center")
        ax[k].set_yticks([])
    ax[0].legend(fontsize=8, ncol=2, loc="upper right"); ax[-1].set_xlabel("time (s)")
    fig.suptitle(f"{w['name']}  reconstruction  (overall R2={r2:.3f})")
    fig.tight_layout(); fig.savefig(f"{outdir}/eval_recon.png", dpi=110); plt.close(fig)

    # Fig 2: OVERVIEW -- behavior strip + latents offset + manifold(model vs raw, 2D+3D) + t-SNE + confusion
    fig = plt.figure(figsize=(15, 13)); gs = fig.add_gridspec(4, 3, height_ratios=[0.5, 1.0, 1.4, 1.4])
    axL = fig.add_subplot(gs[0, :])                                               # behavior strip: model / true / PCA
    axL.imshow(np.stack([pred_full, states, pred_pca_full]), aspect="auto", cmap=cmap, vmin=0, vmax=9,
               extent=[tvec[0], tvec[-1], 0, 3], interpolation="nearest")
    axL.axhline(1.0, color="w", lw=1.5); axL.axhline(2.0, color="w", lw=1.5)
    axL.set_yticks([0.5, 1.5, 2.5]); axL.set_yticklabels(["PCA", "true", "model"], fontsize=8)
    axL.set_title(f"{w['name']}  behavior (SVM blocked-CV): model (top) / true (mid) / PCA-of-data (bottom)   "
                  f"model acc={dec_z['SVM'][0]:.2f}  PCA acc={dec_pca['SVM'][0]:.2f}")
    axL.legend(handles=[Patch(color=cmap(i), label=nm) for i, nm in enumerate(names)],
               ncol=ns, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.5))
    axT = fig.add_subplot(gs[1, :])                                               # filtered latents offset
    for j in range(z_hat.shape[1]):
        axT.plot(tvec, z_hat[:, j] + j * 3, lw=0.6)
    axT.set_yticks([]); axT.set_xlabel("time (s)"); axT.set_title("filtered latents z_hat (offset)")
    zev, yev = zp3.explained_variance_ratio_, yp3.explained_variance_ratio_
    _pred2d(fig.add_subplot(gs[2, 0]), Zp, pred_full, states, "model z_hat PCA1-2 (PREDICTED; x=misclassified)", zev)
    _traj3d(fig.add_subplot(gs[2, 1], projection="3d"), Zp, pred_full, "model z_hat PCA1-2-3 (predicted)", zev)
    _beh2d(fig.add_subplot(gs[3, 0]), Yp, states, "RAW-data PCA1-2 (behavior) [Kato]", yev)
    _traj3d(fig.add_subplot(gs[3, 1], projection="3d"), Yp, states, "RAW-data PCA1-2-3 [Kato]", yev)
    lbl = [names[i] for i in np.unique(states[keep])]

    def _confus(ax, cm, ttl):                                                    # row-normalized confusion matrix
        ax.imshow(cm / cm.sum(1, keepdims=True).clip(1), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(lbl))); ax.set_xticklabels(lbl, fontsize=6, rotation=45)
        ax.set_yticks(range(len(lbl))); ax.set_yticklabels(lbl, fontsize=6)
        ax.set_box_aspect(1); ax.set_title(ttl, fontsize=8)

    _confus(fig.add_subplot(gs[2, 2]), dec_z["SVM"][1], f"MODEL SVM confusion (acc {dec_z['SVM'][0]:.2f})")
    _confus(fig.add_subplot(gs[3, 2]), dec_pca["SVM"][1], f"PCA-of-data SVM confusion (acc {dec_pca['SVM'][0]:.2f})")
    fig.suptitle(f"{w['name']}: latents / manifold / decoding   arrows = direction of travel (start=o end=square); "
                 f"x = misclassified   var-expl z={zp3.explained_variance_ratio_.round(2)} "
                 f"raw={yp3.explained_variance_ratio_.round(2)}", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{outdir}/eval_manifold.png", dpi=110); plt.close(fig)

    # Fig 3: dynamics -- MODEL's learned drift field (latent-PCA plane) vs DATA's empirical flow (raw-PCA plane).
    # Same style, same question from both sides: does the learned drift match how the raw data actually flows?
    spca = PCA(2).fit(z_hat)                                                      # model latent state-PCA
    ypca = PCA(2).fit(Yc)                                                         # raw-data PCA (Yc = standardized obs)

    def _model_flow(basis, P2):                                                  # streamplot field from the learned drift_net
        gx = np.linspace(P2[:, 0].min(), P2[:, 0].max(), 24)
        gy = np.linspace(P2[:, 1].min(), P2[:, 1].max(), 24)
        GX, GY = np.meshgrid(gx, gy)
        grid_z = zbar + GX.ravel()[:, None] * basis[0] + GY.ravel()[:, None] * basis[1]   # (G,d) latent points
        with torch.no_grad():
            fz = model.drift_net.net(torch.from_numpy(grid_z).float().to(dev)).cpu().numpy()
        return GX, GY, (fz @ basis[0]).reshape(GX.shape), (fz @ basis[1]).reshape(GX.shape)

    def _data_flow(P2, nb=24, bw=0.10):                                          # empirical dP/dt, kernel-smoothed on a grid
        V = np.gradient(P2, axis=0)                                              # (T,2) per-step velocity of the DATA
        gx = np.linspace(P2[:, 0].min(), P2[:, 0].max(), nb)
        gy = np.linspace(P2[:, 1].min(), P2[:, 1].max(), nb)
        GX, GY = np.meshgrid(gx, gy)
        G = np.stack([GX.ravel(), GY.ravel()], 1)                               # (Ng,2)
        h = bw * np.hypot(np.ptp(gx), np.ptp(gy))
        d2 = ((G[:, None, :] - P2[None, :, :]) ** 2).sum(-1)                    # (Ng,T) grid-to-sample distances
        W = np.exp(-d2 / (2 * h * h)); Wsum = W.sum(1)                          # Nadaraya-Watson kernel weights
        FU = (W @ V[:, 0]) / (Wsum + 1e-8); FV = (W @ V[:, 1]) / (Wsum + 1e-8)
        sparse = Wsum < 0.02 * Wsum.max()                                       # blank cells with little nearby data
        FU[sparse] = np.nan; FV[sparse] = np.nan
        return GX, GY, FU.reshape(GX.shape), FV.reshape(GX.shape)

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
    _draw(axs[0], *_model_flow(spca.components_, Zs), Zs,
          "MODEL learned drift  (latent-PCA plane)", "latent", spca.explained_variance_ratio_)
    _draw(axs[1], *_data_flow(Yp2), Yp2,
          "DATA empirical flow  (raw-PCA plane)", "raw", ypca.explained_variance_ratio_)
    fig.suptitle(f"{w['name']}  dynamics: model's learned drift vs data's empirical flow  "
                 f"(streamlines colored by speed; black = trajectory, start=o end=square)")
    fig.savefig(f"{outdir}/eval_dynamics.png", dpi=115, bbox_inches="tight"); plt.close(fig)
    print(f"\nsaved eval_{{recon,manifold,dynamics}}.png -> {outdir}")


if __name__ == "__main__":
    main()
