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
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
        out[name] = (accuracy_score(y, pred), confusion_matrix(y, pred, labels=np.unique(y)))
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
    decode(pca_base[keep], states[keep], "PCA-of-data")

    # =================== figures ===================
    tvec = np.arange(T) * ckpt["dt"]
    cmap = plt.get_cmap("tab10")

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

    # Fig 2: latents over time (behavior strip) + PCA/t-SNE colored by behavior
    pca = PCA(2).fit(z_hat); Zp = pca.transform(z_hat)
    fig = plt.figure(figsize=(14, 8)); gs = fig.add_gridspec(3, 3)
    axL = fig.add_subplot(gs[0, :])
    axL.imshow(states[None], aspect="auto", cmap=cmap, vmin=0, vmax=9, extent=[tvec[0], tvec[-1], 0, 1])
    axL.set_yticks([]); axL.set_title(f"{w['name']}  behavior state")
    axL.legend(handles=[Patch(color=cmap(i), label=nm) for i, nm in enumerate(names)],
               ncol=len(names), fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.2))
    axT = fig.add_subplot(gs[1, :])
    for j in range(z_hat.shape[1]):
        axT.plot(tvec, z_hat[:, j] + j * 3, lw=0.6)
    axT.set_yticks([]); axT.set_xlabel("time (s)"); axT.set_title("filtered latents z_hat (offset)")
    for name, emb, axg in [("PCA", Zp, fig.add_subplot(gs[2, 0])),
                           ("t-SNE", TSNE(2, init="pca", perplexity=30, random_state=0).fit_transform(z_hat),
                            fig.add_subplot(gs[2, 1]))]:
        axg.scatter(emb[:, 0], emb[:, 1], c=states, cmap=cmap, vmin=0, vmax=9, s=4, alpha=0.6)
        axg.set_title(f"{name} of z_hat (by behavior)"); axg.set_xticks([]); axg.set_yticks([])
    # confusion matrix (SVM on latents)
    axc = fig.add_subplot(gs[2, 2]); cm = dec_z["SVM"][1]
    axc.imshow(cm / cm.sum(1, keepdims=True).clip(1), cmap="Blues", vmin=0, vmax=1)
    lbl = [names[i] for i in np.unique(states[keep])]
    axc.set_xticks(range(len(lbl))); axc.set_xticklabels(lbl, fontsize=6, rotation=45)
    axc.set_yticks(range(len(lbl))); axc.set_yticklabels(lbl, fontsize=6)
    axc.set_title(f"SVM confusion (acc {dec_z['SVM'][0]:.2f})")
    fig.tight_layout(); fig.savefig(f"{outdir}/eval_latents.png", dpi=110); plt.close(fig)

    # Fig 3: learned drift field in the top-2 PC plane + trajectory
    fig, ax = plt.subplots(figsize=(7, 6))
    gx = np.linspace(Zp[:, 0].min(), Zp[:, 0].max(), 22)
    gy = np.linspace(Zp[:, 1].min(), Zp[:, 1].max(), 22)
    GX, GY = np.meshgrid(gx, gy)
    grid_pc = np.stack([GX.ravel(), GY.ravel()], 1)
    grid_z = pca.inverse_transform(grid_pc)                                  # (G,d) latent
    with torch.no_grad():
        f = model.drift_net.net(torch.from_numpy(grid_z).float().to(dev)).cpu().numpy()   # (G,d)
    f_pc = f @ pca.components_.T                                             # project drift onto PCs
    ax.streamplot(GX, GY, f_pc[:, 0].reshape(GX.shape), f_pc[:, 1].reshape(GX.shape),
                  color="0.6", density=1.1, linewidth=0.6)
    ax.scatter(Zp[:, 0], Zp[:, 1], c=states, cmap=cmap, vmin=0, vmax=9, s=5, alpha=0.6)
    ax.set_title(f"{w['name']}  learned drift field (top-2 PC plane) + latent trajectory")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout(); fig.savefig(f"{outdir}/eval_dynamics.png", dpi=110); plt.close(fig)
    print(f"\nsaved eval_recon.png, eval_latents.png, eval_dynamics.png -> {outdir}")


if __name__ == "__main__":
    main()
