"""Model-agnostic eval core: the metric/figure code, decoupled from any specific model object.

A fitted model is represented by a `Result` (z_hat, y_hat, optional drift); a dataset by an
`EvalContext` (standardized obs + ground truth / behavior labels). Everything here operates on plain
arrays so opssm and every baseline are scored identically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from opssm.models.losses import kl_target_pred


# ------------------------------------------------------------------ data contracts

@dataclass
class Result:
    """What every model/baseline adapter returns for one (model, dataset) cell."""
    z_hat: np.ndarray                                   # (T, d) latent estimate in the model's own gauge
    y_hat: np.ndarray                                   # (T, N) reconstruction in STANDARDIZED obs units
    drift_fn: Optional[Callable] = None                 # z (...,d)->(...,d), model's latent frame (in-process models)
    drift_at_zhat: Optional[np.ndarray] = None          # (T, d) drift evaluated at z_hat (cross-process seam)
    g: Optional[float] = None                           # isotropic process-noise scalar, model's frame
    posterior_type: str = "filter"                      # 'filter' | 'smoother'  (fairness tag)
    window_mode: str = "whole"                          # 'windowed' | 'whole'
    runtime_s: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class EvalContext:
    """Byte-identical preprocessed data + ground truth for one dataset (from opssm/eval/data.py)."""
    name: str
    system: str
    latent_dim: int
    dt: float
    obs_mean: np.ndarray
    obs_scale: float
    noise_std_eff: float
    obs_std_fit: np.ndarray                             # (T,B,N) standardized FIT split
    mask_fit: np.ndarray                                # (T,B,1)
    obs_std_eval: np.ndarray                            # (T,B,N) standardized EVAL split
    mask_eval: np.ndarray                               # (T,B,1)
    obs_raw_eval: Optional[np.ndarray] = None           # (T,N) physical units, for per-neuron recon figs
    # --- ground truth (synthetic only; None on Kato) ---
    z_true: Optional[np.ndarray] = None                 # (T,B,d) or (T,d)
    C_true: Optional[np.ndarray] = None
    d_true: Optional[np.ndarray] = None
    true_drift: Optional[Callable] = None
    sigma_true: Optional[float] = None
    z_grid: Optional[np.ndarray] = None                 # d==1 oracle grid
    filt: Optional[np.ndarray] = None                   # d==1 exact-filter target
    # --- behavior labels (Kato only; None on synthetic) ---
    states: Optional[np.ndarray] = None                 # (T,) int
    state_names: Optional[list] = None
    neuron_ids: Optional[list] = None


# ------------------------------------------------------------------ helpers

def _t(x, device=None, dtype=torch.float32):
    """numpy/torch -> torch tensor (no copy when already a matching tensor)."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype) if device is not None else x.to(dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def _interp1d(vals, grid, query):
    """Batched 1-D linear interpolation: vals (..., Nz) on ascending grid (Nz,), query (Nq,) -> (..., Nq)."""
    idx = torch.searchsorted(grid, query).clamp(1, grid.numel() - 1)
    x0 = grid[idx - 1]; x1 = grid[idx]
    w = ((query - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0.0, 1.0)
    return vals[..., idx - 1] * (1 - w) + vals[..., idx] * w


# ------------------------------------------------------------------ gauge-aligned GT metrics

@torch.no_grad()
def gauge_aligned(z_hat, z_true, *, drift_fn=None, drift_at_zhat=None, true_drift=None,
                  g=None, sigma=None, log_pi=None, filt=None, z_grid=None):
    """PROCRUSTES-aligned latent/drift/g metrics (moved verbatim from ZakaiFilterModule._gauge_aligned).

    A latent SDE with linear-Gaussian obs is identifiable only up to a LINEAR-MAP gauge, so scoring
    inferred-z vs true-z on an absolute frame is ill-posed. Fit the best AFFINE map (z_true ~ A z_hat + b)
    and score in the aligned frame. Dimension-agnostic (A scalar for d==1, d x d for multi-dim).

    Metric families are emitted only when their inputs are supplied, so a baseline that provides just
    (z_hat, z_true) still gets the latent metrics:
      - lat_* / s_fit         : always (needs only z_hat, z_true)
      - drift_*               : iff (drift_fn OR drift_at_zhat) AND true_drift
      - g_*                   : iff g and sigma
      - kl_aln                : iff d==1 and log_pi, filt, z_grid  (the 1-D grid oracle)
    `drift_at_zhat` is the drift evaluated AT the z_hat points (for cross-process/JAX models that can't
    pass a live callable); otherwise `drift_fn` is called on z_hat.
    """
    M0 = _t(z_hat); Zt = _t(z_true, device=M0.device)
    d = Zt.shape[-1] if Zt.dim() >= 2 else 1                         # true latent dim lives in z_true's last axis
    M = M0.reshape(-1, d); Z = Zt.reshape(-1, d)                     # inferred vs true latent points (N,d)
    # AFFINE Procrustes gauge z_true ~ A z_hat + b. The OFFSET b matters for offset latents (Lorenz z3
    # mean ~25); drift transforms under the LINEAR part A only (an offset doesn't change velocities).
    Mh = torch.cat([M, torch.ones_like(M[:, :1])], dim=-1)          # (N,d+1) design [z_hat, 1]
    sol = torch.linalg.lstsq(Mh, Z).solution                        # (d+1,d): Z ~ [M,1] @ sol
    A, b = sol[:d], sol[d]                                          # (d,d) linear part, (d,) offset
    z_al = M @ A + b                                                # aligned latent (offset-corrected)
    gscale = A.det().abs().pow(1.0 / d).item()                     # |A|^(1/d)
    e_lat = (z_al - Z).pow(2).sum(-1)                              # (N,) per-point squared latent error
    z_scale = (Z - Z.mean(0)).pow(2).sum(-1).mean().sqrt().clamp_min(1e-8)   # RMS spread of true latent

    out = {"s_fit": float(A.reshape(-1)[0]) if d == 1 else gscale,
           "lat_l2_raw": (M - Z).pow(2).sum(-1).mean().sqrt().item(),
           "lat_l2_aln": e_lat.mean().sqrt().item(),
           "lat_rmse_aln": (e_lat.mean() / d).sqrt().item(),
           "lat_rel": (e_lat.mean().sqrt() / z_scale).item()}

    # drift metrics -- need the model's drift at z_hat mapped through A, vs the true drift at z_al
    if (drift_fn is not None or drift_at_zhat is not None) and true_drift is not None:
        f_M = _t(drift_at_zhat, device=M.device).reshape(-1, d) if drift_at_zhat is not None \
            else _t(drift_fn(M), device=M.device).reshape(-1, d)
        f_al = f_M @ A                                             # drift maps under A (offset-free)
        f_true = _t(true_drift(z_al), device=M.device).reshape(-1, d)
        on = (z_al.abs().le(1.5).all(-1) if d == 1
              else torch.ones(z_al.shape[0], dtype=torch.bool, device=z_al.device))
        e_drf = (f_al[on] - f_true[on]).pow(2).sum(-1)
        f_scale = f_true[on].pow(2).sum(-1).mean().sqrt().clamp_min(1e-8)
        out.update({"drift_l2_aln": e_drf.mean().sqrt().item(),
                    "drift_rmse_aln": (e_drf.mean() / d).sqrt().item(),
                    "drift_rel": (e_drf.mean().sqrt() / f_scale).item()})

    # diffusion metrics
    if g is not None and sigma is not None:
        out["g_aln"] = gscale * float(g)
        out["g_rel"] = gscale * float(g) / max(float(sigma), 1e-8)

    # KL to the exact filter -- 1-D grid oracle only
    if d == 1 and log_pi is not None and filt is not None and z_grid is not None:
        s = float(A.reshape(-1)[0]); b0 = float(b.reshape(-1)[0]); zg = _t(z_grid, device=M.device)
        lp = _t(log_pi, device=M.device)
        pi_al = _interp1d(lp.exp(), zg, (zg - b0) / s).clamp_min(0) / abs(s)
        pi_al = pi_al / pi_al.sum(-1, keepdim=True).clamp_min(1e-12)
        out["kl_aln"] = kl_target_pred(_t(filt, device=M.device), pi_al.clamp_min(1e-20).log()).item()
    return out


# ------------------------------------------------------------------ obs / decoding metrics (already array-only)

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
