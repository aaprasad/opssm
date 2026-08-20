"""Baselining grid runner: (dataset x model) -> dump/baselines/metrics.csv, every model scored by the
identical eval-core (recon R2, gauge-aligned latent/drift/g on synthetic, blocked-CV decode on Kato).

Usage:
  python scripts/run_baselines.py                              # full grid
  python scripts/run_baselines.py --datasets vanderpol --models opssm,ekf,gpslds
  python scripts/run_baselines.py --steps 800                  # per-model fit budget (subprocess baselines)
"""
import argparse
import csv
import os
import time
import traceback

import numpy as np
import torch

from opssm.eval.data import make_context
from opssm.eval.baselines import get_adapter
from opssm.eval.core import gauge_aligned, recon_metrics, decode

OPSSM_CKPT = {
    "doublewell":   "dump/opssm_dw/model.pt",
    "vanderpol":    "dump/opssm_vdp_mala/model.pt",
    "lorenz":       "dump/opssm_lorenz/model.pt",
    "kato_stim0":   "dump/kato_stim0/model.pt",
    "kato_nostim0": "dump/kato_nostim0/model.pt",
}
ALL_DATASETS = ["doublewell", "vanderpol", "lorenz", "kato_stim0", "kato_nostim0"]
ALL_MODELS = ["opssm", "ekf", "gpslds", "visde", "latent_sde"]
COLS = ["model", "dataset", "posterior_type", "window_mode", "latent_dim", "recon_r2",
        "lat_rel", "lat_rmse_aln", "lat_l2_aln", "lat_rmse_trS", "lat_rmse_trS_std",
        "drift_rel", "drift_rmse_aln", "drift_l2_aln", "drift_rmse_norm",
        "g_rel", "decode_svm", "decode_lda", "decode_base", "runtime_s"]


def score(res, ctx):
    """One metrics row. recon always; gauge-aligned GT on synthetic; blocked-CV decode on Kato."""
    row = {c: np.nan for c in COLS}
    row.update(posterior_type=res.posterior_type, window_mode=res.window_mode,
               latent_dim=ctx.latent_dim, runtime_s=round(float(res.runtime_s), 1))
    zt = None if ctx.z_true is None else np.asarray(ctx.z_true)
    ot = np.asarray(ctx.obs_std_eval)
    zh = np.asarray(res.z_hat)
    # a model may infer fewer eval trials than GT (gpSLDS/SING-GP fits full-batch over a few trials);
    # score it on the matching subset -- workers use the FIRST n eval trials.
    if zh.ndim == 3 and ctx.window is None and zh.shape[1] < ot.shape[1]:
        n = zh.shape[1]
        ot = ot[:, :n]
        zt = zt[:, :n] if zt is not None else None
    y_true = ot[:, 0, :] if ctx.window is not None else ot
    row["recon_r2"] = round(float(recon_metrics(y_true, res.y_hat)[0]), 4)
    if zt is not None:                                                  # synthetic: gauge-aligned GT metrics
        al = gauge_aligned(res.z_hat, zt, drift_fn=res.drift_fn,
                           drift_at_zhat=res.drift_at_zhat, true_drift=ctx.true_drift,
                           g=res.g, sigma=ctx.sigma_true, z_cov=res.z_cov,
                           filt=ctx.filt, z_grid=ctx.z_grid)
        for k in ("lat_rel", "lat_rmse_aln", "lat_l2_aln", "lat_rmse_trS", "lat_rmse_trS_std",
                  "drift_rel", "drift_rmse_aln", "drift_l2_aln", "drift_rmse_norm", "g_rel"):
            if k in al:
                row[k] = round(float(al[k]), 4)
    if ctx.states is not None:                                          # Kato: behavior decoding
        dec, base = decode(np.asarray(res.z_hat), ctx.states, "decode", verbose=False)
        row["decode_svm"] = round(float(dec["SVM"][0]), 4)
        row["decode_lda"] = round(float(dec["LDA"][0]), 4)
        row["decode_base"] = round(float(base), 4)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(ALL_DATASETS))
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--steps", type=int, default=None, help="fit budget for subprocess baselines")
    ap.add_argument("--out", default="dump/baselines")
    a = ap.parse_args()
    datasets = a.datasets.split(",")
    models = a.models.split(",")
    os.makedirs(a.out, exist_ok=True)
    csv_path = os.path.join(a.out, "metrics.csv")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}                                                       # (model,dataset) -> row; upsert so
    if os.path.exists(csv_path):                                       # re-runs add/replace cells, not clobber
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                results[(r["model"], r["dataset"])] = r
    for ds in datasets:
        ctx = make_context(ds, device=dev)
        for m in models:
            cfg = {}
            if m == "opssm":
                ckpt = OPSSM_CKPT.get(ds)
                if not ckpt or not os.path.exists(ckpt):
                    print(f"[skip] opssm/{ds}: no checkpoint at {ckpt}")
                    continue
                cfg["checkpoint"] = ckpt
            elif a.steps is not None:
                cfg["steps"] = a.steps
            print(f"\n=== {m} / {ds} ===", flush=True)
            t0 = time.time()
            try:
                res = get_adapter(m).fit_predict(ctx, cfg)
                row = score(res, ctx)
                row.update(model=m, dataset=ds)
                results[(m, ds)] = row
                print(f"  -> recon={row['recon_r2']} lat_rel={row['lat_rel']} "
                      f"drift_rel={row['drift_rel']} decode_svm={row['decode_svm']} "
                      f"[{row['posterior_type']}] {time.time()-t0:.0f}s", flush=True)
            except Exception:
                print(f"  !! FAILED {m}/{ds}:\n{traceback.format_exc()}", flush=True)
            # incremental write so a mid-grid crash still leaves a usable table
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore", restval="")
                w.writeheader()
                w.writerows(results.values())
    print(f"\nwrote {csv_path} ({len(results)} cells)")


if __name__ == "__main__":
    main()
