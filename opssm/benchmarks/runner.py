"""Train/select on observations, then evaluate an untouched test split."""
import argparse
import csv
from dataclasses import replace
import json
import platform
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

from .data import PRESETS, fingerprint, make_dataset, save_dataset, smoke_config
from .metrics import aggregate

MODELS = ("kf", "ekf", "ukf", "slds", "latent_sde", "sde_matching", "opssm")


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", choices=[*PRESETS, "kato"], default=["doublewell", "vanderpol", "lorenz"])
    ap.add_argument("--kato-mat", type=Path, help="Kato 2015 MATLAB file (recommended: WT_NoStim.mat)")
    ap.add_argument("--kato-worms", nargs="+", type=int, help="Zero-based worm indices; default: all worms")
    ap.add_argument("--kato-latent-dim", type=int, default=10)
    ap.add_argument("--kato-window", type=int, default=200)
    ap.add_argument("--kato-stride", type=int, default=100, help="Training stride; evaluation uses disjoint windows")
    ap.add_argument("--kato-gap", type=int, default=30, help="Unused frames after each temporal split boundary")
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, default=Path("dump/benchmarks"))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--modes", type=int, default=3)
    ap.add_argument("--solver-dt", type=float, default=.01)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--fixed-obs-noise", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="Shortened data and tiny models; NOT benchmark results")
    ap.add_argument("--generate-only", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip cells that already hold a result.json instead of failing. Required for "
                         "SLURM --requeue: a preempted array task reruns and must resume, not abort.")
    ap.add_argument("--dataset-config", type=Path, help="JSON mapping preset names to DatasetConfig overrides")
    ap.add_argument("--opssm-python", default=sys.executable, help="Python in the separate JAX environment")
    ap.add_argument("--opssm-config", type=Path, help="JSON overrides for OPSSM hyperparameters")
    ap.add_argument("--opssm-experiment", help="Hydra experiment for OPSSM; default: matching duncker_*/kato")
    ap.add_argument("--opssm-config-dir", type=Path, help="Hydra configs directory; default: repository configs/")
    ap.add_argument("--opssm-monitor", default="val_forecast_rmse",
                    help="OPSSM checkpoint-selection metric. Default is the ground-truth-free prefix-only "
                         "validation forecast RMSE; recon_r2 saturates and selects far too early. Drift "
                         "metrics (drift_rel/drift_l2) are ORACLE selection and are unavailable on Kato.")
    ap.add_argument("--opssm-monitor-mode", choices=("min", "max"),
                    help="Override the monitor direction; inferred from the metric by default")
    ap.add_argument("--opssm-select-samples", type=int, default=16,
                    help="Forecast samples used per validation for checkpoint selection")
    ap.add_argument("--opssm-gt-diagnostics", action="store_true",
                    help="Also log ground-truth drift/latent metrics each validation (synthetic only). "
                         "Diagnostic only: they are never used to select a checkpoint.")
    args = ap.parse_args(argv)
    if "kato" in args.datasets and (args.kato_mat is None or not args.kato_mat.is_file()):
        ap.error("--datasets kato requires --kato-mat pointing to a downloaded Kato MATLAB file")
    if args.smoke:
        args.steps, args.val_every = min(args.steps, 3), 1
        args.hidden, args.samples = min(args.hidden, 8), min(args.samples, 8)
    if min(args.steps, args.val_every, args.samples, args.batch_size, args.hidden, args.modes) < 1:
        ap.error("Budgets and dimensions must be positive")
    if args.samples < 2 or min(args.solver_dt, args.lr) <= 0 or min(args.seeds) < 0:
        ap.error("Need >=2 posterior samples and positive dt/lr, nonnegative seeds")
    args.out.mkdir(parents=True, exist_ok=True)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    overrides = json.loads(args.dataset_config.read_text()) if args.dataset_config else {}
    if set(overrides) - set(PRESETS):
        ap.error(f"Unknown dataset overrides: {sorted(set(overrides) - set(PRESETS))}")
    write_json(args.out / "invocation.json", dict(settings=settings, python=platform.python_version()))
    failed = False
    datasets = []
    for dataset in args.datasets:
        if dataset == "kato":
            from .kato import KatoConfig, make_kato_dataset
            from opssm.data.kato.load import n_worms
            worms = args.kato_worms if args.kato_worms is not None else range(n_worms(args.kato_mat))
            for worm in worms:
                cfg, data = make_kato_dataset(KatoConfig(str(args.kato_mat), worm=worm,
                    latent_dim=args.kato_latent_dim, window=args.kato_window,
                    train_stride=args.kato_stride, gap_frames=args.kato_gap), smoke=args.smoke)
                datasets.append((cfg, data))
        else:
            cfg = replace(PRESETS[dataset], **overrides.get(dataset, {}))
            datasets.append((smoke_config(cfg) if args.smoke else cfg, None))
    for cfg, fixed_data in datasets:
        for seed in args.seeds:
            data = make_dataset(cfg, seed) if fixed_data is None else fixed_data
            root = args.out / cfg.name / f"seed_{seed}"
            save_dataset(root / "data.npz", data)
            if args.generate_only:
                continue
            for name in args.models:
                directory = root / name
                directory.mkdir(parents=True, exist_ok=True)
                # Preserve existing runs; accidental reruns must not silently overwrite results.
                if (directory / "result.json").exists():
                    if args.skip_existing:
                        print(f"skip existing: {directory}", flush=True)
                        continue
                    raise FileExistsError(f"Run exists: {directory}. Choose a new --out directory.")
                row = dict(dataset=cfg.name, model=name, seed=seed, steps=args.steps, smoke=args.smoke,
                           dataset_hash=fingerprint(data), settings=settings)
                try:
                    if name == "opssm":
                        job = dict(data=str((root / "data.npz").resolve()), out=str(directory.resolve()),
                                   steps=args.steps, seed=seed, smoke=args.smoke, samples=args.samples,
                                   solver_dt=args.solver_dt, fixed_obs_noise=args.fixed_obs_noise,
                                   config=str(args.opssm_config.resolve()) if args.opssm_config else None,
                                   experiment=args.opssm_experiment, val_every=args.val_every,
                                   config_dir=str(args.opssm_config_dir.resolve()) if args.opssm_config_dir else None,
                                   monitor=args.opssm_monitor, monitor_mode=args.opssm_monitor_mode,
                                   select_samples=args.opssm_select_samples,
                                   gt_diagnostics=args.opssm_gt_diagnostics)
                        write_json(directory / "job.json", job)
                        command = [args.opssm_python, "-m", "opssm.benchmarks.opssm_worker", str(directory / "job.json")]
                        subprocess.run(command, check=True)
                        metrics = json.loads((directory / "worker_metrics.json").read_text())
                    else:
                        from .train import fit_jax
                        metrics = fit_jax(name, data, cfg, args, directory, seed)
                    row.update(metrics, status="ok")
                except Exception as exc:
                    failed = True
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    (directory / "error.txt").write_text(traceback.format_exc())
                    print(row["error"], file=sys.stderr, flush=True)
                write_json(directory / "result.json", row)
    rows = [json.loads(p.read_text()) for p in sorted(args.out.glob("*/seed_*/*/result.json"))]
    write_json(args.out / "summary.json", aggregate(rows))
    if rows:
        fields = sorted(set().union(*(row.keys() for row in rows)) - {"settings"})
        with (args.out / "metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
