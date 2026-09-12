"""Train/select on observations, then evaluate an untouched test split."""
import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

from .data import (PRESETS, config_from_data, fingerprint, make_dataset, save_dataset,
                   smoke_config)
from .metrics import aggregate

# SLURM_RESTART_COUNT is the one that answers "was this preempted?": SLURM increments it each time
# it requeues the task, so a nonzero value means this attempt follows a preemption or a timeout.
SLURM_KEYS = {"SLURM_JOB_ID": "job_id", "SLURM_ARRAY_JOB_ID": "array_job_id",
              "SLURM_ARRAY_TASK_ID": "array_task_id", "SLURM_RESTART_COUNT": "restart_count",
              "SLURM_JOB_PARTITION": "partition", "SLURMD_NODENAME": "node",
              "SLURM_CLUSTER_NAME": "cluster", "SLURM_JOB_QOS": "qos"}


def slurm_info():
    """SLURM identifiers for this attempt; empty off-cluster."""
    return {name: os.environ[key] for key, name in SLURM_KEYS.items() if key in os.environ}


def record_attempt(directory, **fields):
    """Append one line per attempt, BEFORE the work starts.

    A preempted task is killed without writing result.json, so this is the only durable trace
    that the cell was ever tried: the line names the SLURM job, node and restart count, which is
    what `sacct -j <job_id>` needs to say whether it was preempted, timed out or failed.
    """
    directory.mkdir(parents=True, exist_ok=True)
    entry = dict(started_at=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                 host=platform.node(), **fields, slurm=slurm_info())
    with (directory / "attempts.jsonl").open("a") as handle:
        handle.write(json.dumps(entry) + "\n")
    return entry

DEFAULT_MODELS = ("kf", "ekf", "ukf", "rslds", "latent_sde", "sde_matching", "opssm")
MODELS = (*DEFAULT_MODELS, "slds")  # Keep legacy SLDS explicit and results identifiable.


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
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(DEFAULT_MODELS))
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
    ap.add_argument("--reuse-data", action="store_true",
                    help="Adopt an existing data.npz instead of regenerating from the preset. Use this to "
                         "add a model to a cell whose other models have already finished: the new run then "
                         "trains on the SAME bytes they did, which is what makes the cell comparable. "
                         "Without it, a data.npz that disagrees with the current preset/code is a hard error.")
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
            root = args.out / cfg.name / f"seed_{seed}"
            existing = root / "data.npz"
            cell_cfg = cfg
            if args.reuse_data and fixed_data is None and existing.is_file():
                # Adopting the file makes it the source of truth: every downstream consumer, and the
                # dataset_hash recorded in the row, then describes the bytes actually trained on. The
                # OPSSM worker reads this path directly, so regenerating instead would let the two
                # paths diverge while still reporting one hash.
                with np.load(existing, allow_pickle=False) as loaded:
                    data = dict(loaded)
                # The file's stored config governs the cell too, not just its arrays: fit_jax takes
                # latent_dim, diffusion_type and obs_dim from here, so leaving the preset in place
                # would configure the baselines for data they are not training on. The OPSSM worker
                # already derives its config this way, and the two must not disagree.
                cell_cfg = config_from_data(data)
                if cell_cfg.name != cfg.name:
                    raise ValueError(f"{existing} holds dataset {cell_cfg.name!r}, not {cfg.name!r}")
                generated = fingerprint(make_dataset(cfg, seed))
                if fingerprint(data) != generated:
                    print(f"reusing existing dataset {existing} (hash {fingerprint(data)[:12]}); the current "
                          f"preset/code would generate {generated[:12]} -- results are tied to the file, "
                          f"not to this checkout's config", flush=True)
            else:
                data = make_dataset(cfg, seed) if fixed_data is None else fixed_data
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
                attempt = record_attempt(directory, dataset=cfg.name, model=name, seed=seed)
                row = dict(dataset=cfg.name, model=name, seed=seed, steps=args.steps, smoke=args.smoke,
                           dataset_hash=fingerprint(data), settings=settings,
                           started_at=attempt["started_at"], host=attempt["host"], **(
                               {f"slurm_{k}": v for k, v in attempt["slurm"].items()}))
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
                        # Capture stderr so a crash reports WHY, not just an exit status. stdout is
                        # left attached, so training progress still streams into the job log live.
                        completed = subprocess.run(command, stderr=subprocess.PIPE, text=True)
                        if completed.returncode:
                            detail = (completed.stderr or "").strip()
                            if detail:
                                (directory / "worker_stderr.txt").write_text(detail + "\n")
                                print(detail, file=sys.stderr, flush=True)
                            cause = detail.splitlines()[-1] if detail else "no stderr captured"
                            raise RuntimeError(
                                f"OPSSM worker exited {completed.returncode}: {cause}"
                                + (f" (full traceback in {directory / 'worker_stderr.txt'})" if detail else "")
                                + f"; reproduce with: {args.opssm_python} -m opssm.benchmarks.opssm_worker "
                                  f"{directory / 'job.json'}")
                        metrics = json.loads((directory / "worker_metrics.json").read_text())
                    else:
                        from .train import fit_jax
                        metrics = fit_jax(name, data, cell_cfg, args, directory, seed)
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
