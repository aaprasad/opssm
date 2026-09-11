"""Hydra entrypoint for the benchmark comparison suite -- one multirun job per GPU cell.

Each job fits ONE (data point, dataset, seed, model) cell and writes into a shared results tree,
so a multirun is a job array over the product of the comma-separated axes. Concurrent training on
one GPU is not supported, hence one cell per job rather than one dataset or one seed per job.

Sweeping the data configs is the point: anything under `dataset_config` overrides that dataset's
DatasetConfig, so `dataset_config.noise_std=0.25,0.5,1.0` is a three-point sweep sharing the rest.
(`model`/`data` are existing Hydra groups in this repo, hence the names `method`/`dataset_config`.)

    python scripts/benchmark_sweep.py -m hydra/launcher=submitit_slurm \
        dataset=vanderpol seed=0,1,2,3,4 method=kf,ekf,ukf,slds,latent_sde,sde_matching,opssm \
        dataset_config.noise_std=0.25,0.5,1.0,1.5,3.0

Jobs are idempotent: a cell that already holds a result.json returns immediately, so SLURM
--requeue after preemption resumes the sweep rather than aborting on partial output.
"""
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import hydra
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Popped at import, so it applies both where the multirun is submitted and inside the task: an
# inherited SLURM_CPU_BIND from an enclosing allocation is applied to the new job step and makes
# srun reject the binding as outside its allocation.
os.environ.pop("SLURM_CPU_BIND", False)


def point_name(overrides):
    """Stable, filesystem-safe directory name for one set of data-config overrides."""
    if not overrides:
        return "baseline"
    parts = [f"{k}{str(v).replace('.', 'p').replace('-', 'm')}" for k, v in sorted(overrides.items())]
    return "_".join(parts)


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark")
def main(cfg):
    from opssm.benchmarks.data import PRESETS
    from opssm.benchmarks.runner import main as run_benchmarks

    # null means "keep the preset's value", so only non-null entries become overrides.
    overrides = {k: v for k, v in (OmegaConf.to_container(cfg.dataset_config, resolve=True) or {}).items()
                 if v is not None}
    if cfg.dataset not in PRESETS:
        raise SystemExit(f"Unknown dataset {cfg.dataset!r}; choose from {sorted(PRESETS)}")
    # Validate the resolved config here, so a bad sweep point fails this job rather than
    # surfacing halfway through training.
    replace(PRESETS[cfg.dataset], **overrides).validate()

    point = point_name(overrides)
    out = Path(cfg.results_root).resolve() / point
    cell = out / cfg.dataset / f"seed_{cfg.seed}" / cfg.method
    if (cell / "result.json").exists():
        print(f"already complete: {cell}", flush=True)
        return 0

    steps = int(OmegaConf.select(cfg, f"steps_per_dataset.{cfg.dataset}") or cfg.steps)
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / f"dataset_config_{point}.json"
    if overrides and not config_path.exists():
        config_path.write_text(json.dumps({cfg.dataset: overrides}, indent=2) + "\n")

    argv = ["--datasets", str(cfg.dataset), "--models", str(cfg.method), "--seeds", str(cfg.seed),
            "--steps", str(steps), "--val-every", str(cfg.val_every), "--samples", str(cfg.samples),
            "--solver-dt", str(cfg.solver_dt), "--batch-size", str(cfg.batch_size),
            "--hidden", str(cfg.hidden), "--modes", str(cfg.modes), "--lr", str(cfg.lr),
            "--out", str(out), "--skip-existing",
            "--opssm-monitor", str(cfg.opssm.monitor),
            "--opssm-select-samples", str(cfg.opssm.select_samples)]
    if overrides:
        argv += ["--dataset-config", str(config_path)]
    if cfg.opssm.monitor_mode:
        argv += ["--opssm-monitor-mode", str(cfg.opssm.monitor_mode)]
    if cfg.opssm.gt_diagnostics:
        argv += ["--opssm-gt-diagnostics"]
    if cfg.opssm.experiment:
        argv += ["--opssm-experiment", str(cfg.opssm.experiment)]
    if cfg.opssm.config_dir:
        argv += ["--opssm-config-dir", str(cfg.opssm.config_dir)]
    if cfg.fixed_obs_noise:
        argv += ["--fixed-obs-noise"]
    if cfg.smoke:
        argv += ["--smoke"]

    # Disabling preallocation is what caused the em_lorenz OOM; leave XLA's allocator alone.
    os.environ.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    print(f"[{point}] {cfg.dataset} seed={cfg.seed} {cfg.method} steps={steps} -> {cell}", flush=True)
    return run_benchmarks(argv)


if __name__ == "__main__":
    raise SystemExit(main())
