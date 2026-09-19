"""Hydra entrypoint for the benchmark comparison suite -- one multirun job per GPU cell.

Each job fits ONE (data point, dataset, seed, model) cell and writes into a shared results tree,
so a multirun is a job array over the product of the comma-separated axes. Concurrent training on
one GPU is not supported, hence one cell per job rather than one dataset or one seed per job.

Sweeping the data configs is the point: anything under `dataset_config` overrides that dataset's
DatasetConfig, so `dataset_config.noise_std=0.25,0.5,1.0` is a three-point sweep sharing the rest.
(`model`/`data` are existing Hydra groups in this repo, hence the names `method`/`dataset_config`.)

    python scripts/benchmark_sweep.py -m hydra/launcher=submitit_slurm \
        dataset=vanderpol seed=0,1,2,3,4 method=kf,ekf,ukf,rslds,latent_sde,sde_matching,opssm \
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


def _parts(overrides, prefix=""):
    return [f"{prefix}{k}{str(v).replace('.', 'p').replace('-', 'm')}"
            for k, v in sorted(overrides.items())]


def point_name(overrides, model_overrides=None, gpslds_overrides=None):
    """Stable, filesystem-safe directory name for one set of config overrides.

    Model overrides are prefixed and folded into the SAME name as the data overrides: a point is a
    configuration, not just a dataset, so two activations must not share a results directory and
    silently skip each other through the `already complete` check. Each model's knobs carry their
    own prefix so an OPSSM point and a gpSLDS point never read as the same configuration.
    """
    parts = (_parts(overrides) + _parts(model_overrides or {}, prefix="opssm_")
             + _parts(gpslds_overrides or {}, prefix="gpslds_"))
    return "_".join(parts) if parts else "baseline"


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark")
def main(cfg):
    from opssm.benchmarks.data import PRESETS
    from opssm.benchmarks.runner import main as run_benchmarks

    # null means "keep the preset's value", so only non-null entries become overrides.
    overrides = {k: v for k, v in (OmegaConf.to_container(cfg.dataset_config, resolve=True) or {}).items()
                 if v is not None}
    # OPSSM hyperparameter overrides, merged over the resolved Hydra config by the worker. Only
    # meaningful for method=opssm, so reject them elsewhere rather than writing a point name that
    # claims a setting the run never applied.
    model_overrides = {k: v for k, v in
                       (OmegaConf.to_container(cfg.opssm.get("overrides"), resolve=True) or {}).items()
                       if v is not None}
    if model_overrides and cfg.method != "opssm":
        raise SystemExit(f"opssm.overrides only applies to method=opssm, not {cfg.method!r}")
    # gpSLDS knobs that differ from their defaults are folded into the point name the same way, so
    # two gpSLDS configurations never share a results directory and silently skip each other.
    gpslds_defaults = {"states": None, "sigma": 1.0, "iters": 50, "iters_e": 15, "iters_m": 50,
                       "iters_infer": 15, "lr": 1e-4, "tau": 0.5,
                       "inducing_per_axis": None, "inducing_pad": 0.25}
    gpslds_cfg = OmegaConf.to_container(cfg.get("gpslds"), resolve=True) or {}
    gpslds_overrides = {k: v for k, v in gpslds_cfg.items()
                        if v is not None and v != gpslds_defaults.get(k)}
    if gpslds_overrides and cfg.method != "gpslds":
        raise SystemExit(f"gpslds.* only applies to method=gpslds, not {cfg.method!r}")
    if "drift_activation" in model_overrides:
        from opssm.models.activations import ACTIVATION_NAMES
        if model_overrides["drift_activation"] not in ACTIVATION_NAMES:
            raise SystemExit(f"drift_activation must be one of {sorted(ACTIVATION_NAMES)}")
    kato = cfg.dataset == "kato"
    if kato:
        # Kato is a real recording, not a generator: it has no DatasetConfig to override or validate,
        # and its sweep axes live under `kato` instead. The runner derives the dataset directory name
        # from the file stem, worm and fold, so reproduce it here to find this job's cell.
        if not cfg.kato.mat:
            raise SystemExit("dataset=kato requires kato.mat=/path/to/WT_NoStim.mat")
        if not Path(cfg.kato.mat).is_file():
            raise SystemExit(f"kato.mat not found: {cfg.kato.mat}")
        if overrides:
            raise SystemExit("dataset_config does not apply to kato; sweep the kato.* knobs instead")
        suffix = ("_full" if cfg.kato.fold < 0 else
                  "" if cfg.kato.folds <= 1 else f"_fold{cfg.kato.fold}of{cfg.kato.folds}")
        dataset_dir = f"kato_{Path(cfg.kato.mat).stem}_worm{cfg.kato.worm}{suffix}"
    elif cfg.dataset not in PRESETS:
        raise SystemExit(f"Unknown dataset {cfg.dataset!r}; choose from {[*sorted(PRESETS), 'kato']}")
    else:
        # Validate the resolved config here, so a bad sweep point fails this job rather than
        # surfacing halfway through training.
        replace(PRESETS[cfg.dataset], **overrides).validate()
        dataset_dir = cfg.dataset

    point = point_name(overrides, model_overrides, gpslds_overrides)
    out = Path(cfg.results_root).resolve() / point
    cell = out / dataset_dir / f"seed_{cfg.seed}" / cfg.method
    if (cell / "result.json").exists():
        print(f"already complete: {cell}", flush=True)
        return 0

    steps = int(OmegaConf.select(cfg, f"steps_per_dataset.{cfg.dataset}") or cfg.steps)
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / f"dataset_config_{point}.json"
    if overrides and not config_path.exists():
        config_path.write_text(json.dumps({cfg.dataset: overrides}, indent=2) + "\n")
    model_config_path = out / f"opssm_config_{point}.json"
    if model_overrides and not model_config_path.exists():
        model_config_path.write_text(json.dumps(model_overrides, indent=2) + "\n")

    argv = ["--datasets", str(cfg.dataset), "--models", str(cfg.method), "--seeds", str(cfg.seed),
            "--steps", str(steps), "--val-every", str(cfg.val_every), "--samples", str(cfg.samples),
            "--solver-dt", str(cfg.solver_dt), "--batch-size", str(cfg.batch_size),
            "--hidden", str(cfg.hidden), "--modes", str(cfg.modes), "--lr", str(cfg.lr),
            "--out", str(out), "--skip-existing",
            "--opssm-monitor", str(cfg.opssm.monitor),
            "--opssm-select-samples", str(cfg.opssm.select_samples)]
    if overrides:
        argv += ["--dataset-config", str(config_path)]
    if model_overrides:
        argv += ["--opssm-config", str(model_config_path)]
    if kato:
        argv += ["--kato-mat", str(cfg.kato.mat), "--kato-worms", str(cfg.kato.worm),
                 "--kato-folds", str(cfg.kato.folds), "--kato-fold", str(cfg.kato.fold),
                 "--kato-latent-dim", str(cfg.kato.latent_dim), "--kato-window", str(cfg.kato.window),
                 "--kato-stride", str(cfg.kato.stride), "--kato-gap", str(cfg.kato.gap)]
    if cfg.opssm.monitor_mode:
        argv += ["--opssm-monitor-mode", str(cfg.opssm.monitor_mode)]
    if cfg.opssm.gt_diagnostics:
        argv += ["--opssm-gt-diagnostics"]
    if cfg.opssm.experiment:
        argv += ["--opssm-experiment", str(cfg.opssm.experiment)]
    if cfg.opssm.config_dir:
        argv += ["--opssm-config-dir", str(cfg.opssm.config_dir)]
    if cfg.method == "gpslds":
        for key, value in gpslds_cfg.items():
            if value is not None:
                argv += [f"--gpslds-{key.replace('_', '-')}", str(value)]
    if cfg.get("reuse_data", False):
        argv += ["--reuse-data"]
    if cfg.fixed_obs_noise:
        argv += ["--fixed-obs-noise"]
    if cfg.smoke:
        argv += ["--smoke"]

    # Disabling preallocation is what caused the em_lorenz OOM; leave XLA's allocator alone.
    os.environ.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    print(f"[{point}] {dataset_dir} seed={cfg.seed} {cfg.method} steps={steps} -> {cell}", flush=True)
    return run_benchmarks(argv)


if __name__ == "__main__":
    raise SystemExit(main())
