"""Build a SLURM array sweep over benchmark data configs.

Emits, under --out:
  points/<point>.json   dataset-config overrides, one file per sweep point (runner --dataset-config)
  manifest.tsv          one line per array task: (point, dataset, seed, model)
  submit.sh             ready-to-run sbatch invocation with the array size filled in
  sweep.json            the resolved spec, for provenance

Task granularity is one (point, dataset, seed, model) cell = one GPU, because concurrent
training on a shared GPU is not supported. Every cell of a given (point, dataset, seed)
reads the SAME pre-generated data.npz, so models remain exactly comparable.

Example -- sweep observation noise on Van der Pol, the axis the SNR finding implicates:

    python scripts/cluster/make_sweep.py --out sweeps/noise \\
        --axis vanderpol.noise_std=0.25,0.5,1.0,1.5,3.0 --seeds 0 1 2 3 4
"""
import argparse
import itertools
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from opssm.benchmarks.data import PRESETS, DatasetConfig
from opssm.benchmarks.runner import DEFAULT_MODELS, MODELS


def parse_axis(text):
    """'vanderpol.noise_std=0.25,0.5' -> ('vanderpol', 'noise_std', [0.25, 0.5])."""
    target, _, values = text.partition("=")
    dataset, _, field = target.partition(".")
    if not (dataset and field and values):
        raise ValueError(f"Malformed --axis {text!r}; expected dataset.field=v1,v2,...")
    if dataset not in PRESETS:
        raise ValueError(f"Unknown dataset {dataset!r} in --axis; choose from {sorted(PRESETS)}")
    if field not in DatasetConfig.__dataclass_fields__:
        raise ValueError(f"Unknown DatasetConfig field {field!r} in --axis")
    kind = DatasetConfig.__dataclass_fields__[field].type
    cast = {int: int, float: float, str: str}.get(kind if isinstance(kind, type) else
        {"int": int, "float": float, "str": str}.get(str(kind), str), str)
    return dataset, field, [cast(v) for v in values.split(",") if v != ""]


def label(value):
    return str(value).replace(".", "p").replace("-", "m")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="Sweep root; results land in <out>/results/<point>")
    ap.add_argument("--axis", action="append", default=[], metavar="DATASET.FIELD=V1,V2",
                    help="Data-config axis to sweep; repeatable. Multiple axes take the product.")
    ap.add_argument("--datasets", nargs="+", default=["doublewell", "vanderpol", "lorenz"],
                    choices=sorted(PRESETS), help="Datasets to fit at every sweep point")
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--steps-override", action="append", default=[], metavar="DATASET=N",
                    help="Per-dataset step budget, e.g. lorenz=3000 (it converges by ~2000)")
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--solver-dt", type=float, default=.01)
    ap.add_argument("--opssm-monitor", default="val_forecast_rmse")
    ap.add_argument("--opssm-gt-diagnostics", action="store_true", default=True)
    ap.add_argument("--no-opssm-gt-diagnostics", dest="opssm_gt_diagnostics", action="store_false")
    # Site settings; every one is also overridable by an env var at submit time.
    ap.add_argument("--partition", default="", help="SLURM partition(s); required by most sites")
    ap.add_argument("--account", default="")
    ap.add_argument("--qos", default="")
    ap.add_argument("--constraint", default="", help='e.g. "a100|h100" to pin a GPU type')
    ap.add_argument("--time", default="24:00:00", help="Wall clock per task")
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--mem-gb", type=int, default=32)
    ap.add_argument("--array-parallelism", type=int, default=32, help="Max concurrent array tasks")
    ap.add_argument("--python", default="python", help="Python in the cluster's JAX benchmark env")
    ap.add_argument("--setup", default="", help="Shell line run on the node before the task (module load ...)")
    args = ap.parse_args(argv)

    axes = [parse_axis(a) for a in args.axis]
    for dataset, field, values in axes:
        if not values:
            raise SystemExit(f"--axis for {dataset}.{field} lists no values")
    step_budget = {d: args.steps for d in args.datasets}
    for item in args.steps_override:
        name, _, count = item.partition("=")
        if name not in step_budget:
            raise SystemExit(f"--steps-override names {name!r}, not in --datasets")
        step_budget[name] = int(count)

    out = args.out.resolve()
    (out / "points").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    # One sweep point per combination of axis values; the baseline point applies no override.
    combos = list(itertools.product(*([(d, f, v) for v in vals] for d, f, vals in axes))) or [()]
    points = {}
    for combo in combos:
        name = "_".join(f"{d}.{f}{label(v)}" for d, f, v in combo) or "baseline"
        overrides = {}
        for dataset, field, value in combo:
            overrides.setdefault(dataset, {})[field] = value
        # Validate now, on the login node, rather than discovering it inside a GPU task.
        for dataset, fields in overrides.items():
            PRESETS[dataset].__class__(**{**PRESETS[dataset].__dict__, **fields}).validate()
        (out / "points" / f"{name}.json").write_text(json.dumps(overrides, indent=2) + "\n")
        points[name] = overrides

    rows = []
    for point in points:
        for dataset in args.datasets:
            for seed in args.seeds:
                for model in args.models:
                    rows.append(dict(point=point, dataset=dataset, seed=seed, model=model,
                                     steps=step_budget[dataset]))
    with (out / "manifest.tsv").open("w") as handle:
        handle.write("idx\tpoint\tdataset\tseed\tmodel\tsteps\n")
        for i, row in enumerate(rows, start=1):
            handle.write(f"{i}\t{row['point']}\t{row['dataset']}\t{row['seed']}\t{row['model']}\t{row['steps']}\n")

    spec = dict(vars(args) | dict(out=str(out)), points=points, total_tasks=len(rows),
                datasets=args.datasets, models=args.models, seeds=args.seeds)
    (out / "sweep.json").write_text(json.dumps(spec, indent=2, default=str) + "\n")

    script = Path(__file__).resolve().parent / "benchmark_array.sbatch"
    env = {"SWEEP_ROOT": str(out), "SWEEP_PYTHON": args.python, "SWEEP_REPO": str(Path(__file__).resolve().parents[2]),
           "SWEEP_VAL_EVERY": str(args.val_every), "SWEEP_SAMPLES": str(args.samples),
           "SWEEP_SOLVER_DT": str(args.solver_dt), "SWEEP_MONITOR": args.opssm_monitor,
           "SWEEP_GT_DIAG": "1" if args.opssm_gt_diagnostics else "0", "SWEEP_SETUP": args.setup}
    directives = [f"--array=1-{len(rows)}%{args.array_parallelism}", f"--time={args.time}",
                  f"--cpus-per-task={args.cpus}", f"--mem={args.mem_gb}G"]
    for flag, value in (("--partition", args.partition), ("--account", args.account),
                        ("--qos", args.qos), ("--constraint", args.constraint)):
        if value:
            directives.append(f"{flag}={value}")
    submit = ["#!/bin/bash", "# Generated by scripts/cluster/make_sweep.py -- edit freely.", "set -euo pipefail", ""]
    submit += [f"export {k}={shlex.quote(v)}" for k, v in env.items()]
    submit += ["", "# Stage 0: generate every dataset once (CPU), so all models share identical data.",
               f"{shlex.quote(args.python)} {shlex.quote(str(Path(__file__).resolve().parent / 'prepare_data.py'))} "
               f"--sweep {shlex.quote(str(out))}", "",
               "# Stage 1: one GPU task per (point, dataset, seed, model).",
               f"sbatch {' '.join(directives)} \\", f"  {shlex.quote(str(script))}", "",
               "# Stage 2, after the array drains:",
               f"#   {shlex.quote(args.python)} scripts/cluster/collect_sweep.py --sweep {shlex.quote(str(out))}"]
    (out / "submit.sh").write_text("\n".join(submit) + "\n")
    (out / "submit.sh").chmod(0o755)

    print(f"sweep root   : {out}")
    print(f"points       : {len(points)}  ({', '.join(points)})")
    print(f"array tasks  : {len(rows)}  = {len(points)} points x {len(args.datasets)} datasets "
          f"x {len(args.seeds)} seeds x {len(args.models)} models")
    if not args.partition:
        print("WARNING: no --partition given; add one to submit.sh before submitting")
    print(f"\nnext: bash {out}/submit.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
