"""Add aligned dynamics RMSE to saved benchmark runs, without retraining.

    python -m opssm.benchmarks.rescore --out dump/benchmark_smoke_verified
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from .data import DatasetConfig, drift, fingerprint
from .metrics import aggregate, score_dynamics
from .runner import write_json


def rescore(directory):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from .dynamax_models import DynamaxBaseline
    from .sde_models import SDEBaseline

    result_path = directory / "result.json"
    result = json.loads(result_path.read_text())
    if result["status"] != "ok":
        return
    with np.load(directory.parent / "data.npz", allow_pickle=False) as loaded:
        data = dict(loaded)
    with np.load(directory / "predictions.npz", allow_pickle=False) as loaded:
        predictions = dict(loaded)
    digest = fingerprint(data)
    if result["dataset_hash"] != digest or str(predictions["dataset_hash"]) != digest:
        raise ValueError(f"Dataset/prediction fingerprint mismatch in {directory}")
    if "z_test" not in data:
        print(f"{directory}: dynamics RMSE unavailable (no ground truth)", flush=True)
        return
    cfg = DatasetConfig(**json.loads(str(data["metadata"]))["config"])
    started = time.perf_counter()
    if result["model"] == "opssm":
        from .opssm_worker import load_benchmark_checkpoint
        jax.config.update("jax_enable_x64", False)
        hp = json.loads((directory / "hparams.json").read_text())
        state = load_benchmark_checkpoint(directory, hp)
        learned = lambda z: state["drift_net"].net(jnp.asarray(z, dtype=jnp.float32))
        kind = "continuous_drift"
    else:
        jax.config.update("jax_enable_x64", True)
        meta = json.loads((directory / "checkpoint.json").read_text())
        if meta["dataset_hash"] != digest:
            raise ValueError(f"Checkpoint fingerprint mismatch in {directory}")
        dt, noise = float(np.diff(data["ts"])[0]), float(data["noise_std_eff"])
        if result["model"] in ("latent_sde", "sde_matching"):
            model = SDEBaseline(result["model"], cfg.latent_dim, noise, meta["hidden"], meta["solver_dt"],
                                meta["fixed_obs_noise"], cfg.diffusion_type != "constant", observation_dt=dt)
        else:
            model = DynamaxBaseline(result["model"], cfg.latent_dim, dt, noise, meta["hidden"],
                                    meta["modes"], meta["fixed_obs_noise"])
        template = model.initialize(jnp.asarray(data["y_train"], dtype=jnp.float64), jax.random.PRNGKey(meta["seed"]))
        theta = eqx.tree_deserialise_leaves(directory / "best.eqx", template)
        learned = lambda z: model.drift_values(theta, jnp.asarray(z), predictions.get("regime_prob"))
        kind = model.dynamics_kind
    metrics, arrays = score_dynamics(learned, lambda z: drift(z, cfg), data, predictions["alignment"], kind)
    result.update(metrics, dynamics_eval_s=time.perf_counter() - started)
    predictions.update(arrays)
    temp = directory / "predictions.tmp.npz"
    np.savez_compressed(temp, **predictions)
    temp.replace(directory / "predictions.npz")
    write_json(result_path, result)
    if (directory / "worker_metrics.json").exists():
        worker = json.loads((directory / "worker_metrics.json").read_text())
        worker.update(metrics)
        write_json(directory / "worker_metrics.json", worker)
    print(f"{result['dataset']}/{result['model']} seed={result['seed']} dynamics_rmse={result['dynamics_rmse']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    files = sorted(args.out.glob("*/seed_*/*/result.json"))
    if not files:
        parser.error(f"No benchmark runs found in {args.out}")
    for path in files:
        rescore(path.parent)
    rows = [json.loads(path.read_text()) for path in files]
    write_json(args.out / "summary.json", aggregate(rows))
    with (args.out / "metrics.csv").open("w", newline="") as f:
        fields = sorted(set().union(*(row.keys() for row in rows)) - {"settings"})
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
