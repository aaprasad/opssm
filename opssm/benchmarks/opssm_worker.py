"""Train the existing OPSSM on the exact benchmark NPZ, in its JAX environment.

This adapter is new; it imports the current model, not the old baseline branch.
Validation selects the checkpoint by a prefix-only forecast score on the validation
observations (see `forecast_from_prefix`), which -- unlike reconstruction R2 --
keeps improving as the drift improves and is available on ground-truth-free data.
Ground truth is excluded from fitting and selection; when the dataset carries true
latents they are logged as diagnostics only (`gt_diagnostics`), never monitored.
"""
import json
import math
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from .data import config_from_data, fingerprint
from .metrics import score_benchmark
from .runner import write_json
from .sde_models import solve_prior


def load_benchmark_checkpoint(directory, hp, checkpoint_name="best"):
    """Restore existing benchmark inference state without running any training."""
    import equinox as eqx
    import optax
    from opssm.models.jax.operator import OperatorFilter
    from opssm.models.jax.dynamics import DriftNet
    from opssm.models.jax.train import _load_ckpt
    d, n = hp["latent_dim"], hp["data_size"]
    op = OperatorFilter(n, hp["gru_hidden"], hp["ctx_dim"], hp["p"], latent_dim=d,
                        key=jr.PRNGKey(0), branch_hidden=hp.get("branch_hidden", 128),
                        trunk_hidden=hp.get("trunk_hidden", 64), trunk_layers=hp.get("trunk_layers", 3),
                        trunk_activation=hp.get("trunk_activation", "softplus"))
    drift_net = DriftNet(hp["drift_hidden"], layers=hp.get("drift_layers", 3), latent_dim=d, key=jr.PRNGKey(1))
    schedule = optax.exponential_decay(hp["lr"], transition_steps=1, decay_rate=hp["sched_gamma"])
    op_state = optax.adam(schedule).init(eqx.filter(op, eqx.is_inexact_array))
    drift_state = optax.adam(hp["drift_lr"]).init(eqx.filter(drift_net.net, eqx.is_inexact_array))
    skeleton = (op, op_state, drift_net, drift_state, jnp.zeros((n, d)), jnp.zeros(n), jnp.eye(d), jnp.ones(n))
    arrays, meta = _load_ckpt(str(directory), skeleton, name=checkpoint_name)
    op, _, drift_net, _, C, offset, L, noise = arrays
    return dict(op=op, drift_net=drift_net, C_cur=C, d_cur=offset, L_cur=L,
                noise_var=noise, g_cur=meta["g_cur"], selected_step=meta["step"])


def posterior_moments(op, obs, C, offset, hp, latent_dim, z_grid, samples, key):
    """Filtering posterior mean/cov for the supplied window only (never sees later frames)."""
    from opssm.models.jax import mstep
    from opssm.models.jax.obs import zhat_from_obs
    mask = jnp.ones((*obs.shape[:2], 1))
    if latent_dim == 1:
        mass = jnp.exp(op.log_posterior(obs, mask, z_grid))
        mean = (mass * z_grid).sum(-1)[..., None]
        cov = (mass * (z_grid - mean) ** 2).sum(-1)[..., None, None]
    else:
        chains, _, _ = mstep._mala_chains(op, obs, mask, zhat_from_obs(obs, C, offset),
            max(samples, 2), hp["mala_steps"], hp["broad_std"], key, chunk_size=hp["chunk_size"])
        mean = chains.mean(2)
        delta = chains - mean[:, :, None]
        cov = jnp.einsum("tbki,tbkj->tbij", delta, delta) / (chains.shape[2] - 1)
    return dict(mean=mean, cov=cov, reconstruction=mean @ C.T + offset)


def forecast_from_prefix(op, drift_net, g_cur, C, offset, L_cur, obs, ts, hp, latent_dim,
                         z_grid, samples, solver_dt, key):
    """Hide the second half of `obs`, then roll the learned prior forward from the prefix filter.

    The single code path behind both validation checkpoint selection and the reported test
    forecast, so the metric that selects a checkpoint is the one it is later scored on.
    """
    from .common import chol
    cut = obs.shape[0] // 2
    k_post, k_z0, k_path = jr.split(key, 3)
    prefix = posterior_moments(op, obs[:cut], C, offset, hp, latent_dim, z_grid, samples, k_post)
    m, cov = prefix["mean"][-1], prefix["cov"][-1]
    z0 = m[None] + jnp.einsum("bij,sbj->sbi", chol(cov),
                              jr.normal(k_z0, (samples, *m.shape)))
    n_sub = max(1, math.ceil(hp["dt"] / solver_dt - 1e-8))
    L = L_cur if hp["diffusion_cov"] else g_cur * jnp.eye(latent_dim)

    def solve(z, k):
        return solve_prior(drift_net.net, lambda z: L, z, ts[cut - 1:], k, solver_dt, n_sub)[1:]

    keys = jr.split(k_path, samples * m.shape[0]).reshape(samples, m.shape[0], 2)
    paths = jnp.transpose(jax.jit(jax.vmap(jax.vmap(solve)))(z0, keys), (2, 0, 1, 3))
    return paths @ C.T + offset, cut


def make_val_forecast_metric(y_val, ts, hp, latent_dim, z_grid, samples, solver_dt):
    """Ground-truth-free validation score: RMSE of the prefix-only forecast on held-out frames.

    Drift quality drives it directly (a bad drift cannot forecast), so unlike recon R2 it does not
    saturate once the autoencoder fits; and it needs no true latents, so Kato uses the same rule.
    """
    def extra_metrics(op, drift_net, g_cur, C_cur, d_cur, noise_var, L_cur, m_op_full, key):
        predictions, cut = forecast_from_prefix(op, drift_net, g_cur, C_cur, d_cur, L_cur,
            y_val, ts, hp, latent_dim, z_grid, samples, solver_dt, key)
        error = y_val[cut:] - predictions.mean(1)
        return {"val_forecast_rmse": float(jnp.sqrt(jnp.mean(error ** 2)))}
    return extra_metrics


def run(job):
    from opssm.models.jax.train import train
    from .opssm_config import resolve_opssm_config
    from opssm.models.jax.systems import make_drift
    with np.load(job["data"], allow_pickle=False) as loaded:
        data = dict(loaded)
    cfg = config_from_data(data)
    if cfg.diffusion_type != "constant":
        raise ValueError("Current OPSSM supports constant diffusion; use the common Lorenz preset for its main comparison")
    out = Path(job["out"])
    system = {"vanderpol": "vanderpol_duncker", "kato": "none"}.get(cfg.system, cfg.system)
    params = {"a": cfg.a} if cfg.system == "doublewell" else ({"tau": cfg.tau, "mu": cfg.mu} if cfg.system == "vanderpol" else {})
    ts = jnp.asarray(data["ts"], dtype=jnp.float32)
    y = {s: jnp.asarray(data[f"y_{s}"], dtype=jnp.float32) for s in ("train", "val", "test")}
    refs = dict(x_train=y["train"], mask_train=jnp.ones((*y["train"].shape[:2], 1)),
                x_val=y["val"], mask_val=jnp.ones((*y["val"].shape[:2], 1)), full_obs=y["train"],
                ts=ts, system=system, true_drift=make_drift(system, **params)[0], sigma=getattr(cfg, "diffusion", .1))
    if cfg.system == "kato":
        refs["true_drift"] = None
    # Grid used only for OPSSM's 1D readout/diagnostics, not to generate supervision.
    refs["z_grid"] = jnp.linspace(-8., 8., 512)
    overrides = json.loads(Path(job["config"]).read_text()) if job["config"] else None
    hp, config_provenance = resolve_opssm_config(cfg.system, experiment=job.get("experiment"),
        config_dir=job.get("config_dir"), overrides=overrides)
    if job["fixed_obs_noise"]:
        hp["learn_obs_noise"] = False
    if job["smoke"]:
        hp.update(gru_hidden=8, ctx_dim=8, p=8, branch_hidden=8, trunk_hidden=8, trunk_layers=2,
                  drift_hidden=8, drift_layers=2, n_colloc=8, n_scoll=2, n_tcoll=4,
                  n_mean=8, mala_chains=8, mala_steps=4, m_every=1, m_inner=2, bootstrap_mstep=False)
    # Dimensions/timing must come from the shared artifact even when tuning the model.
    hp.update(system=system, latent_dim=cfg.latent_dim, data_size=cfg.obs_dim,
              dt=float(data["ts"][1] - data["ts"][0]), noise_std=float(data["noise_std_eff"]))
    # Checkpoint selection. Default: prefix-only validation forecast RMSE (no ground truth, works on Kato).
    monitor = job.get("monitor") or "val_forecast_rmse"
    monitor_mode = job.get("monitor_mode") or ("max" if monitor in ("recon_r2", "c_cos") else "min")
    select_samples = 8 if job["smoke"] else job.get("select_samples", 16)
    extra_metrics = (make_val_forecast_metric(y["val"], ts, hp, cfg.latent_dim, refs["z_grid"],
                                              select_samples, job["solver_dt"])
                     if monitor == "val_forecast_rmse" else None)
    # True latents, when the dataset has them, are logged for diagnosis only -- never monitored.
    if job.get("gt_diagnostics") and "z_val" in data and cfg.system != "kato":
        refs["z_val_true"] = jnp.asarray(data["z_val"], dtype=jnp.float32)
        refs["C_true"] = jnp.asarray(data["C_true"], dtype=jnp.float32)
        if monitor in ("drift_rel", "lat_rel", "drift_l2"):
            raise ValueError(f"monitor={monitor} selects on ground truth; that is oracle selection, "
                             "not the documented protocol -- pass it deliberately via --opssm-monitor")
    write_json(out / "hparams.json", hp)
    config_provenance.update(selection=dict(monitor=monitor, mode=monitor_mode,
        samples=select_samples, ground_truth_free=monitor == "val_forecast_rmse",
        gt_diagnostics_logged=bool(refs.get("z_val_true") is not None)))
    write_json(out / "config_provenance.json", config_provenance)
    # Resume a preempted run from train()'s rolling ckpt.*: SLURM requeues the task, and without
    # this the cell restarts at step 0. Only ever resume a checkpoint this same configuration
    # wrote -- otherwise a rerun against a changed config would silently continue a stale fit.
    signature = dict(config_sha256=config_provenance["resolved_model_sha256"],
                     dataset_hash=fingerprint(data), steps=job["steps"], seed=job["seed"],
                     monitor=monitor, monitor_mode=monitor_mode, smoke=bool(job["smoke"]))
    signature_path = out / "run_signature.json"
    resume = False
    if signature_path.exists() and (out / "ckpt.eqx").exists():
        if json.loads(signature_path.read_text()) == signature:
            resume = True
        else:
            raise ValueError(f"{out} holds a checkpoint from a different configuration; "
                             "use a new results directory rather than resuming a stale fit")
    write_json(signature_path, signature)
    started = time.perf_counter()
    state, _ = train(refs, hp, n_steps=job["steps"], key=jr.PRNGKey(job["seed"]),
                     val_every=1 if job["smoke"] else job.get("val_every", 100), ckpt_dir=str(out),
                     resume=resume, ckpt_every=job.get("ckpt_every", 500),
                     monitor=monitor, monitor_mode=monitor_mode, return_best=True, timing=True,
                     extra_metrics=extra_metrics)
    fit_s = time.perf_counter() - started

    def posterior(obs, key):
        return posterior_moments(state["op"], obs, state["C_cur"], state["d_cur"], hp,
                                 cfg.latent_dim, refs["z_grid"], job["samples"], key)

    started = time.perf_counter()
    val = jax.tree.map(np.asarray, posterior(y["val"], jr.PRNGKey(job["seed"] + 20000)))
    test = jax.tree.map(np.asarray, posterior(y["test"], jr.PRNGKey(job["seed"] + 30000)))
    samples = job["samples"]
    predictions, cut = forecast_from_prefix(state["op"], state["drift_net"], state["g_cur"],
        state["C_cur"], state["d_cur"], state["L_cur"], y["test"], ts, hp, cfg.latent_dim,
        refs["z_grid"], samples, job["solver_dt"], jr.PRNGKey(job["seed"] + 40000))
    variance = state["noise_var"]
    lp = -.5 * (((y["test"][cut:, None] - predictions) ** 2 / variance) + jnp.log(2 * jnp.pi * variance)).sum(-1)
    ahead = dict(forecast_mean=np.asarray(predictions.mean(1)),
                 forecast_loglik=np.asarray(jax.scipy.special.logsumexp(lp, axis=1) - jnp.log(samples)))
    if any(not np.isfinite(v).all() for v in [*val.values(), *test.values(), *ahead.values()]):
        raise FloatingPointError("Nonfinite OPSSM predictions")
    train_mean = (np.asarray(posterior(y["train"], jr.PRNGKey(job["seed"] + 25000))["mean"])
                  if "states_train" in data else None)
    metrics, evaluation_arrays = score_benchmark(val, test, ahead, data, cfg,
        lambda z: state["drift_net"].net(jnp.asarray(z, dtype=jnp.float32)), train_mean=train_mean)
    metrics.update(fit_s=fit_s, eval_s=time.perf_counter() - started, posterior_type="filter",
                   best_step=state["selected_step"], backend="jax", precision="float32",
                   opssm_experiment=config_provenance["experiment"],
                   opssm_config_sha256=config_provenance["resolved_model_sha256"],
                   diffusion_cov=hp["diffusion_cov"], n_colloc=hp["n_colloc"], chunk_size=hp["chunk_size"],
                   opssm_monitor=monitor, opssm_monitor_mode=monitor_mode)
    np.savez_compressed(out / "predictions.npz", **test, **ahead, **evaluation_arrays,
                        forecast_cutoff=np.array(cut), dataset_hash=np.array(fingerprint(data)))
    write_json(out / "worker_metrics.json", metrics)


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text()))
