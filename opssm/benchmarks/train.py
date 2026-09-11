"""Shared JAX fitting, validation selection, timing, and prefix-only forecasting."""
import importlib.metadata
import json
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from .common import decode, observation_nll
from .data import fingerprint
from .metrics import score_benchmark


def forecast(model, theta, y, ts, key, samples):
    """Only the first half is supplied to inference. Future y enters scores only."""
    cut = len(ts) // 2
    kinfer, kprior = jr.split(key)
    post = model.posterior(theta, y[:cut], ts[:cut], kinfer, samples)
    paths = model.forecast_samples(theta, post, ts[cut - 1:], kprior, samples)
    predictions = decode(theta, paths)
    lp = -observation_nll(theta, y[cut:, None], paths, model.noise)
    return dict(forecast_mean=predictions.mean(1),
                forecast_loglik=jax.scipy.special.logsumexp(lp, axis=1) - jnp.log(samples))


def save_fit_checkpoint(directory, arrays, meta):
    """Atomically persist the in-progress fit, so a preemption mid-write cannot corrupt it."""
    arrays_path, meta_path = directory / "fit_ckpt.eqx", directory / "fit_ckpt.json"
    temp = arrays_path.with_name(arrays_path.name + ".tmp")
    eqx.tree_serialise_leaves(temp, arrays)
    os.replace(temp, arrays_path)
    temp = meta_path.with_name(meta_path.name + ".tmp")
    temp.write_text(json.dumps(meta))
    os.replace(temp, meta_path)


def load_fit_checkpoint(directory, skeleton, signature):
    """Restore a preempted fit, but only one produced by this exact configuration.

    Resuming across a changed model, dataset or budget would silently mix two runs, so a
    mismatched signature is ignored and the fit restarts rather than producing a hybrid.
    """
    arrays_path, meta_path = directory / "fit_ckpt.eqx", directory / "fit_ckpt.json"
    if not (arrays_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None
    if meta.get("signature") != signature:
        print(f"ignoring {arrays_path}: written by a different configuration", flush=True)
        return None
    return eqx.tree_deserialise_leaves(arrays_path, skeleton), meta


def fit_jax(name, data, cfg, args, directory, seed):
    from .dynamax_models import DynamaxBaseline
    from .sde_models import SDEBaseline
    from .runner import write_json
    # Avoid cancellation in low-noise Kalman updates and matching second derivatives.
    jax.config.update("jax_enable_x64", True)
    y = {s: jnp.asarray(data[f"y_{s}"], dtype=jnp.float64) for s in ("train", "val", "test")}
    ts = jnp.asarray(data["ts"])
    dt = float(data["ts"][1] - data["ts"][0])
    if not np.allclose(np.diff(data["ts"]), dt):
        raise ValueError("This comparison currently requires a regular observation grid")
    noise = float(data["noise_std_eff"])
    if name in ("latent_sde", "sde_matching"):
        model = SDEBaseline(name, cfg.latent_dim, noise, args.hidden, args.solver_dt,
                            args.fixed_obs_noise, cfg.diffusion_type != "constant", observation_dt=dt)
    else:
        model = DynamaxBaseline(name, cfg.latent_dim, dt, noise, args.hidden, args.modes, args.fixed_obs_noise)
    theta = model.initialize(y["train"], jr.PRNGKey(seed))
    optim = optax.chain(optax.clip_by_global_norm(10.), optax.adam(args.lr))
    state = optim.init(eqx.filter(theta, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(theta, state, batch, key):
        loss, grad = eqx.filter_value_and_grad(model.loss)(theta, batch, ts, key)
        grad_finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grad)]))
        updates, state = optim.update(grad, state, theta)
        return eqx.apply_updates(theta, updates), state, loss, grad_finite

    val_loss = eqx.filter_jit(lambda th: model.loss(th, y["val"], ts, jr.PRNGKey(seed + 10000)))
    history, best, best_step, best_theta = [], float("inf"), 0, theta
    key = jr.PRNGKey(seed + 1)
    start_iteration, elapsed_before = 0, 0.
    # Resume a preempted fit: SLURM requeues the task, and without this the cell restarts at step 0.
    signature = dict(model=name, seed=seed, steps=args.steps, lr=args.lr, hidden=args.hidden,
                     modes=args.modes, batch_size=args.batch_size, val_every=args.val_every,
                     solver_dt=args.solver_dt, fixed_obs_noise=args.fixed_obs_noise,
                     dataset_hash=fingerprint(data))
    restored = load_fit_checkpoint(directory, (theta, state, theta), signature)
    if restored is not None:
        (theta, state, best_theta), meta = restored
        history, best, best_step = meta["history"], meta["best"], meta["best_step"]
        start_iteration, elapsed_before = meta["iteration"], meta["elapsed_s"]
        key = jnp.asarray(meta["key"], dtype=jnp.uint32)
        print(f"{cfg.name}/{name} seed={seed} resuming at step {start_iteration}", flush=True)
    started = time.perf_counter()
    first_step_s = None
    for iteration in range(start_iteration + 1, args.steps + 1):
        key, kb, kl = jr.split(key, 3)
        idx = jr.permutation(kb, y["train"].shape[1])[:args.batch_size]
        before = time.perf_counter()
        theta, state, loss, finite = step(theta, state, y["train"][:, idx], kl)
        loss = float(loss)  # synchronizes accelerator execution before timing
        if first_step_s is None:
            first_step_s = time.perf_counter() - before
        if not np.isfinite(loss) or not bool(finite):
            raise FloatingPointError(f"Nonfinite training loss/gradient at step {iteration}")
        if iteration == 1 or iteration % args.val_every == 0 or iteration == args.steps:
            value = float(val_loss(theta))
            if not np.isfinite(value):
                raise FloatingPointError(f"Nonfinite validation loss at step {iteration}")
            history.append(dict(step=iteration, train_loss=loss, val_loss=value))
            if value < best:
                best, best_step, best_theta = value, iteration, theta
            print(f"{cfg.name}/{name} seed={seed} step={iteration} val={value:.5g}", flush=True)
            save_fit_checkpoint(directory, (theta, state, best_theta),
                dict(signature=signature, iteration=iteration, best=best, best_step=best_step,
                     history=history, key=[int(v) for v in jnp.asarray(key).ravel()],
                     elapsed_s=elapsed_before + time.perf_counter() - started))
    fit_s = elapsed_before + time.perf_counter() - started
    theta = best_theta
    eqx.tree_serialise_leaves(directory / "best.eqx", theta)
    write_json(directory / "history.json", history)
    write_json(directory / "checkpoint.json", dict(model=name, best_step=best_step, seed=seed,
               dataset_hash=fingerprint(data), hidden=args.hidden, modes=args.modes, solver_dt=args.solver_dt,
               fixed_obs_noise=args.fixed_obs_noise, precision="float64"))

    @eqx.filter_jit
    def infer(theta, obs, key):
        out = model.posterior(theta, obs, ts, key, args.samples)
        return {**out, "reconstruction": decode(theta, out["mean"])}

    ahead_fn = eqx.filter_jit(lambda th, obs, key: forecast(model, th, obs, ts, key, args.samples))
    started = time.perf_counter()
    val = jax.tree.map(np.asarray, infer(theta, y["val"], jr.PRNGKey(seed + 20000)))
    test = jax.tree.map(np.asarray, infer(theta, y["test"], jr.PRNGKey(seed + 30000)))
    ahead = jax.tree.map(np.asarray, ahead_fn(theta, y["test"], jr.PRNGKey(seed + 40000)))
    if any(not np.isfinite(v).all() for v in [*val.values(), *test.values(), *ahead.values()]):
        raise FloatingPointError("Nonfinite inference or forecast")
    train_mean = (np.asarray(infer(theta, y["train"], jr.PRNGKey(seed + 25000))["mean"])
                  if "states_train" in data else None)
    metrics, evaluation_arrays = score_benchmark(val, test, ahead, data, cfg,
        lambda z: model.drift_values(theta, jnp.asarray(z), test.get("regime_prob")),
        model.dynamics_kind, train_mean=train_mean)
    eval_s = time.perf_counter() - started
    cut = len(ts) // 2
    metrics.update(fit_s=fit_s, eval_s=eval_s, first_train_step_s=first_step_s,
                   best_step=best_step, val_objective=best,
                   parameters=sum(x.size for x in jax.tree.leaves(eqx.filter(theta, eqx.is_inexact_array))),
                   posterior_type="smoother" if isinstance(model, SDEBaseline) else "filter",
                   backend="jax", devices=[str(device) for device in jax.devices()], precision="float64",
                   library_versions={p: importlib.metadata.version(p) for p in ("jax", "dynamax", "diffrax", "equinox", "optax")})
    if "loglik" in test:
        metrics["filter_predictive_nll"] = float(-test["loglik"].mean() / (len(ts) * cfg.obs_dim))
    np.savez_compressed(directory / "predictions.npz", **test, **ahead, **evaluation_arrays,
                        forecast_cutoff=np.array(cut), dataset_hash=np.array(fingerprint(data)))
    return metrics
