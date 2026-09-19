"""gpSLDS baseline: SING-GP with the smoothly-switching-linear kernel, on the benchmark NPZs.

Why this model earns a slot next to rSLDS. The rSLDS in the sweep is `modes` affine pieces with a
logistic partition, and -- because it is a DISCRETE-TIME map -- it is scored through
`observation_interval_effective_drift`, a one-interval finite difference that carries an O(dt)
handicap the continuous-drift models do not pay. The gpSLDS keeps the locally-linear idea but puts
a GP prior on the drift that interpolates smoothly between regimes, and it is a genuine
continuous-time SDE, so it scores as `continuous_drift` -- the like-for-like comparison.

Implementation is SING (Hu, Smith & Linderman 2025) from `src/sing`, not the original `src/gpslds`
repo: that repo's own README names SING the recommended inference algorithm for gpSLDS models as of
12/2025, and SING's `SSL` kernel IS the gpSLDS prior. (`src/gpslds` also imports wandb.)

Protocol, matched to the other models as far as SING allows:
  - variational EM (inference + learning) on the TRAIN trials;
  - inference ONLY on val and test with the kernel and output params frozen, because SING's
    posterior is per-trial rather than amortized -- there is no encoder to apply to new trials;
  - forecast by rolling the GP posterior-mean drift forward from the prefix posterior, same
    half-trace cutoff the other models use.

NOT checkpointed for preemption: SING's `fit_variational_em` does not expose per-iteration state,
so a requeued cell restarts from scratch. vEM is ~50 iterations, so this is cheap relative to the
10k-step gradient fits; revisit if the wall clock grows.
"""
import importlib.metadata
import math
import sys
import time
from pathlib import Path

import numpy as np

# src/sing is a VENDORED CLONE and src/ is gitignored, so it is not in a fresh checkout. Point the
# failure at the fix rather than letting it surface as a bare "No module named 'sing'" on a node.
_SING = Path(__file__).resolve().parents[2] / "src" / "sing"
if str(_SING) not in sys.path:
    sys.path.insert(0, str(_SING))


def _require_sing():
    try:
        import sing  # noqa: F401
    except ImportError as error:
        raise ImportError(
            f"method=gpslds needs the SING sources at {_SING}, which is gitignored (src/) and so "
            "absent from a fresh clone. Fix with:\n"
            f"    git clone https://github.com/lindermanlab/sing {_SING}\n"
            "It is imported from that path, not installed, but its dependencies "
            "(tensorflow_probability, optax, flax) must be in the environment running this sweep."
        ) from error

# Inducing points per latent axis: the GP cost is O(M^3) in the TOTAL count, so the per-axis count
# has to fall as d grows -- 1-D 16, 2-D 8x8=64, 3-D 6^3=216.
INDUCING_PER_AXIS = {1: 16, 2: 8, 3: 4}
# SING holds the variational posterior for every trial at once and runs quadrature at every
# (trial, timestep, inducing point). Trial counts are NOT comparable across these presets --
# doublewell/vanderpol have 20 training trials, lorenz has 1024 -- so full-batch vEM OOMs on
# lorenz (35 GiB). Cap the SVI minibatch, and chunk val/test inference, at this many trials.
MAX_TRIALS_PER_PASS = {1: 64, 2: 64, 3: 16}


def _version(package):
    """Best-effort version string. tfp is often installed without standard dist metadata, and
    `sing` is vendored rather than installed -- neither is worth failing a finished fit over."""
    try:
        return importlib.metadata.version(package)
    except Exception:
        try:
            return getattr(__import__(package), "__version__", "unknown")
        except Exception:
            return "unknown"


def _build_prior(key, latent_dim, num_states, axis_lims, n_per_axis, tau_init):
    """gpSLDS prior: SSL kernel over a grid of inducing points. Follows initialize_gpslds_prior in
    SING's neural-data demo, generalised past its hard-coded 2-D basis set."""
    import jax.numpy as jnp
    import jax.random as jr
    import tensorflow_probability.substrates.jax as tfp
    from sing.kernels import SSL, FullLinear
    from sing.sde import SparseGP
    tfd = tfp.distributions

    basis_set = lambda x: jnp.concatenate([jnp.ones(1), x])      # linear decision boundaries
    kernel = SSL(FullLinear(latent_dim), basis_set, latent_dim)
    axes = [jnp.linspace(lo, hi, n_per_axis) for lo, hi in axis_lims]
    mesh = jnp.meshgrid(*axes, indexing="ij")
    zs = jnp.stack([m.ravel() for m in mesh], axis=1)
    key_W, key_fp = jr.split(key, 2)
    kernel_params = {
        "linear_params": [{"fixed_point": fp, "log_M": jnp.zeros(latent_dim), "log_noise_var": 0.}
                          for fp in tfd.Normal(0, 1).sample((num_states, latent_dim),
                                                            seed=key_fp).astype(jnp.float64)],
        "W": tfd.Normal(0, 1).sample((1 + latent_dim, num_states - 1), seed=key_W).astype(jnp.float64),
        "log_tau": jnp.log(tau_init),
    }
    return SparseGP(zs, kernel), kernel_params, zs


def _rho_schedule(n_iters):
    import jax.numpy as jnp
    ramp = jnp.logspace(-1, 0, min(10, n_iters))                  # ramp the natural-gradient step
    return jnp.concatenate([ramp, ramp[-1] * jnp.ones(max(0, n_iters - len(ramp)))])


def _infer(fn, kernel_params, output_params, t_grid, ys, sigma, n_iters, n_iters_e, key, chunk):
    """Inference only: drift and output params frozen, SING run on unseen trials.

    Chunked over trials and FULL-BATCH within each chunk. Minibatching here would be wrong, not
    just slow: SING only updates the trials it samples, so with 128 test trials and a batch of 64
    roughly half would be scored at their PCA initialization.
    """
    import jax.numpy as jnp
    import jax.random as jr
    from sing.initialization import initialize_params_pca
    from sing.likelihoods import Gaussian
    from sing.sing import fit_variational_em
    latent_dim = output_params["C"].shape[1]
    out = []
    for start in range(0, ys.shape[0], chunk):
        part = ys[start:start + chunk]
        _, x0 = initialize_params_pca(latent_dim, part)
        init_params = {"mu0": x0, "V0": jnp.eye(latent_dim)[None].repeat(part.shape[0], 0)}
        marginal = fit_variational_em(
            jr.fold_in(key, start), fn, Gaussian(part, jnp.ones(part.shape[:2])), t_grid,
            kernel_params, init_params, output_params, batch_size=None,
            rho_sched=_rho_schedule(n_iters), sigma=sigma, n_iters=n_iters, n_iters_e=n_iters_e,
            perform_m_step=False, learn_output_params=False, print_interval=10 ** 9)[0]
        out.append(marginal)
    return {k: jnp.concatenate([m[k] for m in out], axis=0) for k in ("m", "S")}


def _forecast(fn, gp_post, kernel_params, prefix, output_params, dt, solver_dt, sigma,
              samples, latent_dim, key):
    """Roll the GP posterior-mean drift forward from the prefix posterior (Euler-Maruyama).

    Mirrors the other models' half-trace protocol: filter the first half, then integrate the
    learned prior forward over the second half with no further observations.
    """
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    n_steps = prefix["n_ahead"]
    m, S = prefix["m"], prefix["S"]                               # (trials,d), (trials,d,d)
    chol = jnp.linalg.cholesky(S + 1e-8 * jnp.eye(latent_dim))
    k0, kp = jr.split(key)
    z = m[None] + jnp.einsum("bij,sbj->sbi", chol, jr.normal(k0, (samples, *m.shape)))
    n_sub = max(1, math.ceil(dt / solver_dt - 1e-8))
    h = dt / n_sub
    drift = lambda x: fn.get_posterior_f_mean(gp_post, kernel_params, x)

    def step(carry, key_t):
        z = carry
        for k in jr.split(key_t, n_sub):
            flat = z.reshape(-1, latent_dim)
            f = drift(flat).reshape(z.shape)
            z = z + f * h + sigma * math.sqrt(h) * jr.normal(k, z.shape)
        return z, z

    _, path = jax.lax.scan(step, z, jr.split(kp, n_steps))        # (n_ahead,samples,trials,d)
    return path @ output_params["C"].T + output_params["d"]


def fit_gpslds(name, data, cfg, args, directory, seed):
    """Fit and score one gpSLDS cell. Same contract as train.fit_jax: returns the metrics dict and
    writes predictions.npz / checkpoint.json / history.json into `directory`."""
    _require_sing()
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax.random as jr
    from sing.initialization import initialize_params_pca
    from sing.likelihoods import Gaussian
    from sing.sing import fit_variational_em

    from .data import fingerprint
    from .metrics import score_benchmark
    from .runner import write_json

    get = lambda k, default: getattr(args, k, None) if getattr(args, k, None) is not None else default
    num_states = get("gpslds_states", args.modes)
    sigma = float(get("gpslds_sigma", 1.0))
    n_iters = int(get("gpslds_iters", 50))
    n_iters_e = int(get("gpslds_iters_e", 15))
    n_iters_m = int(get("gpslds_iters_m", 50))
    n_iters_infer = int(get("gpslds_iters_infer", 15))
    lr = float(get("gpslds_lr", 1e-4))
    tau_init = float(get("gpslds_tau", 0.5))
    pad_frac = float(get("gpslds_inducing_pad", 0.25))
    batch_size = get("gpslds_batch_size", None)

    d = cfg.latent_dim
    n_axis = int(get("gpslds_inducing_per_axis", INDUCING_PER_AXIS.get(d, 4)))
    cap = MAX_TRIALS_PER_PASS.get(d, 16)
    chunk = int(get("gpslds_infer_chunk", cap))
    ts = jnp.asarray(data["ts"], dtype=jnp.float64)
    dt = float(data["ts"][1] - data["ts"][0])
    if not np.allclose(np.diff(data["ts"]), dt):
        raise ValueError("gpSLDS currently requires a regular observation grid")
    # benchmark layout is (T, trials, dims); SING wants (trials, T, dims)
    y = {s: jnp.asarray(np.transpose(data[f"y_{s}"], (1, 0, 2)), dtype=jnp.float64)
         for s in ("train", "val", "test")}

    output_params_init, x0_init = initialize_params_pca(d, y["train"])
    # PCA initializes R from the reconstruction residual, which is exactly ZERO whenever the
    # sensor is square and full rank -- lorenz is `sensor=identity` with D=d=3, giving R~7e-31 and
    # a likelihood that divides by it, so the first E-step returns NaN. Floor R at the observation
    # noise the benchmark already hands every other baseline (`noise_std_eff`); the M-step learns
    # it from there, so this only fixes the initialization.
    noise_var = float(data["noise_std_eff"]) ** 2
    output_params_init = {**output_params_init,
                          "R": jnp.maximum(output_params_init["R"], noise_var)}
    scores = (jnp.vstack(y["train"]) - output_params_init["d"]) @ output_params_init["C"]
    lo, hi = np.asarray(scores.min(0)), np.asarray(scores.max(0))
    pad = pad_frac * np.maximum(hi - lo, 1e-6)
    axis_lims = list(zip(lo - pad, hi + pad))      # inducing grid spans the data, with a margin
    fn, kernel_params_init, zs = _build_prior(jr.PRNGKey(seed + 20), d, num_states,
                                              axis_lims, n_axis, tau_init)

    n_trials = int(y["train"].shape[0])
    batch_size = int(batch_size) if batch_size else min(n_trials, cap)
    started = time.perf_counter()
    results = fit_variational_em(
        jr.PRNGKey(seed), fn, Gaussian(y["train"], jnp.ones(y["train"].shape[:2])), ts,
        kernel_params_init, {"mu0": x0_init, "V0": jnp.eye(d)[None].repeat(y["train"].shape[0], 0)},
        output_params_init, batch_size=batch_size, rho_sched=_rho_schedule(n_iters), sigma=sigma,
        n_iters=n_iters, n_iters_e=n_iters_e, perform_m_step=True, n_iters_m=n_iters_m,
        learning_rate=lr * jnp.ones(n_iters), print_interval=max(1, n_iters // 10))
    _, _, gp_post, kernel_params, _, output_params, _, elbos = results
    fit_s = time.perf_counter() - started

    started = time.perf_counter()
    marg = {s: _infer(fn, kernel_params, output_params, ts, y[s], sigma, n_iters_infer,
                      n_iters_e, jr.PRNGKey(seed + off), chunk)
            for s, off in (("val", 20000), ("test", 30000))}

    C, off = np.asarray(output_params["C"]), np.asarray(output_params["d"])

    def pack(m):                                   # (trials,T,.) -> benchmark (T,trials,.)
        mean = np.transpose(np.asarray(m["m"]), (1, 0, 2))
        cov = np.transpose(np.asarray(m["S"]), (1, 0, 2, 3))
        return dict(mean=mean, cov=cov, reconstruction=mean @ C.T + off)

    val, test = pack(marg["val"]), pack(marg["test"])
    cut = len(data["ts"]) // 2
    prefix = {"m": marg["test"]["m"][:, cut - 1], "S": marg["test"]["S"][:, cut - 1],
              "n_ahead": len(data["ts"]) - cut}
    predictions = _forecast(fn, gp_post, kernel_params, prefix, output_params, dt, args.solver_dt,
                            sigma, args.samples, d, jr.PRNGKey(seed + 40000))
    predictions = jnp.transpose(predictions, (0, 1, 2, 3))        # (n_ahead,samples,trials,D)
    R = jnp.asarray(output_params["R"])
    y_obs = y["test"].transpose(1, 0, 2)[cut:]                    # (n_ahead,trials,D)
    lp = -.5 * (((y_obs[:, None] - predictions) ** 2 / R) + jnp.log(2 * jnp.pi * R)).sum(-1)
    ahead = dict(forecast_mean=np.asarray(predictions.mean(1)),
                 forecast_loglik=np.asarray(jax.scipy.special.logsumexp(lp, axis=1)
                                            - jnp.log(args.samples)))
    if any(not np.isfinite(v).all() for v in [*val.values(), *test.values(), *ahead.values()]):
        raise FloatingPointError("Nonfinite gpSLDS inference or forecast")

    # gpSLDS drift is continuous-time -> scored exactly like OPSSM and sde_matching.
    learned = lambda z: np.asarray(fn.get_posterior_f_mean(
        gp_post, kernel_params, jnp.asarray(np.asarray(z, dtype=float).reshape(-1, d)))
    ).reshape(np.shape(z))
    metrics, arrays = score_benchmark(val, test, ahead, data, cfg, learned, "continuous_drift")
    metrics.update(fit_s=fit_s, eval_s=time.perf_counter() - started, modes=num_states,
                   posterior_type="smoother", backend="jax", precision="float64",
                   steps=n_iters,   # override runner's --steps: gpSLDS never reads it
                   gpslds_sigma=sigma, gpslds_iters=n_iters, gpslds_n_inducing=int(zs.shape[0]),
                   gpslds_batch_size=batch_size, gpslds_n_train_trials=n_trials,
                   gpslds_epochs=round(n_iters * batch_size / max(n_trials, 1), 2),
                   gpslds_implementation="sing.SparseGP+SSL",
                   val_objective=float(elbos[-1]) if len(elbos) else None,
                   devices=[str(device) for device in jax.devices()],
                   library_versions={p: _version(p) for p in ("jax", "optax",
                                                                "tensorflow_probability", "sing")})
    write_json(directory / "history.json", [dict(iteration=i, elbo=float(e))
                                            for i, e in enumerate(elbos)])
    write_json(directory / "checkpoint.json", dict(model=name, seed=seed, modes=num_states,
               dataset_hash=fingerprint(data), sigma=sigma, n_inducing=int(zs.shape[0]),
               n_iters=n_iters, solver_dt=args.solver_dt, precision="float64"))
    np.savez_compressed(directory / "predictions.npz", **test, **ahead, **arrays,
                        inducing_points=np.asarray(zs), C=C, d=off,
                        forecast_cutoff=np.array(cut), dataset_hash=np.array(fingerprint(data)))
    return metrics
