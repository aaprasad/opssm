"""Torch-free in-memory synthetic data for the JAX backend.

The .[jax] venv has no torch, so it cannot instantiate the (torch/torchsde/Lightning) DoubleWellDataModule.
This is a numpy re-implementation of that datamodule's LINEAR-SENSOR branches (multi-D VdP/Lorenz, and the
1-D high-D double-well): Euler-Maruyama the latent SDE + a high-D linear sensor + the signal-scale obs
standardization, returning the SAME array dict the bridge npz holds (consumed by
opssm.models.jax.train._refs_from_mapping). So `scripts/train_jax.py` can generate synthetic data in memory
(no npz on disk), exactly as `opssm.data.kato.refs.build_kato_data` does for Kato.

The d==1 exact grid oracle (_grid_filter_highd) IS reproduced here in numpy (validated == the torch oracle
to ~1e-8), so d==1 high-D systems (em_highd) get filt_val -> the exact-filter `kl` metric works in memory
(~0.2s one-time, O(T*B*Nz^2)). d>=2 (VdP/Lorenz) has NO grid oracle (O(N^d) intractable) -> no kl, ever
(as in the torch datamodule). Direct-obs 1-D (doublewell, highd=false) is not supported here (no sensor) --
use the npz bridge. RNG is numpy (differs from the torch bridge), so generated data is a fresh valid draw
from the same generative model, not a byte match to any .npz.
"""
import math

import numpy as np

_DIM = {"doublewell": 1, "vanderpol": 2, "vanderpol_duncker": 2, "lorenz": 3}


def _drift(name, a):
    """Numpy drifts -- MUST match opssm.models.jax.systems / opssm.data.systems (same registered defaults)."""
    if name == "doublewell":
        return lambda z: a * (z - z ** 3)
    if name == "vanderpol":
        def f(z, mu=1.5):
            x, v = z[..., 0], z[..., 1]
            return np.stack([v, mu * (1.0 - x ** 2) * v - x], axis=-1)
        return f
    if name == "vanderpol_duncker":
        def f(z, tau=10.0, mu=2.0):
            x1, x2 = z[..., 0], z[..., 1]
            return np.stack([tau * mu * (x1 - x1 ** 3 / 3 - x2), tau * (x1 / mu)], axis=-1)
        return f
    if name == "lorenz":
        def f(z, s=10.0, r=28.0, b=8.0 / 3.0):
            x, y, w = z[..., 0], z[..., 1], z[..., 2]
            return np.stack([s * (y - x), x * (r - w) - y, x * y - b * w], axis=-1)
        return f
    raise NotImplementedError(f"synthetic in-memory gen: unknown system {name!r}")


def _simulate(name, a, batch, num_steps, dt, sigma, n_sub, init_std, burn_in, x0_uniform, rng):
    """Euler-Maruyama -> z (num_steps, batch, d); n_sub substeps/step; burn_in steps onto the attractor
    before recording. 1:1 with opssm.data.systems.simulate (isotropic diffusion sigma)."""
    drift, d = _drift(name, a), _DIM[name]
    sub = dt / n_sub
    rt = math.sqrt(sub)

    def step(z):
        for _ in range(n_sub):
            z = z + drift(z) * sub + sigma * rt * rng.standard_normal((batch, d))
        return z

    if x0_uniform is not None:
        z = (rng.random((batch, d)) * 2 - 1) * float(x0_uniform)
    else:
        z = init_std * rng.standard_normal((batch, d))
    for _ in range(burn_in):
        z = step(z)
    out = np.empty((num_steps, batch, d), np.float32)
    out[0] = z
    for i in range(1, num_steps):
        z = step(z)
        out[i] = z
    return out


def _linear_sensor(z, obs_dim, noise_std, c_scale, rng):
    """y = C z + d + noise, C (obs_dim, d) raw Gaussian (M-step imposes Stiefel). 1:1 with linear_sensor."""
    d = z.shape[-1]
    C = c_scale * rng.standard_normal((obs_dim, d))
    d_off = rng.standard_normal(obs_dim)
    y = np.einsum("od,tbd->tbo", C, z) + d_off
    y = y + noise_std * rng.standard_normal(y.shape)
    return y.astype(np.float32), C.astype(np.float32), d_off.astype(np.float32)


def _standardize(y, n_val, d_lat, noise_std):
    """Per-dim MEAN + a single GLOBAL SIGNAL scalar (geo-mean of top-d_lat singular values / sqrt(N)) from
    the TRAINING split (mask all-ones here). Mirrors DoubleWellDataModule._standardize_obs (s_scale removed)."""
    yt = y[:, n_val:].reshape(-1, y.shape[-1])                    # training observed entries
    mean = yt.mean(0)
    sv = np.linalg.svd(yt - mean, compute_uv=False)
    scale = max(float(np.exp(np.log(sv[:d_lat]).mean()) / math.sqrt(yt.shape[0])), 1e-6)
    return ((y - mean) / scale).astype(np.float32), mean.astype(np.float32), scale, noise_std / scale


def _softmax(x, axis):
    x = x - x.max(axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis, keepdims=True) + 1e-12)


def _grid_filter_highd(y, z, a, sigma, noise_std, dt, n_sub, C, d):
    """Exact 1-D grid filter p(z_t | y_{0:t}) -> filtered mass (T,B,Nz). numpy port of
    opssm.data.doublewell.oracle.grid_filter_highd (FORWARD pass only -- the d=1 kl uses the filter).
    Column-stochastic Euler-Maruyama transition (true drift) ^ n_sub, high-D Gaussian sensor likelihood
    N(y; C z + d, noise^2 I). Runs on RAW y/C/d (obs-scale-invariant over z). C is (D,), z (Nz,)."""
    dt_sub = dt / n_sub
    f = a * (z - z ** 3)                                          # true drift on the grid
    mean = z + f * dt_sub
    diff = z[:, None] - mean[None, :]                            # (Nz_k, Nz_j)
    T1 = _softmax(-0.5 * diff ** 2 / (sigma ** 2 * dt_sub), axis=0)   # column-stochastic transition
    K = np.linalg.matrix_power(T1, n_sub)                        # (Nz,Nz)  n_sub substeps per obs step
    hz = z[:, None] * C[None, :] + d[None, :]                    # (Nz, D)  C z + d on the grid
    inv2var = 0.5 / noise_std ** 2
    Tn, B, Nz = y.shape[0], y.shape[1], z.shape[0]
    pi = np.tile(_softmax(-0.5 * z ** 2, axis=0), (B, 1))        # (B,Nz) prior
    filt = np.empty((Tn, B, Nz), np.float32)
    for i in range(Tn):
        ll = -inv2var * ((y[i][:, None, :] - hz[None, :, :]) ** 2).sum(-1)   # (B,Nz)
        w = np.exp(ll - ll.max(1, keepdims=True)) * pi           # update (stabilized)
        w = w / (w.sum(1, keepdims=True) + 1e-12)
        filt[i] = w
        pi = w @ K.T                                            # predict
    return filt


def build_synthetic_data(dcfg, seed=0):
    """dcfg: cfg.data mapping. Returns the bridge-shaped dict of numpy arrays for _refs_from_mapping."""
    def g(k, dv=None):
        return dcfg[k] if k in dcfg else dv

    system = str(g("system", "doublewell"))
    highd = bool(g("highd", False))
    if system == "doublewell" and not highd:
        raise NotImplementedError(
            "synthetic in-memory gen supports the LINEAR-SENSOR path only (system != doublewell, or "
            "highd=true). Direct-obs 1-D (doublewell, highd=false) needs the npz bridge (dump/jax_port/bridge.py).")
    d_lat = _DIM[system]
    num_steps, batch, n_val = int(g("num_steps")), int(g("batch_size")), int(g("n_val"))
    t0, t1 = float(g("t0", 0.0)), float(g("t1"))
    dt = (t1 - t0) / num_steps
    sigma, noise_std, a = float(g("sigma")), float(g("noise_std")), float(g("a", 1.0))

    z_true = _simulate(system, a, batch, num_steps, dt, sigma, int(g("n_sub", 1)),
                       float(g("init_std", 1.0)), int(g("burn_in", 0)), g("x0_uniform", None),
                       np.random.default_rng(int(seed)))
    y, C, d_off = _linear_sensor(z_true, int(g("obs_dim")), noise_std, float(g("c_scale", 1.0)),
                                 np.random.default_rng(int(seed) + 1))
    y_std, _, obs_scale, nse = _standardize(y, n_val, d_lat, noise_std)

    D = dict(
        system=system,
        x_train=y_std[:, n_val:], mask_train=np.ones((num_steps, batch - n_val, 1), np.float32),
        x_val=y_std[:, :n_val], mask_val=np.ones((num_steps, n_val, 1), np.float32),
        full_obs=y_std[:, n_val:],
        z_val_true=z_true[:, :n_val], C_true=C, d_true=d_off,
        ts=np.linspace(t0, t1, num_steps, dtype=np.float32),
        dt=np.float32(dt), noise_std_eff=np.float32(nse), a=np.float32(a), sigma=np.float32(sigma),
        obs_scale=np.float32(obs_scale), obs_dim=np.int64(int(g("obs_dim"))),
    )
    if d_lat == 1:                                                # d=1: exact grid oracle IS tractable (O(N))
        zmax, Nz = float(g("zmax", 3.0)), int(g("Nz", 200))
        z_grid = np.linspace(-zmax, zmax, Nz, dtype=np.float32)
        D["z_grid"] = z_grid                                       # grid-mean readout + drift_l2
        D["filt_val"] = _grid_filter_highd(y[:, :n_val], z_grid, a, sigma, noise_std, dt,   # RAW val obs
                                           int(g("n_sub", 1)), C[:, 0], d_off)   # -> exact-filter kl target
    return D
