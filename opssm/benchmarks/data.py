"""Shared, versioned datasets. No framework import and no model-dependent RNG.

All arrays are time x trajectory x channel. Truth is retained ONLY for scoring.
See docs/baselines.md for provenance and deliberate deviations from papers.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    system: str
    latent_dim: int
    obs_dim: int
    duration: float
    num_obs: int
    simulation_dt: float
    diffusion: float
    noise_std: float
    n_train: int = 20
    n_val: int = 5
    n_test: int = 10
    a: float = 4.0
    tau: float = 15.0
    mu: float = 2.0
    initial: str = "normal"
    initial_scale: float = 1.0
    sensor: str = "gaussian"
    diffusion_type: str = "constant"
    noise_units: str = "physical"

    def validate(self):
        if self.system not in ("doublewell", "vanderpol", "lorenz"):
            raise ValueError(f"Unknown system {self.system}")
        expected = {"doublewell": 1, "vanderpol": 2, "lorenz": 3}[self.system]
        if self.latent_dim != expected or self.obs_dim < expected:
            raise ValueError("Inconsistent latent/observation dimensions")
        if self.num_obs < 2 or min(self.n_train, self.n_val, self.n_test) < 1:
            raise ValueError("Need at least two times and nonempty independent splits")
        if min(self.duration, self.simulation_dt, self.noise_std, self.diffusion) <= 0:
            raise ValueError("Time and noise parameters must be positive")
        if self.sensor not in ("gaussian", "identity") or self.initial not in ("normal", "uniform"):
            raise ValueError("Unknown sensor/initial distribution")
        if self.sensor == "identity" and self.obs_dim != self.latent_dim:
            raise ValueError("Identity sensor requires equal dimensions")
        if self.diffusion_type not in ("constant", "lorenz_multiplicative"):
            raise ValueError("Unknown diffusion type")
        if self.diffusion_type == "lorenz_multiplicative" and self.system != "lorenz":
            raise ValueError("Multiplicative reference is Lorenz only")
        if self.noise_units not in ("physical", "normalized_signal"):
            raise ValueError("Unknown observation noise units")


PRESETS = {
    "doublewell": DatasetConfig("doublewell", "doublewell", 1, 15, 3., 301, .001, 1., .5),
    "vanderpol": DatasetConfig("vanderpol", "vanderpol", 2, 20, 3., 301, .001, 1., 1.5,
                              initial="uniform", initial_scale=3.),
    "lorenz": DatasetConfig("lorenz", "lorenz", 3, 3, 2., 100, .001, .3, .01,
                           n_train=1024, n_val=128, n_test=128, sensor="identity",
                           noise_units="normalized_signal"),
    "lorenz_li": DatasetConfig("lorenz_li", "lorenz", 3, 3, 2., 100, .001, 1., .01,
                              n_train=1024, n_val=128, n_test=128, sensor="identity",
                              noise_units="normalized_signal", diffusion_type="lorenz_multiplicative"),
}


def drift(z, cfg):
    if cfg.system not in ("doublewell", "vanderpol", "lorenz"):
        raise ValueError(f"No ground-truth drift available for {cfg.system}")
    if cfg.system == "doublewell":
        return cfg.a * (z - z ** 3)
    if cfg.system == "vanderpol":
        x, y = z[..., 0], z[..., 1]
        return np.stack([cfg.tau * cfg.mu * (x - x ** 3 / 3 - y), cfg.tau * x / cfg.mu], -1)
    x, y, w = z[..., 0], z[..., 1], z[..., 2]
    return np.stack([10 * (y - x), x * (28 - w) - y, x * y - (8 / 3) * w], -1)


def volatility(z, cfg):
    if cfg.diffusion_type == "lorenz_multiplicative":
        return z * np.array([.1, .28, .3])
    return np.full_like(z, cfg.diffusion)


def simulate(cfg, batch, rng):
    ts = np.linspace(0., cfg.duration, cfg.num_obs)
    shape = (batch, cfg.latent_dim)
    z = (rng.uniform(-1., 1., shape) if cfg.initial == "uniform" else rng.standard_normal(shape))
    z *= cfg.initial_scale
    out = [z.copy()]
    for interval in np.diff(ts):
        # Hit observation times exactly, including 2/99 in the Li reference.
        n = max(1, int(np.ceil(interval / cfg.simulation_dt - 1e-10)))
        h = interval / n
        for _ in range(n):
            z = z + drift(z, cfg) * h + volatility(z, cfg) * np.sqrt(h) * rng.standard_normal(shape)
        if not np.isfinite(z).all():
            raise FloatingPointError("Simulation diverged; decrease simulation_dt")
        out.append(z.copy())
    return np.stack(out)


def make_dataset(cfg, seed=0):
    cfg.validate()
    # Split-specific streams ensure test size cannot change training data or sensor.
    streams = np.random.SeedSequence(seed).spawn(7)
    rng = [np.random.default_rng(s) for s in streams]
    C = (np.eye(cfg.latent_dim) if cfg.sensor == "identity" else
         rng[0].standard_normal((cfg.obs_dim, cfg.latent_dim)))
    offset = np.zeros(cfg.obs_dim) if cfg.sensor == "identity" else rng[0].standard_normal(cfg.obs_dim)
    zs = {s: simulate(cfg, n, rng[i + 1]) for i, (s, n) in enumerate(
        [("train", cfg.n_train), ("val", cfg.n_val), ("test", cfg.n_test)])}
    signals = {s: z @ C.T + offset for s, z in zs.items()}
    if cfg.noise_units == "normalized_signal":
        # Part of the synthetic sensor definition, as in Li; train-only constants.
        center = signals["train"].mean((0, 1))
        scale = signals["train"].std((0, 1)).clip(1e-6)
        C, offset = C / scale[:, None], (offset - center) / scale
        signals = {s: (x - center) / scale for s, x in signals.items()}
    ys = {s: x + cfg.noise_std * rng[i + 4].standard_normal(x.shape)
          for i, (s, x) in enumerate(signals.items())}
    # Model preprocessing uses noisy TRAIN observations only, with a common scalar.
    flat = ys["train"].reshape(-1, cfg.obs_dim)
    mean = flat.mean(0)
    sv = np.linalg.svd(flat - mean, compute_uv=False)[:cfg.latent_dim]
    scale = max(float(np.exp(np.log(sv).mean()) / np.sqrt(len(flat))), 1e-6)
    data = dict(ts=np.linspace(0., cfg.duration, cfg.num_obs), obs_mean=mean, obs_scale=np.array(scale),
                C_true=C / scale, d_true=(offset - mean) / scale,
                noise_std_eff=np.array(cfg.noise_std / scale))
    for split in zs:
        data[f"y_{split}"] = ((ys[split] - mean) / scale).astype(np.float32)
        data[f"signal_{split}"] = ((signals[split] - mean) / scale).astype(np.float32)
        data[f"z_{split}"] = zs[split].astype(np.float32)
    metadata = dict(schema_version=1, config=asdict(cfg), seed=seed, integrator="Euler-Maruyama",
                    actual_simulation_dt=(cfg.duration / (cfg.num_obs - 1)) /
                    int(np.ceil(cfg.duration / (cfg.num_obs - 1) / cfg.simulation_dt - 1e-10)),
                    preprocessing="train-only mean and global PCA signal scale")
    data["metadata"] = np.array(json.dumps(metadata, sort_keys=True))
    return data


def fingerprint(data):
    h = hashlib.sha256()
    for key in sorted(data):
        value = np.ascontiguousarray(data[key])
        h.update(key.encode()); h.update(str(value.shape).encode()); h.update(str(value.dtype).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def config_from_data(data):
    config = json.loads(str(data["metadata"]))["config"]
    if config["system"] == "kato":
        from .kato import KatoConfig
        return KatoConfig(**config)
    return DatasetConfig(**config)


def save_dataset(path, data):
    """Write the dataset atomically, so concurrent writers cannot produce a torn file.

    Hydra multirun launches every cell of a sweep at once and they share one data.npz per
    (point, dataset, seed). make_dataset is deterministic in (cfg, seed), so racing writers
    produce identical bytes; a unique temp name plus os.replace makes the swap atomic, and a
    reader therefore sees either no file or a complete one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if fingerprint(dict(old)) != fingerprint(data):
                raise FileExistsError(f"Different dataset already exists at {path}; use a new output directory")
        return
    temp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp.npz")
    try:
        np.savez_compressed(temp, **data)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def smoke_config(cfg):
    return replace(cfg, num_obs=12, duration=11 * cfg.duration / (cfg.num_obs - 1),
                   n_train=4, n_val=2, n_test=2)
