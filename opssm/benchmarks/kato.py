"""Per-worm Kato 2015 benchmark; split raw time BEFORE forming windows.

Only the raw MATLAB reader is shared with the existing Kato application. This
benchmark has its own three-way split, preprocessing, and observation-only scores.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class KatoConfig:
    mat_path: str
    worm: int = 0
    latent_dim: int = 10
    window: int = 200
    train_stride: int = 100
    train_fraction: float = .6
    val_fraction: float = .2
    gap_frames: int = 30
    noise_std: float = .1  # Initialization in standardized units, NOT known noise.
    name: str = "kato"
    system: str = "kato"
    obs_dim: int = 0  # Resolved from the recording.
    diffusion_type: str = "constant"  # Model family; biological diffusion is unknown.


def prepare_recording(recording, cfg, smoke=False, source_sha256=None):
    """Pure NumPy preparation, also usable with tiny fixtures for leakage tests."""
    y = np.asarray(recording["traces"], dtype=np.float64)
    labels = np.asarray(recording["states"], dtype=np.int64)
    if y.ndim != 2 or labels.shape != (len(y),) or not np.isfinite(y).all():
        raise ValueError("Kato requires finite T x neuron traces and one behavior label per frame")
    if not 0 < cfg.train_fraction < 1 or not 0 < cfg.val_fraction < 1 - cfg.train_fraction:
        raise ValueError("Kato train/val fractions must leave a nonempty test block")
    if cfg.gap_frames < 0 or cfg.window < 4 or not 1 <= cfg.train_stride <= cfg.window:
        raise ValueError("Need nonnegative gap, window >=4, and 1 <= train_stride <= window")
    if not 1 <= cfg.latent_dim <= y.shape[1] or cfg.noise_std <= 0:
        raise ValueError("Invalid latent dimension or observation-noise initialization")
    dt = float(recording["dt"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid recorded sampling rate")
    if smoke:
        cfg = replace(cfg, window=min(cfg.window, 12), train_stride=min(cfg.train_stride, 12))
    cfg = replace(cfg, name=f"kato_{recording['name']}", obs_dim=y.shape[1])
    a, b = int(len(y) * cfg.train_fraction), int(len(y) * (cfg.train_fraction + cfg.val_fraction))
    bounds = {"train": (0, a), "val": (a + cfg.gap_frames, b), "test": (b + cfg.gap_frames, len(y))}
    # Standardization uses only the raw training block. No clipping, interpolation,
    # derivatives, smoothing, or extra observation noise is added.
    #
    # Per-neuron mean, then ONE global signal scale from the top-latent_dim singular values --
    # the same rule as the synthetic presets and opssm/data/kato/datamodule.py, so that
    # zhat = C^T (y - d) is O(1). Dividing each neuron by its own SD instead whitens the
    # observations, destroying the relative amplitude structure that fixes the scale of the
    # top-d subspace; the latent scale is then unidentified and the learned diffusion absorbs
    # the mismatch, which diverges (g and obs_noise run away, recon R2 goes sharply negative).
    mean = y[:a].mean(0)
    singular = np.linalg.svd(y[:a] - mean, compute_uv=False)[:cfg.latent_dim]
    scale = np.array(max(float(np.exp(np.log(np.clip(singular, 1e-12, None)).mean())
                               / np.sqrt(a)), 1e-6))
    data = dict(ts=np.arange(cfg.window) * dt, obs_mean=mean, obs_scale=scale,
                noise_std_eff=np.array(cfg.noise_std), neuron_ids=np.asarray(recording["neuron_ids"], dtype=str))
    dropped = {}
    for split, (start, stop) in bounds.items():
        stride = cfg.train_stride if split == "train" else cfg.window
        starts = np.arange(start, stop - cfg.window + 1, stride, dtype=np.int64)
        if smoke and len(starts):
            count = min(len(starts), 4 if split == "train" else 2)
            starts = starts[np.linspace(0, len(starts) - 1, count, dtype=int)]
        if not len(starts):
            raise ValueError(f"Kato {split} block too short for window={cfg.window} with gap={cfg.gap_frames}")
        frames = np.arange(cfg.window)[:, None] + starts[None, :]
        data[f"frame_indices_{split}"] = frames
        data[f"y_{split}"] = ((y[frames] - mean) / scale).astype(np.float32)
        data[f"states_{split}"] = labels[frames]
        dropped[split] = int(stop - (starts[-1] + cfg.window))
    metadata = dict(schema_version=2, dataset_kind="real", config=asdict(cfg), smoke=smoke,
                    source="https://osf.io/2395t/", source_sha256=source_sha256,
                    fps=float(recording["fps"]), n_frames=len(y), split_frame_bounds=bounds,
                    unused_tail_frames=dropped, state_names=recording["state_names"],
                    preprocessing="train-only per-neuron mean + global top-d SVD signal scale; corrected fluorescence; native fps",
                    ground_truth_available=False, noise_std_is_initialization=True)
    data["metadata"] = np.array(json.dumps(metadata, sort_keys=True))
    return cfg, data


def make_kato_dataset(cfg, smoke=False):
    from opssm.data.kato.load import load_worm, n_worms
    path = Path(cfg.mat_path).expanduser().resolve()
    if not 0 <= cfg.worm < n_worms(path):
        raise ValueError(f"Worm index {cfg.worm} outside {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return prepare_recording(load_worm(path, cfg.worm), replace(cfg, mat_path=str(path)), smoke, digest.hexdigest())
