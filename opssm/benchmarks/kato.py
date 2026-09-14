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
    train_fraction: float = .6   # Used only by the legacy single-split protocol (folds <= 1).
    val_fraction: float = .2
    folds: int = 5               # Rotating blocked CV: K contiguous blocks, test rotates over them.
    fold: int = 0                # Which block is the test block, 0-based. -1 = no split at all
                                 #   (train == val == test == the whole trace), the dLDS Table 1
                                 #   protocol. Scores from it are IN-SAMPLE, not held out.
    # Kato traces stay autocorrelated for 34-111 frames (12-36 s, 1/e, WT_NoStim), so the old
    # 30-frame gap left the held-out blocks correlated with training and inflated their scores.
    gap_frames: int = 100
    noise_std: float = .1  # Initialization in standardized units, NOT known noise.
    name: str = "kato"
    system: str = "kato"
    obs_dim: int = 0  # Resolved from the recording.
    diffusion_type: str = "constant"  # Model family; biological diffusion is unknown.


def split_segments(n_frames, cfg):
    """Frame spans per split.

    folds <= 1 keeps the legacy single chronological 60/20/20 split. Otherwise the trace is cut
    into `folds` contiguous blocks and block `fold` is the test block, with the next block (wrapping)
    held out for validation -- so rotating `fold` over 0..folds-1 tests every region of the recording
    exactly once, while each individual fold still trains on all the remaining blocks.

    The gap is trimmed off the held-out blocks only, never off training: that is what decorrelates a
    held-out score from the training data, and taking it from the training side as well would cost
    scarce frames without changing the lag between the two.
    """
    if cfg.fold < 0:
        # No split: fit and score every frame, as dLDS does for its whole-trace R2. There is no
        # held-out data here, so recon R2 measures fit quality, not generalization -- and with a
        # rank-d linear observation model it is bounded above by rank-d PCA, which involves no
        # dynamics at all. Report it against that ceiling, never on its own.
        return {split: [(0, n_frames)] for split in ("train", "val", "test")}
    if cfg.folds <= 1:
        a = int(n_frames * cfg.train_fraction)
        b = int(n_frames * (cfg.train_fraction + cfg.val_fraction))
        return {"train": [(0, a)], "val": [(a + cfg.gap_frames, b)],
                "test": [(b + cfg.gap_frames, n_frames)]}
    if cfg.folds < 3:
        raise ValueError("Rotating Kato CV needs folds >= 3 to leave a training block")
    if not 0 <= cfg.fold < cfg.folds:
        raise ValueError(f"fold {cfg.fold} outside 0..{cfg.folds - 1} (-1 = no split)")
    edges = np.linspace(0, n_frames, cfg.folds + 1).astype(int)
    label = {k: "train" for k in range(cfg.folds)}
    label[cfg.fold] = "test"
    label[(cfg.fold + 1) % cfg.folds] = "val"
    segments = {"train": [], "val": [], "test": []}
    for k in range(cfg.folds):
        start, stop = int(edges[k]), int(edges[k + 1])
        if label[k] != "train":                                   # pull held-out blocks off the seam
            if k > 0 and label[k - 1] != label[k]:
                start += cfg.gap_frames
            if k < cfg.folds - 1 and label[k + 1] != label[k]:
                stop -= cfg.gap_frames
        segments[label[k]].append((start, stop))
    return segments


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
    suffix = "_full" if cfg.fold < 0 else ("" if cfg.folds <= 1 else f"_fold{cfg.fold}of{cfg.folds}")
    cfg = replace(cfg, name=f"kato_{recording['name']}{suffix}", obs_dim=y.shape[1])
    segments = split_segments(len(y), cfg)
    bounds = {split: [list(s) for s in spans] for split, spans in segments.items()}
    # Standardization uses only the raw training block. No clipping, interpolation,
    # derivatives, smoothing, or extra observation noise is added.
    #
    # Per-neuron mean, then ONE global signal scale from the top-latent_dim singular values --
    # the same rule as the synthetic presets and opssm/data/kato/datamodule.py, so that
    # zhat = C^T (y - d) is O(1). Dividing each neuron by its own SD instead whitens the
    # observations, destroying the relative amplitude structure that fixes the scale of the
    # top-d subspace; the latent scale is then unidentified and the learned diffusion absorbs
    # the mismatch, which diverges (g and obs_noise run away, recon R2 goes sharply negative).
    # Only TRAINING frames, gathered across every training span -- with a rotating test block the
    # training set is no longer a prefix of the trace, so a leading slice would span held-out blocks.
    y_fit = np.concatenate([y[start:stop] for start, stop in segments["train"]])
    mean = y_fit.mean(0)
    singular = np.linalg.svd(y_fit - mean, compute_uv=False)[:cfg.latent_dim]
    scale = np.array(max(float(np.exp(np.log(np.clip(singular, 1e-12, None)).mean())
                               / np.sqrt(len(y_fit))), 1e-6))
    data = dict(ts=np.arange(cfg.window) * dt, obs_mean=mean, obs_scale=scale,
                noise_std_eff=np.array(cfg.noise_std), neuron_ids=np.asarray(recording["neuron_ids"], dtype=str))
    dropped = {}
    for split, spans in segments.items():
        stride = cfg.train_stride if split == "train" else cfg.window
        starts = np.concatenate([np.arange(start, stop - cfg.window + 1, stride, dtype=np.int64)
                                 for start, stop in spans] or [np.empty(0, np.int64)])
        if smoke and len(starts):
            count = min(len(starts), 4 if split == "train" else 2)
            starts = starts[np.linspace(0, len(starts) - 1, count, dtype=int)]
        if not len(starts):
            raise ValueError(f"Kato {split} block too short for window={cfg.window} with "
                             f"gap={cfg.gap_frames} and folds={cfg.folds}")
        frames = np.arange(cfg.window)[:, None] + starts[None, :]
        data[f"frame_indices_{split}"] = frames
        data[f"y_{split}"] = ((y[frames] - mean) / scale).astype(np.float32)
        data[f"states_{split}"] = labels[frames]
        dropped[split] = int(sum(stop - start for start, stop in spans)
                             - len(starts) * cfg.window if split != "train" else 0)
    metadata = dict(schema_version=2, dataset_kind="real", config=asdict(cfg), smoke=smoke,
                    source="https://osf.io/2395t/", source_sha256=source_sha256,
                    fps=float(recording["fps"]), n_frames=len(y), split_frame_bounds=bounds,
                    unused_tail_frames=dropped, state_names=recording["state_names"],
                    preprocessing="train-only per-neuron mean + global top-d SVD signal scale; corrected fluorescence; native fps",
                    protocol="whole_trace_in_sample" if cfg.fold < 0 else "rotating_blocked_cv",
                    held_out=cfg.fold >= 0,
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
