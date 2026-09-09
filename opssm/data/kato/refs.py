"""Torch-free, in-memory Kato data prep -- a numpy replica of KatoDataModule.setup, for the JAX backend.

The .[jax] venv has no torch, so it cannot instantiate the (torch/Lightning) datamodule. For a preemptible
grid search we also don't want a bridged .npz on disk per run. This builds the exact same arrays the bridge
writes (minus 'hparams', which the JAX entrypoint fills from cfg.model) directly from the raw .mat, using only
numpy/scipy -- so each sweep task holds its data in memory. Feed the returned dict to
opssm.models.jax.train._refs_from_mapping. Validated to match the torch bridge to ~1e-5 (obs_scale, x_train).
"""
import math

import numpy as np

from opssm.data.kato.load import load_worm


def _window(y, window, stride):
    """y (T, N) -> (x (window, B, N), starts). B = number of windows (>=1; a short segment -> one clipped window)."""
    starts = list(range(0, y.shape[0] - window + 1, stride))
    if not starts:
        starts, window = [0], y.shape[0]
    x = np.stack([y[s:s + window] for s in starts], axis=1)          # (window, B, N)
    return x, starts, window


def build_kato_data(mat_path, worm=0, window=200, stride=100, val_frac=0.2, noise_std=0.1,
                    clip_negative=True, subsample=1, latent_dim=10, a=1.0, sigma=0.1, system="none"):
    """Numpy replica of KatoDataModule.setup -> dict of arrays == the bridge's output (minus 'hparams').
    obs standardized on TRAIN entries (per-dim mean + single SVD signal-scale); noise_std_eff = noise_std/scale."""
    w = load_worm(mat_path, worm)
    y = np.asarray(w["traces"], dtype=np.float32)                    # (T, N) raw dF/F
    if clip_negative:
        y = np.clip(y, 0.0, None)
    if subsample > 1:
        y = y[::subsample]
    T, N = y.shape
    dt = float(w["dt"]) * subsample

    n_val_t = int(val_frac * T)                                      # split by TIME (train early, val late)
    y_tr, y_val = y[:T - n_val_t], y[T - n_val_t:]
    x_tr, _, _ = _window(y_tr, window, stride)
    x_val, starts_val, _ = _window(y_val, window, stride)

    yt = x_tr.reshape(-1, N)                                         # standardize on TRAIN windows
    mean = yt.mean(0)                                                # (N,)
    s = np.linalg.svd(yt - mean, compute_uv=False)                  # signal spectrum (descending)
    scale = float(math.exp(float(np.log(s[:latent_dim]).mean())) / math.sqrt(yt.shape[0]))
    noise_std_eff = noise_std / scale
    x_tr = (x_tr - mean) / scale
    x_val = (x_val - mean) / scale

    states_full = np.asarray(w["states"])[::subsample]              # (T,) behavior labels
    off, win_b = T - n_val_t, x_val.shape[0]                        # val-window states, aligned to the full trace
    states_val = np.stack([states_full[off + st: off + st + win_b] for st in starts_val], axis=1)
    neuron_ids = np.array([(sid if (sid := str(x)) and not sid.isdigit() else "")   # named neurons; blank numeric
                           for x in w["neuron_ids"]], dtype=object)

    return dict(
        system=str(system), name=str(w["name"]),
        x_train=x_tr, mask_train=np.ones((x_tr.shape[0], x_tr.shape[1], 1), np.float32),
        x_val=x_val, mask_val=np.ones((x_val.shape[0], x_val.shape[1], 1), np.float32),
        full_obs=x_tr,
        dt=np.asarray(dt), noise_std_eff=np.asarray(noise_std_eff),
        a=np.asarray(float(a)), sigma=np.asarray(float(sigma)),
        obs_dim=np.array(int(N)), obs_mean=mean, obs_scale=np.asarray(scale),
        ts=(np.arange(window) * dt).astype(np.float32),
        states_full=states_full, states_val=states_val,
        state_names=np.array([str(x) for x in w["state_names"]], dtype=object),
        neuron_ids=neuron_ids,
        y_full_std=(y - mean) / scale, y_full_raw=y,
        window=np.array(int(window)), stride=np.array(int(stride)),
    )
