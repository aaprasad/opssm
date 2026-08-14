"""LightningDataModule for the Kato et al. 2015 C. elegans whole-brain calcium data (per worm).

Real neural data -> NO ground truth: `z_val_true / C_true / d_true / smoothed_val = None`, `filt = None` in
every batch, so all ground-truth-dependent metrics/figures in `filter_module.validation_step` auto-skip
(they guard on `... is not None`). Mirrors the multi-D branch of `DoubleWellDataModule`'s contract.

Windowed batching: the single worm's long trace (T~2000-3300) is sliced into windows of length `window`
with `stride` (overlapping if stride<window). Windows are the batch dim -> `x (window, B=#windows, N)`,
each filtered independently. Train/val split is by TIME (train = early, val = late) so windows don't leak.
"""
import math

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset

from opssm.data.kato.load import load_worm


class _Repeat(Dataset):
    """Yields the same full-batch tuple every step (full-batch training; length drives per-epoch steps)."""
    def __init__(self, data, n):
        self.data, self.n = data, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.data


def _window(y, window, stride):
    """y (T, N) -> (x (window, B, N), starts list). B = number of windows."""
    starts = list(range(0, y.shape[0] - window + 1, stride))
    if not starts:                                          # segment shorter than window -> one clipped window
        starts = [0]
        window = y.shape[0]
    x = torch.stack([y[s:s + window] for s in starts], dim=1)     # (window, B, N)
    return x, starts, window


class KatoDataModule(pl.LightningDataModule):
    def __init__(self, mat_path, worm=0, window=200, stride=100, val_frac=0.2, noise_std=0.5,
                 clip_negative=True, subsample=1, latent_dim=10, num_iters=14000, seed=0,
                 a=1.0, sigma=0.1, system="none", train_dir="./dump/kato"):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        h = self.hparams
        dev = self.trainer.strategy.root_device       # match the Trainer's device
        w = load_worm(h.mat_path, h.worm)
        y = torch.from_numpy(w["traces"]).float()                       # (T, N)
        if h.clip_negative:
            y = y.clamp_min(0.0)
        if h.subsample > 1:
            y = y[::h.subsample]
        y = y.to(dev)
        T, N = y.shape
        dt = w["dt"] * h.subsample

        # split by TIME (train early, val late), then window each segment independently (no leakage)
        n_val_t = int(h.val_frac * T)
        y_tr, y_val = y[:T - n_val_t], y[T - n_val_t:]
        x_tr, starts_tr, win_tr = _window(y_tr, h.window, h.stride)     # (win, B_tr, N)
        x_val, starts_val, win_val = _window(y_val, h.window, h.stride)

        # standardize on TRAIN entries (per-dim mean + single signal-scale so zhat=C^T(y-d) ~ O(1)),
        # apply to both. Mirrors DoubleWellDataModule._standardize_obs.
        d_lat = h.latent_dim
        yt = x_tr.reshape(-1, N)                                        # (T_tr*B_tr, N)
        mean = yt.mean(0)                                               # (N,)
        s = torch.linalg.svdvals(yt - mean)                            # signal spectrum
        scale = float(s[:d_lat].log().mean().exp() / math.sqrt(yt.shape[0]))
        self.obs_mean, self.obs_scale = mean, scale
        self.noise_std_eff = h.noise_std / scale
        x_tr = (x_tr - mean) / scale
        x_val = (x_val - mean) / scale

        m_tr = torch.ones(x_tr.shape[0], x_tr.shape[1], 1, device=dev)
        m_val = torch.ones(x_val.shape[0], x_val.shape[1], 1, device=dev)
        self.train_batch = (x_tr, m_tr, None)
        self.val_batch = (x_val, m_val, None)
        self.full_obs = x_tr

        # ground-truth-free -> None (auto-skips GT metrics/figures)
        self.z_val_true = self.C_true = self.d_true = self.smoothed_val = None
        self.dt = dt
        self.z_grid = torch.linspace(-3.0, 3.0, 200, device=dev)       # dummy (unused at d>1)
        self.ts = torch.arange(h.window, device=dev) * dt
        self.hparams.obs_dim = N                                        # model reads dm.hparams.obs_dim

        # for post-hoc eval (behavior decoding + whole-trace stitching)
        self.states_full = w["states"][::h.subsample]                  # (T,) behavior labels
        self.state_names = w["state_names"]
        self.neuron_ids = w["neuron_ids"]
        self.y_full = y                                                # (T, N) raw (clipped) dF/F
        self.window_starts_train, self.window_starts_val = starts_tr, starts_val
        self.n_neurons, self.name = N, w["name"]

    def train_dataloader(self):
        return DataLoader(_Repeat(self.train_batch, self.hparams.num_iters), batch_size=None)

    def val_dataloader(self):
        return DataLoader(_Repeat(self.val_batch, 1), batch_size=None)
