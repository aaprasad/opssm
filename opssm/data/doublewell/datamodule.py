# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LightningDataModule for the double-well testbed.

The operator is FULL-BATCH (it processes all trajectories every step), so a "batch" here is the
entire training set and an "iteration" is one optimizer step. The dataloaders therefore yield the
same full-batch tuple, `max_steps` times. All static tensors (the z-grid, the exact-filter oracle,
the true latent / sensor for high-D metrics) are computed once in `setup` and exposed as attributes
the LightningModule reads via `self.trainer.datamodule`.

Two modes (via `highd`):
  - 1-D direct observations  x_i = z_i + eps     (the EM-learn-f,g experiment)
  - high-D linear sensor     y   = C z + d + eps  (the Duncker-style learn f,g,C experiment)
"""

import math
import os

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset

from opssm.data.doublewell.sde import make_dataset, make_dataset_highd
from opssm.data.doublewell.oracle import grid_filter_target, grid_filter_highd
from opssm.data.systems import linear_sensor, make_drift, simulate


class _Repeat(Dataset):
    """Yields the same full-batch tuple every step; its length drives the per-epoch step count."""

    def __init__(self, data, n):
        self.data, self.n = data, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.data


class DoubleWellDataModule(pl.LightningDataModule):
    def __init__(self, a=1.0, sigma=0.6, noise_std=0.1, num_steps=100, t0=0.0, t1=10.0,
                 batch_size=256, n_val=32, n_sub=5, Nz=200, zmax=3.0, num_iters=14000,
                 gap_lo=0.5, gap_hi=0.5, highd=False, obs_dim=10, c_scale=1.0, seed=0,
                 system="doublewell", init_std=1.0, burn_in=0,
                 train_dir="./dump/dw_data"):
        super().__init__()
        self.save_hyperparameters()

    def _standardize_obs(self, y, mask, n_val, d_lat):
        """Standardize the linear-sensor obs so the encoder is well-conditioned AND the model needs NO
        internal obs-scale (s_scale removed). Per-dim MEAN + a single GLOBAL SIGNAL scalar, estimated from
        the TRAINING split's observed entries: scale = geo-mean of the top-d_lat singular values / sqrt(N).

        Why the SIGNAL scale (top-d SVD) and not the per-dim RMS: the model's collocation center is the
        pseudo-inverse latent zhat = C^T (y - d) with C orthonormal, so zhat has scale ~||signal||. The
        per-dim RMS is diluted by the D - d_lat noise dims, leaving zhat at scale ~sqrt(D/d)*1 >> 1; the
        operator's O(1) prior / fixed proposals then can't hold the gauge and the latent SCALE RUNS AWAY
        (raw kl blows up). Dividing by the signal scale pins zhat ~ O(1) -- this is exactly what s_scale
        did in the decode, now applied once to the data. A single global scalar keeps the obs noise
        ISOTROPIC and sensor DIRECTIONS intact (true C / c_cos / z_val_true unaffected). Records obs_mean,
        obs_scale, noise_std_eff (the config obs noise in standardized units); returns standardized obs."""
        obs = mask[:, n_val:, 0].bool()                            # training observed entries
        yt = y[:, n_val:][obs]                                     # (Nobs, D)
        mean = yt.mean(0)                                          # (D,) per-dim mean
        _, Sv, _ = torch.linalg.svd(yt - mean, full_matrices=False)
        scale = (Sv[:d_lat].log().mean().exp() / math.sqrt(yt.shape[0])).clamp_min(1e-6)   # signal scalar
        self.obs_mean, self.obs_scale = mean, float(scale)
        self.noise_std_eff = self.hparams.noise_std / self.obs_scale
        return (y - mean) / scale

    def setup(self, stage=None):
        h = self.hparams
        dev = self.trainer.strategy.root_device       # match the Trainer's device (data pinned there)
        os.makedirs(h.train_dir, exist_ok=True)
        self.z_grid = torch.linspace(-h.zmax, h.zmax, h.Nz, device=dev)
        # obs-standardization state; defaults = identity (direct-obs 1-D needs none, obs already ~O(1))
        self.obs_mean, self.obs_scale, self.noise_std_eff = None, 1.0, h.noise_std

        self.latent_dim = make_drift(h.system)[1]                  # d inferred from the system registry
        if h.system != "doublewell":                              # MULTI-D latent (VdP 2-D / Lorenz 3-D): high-D
            dt = (h.t1 - h.t0) / h.num_steps                       #   linear sensor, NO grid oracle (deferred)
            z_true = simulate(h.system, h.batch_size, h.num_steps, dt, h.sigma, h.n_sub,
                              init_std=h.init_std, burn_in=h.burn_in, device=dev, seed=h.seed)  # (T,B,d)
            y, C_true, d_true = linear_sensor(z_true, h.obs_dim, h.noise_std, h.c_scale, dev, seed=h.seed + 1)
            ts = torch.linspace(h.t0, h.t1, h.num_steps, device=dev)
            mask = torch.ones(h.num_steps, h.batch_size, 1, device=dev)
            y = self._standardize_obs(y, mask, h.n_val, self.latent_dim)                    # obs -> ~unit scale (s_scale removed)
            self.train_batch = (y[:, h.n_val:], mask[:, h.n_val:], None)   # filt=None (no d>1 oracle yet)
            self.val_batch = (y[:, :h.n_val], mask[:, :h.n_val], None)
            self.z_val_true, self.C_true, self.d_true = z_true[:, :h.n_val], C_true, d_true
            self.smoothed_val = None
            self.full_obs = y[:, h.n_val:]                         # for PCA / random C init
        elif h.highd:
            z_true, y, ts, C_true, d_true = make_dataset_highd(
                h.t0, h.t1, h.batch_size, h.num_steps, h.sigma, h.a, h.obs_dim,
                h.noise_std, h.c_scale, h.n_sub, dev, seed=h.seed)
            dt = float((ts[1] - ts[0]).item())
            mask = torch.ones(h.num_steps, h.batch_size, 1, device=dev)
            filt, smoothed = grid_filter_highd(y, self.z_grid, h.a, h.sigma, h.noise_std,
                                               dt, h.n_sub, C_true, d_true)   # oracle on RAW y (over z; obs-scale invariant)
            y = self._standardize_obs(y, mask, h.n_val, self.latent_dim)                       # standardize AFTER the oracle (s_scale removed)
            y_tr, y_val = y[:, h.n_val:], y[:, :h.n_val]
            mask_tr, mask_val = mask[:, h.n_val:], mask[:, :h.n_val]
            self.train_batch = (y_tr, mask_tr, filt[:, h.n_val:])
            self.val_batch = (y_val, mask_val, filt[:, :h.n_val])
            # expose with a trailing latent axis (d=1) for the generalized (d-D) model
            self.z_val_true = z_true[:, :h.n_val].unsqueeze(-1)    # (T,n_val,1)
            self.C_true = C_true.unsqueeze(-1)                     # (D,1)
            self.d_true = d_true
            self.smoothed_val = smoothed[:, :h.n_val]             # oracle smoother ground truth (kl_smooth)
            self.full_obs = y_tr                                   # for PCA / random C init
        else:
            xs, ts = make_dataset(h.t0, h.t1, h.batch_size, h.noise_std, h.num_steps,
                                  h.sigma, h.a, h.train_dir, dev)
            dt = float((ts[1] - ts[0]).item())
            mask_t = torch.ones(h.num_steps, dtype=torch.bool, device=dev)
            mask_t[int(h.gap_lo * h.num_steps):int(h.gap_hi * h.num_steps)] = False
            mask = mask_t.view(h.num_steps, 1, 1).float().expand(h.num_steps, h.batch_size, 1).contiguous()
            filtered, smoothed = grid_filter_target(xs, self.z_grid, h.a, h.sigma, h.noise_std,
                                                    dt, h.n_sub, mask=mask_t)
            self.train_batch = (xs[:, h.n_val:], mask[:, h.n_val:], filtered[:, h.n_val:])
            self.val_batch = (xs[:, :h.n_val], mask[:, :h.n_val], filtered[:, :h.n_val])
            self.smoothed_val = smoothed[:, :h.n_val]             # oracle smoother ground truth (kl_smooth)
            self.z_val_true = self.C_true = self.d_true = self.full_obs = None

        self.ts, self.dt = ts, dt

    def train_dataloader(self):
        return DataLoader(_Repeat(self.train_batch, self.hparams.num_iters), batch_size=None)

    def val_dataloader(self):
        return DataLoader(_Repeat(self.val_batch, 1), batch_size=None)
