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

"""Double-well latent SDE: generative model + datasets (1-D direct obs and high-D linear sensor)."""

import logging
import math
import os

import torch

import torchsde


class DoubleWell(object):
    """Stochastic 1D double-well system: dX = a (X - X^3) dt + sigma dW."""

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, a: float = 1.0, sigma: float = 0.6):
        self.a = a
        self.sigma = sigma

    def f(self, t, y):
        return self.a * (y - y ** 3)

    def g(self, t, y):
        return torch.full_like(y, self.sigma)

    @torch.no_grad()
    def sample(self, x0, ts, noise_std):
        xs = torchsde.sdeint(self, x0, ts, dt=1e-2)
        if noise_std > 0:
            xs = xs + torch.randn_like(xs) * noise_std
        return xs


def make_dataset(t0, t1, batch_size, noise_std, num_steps, sigma, a, train_dir, device):
    data_path = os.path.join(train_dir, f"double_well_raw_sigma{sigma}.pth")
    if os.path.exists(data_path):
        data_dict = torch.load(data_path)
        xs, ts = data_dict["xs"], data_dict["ts"]
        logging.warning(f"Loaded toy data at: {data_path}")
        if xs.shape[1] != batch_size:
            raise ValueError("Batch size has changed; delete and regenerate the data.")
        if ts[0] != t0 or ts[-1] != t1:
            raise ValueError("Time interval changed; delete and regenerate the data.")
    else:
        _y0 = torch.randn(batch_size, 1, device=device)
        ts = torch.linspace(t0, t1, steps=num_steps, device=device)
        xs = DoubleWell(a=a, sigma=sigma).sample(_y0, ts, noise_std)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        torch.save({"xs": xs, "ts": ts}, data_path)
        logging.warning(f"Stored toy data at: {data_path}")
    return xs, ts


def make_dataset_highd(t0, t1, batch_size, num_steps, sigma, a, obs_dim, noise_std,
                       c_scale, n_sub, device, seed=0):
    """Latent double-well SDE z (T,B) + high-D linear obs y = C z + d + noise (T,B,D).
    Returns z, y, ts, C (D,), d (D,)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    dt = (t1 - t0) / num_steps
    sub = dt / n_sub
    ts = torch.linspace(t0, t1, num_steps, device=device)
    z = torch.zeros(num_steps, batch_size, device=device)
    z[0] = torch.randn(batch_size, device=device, generator=gen)
    for i in range(1, num_steps):
        zz = z[i - 1].clone()
        for _ in range(n_sub):
            zz = zz + a * (zz - zz ** 3) * sub + sigma * math.sqrt(sub) * \
                torch.randn(batch_size, device=device, generator=gen)
        z[i] = zz
    C = c_scale * torch.randn(obs_dim, device=device, generator=gen)
    d = torch.randn(obs_dim, device=device, generator=gen)
    y = C * z[..., None] + d + noise_std * torch.randn(num_steps, batch_size, obs_dim,
                                                       device=device, generator=gen)
    return z, y, ts, C, d
