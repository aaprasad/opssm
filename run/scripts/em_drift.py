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

"""EM for the drift: tighten the latent-path drift regression past the one-shot 0.67.

The diagnostic (drift_from_smoothing.py) recovered f by regressing the increments of a latent
path inferred by a GENERIC (f=0) smoother. Its residual error was the f=0 prior attenuating the
drift. EM fixes that by using the CURRENT estimate of f in the smoother and iterating:

    E-step:  given f_theta, infer the smoothing posterior over latent PATHS (exact grid filter
             + FFBS sampling -- the nonlinearity-correct posterior, not just the mean).
    M-step:  regress f_theta(z) ~ (z_{t+1}-z_t)/dt over the sampled paths.

A better f sharpens the smoother, which sharpens the path samples, which sharpens f. The grid
filter here is the EXACT E-step (the neural operator replaces it in the integrated version); this
script isolates the EM convergence of the drift. Watch `drift L2 err` fall below 0.67.

To run:
python -m run.scripts.em_drift
"""

import logging
import os

import fire
import matplotlib.pyplot as plt
import torch
import tqdm

from run.scripts.latent_sde_double_well_zakai import (
    make_dataset, build_transition, transition_power, forward_backward, ffbs_sample)
from run.scripts.neural_zakai_filter import mlp


def main(batch_size=512, t0=0.0, t1=10.0, num_steps=100, a=1.0, sigma=0.6, noise_std=0.1,
         Nz=200, zmax=3.0, n_sub=5, n_samp=4, n_em=12, m_steps=800, lr=2e-3,
         train_dir="./dump/em_drift/"):
    torch.manual_seed(0)
    os.makedirs(train_dir, exist_ok=True)
    dev = torch.device("cpu")          # E-step is grid matmuls; CPU avoids the flaky-GPU risk
    xs, ts = make_dataset(t0=t0, t1=t1, batch_size=batch_size, noise_std=noise_std,
                          num_steps=num_steps, sigma=sigma, a=a, train_dir=train_dir, device=dev)
    dt = float((ts[1] - ts[0]).item())
    z_grid = torch.linspace(-zmax, zmax, Nz, device=dev)
    f_true = a * (z_grid - z_grid ** 3)
    supp = z_grid.abs() <= 2.0

    f_net = mlp([1] + [64] * 3 + [1])                       # f_theta; zero last layer => start f=0
    torch.nn.init.zeros_(f_net[-1].weight); torch.nn.init.zeros_(f_net[-1].bias)
    opt = torch.optim.Adam(f_net.parameters(), lr=lr)

    def f_on_grid():
        with torch.no_grad():
            return f_net(z_grid.unsqueeze(-1)).squeeze(-1)

    history = []
    for em in range(n_em):
        # ---- E-step: exact grid smoother + FFBS path samples, using the CURRENT f
        K = transition_power(build_transition(f_on_grid(), z_grid, dt / n_sub, sigma), n_sub)
        filtered, _ = forward_backward(xs, z_grid, K, noise_std)
        paths = ffbs_sample(z_grid, K, filtered, n_samp)    # (T, B, n_samp) posterior latent paths
        z = paths[:-1].reshape(-1)
        dzdt = ((paths[1:] - paths[:-1]) / dt).reshape(-1)
        # ---- M-step: regress f_theta on the path increments (warm-started)
        for _ in range(m_steps):
            opt.zero_grad()
            loss = ((f_net(z.unsqueeze(-1)).squeeze(-1) - dzdt) ** 2).mean()
            loss.backward(); opt.step()
        l2 = (f_on_grid()[supp] - f_true[supp]).pow(2).mean().sqrt().item()
        history.append(l2)
        logging.warning(f"EM iter {em:02d}:  drift L2 err on [-2,2] = {l2:.4f}")

    # plot: drift L2 vs EM iteration, and final f_theta vs truth
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(range(n_em), history, "o-")
    ax[0].axhline(0.67, ls=":", c="gray", label="one-shot (f=0 smoother) 0.67")
    ax[0].set_xlabel("EM iteration"); ax[0].set_ylabel("drift L2 err on $[-2,2]$")
    ax[0].set_title("EM tightens the drift"); ax[0].legend(fontsize=9)
    zg = z_grid.numpy()
    ax[1].plot(zg, f_true.numpy(), "k-", lw=2.5, label="true $a(z-z^3)$")
    ax[1].plot(zg, f_on_grid().numpy(), "C2--", lw=2, label=f"$f_\\theta$ after EM (L2={history[-1]:.3f})")
    ax[1].set_xlim(-2.5, 2.5); ax[1].set_ylim(-4, 4)
    ax[1].set_xlabel("$z$"); ax[1].set_ylabel("$f(z)$")
    ax[1].set_title("learned drift after EM"); ax[1].legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(train_dir, "em_drift.pdf")
    plt.savefig(out); plt.close()
    # save the learned drift MLP (same architecture as DriftNet.net) for the operator to use
    ckpt = os.path.join(train_dir, "drift.pt")
    torch.save(f_net.state_dict(), ckpt)
    logging.warning(f"drift L2: {history[0]:.4f} (iter 0) -> {history[-1]:.4f} (iter {n_em-1}); "
                    f"wrote {out} and {ckpt}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    fire.Fire(main)
