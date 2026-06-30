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

"""Diagnostic: is the drift f recoverable by REGRESSING it from the inferred latent path?

This is a throwaway sanity check of the EM reframing -- it does NOT use the neural operator.
It uses a generic linear-Gaussian RTS smoother (random-walk prior + the true diffusion, but
NO knowledge of f) to denoise the noisy observations x_i = z_i + noise into a latent estimate
z_hat(t), then fits the drift by the conditional-mean regression

        f(z) ~ E[ (z_{t+1}-z_t)/dt | z_t ] .

The single increment Delta z = f dt + sqrt(sigma^2 dt) w is mostly DIFFUSION noise, so the drift
is invisible per-step -- but its CONDITIONAL MEAN over many increments averages that noise away
and recovers f. We also regress from the RAW (un-smoothed) observations as a control: the
observation noise correlates with the regressor and BIASES that fit, which is why denoising
(the E-step) is necessary. If the smoothed regression recovers a(z - z^3) while the marginal
filtering likelihood (Stage 3) could not, the conclusion is: f IS identifiable here -- we were
extracting it through the wrong channel.

To run:
python -m run.scripts.drift_from_smoothing
"""

import logging
import os

import fire
import matplotlib.pyplot as plt
import torch
import tqdm
from torch import nn

from run.scripts.latent_sde_double_well_zakai import make_dataset
from run.scripts.neural_zakai_filter import mlp


def rts_smoother(x, dt, sigma, noise_std, p0=1.0):
    """1D Kalman RTS smoother with a random-walk prior (f=0) + diffusion sigma and obs noise
    noise_std -- a GENERIC denoiser that does NOT know the true drift. x (T,B) -> z_hat (T,B)
    smoothed posterior mean."""
    T, B = x.shape
    Q = sigma ** 2 * dt
    R = noise_std ** 2
    zf = torch.zeros(T, B); Pf = torch.zeros(T, B)          # filtered mean / var
    zp = torch.zeros(T, B); Pp = torch.zeros(T, B)          # one-step-prediction mean / var
    z_pred = torch.zeros(B); P_pred = torch.full((B,), p0)
    for i in range(T):
        zp[i] = z_pred; Pp[i] = P_pred
        K = P_pred / (P_pred + R)                           # Kalman gain
        zf[i] = z_pred + K * (x[i] - z_pred)
        Pf[i] = (1 - K) * P_pred
        z_pred = zf[i]; P_pred = Pf[i] + Q                  # predict (random walk)
    zs = zf.clone()                                         # RTS backward pass
    for i in range(T - 2, -1, -1):
        C = Pf[i] / Pp[i + 1]
        zs[i] = zf[i] + C * (zs[i + 1] - zp[i + 1])
    return zs


def binned_drift(z, dzdt, edges):
    """Nonparametric f estimate: mean of dz/dt within each z-bin -> (centers, f_hat)."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = torch.bucketize(z, edges) - 1
    f_hat = torch.full((len(centers),), float("nan"))
    for b in range(len(centers)):
        m = idx == b
        if m.sum() > 20:
            f_hat[b] = dzdt[m].mean()
    return centers, f_hat


def fit_mlp_drift(z, dzdt, iters=3000, lr=2e-3, hidden=64):
    """Smooth regression f_theta(z) ~ dz/dt by least squares (the EM M-step, done directly)."""
    net = mlp([1] + [hidden] * 3 + [1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    z_in = z.unsqueeze(-1); target = dzdt.unsqueeze(-1)
    for _ in tqdm.tqdm(range(iters)):
        opt.zero_grad()
        loss = ((net(z_in) - target) ** 2).mean()
        loss.backward(); opt.step()
    return net


def main(batch_size=512, t0=0.0, t1=10.0, num_steps=100, a=1.0, sigma=0.6,
         noise_std=0.1, zmax=2.5, train_dir="./dump/drift_from_smoothing/"):
    torch.manual_seed(0)
    os.makedirs(train_dir, exist_ok=True)
    xs, ts = make_dataset(t0=t0, t1=t1, batch_size=batch_size, noise_std=noise_std,
                          num_steps=num_steps, sigma=sigma, a=a, train_dir=train_dir,
                          device=torch.device("cpu"))
    dt = float((ts[1] - ts[0]).item())
    x = xs[..., 0]                                          # (T,B) noisy observations

    z_hat = rts_smoother(x, dt, sigma, noise_std)           # E-step: denoise -> latent estimate

    # M-step regression pairs (z_t, (z_{t+1}-z_t)/dt), from smoothed vs raw observations
    def pairs(z):
        return z[:-1].reshape(-1), ((z[1:] - z[:-1]) / dt).reshape(-1)
    zs_z, zs_dz = pairs(z_hat)                              # from the smoothed path
    raw_z, raw_dz = pairs(x)                                # control: from raw noisy obs

    edges = torch.linspace(-zmax, zmax, 26)
    c_s, f_s = binned_drift(zs_z, zs_dz, edges)             # nonparametric f, smoothed
    c_r, f_r = binned_drift(raw_z, raw_dz, edges)           # nonparametric f, raw (biased)
    net = fit_mlp_drift(zs_z, zs_dz)                        # smooth f_theta from smoothed path

    # L2 error vs the truth on the support [-2, 2]
    zg = torch.linspace(-2, 2, 200)
    f_true = a * (zg - zg ** 3)
    with torch.no_grad():
        f_mlp = net(zg.unsqueeze(-1)).squeeze(-1)
    l2_mlp = (f_mlp - f_true).pow(2).mean().sqrt().item()
    ok = ~f_s.isnan() & (c_s.abs() <= 2)
    l2_binned = (f_s[ok] - a * (c_s[ok] - c_s[ok] ** 3)).pow(2).mean().sqrt().item()
    logging.warning(f"drift L2 err on [-2,2]:  binned(smoothed)={l2_binned:.4f}   "
                    f"MLP(smoothed)={l2_mlp:.4f}   (Stage-3 filtering-likelihood plateaued ~1.2-1.4)")

    zgn = zg.numpy()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(z_hat[:, 0].numpy(), label="smoothed $\\hat z$", lw=2)
    ax[0].plot(x[:, 0].numpy(), ".", ms=3, alpha=0.5, label="noisy obs $x$")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("$z$"); ax[0].set_title("denoising (traj 0)")
    ax[0].legend(fontsize=9)
    ax[1].plot(zgn, f_true.numpy(), "k-", lw=2.5, label="true $a(z-z^3)$")
    ax[1].plot(zgn, f_mlp.numpy(), "C2--", lw=2, label=f"MLP from $\\hat z$ (L2={l2_mlp:.3f})")
    ax[1].plot(c_s.numpy(), f_s.numpy(), "C2o", ms=5, label="binned from $\\hat z$")
    ax[1].plot(c_r.numpy(), f_r.numpy(), "C3x", ms=5, label="binned from raw $x$ (biased)")
    ax[1].set_xlabel("$z$"); ax[1].set_ylabel("$f(z)$"); ax[1].set_ylim(-4, 4); ax[1].set_xlim(-2.5, 2.5)
    ax[1].set_title("drift regressed from the inferred latent path"); ax[1].legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(train_dir, "drift_from_smoothing.pdf")
    plt.savefig(out); plt.close()
    logging.warning(f"wrote {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    fire.Fire(main)
