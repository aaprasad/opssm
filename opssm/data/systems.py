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

"""Registry of latent dynamical systems (the SDE drift f(z)) for the benchmarks.

Each system defines its latent DIMENSION and its drift `f(z) -> (...,d)`; both the data generator and the
true-drift METRIC (filter_module) pull the ground-truth drift from here, so a benchmark is defined in one
place. Mirrors the encoder registry (opssm/models/encoders.py). Diffusion is isotropic scalar `sigma` for
now (a d x m factor G(z) is the later generalization).

    doublewell (1-D): dz = a (z - z^3) dt + sigma dW                     -- bistable (existing benchmark)
    vanderpol  (2-D): dz = [v, mu(1-x^2)v - x] dt + sigma dW             -- limit cycle
    lorenz     (3-D): dz = [s(y-x), x(r-z)-y, xy - b z] dt + sigma dW    -- chaotic attractor
"""
import math

import torch

SYSTEMS = {}  # name -> drift fn (with .dim, .defaults attached)


def register_system(name, dim, defaults):
    def deco(fn):
        fn.dim = dim
        fn.defaults = defaults
        SYSTEMS[name] = fn
        return fn
    return deco


@register_system("doublewell", dim=1, defaults={"a": 1.0})
def doublewell(z, a=1.0):
    """z (...,1) -> (...,1). Bistable double well."""
    return a * (z - z ** 3)


@register_system("vanderpol", dim=2, defaults={"mu": 1.5})
def vanderpol(z, mu=1.5):
    """z=(x,v) (...,2) -> (...,2). Van der Pol limit cycle."""
    x, v = z[..., 0], z[..., 1]
    return torch.stack([v, mu * (1.0 - x ** 2) * v - x], dim=-1)


@register_system("vanderpol_duncker", dim=2, defaults={"tau": 10.0, "mu": 2.0})
def vanderpol_duncker(z, tau=10.0, mu=2.0):
    """z=(x1,x2) (...,2) -> (...,2). Duncker et al. 2019 Lienard-form VdP (their Eq. 25, rho==mu):
    f1 = tau*mu*(x1 - x1^3/3 - x2),  f2 = tau*(x1/mu). tau multiplies -> FAST relaxation oscillator."""
    x1, x2 = z[..., 0], z[..., 1]
    return torch.stack([tau * mu * (x1 - x1 ** 3 / 3 - x2), tau * (x1 / mu)], dim=-1)


@register_system("lorenz", dim=3, defaults={"s": 10.0, "r": 28.0, "b": 8.0 / 3.0})
def lorenz(z, s=10.0, r=28.0, b=8.0 / 3.0):
    """z=(x,y,w) (...,3) -> (...,3). Lorenz attractor (chaotic)."""
    x, y, w = z[..., 0], z[..., 1], z[..., 2]
    return torch.stack([s * (y - x), x * (r - w) - y, x * y - b * w], dim=-1)


def make_drift(name, **params):
    """Return (drift_fn: (...,d)->(...,d) with params baked in, latent dim d)."""
    if name not in SYSTEMS:
        raise KeyError(f"unknown system {name!r}; registered: {sorted(SYSTEMS)}")
    fn = SYSTEMS[name]
    p = {**fn.defaults, **params}
    return (lambda z: fn(z, **p)), fn.dim


@torch.no_grad()
def simulate(name, batch_size, num_steps, dt, sigma, n_sub=5, params=None,
             init_std=1.0, burn_in=0, x0_uniform=None, device="cpu", seed=0):
    """Euler-Maruyama a latent system -> z (num_steps, batch_size, d). Isotropic diffusion `sigma`.
    `burn_in` extra sub-integrated steps before t=0 (lets chaotic/limit-cycle systems reach their
    attractor before the recorded window)."""
    drift, d = make_drift(name, **(params or {}))
    gen = torch.Generator(device=device).manual_seed(seed)
    sub = dt / n_sub
    rt = math.sqrt(sub)

    def step(zz):
        for _ in range(n_sub):
            zz = zz + drift(zz) * sub + sigma * rt * torch.randn(batch_size, d, device=device, generator=gen)
        return zz

    if x0_uniform is not None:                                        # x0 ~ U[-x0_uniform, x0_uniform]^d (Duncker VdP)
        zz = (torch.rand(batch_size, d, device=device, generator=gen) * 2 - 1) * x0_uniform
    else:
        zz = init_std * torch.randn(batch_size, d, device=device, generator=gen)
    for _ in range(burn_in):
        zz = step(zz)
    z = torch.zeros(num_steps, batch_size, d, device=device)
    z[0] = zz
    for i in range(1, num_steps):
        z[i] = step(z[i - 1].clone())
    return z


@torch.no_grad()
def linear_sensor(z, obs_dim, noise_std, c_scale=1.0, device="cpu", seed=0):
    """High-D linear observation of a d-D latent: y = C z + d + noise, C (obs_dim, d), d (obs_dim).
    z (T,B,d) -> y (T,B,obs_dim). Returns y, C, d_off. (C is a raw Gaussian matrix; the model's M-step
    imposes the Stiefel/orthonormal structure, as in the 1-D high-D benchmark.)"""
    gen = torch.Generator(device=device).manual_seed(seed)
    d_lat = z.shape[-1]
    C = c_scale * torch.randn(obs_dim, d_lat, device=device, generator=gen)
    d_off = torch.randn(obs_dim, device=device, generator=gen)
    y = torch.einsum("od,tbd->tbo", C, z) + d_off
    y = y + noise_std * torch.randn(*z.shape[:2], obs_dim, device=device, generator=gen)
    return y, C, d_off
