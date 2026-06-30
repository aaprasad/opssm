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

"""Learnable SDE coefficients: DriftNet f(z) and DiffusionNet g(z), with jvp derivatives."""

import math

import torch
from torch import nn
from torch.func import jvp

from opssm.models.nn import mlp


class DriftNet(nn.Module):
    """Learnable drift f_theta: z -> R, with f and f' = d_z f BOTH by autodiff (jvp), so
    it is a drop-in for the analytic `drift(z) -> (f, df)` callable the Zakai residual
    expects. Mesh-free: queryable at any sampled z. Last layer zero-initialized so f ~ 0
    at the start (pure diffusion) and grows to fit the data."""

    def __init__(self, hidden=64, layers=3):
        super().__init__()
        self.net = mlp([1] + [hidden] * layers + [1])
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def drift(self, z):
        """z (any shape) -> f(z), f'(z), each z.shape (autodiff in z, trainable in theta)."""
        zin = z.reshape(-1, 1)
        f, df = jvp(self.net, (zin,), (torch.ones_like(zin),))
        return f.reshape(z.shape), df.reshape(z.shape)


class DiffusionNet(nn.Module):
    """Learnable diffusion via the FACTOR g(z) = net(z), with g^2(z) = g(z) . g(z) >= 0 by
    construction. In higher-D the same net outputs the diffusion factor G(z) (d x m) and the
    Fokker-Planck diffusion matrix is D = G G^T (symmetric PSD by construction) -- which is why
    we learn the factor and multiply, not the covariance directly. Returns g^2, (g^2)' and
    (g^2)'' by autodiff (jvp; (g^2)'' forward-over-forward), all needed because a state-dependent
    g makes the FP diffusion term
        1/2 d^2_z(g^2 rho) = 1/2 (g^2)'' rho + (g^2)' d_z rho + 1/2 g^2 d^2_z rho ,
    not just 1/2 g^2 d^2_z rho. Last layer init so g ~ g_init (flat); the M-step fits g^2 to the
    conditional variance of the increments. Drop-in for the scalar sigma**2 in the residual."""

    def __init__(self, hidden=64, layers=3, g_init=1.0):
        super().__init__()
        self.net = mlp([1] + [hidden] * layers + [1])
        nn.init.zeros_(self.net[-1].weight)
        self.net[-1].bias.data.fill_(g_init)            # g(z) ~ g_init (flat) => g^2 ~ g_init^2

    def _g2(self, zin):
        g = self.net(zin)
        return g * g                                    # g^2 = g . g  (-> G G^T in higher-D)

    def diffusion(self, z):
        """z (any shape) -> g^2(z), (g^2)'(z), (g^2)''(z), each z.shape (autodiff in z)."""
        zin = z.reshape(-1, 1)
        e = torch.ones_like(zin)
        g2, dg2 = jvp(self._g2, (zin,), (e,))
        _, d2g2 = jvp(lambda x: jvp(self._g2, (x,), (e,))[1], (zin,), (e,))
        return g2.reshape(z.shape), dg2.reshape(z.shape), d2g2.reshape(z.shape)
