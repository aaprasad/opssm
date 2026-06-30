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

"""OperatorFilter: DeepONet conditional log-density (the core mesh-free Zakai operator)."""

import torch
from torch import nn
from torch.func import jvp

from opssm.models.nn import mlp


class OperatorFilter(nn.Module):
    """x_{0:T} -> log filtering density ell(z, s), a DeepONet conditioned on a causal
    observation context, mesh-free in z and CONTINUOUS in the within-interval time
    s = tau/dt in [0,1] (tau = time since the last observation):

        ell_i(z, s) = bias + sum_p b_p(c_i, s) * trunk_p(z)

    The state basis trunk(z) (p functions of z) is fixed; the per-step coefficients
    b(c_i, s) FLOW with s (a Galerkin-in-z, evolve-in-time DeepONet -- the branch takes
    the time, so each observation step gets its own coefficient trajectory). s=0 is the
    post-update filtering density at obs i, s=1 the Fokker-Planck-predicted density just
    before obs i+1. ell is differentiable in s by AUTODIFF (d_s ell = d_s b . trunk),
    so the continuous-time Zakai PINN enforces the FP evolution with no time-stepping /
    Euler (see pinn_zakai_loss)."""

    def __init__(self, data_size=1, gru_hidden=64, ctx_dim=64, p=64,
                 branch_hidden=128, trunk_hidden=64, trunk_layers=3):
        super().__init__()
        # input = [obs (zeroed where missing), observed-mask] so the GRU knows when
        # to update vs predict-only through an observation gap.
        self.gru = nn.GRU(input_size=data_size + 1, hidden_size=gru_hidden)
        self.to_ctx = nn.Linear(gru_hidden, ctx_dim)
        self.branch = mlp([ctx_dim + 1, branch_hidden, p])           # (context, time) -> coeffs
        self.trunk = mlp([1] + [trunk_hidden] * trunk_layers + [p])  # query z -> state basis
        self.bias = nn.Parameter(torch.zeros(()))

    def context(self, xs, mask):
        """xs (T,B,M), mask (T,B,1) observed-indicator -> causal context (T,B,C)."""
        inp = torch.cat([xs * mask, mask], dim=-1)         # (T,B,M+1)
        h, _ = self.gru(inp)                               # (T,B,gru_hidden)
        return self.to_ctx(h)                              # (T,B,C)

    def coeffs(self, ctx, s):
        """Branch coefficients b(c, s) at within-interval times s (Ns,) -> (T,B,Ns,p)."""
        ce = ctx.unsqueeze(2).expand(*ctx.shape[:2], s.numel(), ctx.shape[-1])
        se = s.reshape(1, 1, -1, 1).expand(*ctx.shape[:2], s.numel(), 1)
        return self.branch(torch.cat([ce, se], dim=-1))

    def coeffs_dtime(self, ctx, s):
        """b(c, s) AND d_s b(c, s) by autodiff (forward-mode jvp in the time input) ->
        each (T,B,Ns,p). d_s b is the time derivative used in the Zakai PDE residual."""
        ce = ctx.unsqueeze(2).expand(*ctx.shape[:2], s.numel(), ctx.shape[-1])
        se = s.reshape(1, 1, -1, 1).expand(*ctx.shape[:2], s.numel(), 1)
        inp = torch.cat([ce, se], dim=-1)                  # (T,B,Ns,C+1)
        tan = torch.zeros_like(inp); tan[..., -1] = 1.0    # tangent in the time input
        return jvp(self.branch, (inp,), (tan,))            # b, d_s b

    def trunk_zderivs(self, z):
        """State basis trunk(z) and its z, zz derivatives by autodiff. z any shape
        (mesh-free: sampled points, not a grid) -> tau, d_z, d2_z each (*z.shape, p)."""
        zin = z.reshape(-1, 1)
        e = torch.ones_like(zin)
        tau, dz = jvp(self.trunk, (zin,), (e,))
        _, d2z = jvp(lambda x: jvp(self.trunk, (x,), (e,))[1], (zin,), (e,))
        shp = (*z.shape, -1)
        return tau.reshape(shp), dz.reshape(shp), d2z.reshape(shp)

    def log_density(self, ctx, z, s=0.0):
        """ctx (T,B,C), z (Nz,) at within-interval time s (scalar) -> ell (T,B,Nz)."""
        s_t = torch.as_tensor([s], dtype=z.dtype, device=z.device)
        b = self.coeffs(ctx, s_t)[:, :, 0]                 # (T,B,p)
        return torch.einsum("tbp,zp->tbz", b, self.trunk(z.unsqueeze(-1))) + self.bias

    def log_posterior(self, xs, mask, z):
        """Normalized filtering log-posterior on z (post-update, s=0): (T,B,Nz)."""
        ell = self.log_density(self.context(xs, mask), z, s=0.0)
        return ell - torch.logsumexp(ell, dim=-1, keepdim=True)
