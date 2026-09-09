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

"""OperatorFilter + OperatorBackward: DeepONet conditional log-density (the mesh-free Zakai forward
filter, and its anti-causal backward-message twin for the smoother)."""

import torch
from torch import nn
from torch.func import jvp, vmap

from opssm.models.encoders import make_encoder
from opssm.models.nn import mlp


class OperatorFilter(nn.Module):
    """x_{0:T} -> log filtering density ell(z, s), a DeepONet conditioned on a causal
    observation context, mesh-free in z and CONTINUOUS in the within-interval time
    s = tau/dt in [0,1] (tau = time since the last observation):

        ell_i(z, s) = bias + sum_p b_p(c_i, s) * trunk_p(z)

    Softplus trunk activations allow unbounded log-densities. Tanh remains available for legacy runs.

    The state basis trunk(z) (p functions of z) is fixed; the per-step coefficients
    b(c_i, s) FLOW with s (a Galerkin-in-z, evolve-in-time DeepONet -- the branch takes
    the time, so each observation step gets its own coefficient trajectory). s=0 is the
    post-update filtering density at obs i, s=1 the Fokker-Planck-predicted density just
    before obs i+1. ell is differentiable in s by AUTODIFF (d_s ell = d_s b . trunk),
    so the continuous-time Zakai PINN enforces the FP evolution with no time-stepping /
    Euler (see pinn_zakai_loss)."""

    def __init__(self, data_size=1, gru_hidden=64, ctx_dim=64, p=64,
                 branch_hidden=128, trunk_hidden=64, trunk_layers=3,
                 encoder="gru", encoder_kwargs=None, reverse=False, latent_dim=1,
                 trunk_activation='softplus'):
        super().__init__()
        self.latent_dim = latent_dim                                 # d: LATENT dim (separate from data_size = obs dim)
        activations = {'tanh': nn.Tanh, 'softplus': nn.Softplus}
        if trunk_activation not in activations:
            raise ValueError('trunk_activation must be tanh or softplus')
        self.trunk_activation = trunk_activation
        # CAUSAL context encoder: packs [obs (zeroed where missing), observed-mask] -> (T,B,ctx_dim), so it
        # knows when to update vs predict-only through a gap. Pluggable (encoder=gru|tcn|transformer|...);
        # built FIRST so the default gru keeps the original init RNG order (byte-identical). `gru_hidden`
        # doubles as the generic encoder hidden width.
        enc_kwargs = {"hidden": gru_hidden, **(encoder_kwargs or {})}   # encoder_kwargs may override hidden
        self.encoder = make_encoder(encoder, data_size + 1, ctx_dim, **enc_kwargs)
        self.reverse = reverse                                       # True = anti-causal (backward twin)
        self.branch = mlp([ctx_dim + 1, branch_hidden, p])           # (context, time) -> coeffs
        self.trunk = mlp([latent_dim] + [trunk_hidden] * trunk_layers + [p],
                         act=activations[trunk_activation])  # query z (d) -> state basis
        self.bias = nn.Parameter(torch.zeros(()))

    def context(self, xs, mask):
        """xs (T,B,M), mask (T,B,1) observed-indicator -> context (T,B,C). reverse=False: CAUSAL, ctx[t]
        summarizes y_{0:t}. reverse=True (backward twin): ANTI-CAUSAL via flip -> causal encoder -> flip,
        so ctx[t] summarizes y_{t:T} (inclusive) -- generic over ANY causal encoder. Obs are standardized at
        the DATALOADER level (see the datamodule), so the encoder sees ~unit-scale input."""
        inp = torch.cat([xs * mask, mask], dim=-1)         # (T,B,M+1)
        if self.reverse:
            return self.encoder(inp.flip(0)).flip(0)       # (T,B,C); ctx[t] <- y_{t:T}
        return self.encoder(inp)                           # (T,B,C); ctx[t] <- y_{0:t}

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
        """State basis trunk(z) with its GRADIENT + LAPLACIAN by autodiff (mesh-free; z (...,d)) ->
        tau (...,p), grad_tau (...,d,p) [d_i tau], lap_tau (...,p) [sum_i d_ii tau]. FUSED
        forward-over-forward jvp along each unit tangent e_i yields BOTH d_i tau and d_ii tau; VMAPPED over
        the d tangents (one batched op instead of a Python loop -- fewer kernel launches, d x peak memory).
        Cost ~4d forward passes, independent of p -- vs jacrev/hessian which scale with p, p*d."""
        d = z.shape[-1]
        zin = z.reshape(-1, d)                                        # (N,d)
        eye = torch.eye(d, device=z.device, dtype=z.dtype)           # (d,d) unit tangents

        def along(e):                                               # e (d,) -> derivs along axis e
            v = e.expand_as(zin)                                    # (N,d) tangent
            (tau, di), (_, dii) = jvp(lambda x: jvp(self.trunk, (x,), (v,)), (zin,), (v,))
            return tau, di, dii                                    # each (N,p)

        tau, grad, lap_ax = vmap(along)(eye)                        # (d,N,p) each; tau identical over tangents
        shp = z.shape[:-1]
        return (tau[0].reshape(*shp, -1),                          # tau: take one copy
                grad.movedim(0, -2).reshape(*shp, d, -1),          # (d,N,p) -> (N,d,p)
                lap_ax.sum(0).reshape(*shp, -1))                   # sum axes -> Laplacian (N,p)

    def trunk_zderivs_dirs(self, z, dirs):
        """State basis with its full GRADIENT and the DIRECTIONAL second derivatives summed over `dirs`.

        z (...,d), dirs (m,d) [ROW k = the k-th direction v_k] ->
            tau (...,p), grad_tau (...,d,p) [d_i tau], sec_tau (...,p) [sum_k v_k^T H(tau) v_k].

        The anisotropic-diffusion generalization of `trunk_zderivs`. For a diffusion matrix
        Sigma = L L^T, taking `dirs = L.T` (rows = COLUMNS of L) gives exactly

            sec_tau = sum_k (L[:,k])^T H L[:,k] = tr(L^T H L) = tr(Sigma H),

        the weighted Hessian trace the Fokker-Planck operator needs -- at the SAME cost as the
        isotropic Laplacian (d directions), not the d(d+1)/2 of a full Hessian. With L = g I this
        reduces to g^2 * (Laplacian), so the isotropic path is recovered exactly.

        The gradient is taken separately along the UNIT axes (a single forward jvp each, ~half the
        cost of the nested one) because the drift term f . grad ell needs the gradient in the
        canonical frame, not the L frame -- recovering it by a triangular solve with L^-T would
        amplify error whenever L is ill-conditioned."""
        d = z.shape[-1]
        zin = z.reshape(-1, d)                                       # (N,d)
        eye = torch.eye(d, device=z.device, dtype=z.dtype)          # unit tangents
        tau, grads = vmap(lambda e: jvp(self.trunk, (zin,), (e.expand_as(zin),)))(eye)  # (d,N,p)

        def sec_along(v):                                           # v (d,) -> v^T H tau v  (N,p)
            vv = v.expand_as(zin)                                   # (N,d) tangent
            (_, _), (_, dvv) = jvp(lambda x: jvp(self.trunk, (x,), (vv,)), (zin,), (vv,))
            return dvv

        sec = vmap(sec_along)(dirs)                                 # (m,N,p)
        shp = z.shape[:-1]
        return (tau[0].reshape(*shp, -1),
                grads.movedim(0, -2).reshape(*shp, d, -1),          # (N,d,p)
                sec.sum(0).reshape(*shp, -1))                       # sum_k -> tr(Sigma H)  (N,p)

    def trunk_grad(self, z):
        """State basis trunk(z) with its GRADIENT only (no Laplacian) -> tau (...,p), grad_tau (...,d,p)
        [d_i tau]. Single forward jvp per unit tangent, VMAPPED over the d tangents (= forward-mode
        Jacobian) -- ~half the cost/memory of trunk_zderivs; used by the MALA readout (grad ell only)."""
        d = z.shape[-1]
        zin = z.reshape(-1, d)                                        # (N,d)
        eye = torch.eye(d, device=z.device, dtype=z.dtype)          # (d,d) unit tangents
        tau, grads = vmap(lambda e: jvp(self.trunk, (zin,), (e.expand_as(zin),)))(eye)   # (d,N,p) each
        shp = z.shape[:-1]
        return tau[0].reshape(*shp, -1), grads.movedim(0, -2).reshape(*shp, d, -1)

    def log_density(self, ctx, z, s=0.0):
        """ctx (T,B,C), z (Nz,) [1-D grid] or (Nz,d) at within-interval time s (scalar) -> ell (T,B,Nz)."""
        s_t = torch.as_tensor([s], dtype=z.dtype, device=z.device)
        b = self.coeffs(ctx, s_t)[:, :, 0]                 # (T,B,p)
        zt = z.unsqueeze(-1) if z.dim() == 1 else z        # (Nz,d)
        return torch.einsum("tbp,zp->tbz", b, self.trunk(zt)) + self.bias

    def log_posterior(self, xs, mask, z):
        """Normalized filtering log-posterior on z (post-update, s=0): (T,B,Nz)."""
        ell = self.log_density(self.context(xs, mask), z, s=0.0)
        return ell - torch.logsumexp(ell, dim=-1, keepdim=True)


class OperatorBackward(OperatorFilter):
    """ANTI-CAUSAL twin of OperatorFilter for the BACKWARD adjoint-Zakai smoother. Identical DeepONet
    machinery (trunk / branch / coeffs / coeffs_dtime / trunk_zderivs) with its OWN weights; only the
    context differs -- anti-causal, via reverse=True (flip -> causal encoder -> flip, so ctx[t] summarizes
    y_{t:T} INCLUDING obs t). log_density(ctx_b, z, s=0) = log msg_t(z), the (unnormalized) post-update
    backward MESSAGE msg_t(z) = p(y_{t:T} | z_t) (mirroring the forward post-update pi_t = p(z_t | y_{0:t});
    s>0 transports it backward (adjoint FP) toward obs t-1, and the jump ties it to msg_{t-1} via lik_{t-1}
    -- see pinn_adjoint_loss). The smoothed posterior is gamma_t(z) = p(z_t | y_{0:T}) proportional to
    alpha_t(z) * beta_t(z) = alpha_t * msg_t / lik_t, i.e. softmax_z( ell_forward + log msg - loglik_t )
    (see mstep.log_smoothed)."""

    def __init__(self, *args, **kwargs):
        kwargs["reverse"] = True                           # anti-causal: context = flip -> encoder -> flip
        super().__init__(*args, **kwargs)

    def log_msg(self, xs, mask, z):
        """Normalized backward-message log-density at s=0 on grid z (for viz; the normalizer is
        arbitrary for smoothing -- it cancels in the gamma = alpha * beta softmax)."""
        lm = self.log_density(self.context(xs, mask), z, s=0.0)
        return lm - torch.logsumexp(lm, dim=-1, keepdim=True)
