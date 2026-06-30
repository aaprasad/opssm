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

"""Neural mesh-free Zakai filter -- STAGE 1: representability / amortization.

Can a NEURAL OPERATOR amortize the exact filtering posterior? With the dynamics
KNOWN, train an operator that maps an observation sequence to the filtering
log-density, supervised against the exact grid Zakai filter, and test whether it
(a) fits the posterior, (b) generalizes to HELD-OUT observation sequences in one
forward pass (amortization), (c) represents multimodality a Gaussian cannot.

Model (a DeepONet conditional density -- mesh-free, queryable at any z):
    c_t   = CausalGRU(x_{<=t})                 causal observation context
    ell(z,t) = branch(c_t) . trunk(z)          log-density (DeepONet), any z
    pi(.,t)  = softmax_z ell(.,t)              normalized filtering posterior

Target: the exact grid filter `forward_backward` (latent_sde_double_well_zakai.py)
with the TRUE drift -- the exact p(z_t | x_{0:t}) on a grid. Loss = KL(target || pi).

This is the cheap go/no-go for the whole approach (see run/notes/neural_zakai_filter_design.md):
if an operator can't even FIT + AMORTIZE the true filter here, stop. If it can, Stage 2
drops the supervision and learns the filter from data via the likelihood + Zakai residual.

To run:
python -m run.scripts.neural_zakai_filter
"""

import logging
import math
import os

import fire
import matplotlib.pyplot as plt
import torch
import tqdm
from torch import nn
from torch import optim
from torch.func import jvp

from run.scripts.latent_sde_double_well_zakai import (
    make_dataset,
    build_transition,
    transition_power,
    forward_backward,
)


# ---------------------------------------------------------------------------
# Exact grid Zakai filter (the supervision target), with KNOWN dynamics.
# ---------------------------------------------------------------------------
@torch.no_grad()
def grid_filter_target(xs, z_grid, a, sigma, noise_std, dt_obs, n_sub, mask=None):
    """Exact p(z_t | x_{0:t}) (filtered) and p(z_t | x_{0:T}) (smoothed) on the
    grid, using the TRUE drift f = a(z - z^3). `mask` (T,) bool: False = no
    observation at that step (predict-only) -> the posterior relaxes toward the
    bimodal stationary, which is where multimodality shows up. Returns (T,B,Nz)."""
    f_true = a * (z_grid - z_grid ** 3)
    K = build_transition(f_true, z_grid, dt_obs / n_sub, sigma)
    K_full = transition_power(K, n_sub)
    filtered, smoothed = forward_backward(xs, z_grid, K_full, noise_std, mask=mask)
    return filtered, smoothed


# ---------------------------------------------------------------------------
# Operator: causal encoder + DeepONet conditional log-density.
# ---------------------------------------------------------------------------
def mlp(sizes, act=nn.Tanh):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


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


def kl_target_pred(target, log_pred, eps=1e-12):
    """KL(target || pred), target a mass (T,B,Nz), log_pred the predicted log-mass."""
    t = target.clamp_min(eps)
    return (t * (t.log() - log_pred)).sum(dim=-1).mean()


def recursion_loss(model, xs, mask, z_grid, T_full, prior, noise_std, eps=1e-12):
    """STAGE 2 -- learn the filter WITHOUT the precomputed answer. The operator is
    trained to satisfy the filtering RECURSION itself (KNOWN dynamics): predict via
    the transition T_full (discretized Fokker-Planck), update by the observation
    likelihood, anchor t=0 with the prior. Also returns the data NLL (the normalizer/
    evidence) -- the quantity that will train the DYNAMICS in Stage 3."""
    log_pi = model.log_posterior(xs, mask, z_grid)         # (T,B,Nz)
    pi = log_pi.exp()
    x = xs[..., 0]                                         # (T,B)
    lik = torch.exp(-0.5 * (x.unsqueeze(-1) - z_grid) ** 2 / noise_std ** 2)  # (T,B,Nz)
    m = mask[..., 0:1]                                     # (T,B,1) observed (1) / gap (0)

    # predict: pi_pred[t] = pi[t-1] @ T_full.t() (bootstrapped, detached); pi_pred[0]=prior
    pi_pred = torch.empty_like(pi)
    pi_pred[0] = prior
    pi_pred[1:] = pi[:-1].detach() @ T_full.t()
    # update where observed (lik**m is the identity in the gap), normalize -> target
    upd = (lik ** m) * pi_pred
    pi_tgt = (upd / upd.sum(-1, keepdim=True).clamp_min(eps)).detach()
    loss = -(pi_tgt * log_pi).sum(-1).mean()              # cross-entropy to recursion target

    # data NLL = -sum_t log c_t, evidence c_t = integral lik_t * pi_pred_t (observed steps)
    c = (lik * pi_pred).sum(-1).clamp_min(eps)            # (T,B)
    nll = -(torch.log(c) * m[..., 0]).sum(0).mean()
    return loss, nll


def sample_collocation(xs, mask, n_colloc, near_std, broad_std, center=None):
    """MESH-FREE collocation: draw state points from a data-following proposal q (no grid,
    no zmax). Half are near the observation (the peaked observed posterior), half broad
    (covering the wells / the gap's bimodality where there is no observation). Returns
    z (T,B,K) sampled points and log_q (T,B,K) the proposal log-density (the importance
    weights for the SNIS normalizer). In high-D this q is the importance-sampling lever
    that replaces the O(N^d) grid. `center` (T,B): the LATENT location to draw the near
    component around; default xs[...,0] (1D direct obs). For high-D obs y, pass a z-estimate
    (e.g. the pseudo-inverse C^+ (y-d)) since the obs itself is not in latent space."""
    T, B = xs.shape[0], xs.shape[1]
    dev = xs.device
    obs = mask[..., 0]                                          # (T,B) observed indicator
    center = (xs[..., 0] if center is None else center) * obs   # latent estimate where observed
    # "near" component: tight around the obs where observed. In the GAP there is no obs and
    # the posterior is BIMODAL (mass at the wells, not at 0), so widen the near component to
    # broad there -- otherwise half the samples pile up at z=0, exactly where the gap
    # posterior is empty, and the modes stay under-sampled.
    near_sd = (near_std * obs + broad_std * (1.0 - obs)).unsqueeze(-1)   # (T,B,1) per step
    Kn = n_colloc // 2
    Kb = n_colloc - Kn
    near = center.unsqueeze(-1) + near_sd * torch.randn(T, B, Kn, device=dev)
    broad = broad_std * torch.randn(T, B, Kb, device=dev)
    z = torch.cat([near, broad], dim=-1)                        # (T,B,K)
    c2 = math.log(2.0) + 0.5 * math.log(2 * math.pi)
    lqn = -0.5 * ((z - center.unsqueeze(-1)) / near_sd) ** 2 - near_sd.log() - c2
    lqb = -0.5 * (z / broad_std) ** 2 - math.log(broad_std) - c2
    log_q = torch.logaddexp(lqn, lqb)                           # log(0.5 q_near + 0.5 q_broad)
    return z, log_q


def pinn_zakai_loss(model, xs, mask, z_col, log_q, s_coll, drift, sigma, log_prior,
                    noise_std, dt, n_tcoll=None, res_post=0.0, decode=None):
    """MESH-FREE continuous-time Zakai PINN -- no grid, no time-stepping, no Euler.
    Collocation points z_col (T,B,K) are SAMPLED from the proposal (log_q its log-density);
    the normalizer Z and evidence c are self-normalized importance-sampling (SNIS) estimates
    over them; the drift f, f' are evaluated ANALYTICALLY at the samples (drift(z)->f,df).
        predict (PINN):  d_tau ell = -(f' + f d_z ell) + 1/2 g^2 ((d_z ell)^2 + d2_z ell)
        update  (jump):  pi_{i+1}(.,0) = normalize( lik_{i+1} * pi_i(.,dt) )   [SNIS]
        anchor  (IC):    pi_0(.,0)     = normalize( lik_0 * prior )            [SNIS]
    ell_i(z,s) = b(c_i,s).trunk(z); d_z ell = b.d_z tau, d_s ell = (d_s b).tau, all exact
    autodiff (jvp). pi(.,s) is SNIS weights W = softmax(ell - log_q). d_tau = d_s/dt.
    The recursion/SNIS terms use only the cheap basis tau (a forward, no autodiff) at ALL
    steps; the costly per-sample Jacobians (d_z, d2_z, d_s) for the FP residual are taken
    on a random subsample of n_tcoll steps (stochastic time collocation), so the residual's
    cost is decoupled from T. Returns (residual_loss, jump_loss, ic_loss, data_nll)."""
    T, B, K = z_col.shape
    ctx = model.context(xs, mask)                                  # (T,B,C)

    # ---- cheap path (no autodiff): basis at all steps; coeffs at the interval ends s=0,1
    tau = model.trunk(z_col.reshape(-1, 1)).reshape(T, B, K, -1)   # (T,B,K,p)
    b_ends = model.coeffs(ctx, torch.tensor([0.0, 1.0], device=z_col.device))   # (T,B,2,p)
    ell0 = torch.einsum("tbp,tbkp->tbk", b_ends[:, :, 0], tau) + model.bias   # post-update (s=0)

    if decode is None:                                                # 1D direct obs: h(z) = z
        loglik = -0.5 * (xs[..., 0].unsqueeze(-1) - z_col) ** 2 / noise_std ** 2   # (T,B,K)
    else:                                                             # high-D obs: lik = N(y; h(z), sigma^2 I)
        loglik = -0.5 * ((xs.unsqueeze(2) - decode(z_col)) ** 2).sum(-1) / noise_std ** 2   # (T,B,K)
    m = mask[..., 0]                                                   # (T,B)
    logZ0 = torch.logsumexp(ell0 - log_q, dim=-1, keepdim=True) - math.log(K)
    logpi0 = ell0 - logZ0                                              # normalized log-density

    # predict pi_i(.,dt) as SNIS weights, EVALUATED on step (i+1)'s samples (shared support
    # for the bootstrap): ell of step i (s=1) at z_col[i+1] = b_i(.,1) . trunk(z_col[i+1]).
    ellT_next = torch.einsum("tbp,tbkp->tbk", b_ends[:-1, :, 1], tau[1:]) + model.bias
    logW_pred = torch.log_softmax(ellT_next - log_q[1:], dim=-1).detach()
    logW_tgt = torch.log_softmax(logW_pred + loglik[1:] * m[1:].unsqueeze(-1), dim=-1)
    jump_loss = -(logW_tgt.exp() * logpi0[1:]).sum(-1).mean()
    logW_ic = torch.log_softmax(
        log_prior(z_col[0]) + loglik[0] * m[0].unsqueeze(-1) - log_q[0], dim=-1)
    ic_loss = -(logW_ic.exp() * logpi0[0]).sum(-1).mean()
    logc = torch.logsumexp(logW_pred + loglik[1:], dim=-1)            # evidence c_{i+1}
    nll = -(logc * m[1:]).sum(0).mean()

    # ---- residual path (autodiff) on a random time subsample
    ti = (torch.randperm(T, device=z_col.device)[:n_tcoll] if n_tcoll and n_tcoll < T
          else torch.arange(T, device=z_col.device))
    b_s, ds_b = model.coeffs_dtime(ctx[ti], s_coll)                   # (Ts,B,Ns,p)
    tau_s, dz_tau, d2z_tau = model.trunk_zderivs(z_col[ti])           # (Ts,B,K,p)
    dz_ell = torch.einsum("tbsp,tbkp->tbsk", b_s, dz_tau)
    d2z_ell = torch.einsum("tbsp,tbkp->tbsk", b_s, d2z_tau)
    ds_ell = torch.einsum("tbsp,tbkp->tbsk", ds_b, tau_s)
    f, df = drift(z_col[ti])                                          # analytic at samples
    f = f.unsqueeze(2); df = df.unsqueeze(2)                          # (Ts,B,1,K)
    if callable(sigma):                                              # state-dependent g^2(z)
        g2, dg2, d2g2 = sigma(z_col[ti])                             # (Ts,B,K) each
        g2 = g2.unsqueeze(2); dg2 = dg2.unsqueeze(2); d2g2 = d2g2.unsqueeze(2)
        # 1/2 d^2_z(g^2 rho) in log-space: 1/2 (g^2)'' + (g^2)' ell' + 1/2 g^2 (ell'^2 + ell'')
        rhs = (-(df + f * dz_ell)
               + 0.5 * d2g2 + dg2 * dz_ell + 0.5 * g2 * (dz_ell ** 2 + d2z_ell))
    else:                                                            # constant scalar g
        rhs = -(df + f * dz_ell) + 0.5 * sigma ** 2 * (dz_ell ** 2 + d2z_ell)
    res2 = (ds_ell / dt - rhs) ** 2                                   # (Ts,B,Ns,K)
    if res_post > 0:
        # weight the FP residual by the posterior mass: enforce the physics WHERE THE DENSITY
        # IS (the modes) rather than uniformly over the broad proposal -- targets the gap
        # over-dispersion. w = (1-res_post)/K + res_post * softmax(ell - log_q), sums to 1.
        W = torch.softmax(ell0[ti] - log_q[ti], dim=-1).detach()      # (Ts,B,K) posterior weights
        w = (1.0 - res_post) / K + res_post * W
        res_loss = (w.unsqueeze(2) * res2).sum(-1).mean()
    else:
        res_loss = res2.mean()
    return res_loss, jump_loss, ic_loss, nll


def accumulate_pinn_grads(model, xs, mask, s_coll, drift, sigma, log_prior, noise_std, dt,
                          n_colloc, near_std, broad_std, n_tcoll, chunk_size,
                          w_nll=0.0, num_steps=1, res_post=0.0, decode=None, center=None):
    """MEMORY-CAPPED forward+backward of the mesh-free Zakai PINN loss. The recursion is
    per-trajectory and the residual is per-point, both INDEPENDENT across the batch, so we
    split the batch into chunks of `chunk_size`, run pinn_zakai_loss + backward per chunk
    (freeing each chunk's autodiff graph before the next), and accumulate gradients. Peak
    memory scales with chunk_size rather than the full batch, and the result is EXACT (a
    batch-mean is the size-weighted mean of chunk-means). Set w_nll>0 to add the data-NLL
    term w_nll * NLL/num_steps (Stage 3, learn dynamics). Returns the batch-averaged
    (res, jump, ic, nll) scalars for logging; gradients are left accumulated on the params
    (caller does zero_grad before / step after)."""
    B = xs.shape[1]
    agg = [0.0, 0.0, 0.0, 0.0]
    for st in range(0, B, chunk_size):
        sl = slice(st, min(st + chunk_size, B))
        bw = (sl.stop - sl.start) / B                       # chunk's batch fraction
        ctr = None if center is None else center[:, sl]
        z_col, log_q = sample_collocation(xs[:, sl], mask[:, sl], n_colloc, near_std, broad_std,
                                          center=ctr)
        res, jump, ic, nll = pinn_zakai_loss(
            model, xs[:, sl], mask[:, sl], z_col, log_q, s_coll, drift, sigma, log_prior,
            noise_std, dt, n_tcoll=n_tcoll, res_post=res_post, decode=decode)
        (bw * (res + jump + ic + w_nll * nll / num_steps)).backward()
        for i, v in enumerate((res, jump, ic, nll)):
            agg[i] += bw * v.item()
    return agg


# ---------------------------------------------------------------------------
# Visualization: operator posterior vs exact grid filter on HELD-OUT sequences.
# ---------------------------------------------------------------------------
@torch.no_grad()
def vis(model, xs_val, mask_val, filt_val, z_grid, ts, img_path, n_traj=3):
    z = z_grid.cpu().numpy()
    ts_np = ts.cpu().numpy()
    log_pi = model.log_posterior(xs_val, mask_val, z_grid)  # (T,Bv,Nz)
    pi = log_pi.exp().cpu().numpy()
    filt = filt_val.cpu().numpy()
    data = xs_val.cpu().numpy()[..., 0]                     # (T,Bv)
    obs = mask_val[:, 0, 0].cpu().numpy().astype(bool)      # (T,) observed
    gap = ~obs

    # pick, per trajectory, the time of MAX spread in the exact filter (= where the
    # posterior goes bimodal, typically inside the observation gap).
    mean = (filt_val * z_grid).sum(-1)                     # (T,Bv)
    var = (filt_val * z_grid ** 2).sum(-1) - mean ** 2
    t_star = var[2:-2].argmax(dim=0).cpu().numpy() + 2     # (Bv,)

    fig, axes = plt.subplots(2, n_traj, figsize=(6 * n_traj, 9))
    for j in range(n_traj):
        tj = int(t_star[j])
        ax = axes[0, j]
        ax.plot(z, filt[tj, j], "k-", lw=2, label="exact filter")
        ax.plot(z, pi[tj, j], "C3--", lw=2, label="operator")
        if obs[tj]:
            ax.axvline(data[tj, j], color="C0", lw=1, ls=":", label="obs $x_t$")
        ax.set_xlabel("$z$"); ax.set_title(
            f"held-out traj {j}, $t={tj}$ ({'in gap' if gap[tj] else 'observed'})")
        if j == 0:
            ax.legend(fontsize=9)

        m_op = (log_pi[:, j].exp() * z_grid).sum(-1).cpu().numpy()
        s_op = ((log_pi[:, j].exp() * z_grid ** 2).sum(-1).cpu().numpy() - m_op ** 2).clip(0) ** 0.5
        m_ex = mean[:, j].cpu().numpy()
        ax = axes[1, j]
        if gap.any():
            ax.axvspan(ts_np[gap][0], ts_np[gap][-1], color="gray", alpha=0.15, label="no obs")
        ax.plot(ts_np[obs], data[obs, j], "C0.", ms=3, label="obs")
        ax.plot(ts_np, m_ex, "k-", lw=2, label="exact mean")
        ax.plot(ts_np, m_op, "C3--", lw=2, label="operator mean")
        ax.fill_between(ts_np, m_op - 2 * s_op, m_op + 2 * s_op, color="C3", alpha=0.2)
        ax.set_xlabel("$t$"); ax.set_ylabel("$z$")
        if j == 0:
            ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(img_path); plt.close()


@torch.no_grad()
def recursion_width_diag(model, xs, mask, z_grid, noise_std):
    """Definitive over-dispersion probe: is the operator's posterior broad because the bootstrap
    TARGET is broad (the predict over-diffuses -> structural recursion problem) or because it
    fails to MATCH a sharp target (representability/stiffness)? Returns (op_post_std,
    bootstrap_target_std), averaged over observed steps on a fine grid. target_i+1 = predict_i
    (operator s=1) x likelihood_i+1 -- exactly the jump target, but on the grid not the SNIS samples."""
    ctx = model.context(xs, mask)
    ell0 = model.log_density(ctx, z_grid, s=0.0)                    # post-update (s=0)
    ellT = model.log_density(ctx, z_grid, s=1.0)                    # one-step predict (s=1)
    pi0 = (ell0 - ell0.logsumexp(-1, keepdim=True)).exp()
    loglik = -0.5 * (xs[..., 0].unsqueeze(-1) - z_grid) ** 2 / noise_std ** 2
    logtgt = (ellT[:-1] - ellT[:-1].logsumexp(-1, keepdim=True)) + loglik[1:]
    Wtgt = (logtgt - logtgt.logsumexp(-1, keepdim=True)).exp()      # predict_i x lik_{i+1}

    def std(p):
        m = (p * z_grid).sum(-1)
        return (p * z_grid ** 2).sum(-1).sub(m ** 2).clamp_min(0).sqrt()

    return std(pi0[1:]).mean().item(), std(Wtgt).mean().item()


def main(
    batch_size=512,
    n_val=64,
    t0=0.0,
    t1=10.0,
    num_steps=100,
    a=1.0,
    sigma=0.6,
    noise_std=0.1,
    Nz=200,
    zmax=3.0,
    n_sub=5,
    gap_lo=0.4,                 # observation gap: no obs in [gap_lo, gap_hi] * horizon
    gap_hi=0.7,
    n_scoll=4,                  # within-interval time collocation points (continuous PINN)
    n_tcoll=32,                 # time steps subsampled per iter for the (autodiff) FP residual
    res_post=0.0,               # TARGET posterior-mass weighting of the FP residual (0=uniform)
    res_warm=4000,              # ramp res_post in only after this step (posterior must be reliable)
    res_ramp=2000,              # linear ramp duration for res_post (0 -> target)
    w_res=1.0,                  # weight on the FP-residual term (diagnostic: 0 drops the FP
                                # constraint to test whether it is what holds the post-update broad)
    n_colloc=128,               # MESH-FREE state collocation points sampled per step
    near_std=0.3,               # proposal: spread of the near-observation component
    broad_std=1.2,              # proposal: spread of the broad (gap / both-wells) component
    gru_hidden=64,
    ctx_dim=64,
    p=64,
    train_mode="recursion",     # "supervised" (Stage 1) | "recursion" (Stage 2, no teacher)
    lr=2e-3,
    num_iters=3000,
    pause_every=200,
    train_dir="./dump/neural_zakai_stage2/",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # save the run log (the per-step KL/loss) alongside the figure PDFs in train_dir
    os.makedirs(train_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(train_dir, "train.log"), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(fh)

    xs, ts = make_dataset(t0=t0, t1=t1, batch_size=batch_size, noise_std=noise_std,
                          num_steps=num_steps, sigma=sigma, a=a, train_dir=train_dir,
                          device=device)
    dt_obs = float((ts[1] - ts[0]).item())
    z_grid = torch.linspace(-zmax, zmax, Nz, device=device)
    s_coll = torch.linspace(0.0, 1.0, n_scoll, device=device)   # within-interval time tau/dt

    # observation gap: no observations in [gap_lo, gap_hi] of the horizon, so the
    # filter predicts-only and the posterior relaxes toward the bimodal stationary.
    mask_t = torch.ones(num_steps, dtype=torch.bool, device=device)
    mask_t[int(gap_lo * num_steps):int(gap_hi * num_steps)] = False
    mask = mask_t.view(num_steps, 1, 1).float().expand(num_steps, batch_size, 1).contiguous()

    filtered, _ = grid_filter_target(xs, z_grid, a, sigma, noise_std, dt_obs, n_sub, mask=mask_t)

    # train / held-out split over trajectories
    xs_tr, xs_val = xs[:, n_val:], xs[:, :n_val]
    filt_tr, filt_val = filtered[:, n_val:], filtered[:, :n_val]
    mask_tr, mask_val = mask[:, n_val:], mask[:, :n_val]

    # known dynamics. The "zakai" PINN is MESH-FREE: it evaluates the drift ANALYTICALLY
    # at sampled points via these closures (drift(z) -> f, f'; log_prior(z)), no grid. The
    # grid quantities below are for the "recursion" baseline and for VALIDATION only.
    def drift(z):
        return a * (z - z ** 3), a * (1 - 3 * z ** 2)

    def log_prior(z):
        return -0.5 * z ** 2                                # log N(0,1) up to a const (cancels)

    f_true_grid = a * (z_grid - z_grid ** 3)
    T_full = transition_power(build_transition(f_true_grid, z_grid, dt_obs / n_sub, sigma), n_sub)
    prior = torch.softmax(-0.5 * z_grid ** 2, dim=0)

    model = OperatorFilter(data_size=1, gru_hidden=gru_hidden, ctx_dim=ctx_dim, p=p).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9998)

    for step in tqdm.tqdm(range(1, num_iters + 1)):
        optimizer.zero_grad()
        breakdown = ""
        if train_mode == "supervised":                     # Stage 1: grid filter as teacher
            loss = kl_target_pred(filt_tr, model.log_posterior(xs_tr, mask_tr, z_grid))
            nll = torch.zeros((), device=device)
        elif train_mode == "zakai":              # Stage 2: MESH-FREE continuous-time Zakai PINN
            # ramp the posterior-mass weighting in only once the operator (hence its posterior,
            # which is the weight) is reliable -- early on the uniform residual is used
            rp = res_post * min(1.0, max(0.0, (step - res_warm) / res_ramp))
            z_col, log_q = sample_collocation(xs_tr, mask_tr, n_colloc, near_std, broad_std)
            res, jump, ic, nll = pinn_zakai_loss(
                model, xs_tr, mask_tr, z_col, log_q, s_coll, drift, sigma, log_prior,
                noise_std, dt_obs, n_tcoll=n_tcoll, res_post=rp)
            loss = w_res * res + jump + ic
            breakdown = f"res: {res.item():.4f}, jump: {jump.item():.4f}, ic: {ic.item():.4f}, "
        else:                                              # Stage 2: grid-transition recursion
            loss, nll = recursion_loss(model, xs_tr, mask_tr, z_grid, T_full, prior, noise_std)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % pause_every == 0:
            with torch.no_grad():
                kl_val = kl_target_pred(
                    filt_val, model.log_posterior(xs_val, mask_val, z_grid)).item()
                op_w, tgt_w = recursion_width_diag(model, xs_val, mask_val, z_grid, noise_std)
                exm = (filt_val * z_grid).sum(-1)
                ex_w = (filt_val * z_grid ** 2).sum(-1).sub(exm ** 2).clamp_min(0).sqrt().mean().item()
            logging.warning(f"step {step:05d}, loss: {loss.item():.4f}, {breakdown}"
                            f"NLL: {nll.item():.2f}, "
                            f"KL(operator vs exact filter, held-out): {kl_val:.4f}, "
                            f"std[op={op_w:.3f} target={tgt_w:.3f} exact={ex_w:.3f}]")
            vis(model, xs_val, mask_val, filt_val, z_grid, ts,
                os.path.join(train_dir, f"step_{step:05d}.pdf"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    fire.Fire(main)
