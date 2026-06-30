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

"""STAGE 3 -- learn the DYNAMICS from data with the mesh-free neural Zakai filter.

Stage 2 (run/scripts/neural_zakai_filter.py, train_mode="zakai") learned the
filtering density with KNOWN drift f. Here f is UNKNOWN: a neural network f_theta
whose value and derivative enter the Fokker-Planck operator L*, and which is
identified PURELY from the data likelihood. This is a PINN inverse problem:

    operator (the PDE solution ansatz)  ell_theta(z,s)   -- the filtering log-density
    drift   (the unknown PDE parameter) f_theta(z)        -- with f' by autodiff
    residual ties the operator's time-evolution to f_theta (continuous Zakai PINN);
    the data NLL = -sum_i log c_i  (the normalizer/evidence) makes that evolution
    explain the observations.  Jointly minimizing residual + recursion + NLL over
    (operator, f_theta) drives f_theta -> the true drift, with NO supervision of f.

We reuse the operator, the mesh-free SNIS loss, the data-following proposal, and the
grid oracle from neural_zakai_filter.py unchanged; the only additions are the drift
network and the data-NLL term. Success = f_theta(z) recovers a(z - z^3) on the
support, the NLL approaches the exact grid filter's, and the posterior still matches.

To run:
python -m run.scripts.neural_zakai_filter_learn
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
    make_dataset, build_transition, transition_power)
from run.scripts.neural_zakai_filter import (
    OperatorFilter, accumulate_pinn_grads, grid_filter_target, kl_target_pred, mlp)


@torch.no_grad()
def grid_nll_masked(xs, z, T_full, noise_std, mask):
    """Exact grid-filter data NLL = -sum_i log c_i over OBSERVED steps, using the SAME
    (unnormalized-Gaussian) likelihood as the operator's SNIS evidence so the two are
    directly comparable -- the optimal the learned NLL should approach. mask (T,) bool."""
    T_steps, B = xs.shape[0], xs.shape[1]
    inv2var = 0.5 / noise_std ** 2
    pi_pred = torch.softmax(-0.5 * z ** 2, dim=0).unsqueeze(0).expand(B, z.numel()).clone()
    nll = torch.zeros(B, device=xs.device)
    for i in range(T_steps):
        if bool(mask[i]):
            w = torch.exp(-inv2var * (xs[i, :, 0:1] - z.unsqueeze(0)) ** 2) * pi_pred
            c = w.sum(1)
            nll = nll - torch.log(c + 1e-12)
            pi_pred = w / (c.unsqueeze(1) + 1e-12)
        pi_pred = pi_pred @ T_full.t()
    return nll.mean()


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


@torch.no_grad()
def vis(model, drift_net, xs_val, mask_val, filt_val, z_grid, ts, a, img_path, n_traj=2,
        diff_net=None, sigma=None):
    z = z_grid.cpu().numpy()
    ts_np = ts.cpu().numpy()
    f_learned = drift_net.drift(z_grid)[0].cpu().numpy()
    f_true = (a * (z_grid - z_grid ** 3)).cpu().numpy()
    log_pi = model.log_posterior(xs_val, mask_val, z_grid)
    pi = log_pi.exp().cpu().numpy()
    filt = filt_val.cpu().numpy()
    mean = (filt_val * z_grid).sum(-1)
    var = (filt_val * z_grid ** 2).sum(-1) - mean ** 2
    t_star = var[2:-2].argmax(dim=0).cpu().numpy() + 2
    obs = mask_val[:, 0, 0].cpu().numpy().astype(bool)
    gap = ~obs

    has_g = diff_net is not None
    n_cols = n_traj + 2 + (1 if has_g else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4.5))
    # learned drift vs truth
    ax = axes[0]
    ax.plot(z, f_true, "k-", lw=2, label="true $a(z-z^3)$")
    ax.plot(z, f_learned, "C2--", lw=2, label=r"learned $f_\theta$")
    ax.axvspan(-2, 2, color="C2", alpha=0.06, label="support")
    ax.set_xlabel("$z$"); ax.set_ylabel("$f(z)$"); ax.set_ylim(-4, 4)
    ax.set_title("learned drift"); ax.legend(fontsize=9)
    off = 1
    # learned diffusion g^2(z) vs the true constant (when a DiffusionNet is given)
    if has_g:
        ax = axes[1]
        g2 = diff_net._g2(z_grid.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        ax.plot(z, g2, "C0-", lw=2, label=r"learned $g^2(z)$")
        if sigma is not None:
            ax.axhline(sigma ** 2, ls="--", c="k", lw=2, label=fr"true $\sigma^2={sigma ** 2:.3f}$")
        ax.axvspan(-2, 2, color="C2", alpha=0.06)
        ax.set_xlabel("$z$"); ax.set_ylabel("$g^2(z)$")
        ax.set_ylim(0, max(0.6, float(g2.max()) * 1.2))
        ax.set_title("learned diffusion"); ax.legend(fontsize=9)
        off = 2
    # posterior snapshots in the gap
    for j in range(n_traj):
        tj = int(t_star[j]); ax = axes[off + j]
        ax.plot(z, filt[tj, j], "k-", lw=2, label="exact filter")
        ax.plot(z, pi[tj, j], "C3--", lw=2, label="operator")
        ax.set_xlabel("$z$"); ax.set_title(f"posterior traj {j}, $t={tj}$ (gap)")
        if j == 0:
            ax.legend(fontsize=9)
    # mean +/- 2 std over time, traj 0
    j = 0; ax = axes[-1]
    m_op = (log_pi[:, j].exp() * z_grid).sum(-1).cpu().numpy()
    s_op = ((log_pi[:, j].exp() * z_grid ** 2).sum(-1).cpu().numpy() - m_op ** 2).clip(0) ** 0.5
    if gap.any():
        ax.axvspan(ts_np[gap][0], ts_np[gap][-1], color="gray", alpha=0.15, label="no obs")
    ax.plot(ts_np, mean[:, j].cpu().numpy(), "k-", lw=2, label="exact mean")
    ax.plot(ts_np, m_op, "C3--", lw=2, label="operator mean")
    ax.fill_between(ts_np, m_op - 2 * s_op, m_op + 2 * s_op, color="C3", alpha=0.2)
    ax.set_xlabel("$t$"); ax.set_ylabel("$z$"); ax.set_title("estimate traj 0"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(img_path); plt.close()


def main(
    batch_size=128,
    n_val=32,
    t0=0.0,
    t1=10.0,
    num_steps=100,
    a=1.0,
    sigma=0.6,
    noise_std=0.1,
    Nz=200,
    zmax=3.0,
    n_sub=5,
    gap_lo=0.4,
    gap_hi=0.7,
    obs_every=1,                # 1 = dense obs + a gap; K>1 = observe every K steps (SPARSE:
                                #     long inter-obs prediction horizons -> drift identifiable)
    n_scoll=4,
    n_tcoll=24,
    n_colloc=128,
    chunk_size=16,              # batch chunk for forward+backward (caps peak GPU memory ~3GB)
    near_std=0.3,
    broad_std=1.6,
    gru_hidden=64,
    ctx_dim=64,
    p=64,
    w_nll=1.0,                  # weight on the per-step data NLL (the term that learns f)
    drift_hidden=64,
    drift_lr=2e-3,
    lr=2e-3,
    num_iters=20000,
    warmup=2000,                # train the operator with f frozen (~0) before unfreezing f
    alternate=0,                # 0 = joint; K>0 = alternate K steps operator-only / K drift-only
    drift_ckpt=None,            # load a pre-learned (EM) drift and FREEZE it: operator filters
                                #     with fixed learned dynamics (validates "learn f, then filter")
    pause_every=2000,
    resume=True,                # resume from train_dir/ckpt.pt if present (survive Xid crashes)
    train_dir="./dump/neural_zakai_learn/",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    os.makedirs(train_dir, exist_ok=True)
    ckpt_path = os.path.join(train_dir, "ckpt.pt")
    resuming = resume and os.path.exists(ckpt_path)
    fh = logging.FileHandler(os.path.join(train_dir, "train.log"), mode="a" if resuming else "w")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(fh)

    xs, ts = make_dataset(t0=t0, t1=t1, batch_size=batch_size, noise_std=noise_std,
                          num_steps=num_steps, sigma=sigma, a=a, train_dir=train_dir,
                          device=device)
    dt_obs = float((ts[1] - ts[0]).item())
    z_grid = torch.linspace(-zmax, zmax, Nz, device=device)
    s_coll = torch.linspace(0.0, 1.0, n_scoll, device=device)

    if obs_every > 1:           # SPARSE regime: predict over many steps before each update, so
        mask_t = torch.zeros(num_steps, dtype=torch.bool, device=device)  # f's effect accumulates
        mask_t[::obs_every] = True                                        # and the evidence sees it
    else:                       # dense regime with a single predict-only gap
        mask_t = torch.ones(num_steps, dtype=torch.bool, device=device)
        mask_t[int(gap_lo * num_steps):int(gap_hi * num_steps)] = False
    mask = mask_t.view(num_steps, 1, 1).float().expand(num_steps, batch_size, 1).contiguous()
    filtered, _ = grid_filter_target(xs, z_grid, a, sigma, noise_std, dt_obs, n_sub, mask=mask_t)

    xs_tr, xs_val = xs[:, n_val:], xs[:, :n_val]
    filt_val = filtered[:, :n_val]
    mask_tr, mask_val = mask[:, n_val:], mask[:, :n_val]

    # the exact grid filter NLL with the TRUE drift -- the target the learned NLL should reach
    f_true_grid = a * (z_grid - z_grid ** 3)
    K_true = transition_power(build_transition(f_true_grid, z_grid, dt_obs / n_sub, sigma), n_sub)
    nll_true = grid_nll_masked(xs_tr, z_grid, K_true, noise_std, mask_t).item() / num_steps

    def log_prior(z):
        return -0.5 * z ** 2

    model = OperatorFilter(1, gru_hidden, ctx_dim, p).to(device)
    drift_net = DriftNet(drift_hidden).to(device)
    if drift_ckpt:              # plug in a pre-learned (EM) drift and freeze it
        drift_net.net.load_state_dict(torch.load(drift_ckpt, map_location=device))
        drift_net.requires_grad_(False)
        warmup, w_nll, alternate = 0, 0.0, 0    # use the learned f from step 1, never train it
        logging.warning(f"loaded FROZEN drift from {drift_ckpt}; operator filters with learned f")
    optimizer = optim.Adam([
        {"params": model.parameters(), "lr": lr},
        {"params": drift_net.parameters(), "lr": drift_lr}])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9998)

    # L2 error of the learned drift vs the truth, measured on the support [-2, 2]
    supp = (z_grid.abs() <= 2.0)

    for step in tqdm.tqdm(range(1, num_iters + 1)):
        optimizer.zero_grad()
        # warmup: freeze f ~ 0 so the operator first becomes a sane filter, then learn f
        drift = (lambda z: (torch.zeros_like(z), torch.zeros_like(z))) if step <= warmup \
            else drift_net.drift
        # alternating optimization (after warmup): freeze the other block each phase, so the
        # operator first becomes the data-fitted filter (drift frozen), then f regresses onto
        # that fixed, data-converged evolution (operator frozen) -- decoupling the joint
        # co-adaptation where the flexible operator absorbs the drift error.
        if alternate and step > warmup:
            drift_phase = ((step - warmup - 1) // alternate) % 2 == 1
            model.requires_grad_(not drift_phase)
            drift_net.requires_grad_(drift_phase)
        # chunked forward+backward (memory-capped); grads accumulate over batch chunks
        res, jump, ic, nll = accumulate_pinn_grads(
            model, xs_tr, mask_tr, s_coll, drift, sigma, log_prior, noise_std, dt_obs,
            n_colloc, near_std, broad_std, n_tcoll, chunk_size, w_nll=w_nll, num_steps=num_steps)
        torch.nn.utils.clip_grad_norm_(drift_net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % pause_every == 0:
            with torch.no_grad():
                kl_val = kl_target_pred(
                    filt_val, model.log_posterior(xs_val, mask_val, z_grid)).item()
                f_learned = drift_net.drift(z_grid)[0]
                f_err = (f_learned[supp] - f_true_grid[supp]).pow(2).mean().sqrt().item()
            logging.warning(
                f"step {step:05d}, res: {res:.4f}, jump: {jump:.4f}, ic: {ic:.4f}, "
                f"NLL/T: {nll/num_steps:.4f} (true {nll_true:.4f}), "
                f"drift L2 err: {f_err:.4f}, KL(post): {kl_val:.4f}")
            vis(model, drift_net, xs_val, mask_val, filt_val, z_grid, ts, a,
                os.path.join(train_dir, f"step_{step:05d}.pdf"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    fire.Fire(main)
