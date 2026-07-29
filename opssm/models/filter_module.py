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

"""ZakaiFilterModule: the neural mesh-free Zakai filter as a LightningModule.

Manual optimization, because training alternates two channels:
  - E-step (every step): a gradient step on the OPERATOR via the mesh-free Zakai PINN loss
    (`accumulate_pinn_grads`), with the dynamics held frozen.
  - M-step (every `m_every` steps after `warmup`): closed-form / regression updates of the drift,
    diffusion, and (high-D) the Stiefel observation map, from the operator's inferred latent path.

One configurable module covers all experiments: `learn_dynamics` toggles the M-step (off → the
supervised/known-dynamics operator), `learn_obs` toggles the high-D Stiefel sensor learning, and
`loss` selects the operator objective. 1-D is the degenerate `cstab≡0` case of the high-D curriculum.
"""

import os

import lightning.pytorch as pl
import torch
from torch import optim

from opssm.models.operator import OperatorFilter, OperatorBackward
from opssm.models.dynamics import DriftNet, DiffusionNet
from opssm.models.losses import accumulate_pinn_grads, accumulate_adjoint_grads, kl_target_pred
from opssm.models.mstep import mstep, log_smoothed
from opssm.models.obs import make_decode, zhat_from_obs
from opssm.models.mstep import posterior_mean_fixed
from opssm.data.systems import make_drift
from opssm.analysis import viz


def _log_prior(z):
    return -0.5 * z.pow(2).sum(-1)                             # log N(0, I_d) up to a const (z (...,d) -> (...,))


def _interp1d(vals, grid, query):
    """Batched 1-D linear interpolation: vals (..., Nz) on ascending grid (Nz,), query (Nq,) -> (..., Nq)."""
    idx = torch.searchsorted(grid, query).clamp(1, grid.numel() - 1)
    x0 = grid[idx - 1]; x1 = grid[idx]
    w = ((query - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0.0, 1.0)
    return vals[..., idx - 1] * (1 - w) + vals[..., idx] * w


class ZakaiFilterModule(pl.LightningModule):
    def __init__(self, data_size=1, gru_hidden=64, ctx_dim=64, p=64, drift_hidden=64,
                 lr=2e-3, drift_lr=2e-3, sched_gamma=0.9998,
                 n_scoll=4, n_tcoll=24, n_colloc=128, chunk_size=16, near_std=0.3, broad_std=1.6,
                 warmup=2000, m_every=2000, m_inner=400, reg_lambda=3e-4, reg_lambda_g=3e-3,
                 g_init=1.0, learn_dynamics=True, learn_g=True, g_net=False, learn_obs=False,
                 pca_init=True, c_stable_tol=0.05, meshfree_mean=True, n_mean=256,
                 res_mode="rel", w_res=0.2, learn_smoother=False, joint_g=False,
                 encoder="gru", encoder_kwargs=None,
                 latent_dim=1, loss="zakai", train_dir="./dump/nzf"):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        h = self.hparams
        self.model = OperatorFilter(h.data_size, h.gru_hidden, h.ctx_dim, h.p,
                                    encoder=h.encoder, encoder_kwargs=h.encoder_kwargs, latent_dim=h.latent_dim)
        # backward adjoint-Zakai twin for the two-filter smoother (default off => byte-identical filter)
        self.model_b = OperatorBackward(h.data_size, h.gru_hidden, h.ctx_dim, h.p,
                                        encoder=h.encoder, encoder_kwargs=h.encoder_kwargs, latent_dim=h.latent_dim) \
            if h.learn_smoother else None
        self.drift_net = DriftNet(h.drift_hidden, latent_dim=h.latent_dim); self.drift_net.requires_grad_(False)
        self.diff_net = None
        if h.learn_g and h.g_net:
            self.diff_net = DiffusionNet(h.drift_hidden, g_init=h.g_init, latent_dim=h.latent_dim)
            self.diff_net.requires_grad_(False)
        # mutable EM state (initialized in setup once the data is known)
        self.g_cur = h.g_init
        self.C_cur = self.d_cur = None
        self.dr_opt = self.dg_opt = None

    # -- static config pulled from the datamodule + EM-state init --------------------------------
    def setup(self, stage=None):
        h = self.hparams
        dm = self.trainer.datamodule
        dev = self.device
        self.z_grid = dm.z_grid
        self.dt = dm.dt
        self.a, self.sigma = dm.hparams.a, dm.hparams.sigma
        self.noise_std = dm.noise_std_eff                            # obs noise in the STANDARDIZED units (see datamodule)
        self.system = dm.hparams.system
        self.true_drift = make_drift(self.system)[0]                  # ground-truth drift for the metric (registry)
        self.s_coll = torch.linspace(0.0, 1.0, h.n_scoll, device=dev)
        d = self.model.latent_dim
        if d == 1:                                                    # 1-D grid metrics (kl, drift_l2) + g_net reg
            self.z_reg = torch.linspace(-2.0, 2.0, 80, device=dev)
            self.hr = float(self.z_reg[1] - self.z_reg[0])
            self.f_true_grid = self.a * (self.z_grid - self.z_grid ** 3)
            self.supp = self.z_grid.abs() <= 2.0
        else:                                                         # mesh-free for d>1 (no grid)
            self.z_reg = self.hr = self.f_true_grid = self.supp = None
        if not h.learn_g:
            self.g_cur = self.sigma
        # high-D observation map init. Obs are standardized to ~unit scale at the dataloader level, so the
        # decode is h(z) = C z + d with NO scale factor (s_scale removed) -- C is a D x d Stiefel matrix and
        # d the (now ~zero) intercept.
        if h.learn_obs:
            obs_dim = dm.hparams.obs_dim
            Yc = dm.full_obs.reshape(-1, obs_dim)
            ybar = Yc.mean(0)
            if h.pca_init:
                _, _, Vt = torch.linalg.svd(Yc - ybar, full_matrices=False)
                self.C_cur = Vt[:d].t().contiguous()                  # (D,d) top-d PCA directions as columns
            else:
                self.C_cur = torch.linalg.qr(torch.randn(obs_dim, d, device=dev))[0]   # random Stiefel (D,d)
            self.d_cur = ybar.clone()
        # manual M-step optimizers (not Lightning-managed; the M-step is fully manual)
        self.dr_opt = optim.Adam(self.drift_net.parameters(), lr=h.drift_lr)
        if self.diff_net is not None:
            self.dg_opt = optim.Adam(self.diff_net.parameters(), lr=h.drift_lr)
        os.makedirs(h.train_dir, exist_ok=True)

    def configure_optimizers(self):
        h = self.hparams
        # ONE optimizer over both operators' (disjoint) params -> a single optimizer.step() per batch,
        # so global_step stays 1/batch (two Lightning optimizers under manual opt double-count it, which
        # would halve warmup/m_every/max_steps and skip validation). The forward and backward losses
        # touch disjoint parameters, so a shared Adam is exactly two independent Adams.
        params = list(self.model.parameters())
        if h.learn_smoother:
            params += list(self.model_b.parameters())
        opt = optim.Adam(params, lr=h.lr)
        sched = optim.lr_scheduler.ExponentialLR(opt, gamma=h.sched_gamma)
        return {"optimizer": opt, "lr_scheduler": sched}

    # -- helpers for the high-D decode / collocation-center hooks --------------------------------
    def _decode_center(self, x):
        if not self.hparams.learn_obs:
            return None, None
        decode = make_decode(self.C_cur, self.d_cur)               # h(z) = C z + d (obs standardized upstream)
        center = zhat_from_obs(x, self.C_cur, self.d_cur)
        return decode, center

    # -- E-step ----------------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        h = self.hparams
        x, mask, filt = batch
        opt, sched = self.optimizers(), self.lr_schedulers()
        if h.loss == "supervised":                                # Stage-1 sanity: fit the exact filter
            opt.zero_grad()
            loss = kl_target_pred(filt, self.model.log_posterior(x, mask, self.z_grid))
            self.manual_backward(loss)
            opt.step(); sched.step()
            self.log("kl_train", loss, prog_bar=True)
            return
        step = self.global_step + 1
        drift = (lambda z: (torch.zeros_like(z), torch.zeros(z.shape[:-1], device=z.device, dtype=z.dtype))) \
            if step <= h.warmup else self.drift_net.drift
        diffusion = self.diff_net.diffusion if (h.g_net and step > h.warmup) else self.g_cur
        decode, center = self._decode_center(x)
        opt.zero_grad()
        res, jump, ic, _ = accumulate_pinn_grads(                 # does its own chunked backward
            self.model, x, mask, self.s_coll, drift, diffusion, _log_prior, self.noise_std,
            self.dt, h.n_colloc, h.near_std, h.broad_std, h.n_tcoll, h.chunk_size,
            res_mode=h.res_mode, w_res=h.w_res, decode=decode, center=center)
        if h.learn_smoother:                                      # backward E-step: accumulate into the
            res_b, jump_b, tc_b = accumulate_adjoint_grads(       # SAME optimizer (disjoint model_b params,
                self.model_b, x, mask, self.s_coll, drift, diffusion, self.noise_std, self.dt,  # same frozen dyn)
                h.n_colloc, h.near_std, h.broad_std, h.n_tcoll, h.chunk_size,
                res_mode=h.res_mode, w_res=h.w_res, decode=decode, center=center)
        opt.step()
        sched.step()
        self.log_dict({"res": res, "jump": jump, "ic": ic}, prog_bar=True)
        if h.learn_smoother:
            self.log_dict({"res_b": res_b, "jump_b": jump_b, "tc_b": tc_b}, prog_bar=False)

    # -- M-step ----------------------------------------------------------------------------------
    def on_train_batch_end(self, outputs, batch, batch_idx):
        h = self.hparams
        step = self.global_step
        if not h.learn_dynamics or step <= h.warmup or step % h.m_every != 0:
            return
        x, mask, _ = batch
        out = mstep(self.model, x, mask, self.z_grid, self.dt, self.drift_net, self.dr_opt,
                    self.diff_net, self.dg_opt, self.z_reg, self.hr,
                    learn_g=h.learn_g, g_net=h.g_net, reg_lambda=h.reg_lambda,
                    reg_lambda_g=h.reg_lambda_g, m_inner=h.m_inner,
                    learn_obs=h.learn_obs, c_stable_tol=h.c_stable_tol,
                    C_cur=self.C_cur, d_cur=self.d_cur,
                    meshfree_mean=h.meshfree_mean, n_mean=h.n_mean,
                    near_std=h.near_std, broad_std=h.broad_std,
                    joint_g=h.joint_g, noise_std=self.noise_std, g_cur_in=self.g_cur)
        if out["g_cur"] is not None:
            self.g_cur = out["g_cur"]
        if h.learn_obs:
            self.C_cur, self.d_cur = out["C_cur"], out["d_cur"]
        if out.get("ess") is not None:
            self.log("ess", out["ess"], prog_bar=True)                  # SNIS health diagnostic

    @torch.no_grad()
    def _gauge_aligned(self, log_pi, filt, m_op, z_true):
        """PROCRUSTES-aligned metrics. A latent SDE with linear-Gaussian obs is identifiable only up to a
        LINEAR-MAP gauge (z->A z, f->A f(A^-1 .), g->A g, s_scale absorbs it -- the data distribution is
        unchanged), so scoring operator-z vs true-z on an absolute frame is ill-posed and inflates the gap.
        Fit the best map A (least squares z_true ~ A m_op) and score in the aligned frame.
          DIMENSION-AGNOSTIC (pure linear algebra, no grid): A is a SCALAR for a 1-D latent, a d x d matrix
          for a multi-dim latent; `lat_rmse_aln` follows for any d.
          1-D-MODEL-BOUND (guarded to d==1): drift/diffusion (the drift net is 1->1) and KL (needs the 1-D
          grid oracle -- no analog for a multi-dim latent). These generalize by swapping in a d->d drift net
          and a sample-based KL once the LATENT model goes multi-dim; the metric code is not the blocker."""
        d = m_op.shape[-1] if m_op.dim() >= 3 else 1
        M = m_op.reshape(-1, d); Z = z_true.reshape(-1, d)                # inferred vs true latent points (N,d)
        A = torch.linalg.lstsq(M, Z).solution                            # (d,d): Z ~ M @ A  (scalar for 1-D)
        z_al = M @ A                                                     # aligned latent at the inferred points
        # aligned DRIFT (dimension-agnostic, mesh-free): evaluate f_op at the inferred points and map to the
        # true frame  F = f_op @ A ; compare to the benchmark's true drift at z_al. Works for a 1-D double-well
        # or a 2-D Van der Pol / 3-D Lorenz latent given a matching d->d drift net; the true-drift form below
        # is the ONLY per-benchmark piece (double-well a(z-z^3) here -- swap for VdP/Lorenz's field).
        f_al = self.drift_net.drift(M)[0].reshape(-1, d) @ A
        f_true = self.true_drift(z_al)                                 # ground-truth drift from the systems registry
        on = (z_al.abs().le(1.5).all(-1) if d == 1                     # double-well data region; all points for d>1
              else torch.ones(z_al.shape[0], dtype=torch.bool, device=z_al.device))
        gscale = A.det().abs().pow(1.0 / d).item()                      # |A|^(1/d): scalar |A| for 1-D, dxd det
        out = {"s_fit": float(A.reshape(-1)[0]) if d == 1 else gscale,  # signed 1-D scale (viz) / geo-mean scale
               "lat_rmse": (M - Z).pow(2).sum(-1).mean().sqrt().item(),          # RAW (unaligned)
               "lat_rmse_aln": (z_al - Z).pow(2).sum(-1).mean().sqrt().item(),   # aligned
               "drift_l2_aln": (f_al[on] - f_true[on]).pow(2).sum(-1).mean().sqrt().item(),
               "g_aln": gscale * float(self.g_cur)}                     # isotropic g scales by |A|^(1/d)
        if d == 1:                                                       # KL: grid-bound (1-D oracle only)
            s = float(A.reshape(-1)[0]); zg = self.z_grid
            pi_al = _interp1d(log_pi.exp(), zg, zg / s).clamp_min(0) / abs(s)   # push posterior to true frame
            pi_al = pi_al / pi_al.sum(-1, keepdim=True).clamp_min(1e-12)
            out["kl_aln"] = kl_target_pred(filt, pi_al.clamp_min(1e-20).log()).item()
        return out

    # -- validation: KL vs the exact filter, drift L2, C cos, + a figure -------------------------
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        h = self.hparams
        dm = self.trainer.datamodule
        x, mask, filt = batch
        d = self.model.latent_dim
        logs = {"g": float(self.g_cur)}
        log_pi = None
        if d == 1:                                                   # 1-D GRID metrics (kl, drift_l2 on the grid)
            log_pi = self.model.log_posterior(x, mask, self.z_grid)
            logs["kl"] = kl_target_pred(filt, log_pi).item()
            m_op = (log_pi.exp() * self.z_grid).sum(-1)              # (T,B) grid-quadrature mean
            lo, hi = m_op.quantile(0.01), m_op.quantile(0.99)       # ON-DATA regime (the range the latent visits)
            on = (self.z_grid >= lo) & (self.z_grid <= hi)
            off = self.supp & ~on                                   # |z|<=2 but outside the data regime
            fd = self.drift_net.drift(self.z_grid.unsqueeze(-1))[0].squeeze(-1)   # drift on the 1-D grid
            logs["drift_l2"] = (fd[on] - self.f_true_grid[on]).pow(2).mean().sqrt().item()
            logs["drift_l2_off"] = ((fd[off] - self.f_true_grid[off]).pow(2).mean().sqrt().item()
                                    if bool(off.any()) else float("nan"))
            m_op_al = m_op                                          # (T,B) -> _gauge_aligned (d==1)
        else:                                                       # MESH-FREE mean for d>1 (no grid)
            _, center = self._decode_center(x)
            m_op_al, _ = posterior_mean_fixed(self.model, x, mask, center, h.n_mean, h.near_std, h.broad_std)  # (T,B,d)
        if h.learn_obs and dm.C_true is not None:                   # c_cos: mean principal-angle cosine (any d)
            Cn = dm.C_true / dm.C_true.norm(dim=0, keepdim=True)
            logs["c_cos"] = float(torch.linalg.svdvals(self.C_cur.t() @ Cn).clamp(max=1.0).mean())
        s_fit = None                                                # PROCRUSTES gauge-aligned metrics (well-posed)
        if h.learn_obs and dm.z_val_true is not None:
            al = self._gauge_aligned(log_pi, filt, m_op_al, dm.z_val_true)
            logs.update(al); s_fit = al["s_fit"]
        if d == 1 and h.learn_smoother and getattr(dm, "smoothed_val", None) is not None:
            log_sm = log_smoothed(self.model, self.model_b, x, mask, self.z_grid)
            logs["kl_smooth"] = kl_target_pred(dm.smoothed_val, log_sm).item()
        self.log_dict(logs, prog_bar=True)
        # figure: 1-D the rich Duncker panels (raw + Procrustes-aligned), 2-D the phase-plane drift
        # STREAMPLOT, 3-D the attractor + projected drift quivers. d>3 has no figure.
        img = os.path.join(h.train_dir, f"step_{self.global_step:05d}.pdf")
        if d == 1:
            sm_val = getattr(dm, "smoothed_val", None) if h.learn_smoother else None
            if h.learn_obs:                                          # squeeze the trailing latent axis to the 1-D viz
                zt1, Cc, Ct = dm.z_val_true.squeeze(-1), self.C_cur.squeeze(-1), dm.C_true.squeeze(-1)
                for al, tag in ([(False, "_raw"), (True, "_aligned")] if s_fit is not None else [(False, "")]):
                    viz.vis_highd(self.model, self.drift_net, self.diff_net, x, mask, zt1, filt,
                                  self.z_grid, dm.ts, self.a, self.sigma, Cc, self.d_cur, Ct, dm.d_true,
                                  True, img.replace(".pdf", tag + ".pdf"), g_scalar=self.g_cur,
                                  model_b=self.model_b, smoothed_val=sm_val, s_fit=s_fit, aligned=al,
                                  obs_mean=dm.obs_mean, obs_scale=dm.obs_scale)
            else:
                viz.vis_learn(self.model, self.drift_net, x, mask, filt, self.z_grid, dm.ts, self.a,
                              img, diff_net=self.diff_net, sigma=self.sigma,
                              model_b=self.model_b, smoothed_val=sm_val)
        elif d in (2, 3) and dm.z_val_true is not None:
            vis = viz.vis_latent2d if d == 2 else viz.vis_latent3d
            vis(self.drift_net, m_op_al, dm.z_val_true, dm.ts, self.true_drift, self.g_cur, img)
