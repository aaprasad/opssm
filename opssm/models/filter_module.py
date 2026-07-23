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

import math
import os

import lightning.pytorch as pl
import torch
from torch import optim

from opssm.models.operator import OperatorFilter, OperatorBackward
from opssm.models.dynamics import DriftNet, DiffusionNet
from opssm.models.losses import accumulate_pinn_grads, accumulate_adjoint_grads, kl_target_pred
from opssm.models.mstep import (mstep, log_smoothed, posterior_mean_fixed, smoother_mean_fixed,
                                _fixed_nodes)
from opssm.models.obs import make_decode, zhat_from_obs
from opssm.analysis import viz


def _log_prior(z):
    return -0.5 * z ** 2                                       # log N(0,1) up to a constant


class ZakaiFilterModule(pl.LightningModule):
    def __init__(self, data_size=1, gru_hidden=64, ctx_dim=64, p=64, drift_hidden=64,
                 lr=2e-3, drift_lr=2e-3, sched_gamma=0.9998,
                 n_scoll=4, n_tcoll=24, n_colloc=128, chunk_size=16, near_std=0.3, broad_std=1.6,
                 warmup=2000, m_every=2000, m_inner=400, reg_lambda=3e-4, reg_lambda_g=3e-3,
                 g_init=1.0, learn_dynamics=True, learn_g=True, g_net=False, learn_obs=False,
                 pca_init=True, c_stable_tol=0.05, meshfree_mean=True, n_mean=256,
                 res_mode="rel", w_res=0.2, learn_smoother=False, smoother_mstep=False,
                 drift_method="euler", loss="zakai", train_dir="./dump/nzf"):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        h = self.hparams
        self.model = OperatorFilter(h.data_size, h.gru_hidden, h.ctx_dim, h.p)
        # backward adjoint-Zakai twin for the two-filter smoother (default off => byte-identical filter)
        self.model_b = OperatorBackward(h.data_size, h.gru_hidden, h.ctx_dim, h.p) \
            if h.learn_smoother else None
        self.drift_net = DriftNet(h.drift_hidden); self.drift_net.requires_grad_(False)
        self.diff_net = None
        if h.learn_g and h.g_net:
            self.diff_net = DiffusionNet(h.drift_hidden, g_init=h.g_init)
            self.diff_net.requires_grad_(False)
        # mutable EM state (initialized in setup once the data is known)
        self.g_cur = h.g_init
        self.s_scale = 1.0
        self.C_cur = self.d_cur = None
        self.dr_opt = self.dg_opt = None

    # -- static config pulled from the datamodule + EM-state init --------------------------------
    def setup(self, stage=None):
        h = self.hparams
        dm = self.trainer.datamodule
        dev = self.device
        self.z_grid = dm.z_grid
        self.dt = dm.dt
        self.a, self.sigma, self.noise_std = dm.hparams.a, dm.hparams.sigma, dm.hparams.noise_std
        self.s_coll = torch.linspace(0.0, 1.0, h.n_scoll, device=dev)
        self.z_reg = torch.linspace(-2.0, 2.0, 80, device=dev)
        self.hr = float(self.z_reg[1] - self.z_reg[0])
        self.f_true_grid = self.a * (self.z_grid - self.z_grid ** 3)
        self.supp = self.z_grid.abs() <= 2.0
        if not h.learn_g:
            self.g_cur = self.sigma
        # high-D observation map: fixed scale + PCA/random unit direction (see plan)
        if h.learn_obs:
            obs_dim = dm.hparams.obs_dim
            Yc = dm.full_obs.reshape(-1, obs_dim)
            ybar = Yc.mean(0)
            _, Sv, Vt = torch.linalg.svd(Yc - ybar, full_matrices=False)
            self.s_scale = float(Sv[0] / math.sqrt(Yc.shape[0]))
            if h.pca_init:
                self.C_cur = Vt[0].clone()
            else:
                c = torch.randn(obs_dim, device=dev); self.C_cur = c / c.norm()
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
        decode = make_decode(self.C_cur, self.d_cur, self.s_scale)
        center = zhat_from_obs(x, self.C_cur, self.d_cur) / self.s_scale
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
        drift = (lambda z: (torch.zeros_like(z), torch.zeros_like(z))) \
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
                    C_cur=self.C_cur, d_cur=self.d_cur, s_scale=self.s_scale,
                    meshfree_mean=h.meshfree_mean, n_mean=h.n_mean,
                    near_std=h.near_std, broad_std=h.broad_std,
                    model_b=self.model_b, smoother_mstep=h.smoother_mstep, g_cur_in=self.g_cur,
                    drift_method=h.drift_method)
        if out["g_cur"] is not None:
            self.g_cur = out["g_cur"]
        if h.learn_obs:
            self.C_cur, self.d_cur = out["C_cur"], out["d_cur"]
        if out.get("ess") is not None:
            self.log("ess", out["ess"], prog_bar=True)                  # SNIS health diagnostic

    @torch.no_grad()
    def _drift_source_diag(self, x, mask):
        """CONFOUNDER-FREE (same operator) comparison of the drift from four latent estimates -> RMSE vs
        the true a(z-z^3), on-data |z|<=1.5. filter/smoother MEAN use pure path increments (fully clean,
        mesh-free, no transition). filter/smoother JOINT use the fixed-node CRN pair (mesh-free -- NO grid:
        propose z_{t+1} via current (f,g), reweight by alpha_t * factor_{t+1}, factor = lik or msg); these
        are EM-updates (the joint needs the transition), so read them as 'the update this source produces'.
        Returns {drift_l2_fm, drift_l2_sm, drift_l2_fj, drift_l2_sj}."""
        h = self.hparams; dt = self.dt; dev = self.device; a = self.a; ns = self.noise_std
        center = (zhat_from_obs(x, self.C_cur, self.d_cur) / self.s_scale) if h.learn_obs else x[..., 0]
        decode = make_decode(self.C_cur, self.d_cur, self.s_scale) if h.learn_obs else None
        m = mask[..., 0]; mv = (m[:-1] * m[1:]).bool()
        zf, _ = posterior_mean_fixed(self.model, x, mask, center, h.n_mean, h.near_std, h.broad_std)
        zs, _ = smoother_mean_fixed(self.model, self.model_b, x, mask, center, h.n_mean, h.near_std, h.broad_std)
        z, log_q = _fixed_nodes(center, mask, h.n_mean, h.near_std, h.broad_std)     # (T,B,K)
        gen = torch.Generator(device=dev).manual_seed(1)
        eps = torch.randn(1, z.shape[1], z.shape[2], device=dev, generator=gen)
        z_next = z + self.drift_net.drift(z)[0] * dt + math.sqrt(max(float(self.g_cur), 1e-6) ** 2 * dt) * eps
        ctx_f = self.model.context(x, mask); ctx_b = self.model_b.context(x, mask)
        b0f = self.model.coeffs(ctx_f, torch.zeros(1, device=dev))[:, :, 0]
        b0b = self.model_b.coeffs(ctx_b, torch.zeros(1, device=dev))[:, :, 0]
        l_alpha = torch.einsum("tbp,tbkp->tbk", b0f, self.model.trunk(z.unsqueeze(-1))) + self.model.bias
        l_msg1 = torch.einsum("tbp,tbkp->tbk", b0b[1:], self.model_b.trunk(z_next[:-1].unsqueeze(-1))) + self.model_b.bias
        if decode is None:
            loglik1 = -0.5 * (x[1:, :, 0].unsqueeze(-1) - z_next[:-1]) ** 2 / ns ** 2
        else:
            loglik1 = -0.5 * ((x[1:].unsqueeze(2) - decode(z_next[:-1])) ** 2).sum(-1) / ns ** 2
        a_k = (l_alpha - log_q)[:-1]
        Wf = torch.softmax(a_k + loglik1, dim=-1); Ws = torch.softmax(a_k + l_msg1, dim=-1)
        zt, dzt = z[:-1], (z_next[:-1] - z[:-1]) / dt
        edges = torch.linspace(-2.0, 2.0, 41, device=dev); ctr = 0.5 * (edges[:-1] + edges[1:])
        ft = a * (ctr - ctr ** 3); onb = ctr.abs() <= 1.5

        def rmse(zc, dz, w):
            zc = zc.reshape(-1); dz = dz.reshape(-1); w = w.reshape(-1)
            idx = (torch.bucketize(zc, edges) - 1).clamp(0, 39)
            ws = torch.zeros(40, device=dev).scatter_add(0, idx, w)
            ds = torch.zeros(40, device=dev).scatter_add(0, idx, w * dz)
            fe = ds / ws.clamp_min(1e-12)
            keep = onb & (ws > ws.sum() * 0.003)
            return (fe[keep] - ft[keep]).pow(2).mean().sqrt().item() if bool(keep.any()) else float("nan")
        one = torch.ones_like(zf[:-1])
        return {"drift_l2_fm": rmse(zf[:-1][mv], ((zf[1:] - zf[:-1]) / dt)[mv], one[mv]),
                "drift_l2_sm": rmse(zs[:-1][mv], ((zs[1:] - zs[:-1]) / dt)[mv], one[mv]),
                "drift_l2_fj": rmse(zt[mv], dzt[mv], Wf[mv]),
                "drift_l2_sj": rmse(zt[mv], dzt[mv], Ws[mv])}

    # -- validation: KL vs the exact filter, drift L2, C cos, + a figure -------------------------
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        h = self.hparams
        dm = self.trainer.datamodule
        x, mask, filt = batch
        log_pi = self.model.log_posterior(x, mask, self.z_grid)
        kl = kl_target_pred(filt, log_pi).item()
        # drift error over the DATA REGIME only -- the range the inferred latent actually visits.
        # Evaluating on a fixed [-2,2] is dominated by the tails (|z|>1.5) where there is no data and
        # the regression must extrapolate; that penalizes coverage, not fit.
        # SPLIT the drift RMSE into ON-DATA (inside the range the latent visits, m_op's 1-99
        # percentile) and OFF-DATA (|z|<=2 but outside that range -- where the regression must
        # extrapolate with no data to constrain it, so the drift naturally peels off the truth).
        # Reporting them separately keeps the on-data fit from being masked by off-data divergence.
        m_op = (log_pi.exp() * self.z_grid).sum(-1)                   # (T,B) inferred latent
        lo, hi = m_op.quantile(0.01), m_op.quantile(0.99)
        on = (self.z_grid >= lo) & (self.z_grid <= hi)                # data regime
        off = self.supp & ~on                                        # |z|<=2 but outside the data regime
        fd = self.drift_net.drift(self.z_grid)[0]
        f_err = (fd[on] - self.f_true_grid[on]).pow(2).mean().sqrt().item()
        f_err_off = ((fd[off] - self.f_true_grid[off]).pow(2).mean().sqrt().item()
                     if bool(off.any()) else float("nan"))
        logs = {"kl": kl, "drift_l2": f_err, "drift_l2_off": f_err_off, "g": float(self.g_cur)}
        if h.learn_obs and dm.C_true is not None:
            logs["c_cos"] = abs(float(self.C_cur @ (dm.C_true / dm.C_true.norm())))
        # smoother marginal KL vs the oracle smoother (the v1 gate): should track `kl`.
        if h.learn_smoother and getattr(dm, "smoothed_val", None) is not None:
            log_sm = log_smoothed(self.model, self.model_b, x, mask, self.z_grid)
            logs["kl_smooth"] = kl_target_pred(dm.smoothed_val, log_sm).item()
        if h.learn_smoother:                                          # 4-way drift-source comparison (Option B)
            logs.update(self._drift_source_diag(x, mask))
        self.log_dict(logs, prog_bar=True)
        # figure
        img = os.path.join(h.train_dir, f"step_{self.global_step:05d}.pdf")
        sm_val = getattr(dm, "smoothed_val", None) if h.learn_smoother else None
        if h.learn_obs:
            viz.vis_highd(self.model, self.drift_net, self.diff_net, x, mask, dm.z_val_true, filt,
                          self.z_grid, dm.ts, self.a, self.sigma, self.C_cur, self.d_cur,
                          dm.C_true, dm.d_true, True, img, s_scale=self.s_scale, g_scalar=self.g_cur,
                          model_b=self.model_b, smoothed_val=sm_val)
        else:
            viz.vis_learn(self.model, self.drift_net, x, mask, filt, self.z_grid, dm.ts, self.a,
                          img, diff_net=self.diff_net, sigma=self.sigma,
                          model_b=self.model_b, smoothed_val=sm_val)
