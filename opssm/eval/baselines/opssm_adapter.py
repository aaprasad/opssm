"""opssm reference adapter: load a trained ZakaiFilterModule checkpoint and produce a Result.

Kato (ctx.window set): window the full standardized trace, run the mesh-free MALA filter mean per window,
stitch overlaps -- byte-identical to scripts/eval_kato.filter_full_trace. Synthetic (window None): filter
the val batch directly. opssm's posterior is a CAUSAL FILTER (fairness tag).
"""
import inspect
import time

import numpy as np
import torch

from opssm.eval.core import Result
from opssm.models.filter_module import ZakaiFilterModule
from opssm.models.mstep import filter_mean
from opssm.models.obs import zhat_from_obs


class OpssmAdapter:
    name = "opssm"
    kind = "torch"

    def _load(self, ckpt_path, dev):
        d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp = d["hparams"]
        args = set(inspect.signature(ZakaiFilterModule.__init__).parameters)
        model = ZakaiFilterModule(**{k: v for k, v in hp.items() if k in args})
        model.load_state_dict(d["state_dict"], strict=False)
        model.C_cur, model.d_cur, model.g_cur = d["C_cur"].to(dev), d["d_cur"].to(dev), d["g_cur"]
        return model.to(dev).eval()

    @torch.no_grad()
    def _filter_full(self, model, y_std, window, stride, dev):
        """y_std (T,N) -> z_hat (T,d): windowed MALA filter means, overlaps averaged (== eval_kato)."""
        T, N = y_std.shape
        d = model.model.latent_dim
        h = model.hparams
        starts = list(range(0, T - window + 1, stride)) or [0]
        x = torch.stack([y_std[s:s + window] for s in starts], dim=1).to(dev)   # (window,B,N)
        mask = torch.ones(x.shape[0], x.shape[1], 1, device=dev)
        center = zhat_from_obs(x, model.C_cur, model.d_cur)
        zc, _ = filter_mean(model.model, x, mask, center, method="mala", n_mean=h.n_mean,
                            near_std=h.near_std, broad_std=h.broad_std, mala=model._mala_cfg())
        acc = torch.zeros(T, d, device=dev); cnt = torch.zeros(T, 1, device=dev)
        for b, s in enumerate(starts):
            L = min(window, T - s)
            acc[s:s + L] += zc[:L, b]; cnt[s:s + L] += 1
        return (acc / cnt.clamp_min(1)).cpu().numpy()                           # (T,d)

    @torch.no_grad()
    def fit_predict(self, ctx, cfg, device="cuda"):
        ckpt = cfg.get("checkpoint")
        if ckpt is None:
            raise ValueError("opssm adapter needs cfg['checkpoint'] (a trained model.pt); "
                             "synthetic-from-scratch training is added in Phase 4")
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        model = self._load(ckpt, dev)
        h = model.hparams
        C = model.C_cur.detach().cpu().numpy(); d_off = model.d_cur.detach().cpu().numpy()
        t0 = time.time()
        if ctx.window is not None:                                             # Kato: window + stitch
            y_std = torch.from_numpy(ctx.obs_std_eval[:, 0, :]).float().to(dev)
            z_hat = self._filter_full(model, y_std, ctx.window, ctx.stride, dev)   # (T,d)
            window_mode = "windowed"
        else:                                                                  # synthetic: filter the val batch
            x = torch.from_numpy(ctx.obs_std_eval).float().to(dev)
            mask = torch.from_numpy(ctx.mask_eval).float().to(dev)
            center = zhat_from_obs(x, model.C_cur, model.d_cur)
            zc, _ = filter_mean(model.model, x, mask, center, method=h.mean_method, n_mean=h.n_mean,
                                near_std=h.near_std, broad_std=h.broad_std, mala=model._mala_cfg())
            z_hat = zc.detach().cpu().numpy()                                   # (T,B,d)
            window_mode = "whole"
        y_hat = z_hat @ C.T + d_off                                            # standardized units
        drift_fn = lambda z: model.drift_net.net(
            torch.as_tensor(np.asarray(z), dtype=torch.float32, device=dev)).detach().cpu().numpy()
        return Result(z_hat=z_hat, y_hat=y_hat, drift_fn=drift_fn, g=float(model.g_cur),
                      posterior_type="filter", window_mode=window_mode, runtime_s=time.time() - t0)
