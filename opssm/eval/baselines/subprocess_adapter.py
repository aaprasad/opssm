"""Generic subprocess+npz adapter: run a baseline that lives in ANOTHER venv (torch baselines, or the JAX
env in Phase 3) without importing it in-process. The seam is numpy arrays, so framework clashes (torch vs
jax CUDA/versions) are impossible.

Contract:
  input.npz   (written here)  : obs_std_fit (T,Bf,N), mask_fit, obs_std_eval (T,Be,N), mask_eval,
                                dt, latent_dim, noise_std_eff, window (or -1), stride (or -1), + scalar cfg.
  output.npz  (written by worker): z_hat, y_hat (standardized), g, [drift_at_zhat], [posterior_type],
                                [window_mode], [runtime_s]. All numpy.
A worker is a STANDALONE script (only its model's deps + numpy) run as `<venv_python> <worker.py> in out`.
"""
import os
import subprocess
import time

import numpy as np

from opssm.eval.core import Result


class SubprocessAdapter:
    kind = "subprocess"

    def __init__(self, name, venv_python, worker, posterior_type="smoother"):
        self.name = name
        self.venv_python = os.path.expanduser(venv_python)
        self.worker = os.path.abspath(worker)
        self.posterior_type = posterior_type

    def fit_predict(self, ctx, cfg=None, device="cuda"):
        cfg = cfg or {}
        scratch = cfg.get("scratch_dir", "/tmp/opssm_baselines")
        os.makedirs(scratch, exist_ok=True)
        uid = f"{self.name}_{ctx.name}".replace("/", "_").replace(" ", "_")
        fin, fout = f"{scratch}/{uid}_in.npz", f"{scratch}/{uid}_out.npz"
        payload = dict(
            obs_std_fit=ctx.obs_std_fit.astype(np.float32), mask_fit=ctx.mask_fit.astype(np.float32),
            obs_std_eval=ctx.obs_std_eval.astype(np.float32), mask_eval=ctx.mask_eval.astype(np.float32),
            dt=np.float32(ctx.dt), latent_dim=np.int64(ctx.latent_dim),
            noise_std_eff=np.float32(ctx.noise_std_eff),
            window=np.int64(ctx.window if ctx.window else -1),
            stride=np.int64(ctx.stride if ctx.stride else -1))
        if getattr(ctx, "sigma_true", None) is not None:              # GT diffusion (synthetic); gpSLDS/SING-GP
            payload["sigma_true"] = np.float32(ctx.sigma_true)        # set the fit's sigma to it, as their demo does
        payload.update({k: v for k, v in cfg.items() if np.isscalar(v) and k != "scratch_dir"})
        np.savez(fin, **payload)

        env = dict(os.environ)
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        t0 = time.time()
        r = subprocess.run([self.venv_python, self.worker, fin, fout],
                           capture_output=True, text=True, env=env)
        if r.returncode != 0 or not os.path.exists(fout):
            raise RuntimeError(f"[{self.name}] worker failed on {ctx.name} (rc={r.returncode}):\n"
                               f"--- stdout ---\n{r.stdout[-3000:]}\n--- stderr ---\n{r.stderr[-3000:]}")
        d = np.load(fout, allow_pickle=True)
        return Result(
            z_hat=d["z_hat"], y_hat=d["y_hat"],
            drift_at_zhat=d["drift_at_zhat"] if "drift_at_zhat" in d.files else None,
            z_cov=d["z_cov"] if "z_cov" in d.files else None,
            g=float(d["g"]) if "g" in d.files else None,
            posterior_type=str(d["posterior_type"]) if "posterior_type" in d.files else self.posterior_type,
            window_mode=str(d["window_mode"]) if "window_mode" in d.files else "whole",
            runtime_s=float(d["runtime_s"]) if "runtime_s" in d.files else (time.time() - t0))
