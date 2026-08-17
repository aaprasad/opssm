"""Shared data provider: byte-identical preprocessed data + ground truth for every dataset, so opssm and
all baselines see EXACTLY the same inputs. Reuses the training datamodules (single source of truth).

`make_context(name)` -> EvalContext for name in:
  doublewell (d=1, em_highd) | vanderpol (d=2) | lorenz (d=3) | kato_stim0 | kato_nostim0 (d=10).
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import torch

from opssm.eval.core import EvalContext

_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs"))

# dataset name -> (experiment config, extra hydra overrides)
DATASETS = {
    "doublewell": ("em_highd", []),
    "vanderpol":  ("em_vdp", []),
    "lorenz":     ("em_lorenz", []),
    # noise_std=0.1 matches how the kato_stim0/nostim0 checkpoints were trained (config default is 0.5)
    "kato_stim0": ("kato", ["data.mat_path=/home/aaprasad/data/kato/WT_Stim.mat",
                            "data.worm=0", "model.data_size=107", "data.noise_std=0.1"]),
    "kato_nostim0": ("kato", ["data.mat_path=/home/aaprasad/data/kato/WT_NoStim.mat",
                              "data.worm=0", "model.data_size=109", "data.noise_std=0.1"]),
}


def _np(x):
    return None if x is None else (x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x))


def _compose(experiment, overrides, scratch_dir):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="train",
                       overrides=[f"experiment={experiment}", f"train_dir={scratch_dir}", *overrides])


def make_context(name, *, device="cuda", scratch_dir=None) -> EvalContext:
    """Instantiate the training datamodule for `name`, run its EXACT preprocessing, and package the
    standardized fit/eval splits + ground truth (synthetic) or behavior labels (Kato) into an EvalContext."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    from hydra.utils import instantiate
    import lightning.pytorch as pl

    experiment, overrides = DATASETS[name]
    scratch_dir = scratch_dir or f"/tmp/opssm_eval_{name}"
    os.makedirs(scratch_dir, exist_ok=True)
    cfg = _compose(experiment, overrides, scratch_dir)
    pl.seed_everything(cfg.get("seed", 0), verbose=False)             # reproducible synthetic data

    dm = instantiate(cfg.data)
    _setup_dm(dm, device)
    return _build_context(dm, name)


def context_from_kato_checkpoint(ckpt, *, device="cuda", scratch_dir=None):
    """EvalContext from a saved opssm Kato model.pt (checkpoint-driven; used by scripts/eval_kato.py so it
    works on ANY kato checkpoint by reading its data_hparams -- the exact training preprocessing)."""
    import inspect
    from opssm.data.kato.datamodule import KatoDataModule
    d = ckpt if isinstance(ckpt, dict) else torch.load(ckpt, map_location="cpu", weights_only=False)
    dh = dict(d["data_hparams"])
    dh["train_dir"] = scratch_dir or "/tmp/opssm_eval_ckpt"
    os.makedirs(dh["train_dir"], exist_ok=True)
    valid = set(inspect.signature(KatoDataModule.__init__).parameters)
    dm = KatoDataModule(**{k: v for k, v in dh.items() if k in valid})
    _setup_dm(dm, device)
    return _build_context(dm, getattr(dm, "name", "kato"))


def _setup_dm(dm, device):
    dev = torch.device(device if torch.cuda.is_available() or "cpu" in str(device) else "cpu")
    dm.trainer = SimpleNamespace(strategy=SimpleNamespace(root_device=dev))   # setup() reads this
    dm.setup()


def _build_context(dm, name) -> EvalContext:
    obs_mean = _np(getattr(dm, "obs_mean", None))
    obs_scale = float(getattr(dm, "obs_scale", 1.0))
    common = dict(name=getattr(dm, "name", name), system=dm.hparams.get("system", "doublewell"),
                  dt=float(dm.dt), obs_mean=obs_mean, obs_scale=obs_scale,
                  noise_std_eff=float(dm.noise_std_eff),
                  obs_std_fit=_np(dm.train_batch[0]), mask_fit=_np(dm.train_batch[1]))
    is_kato = dm.z_val_true is None                                  # GT-free datamodule == Kato/real

    if is_kato:
        y_full = _np(dm.y_full)                                      # (T,N) physical (clipped)
        y_std_full = (y_full - obs_mean) / obs_scale                # (T,N) standardized full trace
        return EvalContext(
            latent_dim=int(dm.hparams.latent_dim),
            obs_std_eval=y_std_full[:, None, :].astype(np.float32),  # (T,1,N) -> adapter windows it
            mask_eval=np.ones((y_std_full.shape[0], 1, 1), np.float32),
            obs_raw_eval=y_full,
            states=_np(dm.states_full).astype(int), state_names=list(dm.state_names),
            neuron_ids=list(dm.neuron_ids),
            window=int(dm.hparams.window), stride=int(dm.hparams.stride),
            **common)
    # synthetic: ground truth available
    from opssm.data.systems import make_drift
    true_drift, _ = make_drift(dm.hparams.system)
    return EvalContext(
        latent_dim=int(_np(dm.z_val_true).shape[-1]),
        obs_std_eval=_np(dm.val_batch[0]), mask_eval=_np(dm.val_batch[1]),
        obs_raw_eval=None,
        z_true=_np(dm.z_val_true), C_true=_np(dm.C_true), d_true=_np(dm.d_true),
        true_drift=true_drift, sigma_true=float(dm.hparams.sigma),
        z_grid=_np(dm.z_grid), filt=_np(dm.val_batch[2]),
        window=None, stride=None, **common)
