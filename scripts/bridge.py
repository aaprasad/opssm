"""Bridge the torch datamodule + eval oracle to a backend-agnostic .npz for the JAX backend.

The .[jax] venv has no torch, so JAX training cannot instantiate the (torch/Lightning) datamodule in
process. This script -- run in the .[torch] venv -- composes the SAME Hydra config as scripts/train.py,
builds the datamodule, and dumps one <train_dir>/bridge.npz holding the train/val batches, the resolved +
data-derived hyperparameters (noise_std_eff, dt, data_size, ... which only the datamodule can compute), and
the eval refs. scripts/train_jax.py (in the .[jax] venv) then trains from that npz.

    python scripts/bridge.py experiment=kato data.mat_path=.../WT_NoStim.mat data.worm=0 \
        model.data_size=109 train_dir=dump/kato_nostim0 backend=jax          # .[torch] venv

For d>1 real/synthetic systems without a grid oracle (vdp/lorenz/kato) z_grid/filt_val are absent (handled
downstream). 1:1 with the validated dump/jax_port/bridge.py, promoted to an official entrypoint.
"""
import os
from types import SimpleNamespace

import hydra
import numpy as np
import torch
import lightning.pytorch as pl
from hydra.utils import instantiate

HP_KEYS = ["data_size", "latent_dim", "gru_hidden", "ctx_dim", "p", "branch_hidden", "trunk_hidden",
           "trunk_layers", "drift_hidden", "drift_layers", "lr", "drift_lr",
           "sched_gamma", "n_scoll", "n_tcoll", "n_colloc", "near_std", "broad_std", "warmup", "m_every",
           "m_inner", "reg_lambda", "g_init", "w_res", "res_mode", "learn_obs", "learn_g", "c_stable_tol",
           "n_mean", "mean_method", "mala_chains", "mala_steps", "mala_rng", "drift_target",
           "init_method", "init_dynamics", "bootstrap_mstep", "pca_init"]


def _np(t):
    return None if t is None else np.asarray(t.detach().cpu().numpy())


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg):
    pl.seed_everything(cfg.get("seed", 0), verbose=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = instantiate(cfg.data)
    dm.trainer = SimpleNamespace(strategy=SimpleNamespace(root_device=dev))
    dm.setup("fit")
    model = instantiate(cfg.model)                                   # only its resolved hparams are exported
    h = dict(model.hparams)

    xtr, mtr, ftr = dm.train_batch
    xva, mva, fva = next(iter(dm.val_dataloader()))

    hp = {k: h[k] for k in HP_KEYS if k in h}
    hp["data_size"] = int(dm.hparams.obs_dim)                        # actual neuron/obs count (e.g. Kato worm N)
    hp["system"] = str(dm.hparams.system)

    out = dict(
        seed=np.array(cfg.get("seed", 0)), backend=str(cfg.get("backend", "jax")),
        n_steps=np.array(int(cfg.trainer.max_steps)),                # training-loop knobs (config-resolved here)
        val_every=np.array(int(cfg.trainer.val_check_interval)),
        system=str(dm.hparams.system),
        x_train=_np(xtr), mask_train=_np(mtr), x_val=_np(xva), mask_val=_np(mva),
        full_obs=_np(dm.full_obs),
        dt=np.asarray(float(dm.dt)), noise_std_eff=np.asarray(float(dm.noise_std_eff)),
        a=np.asarray(float(getattr(dm.hparams, "a", 1.0))), sigma=np.asarray(float(dm.hparams.sigma)),
        obs_dim=np.array(int(dm.hparams.obs_dim)),
        z_val_true=_np(dm.z_val_true), C_true=_np(dm.C_true), d_true=_np(getattr(dm, "d_true", None)),
        obs_mean=_np(getattr(dm, "obs_mean", None)), obs_scale=np.asarray(float(getattr(dm, "obs_scale", 1.0))),
        ts=_np(dm.ts), hparams=np.array(hp, dtype=object),
    )
    for k, v in [("filt_val", fva), ("z_grid", getattr(dm, "z_grid", None))]:   # d==1 grid oracle only
        if v is not None:
            out[k] = _np(v) if torch.is_tensor(v) else np.asarray(v)

    if getattr(dm, "states_full", None) is not None:                # Kato: behavior labels + whole-trace recon refs
        T_full = int(dm.y_full.shape[0])
        n_val_t = int(dm.hparams.val_frac * T_full)
        off = T_full - n_val_t
        win = int(xva.shape[0])
        sf = np.asarray(dm.states_full)
        out["states_val"] = np.stack([sf[off + s: off + s + win] for s in dm.window_starts_val], axis=1)
        out["state_names"] = np.array([str(x) for x in dm.state_names], dtype=object)
        y_full_std = (dm.y_full - dm.obs_mean) / dm.obs_scale        # whole-trace co-smoothing (== _filter_full)
        out["y_full_std"] = _np(y_full_std)
        out["y_full_raw"] = _np(dm.y_full)
        out["window"] = np.array(int(dm.hparams.window))
        out["stride"] = np.array(int(dm.hparams.stride))
        out["states_full"] = np.asarray(dm.states_full)
        out["neuron_ids"] = np.array([(s if (s := str(x)) and not s.isdigit() else "") for x in dm.neuron_ids],
                                     dtype=object)                   # keep named neurons, blank numeric placeholders
        out["name"] = str(getattr(dm, "name", str(dm.hparams.system)))

    out = {k: v for k, v in out.items() if v is not None}
    os.makedirs(cfg.train_dir, exist_ok=True)
    npz_path = os.path.join(cfg.train_dir, "bridge.npz")
    np.savez(npz_path, **out)
    print(f"bridged -> {npz_path}  backend={out['backend']} system={out['system']} d={h['latent_dim']} "
          f"data_size={hp['data_size']} dt={float(out['dt']):.5f} noise_std_eff={float(out['noise_std_eff']):.5f}")


if __name__ == "__main__":
    main()
