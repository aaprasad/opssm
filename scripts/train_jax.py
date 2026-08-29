"""JAX-backend training entrypoint -- grid-search ready (Hydra multirun + submitit + preemption).

Run in the .[jax] venv. Reads the SAME Hydra config as scripts/train.py and builds its data IN MEMORY,
torch-free (opssm.data.*.refs) -- no bridged .npz on disk. Per run it writes, into the Hydra run dir:
  - ckpt.{eqx,pkl}   rolling resume point (overwritten; a preempted+requeued task resumes from here)
  - best.{eqx,pkl}   best monitored-metric checkpoint so far
  - metrics.csv      per-validation-step trajectory
  - result.json      final eval (whole-trace recon) + resolved hparams   (no figures are written)

Single run:
    python scripts/train_jax.py experiment=kato data.mat_path=.../WT_NoStim.mat data.worm=0 \
        model.data_size=109 train_dir=dump/kato_nostim0 backend=jax

Grid search on SLURM (one job array across all partitions, preemptible/requeue -- tasks resume on requeue):
    python scripts/train_jax.py -m hydra/launcher=submitit_slurm backend=jax experiment=kato \
        data.mat_path=.../WT_NoStim.mat data.worm=0 model.data_size=109 \
        model.w_res=0.2,0.4,0.6 model.g_init=0.5,1.0
"""
import os
import json

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


def _build_data(cfg):
    """Torch-free in-memory data build (dispatch by dataset). Returns the bridge-shaped dict of numpy arrays."""
    d = cfg.data
    if "mat_path" in d:                                              # Kato (real neural data)
        from opssm.data.kato.refs import build_kato_data
        return build_kato_data(d.mat_path, worm=int(d.get("worm", 0)), window=int(d.window),
                               stride=int(d.stride), val_frac=float(d.val_frac), noise_std=float(d.noise_std),
                               clip_negative=bool(d.get("clip_negative", True)),
                               subsample=int(d.get("subsample", 1)), latent_dim=int(d.latent_dim),
                               a=float(d.get("a", 1.0)), sigma=float(d.get("sigma", 0.1)),
                               system=str(d.get("system", "none")))
    raise SystemExit(f"train_jax's in-memory loader currently supports Kato (data.mat_path) only; "
                     f"system={cfg.data.get('system')!r}. For synthetic data, bridge to an npz + load_refs.")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg):
    if cfg.get("backend", "torch") != "jax":                        # this entrypoint runs only the jax backend
        raise SystemExit(f"cfg.backend={cfg.backend!r}: scripts/train_jax.py runs the jax backend only "
                         f"(pass backend=jax). For torch use scripts/train.py in the .[torch] venv.")
    import jax
    from opssm.models.jax.train import train, _refs_from_mapping, whole_trace_recon

    run_dir = HydraConfig.get().runtime.output_dir                  # per-run (per array-task) output dir
    D = _build_data(cfg)
    hparams = OmegaConf.to_container(cfg.model, resolve=True)        # swept model hp from the config...
    hparams.pop("_target_", None)
    hparams["data_size"] = int(D["obs_dim"])                        # ...but data_size is the ACTUAL obs count
    D["hparams"] = hparams
    refs, hp = _refs_from_mapping(D)

    n_steps = int(cfg.trainer.max_steps)
    val_every = int(cfg.trainer.val_check_interval)
    seed = int(cfg.get("seed", 0))
    ckpt_every = int(cfg.get("ckpt_every", 1000))
    monitor = str(cfg.get("monitor", "recon_r2"))
    print(f"train_jax: run_dir={run_dir} system={hp['system']} d={hp['latent_dim']} data_size={hp['data_size']} "
          f"init={hp.get('init_method')} bootstrap={hp.get('bootstrap_mstep')} n_steps={n_steps} seed={seed} "
          f"noise_std_eff={float(hp['noise_std']):.4f} ckpt_every={ckpt_every} monitor={monitor}", flush=True)

    state, hist = train(refs, hp, n_steps=n_steps, key=jax.random.PRNGKey(seed), val_every=val_every,
                        fig_dir=None, ckpt_dir=run_dir, ckpt_every=ckpt_every,
                        monitor=monitor, monitor_mode=str(cfg.get("monitor_mode", "max")))

    result = {"status": "done", "seed": seed, "n_steps": n_steps, "run_dir": run_dir,
              "hparams": hparams, "final": dict(hist[-1][1]) if hist else {}}
    if "y_full_raw" in refs:                                         # Kato: whole-trace co-smoothing recon (the pub metric)
        r2w, _ = whole_trace_recon(state["op"], state["C_cur"], state["d_cur"], refs, hp, jax.random.PRNGKey(7))
        result["whole_trace_recon_r2"] = float(r2w)
        result["g_final"] = float(state["g_cur"])
        print(f"train_jax: WHOLE-TRACE recon_r2 = {r2w:.4f}  g={float(state['g_cur']):.4f}", flush=True)
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"train_jax: done -> {os.path.join(run_dir, 'result.json')}", flush=True)
    # objective for the hydra sweeper (optuna random/TPE search): maximize whole-trace recon
    return float(result.get("whole_trace_recon_r2", (result.get("final") or {}).get("recon_r2", float("-inf"))))


if __name__ == "__main__":
    main()
