# JAX grid search on SLURM (preemptible)

Large hyperparameter sweeps for the JAX backend, via Hydra multirun + the submitit launcher. Each grid point
is one **array task = 1 GPU**; tasks build their data **in memory** (no `.npz` on disk) and **checkpoint/resume**,
so preempted tasks are requeued by SLURM and pick up where they left off.

## One-time setup

Install the JAX backend into its own venv (separate from the torch venv — the two pin conflicting CUDA runtimes):

```bash
pip install -e '.[jax]'      # jax[cuda12] + equinox + optax + hydra + hydra-submitit-launcher (+ core: h5py, scipy, ...)
```

Then fill the site-specific placeholders in [`configs/hydra/launcher/submitit_slurm.yaml`](../configs/hydra/launcher/submitit_slurm.yaml):

- `partition:` — **required**, comma-separated list of all your partitions, e.g. `"a100,h100,l40s"` (SLURM places
  each task wherever there's a free GPU). Leave the rest (`account`, `qos`, `constraint`, `gres`) unless your
  cluster needs them.
- `array_parallelism:` — max tasks running at once (defaults to 92 ≈ your 20 low-preempt + 72 high-preempt GPUs).

## Launch a sweep

From the `.[jax]` venv:

```bash
python scripts/train_jax.py -m \
    hydra/launcher=submitit_slurm backend=jax \
    experiment=kato data.mat_path=/path/WT_NoStim.mat data.worm=0 model.data_size=109 \
    model.w_res=0.2,0.4,0.6 model.g_init=0.5,1.0 model.drift_lr=1e-3,3e-3
```

`-m` (multirun) submits the cross-product (here 3×2×3 = 18 tasks) as one job array. Per-task outputs land in
`multirun/<date>/<time>/<task>/`. Common knobs: `trainer.max_steps` (default 14000), `ckpt_every` (default 1000),
`monitor`/`monitor_mode` (default `recon_r2`/`max`, selects `best.eqx`).

## Preemption & resume

Resilience is SLURM `--requeue` + our resume -- deliberately **not** signal-catching:

- The launcher sets `additional_parameters.requeue: true`; when a task is preempted (killed), SLURM requeues it as
  the **same** array task, so it reruns with the **same run dir** and `train_jax` resumes from `ckpt.*` there.
- We do **not** install a SIGUSR1/SIGTERM handler. submitit bypasses SIGTERM and does not requeue non-checkpointable
  jobs on its signal, so catching signals would only fight the scheduler. Instead `train()` checkpoints every
  `ckpt_every` steps; a preemption loses at most that many steps (~35 s at the default 1000 -- lower it for the
  high-preemption tier if you want less lost work).
- Resume is faithful: the full EM state, RNG key, and history are checkpointed, so a resumed run continues its own
  trajectory exactly. (Two *independent* fresh runs still differ ~1e-3 from JAX-GPU nondeterminism -- expected.)

## Per-run outputs (in each task's run dir)

| file | what |
|---|---|
| `ckpt.{eqx,pkl}` | rolling resume checkpoint (overwritten every `ckpt_every`) |
| `best.{eqx,pkl}` | best `monitor`-metric checkpoint so far |
| `metrics.csv` | per-validation-step trajectory |
| `result.json` | final eval (whole-trace recon) + resolved hyperparameters |

No data `.npz` and no figures are written.

## Aggregate

```bash
python scripts/aggregate_sweep.py multirun/<date>/<time> --sort whole_trace_recon_r2 --csv sweep.csv
```

Prints every finished run ranked best-first, showing only the hyperparameters that varied. Still-running or
preempted tasks (no `result.json` yet) are simply skipped.

## Single run (debugging, no SLURM)

```bash
python scripts/train_jax.py experiment=kato backend=jax \
    data.mat_path=/path/WT_NoStim.mat data.worm=0 model.data_size=109 \
    trainer.max_steps=300 ckpt_every=100 hydra.run.dir=/tmp/tj hydra.output_subdir=null
```
