# Trunk activation comparison with the standard JAX EM trainer

This experiment uses `scripts/train_jax.py` with `experiment=em_highd`. The tanh and softplus runs
use seed 0 and the current resolved configuration, with `model.tail_std=0.0` in both. The only
configuration differences are the trunk activation and output paths.

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  python scripts/train_jax.py backend=jax experiment=em_highd seed=0 \
  model.trunk_activation=tanh model.tail_std=0.0 \
  train_dir=dump/em_highd_activation_gpu/tanh hydra.run.dir=dump/em_highd_activation_gpu/tanh

JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  python scripts/train_jax.py backend=jax experiment=em_highd seed=0 \
  model.trunk_activation=softplus model.tail_std=0.0 \
  train_dir=dump/em_highd_activation_gpu/softplus hydra.run.dir=dump/em_highd_activation_gpu/softplus

MPLBACKEND=Agg python scripts/plot_training_comparison.py \
  dump/em_highd_activation_gpu/tanh dump/em_highd_activation_gpu/softplus \
  --out dump/em_highd_activation_gpu
```

Both train on the same generated dataset: 224 training and 32 validation trajectories, 100 time
points, true drift `z-z^3`, diffusion 0.6, and a learned 10-dimensional linear sensor. There is no
observation gap. Training runs for 14,000 steps with the configured exponential learning-rate decay,
residual weight 0.4, and a 400-update M-step every 2,000 steps. The trunk and drift networks use width
64 and three hidden layers; branch width is 128. Current defaults enable subspace initialization,
drift warm-starting, and a bootstrap M-step.

The standard trainer reports affine-aligned `drift_rel`, `lat_rel`, `g_aln` and `kl_aln`. These are
appropriate for the learned sensor's latent coordinates; raw diffusion and raw filtering KL are
also retained in each result file. Final-step results are compared, with no selection of the best
drift-error checkpoint. Checkpoints, complete resolved Hydra configs, logs, validation histories and
`result.json` are saved in each run directory.

This is a matched comparison under the current JAX `em_highd` defaults. The historical Torch result
of 0.207 used different initialization defaults and a different random dataset. This run therefore
does not exactly reproduce that historical result. It also does not use the reduced custom harness
in `compare_density_tails.py`; see [the earlier experiment audit](density_tails.md) for that distinction.

## Results

Both runs completed all 14,000 steps on the NVIDIA GeForce RTX 5090 on 2026-09-09. After initially
overlapping, softplus was paused in memory (its last saved checkpoint was step 5,000), tanh finished,
and then softplus resumed to completion. No training progress was discarded. These runs should not
be used to compare throughput, because their execution included that overlap and pause.

| Final metric | Tanh | Softplus |
|---|---:|---:|
| Aligned relative drift error | 0.21782 | 0.23571 |
| Aligned relative latent error | 0.13964 | 0.13997 |
| Aligned filtering KL | 0.04211 | 0.01929 |
| Aligned diffusion (true: 0.6) | 0.55961 | 0.54301 |
| Raw learned diffusion | 0.61732 | 0.59700 |
| Observation reconstruction R² | 0.80401 | 0.80298 |

The standard tanh run recovers drift near the historical 0.207 benchmark. In this single-seed
comparison, softplus reduces aligned filtering KL by 54%, but worsens final drift error by 8%.
Latent accuracy is essentially unchanged, and aligned diffusion moves further below truth. Raw
softplus diffusion being close to 0.6 is not evidence of better diffusion recovery: it must be
transformed into the true latent coordinates using the fitted alignment, as above.

The intermediate drift errors were close and occasionally favored softplus; this table uses the
preselected final step for both. The result supports improved filtering KL under this configuration,
not improved dynamics learning. More seeds would be needed to assess how repeatable the difference is.

All final metrics were finite and both runs recorded every expected validation checkpoint. The plot
script checks that data, seed and training settings match apart from activation/output paths before
comparing results. Training histories are plotted in `dump/em_highd_activation_gpu/comparison.png`
and `.pdf`; `summary.csv` contains all final metrics. These artifacts and raw run directories are
ignored by Git.
