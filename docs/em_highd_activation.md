# Softplus operator trunks

Softplus is the default hidden activation in the state-dependent trunk of both the Torch and JAX
operators. The branch, GRU and dynamics networks retain their original activations. The log density
is the learned branch/trunk inner product, with a linear trunk output and no fixed Gaussian term.
Softplus allows unbounded log densities and smooth spatial derivatives for the PDE loss. It allows
learned decaying tails without guaranteeing normalizability for arbitrary weights.

`model.trunk_activation=tanh` retains the previous representation for baselines and old checkpoints.
The activation changes neither the parameter shapes nor their initialization draws. The Torch backward
operator uses the same configured trunk activation. The Gaussian-tail option and its extra basis
feature have been removed, along with the reduced experimental training harness.

## Configuration and checkpoints

Use the standard training entrypoint and experiment configuration:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
  python scripts/train_jax.py experiment=em_highd backend=jax
```

Torch uses the same model override through `scripts/train.py`. JAX checkpoints record the trunk
activation and reject resume mismatches; Torch restores Lightning hyperparameters and checks the
activation when resuming an existing model. Explicit Lightning hyperparameter overrides retain their
usual behavior. Checkpoints
without activation metadata are treated as tanh, so explicitly use `model.trunk_activation=tanh`
when resuming those runs (or `trunk_activation="tanh"` with Lightning's `load_from_checkpoint`).
Nonzero Gaussian-tail checkpoints are rejected rather than silently interpreted as a different
model; use the experiment revision that created them if they need to be inspected. Checkpoints from
the softplus/tanh comparisons with the Gaussian term disabled remain compatible.

## Matched saved-data experiment

The final comparison uses `train_jax.py experiment=em_highd`, the same saved NPZ data and seed 0.
The resolved configurations differ only in trunk activation. Both completed 14,000 steps on an
NVIDIA GeForce RTX 5090 on 2026-09-09, run sequentially:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
  python scripts/train_jax.py experiment=em_highd backend=jax model.init_method=random \
  model.trunk_activation=softplus +data.npz_path=dump/jax_port/em_highd.npz \
  train_dir=dump/em_highd_repro
# Repeat with model.trunk_activation=tanh for the baseline.
```

The saved dataset has 224 training and 32 validation trajectories, 100 time points, true drift
`z-z^3`, diffusion 0.6, and a learned 10-dimensional linear sensor. There is no observation gap.
Training uses the configured exponential learning-rate decay, residual weight 0.4, and 400-update
M-steps every 2,000 steps. Trunk and drift width/depth are 64/3; branch width is 128. The explicit
random initialization disables the subspace drift warm start; the configured bootstrap M-step
remains enabled, with warmup 0. Effective observation noise is 0.1353 in standardized coordinates.

| Final aligned metric | Tanh | Softplus |
|---|---:|---:|
| Relative drift error | 0.21627 | 0.21511 |
| Relative latent error | 0.11603 | 0.11656 |
| Filtering KL | 0.04841 | 0.02086 |
| Diffusion (true: 0.6) | 0.55419 | 0.54911 |

Drift recovery is effectively tied, near the historical 0.207 benchmark. Softplus lowers aligned
filtering KL by 57% on this seed, with similar latent accuracy. This is a single-seed result, not
established evidence of improved dynamics learning. The table uses the final step for both models,
not the checkpoint with the best drift error. All final metrics were finite.

Posterior diagnostics explain why lower KL need not improve latent RMSE: softplus improves the
posterior width while the point estimate stays similar. Even the grid oracle has nonzero trajectory
error because the observations are noisy.

| Diagnostic | Grid oracle | Tanh | Softplus |
|---|---:|---:|---:|
| Mean posterior standard deviation | 0.10386 | 0.13011 | 0.11698 |
| Posterior mean RMSE against oracle | 0 | 0.01417 | 0.01526 |
| Aligned latent RMSE against true paths | 0.10348 | 0.10419 | 0.10467 |

These use the same finite grid and affine alignment as the standard validation metrics. The oracle
is a numerical approximation; this experiment does not establish discretization convergence.

Hydra saves results/checkpoints under its run directory, independently of `train_dir`. The saved-data
softplus run is `outputs/2026-09-09/14-28-20`; tanh is `outputs/2026-09-09/14-40-41`. To compare them:

```bash
MPLBACKEND=Agg python scripts/plot_training_comparison.py \
  outputs/2026-09-09/14-40-41 outputs/2026-09-09/14-28-20 --out dump/em_highd_repro
```

The plot script checks matching data, seed and settings. It writes `comparison.png`, `comparison.pdf`
and `summary.csv`. Posterior diagnostics are in `dump/em_highd_repro/posterior_diagnostics.json`.
Run artifacts are ignored by Git.

## Verification

Run each backend's tests in its own environment:

```bash
PYTHONPATH=. python tests/test_operator_activation.py -v
JAX_PLATFORMS=cuda PYTHONPATH=. python tests/test_operator_activation_jax.py -v
PYTHONPATH=. python tests/test_mala.py
```

Checks cover analytic softplus derivatives, multidimensional log-density and time derivatives,
differentiable PDE losses, MALA sampling of a known logistic density represented by the softplus
operator, Torch smoothing/readouts, default and legacy activation behavior, and checkpoint metadata.
