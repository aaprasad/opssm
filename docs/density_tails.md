# Normalizable forward density experiment

The bounded tanh trunk originally emits a bounded log-density on unrestricted latent space. Its
exponential has infinite integral. Finite-grid normalization can obscure this problem, and MALA has no
proper global target. This experiment adds a fixed Gaussian reference to the forward log-density:

```text
ell(z, context, s) = neural_residual(z, context, s) - |z|² / (2 tail_std²)
```

The residual remains bounded for finite weights, so the density is integrable for every context and
within-interval time. It can still represent multimodal posteriors. The first version uses a fixed,
isotropic, zero-centered reference: no learned mean, precision or additional parameters.

## Implementation

- `model.tail_std=2.0` enables the experiment in both Torch and JAX; `0.0` preserves the legacy behavior
  and remains the default while the experiment is evaluated.
- `state_basis(z)` appends the Gaussian feature to the learned trunk. `coeffs` appends a constant one,
  and `coeffs_dtime` appends a zero to the time derivative. Losses, spatial derivatives, grid/SNIS/MALA
  readouts, smoothing and diagnostics therefore all use the same log-density.
- Effective basis width is `p+1` when enabled; network parameter names, shapes and initialization are
  unchanged. Custom callers should use `state_basis`, not the raw `trunk`, with `coeffs`.
- Backward likelihood messages retain their original representation. They need not be integrable in
  latent space; the forward density supplies the smoother's tails.
- Use separate output directories for different tail settings. JAX resume checks the saved tail scale
  and rejects a mismatch. Old checkpoints without this metadata are treated as legacy (`tail_std=0`).
- This changes the density model and its PDE derivatives, not the physical initial prior or SDE diffusion.

## Controlled inference comparison

Run in the JAX environment, from the repository root:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  python scripts/compare_density_tails.py --steps 1000 --seeds 0 1 2 \
  --tail-stds 0 2 --out dump/density_tails
```

The script holds the true double-well drift, diffusion (0.6) and direct sensor fixed, and changes only
the density representation. All variants share 16 training trajectories, 8 validation trajectories,
40 observation times at dt=0.1, observation noise 0.2, and a five-observation gap. Seeds vary initialization
and collocation draws, with each baseline/tail pair matched. The reference is a 401-point grid filter
on [-3,3] with 10 Euler-Maruyama substeps. Training uses width 32, 64 collocation points, `w_res=0.2`,
and 1,000 Adam steps at learning rate 0.002. The M-step is excluded to isolate inference quality.

Results on CPU, 2026-09-09 (means over the three initialization seeds):

| Metric | Legacy | Gaussian tail, std=2 |
|---|---:|---:|
| Grid filtering KL | 0.15824 | 0.13657 |
| Grid KL during the gap | 0.11126 | 0.06080 |
| Posterior mean RMSE against grid oracle | 0.10041 | 0.04066 |
| Mean posterior standard deviation | 0.28106 | 0.25751 |
| MALA mean RMSE against model's grid mean | 0.24914 | 0.15252 |
| MALA acceptance | 0.57545 | 0.57315 |

Oracle mean posterior standard deviation: 0.16857. Per-seed grid KL (legacy -> tail):
0.14890 -> 0.14107; 0.12977 -> 0.12027; 0.19606 -> 0.14838.

Filtering KL improved about 14%, gap KL about 45%, and oracle-mean RMSE about 60% in this small experiment.
All three seeds improved on these three metrics. This is an initial inference result on one shared
dataset, not evidence of improved drift/diffusion recovery or broad generalization. The posterior remains
too wide, and MALA still differs materially from quadrature despite healthy acceptance. The legacy KL is
only a finite-grid comparison because its global density is improper. The grid oracle is also a numerical
approximation; discretization convergence was not established in this experiment.

Raw per-seed metrics, training histories, finite-domain mass diagnostics and Equinox weights are written
under `dump/density_tails/` (ignored by Git). The JSON records experiment settings. For an EM trial:

```bash
python scripts/train_jax.py backend=jax experiment=em_highd model.tail_std=2.0 \
  hydra.run.dir=dump/em_highd_gaussian_tail
# Torch uses the same override:
python scripts/train.py experiment=em_highd model.tail_std=2.0 train_dir=dump/torch_gaussian_tail
```

## Verification

In their respective backend environments:

```bash
PYTHONPATH=. python tests/test_density_tails.py -v
JAX_PLATFORMS=cpu PYTHONPATH=. python tests/test_density_tails_jax.py -v
PYTHONPATH=. python tests/test_mala.py
```

Checks cover Gaussian normalization, spatial gradients and Laplacians in d=1 and d=3, the zero time
derivative of the fixed tail, legacy weight compatibility, finite training gradients, Torch smoother and
pair readouts, sampling an actual Gaussian operator, and JAX checkpoint tail compatibility.

Tiny end-to-end training runs passed in both backends, including the Torch backward smoother and JAX
M-step. The Torch smoother smoke exposed an existing metric bug: mean-only alignment tried to evaluate
KL with a missing grid density. Its KL calculation now requires both density inputs; a regression test
covers this case. These two-step smoke runs establish wiring, not convergence or parameter recovery.
