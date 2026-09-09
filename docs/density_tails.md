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

## Learned-dynamics comparison

This is a small custom ablation, not a reproduction of the established `em_highd` double-well
benchmark (drift relative error 0.207). Its poor drift recovery limits conclusions about changes to
that benchmark. See the benchmark audit below.

The same harness can learn drift and scalar diffusion with the existing MALA/`det_mid` M-step while
keeping the direct observation model fixed. This isolates dynamics learning from sensor identification.
Ground-truth drift and diffusion are used for simulation and evaluation, not as training targets.

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  python scripts/compare_density_tails.py --learn-dynamics --steps 6000 --seeds 0 1 2 \
  --tail-stds 0 2 --out dump/density_tails_em_gpu
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg PYTHONPATH=. \
  python scripts/plot_density_tails_em.py dump/density_tails_em_gpu/results.json
```

Both variants start with zero drift and g=1.0, with no supervised warm start or bootstrap. After 1,000
operator steps, the M-step runs every 500 steps with 200 drift-optimizer steps, learning rate 0.002,
weight penalty 0.0003, and 64 MALA chains with 30 sweeps. Ten M-steps occur at steps 1,000 through 5,500;
the final 500 operator steps let inference adjust to the final dynamics. All other data/architecture/loss
settings match the controlled comparison above. M-step random keys are matched separately from E-step
keys, so both variants receive identical random draws at corresponding updates.

The report includes g, absolute diffusion error, drift RMSE and relative error evaluated at true held-out
states with |z| <= 1.5, and latent RMSE against the actual simulated paths. The known direct sensor fixes
the coordinates, so no affine alignment is used. Drift weights, operator weights, M-step histories and
device information are saved with the results. Plotting writes PNG/PDF figures and a per-seed CSV table.

Results on NVIDIA GeForce RTX 5090, 2026-09-09 (means over three initialization seeds):

| Metric | Legacy | Gaussian tail, std=2 |
|---|---:|---:|
| Grid filtering KL | 0.04520 | 0.04409 |
| Grid KL during the gap | 0.07989 | 0.08452 |
| Latent RMSE against simulated paths | 0.16918 | 0.16872 |
| Relative drift RMSE | 0.86578 | 0.83776 |
| Learned diffusion g (true: 0.6) | 0.47326 | 0.47277 |
| MALA mean RMSE against model's grid mean | 0.14641 | 0.02010 |

The tail substantially improves sampler/grid agreement, but dynamics recovery improves only slightly:
relative drift error drops about 3%, while diffusion remains about 21% below truth. Filtering KL improves
about 2%, and gap KL worsens about 6%. Drift error improves in all three seeds; filtering KL improves in
two of three. The learned drift curves still differ materially from the true double-well drift.

These six runs establish that the tail works with learned dynamics, not that it solves parameter
identification. They share one small dataset and a known sensor. The existing mean-based M-step still
discards posterior uncertainty; testing an uncertainty-aware transition objective is a useful next
experiment, although this comparison does not isolate the cause of the remaining parameter bias.
The finite-grid and oracle limitations above also apply here. All six runs completed ten M-steps with
finite logged losses and final scalar metrics. Results, checkpoints, CSV and plots are saved under
`dump/density_tails_em_gpu/` (ignored by Git).

## Softplus activation comparison

`model.trunk_activation=softplus model.tail_std=0.0` replaces every hidden tanh in the state-basis
trunk with softplus, leaving the output layer linear. The branch, GRU and drift network retain their
original activations. This option is supported by both backends; tanh remains the default. Parameter
shapes and initialization draws are unchanged. Torch saves the choice in Lightning hyperparameters;
JAX saves it in checkpoint metadata and rejects a resume with a different activation. Missing metadata
in older JAX checkpoints means tanh.

Softplus is smooth and unbounded, allowing log-density tails that decrease without a fixed Gaussian
term. The learned coefficients still determine whether both tails decrease; the activation alone
does not guarantee normalizability. A Gaussian term can also be combined with softplus, since the
quadratic dominates its at-most-linear growth, but that combination was not tested in this ablation.

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  python scripts/compare_density_tails.py --learn-dynamics --steps 6000 --seeds 0 1 2 \
  --tail-stds 0 --trunk-activation softplus --out dump/softplus_em_gpu
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg PYTHONPATH=. \
  python scripts/plot_density_tails_em.py dump/softplus_em_gpu/results.json \
  --compare-results dump/density_tails_em_gpu/results.json
```

Three softplus runs completed on the RTX 5090 with the same data, seeds, learning rates and EM schedule
as the learned-dynamics comparison. The table reuses the matched tanh baseline above; neither column
uses the Gaussian term. Means over the three initialization seeds:

| Metric | Tanh | Softplus |
|---|---:|---:|
| Grid filtering KL | 0.04520 | 0.05002 |
| Grid KL during the gap | 0.07989 | 0.09978 |
| Latent RMSE against simulated paths | 0.16918 | 0.16893 |
| Relative drift RMSE | 0.86578 | 0.85014 |
| Learned diffusion g (true: 0.6) | 0.47326 | 0.48286 |
| MALA mean RMSE against model's grid mean | 0.14641 | 0.00350 |

Softplus improves MALA/grid agreement about 98%, also outperforming the Gaussian-tail variant on this
diagnostic (0.02010). Latent RMSE is essentially unchanged. Drift error improves about 2%, and diffusion
is slightly closer to truth but remains underestimated. Filtering KL worsens about 11%, and gap KL
about 25%. This supports the activation swap as a way to improve sampling in this experiment, without
establishing a substantial improvement in dynamics recovery. It remains a small, shared-data comparison
with a known sensor; sampler/grid agreement does not prove global normalizability.

All three runs completed ten M-steps with finite logged losses and final scalar metrics. Analytic
softplus value/gradient/Laplacian checks, differentiable PINN losses and checkpoint compatibility pass
in the backend tests. The JAX analytic tests request full-precision GPU matrix multiplication to meet
their existing 1e-6 tolerances; the experiments use the same default precision as the earlier GPU runs.
Raw results, weights and comparison plots are in `dump/softplus_em_gpu/` (ignored by Git).

## Audit against the earlier 0.207 double-well result

The earlier result is confirmed in `dump/dm_em_highd.log`: final step 14,000 reports `drift_rel=0.207`,
`drift_rmse_aln=0.0908`, `g=0.611`, and `g_aln=0.549`. Its saved configuration is
`outputs/2026-08-12/10-27-35/.hydra/config.yaml`; the corresponding implementation is in commit
`fbe1f92`. The activation experiments changed much more than the activation relative to that run:

| Setting | Earlier benchmark | Recent activation ablation |
|---|---|---|
| Backend | Torch | JAX |
| Training / validation trajectories | 224 / 32 | 16 / 8 |
| Time points per trajectory | 100 | 40 |
| Sensor | Learned 10-D linear sensor, standardized | Fixed direct 1-D sensor |
| Missing observations | None | Five-point gap |
| Trunk / drift width and depth | 64, three hidden layers | 32, two hidden layers |
| Operator training steps | 14,000, exponentially decaying LR | 6,000, constant LR |
| FP residual weight | 0.4 | 0.2 |
| M-step interval / inner steps | 2,000 / 400 | 500 / 200 |
| Reported drift metric | Affine-aligned, evaluated at inferred means | Unaligned, evaluated at true states |

The old run predates subspace initialization and bootstrap M-steps; those newer defaults must not be
silently used when reproducing it. It used a random sensor initialization, zero initial drift, and a
2,000-step warmup. Thus supervised or subspace initialization does not explain its better result.

Rescoring all nine recent checkpoints with the earlier affine-aligned metric using grid posterior means
gives mean relative drift errors **1.011 (tanh), 0.985 (tanh + Gaussian), and 1.015 (softplus)**.
The metric change therefore does not explain away the poor recovery. The rescore uses the recent
dataset and 401-point grid, not the old data/grid; it checks the metric definition, not benchmark parity.
Per-seed rescoring is saved in `dump/softplus_em_gpu/metric_audit.json`.

The activation comparisons remain matched to each other within the small experiment. They do not
establish performance relative to the earlier 0.207 result, nor that the existing M-step inherently
cannot recover double-well dynamics. Reproducing the original setup and then changing only the trunk
activation is the appropriate benchmark comparison; the effects of data size, sensor, capacity and
optimization have not been separated here.

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
