# Latent-dynamics comparisons: synthetic systems and Kato 2015

The suite is implemented afresh: **Dynamax** for KF/EKF/UKF and the Gaussian
components of a switching LDS; **Diffrax** for latent SDE simulation; **JAX**
for simulation-free SDE Matching. It does not import the `baselines-harness`
branch. The existing OPSSM is accessed through a new adapter.

Install into a JAX environment:

```bash
pip install -e '.[benchmarks]'
pip install pytest  # to run the mathematical checks
JAX_PLATFORMS=cpu python -m pytest tests/test_benchmarks.py tests/test_kato_benchmarks.py -q
JAX_PLATFORMS=cpu python -m opssm.benchmarks.runner --smoke --seeds 0 --out dump/benchmark_smoke
```

Omit `JAX_PLATFORMS=cpu` to use an installed accelerator backend. The dependency
extra does not install CUDA libraries or Torch. OPSSM normally runs in a fresh
subprocess using the same Python; `--opssm-python /path/to/jax/bin/python` selects
a separate environment if needed. All requested models are included by default.

## Dataset settings and provenance

There is **no single standard dataset protocol** covering these methods. The
defaults below keep recognizable published dynamics and noise scales while
providing a common regular observation grid and independent test trajectories.
These are **adapted comparisons**, not reproductions of published scores.

| Preset | Physical drift | Diffusion factor | Simulation step, at most | Observations | Observation dimensions / noise SD | Independent train / val / test trajectories |
|---|---|---|---|---|---|---|
| `doublewell` | `4(x − x³)` | `G=1` | 0.001 | 301 over [0,3], Δt=0.01 | 15 / 0.5 before preprocessing | 20 / 5 / 10 |
| `vanderpol` | `f₁=τρ(x₁−x₁³/3−x₂)`, `f₂=τx₁/ρ`; ρ=2, τ=15 | `G=I₂` | 0.001 | 301 over [0,3], Δt=0.01 | 20 / 1.5 before preprocessing | 20 / 5 / 10 |
| `lorenz` | Lorenz-63, (10,28,8/3) | `G=0.3 I₃` | 0.001 | 100 over [0,2], Δt=2/99 | 3 / 0.01 after signal normalization | 1024 / 128 / 128 |
| `lorenz_li` | Same Lorenz drift | `diag(0.1x,0.28y,0.3z)` | 0.001 | Same as `lorenz` | Same as `lorenz` | 1024 / 128 / 128 |

`G` is the factor multiplying `dW`, **not** a variance: Euler-Maruyama uses
`G sqrt(h) ε`, with incremental covariance `h GGᵀ`. Each observation interval
is subdivided to land exactly at its endpoint. In particular `2/99` is not
rounded to 0.02. Metadata records the actual simulation step.

The [Duncker et al. paper, sections 5.1–5.2](https://proceedings.mlr.press/v97/duncker19a/duncker19a.pdf)
specifies double-well strength 4, 15 outputs, noise variance 0.25, and 20 trials;
for Van der Pol it gives ρ=2, **τ=15**, 20 outputs, variance 2.25, and 20 trials.
It reports randomly sampled observations (20 per double-well trial), initializes
the sensor at its generating values, and uses a 1 ms inference grid. Our regular
dense grid, extra held-out trajectories, observation-only initialization,
3-second windows, and simulation step are explicit comparison choices. Unit
diffusion and initial distributions are specified assumptions here, rather than
claims of an exact reproduction of that paper's data generator.

For double well, initial states are `N(0,1)`. For Van der Pol they are
`Uniform([-3,3]²)`. Both have a shared random Gaussian sensor `C` and offset `d`,
held fixed across splits. There is no burn-in.

The [official torchsde Lorenz example](https://github.com/google-research/torchsde/blob/master/examples/latent_sde_lorenz.py)
uses the Lorenz parameters above, multiplicative coefficients (0.1,0.28,0.3),
normal initial states, 100 observations over two seconds, 1024 trajectories,
and noise SD 0.01 after normalization. `lorenz_li` preserves those main choices;
we use Euler-Maruyama on an explicit grid, train-only normalization constants,
and add separate validation/test draws. The primary `lorenz` preset replaces
the multiplicative diffusion with additive SD 0.3 so it lies in OPSSM's current
constant-diffusion model class. **That diffusion change is an adaptation.**
OPSSM rejects `lorenz_li` rather than silently treating its diffusion as constant.
EKF/UKF also retain their additive-noise model on that optional stress test;
the two latent SDEs enable coordinate-wise state-dependent diffusion there.

The [SDE Matching paper](https://proceedings.mlr.press/v267/bartosh25a.html)
and [authors' implementation](https://github.com/GrigoryBartosh/sde_matching)
motivate the simulation-free objective. This suite implements the diagonal
Gaussian marginal specialization, not their full experimental architecture.
In particular, all primary baselines use the known *dimension* 1/2/3; the
torchsde example's default latent dimension is 4. No baseline receives the
true latent paths or true sensor coefficients during training.

## What each baseline computes

| Model | Library / learning | Native inference |
|---|---|---|
| `kf` | Dynamax linear Gaussian LDS; learn affine transition, diagonal process covariance, sensor and initial distribution by likelihood gradients | Exact Kalman filter |
| `ekf` | Dynamax EKF; learn neural drift and constant diagonal diffusion by approximate likelihood gradients | First-order Gaussian filter |
| `ukf` | Identical generative model to EKF; Dynamax UKF, α=1, β=2, κ=0 | Sigma-point Gaussian filter |
| `slds` | Dynamax SLDS parameter types and `lgssm_filter` updates; learn Markov transitions and regime-specific affine dynamics using an IMM likelihood | Approximate switching filter |
| `latent_sde` | Diffrax Ito Euler solver, reverse-mode differentiation with recursive checkpointing, Girsanov path KL | Simulated variational smoother |
| `sde_matching` | JAX Gaussian posterior marginals and analytical posterior drift; no solver during training | Gaussian variational smoother |
| `opssm` | Existing JAX operator/EM training, with the same saved data | Operator filter |

The [Dynamax API](https://probml.github.io/dynamax/api.html) supplies the Gaussian
filters. The switching implementation is a **Markov SLDS with IMM approximation**,
not rSLDS, not a deterministic mixture of drifts, and not an exact SLDS likelihood.
It retains a Gaussian per regime and includes between-regime covariance during
mixing. There is no built-in Dynamax IMM fitting API; this small orchestration
layer is implemented here and tested to reduce to Dynamax KF at one regime.
Default regime count is 3; tune `--modes` on validation data.

OPSSM resolves the repository's Hydra model configuration, including inherited
`configs/model/operator.yaml` defaults, using this mapping:

| Dataset | Experiment | `n_colloc` | `n_mean` |
|---|---|---:|---:|
| Double well | `duncker_dw` | 128 | 256 |
| Van der Pol | `duncker_vdp` | 256 | 384 |
| Lorenz | `duncker_lorenz` | 384 | 512 |
| Kato | `kato` | 128 | 256 |

These configurations use scalar diffusion (`diffusion_cov=false`), subspace
initialization, drift warm-starting, and the bootstrap M-step. `n_mean` is retained
from the configuration; with `mean_method=mala`, the actual sampler budget is
`mala_chains=64` and `mala_steps=30`. The configured `chunk_size=16` is implemented
in the JAX E-step as size-weighted full-batch gradient accumulation with one
optimizer update per step. MALA readout also processes trajectories in chunks.
This caps derivative-buffer memory without dropping training trajectories or
reducing collocation points. E-step chunking preserves the full-batch gradient
for identical samples; MALA chunking uses independent chunk RNG streams.

Only model settings are inherited: all methods still use the same benchmark
dataset, preprocessing, observation intervals, dimensions, training-step budget,
and validation cadence. The experiments' legacy dataset settings are not applied
to OPSSM alone. Use `--opssm-experiment NAME` to select another compatible
experiment, `--opssm-config-dir PATH` for a frozen config tree, and
`--opssm-config overrides.json` for explicit model overrides. Each OPSSM run saves
`hparams.json`, `config_provenance.json`, and its experiment/config hash in results.
Unsupported JAX model paths fail explicitly rather than silently ignoring them.

The original September 10 sweep used trainer defaults, PCA initialization, and
full covariance. Those OPSSM results are a separate configuration and must not be
pooled with the corrected experiment-config runs.

EKF/UKF use the conventional observation-step approximation
`z_next = z + Δt f(z) + Normal(0, Δt diag(g²))`. This approximation is distinct
from the finer data simulation. KF/SLDS learn a discrete transition directly.
The [Diffrax solver](https://docs.kidger.site/diffrax/usage/getting-started/)
uses steps at most `--solver-dt` (default 0.01), with every observation time
included in the step grid. Check discretization sensitivity using smaller
solver steps; this flag affects the latent SDEs and OPSSM forecasts, not the
Dynamax discrete observation-step models.

The neural SDEs share the same prior drift, diffusion family, affine emission,
learned Gaussian initial prior, and GRU hidden size. Simulation training uses
`f_q=f_p+g u`, giving path KL `½ ∫ ||u||² dt`. Matching uses
`z=m(t)+s(t)ε`, score `−ε/s`, and posterior drift
`dm/dt + ds/dt ε + ½ div(GGᵀ) + ½ GGᵀ score`. Its uniform-time estimator is
multiplied by the physical duration, and its sampled observation likelihood by
the number of observations. Initial KL is counted once. No time normalization
or KL annealing is applied. Changing the trajectory duration therefore does
not silently change the physical units of the drift/diffusion.

Additional cross-check: [Kiyohara's JAX MoCap implementation](https://github.com/nkiyohara/sde-matching-mocap/blob/main/train_mocap.py)
uses the same Gaussian posterior-drift identity, JVP time derivatives, and
coordinate-wise diffusion correction. Its forward objective combines sampled
terms without our duration/observation-count factors, so it is not a drop-in
equivalent of our full-sequence ELBO. Its nonlinear emission and configurable
posterior features also differ. We retain the shared synthetic sensor model
and record this implementation as a reference, without copying its code.

## Fairness and evaluation

* Data are generated **once per dataset/seed**, saved as a framework-neutral NPZ,
  and identified by a SHA-256 content hash in every result. Splits have independent
  RNG streams and share the sensor. Increasing test size cannot alter training.
* After constructing the sensor, preprocessing uses only noisy training
  observations: per-channel centering and one global PCA signal scale. In the
  Lorenz presets, train-only clean-signal normalization is part of the synthetic
  observation map, following the reference's signal-normalized noise convention.
* The configured observation-noise SD is supplied as a common initial value.
  It is learned by default; `--fixed-obs-noise` holds it fixed for every method.
  No test data or test truth are used for fitting, initialization, or selection.
* Native **filtering and smoothing metrics are labelled separately**. Never rank
  them together as causal inference results. Smoothing encoders see future
  observations within the supplied sequence; no masked-encoder variant is
  incorrectly labelled an exact Bayesian filter.
* The common forecast task hides the **second half** of every test sequence.
  A smoother sees only the first-half prefix, and all future paths come from
  the learned prior. Report clean-signal forecast RMSE and Monte Carlo marginal
  observation NLL. The latter is a finite-sample mixture estimate, **not** a
  joint trajectory likelihood or an ELBO. It can be noisy at very small sensor
  noise; increase `--samples` and check convergence.
* Latent RMSE/normalized RMSE use an affine alignment fitted to **validation**
  latent states after model selection, frozen before test scoring. This resolves
  coordinate ambiguity without fitting away test errors. Calibration truth is
  used only for this evaluation coordinate map. Normalization uses training
  latent standard deviations. No latent alignment affects observation scores.
* This map is **unrestricted affine least squares**, allowing translation,
  rotation/reflection, scale and shear, rather than orthogonal Procrustes.
  The exact map is stored in `predictions.npz` as `alignment`.
* `dynamics_rmse` measures learned **prior drift** in physical state units per
  unit time. For the row-vector map `x = z A + b`, query the learned field at
  `z = (x_true − b) A⁻¹`, then compare `f_learned(z) A` with `f_true(x_true)`.
  Both fields are evaluated at the same ground-truth test states, so latent
  trajectory errors do not directly contaminate this field-recovery metric.
  The translation is not added to velocities. RMSE averages squared errors
  over all times, test trajectories and coordinates before taking the root.
  `dynamics_nrmse` divides this by the RMS true drift on the training states.
  This evaluates the analytic true drift, not finite differences of noisy SDE
  paths. The query states and both drift arrays are saved with predictions.
* `dynamics_kind` distinguishes continuous drift (OPSSM, EKF/UKF, both latent
  SDEs) from the **observation-interval effective drift** of KF/SLDS. For KF this
  is `(A_discrete z + b_discrete − z)/Δt`. For SLDS, the next-regime probabilities
  are the test filter's regime probabilities multiplied by its transition
  matrix; these weight the regime-specific affine increments at the query state.
  This SLDS diagnostic is conditioned on observation history and is not an
  autonomous field. Finite-interval approximation error contributes to the
  discrete models' comparison with the instantaneous true drift. A singular
  alignment or condition number above 1e10 yields an explicit dynamics status
  and null RMSE rather than an arbitrary pseudoinverse comparison.
* Clean reconstruction R²/RMSE compare against the noiseless sensor signal.
  Marginal 95% coverage uses Gaussian moment intervals (including mixture moments
  for SLDS), not exact mixture quantiles. OPSSM uses grid moments in 1D and MALA
  moments in higher dimensions. Its forecast starts from Gaussian moment fits.
* Each baseline selects its best checkpoint using its own **validation observation
  objective** (likelihood or ELBO); OPSSM uses validation observation reconstruction
  R². These training objectives are not scores to rank across methods. Test
  states are evaluated only after checkpoint selection.
* Defaults are 2,000 updates, batch size 16, hidden size 64, learning rate 0.001,
  gradient norm clipping at 10, validation every 100 updates, and 128 posterior/
  forecast samples. OPSSM uses its experiment-config EM schedule and learning
  rates, with full-batch gradients accumulated in chunks; `--batch-size` and
  `--lr` apply to the other baselines. Updates have different costs across methods.
* Timing includes compilation and validation; the first baseline training step
  is also reported separately. Baselines use float64 for covariance stability;
  OPSSM keeps its float32 implementation. Record hardware, libraries, update
  counts and parameter counts alongside runtime. A raw timing difference is
  not a controlled solver-only speed comparison.

The existing `configs/data/duncker_*.yaml` remain legacy experiments and are
not used by this suite. In particular the legacy DW strength 1 and VdP τ=10
are different from this suite's paper-aligned drift coefficients.

## Running experiments

### Kato 2015 real neural recordings

The runner also supports the [Kato 2015 whole-brain calcium dataset](https://osf.io/2395t/)
([original paper](https://doi.org/10.1016/j.cell.2015.09.034)). Download the MATLAB
file from the authors' repository and supply `--kato-mat`. The recommended main
comparison uses **WT_NoStim**, the five spontaneous-activity recordings, fitting
each worm separately with its own neuron count. WT_Stim is also readable, but
stimulus inputs are not modeled; it is a separate, unconditional benchmark.
This per-worm protocol is not a reproduction of hierarchical rSLDS results that
share identified neurons across animals.

Defaults are 10 latent dimensions, the corrected fluorescence traces at native
`fps` (WT_NoStim approximately 2.8–3.1 Hz), and no added noise, clipping, derivative
preprocessing, or temporal smoothing. Raw frames are divided chronologically into
60% training, 20% validation, and 20% test blocks; 30 frames are omitted from the
start of each held-out block as a temporal gap. Only then are windows formed:
200 frames, training stride 100, and nonoverlapping validation/test windows.
Incomplete tails are dropped and their lengths recorded. These split/window/gap
choices are this benchmark's protocol, not settings claimed from the original
paper. Per-neuron mean and standard deviation come from the raw training block
only. The observation-noise initialization is 0.1 in standardized units and is
learned by default; there is no known biological noise or diffusion coefficient.

Real-data scores are `observation_recon_rmse`, `observation_recon_r2`,
`forecast_observation_rmse`, `forecast_marginal_nll`, and a last-observation
`forecast_persistence_rmse` reference. RMSE is also reported in original fluorescence
units with a `_physical` suffix. Reconstruction uses the observations supplied to
inference, so it is descriptive; forecasting hides the second half of each test
window and measures prediction. Kato has no ground-truth latent trajectory,
noise-free signal, or drift field. Latent/dynamics RMSE are null with an explicit
unavailable status; no alignment or biological ground truth is fabricated.

**Linear state decodability** uses the annotated behavior states as a separate
supervised evaluation after the dynamical checkpoint is frozen. It fits a
class-balanced logistic regression on training posterior means, standardizing
features using training statistics only. The L2 inverse-regularization grid is
`C = [0.001, 0.01, 0.1, 1, 10, 100]`, selected by validation balanced accuracy,
with ties favoring stronger regularization. The selected decoder remains fitted
on training data; test labels never enter fitting or selection. Duplicate frame
latents from overlapping windows are averaged before probing. Negative/unknown
labels and the ambiguous `NOSTATE` class are excluded. Reported metrics:

* `linear_decode_accuracy`, `linear_decode_balanced_accuracy` (mean recall over
  test-present classes), and `linear_decode_macro_f1` (true/predicted class union).
* Validation balanced accuracy, chosen C, labeled frame counts, training classes,
  unseen test-class fraction, and a training-majority-class test accuracy reference.

Test classes absent from training count as errors. Empty labeled splits or fewer
than two training classes give an explicit unavailable decoder status, not an
invented score (this can happen in very short smoke windows). Decoder weights,
feature normalization, test predictions/labels, and frame indices are saved in
`predictions.npz`. Labels never supervise or select the dynamical model. These
decoding scores inherit the model's `posterior_type`: smoothing methods see the
full window, while filters see its prefix. Compare within the same information
set or use the common prefix-only forecast metric across all methods.

```bash
# All five WT_NoStim worms, all seven methods, five optimization seeds.
python -m opssm.benchmarks.runner --datasets kato \
  --kato-mat /home/aaprasad/data/kato/WT_NoStim.mat \
  --seeds 0 1 2 3 4 --out dump/kato_comparison

# Installation check on one worm. All models retain 10 latent dimensions.
python -m opssm.benchmarks.runner --datasets kato \
  --kato-mat /home/aaprasad/data/kato/WT_NoStim.mat --kato-worms 0 \
  --smoke --seeds 0 --out dump/kato_smoke
```

Override `--kato-latent-dim`, `--kato-window`, `--kato-stride`, and `--kato-gap`
for validation-driven sensitivity studies. Every method and optimization seed
uses the same recording splits; metadata includes the source SHA-256, actual
sampling rate, split boundaries, neuron identifiers, and preprocessing settings.
Results are grouped by worm (e.g. `kato_WT_NoStim_worm0`), model, and inference type.
Across-seed variation is optimization variation, not independent animal replication;
report per-worm results and treat worms as the biological replicates.
Kato smoke runs retain the native sampling rate and use up to four training and
two validation/test windows of 12 frames, spread across each temporal block.

### Synthetic systems

```bash
# Save all full-size datasets without fitting anything.
python -m opssm.benchmarks.runner --generate-only --seeds 0 1 2 3 4 --out dump/comparison

# Full comparison, including OPSSM, on the exact artifacts above.
python -m opssm.benchmarks.runner --seeds 0 1 2 3 4 --out dump/comparison

# A focused run; use a new directory for a different model/hyperparameter run.
python -m opssm.benchmarks.runner --datasets vanderpol --models kf ekf ukf slds \
  --seeds 0 1 2 3 4 --steps 2000 --modes 4 --out dump/vdp_four_modes

# Optional multiplicative Lorenz reference, outside current OPSSM's model class.
python -m opssm.benchmarks.runner --datasets lorenz_li --models latent_sde sde_matching \
  --seeds 0 1 2 3 4 --out dump/lorenz_multiplicative
```

Use `--smoke` only for installation/integration checks: it shortens trajectories,
reduces split sizes and networks, and runs at most three updates. Its metrics are
**not evidence of model quality**. Full converged comparisons require tuning
on validation data with comparable search budgets, multiple seeds, simulation/
solver convergence checks, and both accuracy-versus-updates and accuracy-versus-
time comparisons. The defaults are a starting protocol, not tuned winning settings.

Dataset sensitivity experiments can use `--dataset-config settings.json`, for example
`{"doublewell": {"simulation_dt": 0.0005, "num_obs": 601, "n_train": 40}}`.
All resolved settings are saved in dataset metadata. Use a different output
directory when changing them; the fingerprint check protects existing datasets.

Add dynamics scores to existing results without retraining:

```bash
python -m opssm.benchmarks.rescore --out dump/benchmark_smoke_verified
```

This restores each selected checkpoint and the saved validation alignment,
checks the dataset fingerprint, and updates predictions, results, and root
CSV/summary files. It does not refit the alignment on test data.

Each run writes `data.npz`, model checkpoints, `history.json`, `predictions.npz`
and `result.json`. The root contains `metrics.csv` and `summary.json` (across-seed
means and sample standard deviations, grouped by model and posterior type).
Failures have explicit status/error files and cause a nonzero exit; they are
excluded from successful-run aggregates. Existing results and mismatched datasets
are not silently overwritten.

## Verification of this implementation

On CPU with JAX 0.11.1, Dynamax 1.0.2, Diffrax 0.7.2, Equinox 0.13.8 and
Optax 0.2.8:

* All 12 new mathematical/leakage tests and nine existing JAX model tests passed.
* The subsequent MoCap-reference audit added a nonstationary Fokker–Planck
  identity test; it and two related matching-objective checks passed.
* All 21 primary smoke runs passed: seven methods, including OPSSM, on three systems.
* Both latent SDE methods completed three training updates and evaluation on all
  three full-sized datasets (six runs), with finite losses and predictions.
* With aligned dynamics scoring and Kato/linear decoding added, all 20 benchmark
  tests and nine existing JAX model tests passed (29 total).
* The experiment-config correction adds seven tests for Hydra inheritance,
  chunked/full-batch gradient equality, and complete MALA readout (36 tests total).
  The revised OPSSM adapter passed smoke runs on double well, Van der Pol,
  Lorenz, and Kato. A GPU compile-only check of the full Lorenz E-step
  (100 times, 1,024 trajectories, 384 collocation points, chunk size 16) reported
  2.65 GiB of temporary buffers; this is a compiler estimate, not a measured
  full-training peak.
* All seven Kato smoke runs completed training, forecasting, and valid linear
  behavior decoding on WT_NoStim worm 0. The KF also completed a one-update
  full-size check with valid decoding on each of the five WT_NoStim worms.
  Both MATLAB schemas were loaded and split successfully (five WT_NoStim and
  seven WT_Stim recordings).

Artifacts from these checks are under `dump/benchmark_smoke_verified/` and
`dump/benchmark_full_shape_check/`; Kato artifacts are under
`dump/kato_benchmark_verified/` and `dump/kato_full_shape_check/`.
These checks establish runnable integrations;
they do not establish convergence or an accuracy ranking.

## Cluster sweeps over data configs

Two deployments, both one GPU per (sweep point, dataset, seed, model) cell -- concurrent
training on a shared GPU is not supported, so the cell is the unit of parallelism.

### Hydra multirun + submitit (preferred)

`scripts/benchmark_sweep.py` reads `configs/benchmark.yaml` and fits one cell per job, so a
multirun is a job array over the product of the comma-separated axes. It reuses the repo's
existing `configs/hydra/launcher/submitit_slurm.yaml`.

```bash
# One cell, locally.
python scripts/benchmark_sweep.py dataset=vanderpol method=opssm seed=0

# Sweep the data configs on SLURM; fill in the launcher's mandatory partition.
python scripts/benchmark_sweep.py -m hydra/launcher=submitit_slurm \
    hydra.launcher.partition=<gpu-partition> hydra.launcher.mem_gb=32 \
    dataset=vanderpol seed=0,1,2,3,4 \
    method=kf,ekf,ukf,slds,latent_sde,sde_matching,opssm \
    dataset_config.noise_std=0.25,0.5,1.0,1.5,3.0 \
    results_root=dump/noise_sweep

python scripts/cluster/collect_sweep.py --results-root dump/noise_sweep
```

`dataset_config.*` overrides the dataset's `DatasetConfig`; every sweepable field is listed in
`configs/benchmark.yaml` as `null` ("keep the preset's value") so it can be swept directly.
**`method` and `dataset_config` are named to avoid the repo's existing `model` and `data` Hydra
config groups**, which would otherwise capture those overrides. The resolved non-null fields name
the results subdirectory, so distinct data configs never collide, and `steps_per_dataset` carries
a per-dataset budget (lorenz defaults to 3000, since it converges by ~2000).

Each job validates its resolved `DatasetConfig` before doing any work, and returns immediately if
its cell already holds a `result.json`, so SLURM `--requeue` after preemption resumes the sweep.
`save_dataset` is atomic (unique temp name plus `os.replace`), so the jobs of a cell may race to
write their shared `data.npz` without producing a torn file; `make_dataset` is deterministic in
(config, seed), so the bytes are identical either way.

### Plain sbatch array

`scripts/cluster/make_sweep.py` builds a self-contained SLURM array instead, for sites where the
submitit plugin is unavailable or a pre-generate stage is preferred.

```bash
# 1. Build the sweep. Each --axis is a DatasetConfig field; multiple axes take the product.
python scripts/cluster/make_sweep.py --out sweeps/noise \
    --axis vanderpol.noise_std=0.25,0.5,1.0,1.5,3.0 \
    --datasets vanderpol --seeds 0 1 2 3 4 \
    --partition <your-gpu-partition> --account <acct> --time 12:00:00

# 2. Generate data once (CPU) and submit the array.
bash sweeps/noise/submit.sh

# 3. After the array drains.
python scripts/cluster/collect_sweep.py --sweep sweeps/noise
```

`make_sweep.py` validates every generated `DatasetConfig` on the login node, so an
invalid combination fails before any GPU time is spent. It writes `points/<name>.json`
(runner `--dataset-config` files), `manifest.tsv` (one row per array task), `sweep.json`
(provenance) and a `submit.sh` with the array size filled in.

Data is generated **once per point** by `prepare_data.py` before the array starts, so every
model in a cell reads the same `data.npz`; `save_dataset` fingerprint-checks rather than
rewriting, so there is no write race. `collect_sweep.py` re-verifies this by comparing
`dataset_hash` across models within each cell and refuses to rank a cell whose models
disagree. It also reports failed and missing cells, and exits nonzero if any are present.

Array tasks are idempotent: a cell with an existing `result.json` exits 0 immediately, so
`--requeue` after preemption resumes the sweep instead of aborting on partial output. This
relies on the runner's `--skip-existing`; without it an existing result is a hard error.

Site settings (`--partition`, `--account`, `--qos`, `--constraint`, `--time`, `--cpus`,
`--mem-gb`, `--array-parallelism`) are flags on `make_sweep.py` and land as `sbatch`
directives in `submit.sh`; `--setup` injects a shell line (module loads) before each task,
and `--python` selects the cluster's JAX environment. `SWEEP_PLATFORM=cpu` runs a task body
locally for a dry run. Per-dataset step budgets come from `--steps-override lorenz=3000`.

The array script deliberately leaves `XLA_PYTHON_CLIENT_PREALLOCATE` unset: disabling
preallocation is what caused the earlier `em_lorenz` OOM, not missing chunking.
