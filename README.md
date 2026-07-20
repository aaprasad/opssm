# opssm — Neural mesh-free Zakai filter

A physics-informed neural **operator** that amortizes the *exact* nonlinear filtering posterior (the
solution of the **Zakai** equation), trained **mesh-free** (no state grid), with the latent **dynamics**
(drift `f`, diffusion `g`) and the **observation map** (`C, d`) learned from data by online EM.

The testbed is a 1-D double-well latent SDE, observed either directly (`x = z + noise`) or through a
high-dimensional linear sensor (`y = C z + d + noise`, the Duncker-style setup). The exact grid Zakai
filter is the validation oracle.

> The exploratory **latent-SDE line** that preceded this (baselines, FP/GNO physics priors, the first
> grid/operator Zakai filters) is archived, in chronological order, on the **`archive/latent-sde`** branch.

## Install
```bash
uv pip install -e .          # torch, torchsde, lightning, hydra-core, matplotlib
```

## Run (Hydra + Lightning)
```bash
python scripts/train.py experiment=supervised   # Stage 1: operator fits the exact filter (KL ~ 2e-4)
python scripts/train.py experiment=em_1d        # Stage 3: learn f, g from 1-D obs (drift L2 ~ 0.4, g ~ 0.6)
python scripts/train.py experiment=em_highd     # Stage 4: learn f, g, and the Stiefel sensor C,d from 10-D obs
```
Override anything from the CLI, e.g. `... experiment=em_highd model.pca_init=false trainer.max_steps=8000`.
Figures + metrics are written under `train_dir` (default `./dump/nzf`).

## Package layout
```
opssm/
  data/doublewell/   sde.py (generative model) · oracle.py (exact grid filter) · datamodule.py (LightningDataModule)
  models/            operator.py (DeepONet log-density) · dynamics.py (DriftNet, DiffusionNet) ·
                     losses.py (mesh-free Zakai PINN + SNIS) · mstep.py (EM M-step: f, g, Stiefel C,d) ·
                     obs.py (high-D sensor) · filter_module.py (ZakaiFilterModule, LightningModule)
  analysis/          viz.py (figures) · diagnostics.py (off-path research probes)
scripts/train.py     Hydra entry point
configs/             Hydra config groups: data / model / experiment
```
The EM is one configurable `LightningModule` with manual optimization: `training_step` is the operator
**E-step** (mesh-free Zakai PINN), `on_train_batch_end` is the **M-step** (closed-form / regression of the
dynamics and sensor from the operator's inferred latent path). The 1-D and high-D M-steps share one code
path — 1-D is the degenerate `cstab≡0` case of the high-D *sensor-before-dynamics* curriculum.

## Adding a system
Each dynamical system is a submodule under `data/` providing the trio `sde.py` (generative model),
`oracle.py` (ground-truth filter), `datamodule.py`. `lorenz/` and `vanderpol/` slot in next.

## Mesh-free M-step readout (what we did)
The M-step needs `E[z_t | y]` to fit `f, g, C, d`. The **default is a deterministic mesh-free readout**
(`meshfree_mean=true`, `posterior_mean_fixed` in `models/mstep.py`): fixed-node importance sampling — the
proposal nodes are drawn *once* and reused every M-step — so `ẑ` is a smooth *deterministic* function of the
operator, at `O(K)` cost independent of the latent dimension. The `O(N^d)` grid quadrature (`meshfree_mean=false`)
stays available and is exact for a low-D latent. Findings (full account in the design notes):
- A **stochastic** readout (fresh importance samples each M-step) **diverges**: `g² = Var[Δẑ]/dt` is a
  *variance* amplified by `1/dt`, so readout noise feeds a runaway feedback with the over-dispersion. The grid
  (deterministic) is stable → **determinism is necessary**, and fixing the nodes supplies it while keeping the
  dimension-agnostic `O(K)` cost.
- With the fixed-node readout, **drift and sensor recover well** (data-regime drift L2 ≈ 0.2, `c_cos → 1`);
  **`g` is over-estimated** (≈ 0.79 vs true 0.6 in high-D — *the grid has this too*, so it's the estimator, not
  the readout).
- Ruled out: gradient **mode**-finding (under-converges; mode ≠ mean) and **Gauss–Hermite** quadrature
  (`O(n^d)`, doesn't scale).

## Known limitations (open)
1. **Posterior over-dispersion** — the operator posterior is ~3× too wide (mean right, width wrong; KL plateaus
   ~0.5 on the dense 1-D case). Traced to the stiffness of representing a sharp post-update's fast Fokker–Planck
   evolution with the time-in-branch DeepONet; the bootstrap *target*'s width matches the exact filter, so it's
   a representational/optimization limit, not a wrong objective. Dynamics/sensor recovery is unaffected (it uses
   the mean). See `analysis/diagnostics.py` (`recursion_width_diag`).
2. **Diffusion over-estimation** — `g ≈ 0.79` vs `σ = 0.6` in high-D. `g² = Var[Δẑ]/dt` is a *variance*
   estimator, so it absorbs the over-dispersion / mean-path roughness as if it were diffusion. The principled
   fix is a **square-then-average** estimator over *joint* posterior path samples, which needs the
   cross-covariance of consecutive states — a smoother or path-MCMC — not the per-time marginals the operator
   provides (independent-marginal sampling over-estimates ~100× since consecutive states are near-perfectly
   correlated). Tied to (1).

## Next directions
- **Diffusion `g`:** build a joint / path sampler (path-MCMC, or a smoother for the cross-covariance) to enable
  the correct `square-then-average` `g` estimator. MCMC-over-the-path is the natural vehicle and is unusually
  parallel here (the operator's marginals are time-independent; DEER can parallelize the within-chain steps).
- **Over-dispersion** — the shared root of both open issues: richer within-interval time features (`√s` /
  Fourier), or decoupling the filtering head from the FP initial condition.
- **High-D latent** (`lorenz/`, `vanderpol/`) — where the mesh-free readout actually earns its keep (the grid
  dies at `O(N^d)`) and where the joint-sample `g` fix becomes necessary rather than optional. Slots in via the
  `data/` submodule trio.
```bash
# legacy setup (conda)
conda env create -n opssm -f environment.yml && conda activate opssm
```
