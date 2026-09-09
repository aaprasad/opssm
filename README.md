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

The state-dependent trunk defaults to softplus in both backends. Use `model.trunk_activation=tanh`
for legacy baselines and checkpoints. See the [activation comparison](docs/em_highd_activation.md)
for JAX commands, checkpoint compatibility, and double-well results.

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
- With the fixed-node readout, **drift and sensor recover well** (`c_cos → 1`). `g` *was* over-estimated (≈ 0.79
  vs true 0.6) — but that was **downstream of the over-dispersion**: with the residual fix below the operator is
  much less over-wide and **`g ≈ 0.6`** (0.61 high-D, 0.52 1-D). The residual bias in `g² = Var[Δẑ]/dt` is real
  but is now second-order (see the `g` estimator limitation).
- Ruled out: gradient **mode**-finding (under-converges; mode ≠ mean) and **Gauss–Hermite** quadrature
  (`O(n^d)`, doesn't scale).

## Over-dispersion fix — scale-invariant FP residual (what we did)
The operator posterior was ~3–5× too wide (mean right, width wrong). We traced this to a **conditioning
pathology of the L2 Fokker–Planck residual**, not the DeepONet or the objective's minimum: in log space the FP
terms scale as `1/σ²`, so for a *sharp* density the absolute residual `(∂ℓ/∂t − rhs)²` explodes at collocation
samples far from the mode (gradient norm ~7800 vs ~200 normalized), and minimizing that magnitude drives the
operator **wide**. Diagnosed by continuing a *perfect* narrow start (supervised, std 0.089) on the Zakai loss with
*true* dynamics: it blows up to std 0.7 — the objective actively pushes away from the exact filter — and an
ablation shows the FP residual **spreads** while the likelihood update **sharpens**, with the residual winning.
**Fix:** two knobs, tuned together. (1) `res_mode="rel"` (default) measures the residual *relative to the local
FP magnitude* (`res2 / (rhs² + (∂ℓ/∂t)² + 1)`), removing the tail blow-up while keeping the broad proposal for
coverage — this moves the stable width from ~3–5× down to ~2× (KL 0.67 → 0.35). (2) `w_res` (default `0.2`)
down-weights the FP residual against the jump/ic recursion: the residual *spreads* and the likelihood update
*sharpens*, so a smaller weight slides the balance to the exact width (`rel` makes this knob **stable** — the raw
L2 residual made it a runaway). Together, full 1-D EM (8k) reaches **KL 0.67 → 0.055 (~6×, near-exact)**. The
residual also couples the operator to the learned `f, g`, so `w_res` trades posterior width against dynamics — at
`w_res=0.2` the drift still fits well **in-regime** (its error is in the off-data tail; the drift metric is now
split into on-data / off-data), and `g≈0.52` (close to the exact filter's own mean-path value ~0.58).

**`w_res` is dimension-dependent:** 1-D uses `0.2`, **high-D uses `0.4`** (`configs/experiment/em_highd.yaml`) — a
higher-D sensor sharpens the true posterior, so it needs less residual down-weighting; the optimum saturates at
0.4 (sweep 0.2→0.6). High-D EM reaches **KL ~0.7 → 0.36**, `c_cos → 1.0`, `g ≈ 0.6`.

**Remaining high-D gap (open):** `w_res=0.4` leaves high-D at KL ~0.36 (operator ~1.5× too wide). We showed this is
a **biased minimum of the self-supervised objective**, *not* a sensor/capacity/resolution limit — supervised
fitting reaches KL 3e-4, and the true filter is not a stable point of the Zakai loss even with true dynamics + a
perfect sensor. Structural fixes tried and **ruled out** (detach-rhs → collapses; variance-growth → no effect, a
differential constraint can't move the absolute width; also Fourier/`res_post`/resolution). Closing it likely
needs the **smoother** (a joint/path sampler), not more objective tuning. Full account + ruled-out alternatives in
`notes/overdispersion.md`.

## Known limitations (open)
1. **Diffusion `g` estimator** — `g² = Var[Δẑ]/dt` is a *variance* estimator over the per-time posterior marginals,
   so it absorbs mean-path roughness as if it were diffusion and reads slightly low/high depending on the filter
   width (≈0.52 at `w_res=0.2`, vs σ=0.6). The principled fix is a **square-then-average** estimator over *joint*
   posterior path samples, which needs the cross-covariance of consecutive states — a smoother or path-MCMC — not
   the per-time marginals the operator provides (independent-marginal sampling over-estimates ~100× since
   consecutive states are near-perfectly correlated).

## Next directions
- **Diffusion `g`:** build a joint / path sampler (path-MCMC, or a smoother for the cross-covariance) to enable
  the correct `square-then-average` `g` estimator. MCMC-over-the-path is the natural vehicle and is unusually
  parallel here (the operator's marginals are time-independent; DEER can parallelize the within-chain steps).
- **High-D latent** (`lorenz/`, `vanderpol/`) — where the mesh-free readout actually earns its keep (the grid
  dies at `O(N^d)`) and where the joint-sample `g` fix becomes necessary rather than optional. Slots in via the
  `data/` submodule trio.
```bash
# legacy setup (conda)
conda env create -n opssm -f environment.yml && conda activate opssm
```
