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

## Known limitation (open)
The self-supervised Zakai recursion is **over-dispersed**: the operator's posterior *mean* tracks the exact
filter, but its *width* is ~3× too wide (KL plateaus ~0.5 on the dense 1-D case). We traced this to the
stiffness of representing a sharp post-update's fast Fokker–Planck evolution with the time-in-branch DeepONet;
the bootstrap *target* is correct (its width matches the exact filter), so it's a representational/optimization
limit, not a wrong objective. The dynamics/sensor recovery is unaffected (it uses the mean). See
`analysis/diagnostics.py` (`recursion_width_diag`).
```bash
# legacy setup (conda)
conda env create -n opssm -f environment.yml && conda activate opssm
```
