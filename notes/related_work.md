# Related work: positioning opssm against neural-Kolmogorov and PINN-filtering methods

Two 2026 papers sit closest to opssm and are the natural anchors for a related-work section:

- **NKE** — Bizzi & Fink, *Neural Kolmogorov Equations* (arXiv:2607.19173). Learns the Kolmogorov **forward**
  generator (advection + diffusion + **jumps**) from **fully-observed** trajectories, via a Gaussian-mixture
  Lagrangian (SPH-style) projection and parallel-in-time operator splitting.
- **PINNs Bayes** — Azimov & Kim, *Integrating PINNs with Bayesian inference for nonlinear filtering* (CNSNS 159,
  2026). A PINN solves the **known** forward Kolmogorov (Fokker–Planck) PDE for the one-step prior; a discrete
  Bayes update folds in observations. Dynamics are assumed known; demonstrated in 1-D.

Both learn or solve *Kolmogorov-type* PDEs for stochastic dynamics with neural networks, which is exactly our
family. What distinguishes opssm reduces to two claims.

## The two positioning claims

**1. Zakai-native, amortized filtering.** opssm learns a *single* neural operator, conditioned on the observation
stream (GRU context), that emits the unnormalized conditional density by solving the **Zakai** SPDE. This differs
from both anchors on two structural axes at once:
- *Equation.* The Zakai equation folds the observation likelihood into the **evolution operator** itself. PINNs
  Bayes instead alternates a prediction PDE (forward Kolmogorov) with a **separate discrete Bayes update** and
  renormalization at each observation time; NKE has no observation term at all (it evolves the free density).
- *Amortization.* Our operator is trained across observation realizations and filters a *new* sequence in a
  **forward pass**. PINNs Bayes *re-optimizes a network per time window per run* (warm-started) — inference is an
  optimization, not an evaluation. NKE fits its generator once per dataset and then *simulates*; it is not a filter.

**2. Learning the sensor under a gauge.** opssm recovers a *hidden* latent through an **unknown high-D linear
observation map `C, d`** learned jointly by online EM (Stiefel M-step). Both anchors observe the state (or a known
identity/`H` map) **directly** — NKE reads spatial clusters of the actual trajectories; PINNs Bayes observes the
scalar state plus noise. Learning the sensor from partial, noisy, high-D observations forces opssm to confront a
**latent scale/sign gauge freedom** that fully-observed methods never encounter (the latent SDE + linear-Gaussian
obs is identifiable only up to `z→z/α, f→f(αz)/α, g→g/α, C→αC`). We surface this honestly by reporting
**gauge-aligned (Procrustes) metrics alongside raw** — a concern that simply does not exist for either anchor.

## How the anchors relate, and the gap opssm fills

| | opssm | NKE | PINNs Bayes |
|---|---|---|---|
| Task | filtering **+** system ID | system ID / generative | filtering (known dynamics) |
| Equation | **Zakai** SPDE | forward Kolmogorov + jump | forward Kolmogorov + discrete Bayes |
| Dynamics `f,g` | learned | learned (+ Lévy jumps) | **known a priori** |
| Observation model | **learned high-D sensor `C,d`** | full state (clusters) | direct noisy scalar |
| Amortized inference | **yes** (forward pass) | fit-once, then simulate | **no** (per-window optimization) |
| Noise class | Gaussian, state-dep `g²(z)` | **general Lévy** (coupled + jumps) | Gaussian, fixed `κ` |
| Dimensionality | **high-D latent** (target) | not extreme-D (GMM limit) | 1-D shown |

- **NKE is our closest cousin on the *dynamics-learning* half** — both learn Kolmogorov generators with neural
  nets, mesh-free, without autoregressive simulation. The gap is the *inverse/filtering* problem: NKE reads the
  actual state, so it never faces a hidden latent, a learned sensor, or a gauge; in exchange it learns much richer
  (Lévy/jump) noise. Read NKE as "opssm's dynamics core with full observations and richer noise."
- **PINNs Bayes is our closest cousin on the *filtering* half** — but a weaker instance of it: known dynamics,
  known observation map, per-window (non-amortized) PINN solves, 1-D. Our Zakai formulation, amortized operator,
  and jointly learned dynamics + sensor each strictly generalize it.

## Classical lineage to cite (brief)

The Zakai equation and two-filter (Pardoux) smoother are the classical continuous-time filtering/smoothing
objects; particle filters and the EnKF (both baselines in PINNs Bayes) are the sampling counterparts that struggle
in high dimension; projection filters (Brigo–Hanzon–Le Gland) and moment-closure are the finite-dimensional
Gaussian-manifold approximations that NKE's Gaussian-mixture projection descends from. opssm replaces the
finite-dimensional projection with a **neural operator** and the per-instance solve with **amortization**.
