# M-step latent readout: sampling the operator posterior

Design notes on how the EM **M-step** reads the latent path out of the operator, why it uses
self-normalized importance sampling (SNIS) today, and where MCMC / SMC do and don't fit as we scale to
high-dimensional latents. The **north star is scalability in the latent dimension `d`** — 1-D is only the
development testbed.

## The problem

The M-step fits the SDE coefficients (`f`, `g`) and the observation map (`C, d`) from the operator's
*inferred latent path*
```
z_hat_t = E[z_t | y_{0:t}]     (and Var[z_t] for g)
```
taken under the operator's own per-time posterior `π_t(z) ∝ exp(ℓ_t(z))`, where `ℓ_t(z) = b(c_t, s=0)·trunk(z) + bias`
is the operator's post-update log-density (see `models/mstep.py`, `models/operator.py`).

The **original** readout was a grid quadrature (`posterior_mean` in `mstep.py`):
```
z_hat_t = Σ_i softmax_i(ℓ_t(z_i)) · z_i ,   z_i ∈ z_grid
```
This is `O(N^d)` in the latent dimension — it dies in high `d` and directly contradicts the whole point of a
**mesh-free** operator (which exists to escape the `O(N^d)` grid filter). So the M-step readout must be
mesh-free *and* scale in `d` too. The grid is now retained **only** for the validation oracle
(`data/doublewell/oracle.py`), where an exact reference is worth its cost.

## Empirical findings — the readout in practice

Running the candidates end-to-end (high-D `em_highd`, to 12k steps) settled which mesh-free readout to use — and
it is **not** the plain SNIS below, which was found to **diverge**. The arc:

- **SNIS (stochastic) diverges.** Once the drift gate opens (the operator posterior widens — the over-dispersion
  below), the readout's estimation noise `η` couples into the diffusion estimate `g² = Var[Δẑ]/dt`. Because `g²`
  is a **variance** (noise *adds*, it cannot average out — unlike the drift/sensor *means*, which are robust)
  and the increment is amplified by `1/dt`, `g` inflates → the E-step over-diffuses → the posterior widens →
  more `η` → **runaway** (`g`: 0.5 → 0.74 → 1.6 → 2.0). It is *not* ESS/proposal mismatch — an ESS-adaptive
  proposal kept ESS ≈ 0.5 and `g` still blew up; it is the *stochasticity of the readout itself*.
- **The grid (deterministic) is stable** at the identical config (`g ≈ 0.7`, flat) — the controlled comparison.
  So **determinism is necessary**, and it is the deterministic *mean* that matters.
- **Default: fixed-node importance sampling** (`posterior_mean_fixed`) — SNIS with the proposal nodes drawn
  *once* and reused every M-step (shared across `t`), so `ẑ` is a smooth deterministic function of the operator:
  the grid's mean, but with data-following nodes instead of a uniform grid → `O(K)`, dimension-agnostic. Result:
  **drift and sensor recover** (data-regime drift L2 ≈ 0.2; `c_cos → 1`), `g` **over-estimated** (≈ 0.79 vs 0.6).
- **Ruled out.** *Mode / Laplace* (gradient ascent on `ℓ`): the finder under-converges and `mode ≠ mean`
  (rougher path → `g` creep). *Gauss–Hermite* quadrature: deterministic and low-variance, but `O(n^d)` — dies in
  high `d` exactly like the grid.

Two caveats survive into the default:
- **The `g` over-estimation is the *estimator*, not the readout** — the *grid* over-estimates `g` in high-D too
  (≈ 0.74). `Var[Δẑ]/dt` absorbs the over-dispersion / mean-path roughness as if it were diffusion. The fix is
  the `mean(square)`-over-joint-samples estimator (see below), which is **blocked** on our marginal operator: it
  needs the cross-covariance of consecutive states, and sampling the marginals independently over-estimates
  ~100× (consecutive states over `dt=0.1` are near-perfectly correlated, so the true increment variance is
  tiny). So it is genuinely the joint-sampler / path-MCMC project, not an estimator swap.
- Drift/`g` are only meaningful over the **data regime** the latent visits (`|z| ≲ 1.5`); outside it the
  regression extrapolates — so both the `drift_l2` metric and the figures are scoped to it.

The rest of this note is the design reasoning behind those choices (why importance sampling, why *not* a Gaussian
proposal, where MCMC/SMC fit as `d` grows) — still the roadmap for the high-D-latent push.

## SNIS — self-normalized importance sampling (the estimator; its stochastic form diverges — see above)

**What it is.** To estimate `E_π[z]` for a density we can only evaluate up to a constant, draw samples from a
proposal `q`, weight them by the (unnormalized) density-over-proposal ratio, and self-normalize:
```
z_k ~ q(·)                         (k = 1..K)
w_k = softmax_k( ℓ(z_k) − log q(z_k) )       # unnormalized weights e^ℓ/q, then normalized so Σ w = 1
E[z] ≈ Σ_k w_k z_k
```
Self-normalization is **forced, not chosen**: the operator density is unnormalized (we only have `ℓ` up to a
constant), so the plain unbiased IS estimator — which needs the normalizing constant — isn't available.

**How it's used here** (`posterior_mean_snis` in `models/mstep.py`, toggled by `meshfree_mean`):
- The proposal `q` is the *same data-following proposal the loss already uses* (`sample_collocation`): half
  the samples "near" the observation, centered on the pseudo-inverse latent estimate
  `C⁺(y − d)` (`obs.zhat_from_obs`), half "broad" to cover the wells / a gap's bimodality.
- Evaluate `ℓ(z_k) = coeffs(ctx, s=0)·trunk(z_k) + bias` at the samples (one batched forward), SNIS-weight,
  return the mean. Reusing the loss's proposal keeps the **E-step and M-step consistent** — they integrate
  the same object with the same measure.

**Advantages**
- **Mesh-free, and cost is dimension-agnostic:** `K` samples is one batched forward regardless of `d`. This is
  the property that lets the M-step scale where the grid could not.
- **Needs only pointwise `ℓ`** — no gradients, no tuning (no step size, no burn-in).
- **Embarrassingly parallel / vectorized** across all `T·B` time-sites.
- **Does *not* assume the target is Gaussian.** SNIS reweights an *arbitrary* target, so the operator's
  non-Gaussian, multimodal posterior survives — the non-Gaussianity is exactly what the operator exists to
  represent, and IS preserves it.
- **Empirically validated** as a drop-in: `em_1d` recovers `drift_l2 ≈ 0.19–0.25` (grid baseline ≈ 0.52) and
  `g ≈ 0.65` (true σ = 0.6); the over-dispersion `kl` is unchanged, as expected (that's an operator property,
  not a readout property). `em_highd` (random init) recovers the sensor to
  `c_cos = 1.000` — the Stiefel cross-covariance is fed the SNIS `z_hat` — with `drift_l2` and `g` tracking the
  grid run. So the mesh-free mean carries the full high-D recovery (sensor **and** dynamics), no grid in the M-step.

**Disadvantages**
- **Weight degeneracy in high `d`.** IS weight variance grows like `exp(KL(π‖q))`, and that KL grows ~linearly
  in `d`, so the effective sample size collapses `~exp(−c·d)`. SNIS is dimension-agnostic in *cost* but
  degrades in *accuracy* as `d` grows **unless the proposal tracks the posterior**. This is the central
  scalability limit.
- **Fails on modes the proposal misses.** If the target is multimodal (observation-gap bimodality; chaotic
  filtering distributions) and the proposal doesn't cover a mode, its weight vanishes and the estimate is
  silently biased.
- **`O(1/K)` self-normalization bias** (negligible at `K ~ 256`).
- **Only as good as the proposal** — which is where the scaling fight actually happens.

**A path we explicitly do *not* take: a Gaussian / Kalman-covariance proposal.** Tempting high-`d` booster:
use the linear-Gaussian posterior covariance `Σ = (CᵀC/σ_obs² + Σ_prior⁻¹)⁻¹` as the proposal covariance.
Rejected — it reintroduces a Gaussian assumption *exactly where non-Gaussianity is the point*. IS doesn't
assume the target is Gaussian, but a *unimodal Gaussian proposal* has vanishing overlap with far-apart modes,
so weights degenerate precisely in the multimodal regime that motivates the mesh-free operator. It fails
silently, and it couples the readout to a linear-sensor assumption we don't want to bake in. Off the table as
a principled move (at most a near-Gaussian micro-optimization, and not worth threading through the code path).

## MCMC — the leading high-`d` candidate

**What it is.** Construct a Markov chain whose stationary distribution is `π`; gradient variants (MALA, HMC,
Langevin) use `∇ log π` to propose efficient moves. Cost per effective sample scales roughly `d^{1/3}`–`d^{1/4}`
(MALA/HMC) versus IS's `exp(d)` — it is *built* for the high-dimensional regime.

**Why it fits *this* model unusually well** — beyond the generic "MCMC is for high-`d`":
1. **The readout is embarrassingly parallel across time.** We read `E[z_t]` from the operator's *per-time
   marginal* `π_t`, and those marginals are **independent across `t`** (each from its own GRU context, already
   computed). So sampling is `T·B` *independent* low-`d` problems with **no sequential-in-time axis** — the
   opposite of a particle filter. That's a large parallelism win we get for free from the operator's
   architecture.
2. **Gradient-native and assumption-free.** Langevin/HMC need only `∂_z ℓ` (and optionally `∂²_z ℓ`), which we
   *already compute* via `trunk_zderivs` (forward-mode `jvp`). No parametric proposal; the chain adapts to
   whatever non-Gaussian / multimodal shape the operator learned.
3. **Great initialization.** The pseudo-inverse `C⁺(y − d)` seeds every chain near the mode → short burn-in,
   which is what makes "a few steps × many chains" enough for a *moment* estimate.

**Parallelization — the reason this is appealing now.** Both axes of MCMC cost are attackable:
- **Parallel chains (breadth):** run `T·B × n_replicas` chains at once on the GPU. Diverse inits
  (pseudo-inverse ± perturbations, or draws from the broad proposal) let replicas populate *multiple modes*,
  handling the multimodality a Gaussian proposal cannot.
- **Parallel-in-time within a chain (depth):** the one remaining sequential axis is the MCMC step recurrence
  `z_{k+1} = z_k + ε ∇log π(z_k) + noise`, a nonlinear recurrence over step index `k`. **DEER**-style methods
  (Lim et al. 2024, *Parallelizing non-linear sequential models over the sequence length* — a Newton/Picard
  fixed-point solved with a parallel scan) parallelize exactly that recurrence. Between parallel chains and
  DEER-parallelized steps, the "MCMC is slow because it's sequential" objection largely dissolves in our
  setting.

**Open questions for the exploration (not implemented):**
- **Cost amortization** — MCMC per readout costs more than one IS pass, but the M-step fires only every
  `m_every` steps; the real comparison is MCMC-readout cost vs. E-step cost over that interval (likely small,
  worth measuring).
- **Multimodality coverage** — do diverse-init parallel chains actually populate the gap-posterior's modes,
  or do we need tempering / an annealed bridge (below)?
- **Diagnostics** — parallel chains give `R̂` and ESS almost for free; the same ESS number is the early-warning
  for when plain SNIS has run out of road.
- **We only need low moments** (`E[z]`, `Var[z]`), which converge faster than the full distribution — the bar
  is lower than general-purpose sampling.

## SMC / particle filters — where they don't fit (and where a variant might)

**What it is.** Sequential Monte Carlo propagates a particle population through time, reweighting by the
likelihood and resampling — the natural Bayesian filter, which yields `E[z_t | y_{0:t}]` as a byproduct.

**Why it's the wrong tool for *this* readout:**
1. **Redundant.** The operator has *already amortized* the filtering recursion; an SMC filter would re-run the
   filter the operator exists to replace.
2. **Circular.** A particle filter needs the transition kernel `p(z_t | z_{t-1})` — i.e. the `f, g` we are
   *learning in the same M-step*. Using it would tie the readout to the current, imperfect dynamics and break
   EM consistency. (The grid **oracle** does exactly this with the *true* dynamics — which is why it's a
   validation tool, not a training signal.)
3. **Path degeneracy** compounds in high `d`.

**Where an SMC-*sampler* idea does help:** not as a re-filter, but as a **bridge for multimodality**. An SMC
sampler / annealed importance sampling (AIS) uses MCMC moves to rejuvenate importance samples across a
temperature ladder that anneals from the broad proposal to the operator marginal `π_t`. That targets the
*same* per-time marginal (no dynamics, no re-filtering, no circularity) and is a legitimate fallback if
diverse-init parallel MCMC fails to cover modes.

## Guiding principle and trajectory

Match the estimator to the regime. Ours is: **low-to-moderate `d`, a good proposal (pseudo-inverse),
only low moments needed, massively parallel across sites, and a pointwise-evaluable *unnormalized* density
with cheap autodiff gradients.** That favors importance sampling now and gradient MCMC as `d` grows — and
argues against both fixed grids and any parametric (Gaussian) proposal.

All estimators sit behind one interface — *given pointwise `ℓ`, its `z`-gradient, and a center → return
`E[z]`* (`meshfree_mean` in `mstep.py`) — so we can swap the sampler without touching the EM:
1. **now:** SNIS — the verified beachhead; cost already scales.
2. **as `d` grows:** gradient **MCMC** (massively parallel chains + DEER-parallelized steps) — assumption-free,
   gradient-native on infrastructure we already have (`trunk_zderivs`), and unusually parallel because our
   marginals are time-independent. Add ESS / `R̂` diagnostics.
3. **validate scalability on Lorenz (2–3D), not 1-D** — 1-D is the easy end; the real answer is at `d > 1`.

**Not taken:** the fixed grid (`O(N^d)`), tensor / Gauss–Hermite quadrature (`O(n^d)` — dies like the grid),
and the Gaussian / Kalman-covariance proposal (fails on multimodal targets). SMC appears only as an AIS /
SMC-sampler *bridge* for multimodality, never as a re-filter of the dynamics we're trying to learn.
