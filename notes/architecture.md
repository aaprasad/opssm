# Architecture: the mesh-free neural Zakai filter

Read this first. The inline comments explain *local* choices; this explains how the pieces fit together.
None of it is obvious from the losses alone.

## What we're solving

We observe a low-`d` latent SDE through a (possibly high-`D`) noisy sensor and want the **filtering
posterior** `p(z_t | y_{0:t})` at every time `t` — for a *nonlinear* drift and a *non-Gaussian* posterior,
mesh-free (no `O(N^d)` grid), while simultaneously **learning** the latent drift `f`, diffusion `g`, and
sensor `C, d` by EM.

- **Latent SDE:** `dz = f(z) dt + g dW` (`f` a learned net, `g` an isotropic scalar).
- **Sensor:** `y = C z + d + noise` (`C` a learned `D×d` Stiefel matrix; direct-obs is `C = I`).
- The exact filter obeys the **Zakai / Fokker–Planck** equation. We amortize it with an operator instead of
  integrating a PDE per trajectory.

## Two timescales (this is the key mental model)

1. **Observation timescale `dt`** — how often measurements arrive (`dt = 0.1` in configs; *coarser* under
   gaps). Sets where the Bayes updates land.
2. **Dynamics timescale** — intrinsic to the SDE (drift/diffusion rates; e.g. Lorenz rotates on ~1 time-unit,
   so `dt=0.1` observes ~10× per rotation). Sets how fast the density flows *between* observations.

Everything below is organized around keeping these two separate.

## The operator: `ell(z, s)`, an UNNORMALIZED log-density

A physics-informed DeepONet emits `ell(z, s) = coeffs(ctx_t, s) · trunk(z) + bias`:

- **context encoder** (causal, GRU by default): compresses the history `y_{0:t}` into `ctx_t` — the learned
  belief state / approximate sufficient statistic. (Filtering needs the *whole* history, not just `y_t`; the
  encoder is that aggregation. See `wants-architecture-docs` / the encoder discussion.)
- **branch:** `ctx_t → coeffs(ctx_t, s)`.
- **trunk:** `z → trunk(z)` (the spatial basis), differentiated in `z` by autodiff (`trunk_grad`,
  `trunk_zderivs`) for the FP operator.
- **`s ∈ [0,1]` is continuous PHYSICAL TIME within one observation interval `[t, t+dt]`**, mapped as
  `phys_time = t + s·dt`. `s=0` is the **post-update** density at `t`; `s=1` is the **predicted** density
  (flowed forward by `dt`, ready for the next observation).
- `ell` is the **unnormalized** (Zakai) log-density on purpose — we never normalize during propagation; we
  normalize only at *readout*. Its total mass is the evidence `p(y_{0:t})`, which decays, so `ell` is pinned
  only up to a per-step constant.

## The recursion = FLOW within a step + JUMP between steps

Per step, two different kinds of equation:

```
                 predict (FP flow, CONTINUOUS in s)          update (Bayes, DISCRETE)
   ell_t(·,0) ───────────────────────────────────▶ ell_t(·,1) ──────────────▶ ell_{t+1}(·,0)
                ∂ell/∂(phys time) = L*_log(ell)              + loglik_{t+1}(z)
```

- **PREDICT — the `res` loss (autodiff, within a step).** Enforce the Fokker–Planck PDE in log-space,
  `∂ell/∂(phys time) = -(div f + f·∇ell) + ½ g² (|∇ell|² + Δℓ)`, at `n_scoll` points in `s` × `n_colloc`
  points in `z` (forward-mode `jvp`). Because it's solved *continuously across the interval* (not one Euler
  step), the predict stays accurate even when `dt` is coarse vs the dynamics — the dynamics timescale is
  resolved at `dt/n_scoll`. This is the "no grid, no time-stepping, no Euler" claim.
- **UPDATE — the jump loss (between steps).** `ell_{t+1}(z,0) = ell_t(z,1) + loglik_{t+1}(z)`, i.e. unnormalized
  Bayes `posterior = predict × likelihood` in log-space. `ell_{t+1}` is a *different context's* output
  (`coeffs(ctx_{t+1}, s=0)`), NOT an autodiff derivative — the observation is a discrete, causal event
  (`obs = Σ_k loglik_k · δ(t−t_k)`: zero during the flow, an impulse at each measurement), so it can't be part
  of the smooth `s`-flow without smearing causality. `loglik(z) = -0.5 ||y − (Cz+d)||² / noise²`.
- **IC — the ic loss.** `ell_0(z,0) = log_prior(z) + loglik_0(z)` (the first impulse, no predict in front).
- **Gaps** are the case where the two timescales differ: no observation for several `dt`, so `obs = 0` and the
  density just flows (pure FP) until the next measurement (the mask zeroes the jump's likelihood).

## EM: E-step fits the density, M-step fits the dynamics

- **E-step (every training step):** gradient-descend `w_res·res + jump + ic` to fit the operator's density to
  the Zakai recursion, with `f, g, C, d` held frozen. `losses.py` (`accumulate_pinn_grads`, chunked over the
  batch for memory).
- **M-step (every `m_every`):** read out the filter mean `ẑ_t = E[z_t|y_{0:t}]` from the operator, then
  regress `f` (drift, from the mean increment `(ẑ_{t+1}−ẑ_t)/dt`), `g` (diffusion, from the increment
  residual variance), and `C, d` (sensor). `mstep.py`. **Known limit:** the drift differentiates the mean path
  → `1/dt` error amplification → `g` inherits the leftover (`g_est² = g² + drift_rmse²·dt`); see
  `g-is-drift-limited` and the FP-inverse item in `TODO.md`.

## Identifiability / metrics

The latent is identifiable only up to an **affine** gauge: `z_true ≈ A·m_op + b` (the offset `b` is essential
for offset latents like Lorenz `z3 ≈ 25`). Validation Procrustes-aligns `(A,b)` then reports L2 / per-dim RMSE
/ relative errors for the latent, drift, and `g` (`filter_module._gauge_aligned`). Obs-space (gauge-free)
metrics are a TODO.

## The three SNIS → MCMC sites (the current work)

Self-normalized importance sampling appears in three places and degrades as `d` grows (`ess ~ exp(-cd)`):

1. **M-step mean readout** — `posterior_mean_*` → **Phase 1: MALA** (`mean_method=mala`). Done.
2. **M-step diffusion `g`** — marginal (mean-increment) vs the lag-one joint → **Phase 2**. Resolved: `g` is
   *drift*-limited, marginal wins (`joint_g=false`).
3. **E-step collocation jump/IC** — `colloc`: `snis` (default) | `mcmc` (contrastive-divergence EBM) |
   `consistency` (unnormalized-Zakai variance regression on the near points) → **Phase 3**. The FP `res` uses a
   plain average over a broad proposal (coverage) and is untouched by the ESS collapse.

## Where things live

- `opssm/models/operator.py` — the DeepONet (`context`, `coeffs`, `trunk`, `trunk_grad`, `trunk_zderivs`).
- `opssm/models/losses.py` — E-step: `sample_collocation`, `pinn_zakai_loss` (FP `res` + SNIS jump/ic + nll),
  `zakai_cd_update` / `zakai_consistency_update` (Phase-3 jump/ic), `accumulate_pinn_grads`.
- `opssm/models/mstep.py` — M-step: `filter_mean` (mala/fixed), `posterior_mean_mala` / `_mala_chains`,
  `fit_drift`, `fit_diffusion`, `filter_pair_*` (lag-one joint for `g`).
- `opssm/models/filter_module.py` — the Lightning module: EM schedule, hparams, gauge-aligned metrics.
- `configs/` — `model/operator.yaml` (all knobs, documented inline), `experiment/em_{highd,vdp,lorenz}.yaml`.
