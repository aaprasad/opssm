# Over-dispersion: root cause and fix (scale-invariant FP residual)

**Symptom.** The operator filtering posterior was ~3–5× too wide (std ≈ 0.44 vs the exact filter's ≈ 0.10 on
the dense 1-D double-well), while its *mean* path was correct. Validation KL to the exact filter plateaued
~0.5–0.7. This over-width also drove the diffusion over-estimation (`g² = Var[Δẑ]/dt` absorbs the extra width).

## What it is NOT (ruled out)

- **Not representational / DeepONet capacity.** Supervised fitting of the exact filter reaches KL ≈ 2e-4 with
  the *same* architecture — the operator can represent the sharp posterior. Richer within-interval time features
  (Fourier `[s, sin/cos(k·2πs)]`, the previously-hypothesized fix) gave **no** improvement (val KL 0.625 vs
  linear 0.669 at 8k — noise). Reverted.
- **Not an init / basin problem.** Continuing from a *perfect* narrow start (operator pre-trained supervised to
  std 0.089, then trained on the Zakai loss with the **true** dynamics) the width **blows up** — std 0.089 → 0.71,
  KL 0.005 → 2.0 — in a few hundred steps. So the self-supervised objective **actively pushes away from the exact
  filter**. The correct filter is an *unstable* point of the loss as originally implemented.

## What it IS (mechanism)

From the same narrow start, an ablation isolates the two forces:

| continue from narrow start (std 0.089, true dynamics) | std_op | KL |
|---|---|---|
| FULL loss (residual + jump + ic) | 0.63 (over-disperse) | 2.5 |
| jump + ic only (no FP residual) | 0.028 (collapse) | 55 |

The **likelihood update (jump) sharpens**; the **FP residual spreads**. The correct width is their balance, and
the residual **over-spreads**. Why: the FP residual is evaluated in **log space**, where the terms scale as
`∂_z ℓ ~ z/σ²`, `∂²_z ℓ ~ 1/σ²`. For a *sharp* density these EXPLODE at collocation samples far from the mode,
so the **absolute L2 residual `(∂ℓ/∂t − rhs)²` is dominated by the far-tail samples** — its gradient norm is
**7796** (l2) vs ~**200** for the normalized versions below. Minimising that absolute magnitude, SGD **widens σ**
to de-amplify the tail terms → over-dispersion. (This is a conditioning pathology of the L2 residual, not a wrong
objective: the true narrow filter has an *exactly zero* residual — `σ²(s) = σ²₀ + g²·dt·s` solves the FP PDE.)

## The fix: `res_mode="rel"` (scale-invariant residual)

Measure the residual **relative to the local FP magnitude** instead of absolutely:

```
res2 = (∂ℓ/∂t − rhs)²
res_loss = mean( res2 / (rhs.detach()² + (∂ℓ/∂t).detach()² + 1) )     # res_mode="rel"
```

A far-tail sample with huge `rhs` and huge `res2` now normalises to ~O(1) instead of swamping the batch, so the
tail can't dominate — **and the broad proposal is kept for coverage** (needed by the M-step and the SNIS
normaliser). `res_mode="log1p"` (log-compress `res2`) is a simpler variant but empirically weaker.

**Result (random init, true dynamics, isolating the operator objective):**

| residual mode | final std_op (exact = 0.089) | final KL |
|---|---|---|
| `l2` (original) | 0.63 | ~1.0 |
| `log1p` | 0.29 | 0.69 |
| **`rel`** | **0.186** | **0.35** |

So `rel` moves the stable point from L2's 0.63 down to ~0.19 (≈3.4× narrower, KL roughly halved). The remaining
~2× gap to the exact 0.089 is a genuine *equilibrium* of the rel-residual↔jump balance (a narrow start relaxes
back up to ~0.19), **not** closed by proposal resolution:

- **`res_post`** (posterior-weight the residual): weak (0.585 vs 0.63) — the uniform floor still admits the
  exploding tail samples.
- **`near_std`** (tighter proposal for jump resolution): *not a lever* — near 0.1 / 0.15 / 0.3 all reach the same
  ~0.19–0.24 with broad kept.
- **Reducing `broad_std`**: *backfires* (worse) — it starves the SNIS normaliser / predict-density coverage.

## Status / open

`rel` is wired as the default (`configs/model/operator.yaml: res_mode`, `ZakaiFilterModule(res_mode=...)`,
threaded through `losses.pinn_zakai_loss` / `accumulate_pinn_grads`).

**Full-EM verification** (`experiment=em_1d`, 8k steps, rel vs l2, same seed/data — the M-step in the loop):

| step | KL (rel / l2) | g (rel / l2), σ=0.6 | drift_l2 data-regime (rel / l2) |
|---|---|---|---|
| 4000 | 0.713 / 0.987 | 0.680 / 0.652 | 0.093 / 0.105 |
| 6000 | 0.431 / 0.770 | 0.624 / 0.633 | 0.071 / 0.078 |
| 8000 | **0.349 / 0.669** | **0.608 / 0.633** | **0.064 / 0.069** |

`rel` roughly **halves the over-dispersion KL** (0.67 → 0.35), brings **g closer to the true 0.6** (0.633 → 0.608
— the over-estimation shrinks, since it was downstream of the over-width), and **preserves drift recovery**
(0.069 → 0.064, slightly better). No downside — wired as the default.

## Closing the last ~2×: residual↔jump weight (`w_res`)

The rel equilibrium (~0.19, still ~2× the exact 0.089) is where the residual's *spread* balances the jump's
*sharpen*. Down-weighting the residual (`w_res` on the residual term, jump/ic kept at 1) slides that balance
toward the exact width — cleanly, because `rel` made it stable (the raw L2 residual made this knob a runaway).
Isolated (true dynamics, random init):

| `w_res` | std_op (exact 0.089) | KL |
|---|---|---|
| 1.0 | 0.186 | 0.35 |
| 0.4 | 0.130 | 0.108 |
| 0.2 | 0.116 | 0.070 |
| 0 (no residual) | 0.028 (collapse) | — |

**Full-EM trade-off** (`experiment=em_1d`, 8k, rel + `w_res`) — the residual also couples the operator to the
learned `f, g`, so pushing `w_res` low trades posterior width against dynamics recovery:

| `w_res` | KL | drift_l2 (on-data) | g (σ=0.6) |
|---|---|---|---|
| 1.0 | 0.35 | 0.064 | 0.608 |
| 0.4 | 0.128 | 0.094 | 0.564 |
| **0.2** | **0.055** | 0.13* | 0.520 |

*The `w_res=0.2` drift *looks worse by the old metric* but fits the true cubic **well in-regime** — the error is
concentrated in the off-data tail (|z| ≳ 1.4) where the regression extrapolates unconstrained. The drift metric is
now **split into on-data / off-data** (`drift_l2` / `drift_l2_off`) so the in-regime fit isn't masked by tail
divergence. `g=0.52` is *under* 0.6 but close to what the exact filter's own mean path yields (~0.58); the proper
`g` fix is the joint-sample estimator (open), not `w_res`.

**1-D default: `res_mode=rel`, `w_res=0.2`** — the closest match to the exact filter (KL 0.67 → 0.055, ~6×), with
the drift still recovering in-regime.

## `w_res` is dimension-dependent (high-D wants a higher weight)

The optimal `w_res` **rises with latent-observability**: a higher-D sensor (here 10-D, `y = Cz + d + ε`) is 10
independent measurements of the same scalar `z`, so the *true* posterior is much sharper than the 1-D direct-obs
case — and a sharper target needs *less* residual down-weighting to reach. High-D sweep (`experiment=em_highd`,
14k, `C cos → 1.0` throughout):

| `w_res` | KL | drift_l2 (on-data) | g (σ=0.6) |
|---|---|---|---|
| 0.2 | 0.420 | 0.380 | 0.585 |
| 0.3 | 0.374 | 0.387 | 0.597 |
| **0.4** | **0.356** | 0.374 | 0.610 |
| 0.6 | 0.356 | 0.360 | 0.632 |

KL falls 0.42 → 0.356 and then **saturates at `w_res=0.4`** (0.6 is identical); `g` is on-target near 0.4 and
starts over-shooting past it. So **high-D default `w_res=0.4`** (`configs/experiment/em_highd.yaml`), vs 1-D's 0.2.
Note the high-D drift stays ~0.36–0.39 for *all* `w_res` (much softer than 1-D's ~0.13) — the drift floor there is
set by the latent-readout noise (`C⁺(y−d)`) and the sensor-before-dynamics curriculum, not by `w_res`; sensor
(`C cos=1.0`) and `g` (~0.6) are both excellent regardless.
