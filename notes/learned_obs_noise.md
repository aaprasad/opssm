# Learning the observation noise R (and why `recon_r2` is a dangerous objective)

`R` was the only generative parameter the M-step never fitted, yet it sets the likelihood sharpness in the
jump term, hence the posterior width, hence -- through the increment residual -- both `g` and the drift. On
real data it is a guessed hyperparameter (`configs/data/kato.yaml` notes 0.1 vs 0.5 swings whole-trace
recon 0.84 <-> 0.77). This note records the experiment that justified fitting it.

## Design note: `em_highd` is a NULL test by construction

`em_highd`'s assumed noise IS the true generative value (`noise_std=0.3`, divided by the same `obs_scale`
the data uses -> 0.1565). So R has nothing to fix there and the plain A/B can only show whether learning it
HURTS. Measured, 3 model seeds on a frozen dataset: it does not. Paired deltas were drift_rel +0.005,
lat_rel -0.000, kl_aln +0.000, g_rel +0.003, recon_r2 +0.000, with the drift sign flipping across seeds
(+0.014/+0.004/-0.004) and everything inside the 0.0155 model-seed spread. R itself came out 0.1563 in all
three seeds. A clean regression check, nothing more.

The informative experiment keeps the OBSERVATIONS BYTE-IDENTICAL and corrupts only the model's BELIEF about
the noise, by overwriting `noise_std_eff` in the frozen npz (the model reads its assumed R from there;
the actual noise is baked into `x_train`).

## Result: R is load-bearing, and the estimator recovers a 3x error either way

True R std 0.1565. Same observations in every row. 14k steps, seed 0.

| assumed R | drift_rel | lat_rel | kl_aln | g_rel | recon_r2 |
|---|---|---|---|---|---|
| correct 0.1565, fixed | 0.2171 | 0.1401 | 0.0187 | 0.9034 | **0.8029** |
| 3x too big (0.4695), fixed | 0.8428 | 0.3875 | 3.2771 | 0.1883 | 0.6725 |
| 3x too big, LEARNED (-> 0.1611) | 0.3280 | 0.1455 | 0.0521 | 0.7889 | 0.7984 |
| 1/3 too small (0.0522), fixed | **1.0596** | 0.1575 | 0.2703 | 1.4064 | **0.8113** |
| 1/3 too small, LEARNED (-> 0.1557) | 0.2039 | 0.1398 | 0.0229 | 0.9262 | 0.8036 |

Damage removed by learning R: drift 82% / 102%, latent 98% / 101%, kl_aln 99% / 98%, g 84% / 96%,
recon 97% / 92% (3x-too-big / 1/3-too-small). The ~101% entries are NOT a real gain -- learned R landed at
0.1557 vs the true 0.1565, so those runs are effectively correctly specified and the excess is inside the
0.0155 model-seed spread.

**The two directions break the model through OPPOSITE mechanisms**, both consistent with the causal chain:
- **R too big** -> likelihood too flat -> prior dominates -> filter over-smooths -> increments too smooth ->
  `g` COLLAPSES (g_rel 0.19) and the latent/KL degrade badly (kl_aln 175x worse).
- **R too small** -> likelihood too sharp -> filter tracks OBSERVATION NOISE -> increments noise-dominated ->
  `g` OVER-reads (g_rel 1.41) and the drift is swamped (drift_rel 1.06 = no usable drift).

## `est="perp"` is why it works

The default estimator uses only the `D-d` observation directions ORTHOGONAL to `span(C)`, where no latent can
contribute: `R_i = E[perp_i^2] / (1 - ||C_[i,:]||^2)`, `perp = (I - C C^T)(y-d)`. It **read 0.1563 at EVERY
M-step of BOTH misspecified runs** -- invariant to whether the model started 3x too high or 3x too low, and
to the resulting badly-wrong posterior. The textbook `est="posterior"` estimator (`E_q[(y-Cz-d)^2]` over MALA
samples) is contaminated exactly as predicted: it read 0.1932 at step 2000 of the 3x run (true 0.1565),
inflated by the over-wide posterior, and only crept to 0.1601 as the operator sharpened. Correct-if-q-were-exact
is not good enough when q is known to be 1.5-2x too wide (`notes/overdispersion.md`).

**RESOLVED: the EMA damping WAS the bottleneck; default is now 0.** `obs_noise_damp=0.5` halves the gap per M-step, so R
crawled 0.4695 -> 0.3499 -> ... -> 0.1611 over 7 M-steps while the correct answer was available at step 2000.
That is why the damped 3x case recovers only 82% on drift. The 2x2 settles it -- damage removed on drift_rel:

| estimator | damp 0.5 | damp 0 |
|---|---|---|
| perp (default) | 82.3% | **101.2%** |
| posterior | - | 89% |

Undamped `perp` jumps to 0.1563 at the FIRST M-step and finishes at drift_rel 0.2098 vs the correctly-
specified 0.2171 -- a 3x misspecification becomes essentially INVISIBLE. And the feedback loop the damping
guarded against does not materialize even for `posterior`, which self-corrects 0.1935 -> 0.1595 (true
0.1565) rather than latching on: the D-d directions orthogonal to C anchor it, so the gain is only ~d/D
(~0.1 here), exactly as predicted up front. **`obs_noise_damp` now defaults to 0.0.**

## `recon_r2` is ANTI-correlated with dynamics quality here

The 1/3-too-small run has the **best reconstruction of all five** (0.8113 vs the correctly-specified 0.8029)
while having the **worst drift** (1.06 = essentially no usable drift). It scores well *because* it fits the
observation noise. This is direct evidence for the concern `docs/grid_search.md` already flags as an
"objective caveat": the Optuna sweep maximizes `recon_r2`, so it can actively PREFER a model whose dynamics
are destroyed. Selecting on reconstruction is not safe; a forward-simulation k-step prediction metric is.
