# TODO

Cross-session task tracker for follow-up work that doesn't belong in the current PR.

## Fit drift/diffusion as an FP INVERSE PROBLEM (NEXT major work, after the MCMC ladder)

**Motivation (Phase-2 finding).** The M-step fits `f` by regressing the filter-MEAN increment
`(ẑ_{t+1}-ẑ_t)/dt`, i.e. it DIFFERENTIATES the mean path -- the `1/dt` amplification is why `drift_rel`
~57% at Lorenz from only ~6% `lat_rel`, and `g` inherits the leftover: measured `g_est^2 = g_true^2 +
drift_rmse^2*dt` (VdP 0.35 vs 0.36; Lorenz 11.8 vs 11.3; Lorenz drift-residual term 131 >> true g^2 9). So
`g` is DRIFT-LIMITED, not sampler-limited -- see the `g-is-drift-limited` memory. The fix is to stop
differentiating the mean and instead recover `f,g` as the unknown COEFFICIENTS of the PDE the density
already satisfies (a PINN inverse problem).

**Reference.** Liu, Kou, Park, Lee, "Solving the inverse problem of time independent Fokker-Planck equation
with a self supervised neural network method", Sci. Reports 11:15540 (2021), doi:10.1038/s41598-021-94712-5.
FPE-NN embeds the FP terms as trainable weights and recovers them from observed densities by ALTERNATING
(a) fit terms with density frozen, (b) denoise density with terms frozen -- structurally our EM (M-step /
E-step). Findings: (1) recovers BOTH drift and diffusion (as functions of state); (2) alternating, NOT
one-shot -- a single linear-least-squares fit on the smoothed-noisy pdf fails badly (their Fig 4), the terms
only come out once the density is DENOISED in the loop (their `L_P` data-anchor is what gives the residual
teeth); (3) they are 1-D, so they DODGE the rotational-current non-identifiability that bites us at d>=2.

**How it maps onto us (small change -- ~90% is built).** `pinn_zakai_loss` already computes the log-space
residual `d_t ell = -(div f + f.grad ell) + 1/2 g^2 (|grad ell|^2 + Laplacian ell)`; today its gradient
flows only to `ell` (`f,g` detached from the regression M-step). The change: STOP detaching -- fit `f,g` to
this residual in the M-step (DriftNet via autograd; scalar `g` in closed form). We are strictly better-
equipped than the paper: mesh-free + exact autodiff + CONTINUOUS-time residual (no x-grid, no finite-diff, no
multi-step Euler truncation -> not stuck in 1-D), and our density is already data-anchored by the jump/IC
(observation-likelihood) terms = the analog of their `L_P` safeguard.
- **Closed-form `g^2`** (residual is linear in `g^2`): `g^2 = <A,B>/<B,B>`, `A = d_t ell + div f + f.grad ell`,
  `B = 1/2 (|grad ell|^2 + Laplacian ell)`, summed over collocation points. Fits `g` to the LAPLACIAN physics
  signature, not the leftover increment variance -> dissolves the `g_est^2 = g^2 + drift_rmse^2*dt` coupling.
- **Two things that decide whether it beats regression:**
  1. **Identifiability at d>=2 (the gauge the paper dodges via 1-D).** The FP residual pins the compressive/
     gradient drift + `g` cleanly but leaves the divergence-free (ROTATIONAL) current free -- exactly the
     Helmholtz-SDE rotational-drift gauge (see the divergence-free diagnostic idea below). Increments DO see
     rotation. So the likely winner is a HYBRID: residual for the clean part + `g`, increments (or the Zakai
     observation term, which the free-FP argument ignores) for the rotational part. Test residual-only vs
     hybrid.
  2. **Keep `ell` DATA-driven, not term-driven.** If the E-step bends `ell` too hard toward the current
     (wrong) `f,g` (`w_res` too high vs jump/IC), the M-step residual-fit just confirms them (co-adaptation
     fixed point). The jump/IC anchor prevents runaway; the `w_res` balance is the knob to watch.

**Plan.** Start d=1 on `em_highd` -- the paper's exact regime (no rotational gauge) + we have the exact-filter
KL as ground truth. A/B the residual-fit `f,g` (closed-form `g^2` FIRST -- a few lines, directly tests "does
physics-`g` beat increment-`g`") vs the regression M-step. If it wins at d=1 (it should), go to d=2 (VdP) /
d=3 (Lorenz) where the rotational-gauge question decides residual-only vs hybrid. Subsumes the deferred drift
work (higher-order increment / finer dt below stay as complementary levers for the rotational part).

## Port the E-step batch CHUNKING to the JAX backend (memory parity with torch)

The torch E-step splits the batch and accumulates gradients per chunk (`losses.accumulate_pinn_grads`,
`chunk_size=16`), so peak memory tracks the chunk, not the batch. The JAX `estep` (`models/jax/train.py`,
`make_estep`) is ONE whole-batch `eqx.filter_jit` call with no equivalent. Consequence measured 2026-09-09
on a 32 GB RTX 5090: **`experiment=em_lorenz` cannot run at its default config on the JAX backend at all** --
the BASELINE arm (`diffusion_cov=false`, i.e. nothing exotic) dies with
`RESOURCE_EXHAUSTED: ... allocate 22.00GiB` in `jit_estep`. `n_colloc=384 n_tcoll=24` OOMs;
`n_colloc=256 n_tcoll=12` and below fit.

So the two backends do NOT have the same reachable configuration space, and the published Lorenz settings
are torch-only. That also silently biases any JAX grid search away from large `n_colloc` at d=3.

Fix: chunk the JAX E-step over the batch axis and sum the gradients (a `lax.scan`/`fori_loop` over chunks
accumulating into a grad pytree, or `jax.checkpoint` on the residual path). The residual is per-point and the
recursion is per-trajectory -- both independent across the batch -- so the chunked result is EXACT, exactly as
in torch. Cheap correctness check: chunked vs unchunked gradients must match to float tolerance on a small
config.

## Fix the `nll` gradient detach (dormant; do before enabling Stage-3 NLL training)

In `pinn_zakai_loss` the data-NLL is `logc = logsumexp(logW_pred + loglik)` with `logW_pred =
log_softmax(ellT_next - log_q).detach()`. The `.detach()` is correct for the JUMP (there `logW_pred` is the
bootstrap *target*), but it's REUSED for `nll`, where it kills the gradient through the operator's predict
density `ellT_next` (and thus through `f`,`g`). So `nll`'s gradient reaches ONLY the sensor `C` (via `loglik`).
Currently harmless -- `w_nll=0` everywhere, so `nll` is logged, never backpropped -- but it would silently
cripple `w_nll>0` ("Stage 3, learn dynamics"). Fix (value-preserving, restores the operator/dynamics gradient):
compute the grad-carrying self-normalized log-evidence WITHOUT detaching the `ellT` part:
`logc = logsumexp(ellT_next - log_q[1:] + loglik[1:], dim=-1) - logsumexp(ellT_next - log_q[1:], dim=-1)`,
keeping the separate `logW_pred.detach()` for the jump. (Caught in review; see the Phase-3 plan.)

## Remove `pca_init` from the codebase (own PR)

`pca_init` seeds the unit sensor `C` at the data's top principal-component direction (~= the true
sensor direction) — a data-driven head start that pre-solves the sensor and makes the "we learn the
sensor from scratch" claim vacuous. The M-step recovers `|cos| -> 1` from a **random** init on its own,
so the flag has no honest use. Default is already flipped to `false` (see `configs/model/operator.yaml`);
this task is to **delete the code path entirely**:

- `opssm/models/filter_module.py`: remove the `pca_init` hparam (ctor ~L61) and the `if h.pca_init:` PCA
  seeding block (~L105); always random-init `C`.
- `configs/model/operator.yaml`: drop the `pca_init` key + comment.
- Grep for any other `pca_init` references (experiment configs, notebooks) and clean up.

## Code style: type hints + Google-style docstrings (own PR)

Standardize the codebase on PEP 484 type hints on function signatures and Google-style docstrings
(`Args:` / `Returns:` / `Raises:`) wherever practical. Prioritize the public APIs -- `opssm/models/*.py`
(operator, mstep, losses, dynamics, filter_module) and the datamodule. KEEP the dense inline "why" comments
(they carry the hard-won design rationale); this is about adding signatures + structured docstrings, not
rewriting the commentary. Do it as a mechanical sweep in its own PR so it doesn't clutter feature diffs.

## Profile the training step, then decide on a JAX/Julia backend (research spike)

**STEP 1 -- full profile FIRST.** Run a `torch.profiler` pass over a training step (E-step residual + jump/ic
+ GRU context + M-step readout) and get the per-op / per-kernel breakdown + kernel-launch vs compute time,
at d=1,2,3. This decides whether a backend switch pays off and where. Prior evidence points to a
launch-overhead / no-fusion bottleneck (not a single tunable op): `chunk_size` 4->16 bought only ~18% and
vmap-ing the tangent loop was perf-neutral, i.e. the cost is spread across MANY small ops (the per-chunk `jvp`
Jacobians + einsums + the chunk loop). Measure: fraction of time in launches vs kernels, and the top ops.

**STEP 2 -- JAX (and/or Julia) backend IF the profile supports it.** Many small ops with launch overhead is
exactly what XLA `jit` fusion targets. Hot paths: the forward-mode `jvp` Jacobians (trunk gradient/Laplacian,
drift divergence -- `operator.trunk_zderivs` / `trunk_grad`, `dynamics.DriftNet.drift`) and the MALA /
collocation sampling loops. JAX's `jit` + `vmap` + forward-mode AD (or Julia's compiled AD) could fuse the
per-chunk ops and make the DEER parallel-scan (MCMC Phase 4) natural to express. Scope: prototype the operator
+ one loss in JAX, benchmark vs the PyTorch path, decide whether a full port is worth it. Large lift -- a
spike/decision, not a scheduled port.

References for the port:
- **JaxLightning** -- https://github.com/ludwigwinkler/JaxLightning -- a PyTorch-Lightning-style training loop
  in JAX; a template for porting the `ZakaiFilterModule` / Trainer structure without hand-rolling the loop.
- **jNO (JAX Neural Operators)** -- https://fhg-iisb.github.io/jNO/ -- for the operator / FNO layers in JAX
  (the DeepONet trunk/branch, and the differential-operator layers).

## Gauge-invariant observation-space metrics (own PR)

Our headline latent metrics (`lat_rmse_aln`, `drift_l2_aln`) score in LATENT space, so they need the Procrustes
gauge and are sensitive to the frame -- this is what forced the AFFINE gauge for Lorenz's z3 offset (~25).
Helmholtz-SDE (Smith, Trippe, Linderman 2026) avoids the issue entirely by scoring GAUGE-INVARIANTLY in
observation/output space. Add the same, ALONGSIDE the aligned latent metrics:
- **obs-space posterior LL**: log-likelihood of the true latents pushed through the sensor `Cz+d` under the
  operator posterior predictive over `y = Cz+d+noise`, averaged over t and trials -- no gauge needed.
- **prior-pushforward symmetric KL**: sample the learned prior SDE (drift/g), push through `Cz+d`, and compare
  the distribution to the true prior's pushforward (KDE-based sym-KL, as in Helmholtz Fig 3 / Kiyohara).
- **output autocorrelation error** (multi-time): lagged autocorr of the pushed-through samples vs true.
Frame-free (the sensor absorbs the latent gauge), robust to offset/anisotropy, and make us directly comparable
to the Helmholtz-SDE / SING / SDE-Matching numbers.

## Duncker/Linderman benchmark suite — BUILT; deferred follow-ups

The three Duncker et al. 2019 systems are implemented and run (`configs/experiment/duncker_{dw,vdp,lorenz}.yaml`;
`vanderpol_duncker` drift + `x0_uniform` in `systems.py`; full results + settings + deviations in
`notes/duncker_bench.md`). Run in a well-covered dense-obs regime: Lorenz strong (`lat_rel 0.010`, `drift_rel
0.176`), VdP decent (0.18 / 0.45), double-well healthy at a=1 (0.406). Three follow-ups to make this a real
head-to-head:

1. **Obs-space gauge-invariant metrics FIRST** (see the section above). The Duncker/Helmholtz numbers are
   obs-space; our latent-gauge `drift_rel`/`lat_rel` are NOT directly comparable. Wire the obs-space metrics into
   the `duncker_*` experiments -> then the comparison is meaningful. This is the gating dependency for the
   Baselines below.
2. **Sparse/irregular-obs ablation.** Our runs (and the drift oracle) use DENSE regular obs; Duncker observes at
   ~20 random uneven time-points/trial. Add a random observation `mask` + operator variable-dt support, then
   compare dense vs sparse. This is the regime where their GP-smoothed continuous-time approach earns its keep.
3. **a=4 double-well at finer obs dt.** Duncker's steep `4x(1-x^2)` breaks the drift M-step at obs dt=0.1
   (`|f|*dt=0.75`, large-step; both forward AND det_mid; a quick dt=0.025 test destabilized). A careful dt sweep
   with stable `n_sub` to see whether resolution alone rescues it, or whether the sharp 1-D well needs a
   structural fix (coverage/importance). Details in `notes/duncker_bench.md`.

## Baselines

External baselines to benchmark against once the MCMC ladder + obs-space metrics land.

- **Helmholtz-SDE noisy-Lorenz** (Smith, Trippe, Linderman 2026, "Closing the Approximation Gap in Simulation-
  free Latent SDEs", arXiv:2606.16138; same Linderman-lab lineage as SING [Hu et al.] and the Duncker GPSLDS we
  already cite). A clean, published comparison point in exactly our problem space, deliberately in the HIGH
  posterior-uncertainty regime. Setup: K=4 latent, Lorenz drift `(alpha,rho,beta)=(10,28,8/3)`, process noise
  sigma=5, observed every dt=0.25, obs noise eta=0.3, standardized linear-Gaussian sensor
  `Ctrue=diag(s^-1), dtrue=-m*s^-1`, 1024 trials. Gauge-invariant metrics (use the obs-space metrics above):
  posterior LL of true latents through the output map, symmetric KL to the true prior (KDE), global + LOBE-WISE
  lagged autocorrelation error. Their reported Helmholtz-SDE numbers: latents LL -0.09, sym-KL 0.06, lobe autocorr
  0.052 (Table 1 / Fig 3); simulation-free, ~20x faster than simulation-based SING at matched nELBO. Run OURS on
  this exact setup and compare. Contrast probed: they use GAUSSIAN one-time marginals + a closed-form
  divergence-free drift correction (VI); we use a NON-Gaussian FP posterior + Zakai filtering -- so the head-to-
  head directly tests the Gaussian-marginal ceiling vs our sampling cost. NB their latents are identifiable only
  up to an AFFINE gauge (Sigma fixed=I -> orthogonal+translation); they sidestep it with obs-space metrics.

## Ideas / future directions (not scheduled)

From the NKE and PINNs-Bayes reading (see `related_work.md`):

- **Strang / trapezoidal 2nd-order drift.** NKE frames Euler-Maruyama as 1st-order Lie-Trotter splitting and
  uses a trapezoidal "meet-me-halfway" advection loss (their Eq. 33) for 2nd-order accuracy. We *proved* the
  trapezoidal drift is invalid on the filter MEAN (a propagation, not an SDE integral) and the smoother mean
  over-attenuates -- so this is blocked until the smoother-mean-for-drift problem is solved. Adopt the
  splitting *framing* in the writeup; the concrete gain waits.
- **Temporal FD-conv drift target (higher-order dz/dt).** Compute the drift increment `f(z)·dt` via a
  finite-difference convolution (`neuralop.layers.differential_conv.FiniteDifferenceConvolution`) along the
  TIME axis -- a free 1-D uniform grid, so NO `O(N^d)` cost. A central/higher-order stencil removes the O(dt)
  forward-difference bias in the current target `(z_{t+1}-z_t)/dt`, most plausibly helping STIFF drifts
  (Lorenz). SAME caveat as the 2nd-order-drift item above: `ẑ(t)` is a FILTER MEAN (predict + observation-update
  jumps, and a wide stencil subtracts means under DIFFERENT filtrations `y_{0:t-1}` vs `y_{0:t+1}`) -- not a
  smooth SDE path, so a wider-in-time stencil blends updates/conditioning. Cheap to try AFTER the MCMC ladder
  (MALA already smooths the mean, so the effect shows in isolation); MEASURE whether it actually lowers
  `drift_l2_aln` on Lorenz rather than assume it. NB rejected alternatives: (a) autodiff `dz/dt` -- `d/ds` of
  the operator mean equals the operator's OWN drift (circular, no data signal); (b) parametrizing `f(z)` itself
  with FD-conv over the LATENT grid -- reintroduces `O(N^d)` and breaks mesh-free FP evaluation (we already have
  a mesh-free differential-operator drift: `DriftNet` + autodiff `div`/`grad`).
- **Divergence-free / rotational drift diagnostic** (from the Helmholtz-SDE reading). Their thesis is that
  marginal-based posteriors fail on the ROTATIONAL (q-divergence-free) component of the drift -- the circulatory
  probability current -- precisely where posterior uncertainty is high (VdP limit cycle, Lorenz within-lobe
  spiral). Check whether OUR learned drift captures that current: Helmholtz-decompose the learned vs true drift
  (gradient + divergence-free parts) on the data regime and compare the ROTATIONAL component, not just the scalar
  `drift_l2_aln`. Our drift is learned from increments (not derived from marginals), so we SHOULD get it -- but
  the filter-mean increment target + observation-update conflation could damp the rotational part; this measures
  it. Cheap once the affine gauge + a drift field are in hand.
- **Continuous-time switching LDS via probabilistic jumps (own paper).** Represent the jump kernel as a
  categorical-over-latent-states generator (or, equivalently, a *probabilistic* operator split -- mix over
  which sub-generator acts), giving discrete regimes inside the Zakai/KFE machinery. Strategically relevant
  for neural-activity latents (up/down states, regime switches). NOT low-hanging: filtering can't threshold
  hidden jumps (NKE's mechanism needs observed increments), and Zakai-with-jumps is a PIDE-SPDE. Separate paper.
- **True EM: an expected-NLL M-step over the posterior, not a plug-in mean (research spike).** The M-step
  today is *approximate* (mean-field) EM. `fit_drift` / `fit_diffusion` / `fit_obs_map_stiefel` already
  minimize the Gaussian complete-data NLL of the generative model `z_{t+1}|z_t ~ N(z_t + f(z_t) dt, g^2 dt)`,
  `y_t|z_t ~ N(C z_t + d, sigma^2)` -- but evaluated at the SINGLE posterior mean `ẑ = E[z|y]`, not as
  `E_q[.]` over the full posterior. That point-collapse is the entire gap to true EM and is what forces the
  existing patches: it drops (a) the posterior covariance -> `g` under-reads (`g_est^2 = g^2 + drift_rmse^2
  dt`; the `joint_g` workaround), and (b) `E[f(z)]` vs `f(E[z])` -> the Jensen / errors-in-variables bias (the
  `det_mid` / `ito_correction` workarounds). Fix: fit `f, g, C` to posterior SAMPLES and average the NLL over
  them (Monte-Carlo / stochastic EM) instead of collapsing to `ẑ` first -- the MALA chains in the E-step
  ALREADY draw these samples; today they are averaged into `ẑ` before the fit. Use the JOINT `q(z_t, z_{t+1})`
  (the `filter_pair_*` machinery) so the increment variance is what identifies `g`. Payoff: reintroducing the
  spread should retire `joint_g` AND `det_mid`/`ito` together (all three claw back terms the mean-collapse
  discards). Alt route (Duncker / gpSLDS style): keep a Gaussian `q` with an explicit covariance and take the
  `E_q[.]` terms ANALYTICALLY rather than by sampling. Caveat: this is orthogonal to the filter-vs-smoother
  approximation -- the E-step would still be the causal FILTER, not the smoother that exact EM for an SSM
  wants; a faithful EM would need both fixes. See also the FP-inverse-problem section above (a different
  attack on the same "stop differentiating the mean path" M-step problem).
