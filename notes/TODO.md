# TODO

Cross-session task tracker for follow-up work that doesn't belong in the current PR.

## ~~Fit drift/diffusion as an FP INVERSE PROBLEM~~ -- TRIED, DEAD END (2026-08-11)

**CONCLUDED NEGATIVE. Do not re-attempt.** Full write-up + numbers: `notes/fp_inverse_status.md` (branch
`fp-inverse`, do NOT merge); memory `fp-inverse-drift-dead-end`. Both residual modes diverge on em_highd d=1:
- **FP-flow** (`inv_mode=fp`, `d_t ell = L*(f,g)`, no data): `f,g` never see the observations (the likelihood
  lives in the jump, not the flow), so they co-adapt with the residual-trained `ell` and slide off the true
  filter. drift_rel 1.0->1.4, kl 0.72-0.99. NO weighting/anchoring (`w_res`, `w_g`) rescues it.
- **Zakai-step** (`inv_mode=zakai`, adds the loglik so `f,g` DO see data): diverges HARDER -- g runs away UP to
  ~17, drift_rel ~220. `A=(ell_{t+1}-ell_t-loglik)/dt` divides the discrete O(1) likelihood jump by dt, so the
  operator's imperfect jump reconstruction is amplified 1/dt and `g^2` inflates to absorb it.

Root cause vs the FPE-NN paper (Liu et al 2021, Sci Reports 11:15540): they OBSERVE the density (fixed
data anchor incl. its WIDTH -> well-posed inverse); we INFER it from noisy POINT obs, so `ell` is free to slide
with `f,g` and the width `g` sets is unobserved. The drift bottleneck below stays open; the LIVE lever is the
increment-side work (keep the stable data-driven `f`, cut its differentiation error), not the inverse.

## Drift bottleneck (OPEN) -- the live lever is INCREMENT-side, not the inverse

The M-step fits `f` by regressing the filter-MEAN increment `(ẑ_{t+1}-ẑ_t)/dt`, i.e. it DIFFERENTIATES the
mean path -- the `1/dt` amplification is why `drift_rel` ~57% at Lorenz from only ~6% `lat_rel`, and `g`
inherits the leftover: `g_est^2 = g_true^2 + drift_rmse^2*dt` (VdP 0.35 vs 0.36; Lorenz 11.8 vs 11.3). So `g`
is DRIFT-LIMITED, not sampler-limited (memory `g-is-drift-limited`). The FP-inverse was the attempt to escape
the `1/dt` by fitting `f,g` as PDE coefficients instead -- DEAD (above). What remains is to keep the stable
data-driven increment `f` and REDUCE its differentiation error: the **Temporal FD-conv / higher-order dz/dt
target** and **finer dt** in "Ideas / future directions" below (central/higher-order stencil kills the O(dt)
forward-difference bias). Cheap to A/B on Lorenz; MEASURE `drift_l2_aln`, don't assume.

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
