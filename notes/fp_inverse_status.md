# FP-inverse drift/diffusion — status & handoff

Branch: `fp-inverse` (off `main` @ 84655fb).

## What's implemented

The M-step fits `f` AND `g` **jointly** from a **weighted** loss (`mstep.fit_dynamics`), replacing the
drift-limited mean-increment regression (`fit_drift`):

    loss = w_em * ||f(z_t) - dz||^2                                           (EM increment; f)
         + w_inv * (ds_ell/dt + div f + f.grad ell - g^2 * 1/2(|grad ell|^2 + Laplacian ell))^2   (FP residual; f AND g)
         + reg_lambda * ||f||^2

- The FP residual is **bilinear in (f, g)**, so `f` and `g` descend **together** each inner step (not alternated).
- `g` is the learned **factor** (`g^2` in the term, stays >=0); warm-started each M-step from the classic
  increment estimate then **FP-refined** (init only, not an anchor -- the increment is the drift-limited one).
- Config knobs (`configs/model/operator.yaml`): `w_em`, `w_inv`, `g_lr`. **`w_inv=0` = classic EM, unchanged default.**
- `_fp_residual_terms` computes the operator density-derivatives once per M-step (detached, mirrors the FP block
  of `pinn_zakai_loss`). `mstep.py` commits: b585791 (joint fit) + 5eef11d (increment-g init).

## d=1 RESULT (2026-07-31): NEGATIVE -- the naive joint FP-inverse DIVERGES

d=1 A/B on `em_highd` (exact-filter KL ground truth), 14000 steps, `mean_method=mala`:

    config                     kl      drift_rel   g_rel    lat_rel
    em    (w_em=1, w_inv=0)    0.355   0.257       0.914    0.116     <- EM baseline (good, stable)
    blend (w_em=1, w_inv=1)    0.767   1.440       0.507    0.129     <- WORSE, and DIVERGING
    inv   (w_em=0, w_inv=1)    0.756   1.390       0.508    0.129     <- WORSE, and DIVERGING

Trajectories (per validation) show it DRIFTS AWAY over training, not a plateau:
- `kl`:        blend/inv 0.36 -> 0.50 -> 0.62 -> 0.69 -> 0.77   (EM: 0.36 -> 0.355 stable)
- `drift_rel`: blend/inv 1.0 -> 1.15 -> 1.24 -> 1.36 -> 1.44    (EM: 1.0 -> 0.26 converges)
- `g_rel`:     blend/inv 0.65 -> 0.60 -> 0.56 -> 0.53 -> 0.51   (g SHRINKS -- under-reads)
Even the BLEND (increment anchor on f) diverges, so anchoring f alone is not enough.

**Diagnosis -- co-adaptation / g-runaway.** The FP residual is now minimized by BOTH the E-step (w.r.t. `ell`)
AND the M-step (w.r.t. `f,g`). It's under-determined, so with only a weak data anchor (jump/IC vs `w_res=0.2`)
they co-adapt to a DEGENERATE self-consistent solution: `g` shrinks -> the density sharpens -> the residual's
`g`-fit shrinks `g` further -> `f` compensates toward a wrong drift -> `ell` drifts off the true filter (`kl`
climbs). The E-step's `w_res` already lets the residual bend `ell`; the M-step residual-fit adds pressure the
anchor can't hold. This is exactly the "keep `ell` data-driven" caveat -- it wasn't held.

## UPDATE: the paper fits g from the FPE too -- the real issue is the WIDTH anchor

Re-reading Liu et al: their gh-step fits BOTH drift AND diffusion from the FPE, and it's stable -- because they
OBSERVE the density, so its WIDTH (which g controls) is data-pinned (P^0). We observe only point locations, so
the jump/IC pin the posterior LOCATION but NOT its WIDTH -> g-from-FP is under-determined -> the runaway. This
is a limit of our OBSERVATION MODEL (points vs densities), not a wrong choice to fit g from the FP.

**Fix implemented -- OPTION 1 (symmetric g anchor), config `w_g`:** give g the same two-term treatment as f --
a DATA anchor (`w_g` * (g - g_increment)^2, the increment-g being our stand-in for the paper's observed width)
PLUS the FP residual (`w_inv`). The anchor stops the runaway; the residual refines. `w_g` has its OWN weight
because the anchor and the residual are at very different scales (rough analysis: w_g ~ 10s to balance w_inv).
Default `w_g=0` (off). TODO test: `experiment=em_highd model.mean_method=mala model.w_em=1 model.w_inv=1
model.w_g={1,10,50}` -- find the w_g that tames g_rel without over-pinning to the (biased) increment.

## RESULTS (2026-08-11): the anchors DON'T fix it -- divergence is robust

Two fixes tested on em_highd d=1, BOTH NEGATIVE:

    config                              kl      drift_rel   g_rel     vs EM (0.355 / 0.257 / 0.914)
    w_res=0.05 (strong ell anchor)      0.99    1.46        0.49      still DIVERGES
    w_g=1   (w_em=1, w_inv=1)           0.755   1.56        0.505     still DIVERGES
    w_g=10                              0.754   1.48        0.507     still DIVERGES
    w_g=50                              0.748   1.39        0.510     still DIVERGES

`g_rel` trajectory is **IDENTICAL** across w_g=1/10/50: `4.04 -> 2.01 -> 0.654 -> 0.600 -> 0.561 -> 0.532 ->
0.505`. `drift_rel` climbs `1.0 -> 1.4` for every setting. The w_g anchor is essentially inert.

**Reads:**
1. w_g inert across a 50x range => it isn't controlling `g`. Likely cause: `g_opt` is Adam (per-parameter
   scale-normalized), so scaling the anchor weight barely changes `g`'s step -- a hard clamp / SGD-for-`g`
   would test it. But secondary (see 3).
2. Neither the `ell` anchor (`w_res`) NOR the `g` anchor (`w_g`) stops the divergence => not a simple
   anchor-strength problem. The Tikhonov / regularized-inverse framing was right in spirit but doesn't bite here.
3. **The real driver is the DRIFT `f`, not `g`.** `drift_rel` climbs `1.0 -> 1.4` regardless of the anchors,
   EVEN WITH the increment anchor on `f` (`w_em=1`). So the `w_inv=1` FP-residual term **OVERPOWERS** the
   `w_em` increment anchor and drives `f` wrong; the E-step `ell` and M-step `f` co-adapt on the shared
   residual and the strong `w_inv` wins. `g` is a passenger.

**Bottom line:** the FP-inverse as a STRONG term (`w_inv ~ 1`) diverges robustly at d=1; anchoring `ell` or `g`
does not rescue it.

**Key untested knob -> WEAK `w_inv` (0.1-0.2):** the residual as a gentle regularizer ON TOP of the
increment-dominated EM, so it cannot overpower the `w_em` data anchor on `f`. The `winv02` run (`w_inv=0.2`)
was queued but KILLED before it finished -- RE-RUN it: `experiment=em_highd model.mean_method=mala model.w_em=1
model.w_inv=0.2`. If even a weak `w_inv` fails to beat EM, the FP-inverse for the drift is a dead end in this
setup, and the drift bottleneck stays open -> fall back to higher-order / finer-dt increment (notes/TODO.md).

## Things to try next (prioritized)

1. **Stronger data anchor:** lower `w_res` (e.g. 0.05) so the E-step keeps `ell` data-driven; the M-step then
   fits `f,g` to a TRUE filter, not a co-adapted one. (Config-only; the cheapest test of the diagnosis.)
2. **Weak inverse regularizer:** `w_inv` small (0.1-0.3) on top of EM (`w_em=1`) -- a gentle physics nudge that
   may not overpower the anchor. If this beats EM without diverging, it's the usable regime.
3. **Fix the g-runaway:** fit `f` from the residual but keep `g` from the INCREMENT (don't jointly fit `g`).
   Needs a small flag in `fit_dynamics`/`mstep` (fit `f` only via the FP term, `g` via `fit_diffusion`). Tests
   whether `g` is the driver.
4. Deeper: the residual-fit assumes `ell` is the TRUE data-driven filter; the E-step training `ell` to satisfy
   the SAME residual makes it circular. May need to fit `f,g` against `ell` derivatives that are anchored purely
   by jump/IC (or a smoother/held-out `ell`), not the residual-trained one.

## Local runs (workstation)

Logs: `dump/fpinv_highd_{em,blend,inv}.log` (dump/ is gitignored, so LOCAL to the workstation). Quick read:
`for f in dump/fpinv_highd_*.log; do echo $f; tr '\r' '\n' <$f | grep -oE "kl=[0-9.]+|drift_rel=[0-9.]+|g_rel=[0-9.]+" | tail -1; done`

## Baselines to beat

- **d=1 EM** (em_highd): kl 0.355, drift_rel 0.257, g_rel 0.914. d=1 is decent already -> it's the CORRECTNESS
  VALIDATION (does the inverse match EM's kl?), not the payoff.
- **d=3 EM** (em_lorenz): drift_rel 0.571, g_rel 3.76 -- **THE REAL TARGET** (drift-limited). This is where the
  inverse should help: `experiment=em_lorenz model.mean_method=mala model.w_inv=1.0` (blend: also `model.w_em=1.0`).

## Next steps

1. Read the d=1 A/B: blend/inv should ~match EM's kl (correctness) and ideally improve drift_rel / g_rel.
2. If d=1 holds -> **d=3 Lorenz** (the real test). Watch the **rotational-gauge** caveat: the FP residual leaves
   the divergence-free drift unpinned at d>=2, so the PURE inverse (w_em=0) may miss rotation -- the **blend**
   (w_em>0) keeps the increment for it. Compare pure-inverse vs blend at d=2/3.
3. Tune `w_em`/`w_inv`/`g_lr`; `m_inner` may need to be large enough for the scalar `g` to converge in the FP fit.

## Caveats / open questions

- **Cost:** the FP fit runs `drift_net.drift` over `T*B*n_colloc` points x `m_inner` iters per M-step (jvp for
  `div f`) -- amortized over `m_every=2000`, but watch it/s at d=3 (Laplacian is heavier). Subsample time or drop
  `n_colloc` if it's slow.
- **Keep `ell` data-driven:** the E-step trains `ell` to satisfy the residual with the CURRENT `f,g`, so the
  M-step residual-fit could be circular. The jump/IC (observation) terms anchor `ell` to data and break it; the
  `w_res` (0.2) vs jump/ic balance is the knob. If the inverse looks self-confirming, lower `w_res`.
- Rotational identifiability at d>=2 is the main scientific unknown (see step 2 + `notes/TODO.md` FP-inverse item).
