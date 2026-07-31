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

## Running now (workstation, task bycvcugid)

d=1 A/B on `em_highd` (exact-filter KL as ground truth), 14000 steps each, chained one-at-a-time:
`em` (w_em=1,w_inv=0) -> `blend` (1,1) -> `inv` (0,1). EM baseline was ~8.75 it/s.
Logs: `dump/fpinv_highd_{em,blend,inv}.log` (+ `dump/fpinv_highd_chain.log`). These logs are LOCAL to the
workstation (dump/ is gitignored) -- read them there when the chain finishes. Quick read:
`for f in dump/fpinv_highd_*.log; do echo $f; tr '\r' '\n' <$f | grep -oE "kl=[0-9.]+|drift_rel=[0-9.]+|g_rel=[0-9.]+" | tail -3; done`

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
