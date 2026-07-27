# TODO

Cross-session task tracker for follow-up work that doesn't belong in the current PR.

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

## Ideas / future directions (not scheduled)

From the NKE and PINNs-Bayes reading (see `related_work.md`):

- **Strang / trapezoidal 2nd-order drift.** NKE frames Euler-Maruyama as 1st-order Lie-Trotter splitting and
  uses a trapezoidal "meet-me-halfway" advection loss (their Eq. 33) for 2nd-order accuracy. We *proved* the
  trapezoidal drift is invalid on the filter MEAN (a propagation, not an SDE integral) and the smoother mean
  over-attenuates -- so this is blocked until the smoother-mean-for-drift problem is solved. Adopt the
  splitting *framing* in the writeup; the concrete gain waits.
- **Continuous-time switching LDS via probabilistic jumps (own paper).** Represent the jump kernel as a
  categorical-over-latent-states generator (or, equivalently, a *probabilistic* operator split -- mix over
  which sub-generator acts), giving discrete regimes inside the Zakai/KFE machinery. Strategically relevant
  for neural-activity latents (up/down states, regime switches). NOT low-hanging: filtering can't threshold
  hidden jumps (NKE's mechanism needs observed increments), and Zakai-with-jumps is a PIDE-SPDE. Separate paper.
