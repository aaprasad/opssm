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
