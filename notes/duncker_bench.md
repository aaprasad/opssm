# Duncker / Linderman benchmark suite — results & handoff

Three benchmark systems from Duncker et al. 2019 (GPSLDS; same lineage as SING / Helmholtz-SDE), added to
opssm to set up a comparison against that baseline family. Run in a **well-covered regime** (long trajectories
for coverage) with DENSE observation and the `det_mid` drift target (main default). Configs:
`configs/experiment/duncker_{dw,vdp,lorenz}.yaml` + `configs/data/duncker_*.yaml`.

## Settings

| system | latent drift | σ (Σ=I) | obs N / noise-std | trials (+val) | T | obs dt | num_steps | coverage `trials·T` | x0 |
|---|---|---|---|---|---|---|---|---|---|
| double-well | `a·x(1−x²)`, **a=1** | 1.0 | 15 / 0.5 | 32 (+8) | 30 | 0.1 | 300 | 960 | N(0,1) |
| VdP (`vanderpol_duncker`) | `f1=τμ(x1−⅓x1³−x2), f2=τx1/μ`, τ=10 μ=2 | 1.0 | 20 / 1.5 | 32 (+8) | 30 | 0.1 | 300 | 960 | U[−3,3]² |
| Lorenz | (10, 28, 8/3) | 0.15 | 20 / 0.01 | 32 (+8) | 30 | 0.1 | 300 | 960 | N(0,I) |

## Deviations from Duncker (all intentional or forced)

- **Dense obs, not sparse random.** Duncker observes at ~20 random uneven time-points/trial; we observe every
  step (`mask` all-ones). The sparse/irregular-obs regime is a deferred ablation (needs a random mask + operator
  variable-dt support). Our oracle diagnostics also used dense TRUE latents, so they bypass the sparse-inference
  problem entirely — see `notes/TODO.md`.
- **Double-well uses a=1, NOT Duncker's a=4.** Duncker's `4x(1−x²)` is unresolvable at obs dt=0.1 (see below).
- **Lorenz trials 1024 → 32.** opssm's E-step tensors are `(T,B,K,d)`; B=1024 blows memory. 32 long trajectories
  give coverage 960, comfortably off the drift floor.
- **VdP / Lorenz obs models are best-guessed.** VdP N=20 / noise-var 2.25 from Duncker §5.2; Lorenz N=20 /
  noise-std 0.01 from the Li et al. regime. Not verbatim from a single spec.
- **Small held-out val split** (8 trajectories) for opssm's Procrustes-gauged metrics.

## Results (14000 steps, det_mid, single seed)

| system | lat_rel | drift_rel | g_rel | c_cos | kl |
|---|---|---|---|---|---|
| double-well (a=1) | 0.122 | 0.406 | 0.957 | 1.000 | 0.281 |
| VdP (τ=10) | 0.181 | 0.454 | 1.56 | 0.972 | — |
| Lorenz | 0.010 | **0.176** | 36.9† | 0.949 | — |

- **Lorenz is the star:** near-perfect latent (`lat_rel 0.010`, tiny obs noise) and the best drift we've gotten
  on it (`0.176` vs the home-benchmark `0.211`). The fast VdP (τ=10) also tracks well at dt=0.1 (`lat_rel 0.18`).
- **DW a=1 is healthy but noisier than home `em_highd` (det_mid 0.207)** — the Duncker framing has σ=1 (vs 0.6)
  and only 32 trials (vs 224), so `drift_rel 0.406` is expected, not a regression.
- **† Lorenz g_rel is huge only because true σ=0.15 is tiny** (`g_aln≈5.5`). `g` is entirely drift-residual-
  dominated there (`g_est² ≈ drift_rmse²·dt`; the `g` is drift-limited, see `g-is-drift-limited` memory), not a
  fixable estimator issue — with σ that small there's nothing for any estimator to recover.

## The a=4 finding (why the double-well uses a=1)

Duncker's actual double-well drift is `4x(1−x²)` (a=4). At obs dt=0.1 this **breaks the drift M-step
end-to-end** while the filter itself stays fine (`lat_rel 0.13`, `kl 0.28`):

| Duncker-DW config, det_mid | drift_rel |
|---|---|
| a=1 (shallow) | 0.318 (8k) / 0.406 (14k) ✅ |
| a=4 (Duncker's) | 2.56 💥 |
| a=4, forward (not det_mid) | 1.27 💥 |

Same config, only the well steepness changes. Mechanism: a=4 → `|f|` up to 7.5 → `|f|·dt = 0.75` at dt=0.1 — a
**large-step regime** (like Lorenz), but in **1-D with three fixed points** where `f` changes sign. There the
O(dt) increment error is *colinear* with `f` and can flip the target's sign; the deep wells also pile the filter
means at the well bottoms (`f≈0`) and starve the fast transition regions — so the drift M-step learns an
inflated/wrong `f` (drift_rel >1), which then poisons the FP density. det_mid's bounded shift (`(dt/2)|f|=0.375`)
hops across the fixed points, making it slightly worse than forward. **Not a det_mid bug** — both targets fail.

Reconciling with the oracle: the drift-fit oracle (TRUE latents, forward, a=4) gives `drift_rel ≈ 0.5` *flat
across dt* — so the increment TARGET is fine; the end-to-end failure is in the FILTER/M-step pipeline at this
resolution. A quick finer-dt test (dt=0.025) **made it worse** (drift 2.49, kl→38.5), so resolution alone
doesn't obviously rescue it — needs a careful sweep. Deferred (see `notes/TODO.md`).

## Deferred follow-ups (see `notes/TODO.md`)

1. **Obs-space gauge-invariant metrics** — the real head-to-head vs Duncker/Helmholtz-SDE (our latent-gauge
   `drift_rel`/`lat_rel` aren't directly comparable to their published numbers).
2. **Sparse/irregular-obs ablation** — Duncker's actual sampling (random mask + variable dt).
3. **a=4 double-well at finer obs dt** — a careful dt sweep with stable `n_sub`; whether resolution alone fixes it.

## Repro

`~/venvs/neuraloperator/bin/python scripts/train.py experiment=duncker_{dw,vdp,lorenz} trainer.max_steps=14000
train_dir=dump/...` (one GPU at a time; det_mid is the default drift_target).
