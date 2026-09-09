# Why a scalar `g` is misspecified at latent d>1 (even though the DGP diffusion IS scalar)

Written in answer to a fair objection: *our synthetic systems use an isotropic scalar diffusion `sigma`, so
why learn a full covariance? It should either be wrong, or converge to a scalar.* The answer is that
isotropy is a statement about the TRUE latent frame, and the model does not work in that frame.

## The mechanism

The generator uses a RAW GAUSSIAN sensor (`data/systems.linear_sensor`, `data/synth_refs._linear_sensor`):
`y = C_true z_true + d + eps`, `C_true (D,d)` with i.i.d. normal entries. The M-step, however, constrains its
sensor to the STIEFEL manifold (`fit_obs_map_stiefel`: `C = U V^T`, orthonormal columns). Matching the same
observations forces the model latent to be a LINEAR IMAGE of the true one:

    z_model = M z_true ,      M = C_model^T C_true

Write `C_true = U S V^T`. Once the sensor is recovered (`c_cos -> 1`) we have `span(C_model) = span(U)`, so
`C_model = U Q` for some orthogonal `Q`, and

    M M^T = Q^T S^2 Q  ,      Sigma_model = sigma^2 M M^T = sigma^2 Q^T S^2 Q.

So the model's own-frame diffusion is:
- **anisotropic**, with eigenvalue ratio equal to the SQUARED singular-value ratio of `C_true`;
- **not diagonal**, because `Q` is whatever rotation the Procrustes step lands on -- nothing in the objective
  drives it to diagonalize `Sigma`. (Over random `Q`, the off-diagonal mass is a median ~0.39 of the diagonal.)

**Consequence: orthonormal `C` together with a scalar `g` cannot represent the data-generating process at
d>1.** A scalar gives `Sigma = g^2 I`, which equals `sigma^2 M M^T` only if `C_true` is conformal (all
singular values equal) -- generically false. This is a misspecification in the CURRENT default, not a
modelling luxury in the new one. At d=1 `M` is a scalar, `M M^T ∝ I` trivially, so `em_highd` is unaffected --
which is exactly why the covariance change is a no-op there and only `learn_obs_noise` moves that benchmark.

## Oracle confirmation (no filter, no EM, no dt-bias)

TRUE Lorenz latents, the exactly-transported true drift `f_model(z) = M f_true(M^-1 z)`, fine `dt = 0.001` so
the Euler residual is essentially pure noise, `sigma = 3.0`, `D=10, d=3`:

| check | result |
|---|---|
| `Sigma` from increments vs predicted `sigma^2 M M^T` | max abs error 0.17 on entries up to 179 (~0.1%) |
| anisotropy (std) of recovered `Sigma` | 2.438 |
| singular-value ratio of `C_true` | 2.440 |
| `|max off-diag| / |max diag|` | 0.177 |
| best possible isotropic fit, `min_g ||g^2 I - Sigma||_F / ||Sigma||_F` | 0.528 |

and the reported diffusion, against the true `sigma = 3.0`:

| estimator | `g_aln` | `g_rel` |
|---|---|---|
| scalar `g` = `sqrt(tr(Sigma)/d)` (old default) | 3.382 | **1.127** |
| `det(Sigma)^(1/2d)` (new default) | 3.000 | **1.000** |

The det-based scalar is exact because it is GAUGE-COVARIANT: the Procrustes gauge is `A = M^-1`, so
`gscale = |det A|^(1/d) = |det M|^(-1/d)` and `det(Sigma_model)^(1/2d) = sigma |det M|^(1/d)` -- the two
`det M` factors cancel identically, leaving `sigma`.

## Refinement of [[g-is-drift-limited]] (does NOT contradict it)

`g_est^2 = g_true^2 + drift_rmse^2 dt` still holds in the MODEL frame; what that note calls `g_true^2` is
really `tr(sigma^2 M M^T)/d`, the gauge-transformed diffusion, not `sigma^2`. So the `g_rel` over-read at d>1
has (at least) TWO sources:

1. **drift residual** -- additive in `Sigma`, the documented and dominant term on Lorenz;
2. **estimator/gauge mismatch** -- reporting `sqrt(tr/d)` (an ARITHMETIC mean of eigenvalues) instead of
   `det^(1/2d)` (a GEOMETRIC mean) for an anisotropic `Sigma`. This is a pure MULTIPLICATIVE bias
   `sqrt(arithmetic/geometric mean)`, present even with a perfect drift.

Source 2 is removed for free by the change, independently of the drift. For the actual `em_lorenz` sensor
(numpy `default_rng(seed+1)`, singular values 3.70 / 2.50 / 0.84) it alone predicts a **~32% over-read**, which
is a real slice of that benchmark's long-standing `g_rel`.

## The alternative gauge fixing (where the objection IS right)

There are two consistent ways to fix the latent gauge, and the current code accidentally uses NEITHER
completely -- it constrains BOTH `C` (Stiefel) and `Sigma` (scalar), which over-determines the model:

- **ours**: orthonormal `C`, FREE `Sigma`. The latent frame is pinned by the sensor; the diffusion absorbs
  the gauge. Better for real data (Kato), where nothing suggests the true diffusion is isotropic anyway.
- **Duncker / gpSLDS / Helmholtz-SDE**: FREE `C`, FIXED `Sigma` (they set `Sigma = I`). The diffusion is
  pinned; the sensor absorbs the gauge. Here a single scalar IS right and a full covariance would be wasted
  -- and this is the fairer setup for reproducing those baselines' numbers.

Not built. If a head-to-head against those baselines needs it, add it as a switch rather than replacing the
above; see `notes/TODO.md`.
