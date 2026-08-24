"""JAX EM training loop (em_highd, d=1) -- the functional replacement for filter_module's Lightning loop.

Flow (mirrors ZakaiFilterModule): subspace-ID init (C, A) -> warm-start drift to A -> bootstrap M-step ->
per step { E-step (mesh-free Zakai PINN grad, exp-decay LR) ; M-step every m_every } -> validate. The
mutable EM state (op/op_opt_state, drift_net/dr_state, g_cur, C_cur, d_cur) is threaded explicitly (JAX is
functional); the E-step is one jitted call, the M-step is eager (amortized to ~0 at m_every=2000).

Validation ports the d=1 metrics: kl vs the exact filter, on-data drift_l2, c_cos, and the Procrustes
gauge-aligned kl_aln / lat_rel / drift_rel / g_rel (see filter_module._gauge_aligned).
"""
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from opssm.models.jax.operator import OperatorFilter
from opssm.models.jax.dynamics import DriftNet
from opssm.models.jax.obs import zhat_from_obs, make_decode
from opssm.models.jax.losses import pinn_zakai_loss, sample_collocation, kl_target_pred
from opssm.models.jax import mstep as M
from opssm.models.jax.init import subspace_id, warmstart_drift


def _log_prior(z):
    return -0.5 * (z ** 2).sum(-1)


def _interp1d(vals, grid, query):                                  # vals (...,Nz) on ascending grid, query (Nq,)
    idx = jnp.clip(jnp.searchsorted(grid, query), 1, grid.shape[0] - 1)
    x0, x1 = grid[idx - 1], grid[idx]
    w = jnp.clip((query - x0) / jnp.maximum(x1 - x0, 1e-12), 0.0, 1.0)
    return vals[..., idx - 1] * (1 - w) + vals[..., idx] * w


DEFAULTS = dict(  # em_highd resolved hparams (configs/model/operator.yaml + experiment/em_highd.yaml)
    data_size=10, latent_dim=1, gru_hidden=64, ctx_dim=64, p=64, drift_hidden=64,
    lr=2e-3, drift_lr=2e-3, sched_gamma=0.9998, n_scoll=4, n_tcoll=24, n_colloc=128,
    near_std=0.3, broad_std=1.6, warmup=0, m_every=2000, m_inner=400, reg_lambda=3e-4,
    g_init=1.0, w_res=0.4, res_mode="rel", learn_obs=True, learn_g=True, c_stable_tol=0.05,
    n_mean=256, mean_method="mala", mala_chains=64, mala_steps=30, mala_rng="stochastic",
    drift_target="det_mid",
)


def _mala_cfg(hp):
    return dict(n_chains=hp["mala_chains"], n_steps=hp["mala_steps"], rng=hp["mala_rng"])


def _run_mstep(op, xs, mask, drift_net, dr_opt, dr_state, C, d, g, hp, key, bootstrap=False):
    info, drift_net, dr_state = M.mstep(
        op, xs, mask, hp["dt"], drift_net, dr_opt, dr_state, key,
        learn_g=hp["learn_g"], reg_lambda=hp["reg_lambda"], m_inner=hp["m_inner"],
        learn_obs=hp["learn_obs"], c_stable_tol=hp["c_stable_tol"], C_cur=C, d_cur=d,
        n_mean=hp["n_mean"], near_std=hp["near_std"], broad_std=hp["broad_std"],
        mean_method=hp["mean_method"], mala=_mala_cfg(hp), drift_target=hp["drift_target"],
        bootstrap=bootstrap)
    g = info["g_cur"] if info["g_cur"] is not None else g
    if hp["learn_obs"]:
        C, d = info["C_cur"], info["d_cur"]
    return drift_net, dr_state, C, d, g, info


def make_estep(xs, mask, s_coll, optim, hp):
    """Factory -> one jitted E-step. drift_net/g/C/d passed as dynamic args (no recompile per M-step)."""
    T = xs.shape[0]

    @eqx.filter_jit
    def estep(op, opt_state, drift_net, g, C, d, key):
        kc, kt = jax.random.split(key)
        center = zhat_from_obs(xs, C, d)
        decode = make_decode(C, d)
        z_col, log_q = sample_collocation(kc, xs, mask, hp["n_colloc"], hp["near_std"], hp["broad_std"], center)
        ti = jax.random.permutation(kt, T)[:hp["n_tcoll"]]

        def loss_fn(op):
            res, jump, ic, nll = pinn_zakai_loss(op, xs, mask, z_col, log_q, s_coll, drift_net.drift, g,
                                                 _log_prior, hp["noise_std"], hp["dt"], ti=ti,
                                                 res_mode=hp["res_mode"], decode=decode)
            return hp["w_res"] * res + jump + ic, (res, jump, ic)

        (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(op)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(op, eqx.is_inexact_array))
        return eqx.apply_updates(op, updates), opt_state, aux

    return estep


def gauge_aligned_1d(log_pi, filt, m_op, z_true, drift_net, g_cur, sigma, a, z_grid):
    """d=1 Procrustes gauge-aligned metrics (kl_aln, lat_rel, drift_rel, g_rel, s_fit). m_op (T,B),
    z_true (T,B,1), log_pi/filt (T,B,Nz)."""
    Mm = m_op.reshape(-1, 1)
    Z = z_true.reshape(-1, 1)
    Mh = jnp.concatenate([Mm, jnp.ones_like(Mm[:, :1])], -1)       # (N,2)
    sol = jnp.linalg.lstsq(Mh, Z)[0]                              # (2,1): Z ~ [M,1]@sol
    A, b = sol[:1], sol[1]                                        # (1,1),(1,)
    z_al = Mm @ A + b
    f_al = drift_net.net(Mm) @ A                                 # (N,1) drift maps under A
    f_true = a * (z_al - z_al ** 3)                             # double-well true drift
    on = (jnp.abs(z_al[:, 0]) <= 1.5).astype(z_al.dtype)
    s = A.reshape(-1)[0]
    gscale = jnp.abs(s)
    e_lat = ((z_al - Z) ** 2).sum(-1)
    e_drf = ((f_al - f_true) ** 2).sum(-1)
    z_scale = jnp.maximum(jnp.sqrt(((Z - Z.mean(0)) ** 2).sum(-1).mean()), 1e-8)
    f_scale = jnp.maximum(jnp.sqrt((f_true[:, 0] ** 2 * on).sum() / jnp.maximum(on.sum(), 1)), 1e-8)
    out = {
        "s_fit": float(s),
        "lat_rel": float(jnp.sqrt(e_lat.mean()) / z_scale),
        "drift_rel": float(jnp.sqrt((e_drf * on).sum() / jnp.maximum(on.sum(), 1)) / f_scale),
        "g_aln": float(gscale * g_cur),
        "g_rel": float(gscale * g_cur / max(sigma, 1e-8)),
    }
    b0, sv = float(b.reshape(-1)[0]), float(s)
    pi_al = _interp1d(jnp.exp(log_pi), z_grid, (z_grid - b0) / sv)
    pi_al = jnp.maximum(pi_al, 0.0) / abs(sv)
    pi_al = pi_al / jnp.maximum(pi_al.sum(-1, keepdims=True), 1e-12)
    out["kl_aln"] = float(kl_target_pred(filt, jnp.log(jnp.maximum(pi_al, 1e-20))))
    return out


def validate(op, drift_net, g_cur, C_cur, d_cur, refs, hp):
    """d=1 validation metrics vs the exact filter + gauge-aligned latent/drift/g."""
    xv, mv, filt = refs["x_val"], refs["mask_val"], refs["filt_val"]
    zg, z_true, C_true = refs["z_grid"], refs["z_val_true"], refs["C_true"]
    a, sigma = refs["a"], refs["sigma"]
    log_pi = op.log_posterior(xv, mv, zg)                         # (T,B,Nz)
    logs = {"g": float(g_cur), "kl": float(kl_target_pred(filt, log_pi))}
    m_op = (jnp.exp(log_pi) * zg).sum(-1)                         # (T,B) grid mean
    lo, hi = jnp.quantile(m_op, 0.01), jnp.quantile(m_op, 0.99)
    on = (zg >= lo) & (zg <= hi)
    fd = drift_net.net(zg[:, None])[:, 0]
    f_true_grid = a * (zg - zg ** 3)
    logs["drift_l2"] = float(jnp.sqrt((((fd - f_true_grid) ** 2) * on).sum() / jnp.maximum(on.sum(), 1)))
    Cn = C_true / jnp.linalg.norm(C_true, axis=0, keepdims=True)
    logs["c_cos"] = float(jnp.minimum(jnp.linalg.svd(C_cur.T @ Cn, compute_uv=False), 1.0).mean())
    logs.update(gauge_aligned_1d(log_pi, filt, m_op, z_true, drift_net, g_cur, sigma, a, zg))
    return logs


def train(refs, hp, n_steps, key, val_every=2000, log_fn=print):
    """Run em_highd EM training in JAX. refs: bridged arrays (jnp). Returns final state + metric history."""
    d = hp["latent_dim"]
    xs, mask = refs["x_train"], refs["mask_train"]
    s_coll = jnp.linspace(0.0, 1.0, hp["n_scoll"])
    key, ko, kd, kw, kb = jax.random.split(key, 5)

    # --- init: operator, subspace-ID emission + dynamics, warm-start drift ---
    op = OperatorFilter(hp["data_size"], hp["gru_hidden"], hp["ctx_dim"], hp["p"], latent_dim=d, key=ko)
    C_cur, A_init = subspace_id(refs["full_obs"], d, hp["dt"])
    d_cur = refs["full_obs"].reshape(-1, hp["data_size"]).mean(0)
    drift_net = DriftNet(hp["drift_hidden"], latent_dim=d, key=kd)
    drift_net, wmse = warmstart_drift(drift_net, A_init, kw)
    g_cur = hp["g_init"]

    # --- optimizers ---
    sched = optax.exponential_decay(hp["lr"], transition_steps=1, decay_rate=hp["sched_gamma"])
    optim = optax.adam(sched)
    op_opt_state = optim.init(eqx.filter(op, eqx.is_inexact_array))
    dr_opt = optax.adam(hp["drift_lr"])
    dr_state = dr_opt.init(eqx.filter(drift_net.net, eqx.is_inexact_array))

    # --- bootstrap M-step (from the init projection, before any E-step) ---
    drift_net, dr_state, C_cur, d_cur, g_cur, info = _run_mstep(
        op, xs, mask, drift_net, dr_opt, dr_state, C_cur, d_cur, g_cur, hp, kb, bootstrap=True)
    log_fn(f"[init] warmstart_mse={wmse:.4f} bootstrap cstab={info['cstab']:.4f} g={g_cur:.4f}")

    estep = make_estep(xs, mask, s_coll, optim, hp)
    history = []
    m0 = validate(op, drift_net, g_cur, jnp.asarray(C_cur), d_cur, refs, hp)
    log_fn(f"[step 0] " + " ".join(f"{k}={v:.4f}" for k, v in m0.items()))
    history.append((0, m0))

    for step in range(1, n_steps + 1):
        key, sk = jax.random.split(key)
        op, op_opt_state, aux = estep(op, op_opt_state, drift_net, jnp.asarray(g_cur),
                                      jnp.asarray(C_cur), d_cur, sk)
        if step > hp["warmup"] and step % hp["m_every"] == 0:
            key, mk = jax.random.split(key)
            drift_net, dr_state, C_cur, d_cur, g_cur, info = _run_mstep(
                op, xs, mask, drift_net, dr_opt, dr_state, C_cur, d_cur, g_cur, hp, mk)
        if step % val_every == 0 or step == n_steps:
            m = validate(op, drift_net, g_cur, jnp.asarray(C_cur), d_cur, refs, hp)
            res, jump, ic = (float(a) for a in aux)
            log_fn(f"[step {step}] res={res:.3f} jump={jump:.3f} ic={ic:.3f} | "
                   + " ".join(f"{k}={v:.4f}" for k, v in m.items()))
            history.append((step, m))
    return dict(op=op, drift_net=drift_net, g_cur=g_cur, C_cur=C_cur, d_cur=d_cur), history


def load_refs(npz_path):
    """Load the bridged em_highd arrays as jnp (float32) + scalars."""
    D = np.load(npz_path)
    arr = lambda k: jnp.asarray(D[k])
    refs = {k: arr(k) for k in ("x_train", "mask_train", "x_val", "mask_val", "filt_val",
                                "z_grid", "full_obs", "z_val_true", "C_true", "d_true")}
    refs["a"] = float(D["a"]); refs["sigma"] = float(D["sigma"])
    return refs, dict(dt=float(D["dt"]), noise_std=float(D["noise_std_eff"]))
