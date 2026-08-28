"""JAX EM training loop -- functional replacement for filter_module's Lightning loop. Any latent dim d.

Flow (mirrors ZakaiFilterModule): subspace-ID init (C, A) -> warm-start drift -> bootstrap M-step ->
per step { E-step (mesh-free Zakai PINN grad, exp-decay LR) ; M-step every m_every } -> validate. The
mutable EM state (op/op_opt_state, drift_net/dr_state, g_cur, C_cur, d_cur) is threaded explicitly.

Validation: d==1 uses the grid (kl vs exact filter, drift_l2, grid mean); d>1 uses the mesh-free MALA
filter mean (no grid). Both score the Procrustes gauge-aligned lat_rel / drift_rel / g_rel (+ kl_aln, d==1)
and c_cos / recon_r2. Figures via opssm.models.jax.viz_jax (d=1 posterior panels, d=2/3 phase-space).
"""
import time

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
from opssm.models.jax.init import subspace_id, warmstart_drift, linear_dynamics
from opssm.models.jax.obs import zhat_from_obs as _zhat
from opssm.models.jax.systems import make_drift


def _log_prior(z):
    return -0.5 * (z ** 2).sum(-1)


def _interp1d(vals, grid, query):                                  # vals (...,Nz) on ascending grid, query (Nq,)
    idx = jnp.clip(jnp.searchsorted(grid, query), 1, grid.shape[0] - 1)
    x0, x1 = grid[idx - 1], grid[idx]
    w = jnp.clip((query - x0) / jnp.maximum(x1 - x0, 1e-12), 0.0, 1.0)
    return vals[..., idx - 1] * (1 - w) + vals[..., idx] * w


DEFAULTS = dict(  # em_highd resolved hparams; the bridge overrides per-experiment (latent_dim, n_colloc, ...)
    system="doublewell", data_size=10, latent_dim=1, gru_hidden=64, ctx_dim=64, p=64, drift_hidden=64,
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


def gauge_aligned(m_op, z_true, drift_net, true_drift, g_cur, sigma, log_pi=None, filt=None, z_grid=None):
    """Procrustes gauge-aligned metrics (any d): z_true ~ A m_op + b, score in the aligned frame.
    lat_rel/drift_rel/g_rel/s_fit always; kl_aln only d==1 (grid oracle). m_op/z_true (T,B,d)."""
    d = z_true.shape[-1]
    Mm = m_op.reshape(-1, d)
    Z = z_true.reshape(-1, d)
    sol = jnp.linalg.lstsq(jnp.concatenate([Mm, jnp.ones_like(Mm[:, :1])], -1), Z)[0]   # (d+1,d)
    A, b = sol[:d], sol[d]
    z_al = Mm @ A + b
    f_al = drift_net.net(Mm) @ A                                  # drift maps under the linear part
    f_true = true_drift(z_al)
    on = (jnp.abs(z_al[:, 0]) <= 1.5 if d == 1 else jnp.ones(z_al.shape[0], bool)).astype(z_al.dtype)
    gscale = jnp.abs(jnp.linalg.det(A)) ** (1.0 / d)
    e_lat = ((z_al - Z) ** 2).sum(-1)
    e_drf = ((f_al - f_true) ** 2).sum(-1)
    z_scale = jnp.maximum(jnp.sqrt(((Z - Z.mean(0)) ** 2).sum(-1).mean()), 1e-8)
    f_scale = jnp.maximum(jnp.sqrt(((f_true ** 2).sum(-1) * on).sum() / jnp.maximum(on.sum(), 1)), 1e-8)
    out = {
        "s_fit": float(A.reshape(-1)[0]) if d == 1 else float(gscale),
        "lat_rel": float(jnp.sqrt(e_lat.mean()) / z_scale),
        "lat_rmse_aln": float(jnp.sqrt(e_lat.mean() / d)),
        "drift_rel": float(jnp.sqrt((e_drf * on).sum() / jnp.maximum(on.sum(), 1)) / f_scale),
        "g_aln": float(gscale * g_cur),
        "g_rel": float(gscale * g_cur / max(sigma, 1e-8)),
    }
    if d == 1 and log_pi is not None and filt is not None and z_grid is not None:
        s, b0 = float(A.reshape(-1)[0]), float(b.reshape(-1)[0])
        pi_al = _interp1d(jnp.exp(log_pi), z_grid, (z_grid - b0) / s)
        pi_al = jnp.maximum(pi_al, 0.0) / abs(s)
        pi_al = pi_al / jnp.maximum(pi_al.sum(-1, keepdims=True), 1e-12)
        out["kl_aln"] = float(kl_target_pred(filt, jnp.log(jnp.maximum(pi_al, 1e-20))))
    return out


def validate(op, drift_net, g_cur, C_cur, d_cur, refs, hp, key):
    """Validation metrics (any d). GT-free-safe (Kato): gauge/c_cos auto-skip when z_true/C_true absent,
    recon_r2 still logged. Returns (logs, m_op_full (T,B,d))."""
    d = hp["latent_dim"]
    xv, mv = refs["x_val"], refs["mask_val"]
    has_zt = refs.get("z_val_true") is not None
    has_ct = refs.get("C_true") is not None
    logs = {"g": float(g_cur)}
    log_pi = None
    if d == 1:
        zg = refs["z_grid"]
        log_pi = op.log_posterior(xv, mv, zg)
        if refs.get("filt_val") is not None:
            logs["kl"] = float(kl_target_pred(refs["filt_val"], log_pi))
        m_op = (jnp.exp(log_pi) * zg).sum(-1)                     # (T,B)
        lo, hi = jnp.quantile(m_op, 0.01), jnp.quantile(m_op, 0.99)
        on = (zg >= lo) & (zg <= hi)
        fd = drift_net.net(zg[:, None])[:, 0]
        ftrue_g = refs["true_drift"](zg[:, None])[:, 0]
        logs["drift_l2"] = float(jnp.sqrt((((fd - ftrue_g) ** 2) * on).sum() / jnp.maximum(on.sum(), 1)))
        m_op_full = m_op[..., None]                               # (T,B,1)
    else:
        center = zhat_from_obs(xv, C_cur, d_cur)
        m_op_full, _ = M.posterior_mean_mala(op, xv, mv, center, hp["broad_std"], key,
                                             n_chains=hp["mala_chains"], n_steps=hp["mala_steps"],
                                             rng=hp["mala_rng"])   # (T,B,d)
    if has_ct:
        Cn = refs["C_true"] / jnp.linalg.norm(refs["C_true"], axis=0, keepdims=True)
        logs["c_cos"] = float(jnp.minimum(jnp.linalg.svd(C_cur.T @ Cn, compute_uv=False), 1.0).mean())
    if hp["learn_obs"]:
        y_hat = m_op_full @ C_cur.T + d_cur
        xf = xv.reshape(-1, xv.shape[-1]); yf = y_hat.reshape(-1, xv.shape[-1])
        ss_res = ((xf - yf) ** 2).sum()
        ss_tot = jnp.maximum(((xf - xf.mean(0)) ** 2).sum(), 1e-8)
        logs["recon_r2"] = float(1.0 - ss_res / ss_tot)
    if has_zt:                                                    # GT-only Procrustes metrics
        logs.update(gauge_aligned(m_op_full, refs["z_val_true"], drift_net, refs["true_drift"],
                                  g_cur, refs["sigma"], log_pi, refs.get("filt_val"), refs.get("z_grid")))
    return logs, m_op_full


def train(refs, hp, n_steps, key, val_every=2000, log_fn=print, fig_dir=None, timing=False):
    """Run EM training in JAX (any d). refs: bridged arrays (jnp) + 'true_drift'. Returns (state, history).
    timing=True blocks each phase (block_until_ready) + records per-phase wall-clock into state['timing']."""
    t_train0 = time.perf_counter()
    tm = {"estep": [], "mstep": [], "val": [], "fig": []}

    def _block(x):
        if timing:
            jax.block_until_ready(x)
        return x

    d = hp["latent_dim"]
    xs, mask = refs["x_train"], refs["mask_train"]
    D_obs = hp["data_size"]
    s_coll = jnp.linspace(0.0, 1.0, hp["n_scoll"])
    key, ko, kd, kw, kb, kc = jax.random.split(key, 6)

    op = OperatorFilter(D_obs, hp["gru_hidden"], hp["ctx_dim"], hp["p"], latent_dim=d, key=ko)
    full_obs = refs["full_obs"]
    ybar = full_obs.reshape(-1, D_obs).mean(0)
    d_cur = ybar
    # obs-model init: subspace-ID (C + dynamics A) | pca (top-d PCs) | random Stiefel (matches filter_module.setup)
    method = hp.get("init_method") or ("pca" if hp.get("pca_init") else "random")
    A_init = None
    if method == "subspace":
        C_cur, A_init = subspace_id(full_obs, d, hp["dt"])
    elif method == "pca":
        _, _, Vt = jnp.linalg.svd(full_obs.reshape(-1, D_obs) - ybar, full_matrices=False)
        C_cur = Vt[:d].T
    else:                                                          # random Stiefel
        C_cur = jnp.linalg.qr(jax.random.normal(kc, (D_obs, d)))[0]
    drift_net = DriftNet(hp["drift_hidden"], latent_dim=d, key=kd)
    wmse = float("nan")
    if hp.get("init_dynamics", True) and method in ("pca", "subspace"):   # warm-start drift to the init's linear A
        A = A_init if A_init is not None else linear_dynamics(_zhat(full_obs, C_cur, d_cur), hp["dt"])
        drift_net, wmse = warmstart_drift(drift_net, A, kw)
    g_cur = hp["g_init"]

    sched = optax.exponential_decay(hp["lr"], transition_steps=1, decay_rate=hp["sched_gamma"])
    optim = optax.adam(sched)
    op_opt_state = optim.init(eqx.filter(op, eqx.is_inexact_array))
    dr_opt = optax.adam(hp["drift_lr"])
    dr_state = dr_opt.init(eqx.filter(drift_net.net, eqx.is_inexact_array))

    cstab0 = float("nan")
    if hp.get("bootstrap_mstep", True):                           # one init-time M-step from the init projection
        drift_net, dr_state, C_cur, d_cur, g_cur, info = _run_mstep(
            op, xs, mask, drift_net, dr_opt, dr_state, C_cur, d_cur, g_cur, hp, kb, bootstrap=True)
        cstab0 = info["cstab"]
    _block(eqx.filter(drift_net, eqx.is_inexact_array))
    init_s = time.perf_counter() - t_train0                       # init + warmstart + bootstrap (+ their compiles)
    log_fn(f"[init] d={d} method={method} warmstart_mse={wmse:.4f} bootstrap cstab={cstab0:.4f} g={g_cur:.4f}")

    _plot = None
    if fig_dir is not None:
        from opssm.models.jax.viz_jax import plot_validation as _plot

    def _validate_and_plot(step):
        nonlocal key
        key, vk = jax.random.split(key)
        tv = time.perf_counter()
        m, m_op = validate(op, drift_net, g_cur, jnp.asarray(C_cur), d_cur, refs, hp, vk)   # returns floats -> synced
        tm["val"].append(time.perf_counter() - tv)
        if _plot is not None:
            tf = time.perf_counter()
            _plot(op, drift_net, g_cur, jnp.asarray(C_cur), d_cur, refs, hp, step, fig_dir, m, m_op)
            tm["fig"].append(time.perf_counter() - tf)
        return m

    estep = make_estep(xs, mask, s_coll, optim, hp)
    history = []
    m0 = _validate_and_plot(0)
    log_fn(f"[step 0] " + " ".join(f"{k}={v:.4f}" for k, v in m0.items()))
    history.append((0, m0))

    for step in range(1, n_steps + 1):
        key, sk = jax.random.split(key)
        te = time.perf_counter()
        op, op_opt_state, aux = estep(op, op_opt_state, drift_net, jnp.asarray(g_cur),
                                      jnp.asarray(C_cur), d_cur, sk)
        _block(aux)
        tm["estep"].append(time.perf_counter() - te)
        if step > hp["warmup"] and step % hp["m_every"] == 0:
            key, mk = jax.random.split(key)
            tms = time.perf_counter()
            drift_net, dr_state, C_cur, d_cur, g_cur, info = _run_mstep(
                op, xs, mask, drift_net, dr_opt, dr_state, C_cur, d_cur, g_cur, hp, mk)
            _block(eqx.filter(drift_net, eqx.is_inexact_array))
            tm["mstep"].append(time.perf_counter() - tms)
        if step % val_every == 0 or step == n_steps:
            m = _validate_and_plot(step)
            res, jump, ic = (float(a) for a in aux)
            log_fn(f"[step {step}] res={res:.3f} jump={jump:.3f} ic={ic:.3f} | "
                   + " ".join(f"{k}={v:.4f}" for k, v in m.items()))
            history.append((step, m))
    total_s = time.perf_counter() - t_train0
    report = _timing_report(tm, init_s, total_s, n_steps)
    if timing:
        log_fn(_fmt_timing(report))
    return dict(op=op, drift_net=drift_net, g_cur=g_cur, C_cur=C_cur, d_cur=d_cur, timing=report), history


def _timing_report(tm, init_s, total_s, n_steps):
    """Split per-phase wall-clock into first-call (compile-inclusive) vs steady, derive rates."""
    def split(xs):
        if not xs:
            return 0.0, 0.0
        first = xs[0]
        steady = float(np.mean(xs[1:])) if len(xs) > 1 else xs[0]
        return first, steady
    e_first, e_steady = split(tm["estep"])
    m_first, m_steady = split(tm["mstep"])
    v_first, v_steady = split(tm["val"])
    fig_ms = float(np.mean(tm["fig"])) * 1e3 if tm["fig"] else 0.0
    compile_s = max(e_first - e_steady, 0) + max(m_first - m_steady, 0) + max(v_first - v_steady, 0)
    return {
        "total_s": total_s, "init_s": init_s, "compile_s": compile_s,
        "n_estep": len(tm["estep"]), "n_mstep": len(tm["mstep"]), "n_val": len(tm["val"]),
        "estep_first_ms": e_first * 1e3, "estep_ms": e_steady * 1e3,
        "estep_per_s": (1.0 / e_steady) if e_steady else 0.0,
        "mstep_first_ms": m_first * 1e3, "mstep_ms": m_steady * 1e3,
        "mstep_per_s": (1.0 / m_steady) if m_steady else 0.0,
        "val_first_ms": v_first * 1e3, "val_ms": v_steady * 1e3,
        "val_per_s": (1.0 / v_steady) if v_steady else 0.0,
        "fig_ms": fig_ms,
        "steps_per_s_overall": n_steps / total_s if total_s else 0.0,
    }


def _fmt_timing(r):
    return (f"[timing] total={r['total_s']:.1f}s  init/compile: init={r['init_s']:.1f}s "
            f"xla_compile≈{r['compile_s']:.1f}s\n"
            f"  overall {r['steps_per_s_overall']:.1f} steps/s (incl val/mstep)\n"
            f"  E-step  {r['estep_ms']:.1f} ms  -> {r['estep_per_s']:.1f}/s   (first {r['estep_first_ms']:.0f} ms, compile)\n"
            f"  M-step  {r['mstep_ms']:.1f} ms  -> {r['mstep_per_s']:.2f}/s  (first {r['mstep_first_ms']:.0f} ms, compile; n={r['n_mstep']})\n"
            f"  val     {r['val_ms']:.1f} ms  -> {r['val_per_s']:.2f}/s  (first {r['val_first_ms']:.0f} ms, compile; n={r['n_val']})\n"
            f"  figure  {r['fig_ms']:.0f} ms/fig")


def whole_trace_recon(op, C_cur, d_cur, refs, hp, key):
    """Whole-trace co-smoothing recon (== opssm_adapter._filter_full / the Kato publication metric): window the
    full standardized trace, MALA filter mean per window, average overlaps -> z_hat (T,d); recon vs the full
    trace. Returns (r2, z_hat (T,d)). This is HIGHER than the held-out val-window recon."""
    y = np.asarray(refs["y_full_std"])                           # (T,N) standardized full trace
    T, N = y.shape
    win, stride, d = int(hp["window"]), int(hp["stride"]), hp["latent_dim"]
    starts = list(range(0, T - win + 1, stride)) or [0]
    x = jnp.stack([jnp.asarray(y[s:s + win]) for s in starts], axis=1)   # (win,B,N)
    mask = jnp.ones((x.shape[0], x.shape[1], 1), x.dtype)
    center = _zhat(x, C_cur, d_cur)
    zc, _ = M.posterior_mean_mala(op, x, mask, center, hp["broad_std"], key,
                                  n_chains=hp["mala_chains"], n_steps=hp["mala_steps"], rng=hp["mala_rng"])
    zc = np.asarray(zc)                                          # (win,B,d)
    acc = np.zeros((T, d)); cnt = np.zeros((T, 1))
    for b, s in enumerate(starts):                              # stitch overlapping windows (average)
        L = min(win, T - s)
        acc[s:s + L] += zc[:L, b]; cnt[s:s + L] += 1
    z_hat = acc / np.maximum(cnt, 1)                            # (T,d)
    y_hat = z_hat @ np.asarray(C_cur).T + np.asarray(d_cur)
    r2 = 1.0 - ((y - y_hat) ** 2).sum() / max(((y - y.mean(0)) ** 2).sum(), 1e-8)
    return float(r2), z_hat


def load_refs(npz_path):
    """Load bridged arrays (jnp) + hparams. Returns (refs, hp). refs['true_drift'] baked from the system."""
    D = np.load(npz_path, allow_pickle=True)
    keys = ["x_train", "mask_train", "x_val", "mask_val", "filt_val", "z_grid", "full_obs",
            "z_val_true", "C_true", "d_true", "ts"]
    refs = {k: jnp.asarray(D[k]) for k in keys if k in D.files}
    refs["a"] = float(D["a"]); refs["sigma"] = float(D["sigma"])
    if "states_val" in D.files:                                   # Kato behavior labels (numpy, for figures)
        refs["states_val"] = np.asarray(D["states_val"])
        refs["state_names"] = list(D["state_names"]) if "state_names" in D.files else None
    for k in ("y_full_std", "states_full", "y_full_raw", "obs_mean"):   # Kato whole-trace recon + figure inputs
        if k in D.files:
            refs[k] = np.asarray(D[k])
    if "neuron_ids" in D.files:
        refs["neuron_ids"] = list(D["neuron_ids"])
    refs["obs_scale"] = float(D["obs_scale"]) if "obs_scale" in D.files else 1.0
    if "name" in D.files:
        refs["name"] = str(D["name"])
    system = str(D["system"]) if "system" in D.files else "doublewell"
    refs["system"] = system
    refs["true_drift"] = make_drift(system)[0]
    hp = dict(DEFAULTS)
    if "hparams" in D.files:                                      # per-experiment resolved hparams
        hp.update({k: (v.item() if hasattr(v, "item") else v) for k, v in D["hparams"].item().items()})
    hp["system"] = system
    hp["dt"] = float(D["dt"]); hp["noise_std"] = float(D["noise_std_eff"])
    for k in ("window", "stride"):                                # Kato windowing (for whole-trace recon)
        if k in D.files:
            hp[k] = int(D[k])
    return refs, hp
