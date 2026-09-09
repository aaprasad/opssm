"""EM M-step (JAX) -- mirror of opssm.models.mstep (torch), ACTIVE em_highd path only:
  filter MEAN via MALA (mean_method='mala'), Stiefel obs-map (orthogonal Procrustes), drift regression
  (drift_target det_mid/forward), scalar diffusion from the mean-increment residual (joint_g=false).
The mesh-free MALA is hand-rolled (lax.scan) to match torch's per-site adaptive-step MALA (step->0.574)
exactly in algorithm; being stochastic it is validated statistically (mean agreement + accept~0.574), not
to machine precision. DEFERRED (not on the em_highd path): grid/fixed-node means, smoother, pair samplers
(joint_g), g_net, ito_correction -- add when a config needs them.
"""
import math

import jax
import jax.numpy as jnp
import equinox as eqx

from opssm.models.jax.obs import zhat_from_obs


def _mala_chains(model, x, mask, center, n_chains, n_steps, broad_std, key,
                 rng="stochastic", step_size=0.1, burn_in=None, init_std=0.0, adapt=True):
    """T*B independent d-dim MALA chains on the per-time filter marginal pi_t ~ exp(ell_t). ell_t from
    b0=coeffs(ctx,s=0); grad via trunk_grad. Per-site step adapts toward the 0.574 MALA optimum during
    burn_in. Returns (z_final (T,B,K,d), z_mean (T,B,d) post-burn-in chain mean, accept). 1:1 with torch
    _mala_chains; RNG differs cross-framework so validate statistics, not values."""
    T, B, d = center.shape
    if burn_in is None:
        burn_in = max(1, n_steps // 3)
    if rng == "crn":
        key = jax.random.PRNGKey(0)                              # fixed-seed determinism (within JAX)
    elif rng != "stochastic":
        raise NotImplementedError(f"rng={rng!r}: qmc deferred (see torch _mala_chains)")
    ctx = model.context(x, mask)
    b0 = model.coeffs(ctx, jnp.zeros(1, center.dtype))[:, :, 0]  # (T,B,p) log-density coeffs at s=0

    def ell_grad(z):                                            # z (T,B,K,d) -> ell (T,B,K), grad (T,B,K,d)
        tau, gtau = model.trunk_grad(z)                        # (T,B,K,p),(T,B,K,d,p)
        ell = jnp.einsum("tbp,tbkp->tbk", b0, tau) + model.bias
        return ell, jnp.einsum("tbp,tbkdp->tbkd", b0, gtau)

    obs = mask[..., 0]                                          # (T,B)
    c = (center * obs[..., None])[:, :, None]                   # (T,B,1,d) zero the meaningless gap center
    sd = (init_std * obs + broad_std * (1.0 - obs))[..., None, None]   # (T,B,1,1) BROAD init at gaps
    key, k0 = jax.random.split(key)
    z = c + sd * jax.random.normal(k0, (T, B, n_chains, d), center.dtype)
    ell, grad = ell_grad(z)
    eps = jnp.full((T, B, 1, 1), float(step_size), center.dtype)

    def body(carry, k):
        z, ell, grad, eps, key, zsum, ncol, acc_sum = carry
        key, kn, ku = jax.random.split(key, 3)
        noise = jax.random.normal(kn, z.shape, z.dtype)
        z_p = z + eps * grad + jnp.sqrt(2 * eps) * noise         # Langevin proposal
        ell_p, grad_p = ell_grad(z_p)
        e4 = 4 * eps[..., 0]                                     # (T,B,1)
        log_q_fwd = -((z_p - z - eps * grad) ** 2).sum(-1) / e4  # (T,B,K)
        log_q_bwd = -((z - z_p - eps * grad_p) ** 2).sum(-1) / e4
        log_alpha = (ell_p - ell) + (log_q_bwd - log_q_fwd)     # (T,B,K)
        acc = jnp.log(jax.random.uniform(ku, log_alpha.shape, log_alpha.dtype)) < log_alpha
        ae = acc[..., None]
        z = jnp.where(ae, z_p, z); ell = jnp.where(acc, ell_p, ell); grad = jnp.where(ae, grad_p, grad)
        acc_sum = acc_sum + acc.mean()
        ap = acc.mean(-1)[..., None, None]                      # (T,B,1,1) per-site accept
        eps_ad = jnp.clip(eps * jnp.exp(0.75 * (ap - 0.574)), 1e-4, 2.0)
        eps = jnp.where(adapt & (k < burn_in), eps_ad, eps)
        post = k >= burn_in
        zsum = zsum + jnp.where(post, z.mean(2), jnp.zeros_like(center))
        ncol = ncol + post.astype(jnp.int32)
        return (z, ell, grad, eps, key, zsum, ncol, acc_sum), None

    carry = (z, ell, grad, eps, key, jnp.zeros_like(center), jnp.array(0, jnp.int32), jnp.array(0.0, center.dtype))
    (z, ell, grad, eps, key, zsum, ncol, acc_sum), _ = jax.lax.scan(body, carry, jnp.arange(n_steps))
    return z, zsum / jnp.maximum(ncol, 1), float(acc_sum) / max(n_steps, 1)


def posterior_mean_mala(model, x, mask, center, broad_std, key, n_chains=64, n_steps=30, rng="stochastic"):
    """MALA readout of the per-time filter mean E[z_t|y_{0:t}] -> (z_hat (T,B,d), accept)."""
    _, z_mean, acc = _mala_chains(model, x, mask, center, n_chains, n_steps, broad_std, key, rng=rng)
    return z_mean, acc


def fit_diffusion_cov(drift_net, zc, zc_next, dz, dt, g_floor=0.05):
    """FULL diffusion covariance Sigma = L L^T from the trapezoidal increment residual (mirror of torch).
    Delta = z_{t+1}-z_t-1/2(f+f')dt ~ N(0, Sigma dt) -> Sigma = E[r r^T] dt with r = dz - f_trap.
    Returns the lower-triangular Cholesky factor L; a g_floor^2 I ridge keeps it well-conditioned."""
    f_trap = 0.5 * (drift_net.net(zc) + drift_net.net(zc_next))
    r = dz - f_trap                                                  # (N,d) = Delta/dt
    d = r.shape[-1]
    Sig = (r.T @ r) / max(r.shape[0], 1) * dt
    Sig = 0.5 * (Sig + Sig.T) + (g_floor ** 2) * jnp.eye(d, dtype=r.dtype)
    return jnp.linalg.cholesky(Sig)


def chol_summary(L):
    """Sigma = L L^T -> (g_det = det(Sigma)^{1/2d}, g_iso = sqrt(tr/d), aniso = sqrt(lmax/lmin)).
    g_det is the gauge-covariant scalar (Sigma -> A Sigma A^T scales it by |det A|^{1/d}) and equals g
    exactly when Sigma = g^2 I, so `g` stays comparable with every isotropic run."""
    ev = jnp.clip(jnp.linalg.eigvalsh(L @ L.T), 1e-24, None)
    return (float(jnp.exp(0.5 * jnp.mean(jnp.log(ev)))), float(jnp.sqrt(jnp.mean(ev))),
            float(jnp.sqrt(ev.max() / ev.min())))


def fit_obs_noise(x, mask, z_samp, w_samp, C_cur, d_cur, mode="diag", est="perp", floor=1e-3):
    """Observation-noise M-step (mirror of torch fit_obs_noise) -> (R (D,), diag with both estimators).

    est='posterior': R = E_q[(y-Cz-d)^2] over MALA samples -- textbook EM, but inherits the operator's
      known posterior over-dispersion, so it reads high.
    est='perp' (DEFAULT): R_i = E[perp_i^2] / (1 - ||C_[i,:]||^2) with perp = (I - C C^T)(y-d), using only
      the D-d observation directions no latent can explain -- immune to both the over-dispersion bias and
      the R -> wider-posterior -> larger-R feedback."""
    h = z_samp if C_cur is None else jnp.einsum("od,tbkd->tbko", C_cur, z_samp) + d_cur
    obs = mask[..., 0] > 0
    nrm = jnp.maximum(obs.sum(), 1)
    R_post = (((w_samp[..., None] * (x[:, :, None] - h) ** 2).sum(2)) * obs[..., None]).sum((0, 1)) / nrm
    R_perp = None
    if C_cur is not None and C_cur.shape[0] > C_cur.shape[1]:
        yc = x - d_cur
        perp = yc - yc @ C_cur @ C_cur.T                             # (I - C C^T)(y-d)
        Pii = jnp.maximum(1.0 - (C_cur ** 2).sum(-1), 1e-3)          # diag of the projector
        R_perp = ((perp ** 2) * obs[..., None]).sum((0, 1)) / nrm / Pii
    if est not in ("posterior", "perp"):
        raise ValueError(f"obs_noise_est must be 'posterior' or 'perp', got {est!r}")
    R = R_post if (est == "posterior" or R_perp is None) else R_perp
    if mode == "scalar":
        R = jnp.full_like(R, R.mean())
    elif mode != "diag":
        raise ValueError(f"obs_noise_mode must be 'diag' or 'scalar', got {mode!r}")
    diag = {"R_post": float(R_post.mean()),
            "R_perp": float("nan") if R_perp is None else float(R_perp.mean())}
    return jnp.maximum(R, floor ** 2), diag


def fit_obs_map_stiefel(z_hat, y, C_cur):
    """High-D Stiefel obs map (orthogonal Procrustes): C = nearest orthonormal-cols to cross-cov(y,z_hat);
    d = intercept; cstab = 1 - mean principal-angle cosine. z_hat (T,B,d), y (T,B,D), C_cur (D,d)."""
    d = z_hat.shape[-1]
    Z = z_hat.reshape(-1, d)
    Y = y.reshape(-1, y.shape[-1])
    M = (Y - Y.mean(0)).T @ (Z - Z.mean(0))                     # (D,d) cross-cov
    U, _, Vt = jnp.linalg.svd(M, full_matrices=False)
    C_new = U @ Vt                                              # (D,d) Stiefel
    cos = jnp.minimum(jnp.linalg.svd(C_new.T @ C_cur, compute_uv=False), 1.0)
    cstab = 1.0 - float(cos.mean())
    d_new = Y.mean(0) - C_new @ Z.mean(0)                       # (D,) intercept
    return C_new, d_new, cstab


def fit_drift(drift_net, dr_opt, dr_state, zc, dz, reg_lambda, m_inner):
    """Regress f_theta(z) ~ dz with L2 weight decay via optax. Returns (drift_net, dr_state). zc,dz (N,d)."""
    net = drift_net.net

    @eqx.filter_jit
    def step(net, state):
        def loss_fn(net):
            f = net(zc)
            reg = sum((w ** 2).sum() for w in net.weights) + sum((b ** 2).sum() for b in net.biases)
            return ((f - dz) ** 2).sum(-1).mean() + reg_lambda * reg
        grads = eqx.filter_grad(loss_fn)(net)
        updates, state = dr_opt.update(grads, state, eqx.filter(net, eqx.is_inexact_array))
        return eqx.apply_updates(net, updates), state

    for _ in range(m_inner):
        net, dr_state = step(net, dr_state)
    return eqx.tree_at(lambda dn: dn.net, drift_net, net), dr_state


def fit_diffusion_scalar(drift_net, zc, zc_next, dz, dt):
    """Scalar g from the trapezoidal mean-increment residual r = dz - 1/2(f(z_t)+f(z_{t+1})):
    g = sqrt(mean|r|^2 * dt / d), clamped >= 0.05. (joint_g=false / g_net=false path.)"""
    net = drift_net.net
    f_trap = 0.5 * (net(zc) + net(zc_next))
    r = dz - f_trap
    dcnt = r.shape[-1]
    return float(jnp.maximum(jnp.sqrt((r ** 2).sum(-1).mean() * dt / dcnt), 0.05))


def mstep(model, x, mask, dt, drift_net, dr_opt, dr_state, key, *,
          learn_g, reg_lambda, m_inner, learn_obs=False, c_stable_tol=0.05, C_cur=None, d_cur=None,
          n_mean=256, near_std=0.3, broad_std=1.6, mean_method="mala", mala=None,
          drift_target="det_mid", bootstrap=False, diffusion_cov=False, g_floor=0.05,
          learn_obs_noise=False, obs_noise_mode="diag", obs_noise_est="perp", noise_var_in=None,
          obs_noise_damp=0.5):
    """One EM M-step (em_highd active path). Order: filter-mean (MALA) increments -> Stiefel obs-map + cstab ->
    drift GATED on cstab<c_stable_tol (det_mid RK2-midpoint input shift) -> scalar diffusion. Returns
    (info dict {g_cur,C_cur,d_cur,cstab,accept}, drift_net, dr_state)."""
    center = zhat_from_obs(x, C_cur, d_cur) if learn_obs else x   # (T,B,d) (obs standardized upstream)
    accept = None
    z_samp = w_samp = None                                        # posterior samples (obs-noise M-step)
    if bootstrap:                                                # no-warmup seed: fit from the init projection
        z_hat = center
    elif mean_method == "mala":
        mk = dict(mala or {})
        z_samp, z_hat, accept = _mala_chains(model, x, mask, center, mk.pop("n_chains", 64),
                                             mk.pop("n_steps", 30), broad_std, key, **mk)
        K = z_samp.shape[2]                                      # chain states = equally weighted draws
        w_samp = jnp.full(z_samp.shape[:3], 1.0 / K, z_samp.dtype)
    else:
        raise NotImplementedError(f"mean_method={mean_method!r}: only 'mala' ported (em_highd path)")
    m = mask[..., 0]
    valid = (m[:-1] * m[1:]) > 0                                  # (T-1,B) data-anchored increments
    zc = z_hat[:-1][valid]                                        # (N,d)
    zc_next = z_hat[1:][valid]
    dz = ((z_hat[1:] - z_hat[:-1]) / dt)[valid]                   # (N,d)

    cstab = 0.0
    if learn_obs:
        C_cur, d_cur, cstab = fit_obs_map_stiefel(z_hat, x, C_cur)
    if cstab < c_stable_tol:                                      # sensor-before-dynamics gate
        zc_fit = zc
        if drift_target == "det_mid":                            # RK2 predicted-midpoint input shift
            zc_fit = zc + 0.5 * dt * drift_net.net(zc)
        elif drift_target != "forward":
            raise NotImplementedError(f"drift_target={drift_target!r}: only det_mid/forward ported")
        drift_net, dr_state = fit_drift(drift_net, dr_opt, dr_state, zc_fit, dz, reg_lambda, m_inner)
    g_cur = L_cur = None
    g_iso = aniso = float("nan")
    if learn_g:
        if diffusion_cov:                                         # FULL matrix diffusion Sigma = L L^T
            L_cur = fit_diffusion_cov(drift_net, zc, zc_next, dz, dt, g_floor=g_floor)
            g_cur, g_iso, aniso = chol_summary(L_cur)
        else:
            g_cur = fit_diffusion_scalar(drift_net, zc, zc_next, dz, dt)
    noise_var, R_perp, R_post = noise_var_in, float("nan"), float("nan")
    if learn_obs_noise and not bootstrap and z_samp is not None:
        R_new, rdiag = fit_obs_noise(x, mask, z_samp, w_samp, C_cur if learn_obs else None,
                                     d_cur if learn_obs else None, mode=obs_noise_mode, est=obs_noise_est)
        R_perp, R_post = rdiag["R_perp"], rdiag["R_post"]
        # DAMPED: R feeds back into likelihood sharpness, so bound how far one M-step can move it.
        noise_var = (R_new if noise_var_in is None else
                     obs_noise_damp * noise_var_in + (1.0 - obs_noise_damp) * R_new)
    return (dict(g_cur=g_cur, L_cur=L_cur, C_cur=C_cur, d_cur=d_cur, cstab=cstab, accept=accept,
                 noise_var=noise_var, R_perp=R_perp, R_post=R_post, g_iso=g_iso, aniso=aniso),
            drift_net, dr_state)
