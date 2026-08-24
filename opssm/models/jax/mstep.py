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
          drift_target="det_mid", bootstrap=False):
    """One EM M-step (em_highd active path). Order: filter-mean (MALA) increments -> Stiefel obs-map + cstab ->
    drift GATED on cstab<c_stable_tol (det_mid RK2-midpoint input shift) -> scalar diffusion. Returns
    (info dict {g_cur,C_cur,d_cur,cstab,accept}, drift_net, dr_state)."""
    center = zhat_from_obs(x, C_cur, d_cur) if learn_obs else x   # (T,B,d) (obs standardized upstream)
    accept = None
    if bootstrap:                                                # no-warmup seed: fit from the init projection
        z_hat = center
    elif mean_method == "mala":
        z_hat, accept = posterior_mean_mala(model, x, mask, center, broad_std, key, **(mala or {}))
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
    g_cur = None
    if learn_g:
        g_cur = fit_diffusion_scalar(drift_net, zc, zc_next, dz, dt)
    return dict(g_cur=g_cur, C_cur=C_cur, d_cur=d_cur, cstab=cstab, accept=accept), drift_net, dr_state
