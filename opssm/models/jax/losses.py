"""Mesh-free continuous-time Zakai PINN loss + SNIS collocation (JAX) -- mirror of opssm.models.losses.

Pure functional: no in-place, `.detach()` -> `jax.lax.stop_gradient`, torch reductions -> jnp/jax.nn.
The E-step is `eqx.filter_grad(pinn_zakai_loss)` -- reverse-through-forward (the loss internally calls
`jax.jvp`/`vmap` via model.trunk_zderivs / coeffs_dtime and drift), which composes natively in JAX.
Randomness (collocation draw, stochastic time subsample) is threaded via explicit PRNG keys / a
precomputed time index `ti`, so the loss is a deterministic jittable function of (model, batch, key).
"""
import math

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def kl_target_pred(target, log_pred, eps=1e-12):
    """KL(target || pred), target a mass (T,B,Nz), log_pred the predicted log-mass."""
    t = jnp.maximum(target, eps)
    return (t * (jnp.log(t) - log_pred)).sum(axis=-1).mean()


def sample_collocation(key, xs, mask, n_colloc, near_std, broad_std, center=None):
    """MESH-FREE collocation: draw state points from a data-following proposal q (no grid, no zmax).
    Half near the observation (peaked observed posterior), half broad (covering the wells / the gap's
    bimodality where there is no observation). Returns z (T,B,K,d) sampled points and log_q (T,B,K) the
    ISOTROPIC-Gaussian-mixture proposal log-density (the SNIS weights). `center` (T,B,d): the LATENT
    location for the near component; default = xs (direct obs). For a high-D linear sensor pass a
    z-estimate (the pseudo-inverse (C^T C)^-1 C^T (y-d)) since the obs is not in latent space."""
    T, B = xs.shape[0], xs.shape[1]
    ctr = xs if center is None else center                       # (T,B,d)
    d = ctr.shape[-1]
    obs = mask[..., 0]                                           # (T,B) observed indicator
    ctr = ctr * obs[..., None]                                   # latent estimate where observed (T,B,d)
    # "near" tight around the obs where observed; in the GAP no obs, posterior is BIMODAL -> widen to broad.
    near_sd = near_std * obs + broad_std * (1.0 - obs)           # (T,B) per step
    Kn = n_colloc // 2
    Kb = n_colloc - Kn
    kn, kb = jax.random.split(key)
    near = ctr[:, :, None] + near_sd[..., None, None] * jax.random.normal(kn, (T, B, Kn, d), ctr.dtype)
    broad = broad_std * jax.random.normal(kb, (T, B, Kb, d), ctr.dtype)
    z = jnp.concatenate([near, broad], axis=2)                  # (T,B,K,d)
    off = math.log(2.0) + 0.5 * d * math.log(2 * math.pi)       # d-dim mixture normalizer
    lqn = (-0.5 * (((z - ctr[:, :, None]) / near_sd[..., None, None]) ** 2).sum(-1)
           - d * jnp.log(near_sd[..., None]) - off)             # (T,B,K)
    lqb = -0.5 * ((z / broad_std) ** 2).sum(-1) - d * math.log(broad_std) - off
    log_q = jnp.logaddexp(lqn, lqb)                             # log(0.5 q_near + 0.5 q_broad), (T,B,K)
    return z, log_q


def as_chol(sigma):
    """Diffusion spec -> (d,d) Cholesky factor L with Sigma = L L^T, or None for the isotropic/g_net path."""
    if callable(sigma):
        return None
    a = jnp.asarray(sigma)
    return a if a.ndim == 2 else None


def gauss_loglik(xs, z_col, decode, noise_var):
    """log N(y; h(z), R) up to a constant, R diagonal. noise_var: scalar or (D,) VARIANCES."""
    resid = xs[:, :, None] - (z_col if decode is None else decode(z_col))
    return -0.5 * (resid ** 2 / noise_var).sum(-1)


def fp_diffusion_term(b_s, grad_ell, sec_tau, L):
    """grad ell^T Sigma grad ell + tr(Sigma H(ell)) = |L^T grad ell|^2 + sum_p b_p tr(Sigma H(tau_p))."""
    Lg = jnp.einsum("tbskd,de->tbske", grad_ell, L)
    return (Lg ** 2).sum(-1) + jnp.einsum("tbsp,tbkp->tbsk", b_s, sec_tau)


def pinn_zakai_loss(model, xs, mask, z_col, log_q, s_coll, drift, sigma, log_prior,
                    noise_var, dt, ti=None, res_post=0.0, res_mode="l2", decode=None):
    """MESH-FREE continuous-time Zakai PINN -- no grid, no time-stepping, no Euler. 1:1 with the torch
    pinn_zakai_loss (losses.py:59-144). Collocation z_col (T,B,K,d) is SAMPLED (log_q its log-density);
    Z and evidence c are SNIS estimates; the drift f,f' are analytic at the samples (drift(z)->f,div).
        predict (PINN):  d_tau ell = -(div f + f.grad ell) + 1/2 g^2 (|grad ell|^2 + lap ell)
        update  (jump):  pi_{i+1}(.,0) = normalize( lik_{i+1} * pi_i(.,dt) )   [SNIS]
        anchor  (IC):    pi_0(.,0)     = normalize( lik_0 * prior )            [SNIS]
    The residual (grad/lap/d_s Jacobians) is taken on the time subsample `ti` (stochastic time collocation,
    threaded in by the caller; default = all T). Returns (residual_loss, jump_loss, ic_loss, data_nll)."""
    T, B, K, d = z_col.shape
    ctx = model.context(xs, mask)                               # (T,B,C)

    # ---- cheap path (no autodiff): basis at all steps; coeffs at the interval ends s=0,1
    tau = model.trunk(z_col.reshape(-1, d)).reshape(T, B, K, -1)   # (T,B,K,p)
    b_ends = model.coeffs(ctx, jnp.asarray([0.0, 1.0], dtype=z_col.dtype))   # (T,B,2,p)
    ell0 = jnp.einsum("tbp,tbkp->tbk", b_ends[:, :, 0], tau) + model.bias   # post-update (s=0)

    loglik = gauss_loglik(xs, z_col, decode, noise_var)            # (T,B,K); R scalar or (D,) diagonal
    m = mask[..., 0]                                              # (T,B)
    logZ0 = logsumexp(ell0 - log_q, axis=-1, keepdims=True) - math.log(K)
    logpi0 = ell0 - logZ0                                         # normalized log-density

    # predict pi_i(.,dt) as SNIS weights on step (i+1)'s samples: ell of step i (s=1) at z_col[i+1].
    ellT_next = jnp.einsum("tbp,tbkp->tbk", b_ends[:-1, :, 1], tau[1:]) + model.bias
    logW_pred = jax.lax.stop_gradient(jax.nn.log_softmax(ellT_next - log_q[1:], axis=-1))
    logW_tgt = jax.nn.log_softmax(logW_pred + loglik[1:] * m[1:][..., None], axis=-1)
    jump_loss = -(jnp.exp(logW_tgt) * logpi0[1:]).sum(-1).mean()
    logW_ic = jax.nn.log_softmax(
        log_prior(z_col[0]) + loglik[0] * m[0][..., None] - log_q[0], axis=-1)
    ic_loss = -(jnp.exp(logW_ic) * logpi0[0]).sum(-1).mean()
    logc = logsumexp(logW_pred + loglik[1:], axis=-1)            # evidence c_{i+1}
    nll = -(logc * m[1:]).sum(0).mean()

    # ---- residual path (autodiff) on the time subsample `ti`
    if ti is None:
        ti = jnp.arange(T)
    b_s, ds_b = model.coeffs_dtime(ctx[ti], s_coll)              # (Ts,B,Ns,p)
    L = as_chol(sigma)                                           # (d,d) Cholesky, or None (iso / g_net)
    if L is not None:                # anisotropic Sigma = L L^T: 2nd derivs along L's COLUMNS -> tr(Sigma H)
        tau_s, grad_tau, sec_tau = model.trunk_zderivs_dirs(z_col[ti], L.T)
        lap_tau = None
    else:
        tau_s, grad_tau, lap_tau = model.trunk_zderivs(z_col[ti])    # (Ts,B,K,p),(...,K,d,p),(...,K,p)
    grad_ell = jnp.einsum("tbsp,tbkdp->tbskd", b_s, grad_tau)    # (Ts,B,Ns,K,d)  grad ell
    grad_ell_sq = (grad_ell ** 2).sum(-1)                        # (Ts,B,Ns,K)    |grad ell|^2
    lap_ell = (None if lap_tau is None else
               jnp.einsum("tbsp,tbkp->tbsk", b_s, lap_tau))       # (Ts,B,Ns,K)    Laplacian ell
    ds_ell = jnp.einsum("tbsp,tbkp->tbsk", ds_b, tau_s)
    f, div_f = drift(z_col[ti])                                  # f (Ts,B,K,d), div f (Ts,B,K)
    f_dot = jnp.einsum("tbskd,tbkd->tbsk", grad_ell, f)         # f . grad ell  (Ts,B,Ns,K)
    divf = div_f[:, :, None]                                     # (Ts,B,1,K)
    if L is not None:                                            # MATRIX diffusion Sigma = L L^T (anisotropic)
        rhs = -(divf + f_dot) + 0.5 * fp_diffusion_term(b_s, grad_ell, sec_tau, L)
    elif callable(sigma):                                        # state-dependent g^2(z) -- 1-D only (g_net)
        g2, dg2, d2g2 = sigma(z_col[ti])                        # (Ts,B,K,1) each (DiffusionNet.diffusion 1-D)
        g2 = g2[:, :, None]; dg2 = dg2[:, :, None]; d2g2 = d2g2[:, :, None]
        dz_ell = grad_ell[..., 0]                               # d==1 gradient component
        # 1/2 d^2_z(g^2 rho) in log-space: 1/2 (g^2)'' + (g^2)' ell' + 1/2 g^2 (ell'^2 + ell'')
        rhs = (-(divf + f_dot)
               + 0.5 * d2g2 + dg2 * dz_ell + 0.5 * g2 * (grad_ell_sq + lap_ell))
    else:                                                        # constant scalar g (isotropic D = g^2 I)
        rhs = -(divf + f_dot) + 0.5 * sigma ** 2 * (grad_ell_sq + lap_ell)
    res2 = (ds_ell / dt - rhs) ** 2                              # (Ts,B,Ns,K)
    # SCALE-INVARIANT residual: in LOG space the FP terms scale ~1/sigma^2, so for a SHARP density the L2
    # residual EXPLODES far from the mode (biasing WIDE). "rel"/"log1p" tame the tail; "l2" = absolute.
    if res_mode == "rel":
        scale = jax.lax.stop_gradient(rhs) ** 2 + jax.lax.stop_gradient(ds_ell / dt) ** 2 + 1.0
        res2 = res2 / scale
    elif res_mode == "log1p":
        res2 = jnp.log1p(res2)
    if res_post > 0:
        # weight the FP residual by posterior mass: enforce the physics WHERE THE DENSITY IS (the modes).
        W = jax.lax.stop_gradient(jax.nn.softmax(ell0[ti] - log_q[ti], axis=-1))   # (Ts,B,K)
        w = (1.0 - res_post) / K + res_post * W
        res_loss = (w[:, :, None] * res2).sum(-1).mean()
    else:
        res_loss = res2.mean()
    return res_loss, jump_loss, ic_loss, nll
