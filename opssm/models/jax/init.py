"""Initialization helpers (JAX) -- mirror of opssm.models.init (torch): subspace-ID emission + dynamics,
least-squares linear dynamics, and an optax warm-start of the drift MLP to a linear drift.

The realization is defined only up to a similarity transform (the model's latent gauge), so C matches torch
only up to a sign/gauge -- the gauge-invariant object (the projector C C^T, and linear_dynamics on a FIXED
z_hat) is what parity checks. See opssm/models/init.py for the full rationale.
"""
import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from opssm.models.jax.obs import zhat_from_obs
from opssm.models.jax.dynamics import DriftNet


def _future_hankel(Y, k):
    """Y (T,B,D) -> mean-centered future block-Hankel (k*D, N), N=(T-k+1)*B. Column (b,t) stacks
    [y_t,...,y_{t+k-1}] for trajectory b."""
    T, B, D = Y.shape
    cols = [jnp.transpose(Y[t:t + k], (1, 0, 2)).reshape(B, k * D) for t in range(T - k + 1)]
    H = jnp.concatenate(cols, axis=0).T                             # (k*D, N)
    return H - H.mean(axis=1, keepdims=True)


def linear_dynamics(z_hat, dt):
    """Least-squares continuous-time drift A_cont=(A_disc-I)/dt from z_{t+1} ~ A_disc z_t on z_hat (T,B,d)."""
    d = z_hat.shape[-1]
    Z0, Z1 = z_hat[:-1].reshape(-1, d), z_hat[1:].reshape(-1, d)
    A_disc = jnp.linalg.lstsq(Z0, Z1)[0].T                         # Z1 ~ Z0 A_disc^T -> (d,d)
    return (A_disc - jnp.eye(d, dtype=z_hat.dtype)) / dt


def subspace_id(full_obs, d, dt, horizon=None):
    """Order-d subspace ID from standardized obs full_obs (T,B,D). C = first block row of the future-Hankel's
    top-d left singular subspace (Stiefel); A_cont = least-squares on the subspace scores. Returns (C (D,d), A_cont (d,d))."""
    T, B, D = full_obs.shape
    k = horizon or max(2, min(20, T // 5))
    U, _, _ = jnp.linalg.svd(_future_hankel(full_obs, k), full_matrices=False)
    C = jnp.linalg.qr(U[:D, :d])[0]                                # temporally-informed emission (Stiefel)
    ybar = full_obs.reshape(-1, D).mean(0)
    A_cont = linear_dynamics(zhat_from_obs(full_obs, C, ybar), dt)
    return C, A_cont


def warmstart_drift(drift_net, A_cont, key, *, zmax=2.0, n=4096, steps=400, lr=1e-2):
    """Pre-fit the (zero-init) drift MLP to the linear drift f(z)=A_cont z on z in [-zmax,zmax]^d via optax.
    Returns (drift_net_fitted, final_mse). Functional: returns a NEW drift_net (the M-step's dr_opt state is
    fresh regardless)."""
    d = A_cont.shape[0]
    opt = optax.adam(lr)
    net = drift_net.net
    opt_state = opt.init(eqx.filter(net, eqx.is_inexact_array))

    @eqx.filter_jit
    def upd(net, opt_state, z):
        def loss_fn(net):
            return ((net(z) - z @ A_cont.T) ** 2).mean()          # MLP is batched: (n,d)->(n,d)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(net)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(net, eqx.is_inexact_array))
        return eqx.apply_updates(net, updates), opt_state, loss

    loss = jnp.array(0.0)
    for i in range(steps):
        key, sk = jax.random.split(key)
        z = (jax.random.uniform(sk, (n, d)) * 2 - 1) * zmax
        net, opt_state, loss = upd(net, opt_state, z)
    return eqx.tree_at(lambda dn: dn.net, drift_net, net), float(loss)
