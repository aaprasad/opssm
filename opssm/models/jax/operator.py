"""JAX/equinox operator (DeepONet Zakai filter) -- mirror of opssm.models.operator (torch).

This module is being ported incrementally. STAGE A (here): the mesh-free state-basis derivative primitives
`trunk_zderivs` (grad + Laplacian, nested jvp + vmap) and `trunk_grad` (grad only) -- the E-step hot path.
They take the trunk MLP as an explicit arg so they're testable standalone and reusable by the OperatorFilter
class (STAGE B: encoder + branch + trunk + context/coeffs/coeffs_dtime/log_density -- added next).
Direct 1:1 translation of operator.py:82-112 (torch.func.jvp/vmap -> jax.jvp/vmap).
"""
import jax
import jax.numpy as jnp


def trunk_zderivs(trunk, z):
    """State basis trunk(z) with GRADIENT + LAPLACIAN by autodiff. z (...,d) ->
    tau (...,p), grad_tau (...,d,p) [d_i tau], lap_tau (...,p) [sum_i d_ii tau]. Forward-over-forward
    jvp along each unit tangent e_i yields BOTH d_i tau and d_ii tau; vmapped over the d tangents."""
    d = z.shape[-1]
    zin = z.reshape(-1, d)                                        # (N,d)
    eye = jnp.eye(d, dtype=z.dtype)

    def along(e):                                                 # e (d,) -> (tau, d_i tau, d_ii tau), each (N,p)
        v = jnp.broadcast_to(e, zin.shape)
        (tau, di), (_, dii) = jax.jvp(lambda x: jax.jvp(trunk, (x,), (v,)), (zin,), (v,))
        return tau, di, dii

    tau, grad, lap_ax = jax.vmap(along)(eye)                      # (d,N,p) each; tau identical over tangents
    shp = z.shape[:-1]
    return (tau[0].reshape(*shp, -1),                            # tau
            jnp.moveaxis(grad, 0, -2).reshape(*shp, d, -1),      # (d,N,p) -> (N,d,p)
            lap_ax.sum(0).reshape(*shp, -1))                     # sum axes -> Laplacian (N,p)


def trunk_grad(trunk, z):
    """State basis trunk(z) with GRADIENT only -> tau (...,p), grad_tau (...,d,p). Single forward jvp per
    unit tangent, vmapped over the d tangents (forward-mode Jacobian). Used by the MALA readout."""
    d = z.shape[-1]
    zin = z.reshape(-1, d)
    eye = jnp.eye(d, dtype=z.dtype)
    tau, grads = jax.vmap(lambda e: jax.jvp(trunk, (zin,), (jnp.broadcast_to(e, zin.shape),)))(eye)  # (d,N,p)
    shp = z.shape[:-1]
    return tau[0].reshape(*shp, -1), jnp.moveaxis(grads, 0, -2).reshape(*shp, d, -1)
