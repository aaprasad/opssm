"""High-D linear-sensor helpers (JAX) -- mirror of opssm.models.obs (torch).

Pseudo-inverse latent estimate + decode factory. jnp.einsum / jnp.linalg.solve.
"""
import jax.numpy as jnp


def zhat_from_obs(y, C, d):
    """Pseudo-inverse latent estimate z_hat = (C^T C)^-1 C^T (y - d) -> (...,d_lat). C (D,d_lat),
    y (...,D). On the Stiefel manifold C^T C = I so this is C^T(y-d); the solve is kept for robustness
    at random init. The collocation / M-step center."""
    r = jnp.einsum("od,...o->...d", C, y - d)                     # C^T (y-d)  (...,d_lat)
    gram = C.T @ C                                                # (d_lat,d_lat)
    return jnp.linalg.solve(gram, r[..., None])[..., 0]          # (C^T C)^-1 C^T (y-d)


def make_decode(C, d):
    """h(z) = C z + d, C (D,d_lat), z (...,d_lat) -> (...,D). Obs standardized to ~unit scale at the
    dataloader level, so the decode carries no separate obs-scale factor (s_scale removed)."""
    return lambda z: jnp.einsum("od,...d->...o", C, z) + d
