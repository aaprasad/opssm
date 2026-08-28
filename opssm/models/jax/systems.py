"""JAX true-drift functions (for gauge-aligned metrics/figures) -- mirror of opssm.data.systems drifts,
same registered DEFAULTS so the JAX metric matches torch's make_drift(system) exactly."""
import jax.numpy as jnp

_DIM = {"doublewell": 1, "vanderpol": 2, "vanderpol_duncker": 2, "lorenz": 3, "none": 1}


def doublewell(z, a=1.0):
    return a * (z - z ** 3)


def vanderpol(z, mu=1.5):
    x, v = z[..., 0], z[..., 1]
    return jnp.stack([v, mu * (1.0 - x ** 2) * v - x], axis=-1)


def vanderpol_duncker(z, tau=10.0, mu=2.0):
    x1, x2 = z[..., 0], z[..., 1]
    return jnp.stack([tau * mu * (x1 - x1 ** 3 / 3 - x2), tau * (x1 / mu)], axis=-1)


def lorenz(z, s=10.0, r=28.0, b=8.0 / 3.0):
    x, y, w = z[..., 0], z[..., 1], z[..., 2]
    return jnp.stack([s * (y - x), x * (r - w) - y, x * y - b * w], axis=-1)


def none_drift(z):
    return jnp.zeros_like(z)


_FN = {"doublewell": doublewell, "vanderpol": vanderpol, "vanderpol_duncker": vanderpol_duncker,
       "lorenz": lorenz, "none": none_drift}


def make_drift(name, **params):
    """Return (drift_fn (...,d)->(...,d), dim). Registered defaults baked in (matches torch make_drift)."""
    if name not in _FN:
        raise KeyError(f"unknown system {name!r}; known: {sorted(_FN)}")
    fn = _FN[name]
    return (lambda z: fn(z, **params)), _DIM[name]
