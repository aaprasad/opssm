"""JAX/equinox SDE coefficients -- mirror of opssm.models.dynamics (torch).

DriftNet f(z) with its divergence via forward-mode jax.jvp (vmapped over unit tangents, then trace),
and DiffusionNet g(z) with g^2 and its 1-D derivatives (nested jvp). Direct 1:1 translation of the
torch.func.jvp code in opssm/models/dynamics.py.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from opssm.models.jax.nn import MLP


class DriftNet(eqx.Module):
    net: MLP

    def __init__(self, hidden=64, layers=3, latent_dim=1, key=None, weights=None, biases=None):
        if weights is not None:
            self.net = MLP(weights=weights, biases=biases)
        else:
            self.net = MLP([latent_dim] + [hidden] * layers + [latent_dim], key, zero_last=True)

    def drift(self, z):                                           # z (...,d) -> f (...,d), div_f (...,)
        d = z.shape[-1]
        zin = z.reshape(-1, d)                                    # (N,d)
        eye = jnp.eye(d, dtype=z.dtype)                           # unit tangents

        def dir_jvp(e):                                           # e (d,) -> (f (N,d), Jf=J e (N,d))
            return jax.jvp(self.net, (zin,), (jnp.broadcast_to(e, zin.shape),))

        f, jf = jax.vmap(dir_jvp)(eye)                            # f (d,N,d) [same over tangents], jf (d,N,d)
        div = jnp.einsum("ini->n", jf)                           # trace_i d f_i/d z_i  (N,)
        return f[0].reshape(z.shape), div.reshape(z.shape[:-1])


class DiffusionNet(eqx.Module):
    net: MLP

    def __init__(self, hidden=64, layers=3, g_init=1.0, latent_dim=1, key=None, weights=None, biases=None):
        if weights is not None:
            self.net = MLP(weights=weights, biases=biases)
        else:
            m = MLP([latent_dim] + [hidden] * layers + [1], key, zero_last=True)   # last weight zeroed
            self.net = eqx.tree_at(lambda t: t.biases[-1], m, jnp.full_like(m.biases[-1], g_init))  # bias -> g_init

    def _g2(self, zin):
        g = self.net(zin)
        return g * g                                             # g^2 = g . g  (...,1)

    def diffusion(self, z):                                      # 1-D only: g2, (g2)', (g2)'' (each z.shape)
        zin = z.reshape(-1, 1)
        e = jnp.ones_like(zin)
        g2, dg2 = jax.jvp(self._g2, (zin,), (e,))
        _, d2g2 = jax.jvp(lambda x: jax.jvp(self._g2, (x,), (e,))[1], (zin,), (e,))
        return g2.reshape(z.shape), dg2.reshape(z.shape), d2g2.reshape(z.shape)
