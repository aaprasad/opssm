"""JAX/equinox operator (DeepONet Zakai filter) -- mirror of opssm.models.operator (torch).

`trunk_zderivs`/`trunk_grad`: mesh-free state-basis grad+Laplacian / grad (nested jvp + vmap).
`GRUEncoder`: strictly-causal 1-layer GRU + linear projection, hand-rolled to match torch nn.GRU EXACTLY
  (gate order reset,update,new; separate bias_ih/bias_hh) so weights transfer 1:1 for parity.
`OperatorFilter`: encoder + branch MLP + trunk MLP + scalar bias, with context/coeffs/coeffs_dtime/
  log_density/log_posterior. (OperatorBackward, reverse=True, deferred to the smoother stage.)
1:1 translation of operator.py:41-146 + encoders.py GRUEncoder.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from jax.scipy.special import logsumexp

from opssm.models.jax.nn import MLP


def trunk_zderivs(trunk, z):
    """trunk(z) with GRADIENT + LAPLACIAN. z (...,d) -> tau (...,p), grad (...,d,p), lap (...,p)."""
    d = z.shape[-1]
    zin = z.reshape(-1, d)
    eye = jnp.eye(d, dtype=z.dtype)

    def along(e):
        v = jnp.broadcast_to(e, zin.shape)
        (tau, di), (_, dii) = jax.jvp(lambda x: jax.jvp(trunk, (x,), (v,)), (zin,), (v,))
        return tau, di, dii

    tau, grad, lap_ax = jax.vmap(along)(eye)
    shp = z.shape[:-1]
    return (tau[0].reshape(*shp, -1),
            jnp.moveaxis(grad, 0, -2).reshape(*shp, d, -1),
            lap_ax.sum(0).reshape(*shp, -1))


def trunk_grad(trunk, z):
    """trunk(z) with GRADIENT only -> tau (...,p), grad (...,d,p) (forward-mode Jacobian; MALA readout)."""
    d = z.shape[-1]
    zin = z.reshape(-1, d)
    eye = jnp.eye(d, dtype=z.dtype)
    tau, grads = jax.vmap(lambda e: jax.jvp(trunk, (zin,), (jnp.broadcast_to(e, zin.shape),)))(eye)
    shp = z.shape[:-1]
    return tau[0].reshape(*shp, -1), jnp.moveaxis(grads, 0, -2).reshape(*shp, d, -1)


class GRUEncoder(eqx.Module):
    """1-layer causal GRU + linear -> ctx. Cell hand-rolled to match torch nn.GRU (gates r,z,n; separate
    input/hidden biases) so torch weights transfer 1:1. inp (T,B,in) -> (T,B,ctx)."""
    W_ih: jax.Array
    W_hh: jax.Array
    b_ih: jax.Array
    b_hh: jax.Array
    Wc: jax.Array
    bc: jax.Array
    hidden: int = eqx.field(static=True)

    def __init__(self, in_dim, ctx_dim, hidden=64, key=None, params=None):
        self.hidden = hidden
        if params is not None:
            self.W_ih, self.W_hh, self.b_ih, self.b_hh, self.Wc, self.bc = [jnp.asarray(p) for p in params]
            return
        ks = jax.random.split(key, 3)
        lg = 1.0 / jnp.sqrt(hidden)
        self.W_ih = jax.random.uniform(ks[0], (3 * hidden, in_dim), minval=-lg, maxval=lg)
        self.W_hh = jax.random.uniform(ks[1], (3 * hidden, hidden), minval=-lg, maxval=lg)
        self.b_ih = jnp.zeros(3 * hidden)
        self.b_hh = jnp.zeros(3 * hidden)
        li = 1.0 / jnp.sqrt(hidden)
        self.Wc = jax.random.uniform(ks[2], (ctx_dim, hidden), minval=-li, maxval=li)
        self.bc = jnp.zeros(ctx_dim)

    def __call__(self, inp):                                      # (T,B,in) -> (T,B,ctx)
        B = inp.shape[1]

        def step(h, x):                                          # h (B,H), x (B,in)
            gi = x @ self.W_ih.T + self.b_ih                    # (B,3H)
            gh = h @ self.W_hh.T + self.b_hh
            ir, iz, in_ = jnp.split(gi, 3, -1)
            hr, hz, hn = jnp.split(gh, 3, -1)
            r = jax.nn.sigmoid(ir + hr)
            zg = jax.nn.sigmoid(iz + hz)
            n = jnp.tanh(in_ + r * hn)
            h_new = (1.0 - zg) * n + zg * h
            return h_new, h_new

        _, hs = jax.lax.scan(step, jnp.zeros((B, self.hidden), inp.dtype), inp)   # hs (T,B,H)
        return hs @ self.Wc.T + self.bc                          # (T,B,ctx)


class OperatorFilter(eqx.Module):
    encoder: GRUEncoder
    branch: MLP
    trunk: MLP
    bias: jax.Array
    reverse: bool = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)

    def __init__(self, data_size=1, gru_hidden=64, ctx_dim=64, p=64, branch_hidden=128,
                 trunk_hidden=64, trunk_layers=3, latent_dim=1, reverse=False, key=None, submodules=None):
        self.reverse = reverse
        self.latent_dim = latent_dim
        if submodules is not None:                               # weight-transfer/parity path
            self.encoder, self.branch, self.trunk, self.bias = submodules
            return
        k1, k2, k3 = jax.random.split(key, 3)
        self.encoder = GRUEncoder(data_size + 1, ctx_dim, gru_hidden, key=k1)
        self.branch = MLP([ctx_dim + 1, branch_hidden, p], k2)
        self.trunk = MLP([latent_dim] + [trunk_hidden] * trunk_layers + [p], k3)
        self.bias = jnp.zeros(())

    def context(self, xs, mask):                                 # (T,B,M),(T,B,1) -> (T,B,C)
        inp = jnp.concatenate([xs * mask, mask], axis=-1)
        if self.reverse:
            return self.encoder(inp[::-1])[::-1]                 # anti-causal: flip -> causal enc -> flip
        return self.encoder(inp)

    def coeffs(self, ctx, s):                                    # ctx (T,B,C), s (Ns,) -> (T,B,Ns,p)
        T, B, C = ctx.shape
        Ns = s.shape[0]
        ce = jnp.broadcast_to(ctx[:, :, None, :], (T, B, Ns, C))
        se = jnp.broadcast_to(s.reshape(1, 1, -1, 1), (T, B, Ns, 1))
        return self.branch(jnp.concatenate([ce, se], axis=-1))

    def coeffs_dtime(self, ctx, s):                             # (b, d_s b), each (T,B,Ns,p)
        T, B, C = ctx.shape
        Ns = s.shape[0]
        ce = jnp.broadcast_to(ctx[:, :, None, :], (T, B, Ns, C))
        se = jnp.broadcast_to(s.reshape(1, 1, -1, 1), (T, B, Ns, 1))
        inp = jnp.concatenate([ce, se], axis=-1)
        tan = jnp.zeros_like(inp).at[..., -1].set(1.0)          # unit tangent in the time channel
        return jax.jvp(self.branch, (inp,), (tan,))

    def log_density(self, ctx, z, s=0.0):                       # ctx (T,B,C), z (Nz,) or (Nz,d) -> (T,B,Nz)
        b = self.coeffs(ctx, jnp.asarray([s], dtype=z.dtype))[:, :, 0]   # (T,B,p)
        zt = z[..., None] if z.ndim == 1 else z
        return jnp.einsum("tbp,zp->tbz", b, self.trunk(zt)) + self.bias

    def log_posterior(self, xs, mask, z):                      # normalized filtering posterior (s=0)
        ell = self.log_density(self.context(xs, mask), z, 0.0)
        return ell - logsumexp(ell, axis=-1, keepdims=True)

    def trunk_zderivs(self, z):
        return trunk_zderivs(self.trunk, z)

    def trunk_grad(self, z):
        return trunk_grad(self.trunk, z)
