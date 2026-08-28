"""JAX/equinox NN primitive -- mirror of opssm.models.nn (torch).

Batch-native tanh MLP: Linear-Tanh-...-Linear with NO activation on the last layer, exactly like
`opssm.models.nn.mlp`. Params are stored as weight/bias lists (a clean pytree for optax + trivial
torch weight-transfer via the `weights=`/`biases=` constructor path used by the parity harness).
Weight convention matches torch nn.Linear: W is (out, in), and forward is `x @ W.T + b`.
"""
import jax
import jax.numpy as jnp
import equinox as eqx


class MLP(eqx.Module):
    weights: list
    biases: list

    def __init__(self, sizes=None, key=None, zero_last=False, weights=None, biases=None):
        if weights is not None:                                   # build from given arrays (weight-transfer/parity)
            self.weights = [jnp.asarray(w) for w in weights]
            self.biases = [jnp.asarray(b) for b in biases]
            return
        ks = jax.random.split(key, len(sizes) - 1)
        ws, bs = [], []
        for i, k in enumerate(ks):                                # torch nn.Linear init: U(-1/sqrt(in), 1/sqrt(in))
            lim = 1.0 / jnp.sqrt(sizes[i])
            ws.append(jax.random.uniform(k, (sizes[i + 1], sizes[i]), minval=-lim, maxval=lim))
            bs.append(jax.random.uniform(jax.random.fold_in(k, 1), (sizes[i + 1],), minval=-lim, maxval=lim))
        if zero_last:                                             # DriftNet/DiffusionNet zero the last layer at init
            ws[-1] = jnp.zeros_like(ws[-1])
            bs[-1] = jnp.zeros_like(bs[-1])
        self.weights, self.biases = ws, bs

    def __call__(self, x):                                        # (...,in) -> (...,out)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ W.T + b
            if i < len(self.weights) - 1:
                x = jnp.tanh(x)
        return x
