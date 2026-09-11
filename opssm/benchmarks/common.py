"""JAX parameterizations shared by the baselines; time is in physical units."""
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


def positive(x):
    return jax.nn.softplus(x) + 1e-4


def chol(p):
    return jnp.linalg.cholesky((p + jnp.swapaxes(p, -1, -2)) / 2 + 1e-7 * jnp.eye(p.shape[-1]))


def network(d_in, d_out, hidden, key, zero_last=False):
    net = eqx.nn.MLP(d_in, d_out, hidden, depth=2, activation=jax.nn.tanh, key=key)
    if zero_last:
        net = eqx.tree_at(lambda n: (n.layers[-1].weight, n.layers[-1].bias), net,
                          (jnp.zeros_like(net.layers[-1].weight), jnp.zeros_like(net.layers[-1].bias)))
    return net


def init_common(y, d, hidden, key, nonlinear=True, state_diffusion=False):
    flat = y.reshape(-1, y.shape[-1])
    mean = flat.mean(0)
    _, _, vt = jnp.linalg.svd(flat - mean, full_matrices=False)
    theta = dict(C=vt[:d].T, offset=mean, m0=jnp.zeros(d), raw_s0=jnp.zeros(d),
                 raw_g=jnp.full(d, jnp.log(jnp.expm1(.3))))
    keys = jr.split(key, d + 1)
    if nonlinear:
        theta["drift"] = network(d, d, hidden, keys[0], zero_last=True)
        if state_diffusion:
            theta["g_nets"] = tuple(network(1, 1, hidden, keys[i + 1], zero_last=True) for i in range(d))
    return theta


def diffusion(theta, z):
    raw = theta["raw_g"]
    if "g_nets" in theta:
        raw = raw + jnp.concatenate([net(z[i:i + 1]) for i, net in enumerate(theta["g_nets"])])
    return jnp.broadcast_to(positive(raw), z.shape)


def decode(theta, z):
    return z @ theta["C"].T + theta["offset"]


def obs_noise(theta, fixed_noise):
    return positive(theta["raw_noise"]) if "raw_noise" in theta else fixed_noise


def init_noise(theta, noise, obs_dim, fixed):
    if not fixed:
        theta["raw_noise"] = jnp.full(obs_dim, jnp.log(jnp.expm1(jnp.maximum(noise - 1e-4, 1e-4))))
    return theta


def observation_nll(theta, y, z, noise):
    s = obs_noise(theta, noise)
    return (.5 * ((y - decode(theta, z)) / s) ** 2 + jnp.log(s) + .5 * jnp.log(2 * jnp.pi)).sum(-1)


def diagonal_kl(m, s, pm, ps):
    return (jnp.log(ps / s) + (s ** 2 + (m - pm) ** 2) / (2 * ps ** 2) - .5).sum(-1)
