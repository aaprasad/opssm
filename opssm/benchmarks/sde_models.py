"""Diffrax latent SDE and simulation-free Gaussian SDE Matching in JAX.

The two methods share prior, diffusion, emission, and GRU size. Native inference
uses future observations and is always labelled smoothing. Training never sees
true states. Matching samples marginals without solving an SDE; Diffrax is used
for simulation-based training and prior forecasts for both methods.
"""
import math

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from .common import (chol, diagonal_kl, diffusion, init_common, init_noise, network,
                     observation_nll, positive)


def gaussian_posterior_drift(m, s, dm, ds, eps, g2, divergence):
    return dm + ds * eps + .5 * divergence - .5 * g2 * eps / s


def matching_weights(initial_kl, path_integrand, sampled_nll, duration, n_obs):
    """Unbiased time-integral and observation-SUM estimators (physical units)."""
    return initial_kl + duration * path_integrand + n_obs * sampled_nll


def integration_grid(ts, n_sub):
    fractions = jnp.arange(n_sub) / n_sub
    inner = ts[:-1, None] + jnp.diff(ts)[:, None] * fractions[None]
    return jnp.concatenate([inner.reshape(-1), ts[-1:]])


def solve_prior(drift, diffusion_fn, z0, ts, key, solver_dt, n_sub=1):
    """Ito Euler-Maruyama solve; step grid hits every requested observation time."""
    bm = diffrax.VirtualBrownianTree(ts[0], ts[-1], tol=solver_dt / 10,
                                    shape=z0.shape, key=key)
    def vol(t, z, args):
        g = diffusion_fn(z)
        return jnp.diag(g) if g.ndim == 1 else g
    terms = diffrax.MultiTerm(diffrax.ODETerm(lambda t, z, args: drift(z)),
                             diffrax.ControlTerm(vol, bm))
    grid = integration_grid(ts, n_sub)
    solution = diffrax.diffeqsolve(terms, diffrax.Euler(), t0=ts[0], t1=ts[-1], dt0=None,
                                 y0=z0, saveat=diffrax.SaveAt(ts=ts),
                                 stepsize_controller=diffrax.StepTo(ts=grid),
                                 max_steps=len(grid), adjoint=diffrax.RecursiveCheckpointAdjoint())
    return solution.ys


class SDEBaseline:
    dynamics_kind = "continuous_drift"

    def __init__(self, kind, d, noise, hidden=64, solver_dt=.01, fixed_noise=False,
                 state_diffusion=False, matching_points=4, observation_dt=.01):
        self.kind, self.d, self.noise = kind, d, noise
        self.hidden, self.solver_dt, self.fixed_noise = hidden, solver_dt, fixed_noise
        self.state_diffusion, self.matching_points = state_diffusion, matching_points
        self.n_sub = max(1, math.ceil(observation_dt / solver_dt - 1e-8))

    def initialize(self, y, key):
        kp, kg, ki, kc = jr.split(key, 4)
        theta = init_common(y, self.d, self.hidden, kp, state_diffusion=self.state_diffusion)
        theta = init_noise(theta, self.noise, y.shape[-1], self.fixed_noise)
        theta["gru"] = eqx.nn.GRUCell(y.shape[-1], self.hidden, key=kg)
        if self.kind == "latent_sde":
            theta["initial"] = eqx.nn.Linear(self.hidden, 2 * self.d, key=ki)
            theta["control"] = network(self.d + self.hidden, self.d, self.hidden, kc)
        else:
            theta["marginal"] = network(2 * self.hidden + 1, 2 * self.d, self.hidden, kc)
        return theta

    def encode(self, theta, y):
        def step(h, x):
            h = theta["gru"](x, h)
            return h, h
        return jax.lax.scan(step, jnp.zeros(self.hidden), y, reverse=True)[1]

    def marginal(self, theta, ctx, ts, t):
        # Smooth attention over fixed context supports time JVPs at arbitrary t.
        width = (ts[-1] - ts[0]) / (len(ts) - 1)
        weights = jax.nn.softmax(-.5 * ((ts - t) / width) ** 2)
        local = jnp.sum(weights[:, None] * ctx, axis=0)
        out = theta["marginal"](jnp.concatenate([ctx[0], local, jnp.atleast_1d(t)]))
        m, raw_s = jnp.split(out, 2)
        return m, positive(raw_s)

    def initial(self, theta, ctx, ts):
        if self.kind == "sde_matching":
            return self.marginal(theta, ctx, ts, ts[0])
        m, raw_s = jnp.split(theta["initial"](ctx[0]), 2)
        return m, positive(raw_s)

    def posterior_path(self, theta, ctx, ts, key):
        """Augment the Diffrax solve with the Girsanov energy integral."""
        ki, kb = jr.split(key)
        m, s = self.initial(theta, ctx, ts)
        z0 = m + s * jr.normal(ki, (self.d,))
        state0 = jnp.concatenate([z0, jnp.zeros(1)])

        def drift(t, state, args):
            z = state[:-1]
            # Future context in each interval; its jumps occur at observation times.
            index = jnp.clip(jnp.searchsorted(ts, t, side="right"), 0, len(ts) - 1)
            u = theta["control"](jnp.concatenate([z, ctx[index]]))
            f = theta["drift"](z) + diffusion(theta, z) * u
            return jnp.concatenate([f, jnp.atleast_1d(.5 * jnp.sum(u ** 2))])

        def vol(t, state, args):
            return jnp.concatenate([jnp.diag(diffusion(theta, state[:-1])), jnp.zeros((1, self.d))])

        bm = diffrax.VirtualBrownianTree(ts[0], ts[-1], tol=self.solver_dt / 10,
                                        shape=(self.d,), key=kb)
        terms = diffrax.MultiTerm(diffrax.ODETerm(drift), diffrax.ControlTerm(vol, bm))
        grid = integration_grid(ts, self.n_sub)
        sol = diffrax.diffeqsolve(terms, diffrax.Euler(), t0=ts[0], t1=ts[-1], dt0=None,
                                 y0=state0, saveat=diffrax.SaveAt(ts=ts),
                                 stepsize_controller=diffrax.StepTo(ts=grid),
                                 max_steps=len(grid), adjoint=diffrax.RecursiveCheckpointAdjoint())
        kl0 = diagonal_kl(m, s, theta["m0"], positive(theta["raw_s0"]))
        return sol.ys[:, :-1], sol.ys[-1, -1] + kl0

    def matching_integrand(self, theta, ctx, ts, t, eps):
        (m, s), (dm, ds) = jax.jvp(lambda time: self.marginal(theta, ctx, ts, time),
                                   (t,), (jnp.ones_like(t),))
        z = m + s * eps
        g2 = diffusion(theta, z) ** 2
        # Coordinate-wise diagonal diffusion => divergence_i = d(g_i²)/dz_i.
        divergence = (jax.grad(lambda zz: jnp.sum(diffusion(theta, zz) ** 2))(z)
                      if self.state_diffusion else jnp.zeros_like(z))
        qdrift = gaussian_posterior_drift(m, s, dm, ds, eps, g2, divergence)
        return .5 * jnp.sum((qdrift - theta["drift"](z)) ** 2 / g2)

    def one_loss(self, theta, y, ts, key):
        ctx = self.encode(theta, y)
        if self.kind == "latent_sde":
            path, kl = self.posterior_path(theta, ctx, ts, key)
            return (observation_nll(theta, y, path, self.noise).sum() + kl) / y.size
        kt, ke, ko, kr = jr.split(key, 4)
        m0, s0 = self.initial(theta, ctx, ts)
        kl0 = diagonal_kl(m0, s0, theta["m0"], positive(theta["raw_s0"]))
        duration = ts[-1] - ts[0]
        t = ts[0] + duration * jr.uniform(kt, (self.matching_points,))
        eps = jr.normal(ke, (self.matching_points, self.d))
        energy = jax.vmap(lambda t, eps: self.matching_integrand(theta, ctx, ts, t, eps))(t, eps).mean()
        idx = jr.randint(ko, (), 0, len(ts))
        m, s = self.marginal(theta, ctx, ts, ts[idx])
        nll = observation_nll(theta, y[idx], m + s * jr.normal(kr, (self.d,)), self.noise)
        return matching_weights(kl0, energy, nll, duration, len(ts)) / y.size

    def loss(self, theta, y, ts, key):
        return jax.vmap(lambda obs, k: self.one_loss(theta, obs, ts, k), in_axes=(1, 0))(
            y, jr.split(key, y.shape[1])).mean()

    def drift_values(self, theta, z, regime_prob=None):
        # Score the generative PRIOR drift, never the observation-conditioned control.
        return jax.vmap(theta["drift"])(z.reshape(-1, self.d)).reshape(z.shape)

    def posterior(self, theta, y, ts, key, samples):
        def one(y, key):
            ctx = self.encode(theta, y)
            if self.kind == "sde_matching":
                m, s = jax.vmap(lambda t: self.marginal(theta, ctx, ts, t))(ts)
                return dict(mean=m, cov=jax.vmap(jnp.diag)(s ** 2))
            paths = jax.vmap(lambda k: self.posterior_path(theta, ctx, ts, k)[0])(jr.split(key, samples))
            mean = paths.mean(0)
            delta = paths - mean[None]
            cov = jnp.einsum("sti,stj->tij", delta, delta) / (samples - 1)
            return dict(mean=mean, cov=cov, final_samples=paths[:, -1])
        return jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1),
                            jax.vmap(one, in_axes=(1, 0))(y, jr.split(key, y.shape[1])))

    def forecast_samples(self, theta, post, times, key, samples):
        ki, kp = jr.split(key)
        m, p = post["mean"][-1], post["cov"][-1]
        z0 = (post["final_samples"] if "final_samples" in post else
              m[None] + jnp.einsum("bij,sbj->sbi", chol(p), jr.normal(ki, (samples, *m.shape))))
        def one(z, k):
            return solve_prior(theta["drift"], lambda z: diffusion(theta, z), z, times, k, self.solver_dt, self.n_sub)[1:]
        keys = jr.split(kp, samples * m.shape[0]).reshape(samples, m.shape[0], 2)
        paths = jax.vmap(jax.vmap(one))(z0, keys)  # S,B,T,d
        return jnp.transpose(paths, (2, 0, 1, 3))
