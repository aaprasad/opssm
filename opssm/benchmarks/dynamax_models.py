"""Dynamax KF/EKF/UKF and a differentiable IMM for Dynamax SLDS parameters.

The KF is a learned linear Gaussian LDS. EKF/UKF learn the same neural nonlinear
Gaussian SSM using Dynamax's public inference functions. Their transition is
Euler z'=z+dt*f(z), Q=dt*diag(g²), the conventional additive-noise discretization.
The SLDS uses Dynamax's linear filter for every regime's measurement update and
custom Gaussian mixture reduction, since Dynamax has no IMM fitting API.
"""
import jax
import jax.numpy as jnp
import jax.random as jr
from dynamax.linear_gaussian_ssm import lgssm_filter
from dynamax.linear_gaussian_ssm.inference import (
    ParamsLGSSM, ParamsLGSSMInitial, ParamsLGSSMDynamics, ParamsLGSSMEmissions,
)
from dynamax.nonlinear_gaussian_ssm import (
    ParamsNLGSSM, UKFHyperParams, extended_kalman_filter, unscented_kalman_filter,
)
from dynamax.slds import ParamsSLDS, DiscreteParamsSLDS, LGParamsSLDS

from .common import chol, decode, init_common, init_noise, obs_noise, positive


def linear_params(theta, noise, m, p, A, b, Q):
    """Public Dynamax parameter objects, including a shared affine emission."""
    return ParamsLGSSM(ParamsLGSSMInitial(m, p), ParamsLGSSMDynamics(A, b, None, Q),
                       ParamsLGSSMEmissions(theta["C"], theta["offset"], None,
                                           jnp.eye(theta["C"].shape[0]) * obs_noise(theta, noise) ** 2))


class DynamaxBaseline:
    def __init__(self, kind, d, dt, noise, hidden=64, modes=3, fixed_noise=False):
        self.kind, self.d, self.dt, self.noise = kind, d, dt, noise
        self.hidden, self.modes, self.fixed_noise = hidden, modes, fixed_noise

    def initialize(self, y, key):
        common_key, a_key, b_key = jr.split(key, 3)
        theta = init_common(y, self.d, self.hidden, common_key, nonlinear=self.kind in ("ekf", "ukf"))
        theta = init_noise(theta, self.noise, y.shape[-1], self.fixed_noise)
        if self.kind in ("kf", "slds"):
            theta.pop("raw_g")
            theta.update(A=.98 * jnp.eye(self.d), b=jnp.zeros(self.d), raw_q=jnp.full(self.d, -3.))
        if self.kind == "slds":
            K, d = self.modes, self.d
            theta.update(A=jnp.broadcast_to(theta["A"], (K, d, d)) + .02 * jr.normal(a_key, (K, d, d)),
                         b=.02 * jr.normal(b_key, (K, d)), raw_q=jnp.full((K, d), -3.),
                         m0=jnp.zeros((K, d)), raw_s0=jnp.zeros((K, d)),
                         initial_logits=jnp.zeros(K), transition_logits=3. * jnp.eye(K))
        return theta

    def params(self, th):
        P = jnp.diag(positive(th["raw_s0"]) ** 2)
        if self.kind == "kf":
            return linear_params(th, self.noise, th["m0"], P, th["A"], th["b"], jnp.diag(positive(th["raw_q"]) ** 2))
        return ParamsNLGSSM(th["m0"], P, lambda z: z + self.dt * th["drift"](z),
                           self.dt * jnp.diag(positive(th["raw_g"]) ** 2), lambda z: decode(th, z),
                           jnp.eye(th["C"].shape[0]) * obs_noise(th, self.noise) ** 2)

    def slds_params(self, th):
        K, d, n = self.modes, self.d, th["C"].shape[0]
        transition = jax.nn.softmax(th["transition_logits"], -1)
        return ParamsSLDS(
            DiscreteParamsSLDS(jax.nn.softmax(th["initial_logits"]), transition, transition),
            LGParamsSLDS(th["m0"], jax.vmap(jnp.diag)(positive(th["raw_s0"]) ** 2), th["A"],
                         jax.vmap(jnp.diag)(positive(th["raw_q"]) ** 2), th["b"], jnp.zeros((K, d, 1)),
                         jnp.broadcast_to(th["C"], (K, n, d)),
                         jnp.broadcast_to(jnp.eye(n) * obs_noise(th, self.noise) ** 2, (K, n, n)),
                         jnp.broadcast_to(th["offset"], (K, n)), jnp.zeros((K, n, 1)), True))

    def _imm(self, th, y):
        """One Gaussian per destination regime, including between-mode covariance."""
        params = self.slds_params(th)
        lg, discrete = params.linear_gaussian, params.discrete

        def step(carry, inputs):
            m, p, prob = carry
            t, obs = inputs

            def predict(carry):
                m, p, prob = carry
                joint = prob[:, None] * discrete.transition_matrix
                dest = joint.sum(0)
                weights = joint / jnp.maximum(dest[None], 1e-30)
                mixed_mean = jnp.einsum("ij,id->jd", weights, m)
                delta = m[:, None] - mixed_mean[None]
                mixed_cov = jnp.einsum("ij,ijde->jde", weights, p[:, None] + delta[..., :, None] * delta[..., None, :])
                return (jnp.einsum("kij,kj->ki", lg.dynamics_weights, mixed_mean) + lg.dynamics_bias,
                        lg.dynamics_weights @ mixed_cov @ jnp.swapaxes(lg.dynamics_weights, -1, -2) + lg.dynamics_cov, dest)

            m, p, prob = jax.lax.cond(t > 0, predict, lambda c: c, (m, p, prob))
            def condition(mm, pp):
                # Length-one public filter performs the Gaussian observation update
                # at t=0 without an extra transition before the first observation.
                pr = linear_params(th, self.noise, mm, pp, jnp.eye(self.d), jnp.zeros(self.d), jnp.eye(self.d))
                result = lgssm_filter(pr, obs[None])
                return result.filtered_means[0], result.filtered_covariances[0], result.marginal_loglik
            m, p, ll = jax.vmap(condition)(m, p)
            logw = jnp.log(jnp.maximum(prob, 1e-30)) + ll
            total = jax.scipy.special.logsumexp(logw)
            prob = jnp.exp(logw - total)
            mean = (prob[:, None] * m).sum(0)
            delta = m - mean
            cov = (prob[:, None, None] * (p + delta[..., :, None] * delta[..., None, :])).sum(0)
            return (m, p, prob), (mean, cov, total, prob)

        carry = (lg.initial_mean, lg.initial_cov, discrete.initial_distribution)
        (m, p, _), (means, covs, ll, probs) = jax.lax.scan(step, carry, (jnp.arange(len(y)), y))
        return dict(mean=means, cov=covs, loglik=ll.sum(), regime_prob=probs,
                    final_mode_mean=m, final_mode_cov=p)

    def one_filter(self, theta, y):
        if self.kind == "slds":
            return self._imm(theta, y)
        params = self.params(theta)
        if self.kind == "kf":
            post = lgssm_filter(params, y)
        elif self.kind == "ekf":
            post = extended_kalman_filter(params, y)
        else:
            post = unscented_kalman_filter(params, y, UKFHyperParams(alpha=1., beta=2., kappa=0.))
        return dict(mean=post.filtered_means, cov=post.filtered_covariances, loglik=post.marginal_loglik)

    def posterior(self, theta, y, ts, key, samples):
        post = jax.vmap(lambda obs: self.one_filter(theta, obs), in_axes=1)(y)
        # vmap gives B,T,...; final regime-specific moments have no time axis.
        return {k: (jnp.swapaxes(v, 0, 1) if k in ("mean", "cov", "regime_prob") else v) for k, v in post.items()}

    def loss(self, theta, y, ts, key):
        return -jax.vmap(lambda obs: self.one_filter(theta, obs)["loglik"], in_axes=1)(y).mean() / (len(y) * y.shape[-1])

    @property
    def dynamics_kind(self):
        return "observation_interval_effective_drift" if self.kind in ("kf", "slds") else "continuous_drift"

    def drift_values(self, theta, z, regime_prob=None):
        if self.kind == "kf":
            return (z @ theta["A"].T + theta["b"] - z) / self.dt
        if self.kind == "slds":
            if regime_prob is None or regime_prob.shape != (*z.shape[:-1], self.modes):
                raise ValueError("SLDS dynamics evaluation requires filtered regime probabilities at each query")
            # Regime j determines the NEXT transition, so advance weights with P.
            weights = regime_prob @ jax.nn.softmax(theta["transition_logits"], -1)
            next_z = jnp.einsum("kij,...j->...ki", theta["A"], z) + theta["b"]
            return (jnp.sum(weights[..., None] * next_z, axis=-2) - z) / self.dt
        return jax.vmap(theta["drift"])(z.reshape(-1, self.d)).reshape(z.shape)

    def forecast_samples(self, theta, post, times, key, samples):
        key, ki, ke = jr.split(key, 3)
        B, d = post["mean"].shape[1:]
        if self.kind == "slds":
            prob = post["regime_prob"][-1]
            modes = jr.categorical(ki, jnp.log(prob)[None], shape=(samples, B))
            batch = jnp.broadcast_to(jnp.arange(B), (samples, B))
            m, p = post["final_mode_mean"][batch, modes], post["final_mode_cov"][batch, modes]
        else:
            m, p = post["mean"][-1][None], post["cov"][-1][None]
            modes = jnp.zeros((samples, B), dtype=jnp.int32)
        z = m + jnp.einsum("...ij,...j->...i", chol(p), jr.normal(ke, (samples, B, d)))

        def step(carry, key):
            z, modes = carry
            k1, k2 = jr.split(key)
            eps = jr.normal(k2, z.shape)
            if self.kind == "slds":
                modes = jr.categorical(k1, theta["transition_logits"][modes])
                z = jnp.einsum("...ij,...j->...i", theta["A"][modes], z) + theta["b"][modes] + positive(theta["raw_q"][modes]) * eps
            elif self.kind == "kf":
                z = z @ theta["A"].T + theta["b"] + positive(theta["raw_q"]) * eps
            else:
                f = jax.vmap(jax.vmap(theta["drift"]))(z)
                z = z + self.dt * f + jnp.sqrt(self.dt) * positive(theta["raw_g"]) * eps
            return (z, modes), z
        return jax.lax.scan(step, (z, modes), jr.split(key, len(times) - 1))[1]
