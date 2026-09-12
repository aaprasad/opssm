"""Check recurrence, Gaussian mixture moments, and consistent generative dynamics."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("dynamax")
import jax.numpy as jnp
from opssm.benchmarks.dynamax_models import DynamaxBaseline


@pytest.fixture(autouse=True)
def double_precision():
    jax.config.update("jax_enable_x64", True)


def test_recurrent_probabilities_depend_on_state_and_previous_mode():
    model = DynamaxBaseline("rslds", 1, .1, .1, modes=2)
    th = dict(transition_logits=jnp.array([[2., 0.], [0., 2.]]),
              recurrent_weights=jnp.array([[-3.], [3.]]))
    p = np.exp(model.transition_log_probs(th, jnp.array([[-2.], [0.], [2.]])))
    np.testing.assert_allclose(p.sum(-1), 1.)
    assert (p[0, :, 0] > .999).all() and (p[2, :, 1] > .999).all()
    assert p[1, 0, 0] > .8 and p[1, 1, 1] > .8


@pytest.mark.parametrize("d", [1, 2, 3])
def test_zero_recurrence_reduces_to_markov_imm(d):
    y = jax.random.normal(jax.random.PRNGKey(1), (6, 2, d + 2))
    markov = DynamaxBaseline("slds", d, .1, .2, modes=3)
    recurrent = DynamaxBaseline("rslds", d, .1, .2, modes=3)
    th = markov.initialize(y, jax.random.PRNGKey(2))
    rs = dict(th, recurrent_weights=jnp.zeros((3, d)))
    one, two = markov.one_filter(th, y[:, 0]), recurrent.one_filter(rs, y[:, 0])
    for key in one:
        np.testing.assert_allclose(one[key], two[key], atol=1e-10, rtol=1e-9)


def test_single_recurrent_mode_reduces_to_kalman_filter():
    y = jax.random.normal(jax.random.PRNGKey(1), (6, 2, 3))
    kf = DynamaxBaseline("kf", 2, .1, .2)
    rslds = DynamaxBaseline("rslds", 2, .1, .2, modes=1)
    th = kf.initialize(y, jax.random.PRNGKey(2))
    rs = {**th, **{key: th[key][None] for key in ("A", "b", "raw_q", "m0", "raw_s0")},
          "initial_logits": jnp.zeros(1), "transition_logits": jnp.zeros((1, 1)),
          "recurrent_weights": jnp.array([[3., -4.]])}
    one, two = kf.one_filter(th, y[:, 0]), rslds.one_filter(rs, y[:, 0])
    for key in ("mean", "cov", "loglik"):
        np.testing.assert_allclose(one[key], two[key], atol=1e-10, rtol=1e-9)


def test_gate_conditioning_preserves_total_moments_and_separates_states():
    model = DynamaxBaseline("rslds", 1, .1, .1, modes=2)
    th = dict(transition_logits=jnp.zeros((2, 2)), recurrent_weights=jnp.array([[-2.], [2.]]))
    m, cov, prob = model._recurrent_mix(th, jnp.zeros((2, 1)),
                                       jnp.ones((2, 1, 1)), jnp.array([.3, .7]))
    np.testing.assert_allclose(prob, [.5, .5], atol=1e-12)
    assert m[0, 0] < -.9 and m[1, 0] > .9  # A mean-only gate incorrectly gives zero.
    np.testing.assert_allclose((prob[:, None] * m).sum(0), [0.], atol=1e-12)
    np.testing.assert_allclose((prob[:, None, None] * (cov + m[:, :, None] * m[:, None, :])).sum(0),
                               [[1.]], atol=1e-12)
    assert np.linalg.eigvalsh(cov).min() > 0


def test_recurrent_filter_is_causal_and_recurrence_has_likelihood_gradients():
    model = DynamaxBaseline("rslds", 2, .1, .2, modes=2)
    y = jax.random.normal(jax.random.PRNGKey(1), (6, 2, 4))
    th = model.initialize(y, jax.random.PRNGKey(2))
    th["recurrent_weights"] = jnp.array([[.6, -.4], [-.3, .5]])
    first = model.one_filter(th, y[:, 0])
    second = model.one_filter(th, y[:, 0].at[3:].add(100.))
    np.testing.assert_array_equal(first["mean"][:3], second["mean"][:3])
    np.testing.assert_allclose(first["regime_prob"].sum(-1), 1.)
    assert np.linalg.eigvalsh(np.asarray(first["cov"])).min() > 0
    loss, grad = jax.value_and_grad(model.loss)(th, y, jnp.arange(6) * .1, jax.random.PRNGKey(0))
    assert np.isfinite(loss) and all(np.isfinite(x).all() for x in jax.tree.leaves(grad))
    assert np.linalg.norm(grad["recurrent_weights"]) > 1e-8
    # Recurrence must not introduce an extra transition at the first observation.
    zero = model.one_filter(dict(th, recurrent_weights=jnp.zeros((2, 2))), y[:, 0])
    np.testing.assert_allclose(first["mean"][0], zero["mean"][0], atol=1e-12)
    assert not np.allclose(first["mean"][1:], zero["mean"][1:])


def test_forecast_and_drift_use_recurrence_at_current_continuous_state():
    model = DynamaxBaseline("rslds", 1, .1, .1, modes=2)
    th = dict(A=jnp.ones((2, 1, 1)), b=jnp.array([[-1.], [1.]]), raw_q=jnp.full((2, 1), -30.),
              transition_logits=jnp.zeros((2, 2)), recurrent_weights=jnp.array([[-10.], [10.]]))
    z = jnp.array([[[-2.], [2.]]])
    prob = jnp.array([[[1., 0.], [1., 0.]]])
    expected_drift = model.drift_values(th, z, prob)
    np.testing.assert_allclose(expected_drift, [[[-10.], [10.]]], atol=1e-10)
    post = dict(mean=z, cov=jnp.full((1, 2, 1, 1), 1e-12), regime_prob=prob,
                final_mode_mean=jnp.broadcast_to(z[0, :, None, :], (2, 2, 1)),
                final_mode_cov=jnp.full((2, 2, 1, 1), 1e-12))
    paths = model.forecast_samples(th, post, jnp.array([0., .1, .2]), jax.random.PRNGKey(3), 1024)
    np.testing.assert_allclose(paths[0].mean(0), z[0] + .1 * expected_drift[0], atol=.002)
    np.testing.assert_allclose(paths[1].mean(0), [[-4.], [4.]], atol=.002)
