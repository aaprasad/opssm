"""Mathematical and information-leakage checks for the shared benchmark suite."""
from dataclasses import replace

import numpy as np
import pytest

from opssm.benchmarks.data import PRESETS, drift, fingerprint, make_dataset, save_dataset, smoke_config
from opssm.benchmarks.metrics import fit_alignment, score_dynamics, score_posterior


def test_published_drift_parameters():
    np.testing.assert_allclose(drift(np.array([[.5]]), PRESETS["doublewell"]), [[1.5]])
    np.testing.assert_allclose(drift(np.array([[1., 0.]]), PRESETS["vanderpol"]), [[20., 7.5]])


def test_dataset_splits_and_preprocessing_do_not_leak(tmp_path):
    cfg = smoke_config(PRESETS["lorenz"])
    first = make_dataset(cfg, 42)
    bigger_test = make_dataset(replace(cfg, n_test=7), 42)
    for key in ("y_train", "y_val", "z_train", "obs_mean", "obs_scale", "C_true"):
        np.testing.assert_array_equal(first[key], bigger_test[key])
    assert fingerprint(first) == fingerprint(make_dataset(cfg, 42))
    assert not np.array_equal(first["z_train"][:, :2], first["z_test"])
    np.testing.assert_allclose(first["y_train"].mean((0, 1)), 0, atol=1e-7)
    assert first["ts"][-1] == cfg.duration
    path = tmp_path / "data.npz"
    save_dataset(path, first)
    save_dataset(path, first)
    with pytest.raises(FileExistsError):
        save_dataset(path, bigger_test)


def test_alignment_is_frozen_before_test_scoring():
    rng = np.random.default_rng(4)
    zval = rng.normal(size=(8, 2, 2))
    A = np.array([[2., .5], [-.3, 1.]])
    b = np.array([1., -2.])
    alignment = fit_alignment(zval, zval @ A + b)
    np.testing.assert_allclose(alignment, np.concatenate([A, b[None]]), atol=1e-10)
    ztest = rng.normal(size=(8, 2, 2))
    truth = ztest @ A + b + 3  # Test offset must NOT be fitted away.
    data = dict(z_test=truth, z_train=truth, signal_test=truth)
    result = dict(mean=ztest, cov=np.broadcast_to(np.eye(2), (8, 2, 2, 2)), reconstruction=truth)
    assert score_posterior(result, data, alignment)["latent_rmse"] == pytest.approx(3.)


def test_dynamics_rmse_uses_same_state_and_transforms_velocities():
    rng = np.random.default_rng(15)
    A = np.array([[2., .4], [-.3, .7]])  # Include shear/scale: transpose mistakes cannot cancel.
    b = np.array([1.2, -2.3])
    zval = rng.normal(size=(7, 2, 2))
    alignment = fit_alignment(zval, zval @ A + b)
    data = dict(z_train=rng.normal(size=(10, 3, 2)), z_test=rng.normal(size=(8, 2, 2)) + 2.)
    def true(x):
        return np.stack([x[..., 0] - x[..., 1] ** 2, x[..., 0] * x[..., 1] + .2], -1)
    def learned(z):
        return true(z @ A + b) @ np.linalg.inv(A)
    metrics, arrays = score_dynamics(learned, true, data, alignment)
    assert metrics["dynamics_rmse"] < 1e-12
    np.testing.assert_allclose(arrays["dynamics_query_model"] @ A + b, data["z_test"], atol=1e-12)
    np.testing.assert_allclose(arrays["dynamics_pred_aligned"], true(data["z_test"]), atol=1e-12)
    # A known model-coordinate velocity error must transform with A, with NO +b.
    bias = np.array([.3, -.2])
    metrics, _ = score_dynamics(lambda z: learned(z) + bias, true, data, alignment)
    expected = np.sqrt(np.mean((bias @ A) ** 2))
    assert metrics["dynamics_rmse"] == pytest.approx(expected)
    assert metrics["dynamics_nrmse"] == pytest.approx(expected / np.sqrt(np.mean(true(data["z_train"]) ** 2)))


def test_dynamics_rmse_rejects_collapsed_alignment():
    data = dict(z_test=np.ones((3, 2, 2)), z_train=np.ones((3, 2, 2)))
    def forbidden(_):
        raise AssertionError("Do not evaluate a vector field through a singular coordinate map")
    metrics, arrays = score_dynamics(forbidden, forbidden, data, np.zeros((3, 2)))
    assert metrics["dynamics_rmse"] is None
    assert metrics["dynamics_status"] == "noninvertible_or_ill_conditioned_alignment"
    assert arrays == {}


@pytest.fixture(scope="module")
def jax_data():
    jax = pytest.importorskip("jax")
    pytest.importorskip("dynamax")
    pytest.importorskip("diffrax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    cfg = replace(smoke_config(PRESETS["doublewell"]), num_obs=6, duration=.05)
    data = make_dataset(cfg, 0)
    return jnp.asarray(data["y_train"], dtype=jnp.float64), jnp.asarray(data["ts"]), float(data["noise_std_eff"])


def test_slds_single_mode_reduces_to_dynamax_kf(jax_data):
    import jax
    import jax.numpy as jnp
    from opssm.benchmarks.dynamax_models import DynamaxBaseline
    y, ts, noise = jax_data
    kf = DynamaxBaseline("kf", 1, .01, noise, hidden=4)
    slds = DynamaxBaseline("slds", 1, .01, noise, hidden=4, modes=1)
    th = kf.initialize(y, jax.random.PRNGKey(1))
    switching = {**th, **{key: th[key][None] for key in ("A", "b", "raw_q", "m0", "raw_s0")},
                 "initial_logits": jnp.zeros(1), "transition_logits": jnp.zeros((1, 1))}
    one = kf.one_filter(th, y[:, 0])
    two = slds.one_filter(switching, y[:, 0])
    for key in ("mean", "cov", "loglik"):
        np.testing.assert_allclose(one[key], two[key], atol=1e-10)


def test_discrete_dynamics_use_physical_time_and_next_regime_weights(jax_data):
    import jax.numpy as jnp
    from opssm.benchmarks.dynamax_models import DynamaxBaseline
    z = jnp.array([[[2.]]])
    kf = DynamaxBaseline("kf", 1, .1, .1)
    th = dict(A=jnp.array([[1.5]]), b=jnp.array([.2]))
    np.testing.assert_allclose(kf.drift_values(th, z), [[[12.]]])
    slds = DynamaxBaseline("slds", 1, .1, .1, modes=2)
    th = dict(A=jnp.ones((2, 1, 1)), b=jnp.array([[.1], [.3]]),
              transition_logits=jnp.array([[-100., 100.], [100., -100.]]))
    np.testing.assert_allclose(slds.drift_values(th, z, jnp.array([[[1., 0.]]])), [[[3.]]])
    assert slds.dynamics_kind == "observation_interval_effective_drift"


@pytest.mark.parametrize("kind", ["kf", "ekf", "ukf", "slds"])
def test_dynamax_filters_are_causal_with_finite_gradients(kind, jax_data):
    import equinox as eqx
    import jax
    from opssm.benchmarks.dynamax_models import DynamaxBaseline
    y, ts, noise = jax_data
    model = DynamaxBaseline(kind, 1, .01, noise, hidden=4, modes=2)
    key = jax.random.PRNGKey(3)
    th = model.initialize(y, key)
    first = model.one_filter(th, y[:, 0])
    second = model.one_filter(th, y[:, 0].at[3:].add(100.))
    np.testing.assert_array_equal(first["mean"][:3], second["mean"][:3])
    assert np.linalg.eigvalsh(np.asarray(first["cov"])).min() > 0
    loss, grad = eqx.filter_value_and_grad(model.loss)(th, y, ts, key)
    assert np.isfinite(float(loss))
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grad))


def test_matching_gaussian_drift_and_physical_time_weights(jax_data):
    import jax.numpy as jnp
    from opssm.benchmarks.sde_models import gaussian_posterior_drift, matching_weights
    # Stationary N(0,s²) with diffusion g has OU drift -g² z/(2s²).
    s, g, eps = jnp.array([2.]), jnp.array([.7]), jnp.array([1.3])
    z = s * eps
    drift = gaussian_posterior_drift(0., s, 0., 0., eps, g ** 2, 0.)
    np.testing.assert_allclose(drift, -g ** 2 * z / (2 * s ** 2))
    # Initial KL is NOT multiplied by time; likelihood estimates a sum, not a mean.
    assert float(matching_weights(2., 3., 5., 7., 11)) == 78.


def test_matching_drift_satisfies_fokker_planck_for_changing_gaussian(jax_data):
    """Independently check the density evolution, including nonzero div(GGᵀ).

    A stationary OU example cannot detect missing mean/scale derivatives or
    a missing state-dependent diffusion correction; this case exercises all three.
    """
    import jax
    import jax.numpy as jnp
    from opssm.benchmarks.sde_models import gaussian_posterior_drift

    def moments(t):
        m = jnp.array([.3, -.4]) + jnp.array([.7, -.2]) * t + .1 * t ** 2
        s = jnp.exp(jnp.array([.2, -.3]) * t) + .2
        return m, s

    def density(t, z):
        m, s = moments(t)
        return jnp.exp(-.5 * jnp.sum(((z - m) / s) ** 2)) / (2 * jnp.pi * jnp.prod(s))

    def diffusion_variance(z):
        return (jnp.array([.5, .8]) + jnp.array([.1, .2]) * z ** 2) ** 2

    def posterior_drift(t, z):
        (m, s), (dm, ds) = jax.jvp(moments, (t,), (jnp.ones_like(t),))
        div = jnp.diag(jax.jacfwd(diffusion_variance)(z))
        return gaussian_posterior_drift(m, s, dm, ds, (z - m) / s, diffusion_variance(z), div)

    for time, location in [(.4, [.6, -.2]), (1.3, [-.5, .7])]:
        t, z = jnp.asarray(time), jnp.asarray(location)
        density_dt = jax.grad(density, argnums=0)(t, z)
        advective = -jnp.trace(jax.jacfwd(lambda x: posterior_drift(t, x) * density(t, x))(z))
        diffusive = .5 * sum(jax.hessian(lambda x: diffusion_variance(x)[i] * density(t, x))(z)[i, i]
                             for i in range(2))
        np.testing.assert_allclose(density_dt, advective + diffusive, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("kind", ["latent_sde", "sde_matching"])
def test_sde_loss_gradients_and_forecasts_do_not_read_future(kind, jax_data):
    import equinox as eqx
    import jax
    from opssm.benchmarks.sde_models import SDEBaseline
    from opssm.benchmarks.train import forecast
    y, ts, noise = jax_data
    y = y[:, :1]
    key = jax.random.PRNGKey(3)
    model = SDEBaseline(kind, 1, noise, hidden=4, solver_dt=.01, matching_points=2)
    th = model.initialize(y, key)
    loss, grad = eqx.filter_jit(eqx.filter_value_and_grad(model.loss))(th, y, ts, key)
    assert np.isfinite(float(loss))
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grad))
    assert sum(float(np.sum(np.asarray(x) ** 2)) for x in jax.tree.leaves(grad["drift"])) > 0
    f = eqx.filter_jit(lambda obs: forecast(model, th, obs, ts, key, 4))
    a = f(y)
    b = f(y.at[len(ts) // 2:].add(100.))
    np.testing.assert_array_equal(a["forecast_mean"], b["forecast_mean"])
    assert not np.allclose(a["forecast_loglik"], b["forecast_loglik"])


def test_matching_training_never_calls_solver(jax_data, monkeypatch):
    import jax
    import diffrax
    from opssm.benchmarks.sde_models import SDEBaseline
    y, ts, noise = jax_data
    def forbidden(*args, **kwargs):
        raise AssertionError("Matching training must be simulation-free")
    monkeypatch.setattr(diffrax, "diffeqsolve", forbidden)
    model = SDEBaseline("sde_matching", 1, noise, hidden=4, state_diffusion=True)
    th = model.initialize(y, jax.random.PRNGKey(0))
    assert np.isfinite(float(model.loss(th, y, ts, jax.random.PRNGKey(1))))
