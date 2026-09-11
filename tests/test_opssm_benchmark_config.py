"""Experiment inheritance and exact memory-bounded OPSSM gradient accumulation."""
import numpy as np
import pytest


@pytest.mark.parametrize("system,experiment,points,means", [
    ("doublewell", "duncker_dw", 128, 256), ("vanderpol", "duncker_vdp", 256, 384),
    ("lorenz", "duncker_lorenz", 384, 512), ("kato", "kato", 128, 256)])
def test_matching_experiment_is_inherited(system, experiment, points, means):
    pytest.importorskip("hydra")
    from opssm.benchmarks.opssm_config import resolve_opssm_config
    hp, source = resolve_opssm_config(system)
    assert source["experiment"] == experiment
    assert hp["n_colloc"] == points and hp["n_mean"] == means
    assert hp["diffusion_cov"] is False
    assert hp["chunk_size"] == 16
    assert hp["init_method"] == "subspace" and hp["init_dynamics"] is True
    assert hp["m_every"] == 2000 and hp["m_inner"] == 400
    assert len(source["source_sha256"]) == 3
    with pytest.raises(ValueError, match="does not implement"):
        resolve_opssm_config(system, overrides={"joint_g": True})


@pytest.mark.parametrize("chunk_size", [2, 4])
def test_chunked_gradient_matches_full_batch_with_uneven_tail(chunk_size):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import equinox as eqx
    from opssm.models.jax.operator import OperatorFilter
    from opssm.models.jax.dynamics import DriftNet
    from opssm.models.jax.losses import sample_collocation, pinn_zakai_value_and_grad
    jax.config.update("jax_enable_x64", True)
    op = OperatorFilter(2, 3, 3, 3, branch_hidden=3, trunk_hidden=3, trunk_layers=1,
                        latent_dim=2, key=jax.random.PRNGKey(0))
    drift = DriftNet(3, layers=1, latent_dim=2, key=jax.random.PRNGKey(1))
    x = jax.random.normal(jax.random.PRNGKey(2), (4, 5, 2))
    mask = jnp.ones((4, 5, 1))
    z, q = sample_collocation(jax.random.PRNGKey(3), x, mask, 6, .3, 1.6)
    def evaluate(size):
        return pinn_zakai_value_and_grad(op, x, mask, z, q, jnp.array([0., 1.]),
            drift.drift, .4, lambda z: -.5 * (z ** 2).sum(-1), .1, .02, jnp.array([1, 3]),
            w_res=.4, res_mode="rel", decode=None, chunk_size=size)
    expected = eqx.filter_jit(evaluate)(None)
    actual = eqx.filter_jit(evaluate)(chunk_size)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)


def test_chunked_mala_keeps_every_trajectory_and_weighted_acceptance():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from opssm.models.jax.operator import OperatorFilter
    from opssm.models.jax.mstep import _mala_chains
    jax.config.update("jax_enable_x64", True)
    op = OperatorFilter(2, 3, 3, 3, branch_hidden=3, trunk_hidden=3, trunk_layers=1,
                        latent_dim=2, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 5, 2))
    mask = jnp.ones((3, 5, 1)); key = jax.random.PRNGKey(2)
    actual = _mala_chains(op, x, mask, x, 4, 4, 1.6, key, chunk_size=2)
    parts = [_mala_chains(op, x[:, i:i+2], mask[:, i:i+2], x[:, i:i+2], 4, 4, 1.6,
                         jax.random.fold_in(key, i)) for i in range(0, 5, 2)]
    np.testing.assert_allclose(actual[0], np.concatenate([p[0] for p in parts], axis=1), atol=1e-9)
    np.testing.assert_allclose(actual[1], np.concatenate([p[1] for p in parts], axis=1), atol=1e-9)
    assert actual[2] == pytest.approx(sum(p[2] * p[0].shape[1] / 5 for p in parts))
