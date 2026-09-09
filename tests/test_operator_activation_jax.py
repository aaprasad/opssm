"""JAX activation consistency and differentiable PINN training; run in the JAX environment.

Run: python -m unittest discover -s tests -p test_operator_activation_jax.py
"""
import importlib.util
import tempfile
import unittest

if any(importlib.util.find_spec(name) is None for name in ('jax', 'equinox', 'optax')):
    raise unittest.SkipTest('Run the JAX activation tests in the separate JAX environment')

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# These analytic identities use 1e-6 tolerances; GPU TF32 matmuls need full precision here.
jax.config.update('jax_default_matmul_precision', 'highest')

from opssm.models.jax.operator import OperatorFilter
from opssm.models.jax.losses import pinn_zakai_loss, sample_collocation


def operator(d=1, trunk_activation='softplus'):
    return OperatorFilter(data_size=d, latent_dim=d, gru_hidden=8, ctx_dim=8, p=8,
                          branch_hidden=8, trunk_hidden=8, trunk_layers=1,
                          trunk_activation=trunk_activation, key=jax.random.PRNGKey(4))


class OperatorActivationJaxTest(unittest.TestCase):
    def test_default_and_legacy_activation(self):
        model, legacy = operator(), operator(trunk_activation='tanh')
        self.assertEqual(model.trunk.activation, 'softplus')
        self.assertEqual(model.branch.activation, 'tanh')
        for a, b in zip(jax.tree_util.tree_leaves(model), jax.tree_util.tree_leaves(legacy)):
            np.testing.assert_array_equal(a, b)
        self.assertFalse(np.allclose(model.trunk(jnp.ones((1, 1))), legacy.trunk(jnp.ones((1, 1)))))
        with self.assertRaises(ValueError):
            operator(trunk_activation='invalid')

    def test_density_and_derivatives(self):
        for activation in ('softplus', 'tanh'):
            for d in (1, 3):
                model = operator(d, trunk_activation=activation)
                ctx = model.context(jnp.ones((4, 2, d)), jnp.ones((4, 2, 1)))
                z = jax.random.normal(jax.random.PRNGKey(2), (7, d))
                tau, grad, lap = model.trunk_zderivs(z)
                np.testing.assert_allclose(tau, model.trunk(z), atol=1e-6)
                self.assertEqual(tau.shape, (7, 8))
                np.testing.assert_allclose(model.trunk_grad(z)[1], grad, atol=1e-6)
                s = jnp.array([0., .4, 1.])
                b, db = model.coeffs_dtime(ctx, s)
                np.testing.assert_allclose(b, model.coeffs(ctx, s), atol=1e-6)
                np.testing.assert_allclose(jax.grad(lambda time: model.coeffs(ctx, time).sum())(s),
                                           db.sum((0, 1, 3)), atol=1e-6)
                ell = lambda point: model.log_density(ctx[:1, :1], point[None])[0, 0, 0]
                np.testing.assert_allclose(jax.vmap(jax.grad(ell))(z),
                                           jnp.einsum('p,zdp->zd', b[0, 0, 0], grad), atol=1e-6)
                np.testing.assert_allclose(jax.vmap(jax.hessian(ell))(z).trace(axis1=-2, axis2=-1),
                                           jnp.einsum('p,zp->z', b[0, 0, 0], lap), atol=1e-6)

    def test_softplus_unbounded_basis_and_pde_derivatives(self):
        model = operator(trunk_activation='softplus')
        w0 = jnp.zeros_like(model.trunk.weights[0]).at[0, 0].set(1.).at[1, 0].set(-1.)
        w1 = jnp.zeros_like(model.trunk.weights[1]).at[0, :2].set(-1.)
        model = eqx.tree_at(lambda op: (op.trunk.weights, op.trunk.biases), model,
                            ([w0, w1], [jnp.zeros_like(b) for b in model.trunk.biases]))
        z = jnp.array([[-100.], [-2.], [0.], [2.], [100.]])
        tau, grad, lap = model.trunk_zderivs(z)
        np.testing.assert_allclose(tau[:, 0], -jnp.logaddexp(z[:, 0], 0.)
                                   -jnp.logaddexp(-z[:, 0], 0.), atol=1e-6)
        np.testing.assert_allclose(grad[:, 0, 0], -jnp.tanh(z[:, 0]/2), atol=1e-6)
        np.testing.assert_allclose(lap[:, 0], -.5/jnp.cosh(z[:, 0]/2)**2, atol=1e-6)
        x, mask = jnp.ones((4, 2, 1)), jnp.ones((4, 2, 1))
        nodes, logq = sample_collocation(jax.random.PRNGKey(0), x, mask, 16, .3, 1.6)
        def loss(op):
            res, jump, ic, _ = pinn_zakai_loss(op, x, mask, nodes, logq, jnp.array([0., .5, 1.]),
                lambda z: (z-z**3, (1-3*z*z).sum(-1)), .6, lambda z: -(z*z).sum(-1)/2,
                .3, .1, res_mode='rel')
            return res+jump+ic
        value, grads = eqx.filter_jit(eqx.filter_value_and_grad(loss))(model)
        self.assertTrue(np.isfinite(value))
        self.assertTrue(all(np.isfinite(a).all() for a in jax.tree_util.tree_leaves(grads)))

    def test_checkpoint_preserves_activation(self):
        import pickle
        from pathlib import Path
        from opssm.models.jax.train import _save_ckpt, _load_ckpt
        with tempfile.TemporaryDirectory() as directory:
            model = operator(trunk_activation='softplus')
            _save_ckpt(directory, (model,), 3, .6, jax.random.PRNGKey(0), [])
            arrays, meta = _load_ckpt(directory, (model,))
            self.assertEqual(arrays[0].trunk.activation, 'softplus')
            self.assertEqual(meta['trunk_activation'], 'softplus')
            with self.assertRaisesRegex(ValueError, 'trunk_activation'):
                _load_ckpt(directory, (operator(trunk_activation='tanh'),))
            # Old checkpoints have no activation metadata and must retain tanh semantics.
            _save_ckpt(directory, (operator(trunk_activation='tanh'),), 3, .6, jax.random.PRNGKey(0), [])
            path = Path(directory) / 'ckpt.pkl'
            meta = pickle.loads(path.read_bytes())
            meta.pop('trunk_activation')
            path.write_bytes(pickle.dumps(meta))
            arrays, _ = _load_ckpt(directory, (operator(trunk_activation='tanh'),))
            self.assertEqual(arrays[0].trunk.activation, 'tanh')
            with self.assertRaisesRegex(ValueError, 'trunk_activation'):
                _load_ckpt(directory, (operator(),))
            meta['tail_std'] = 2.
            path.write_bytes(pickle.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'removed Gaussian tails'):
                _load_ckpt(directory, (operator(trunk_activation='tanh'),))


if __name__ == '__main__':
    unittest.main()
