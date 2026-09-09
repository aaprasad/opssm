"""JAX tail consistency and differentiable PINN training; run in the JAX environment.

Run: python -m unittest discover -s tests -p test_density_tails_jax.py
"""
import math
import importlib.util
import tempfile
import unittest

if any(importlib.util.find_spec(name) is None for name in ('jax', 'equinox', 'optax')):
    raise unittest.SkipTest('Run the JAX tail tests in the separate JAX environment')

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from opssm.models.jax.operator import OperatorFilter
from opssm.models.jax.losses import pinn_zakai_loss, sample_collocation


def operator(d=1, tail_std=0.0):
    return OperatorFilter(data_size=d, latent_dim=d, gru_hidden=8, ctx_dim=8, p=8,
                          branch_hidden=8, trunk_hidden=8, trunk_layers=1,
                          tail_std=tail_std, key=jax.random.PRNGKey(4))


class DensityTailsJaxTest(unittest.TestCase):
    def test_density_and_derivatives(self):
        for d in (1, 3):
            with self.subTest(d=d):
                old, new = operator(d), operator(d, 2.0)
                x, mask = jnp.ones((4, 2, d)), jnp.ones((4, 2, 1))
                ctx = new.context(x, mask)
                z = jax.random.normal(jax.random.PRNGKey(2), (7, d))
                for s in (0.0, 0.4, 1.0):
                    np.testing.assert_allclose(new.log_density(ctx, z, s),
                                               old.log_density(ctx, z, s) - (z*z).sum(-1)/8, atol=1e-6)
                tau, grad, lap = new.trunk_zderivs(z)
                np.testing.assert_allclose(tau, new.state_basis(z), atol=1e-6)
                np.testing.assert_allclose(grad[..., -1], -z/4, atol=1e-6)
                np.testing.assert_allclose(lap[..., -1], -d/4, atol=1e-6)
                np.testing.assert_allclose(new.trunk_grad(z)[1], grad, atol=1e-6)
                s = jnp.array([0., .4, 1.])
                b, db = new.coeffs_dtime(ctx, s)
                np.testing.assert_allclose(b, new.coeffs(ctx, s), atol=1e-6)
                np.testing.assert_array_equal(b[..., -1], 1.)
                np.testing.assert_array_equal(db[..., -1], 0.)
                # End-to-end derivatives, independent of the factorized helper.
                ell = lambda point: new.log_density(ctx[:1, :1], point[None])[0, 0, 0]
                np.testing.assert_allclose(jax.vmap(jax.grad(ell))(z),
                                           jnp.einsum('p,zdp->zd', b[0, 0, 0], grad), atol=1e-6)
                np.testing.assert_allclose(jax.vmap(jax.hessian(ell))(z).trace(axis1=-2, axis2=-1),
                                           jnp.einsum('p,zp->z', b[0, 0, 0], lap), atol=1e-6)

    def test_integral_and_training_gradient(self):
        model = operator(tail_std=1.3)
        gaussian = jax.tree_util.tree_map(lambda a: jnp.zeros_like(a) if eqx.is_array(a) else a, model)
        x, mask = jnp.ones((4, 2, 1)), jnp.ones((4, 2, 1))
        z = jnp.linspace(-12., 12., 4001)
        mass = jnp.trapezoid(jnp.exp(gaussian.log_density(gaussian.context(x, mask), z)), z)
        np.testing.assert_allclose(mass, math.sqrt(2*math.pi)*1.3, rtol=1e-5)
        nodes, logq = sample_collocation(jax.random.PRNGKey(0), x, mask, 16, .3, 1.6)
        def loss(op):
            res, jump, ic, _ = pinn_zakai_loss(op, x, mask, nodes, logq, jnp.array([0., .5, 1.]),
                lambda z: (z-z**3, (1-3*z*z).sum(-1)), .6, lambda z: -(z*z).sum(-1)/2,
                .3, .1, res_mode='rel')
            return res+jump+ic
        value, grad = eqx.filter_jit(eqx.filter_value_and_grad(loss))(model)
        self.assertTrue(np.isfinite(value))
        self.assertTrue(all(np.isfinite(a).all() for a in jax.tree_util.tree_leaves(grad)))

    def test_invalid_tail(self):
        for value in (-1., float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                operator(tail_std=value)

    def test_checkpoint_preserves_tail_semantics(self):
        from opssm.models.jax.train import _save_ckpt, _load_ckpt
        model = operator(tail_std=2.)
        with tempfile.TemporaryDirectory() as directory:
            _save_ckpt(directory, (model,), 3, .6, jax.random.PRNGKey(0), [])
            arrays, meta = _load_ckpt(directory, (model,))
            self.assertEqual(meta['tail_std'], 2.)
            self.assertEqual(arrays[0].tail_std, 2.)
            with self.assertRaisesRegex(ValueError, 'tail_std'):
                _load_ckpt(directory, (operator(),))


if __name__ == '__main__':
    unittest.main()
