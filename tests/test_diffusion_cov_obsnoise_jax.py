"""JAX mirror of tests/test_diffusion_cov_obsnoise.py -- matrix diffusion + learned observation noise.

Same load-bearing claims, checked against the JAX implementations: the weighted Hessian trace is exact,
L = g I / an equal-entry diagonal R reduce the loss EXACTLY to the isotropic scalar path, and the
closed-form M-steps recover a known Sigma and a known R.
"""
import math
import unittest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from opssm.models.jax.operator import OperatorFilter                    # noqa: E402
from opssm.models.jax.dynamics import DriftNet                          # noqa: E402
from opssm.models.jax.losses import pinn_zakai_loss, sample_collocation  # noqa: E402
from opssm.models.jax.mstep import fit_diffusion_cov, fit_obs_noise, chol_summary  # noqa: E402


def _setup(d, D=4, T=5, B=2, K=8):
    k = jax.random.split(jax.random.PRNGKey(0), 6)
    op = OperatorFilter(data_size=D, latent_dim=d, p=6, trunk_hidden=8, trunk_layers=2,
                        gru_hidden=8, ctx_dim=8, branch_hidden=8, key=k[0])
    dn = DriftNet(8, layers=2, latent_dim=d, key=k[1])
    xs = jax.random.normal(k[2], (T, B, D))
    mask = jnp.ones((T, B, 1))
    C = jnp.linalg.qr(jax.random.normal(k[3], (D, d)))[0]
    off = jax.random.normal(k[4], (D,))
    decode = lambda z: jnp.einsum("od,...d->...o", C, z) + off          # noqa: E731
    ctr = jnp.einsum("od,tbo->tbd", C, xs - off)
    z_col, log_q = sample_collocation(k[5], xs, mask, K, 0.3, 1.6, ctr)
    return op, dn, xs, mask, z_col, log_q, decode, C, off


class TestWeightedHessianTrace(unittest.TestCase):
    def test_matches_explicit_hessian_and_reduces(self):
        for d in (1, 2, 3):
            op, *_ = _setup(d)
            z = jax.random.normal(jax.random.PRNGKey(7), (6, d))
            L = jnp.tril(jax.random.normal(jax.random.PRNGKey(d), (d, d))) + jnp.eye(d) * 1.5
            _, grad, sec = op.trunk_zderivs_dirs(z, L.T)
            H = jnp.stack([jax.hessian(lambda x, j=j: op.trunk(x[None])[0, j])(z[0]) for j in range(6)])
            ref = jnp.einsum("ab,jab->j", L @ L.T, H)
            self.assertLess(float(jnp.abs(sec[0] - ref).max()), 1e-10)
            _, grad_ref, lap = op.trunk_zderivs(z)
            self.assertLess(float(jnp.abs(grad - grad_ref).max()), 1e-12)
            _, _, sec_iso = op.trunk_zderivs_dirs(z, 0.7 * jnp.eye(d))
            self.assertLess(float(jnp.abs(sec_iso - 0.49 * lap).max()), 1e-12)


class TestLossReduction(unittest.TestCase):
    def test_matrix_and_diag_reduce_exactly(self):
        s_coll = jnp.linspace(0, 1, 3)
        lp = lambda z: -0.5 * (z ** 2).sum(-1)                          # noqa: E731
        for d in (1, 2, 3):
            op, dn, xs, mask, z_col, log_q, decode, C, off = _setup(d)
            g, var = 0.7, 0.09
            args = (op, xs, mask, z_col, log_q, s_coll, dn.drift)
            base = pinn_zakai_loss(*args, g, lp, var, 0.1, res_mode="rel", decode=decode)
            mat = pinn_zakai_loss(*args, g * jnp.eye(d), lp, var, 0.1, res_mode="rel", decode=decode)
            dia = pinn_zakai_loss(*args, g, lp, jnp.full((4,), var), 0.1, res_mode="rel", decode=decode)
            for a, b, c in zip(base, mat, dia):
                self.assertEqual(float(a), float(b))
                self.assertEqual(float(a), float(c))


class TestMStepRecovery(unittest.TestCase):
    def test_recovers_anisotropic_sigma(self):
        d, N, dt = 3, 300000, 0.1
        Ltrue = jnp.array([[0.8, 0., 0.], [0.3, 0.5, 0.], [-0.2, 0.1, 1.4]])
        k = jax.random.split(jax.random.PRNGKey(3), 2)
        zc = jax.random.normal(k[0], (N, d)) * 0.5
        delta = (Ltrue @ jax.random.normal(k[1], (d, N))).T * math.sqrt(dt)
        dn = DriftNet(8, layers=2, latent_dim=d, key=jax.random.PRNGKey(0))   # zero-init last layer
        L = fit_diffusion_cov(dn, zc, zc + delta, delta / dt, dt, g_floor=1e-6)
        self.assertLess(float(jnp.abs((L @ L.T) - (Ltrue @ Ltrue.T)).max()), 0.05)

    def test_chol_summary_isotropic(self):
        g_det, g_iso, aniso = chol_summary(0.6 * jnp.eye(3))
        self.assertAlmostEqual(g_det, 0.6, places=5)
        self.assertAlmostEqual(g_iso, 0.6, places=5)
        self.assertAlmostEqual(aniso, 1.0, places=5)

    def test_perp_estimator_is_immune_to_an_overwide_posterior(self):
        k = jax.random.split(jax.random.PRNGKey(5), 4)
        T, B, K, D, dl, Rtrue = 40, 16, 64, 10, 3, 0.1
        C = jnp.linalg.qr(jax.random.normal(k[0], (D, dl)))[0]
        off = jax.random.normal(k[1], (D,))
        ztrue = jax.random.normal(k[2], (T, B, dl))
        x = jnp.einsum("od,tbd->tbo", C, ztrue) + off + jax.random.normal(k[3], (T, B, D)) * math.sqrt(Rtrue)
        mask, w = jnp.ones((T, B, 1)), jnp.full((T, B, K), 1.0 / K)
        out = {}
        for width in (0.05, 0.6):
            z_s = ztrue[:, :, None] + width * jax.random.normal(jax.random.PRNGKey(9), (T, B, K, dl))
            for est in ("posterior", "perp"):
                R, _ = fit_obs_noise(x, mask, z_s, w, C, off, mode="scalar", est=est)
                out[(est, width)] = float(R.mean())
        self.assertAlmostEqual(out[("perp", 0.05)], out[("perp", 0.6)], places=6)
        self.assertGreater(out[("posterior", 0.6)], 2 * out[("posterior", 0.05)])
        self.assertLess(abs(out[("perp", 0.6)] - Rtrue) / Rtrue, 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
