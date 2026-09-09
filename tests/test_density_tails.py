"""Forward density tails, derivative consistency and all Torch inference paths.

Run: python -m unittest discover -s tests -p test_density_tails.py
"""
import math
import unittest
from types import SimpleNamespace

import torch

from opssm.models.operator import OperatorFilter, OperatorBackward
from opssm.models.losses import pinn_zakai_loss, pinn_adjoint_loss, sample_collocation
from opssm.models.mstep import (
    _mala_chains, posterior_mean_fixed, smoother_mean_fixed, filter_pair_fixed, smoother_pair_fixed,
)
from opssm.models.dynamics import DriftNet


def operator(d=1, tail_std=0.0, backward=False):
    cls = OperatorBackward if backward else OperatorFilter
    return cls(data_size=d, latent_dim=d, gru_hidden=8, ctx_dim=8, p=8,
               branch_hidden=8, trunk_hidden=8, trunk_layers=1, tail_std=tail_std).double()


class DensityTailsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        self.x = torch.randn(4, 2, 1)
        self.mask = torch.ones(4, 2, 1)

    def tearDown(self):
        torch.set_default_dtype(self.old_dtype)

    def test_legacy_weights_and_tail_derivatives(self):
        for d in (1, 3):
            with self.subTest(d=d):
                old, new = operator(d), operator(d, 2.0)
                new.load_state_dict(old.state_dict(), strict=True)
                x = torch.randn(4, 2, d)
                ctx = new.context(x, self.mask)
                z = torch.randn(7, d, requires_grad=True)
                for s in (0.0, 0.4, 1.0):
                    actual = new.log_density(ctx, z, s)
                    expected = old.log_density(ctx, z, s) - z.square().sum(-1) / 8
                    torch.testing.assert_close(actual, expected)
                tau, grad, lap = new.trunk_zderivs(z)
                torch.testing.assert_close(tau, new.state_basis(z))
                torch.testing.assert_close(grad[..., -1], -z / 4)
                torch.testing.assert_close(lap[..., -1], torch.full((7,), -d / 4))
                torch.testing.assert_close(new.trunk_grad(z)[1], grad)
                s = torch.tensor([0.0, 0.4, 1.0])
                b, db = new.coeffs_dtime(ctx, s)
                torch.testing.assert_close(b, new.coeffs(ctx, s))
                torch.testing.assert_close(b[..., -1], torch.ones_like(b[..., -1]))
                torch.testing.assert_close(db[..., -1], torch.zeros_like(db[..., -1]))
                # Full log-density spatial derivatives agree with the factorized PDE path.
                ell = new.log_density(ctx[:1, :1], z)[0, 0]
                grad_direct = torch.autograd.grad(ell.sum(), z, create_graph=True)[0]
                lap_direct = sum(torch.autograd.grad(grad_direct[:, i].sum(), z, retain_graph=True)[0][:, i]
                                 for i in range(d))
                torch.testing.assert_close(grad_direct, torch.einsum('p,zdp->zd', b[0, 0, 0], grad))
                torch.testing.assert_close(lap_direct, torch.einsum('p,zp->z', b[0, 0, 0], lap))
                zbase = old.trunk(z)
                torch.testing.assert_close(old.state_basis(z), zbase, rtol=0, atol=0)

    def test_normalizer_and_mala_on_actual_gaussian_operator(self):
        model = operator(tail_std=1.3)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            ctx = model.context(self.x, self.mask)
            for radius in (8.0, 16.0):
                z = torch.linspace(-radius, radius, 4001)
                mass = torch.trapezoid(model.log_density(ctx, z).exp(), z)
                torch.testing.assert_close(mass, torch.full((4, 2), math.sqrt(2 * math.pi) * 1.3))
        samples, mean, acc = _mala_chains(model, self.x, self.mask, torch.ones_like(self.x) * 3,
                                         128, 150, 2.0, rng='crn')
        self.assertLess(float(mean.abs().max()), 0.1)
        self.assertLess(abs(float(samples.var()) - 1.3 ** 2), 0.25)
        self.assertGreater(acc, 0.3)

    def test_losses_and_readouts_share_basis(self):
        model, backward = operator(tail_std=2.0), operator(tail_std=2.0, backward=True)
        self.assertEqual(backward.tail_std, 0.0)
        drift = DriftNet(8, layers=1).double()
        drift.requires_grad_(False)
        z, logq = sample_collocation(self.x, self.mask, 16, 0.3, 1.6)
        args = (self.x, self.mask, z, logq, torch.tensor([0., 0.5, 1.]), drift.drift, 0.6)
        res, jump, ic, _ = pinn_zakai_loss(model, *args, lambda z: -z.square().sum(-1)/2,
                                          0.3, 0.1, res_mode='rel')
        (res + jump + ic).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        losses_b = pinn_adjoint_loss(backward, *args, 0.3, 0.1)
        sum(losses_b).backward()
        self.assertTrue(all(torch.isfinite(l) for l in losses_b))
        mean, _ = posterior_mean_fixed(model, self.x, self.mask, self.x, 64, .3, 1.6)
        smooth, _ = smoother_mean_fixed(model, backward, self.x, self.mask, self.x, 64, .3, 1.6)
        self.assertTrue(torch.isfinite(mean).all() and torch.isfinite(smooth).all())
        for pair in (
            filter_pair_fixed(model, self.x, self.mask, self.x, drift, .6, .1, 32, .3, 1.6, .3),
            smoother_pair_fixed(model, backward, self.x, self.mask, self.x, drift, .6, .1, 32, .3, 1.6),
        ):
            self.assertTrue(all(torch.isfinite(t).all() for t in pair))

    def test_reject_invalid_tail(self):
        for value in (-1., float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                operator(tail_std=value)
        with self.assertRaises(ValueError):
            OperatorFilter(tail_std=2.0, trunk_layers=0)

    def test_smoother_mean_metrics_without_grid_density(self):
        from opssm.models.filter_module import ZakaiFilterModule
        module = SimpleNamespace(drift_net=DriftNet(8, layers=1).double(), g_cur=.6, sigma=.6,
                                 true_drift=lambda z: z-z**3)
        mean = torch.linspace(-1., 1., 8).reshape(4, 2, 1)
        metrics = ZakaiFilterModule._gauge_aligned(module, None, None, mean, mean)
        self.assertLess(metrics['lat_rel'], 1e-10)
        self.assertNotIn('kl_aln', metrics)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
