"""Trunk activation, derivative consistency and Torch inference paths.

Run: python -m unittest discover -s tests -p test_operator_activation.py
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


def operator(d=1, backward=False, trunk_activation='softplus'):
    cls = OperatorBackward if backward else OperatorFilter
    return cls(data_size=d, latent_dim=d, gru_hidden=8, ctx_dim=8, p=8,
               branch_hidden=8, trunk_hidden=8, trunk_layers=1,
               trunk_activation=trunk_activation).double()


class OperatorActivationTest(unittest.TestCase):
    def test_default_and_legacy_activation(self):
        model, legacy = operator(), operator(trunk_activation='tanh')
        self.assertIsInstance(model.trunk[1], torch.nn.Softplus)
        self.assertIsInstance(model.branch[1], torch.nn.Tanh)
        legacy.load_state_dict(model.state_dict(), strict=True)
        z = torch.tensor([[2.]])
        self.assertFalse(torch.allclose(model.trunk(z), legacy.trunk(z)))
        with self.assertRaises(ValueError):
            operator(trunk_activation='invalid')

    def test_density_and_derivatives(self):
        for activation in ('softplus', 'tanh'):
            for d in (1, 3):
                model = operator(d, trunk_activation=activation)
                ctx = model.context(torch.randn(4, 2, d), self.mask)
                z = torch.randn(7, d, requires_grad=True)
                tau, grad, lap = model.trunk_zderivs(z)
                torch.testing.assert_close(tau, model.trunk(z))
                self.assertEqual(tau.shape, (7, 8))
                torch.testing.assert_close(model.trunk_grad(z)[1], grad)
                s = torch.tensor([0., .4, 1.], requires_grad=True)
                b, db = model.coeffs_dtime(ctx, s)
                torch.testing.assert_close(b, model.coeffs(ctx, s))
                torch.testing.assert_close(torch.autograd.grad(b.sum(), s, retain_graph=True)[0],
                                           db.sum((0, 1, 3)))
                ell = model.log_density(ctx[:1, :1], z)[0, 0]
                grad_direct = torch.autograd.grad(ell.sum(), z, create_graph=True)[0]
                lap_direct = sum(torch.autograd.grad(grad_direct[:, i].sum(), z, retain_graph=True)[0][:, i]
                                 for i in range(d))
                torch.testing.assert_close(grad_direct, torch.einsum('p,zdp->zd', b[0, 0, 0], grad))
                torch.testing.assert_close(lap_direct, torch.einsum('p,zp->z', b[0, 0, 0], lap))

    def test_mala_on_softplus_logistic_density(self):
        model = operator()
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.trunk[0].weight[0, 0] = 1.
            model.trunk[0].weight[1, 0] = -1.
            model.trunk[2].weight[0, :2] = -1.
            model.branch[2].bias[0] = 1.
            z = torch.linspace(-20., 20., 4001)
            density = model.log_density(model.context(self.x, self.mask), z).exp()
            torch.testing.assert_close(torch.trapezoid(density, z), torch.ones(4, 2))
        samples, mean, acc = _mala_chains(model, self.x, self.mask, torch.ones_like(self.x)*3,
                                         128, 150, 2., rng='crn')
        self.assertLess(float(mean.abs().max()), .2)
        self.assertLess(abs(float(samples.var())-math.pi**2/3), .5)
        self.assertGreater(acc, .3)

    def test_checkpoint_activation_and_removed_tail(self):
        import tempfile
        from pathlib import Path
        import lightning.pytorch as pl
        from opssm.models.filter_module import ZakaiFilterModule
        model = ZakaiFilterModule(gru_hidden=8, ctx_dim=8, p=8, branch_hidden=8,
                                  trunk_hidden=8, trunk_layers=1, drift_hidden=8, drift_layers=1)
        checkpoint = {'state_dict': model.state_dict(), 'hyper_parameters': dict(model.hparams),
                      'pytorch-lightning_version': pl.__version__}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.ckpt'
            torch.save(checkpoint, path)
            restored = ZakaiFilterModule.load_from_checkpoint(path)
            self.assertIsInstance(restored.model.trunk[1], torch.nn.Softplus)
            with self.assertRaisesRegex(ValueError, 'trunk_activation'):
                ZakaiFilterModule(trunk_activation='tanh').on_load_checkpoint(checkpoint)
            checkpoint['hyper_parameters'].pop('trunk_activation')
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'trunk_activation'):
                ZakaiFilterModule.load_from_checkpoint(path)
            restored = ZakaiFilterModule.load_from_checkpoint(path, trunk_activation='tanh')
            self.assertIsInstance(restored.model.trunk[1], torch.nn.Tanh)
            checkpoint['hyper_parameters']['tail_std'] = 2.
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'removed Gaussian tails'):
                ZakaiFilterModule.load_from_checkpoint(path, trunk_activation='tanh')

    def test_softplus_unbounded_basis_and_pde_derivatives(self):
        model = operator(trunk_activation='softplus')
        # Construct -softplus(z)-softplus(-z), whose value, score and curvature are known.
        with torch.no_grad():
            for p in model.trunk.parameters():
                p.zero_()
            model.trunk[0].weight[0, 0] = 1.
            model.trunk[0].weight[1, 0] = -1.
            model.trunk[2].weight[0, :2] = -1.
        z = torch.tensor([[-100.], [-2.], [0.], [2.], [100.]])
        tau, grad, lap = model.trunk_zderivs(z)
        torch.testing.assert_close(tau[:, 0], -torch.logaddexp(z[:, 0], torch.zeros(5))
                                   - torch.logaddexp(-z[:, 0], torch.zeros(5)))
        torch.testing.assert_close(grad[:, 0, 0], -torch.tanh(z[:, 0]/2))
        torch.testing.assert_close(lap[:, 0], -.5/torch.cosh(z[:, 0]/2).square())
        nodes, logq = sample_collocation(self.x, self.mask, 16, .3, 1.6)
        res, jump, ic, _ = pinn_zakai_loss(model, self.x, self.mask, nodes, logq,
            torch.tensor([0., .5, 1.]), lambda z: (z-z**3, (1-3*z*z).sum(-1)),
            .6, lambda z: -z.square().sum(-1)/2, .3, .1, res_mode='rel')
        loss = res+jump+ic
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def setUp(self):
        torch.manual_seed(4)
        self.old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        self.x = torch.randn(4, 2, 1)
        self.mask = torch.ones(4, 2, 1)

    def tearDown(self):
        torch.set_default_dtype(self.old_dtype)


    def test_losses_and_readouts_share_basis(self):
        model, backward = operator(), operator(backward=True)
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
