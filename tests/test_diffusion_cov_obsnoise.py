"""Matrix diffusion (Sigma = L L^T) and the learned observation noise R -- both backends.

The load-bearing claims:
  1. tr(Sigma H) computed by directional 2nd derivatives along L's COLUMNS equals the explicit
     contraction of Sigma with the autodiff Hessian.
  2. With L = g I the whole loss is BIT-IDENTICAL to the isotropic scalar path, and a diagonal R whose
     entries are all equal is bit-identical to a scalar R -- so the new defaults cannot silently move
     any existing d==1 / isotropic result.
  3. The closed-form M-steps recover a known anisotropic Sigma and a known diagonal R.
"""
import math
import unittest

import torch

from opssm.models.operator import OperatorFilter
from opssm.models.dynamics import DriftNet
from opssm.models.losses import pinn_zakai_loss, sample_collocation
from opssm.models.mstep import fit_diffusion_cov, fit_obs_noise, chol_summary

torch.manual_seed(0)


def _rand_chol(d, seed=0):
    g = torch.Generator().manual_seed(seed)
    L = torch.tril(torch.randn(d, d, generator=g, dtype=torch.double))
    return L + torch.eye(d, dtype=torch.double) * 1.5           # well-conditioned lower-triangular


def _setup(d, D=4, T=5, B=2, K=8):
    op = OperatorFilter(data_size=D, latent_dim=d, p=6, trunk_hidden=8, trunk_layers=2,
                        gru_hidden=8, ctx_dim=8, branch_hidden=8).double()
    dn = DriftNet(8, layers=2, latent_dim=d).double()
    for p in dn.net.parameters():
        torch.nn.init.normal_(p, std=0.3)
    xs = torch.randn(T, B, D, dtype=torch.double)
    mask = torch.ones(T, B, 1, dtype=torch.double)
    C = torch.linalg.qr(torch.randn(D, d, dtype=torch.double))[0]
    off = torch.randn(D, dtype=torch.double)
    decode = lambda z: torch.einsum("od,...d->...o", C, z) + off        # noqa: E731
    ctr = torch.einsum("od,tbo->tbd", C, xs - off)
    torch.manual_seed(1)
    z_col, log_q = sample_collocation(xs, mask, K, 0.3, 1.6, ctr)
    return op, dn, xs, mask, z_col, log_q, decode, C, off


class TestWeightedHessianTrace(unittest.TestCase):
    def test_matches_explicit_hessian(self):
        for d in (1, 2, 3):
            op, *_ = _setup(d)
            z = torch.randn(6, d, dtype=torch.double)
            L = _rand_chol(d, seed=d)
            _, dgrad, sec = op.trunk_zderivs_dirs(z, L.t())
            Sig = L @ L.t()
            H = torch.stack([torch.autograd.functional.hessian(
                lambda x: op.trunk(x)[j], z[0].clone(), vectorize=True) for j in range(6)])
            self.assertLess(float((sec[0] - torch.einsum("ab,jab->j", Sig, H)).abs().max()), 1e-10)
            # the single fused pass returns DIRECTIONAL first derivatives = L^T grad tau (no second pass)
            _, grad_ref, _ = op.trunk_zderivs(z)                     # (...,d,p) full gradient
            self.assertLess(float((dgrad - torch.einsum("de,ndp->nep", L, grad_ref)).abs().max()), 1e-12)

    def test_isotropic_reduces_to_laplacian(self):
        for d in (1, 2, 3):
            op, *_ = _setup(d)
            z = torch.randn(6, d, dtype=torch.double)
            _, _, lap = op.trunk_zderivs(z)
            _, dgrad, sec = op.trunk_zderivs_dirs(z, 0.7 * torch.eye(d, dtype=torch.double))
            self.assertLess(float((sec - 0.49 * lap).abs().max()), 1e-12)
            _, grad_ref, _ = op.trunk_zderivs(z)                     # dirs = g I -> dgrad = g * grad
            self.assertLess(float((dgrad - 0.7 * grad_ref).abs().max()), 1e-12)


class TestLossReduction(unittest.TestCase):
    """L = g I and an equal-entry diagonal R must be BIT-identical to the legacy scalar path."""

    def test_matrix_and_diag_reduce_exactly(self):
        s_coll = torch.linspace(0, 1, 3, dtype=torch.double)
        lp = lambda z: -0.5 * (z ** 2).sum(-1)                        # noqa: E731
        for d in (1, 2, 3):
            op, dn, xs, mask, z_col, log_q, decode, C, off = _setup(d)
            g, var = 0.7, 0.09
            args = (op, xs, mask, z_col, log_q, s_coll, dn.drift)
            base = pinn_zakai_loss(*args, g, lp, torch.tensor(var, dtype=torch.double), 0.1,
                                   res_mode="rel", decode=decode)
            mat = pinn_zakai_loss(*args, g * torch.eye(d, dtype=torch.double), lp,
                                  torch.tensor(var, dtype=torch.double), 0.1,
                                  res_mode="rel", decode=decode)
            dia = pinn_zakai_loss(*args, g, lp, torch.full((4,), var, dtype=torch.double), 0.1,
                                  res_mode="rel", decode=decode)
            for a, b, c in zip(base, mat, dia):
                self.assertEqual(a.item(), b.item())                  # exact, not approximate
                self.assertEqual(a.item(), c.item())


class TestMStepRecovery(unittest.TestCase):
    def test_recovers_anisotropic_sigma(self):
        d, N, dt = 3, 300000, 0.1
        Ltrue = torch.tensor([[0.8, 0., 0.], [0.3, 0.5, 0.], [-0.2, 0.1, 1.4]])
        torch.manual_seed(3)
        zc = torch.randn(N, d) * 0.5
        delta = (Ltrue @ torch.randn(d, N)).t() * math.sqrt(dt)
        dn = DriftNet(8, layers=2, latent_dim=d)
        for p in dn.net.parameters():
            torch.nn.init.zeros_(p)                                   # zero drift -> residual IS the noise
        L = fit_diffusion_cov(dn, zc, zc + delta, delta / dt, dt, g_floor=1e-6)
        self.assertLess(float(((L @ L.t()) - (Ltrue @ Ltrue.t())).abs().max()), 0.05)

    def test_chol_summary_isotropic(self):
        g_det, g_iso, aniso = chol_summary(0.6 * torch.eye(3))
        self.assertAlmostEqual(g_det, 0.6, places=5)
        self.assertAlmostEqual(g_iso, 0.6, places=5)
        self.assertAlmostEqual(aniso, 1.0, places=5)

    def test_d1_covariance_equals_scalar_path(self):
        """At d==1 the covariance M-step must agree with the scalar one to floating-point round-off, so a
        d==1 A/B (e.g. em_highd) isolates the OTHER knobs instead of silently comparing two diffusion floors.
        Not bit-exact: the matrix path reduces via `r^T r` and round-trips through eigh, so it differs from
        the scalar path's `mean(r^2)` in the last ulp (~1e-7 relative). What matters is that there is no
        SYSTEMATIC offset -- the `+ g_floor^2 I` ridge this replaced biased g^2 by a fixed 0.0025 (~0.7% on
        this benchmark), which is a real effect an A/B could mistake for a result."""
        from opssm.models.mstep import fit_diffusion
        torch.manual_seed(11)
        dn = DriftNet(8, layers=2, latent_dim=1)
        dt = 0.1
        r = torch.randn(60000, 1) * 0.6                               # normal regime, floor inactive
        zc = torch.randn(60000, 1)
        g_cov = chol_summary(fit_diffusion_cov(dn, zc, zc, r, dt, g_floor=0.05))[0]
        g_sca = fit_diffusion(None, None, dn, zc, zc, r, None, None, dt, False, 0.0, 0)
        self.assertAlmostEqual(g_cov, g_sca, places=6)                # round-off, no systematic offset
        self.assertLess(abs(g_cov - g_sca) / g_sca, 1e-5)
        r_small = torch.randn(60000, 1) * 0.01                        # floor active -> equal to float round-off
        g_cov = chol_summary(fit_diffusion_cov(dn, zc, zc, r_small, dt, g_floor=0.05))[0]
        g_sca = fit_diffusion(None, None, dn, zc, zc, r_small, None, None, dt, False, 0.0, 0)
        self.assertAlmostEqual(g_cov, g_sca, places=6)

    def test_floor_does_not_inflate_healthy_eigenvalues(self):
        """A `+ g_floor^2 I` ridge would shift EVERY eigenvalue; the clamp must leave healthy ones alone."""
        d, N, dt = 3, 200000, 0.1
        torch.manual_seed(12)
        Ltrue = torch.diag(torch.tensor([1.2, 0.9, 0.7]))
        zc = torch.randn(N, d)
        delta = (Ltrue @ torch.randn(d, N)).t() * math.sqrt(dt)
        dn = DriftNet(8, layers=2, latent_dim=d)
        for p_ in dn.net.parameters():
            torch.nn.init.zeros_(p_)
        L = fit_diffusion_cov(dn, zc, zc + delta, delta / dt, dt, g_floor=0.05)
        ev = torch.linalg.eigvalsh(L @ L.t())
        self.assertLess(float((ev.sort().values - torch.tensor([0.49, 0.81, 1.44])).abs().max()), 0.02)

    def test_recovers_diagonal_R(self):
        torch.manual_seed(4)
        T, B, K, D, dl = 40, 16, 64, 10, 3
        C = torch.linalg.qr(torch.randn(D, dl))[0]
        off = torch.randn(D)
        ztrue = torch.randn(T, B, dl)
        Rtrue = torch.rand(D) * 0.2 + 0.05
        x = torch.einsum("od,tbd->tbo", C, ztrue) + off + torch.randn(T, B, D) * Rtrue.sqrt()
        z_s = ztrue.unsqueeze(2) + 0.05 * torch.randn(T, B, K, dl)    # posterior samples with spread
        w = torch.full((T, B, K), 1.0 / K)
        mask = torch.ones(T, B, 1)
        for est in ("posterior", "perp"):
            R, diag = fit_obs_noise(x, mask, z_s, w, C, off, mode="diag", est=est)
            rel = float(((R - Rtrue).abs() / Rtrue).mean())
            self.assertLess(rel, 0.15, f"{est} mean relative error {rel:.3f}")
            self.assertTrue(math.isfinite(diag["R_perp"]) and math.isfinite(diag["R_post"]))

    def test_perp_estimator_is_immune_to_an_overwide_posterior(self):
        """The whole point of est='perp': inflating the posterior width must not inflate R."""
        torch.manual_seed(5)
        T, B, K, D, dl = 40, 16, 64, 10, 3
        C = torch.linalg.qr(torch.randn(D, dl))[0]
        off = torch.randn(D)
        ztrue = torch.randn(T, B, dl)
        Rtrue = 0.1
        x = torch.einsum("od,tbd->tbo", C, ztrue) + off + torch.randn(T, B, D) * math.sqrt(Rtrue)
        mask = torch.ones(T, B, 1)
        w = torch.full((T, B, K), 1.0 / K)
        out = {}
        for width in (0.05, 0.6):                                     # a 12x too-wide posterior
            z_s = ztrue.unsqueeze(2) + width * torch.randn(T, B, K, dl)
            for est in ("posterior", "perp"):
                R, _ = fit_obs_noise(x, mask, z_s, w, C, off, mode="scalar", est=est)
                out[(est, width)] = float(R.mean())
        self.assertAlmostEqual(out[("perp", 0.05)], out[("perp", 0.6)], places=6)   # unchanged
        self.assertGreater(out[("posterior", 0.6)], 2 * out[("posterior", 0.05)])   # inflated
        self.assertLess(abs(out[("perp", 0.6)] - Rtrue) / Rtrue, 0.1)               # and accurate


if __name__ == "__main__":
    unittest.main(verbosity=2)
