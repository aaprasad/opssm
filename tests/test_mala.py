"""Phase-1 MALA readout verification (opssm.models.mstep.posterior_mean_mala):
  1. trunk_grad matches trunk_zderivs' gradient (gradient-only fast path is correct).
  2. determinism: rng="crn" is bit-exact across calls; rng="stochastic" differs.
  3. Metropolis correctness: on an ANALYTIC Gaussian target N(mu, s^2), MALA recovers the mean mu
     (independent of the chain init) -- a robust ground truth (an untrained operator density is not).
  4. acceptance sits in a healthy band after adaptation.

Run: PYTHONPATH=<repo> ~/venvs/neuraloperator/bin/python tests/test_mala.py
"""
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
from opssm.models.operator import OperatorFilter
from opssm.models import mstep as M

ok = True
D = 6

# 1. trunk_grad vs trunk_zderivs gradient ------------------------------------------------
for d in (1, 2, 3):
    m = OperatorFilter(data_size=D, latent_dim=d)
    z = torch.randn(4, 3, 5, d)
    _, grad_g = m.trunk_grad(z)
    _, grad_z, _ = m.trunk_zderivs(z)
    err = (grad_g - grad_z).abs().max().item()
    print(f"d={d}: trunk_grad vs trunk_zderivs grad max err = {err:.2e}")
    ok &= err < 1e-9

# 2. determinism -------------------------------------------------------------------------
m = OperatorFilter(data_size=D, latent_dim=2)
x, mask, center = torch.randn(6, 4, D), torch.ones(6, 4, 1), torch.randn(6, 4, 2) * 0.5
args = (m, x, mask, center, 32, 20, 1.6)                       # n_chains, n_steps, broad_std (rest auto)
z1, acc = M.posterior_mean_mala(*args, rng="crn")
z2, _ = M.posterior_mean_mala(*args, rng="crn")
zs1, _ = M.posterior_mean_mala(*args, rng="stochastic")
zs2, _ = M.posterior_mean_mala(*args, rng="stochastic")
crn_det, stoch_diff = torch.equal(z1, z2), not torch.equal(zs1, zs2)
print(f"determinism: crn bit-exact={crn_det}  stochastic differs={stoch_diff}  accept={acc:.3f}")
ok &= crn_det and stoch_diff


# 3. Metropolis correctness on an analytic Gaussian N(mu, s^2 I) --------------------------
class GaussModel:
    """Fakes the operator API used by MALA so that ell(z) = -0.5*sum(((z-mu)/s)^2), i.e. pi = N(mu, s^2 I).
    b0 = ones(T,B,1); trunk_grad returns (tau (...,1), grad_tau (...,d,1)) with grad ell = -(z-mu)/s^2."""
    bias = 0.0

    def __init__(self, mu, s):
        self.mu, self.s = mu, s

    def context(self, x, mask):
        return x.shape[:2]                                    # (T,B)

    def coeffs(self, ctx, s0):
        T, B = ctx
        return torch.ones(T, B, 1, 1)                         # [:, :, 0] -> b0 = ones(T,B,1)

    def trunk_grad(self, z):                                  # z (T,B,K,d)
        tau = (-0.5 * ((z - self.mu) / self.s).pow(2).sum(-1, keepdim=True))   # (T,B,K,1)
        gtau = (-(z - self.mu) / self.s ** 2).unsqueeze(-1)                    # (T,B,K,d,1)
        return tau, gtau


for d, mu, s in [(1, 0.7, 1.0), (2, -0.4, 0.8), (3, 1.2, 1.3)]:
    gm = GaussModel(mu, s)
    x, mask = torch.randn(6, 4, D), torch.ones(6, 4, 1)
    center = torch.randn(6, 4, d) * 3.0                       # init FAR from mu -> AUTO defaults must still converge
    z_hat, acc = M.posterior_mean_mala(gm, x, mask, center, 128, 100, 2.0, rng="crn")   # only budget + broad_std
    err = (z_hat - mu).abs().max().item()
    print(f"d={d}: MALA mean vs analytic mu={mu}: max err = {err:.4f}  accept = {acc:.3f}")
    ok &= err < 0.1 and 0.30 < acc < 0.88                     # recovers mu (finite chains); acceptance healthy

print("MALA VERIFICATION PASS" if ok else "*** MALA VERIFICATION FAIL ***")
