"""SDE Matching baseline worker (Bartosh, Vetrov & Naesseth 2025, arXiv:2502.02472). Standalone torch
worker run by ~/venvs/baselines as `python sdematching_worker.py in.npz out.npz` (needs only torch+numpy).

Simulation-FREE latent-SDE training: rather than solving/backprop-through SDE solutions (torchsde), it
matches the amortized posterior's analytic drift to the prior SDE drift (score/flow-matching style). Code
adapted from the authors' from-scratch notebook, wired to our linear-Gaussian emission Cz+d (PCA-init) and
the npz seam. Amortized GRU posterior over the whole sequence -> a SMOOTHER (fit on FIT, infer on EVAL).
Sim-free foil to torchsde (sim-based); shares the sim-free/amortized axis with opssm and visde.

Time note: the method trains on normalized t in [0,1], so the learned prior drift is dz/dt_norm; we divide
by T_phys=(T-1)*dt to report physical drift (and g by sqrt(T_phys)) so drift/g align with the true flow.
"""
import sys
import time

import numpy as np
import torch
from torch import nn, Tensor
from torch import distributions as D


def _jvp(f, x, v):
    return torch.autograd.functional.jvp(f, x, v, create_graph=torch.is_grad_enabled())


def _t_dir(f, t):
    return _jvp(f, t, torch.ones_like(t))


def _grad(f, x):                                                    # y=f(x) and dy/dx (Ito correction term)
    create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        x = x.clone()
        if not x.requires_grad:
            x.requires_grad = True
        y = f(x)
        (g,) = torch.autograd.grad(y.sum(), x, create_graph=create_graph)
    return y, g


class SDE(nn.Module):
    def forward(self, z, t, *a):
        return self.drift(z, t, *a), self.vol(z, t, *a)


class PriorInitDistribution(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.m = nn.Parameter(torch.zeros(1, d))
        self.log_s = nn.Parameter(torch.zeros(1, d))

    def forward(self):
        return D.Independent(D.Normal(self.m, torch.exp(self.log_s)), 1)


class PriorSDE(SDE):
    def __init__(self, d, h):
        super().__init__()
        self.drift_net = nn.Sequential(nn.Linear(d, h), nn.Softplus(), nn.Linear(h, d))
        self.vol_nets = nn.ModuleList([nn.Sequential(nn.Linear(1, h), nn.Softplus(),
                                                     nn.Linear(h, 1), nn.Sigmoid()) for _ in range(d)])

    def drift(self, z, t, *a):
        return self.drift_net(z)

    def vol(self, z, t, *a):
        zs = torch.split(z, 1, dim=1)
        return torch.cat([net(zi) for net, zi in zip(self.vol_nets, zs)], dim=1)


class PriorObservation(nn.Module):                                 # p(x|z) = N(Cz+d, noise^2 I) -- our emission
    def __init__(self, d, N, noise_std):
        super().__init__()
        self.net = nn.Linear(d, N)
        self.noise_std = noise_std

    def get_coeffs(self, z):
        m = self.net(z)
        return m, torch.ones_like(m) * self.noise_std

    def forward(self, z):
        m, s = self.get_coeffs(z)
        return D.Independent(D.Normal(m, s), 1)


class PosteriorEncoder(nn.Module):
    def __init__(self, N, h):
        super().__init__()
        self.gru = nn.GRU(N, h, batch_first=True)

    def forward(self, x):
        out, hh = self.gru(x)
        return torch.cat([hh[0, :, None], out], dim=1)


class PosteriorAffine(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h + 1, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, 2 * d))
        self.sm = nn.Softmax(dim=-1)

    def get_coeffs(self, ctx, t):
        l = ctx.shape[1] - 1
        h, out = ctx[:, 0], ctx[:, 1:]
        ts = torch.linspace(0, 1, l, device=ctx.device)[None, :]
        c = self.sm(-(l * (ts - t)) ** 2)
        out = (out * c[:, :, None]).sum(dim=1)
        m, log_s = self.net(torch.cat([h + out, t], dim=1)).chunk(2, dim=1)
        return m, torch.exp(log_s)

    def forward(self, ctx, t, return_t_dir=False):
        if return_t_dir:
            return _t_dir(lambda tt: self.get_coeffs(ctx, tt), t)
        return self.get_coeffs(ctx, t)


class MatchingSDE(nn.Module):
    def __init__(self, p_init, p_sde, p_obs, q_enc, q_aff):
        super().__init__()
        self.p_init_distr, self.p_sde, self.p_observe = p_init, p_sde, p_obs
        self.q_enc, self.q_affine = q_enc, q_aff

    def loss_prior(self, ctx):
        t0 = torch.zeros(ctx.shape[0], 1, device=ctx.device)
        m0, s0 = self.q_affine(ctx, t0)
        return D.kl_divergence(D.Independent(D.Normal(m0, s0), 1), self.p_init_distr())

    def loss_diff(self, ctx, t):
        (m, s), (dm, ds) = self.q_affine(ctx, t, return_t_dir=True)
        eps = torch.randn_like(m)
        z = m + s * eps
        g2, d_g2 = _grad(lambda zin: self.p_sde.vol(zin, t) ** 2, z)
        q_drift = (dm + ds * eps) + 0.5 * g2 * (-eps / s) + 0.5 * d_g2   # posterior drift (analytic, sim-free)
        return (0.5 * (q_drift - self.p_sde.drift(z, t)) ** 2 / g2).sum(dim=1)

    def loss_recon(self, ctx, x, t):
        m, s = self.q_affine(ctx, t)
        z = m + s * torch.randn_like(m)
        return -self.p_observe(z).log_prob(x)

    def forward(self, xs, ts):
        bs, n = xs.shape[0], xs.shape[1]
        ctx = self.q_enc(xs)
        t = torch.rand(bs, 1, device=xs.device) * (ts[:, -1] - ts[:, 0]) + ts[:, 0]
        rng = torch.arange(bs, device=xs.device)
        u = torch.randint(n, [bs], device=xs.device)
        return self.loss_prior(ctx) + self.loss_diff(ctx, t) + self.loss_recon(ctx, xs[rng, u], ts[rng, u])


def main(fin, fout):
    z = np.load(fin)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xf = torch.tensor(np.asarray(z['obs_std_fit']).transpose(1, 0, 2), dtype=torch.float32, device=dev)
    xe = torch.tensor(np.asarray(z['obs_std_eval']).transpose(1, 0, 2), dtype=torch.float32, device=dev)
    dt, d, noise = float(z['dt']), int(z['latent_dim']), float(z['noise_std_eff'])
    Bf, T, N = xf.shape
    steps = int(z['steps']) if 'steps' in z.files else 3000
    h = 128
    Tphys = (T - 1) * dt
    ts = torch.linspace(0, 1, T, device=dev)[None, :, None].repeat(Bf, 1, 1)     # normalized time in [0,1]

    torch.manual_seed(0)
    p_sde = PriorSDE(d, h).to(dev)
    p_obs = PriorObservation(d, N, noise).to(dev)
    Yf = xf.reshape(-1, N)
    mean = Yf.mean(0)
    _, _, Vt = torch.linalg.svd(Yf - mean, full_matrices=False)                  # PCA-init emission
    with torch.no_grad():
        p_obs.net.weight.copy_(Vt[:d].T)
        p_obs.net.bias.copy_(mean)
    model = MatchingSDE(PriorInitDistribution(d).to(dev), p_sde, p_obs,
                        PosteriorEncoder(N, h).to(dev), PosteriorAffine(d, h).to(dev)).to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    for it in range(steps):
        loss = model(xf, ts).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 500 == 0:
            print(f"  it {it:5d}  loss {loss.item():.3f}", flush=True)

    model.eval()
    Be = xe.shape[0]
    with torch.no_grad():
        ctx = model.q_enc(xe)
        outs = [model.q_affine(ctx, tk.repeat(Be, 1)) for tk in torch.linspace(0, 1, T, device=dev)]
        z_hat = torch.stack([o[0] for o in outs])                                # (T,Be,d) posterior mean
        z_cov = torch.diag_embed(torch.stack([o[1] for o in outs]) ** 2)         # (T,Be,d,d) diag posterior cov
        flat = z_hat.reshape(-1, d)
        y_hat = model.p_observe.get_coeffs(flat)[0].reshape(T, Be, N)
        drift = (model.p_sde.drift(flat, None) / Tphys).reshape(T, Be, d)        # -> physical dz/dt
        g = float(model.p_sde.vol(flat, None).mean().item() / (Tphys ** 0.5))
    z_hat, y_hat, drift = z_hat.cpu().numpy(), y_hat.cpu().numpy(), drift.cpu().numpy()
    z_cov = z_cov.cpu().numpy()
    if int(z['window']) > 0:                                                     # Kato: whole-trace, squeeze B=1
        z_hat, y_hat, drift, z_cov = z_hat[:, 0], y_hat[:, 0], drift[:, 0], z_cov[:, 0]
    np.savez(fout, z_hat=z_hat.astype(np.float32), y_hat=y_hat.astype(np.float32),
             drift_at_zhat=drift.astype(np.float32), g=np.float32(g), z_cov=z_cov.astype(np.float32),
             posterior_type="smoother", window_mode="whole", runtime_s=np.float32(time.time() - t0))
    print(f"done: z_hat={z_hat.shape} runtime={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
