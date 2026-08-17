"""Vanilla SIMULATION-BASED latent SDE (Li et al. 2020, torchsde) baseline worker.

Standalone: run by the baselines venv as `python latent_sde_worker.py in.npz out.npz`. Only torch/torchsde
+ numpy (NOT opssm). Amortized VI with SDE solves in the E-step (this is the sim-BASED foil to opssm/visde,
which are simulation-free). Linear-Gaussian emission y = C z + d (PCA-init C), diagonal diffusion, MLP
prior/posterior drifts, backward-GRU amortized posterior (=> a SMOOTHER: uses future obs).
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchsde


def _mlp(i, o, h=128, layers=2):
    net = [nn.Linear(i, h), nn.SiLU()]
    for _ in range(layers - 1):
        net += [nn.Linear(h, h), nn.SiLU()]
    net += [nn.Linear(h, o)]
    return nn.Sequential(*net)


class LatentSDE(nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"                                       # additive (state-independent) diffusion: Ito == Stratonovich

    def __init__(self, N, d, dt, noise_std, C0, hid=128, ctx=64):
        super().__init__()
        self.d, self.dt, self.noise_std = d, float(dt), float(noise_std)
        self.enc = nn.GRU(N, ctx)                          # backward GRU -> per-time context (smoother)
        self.qz0 = _mlp(ctx, 2 * d, hid, 1)                # posterior over z0 from ctx[0]
        self.post = _mlp(d + ctx, d, hid, 2)               # posterior/approx drift f(z, ctx_t)
        self.prior = _mlp(d, d, hid, 2)                    # prior drift h(z)
        self.logg = nn.Parameter(torch.full((d,), -1.0))   # diagonal diffusion
        self.dec = nn.Linear(d, N)                          # emission C z + d
        with torch.no_grad():
            self.dec.weight.copy_(torch.as_tensor(C0))     # PCA-init emission
            self.dec.bias.zero_()
        self.prior_std = 1.0
        self._ctx = self._ts = None

    def contextualize(self, ctx, ts):
        self._ctx, self._ts = ctx, ts                      # ctx (T,B,C), ts (T,)

    def _ctx_at(self, t):
        i = torch.searchsorted(self._ts, t.detach()).clamp(0, self._ts.numel() - 1)
        return self._ctx[i]

    def f(self, t, z):                                     # posterior drift (data-dependent)
        return self.post(torch.cat([z, self._ctx_at(t)], -1))

    def h(self, t, z):                                     # prior drift
        return self.prior(z)

    def g(self, t, z):
        return self.logg.exp().expand(z.shape[0], self.d)

    def elbo(self, ys, ts, kl_w=1.0):
        ctx, _ = self.enc(torch.flip(ys, [0]))             # backward in time
        ctx = torch.flip(ctx, [0])                         # (T,B,C)
        self.contextualize(ctx, ts)
        q = self.qz0(ctx[0]); qm, qlogs = q[:, :self.d], q[:, self.d:]
        z0 = qm + torch.randn_like(qm) * qlogs.exp()
        zs, logqp = torchsde.sdeint(self, z0, ts, method="euler", dt=self.dt, logqp=True)  # (T,B,d),(T-1,B)
        y_hat = self.dec(zs)
        ll = (-0.5 * ((ys - y_hat) / self.noise_std) ** 2 - np.log(self.noise_std)).sum((0, 2)).mean()
        klz0 = (0.5 * (qm ** 2 + (2 * qlogs).exp()) / self.prior_std ** 2 - qlogs).sum(-1).mean()
        klpath = logqp.sum(0).mean()
        return -(ll - kl_w * (klpath + klz0)), zs, y_hat

    @torch.no_grad()
    def infer(self, ys, ts):
        ctx, _ = self.enc(torch.flip(ys, [0])); ctx = torch.flip(ctx, [0])
        self.contextualize(ctx, ts)
        z0 = self.qz0(ctx[0])[:, :self.d]                  # posterior mean of z0
        zs = torchsde.sdeint(self, z0, ts, method="euler", dt=self.dt)   # (T,B,d)
        return zs, self.dec(zs)


def main(fin, fout):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(fin)
    xf = torch.tensor(z["obs_std_fit"], dtype=torch.float32, device=dev)      # (T,Bf,N)
    xe = torch.tensor(z["obs_std_eval"], dtype=torch.float32, device=dev)     # (T,Be,N)
    dt, d, noise = float(z["dt"]), int(z["latent_dim"]), float(z["noise_std_eff"])
    T, Bf, N = xf.shape
    steps = int(z["steps"]) if "steps" in z.files else 1500
    torch.manual_seed(0)

    # PCA-init emission C0 (N,d) from the fit obs
    Yf = xf.reshape(-1, N)
    _, _, Vt = torch.linalg.svd(Yf - Yf.mean(0), full_matrices=False)
    C0 = Vt[:d].t().contiguous()                                            # (N,d)

    model = LatentSDE(N, d, dt, noise, C0).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    ts_f = torch.arange(T, device=dev, dtype=torch.float32) * dt
    t0 = time.time()
    for it in range(steps):
        opt.zero_grad()
        kl_w = min(1.0, it / (steps * 0.3))                                 # KL warmup
        loss, _, _ = model.elbo(xf, ts_f, kl_w)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
        if it % 300 == 0:
            print(f"  it {it:5d}  elbo_loss {loss.item():.2f}", flush=True)

    # inference on EVAL split (full trace for Kato -> (Te,1,N))
    Te = xe.shape[0]
    ts_e = torch.arange(Te, device=dev, dtype=torch.float32) * dt
    zs, yh = model.infer(xe, ts_e)                                          # (Te,Be,d),(Te,Be,N)
    with torch.no_grad():
        drift = model.prior(zs)                                             # prior drift at z_hat
    z_hat = zs.cpu().numpy(); y_hat = yh.cpu().numpy(); drift = drift.cpu().numpy()
    windowed = int(z["window"]) > 0
    if windowed:                                                           # Kato: squeeze B=1 full trace
        z_hat, y_hat, drift = z_hat[:, 0], y_hat[:, 0], drift[:, 0]
    np.savez(fout, z_hat=z_hat.astype(np.float32), y_hat=y_hat.astype(np.float32),
             drift_at_zhat=drift.astype(np.float32), g=np.float32(model.logg.exp().mean().item()),
             posterior_type="smoother", window_mode=("whole" if windowed else "whole"),
             runtime_s=np.float32(time.time() - t0))
    print(f"done: z_hat={z_hat.shape} y_hat={y_hat.shape} runtime={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
