"""visde (Course & Nair / Ilersich) SIMULATION-FREE amortized latent SDE baseline worker.

Standalone: run by the baselines venv as `python visde_worker.py in.npz out.npz` (needs `visde` installed
there, NOT opssm). Adapts visde's LatentSDE (its conv encoder/decoder -> our MLP encoder + LINEAR decoder,
so the emission stays linear-Gaussian like opssm; reuses their MLP drift / diagonal dispersion / GP kernel).
Our data has no parameters/forcing (dim_mu=dim_f=1 dummy constants). z_hat = the amortized posterior mean
(encoder per timestep, the GP's pseudo-obs) => a SMOOTHER; y_hat = linear decode.
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import visde


def _mlp(i, o, h=256, layers=2, act=nn.SiLU):
    net = [nn.Linear(i, h), act()]
    for _ in range(layers - 1):
        net += [nn.Linear(h, h), act()]
    net += [nn.Linear(h, o)]
    return nn.Sequential(*net)


class EncMean(nn.Module):                                  # (mu, x_win (B,n_win,D)) -> z mean (B,d)
    def __init__(self, cfg, mean, std):
        super().__init__()
        self.net = _mlp(cfg.n_win * cfg.dim_x, cfg.dim_z)
        self.register_buffer("m", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("s", torch.as_tensor(std, dtype=torch.float32))

    def forward(self, mu, x_win):
        x = (x_win.reshape(x_win.shape[0], -1) - self.m) / self.s
        return self.net(x)


class FixedVar(nn.Module):                                 # constant per-dim variance (encoder/dispersion)
    def __init__(self, dim, init=-4.0):
        super().__init__()
        self.v = nn.Parameter(torch.full((1, dim), init)); self.act = nn.Softplus()

    def forward(self, *a):
        b = a[-1].shape[0]
        return self.act(self.v).expand(b, -1)


class DecMean(nn.Module):                                  # LINEAR emission C z + d (PCA-init) -> (B,D)
    def __init__(self, cfg, C0):
        super().__init__()
        self.lin = nn.Linear(cfg.dim_z, cfg.dim_x)
        with torch.no_grad():
            self.lin.weight.copy_(torch.as_tensor(C0)); self.lin.bias.zero_()

    def forward(self, mu, z):
        return self.lin(z)


class DecFixedVar(nn.Module):                              # obs noise variance (fixed) -> (B,D)
    def __init__(self, D, noise):
        super().__init__()
        self.register_buffer("v", torch.full((1, D), float(noise) ** 2))

    def forward(self, mu, z):
        return self.v.expand(z.shape[0], -1)


class DriftNet(nn.Module):                                 # (mu,t,z,f) -> dz ; cat(z,f)
    def __init__(self, cfg):
        super().__init__()
        self.net = _mlp(cfg.dim_z + cfg.dim_f, cfg.dim_z, 256, 2, nn.LeakyReLU)

    def forward(self, mu, t, z, f):
        return self.net(torch.cat([z, f], -1))


class KernelNet(nn.Module):                                # scalar time-warp for the GP kernel
    def __init__(self):
        super().__init__()
        self.net = _mlp(1, 1, 128, 2, nn.LeakyReLU)

    def forward(self, t):
        return self.net(t)


def build_model(D, d, dt, noise, C0, n_batch, n_total, device):
    vae = visde.VarAutoencoderConfig(dim_mu=1, dim_x=D, dim_z=d, shape_x=(D,), n_win=1)
    mean, std = 0.0, 1.0                                    # obs already standardized
    encoder = visde.VarEncoderNoPrior(vae, EncMean(vae, mean * np.ones(D), std * np.ones(D)), FixedVar(d))
    decoder = visde.VarDecoderNoPrior(vae, DecMean(vae, C0), DecFixedVar(D, noise))
    dcfg = visde.LatentDriftConfig(dim_mu=1, dim_z=d, dim_f=1)
    drift = visde.LatentDriftNoPrior(dcfg, DriftNet(dcfg))
    pcfg = visde.LatentDispersionConfig(dim_mu=1, dim_z=d)
    dispersion = visde.LatentDispersionNoPrior(pcfg, FixedVar(d, init=-3.0))
    kcfg = visde.LatentVarConfig(dim_mu=1, dim_z=d)
    kernel = visde.DeepGaussianKernel(KernelNet(), n_batch, float(dt))
    latentvar = visde.AmortizedLatentVarGP(kcfg, kernel, encoder)
    cfg = visde.LatentSDEConfig(n_totaldata=n_total, n_samples=1, n_tquad=0,
                                n_warmup=0, n_transition=n_total, lr=1e-3, lr_sched_freq=2000)
    return visde.LatentSDE(config=cfg, encoder=encoder, decoder=decoder, drift=drift,
                           dispersion=dispersion, loglikelihood=visde.LogLikeGaussian(),
                           latentvar=latentvar).to(device)


def main(fin, fout):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(fin)
    xf = np.asarray(z["obs_std_fit"], np.float32)          # (T,Bf,N)
    xe = np.asarray(z["obs_std_eval"], np.float32)         # (T,Be,N)
    dt, d, noise = float(z["dt"]), int(z["latent_dim"]), float(z["noise_std_eff"])
    epochs = int(z["epochs"]) if "epochs" in z.files else 200
    T, Bf, N = xf.shape
    torch.manual_seed(0)

    # data tensors (N_T=Bf trajectories, N=T timesteps, D=N) + dummy mu/forcing
    U = torch.tensor(xf.transpose(1, 0, 2))                # (Bf,T,N)
    M = torch.zeros(Bf, 1); F = torch.zeros(Bf, T, 1); Tt = (torch.arange(T) * dt).expand(Bf, T).clone()
    data = visde.MultiEvenlySpacedTensors(M, Tt, U, F, 1)
    n_batch = min(64, Bf * T)
    sampler = visde.MultiTemporalSampler(data, n_batch, n_repeats=1)
    loader = DataLoader(data, batch_sampler=sampler, num_workers=0)

    Yf = xf.reshape(-1, N); _, _, Vt = np.linalg.svd(Yf - Yf.mean(0), full_matrices=False)
    C0 = Vt[:d]                                            # (d,N) -> nn.Linear weight (out=N,in=d) is C0.T? see DecMean
    model = build_model(N, d, dt, noise, C0.T.copy(), n_batch, Bf * T, dev)

    trainer = pl.Trainer(accelerator=dev.type, max_epochs=epochs, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False, num_sanity_val_steps=0)
    t0 = time.time()
    trainer.fit(model, loader)

    # amortized posterior mean at each EVAL timestep (encoder per frame) = z_hat (smoother)
    model.eval().to(dev)
    Te, Be, _ = xe.shape
    xw = torch.tensor(xe.reshape(Te * Be, 1, N), device=dev)   # (Te*Be,1,N) n_win=1
    mu = torch.zeros(Te * Be, 1, device=dev)
    with torch.no_grad():
        zmean, _ = model.encoder(mu, xw)                       # (Te*Be,d)
        yhat, _ = model.decoder(mu, zmean)                     # (Te*Be,N)
        tt = torch.zeros(Te * Be, 1, device=dev); ff = torch.zeros(Te * Be, 1, device=dev)
        drift = model.drift(mu, tt, zmean, ff)                 # prior drift at z_hat (wrapper forward)
        g = float(model.dispersion(mu, tt).mean())             # isotropic diffusion coefficient
    z_hat = zmean.cpu().numpy().reshape(Te, Be, d)
    y_hat = yhat.cpu().numpy().reshape(Te, Be, N)
    drift = drift.cpu().numpy().reshape(Te, Be, d)
    if int(z["window"]) > 0:                                   # Kato: squeeze B=1
        z_hat, y_hat, drift = z_hat[:, 0], y_hat[:, 0], drift[:, 0]
    np.savez(fout, z_hat=z_hat.astype(np.float32), y_hat=y_hat.astype(np.float32),
             drift_at_zhat=drift.astype(np.float32), g=np.float32(g),
             posterior_type="smoother", window_mode="whole", runtime_s=np.float32(time.time() - t0))
    print(f"done: z_hat={z_hat.shape} runtime={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
