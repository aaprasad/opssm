"""Initialization helpers for shortening/removing the EM warmup.

Three pieces, all seeded from the standardized training observations:
  - subspace_id: output-only subspace identification (N4SID / Ho-Kalman style) of an order-d LTI system
      z_{t+1} = A z_t,  y_t = C z_t  -> returns an orthonormal (Stiefel) emission C AND a continuous-time
      drift matrix A_cont = (A_disc - I)/dt. Unlike PCA (a static SVD of the obs), this uses the temporal
      (Hankel) structure, so it also yields the dynamics.
  - linear_dynamics: least-squares A on a latent-score trajectory (used to get an A for the PCA path,
      which has no dynamics of its own).
  - warmstart_drift: pre-fit the (zero-initialized) drift MLP to the linear drift f(z) = A_cont z, so the
      M-step starts from a linear warm-start instead of f = 0.

The realization is defined only up to a similarity transform (the model's arbitrary latent gauge), which is
harmless: eval metrics are Procrustes-aligned and the drift warm-start uses A_cont in the SAME gauge as C.
Assumes fully-observed obs (no mask); the double-well / em_highd data is fully observed.
"""
import torch

from opssm.models.obs import zhat_from_obs


def _future_hankel(Y, k):
    """Y (T,B,D) -> mean-centered future block-Hankel (k*D, N), N = (T-k+1)*B. Column (b,t) stacks
    [y_t, y_{t+1}, ..., y_{t+k-1}] for trajectory b."""
    T, B, D = Y.shape
    cols = [Y[t:t + k].permute(1, 0, 2).reshape(B, k * D) for t in range(T - k + 1)]
    H = torch.cat(cols, dim=0).t()                                  # (k*D, N)
    return H - H.mean(dim=1, keepdim=True)


def subspace_id(full_obs, d, dt, horizon=None):
    """Order-d subspace ID from standardized obs full_obs (T,B,D). The emission C is the first block row of
    the future-Hankel's top-d left singular subspace -- this exploits that the SIGNAL is temporally
    correlated while obs noise is white, so C separates signal from noise better than a static PCA of the
    obs covariance. The dynamics A_cont is then fit by least squares on the subspace scores (noise-robust;
    the deterministic shift-invariance of the observability matrix is biased by process noise for a
    stochastic system). Returns (C (D,d) orthonormal Stiefel, A_cont (d,d) continuous-time drift)."""
    T, B, D = full_obs.shape
    k = horizon or max(2, min(20, T // 5))
    U, _, _ = torch.linalg.svd(_future_hankel(full_obs, k), full_matrices=False)
    C = torch.linalg.qr(U[:D, :d])[0].contiguous()                 # temporally-informed emission (Stiefel)
    ybar = full_obs.reshape(-1, D).mean(0)
    A_cont = linear_dynamics(zhat_from_obs(full_obs, C, ybar), dt)  # A from the subspace scores
    return C, A_cont


def linear_dynamics(z_hat, dt):
    """Least-squares continuous-time drift matrix A_cont=(A_disc-I)/dt from z_{t+1} ~ A_disc z_t on a latent
    trajectory z_hat (T,B,d). Used to seed the drift from PCA scores (PCA gives no dynamics)."""
    d = z_hat.shape[-1]
    Z0, Z1 = z_hat[:-1].reshape(-1, d), z_hat[1:].reshape(-1, d)
    A_disc = torch.linalg.lstsq(Z0, Z1).solution.t()               # Z1 ~ Z0 A_disc^T  ->  (d,d)
    eye = torch.eye(d, device=z_hat.device, dtype=z_hat.dtype)
    return (A_disc - eye) / dt


def warmstart_drift(drift_net, A_cont, *, zmax=2.0, n=4096, steps=400, lr=1e-2):
    """Pre-fit the (zero-init) drift MLP to the linear drift f(z)=A_cont z on z in [-zmax,zmax]^d, in place,
    with a local optimizer (leaves the M-step's dr_opt state fresh). Restores requires_grad. Returns final MSE."""
    dev = next(drift_net.parameters()).device
    A = A_cont.to(dev)
    d = A.shape[0]
    prev = [p.requires_grad for p in drift_net.parameters()]
    drift_net.requires_grad_(True)
    opt = torch.optim.Adam(drift_net.parameters(), lr=lr)
    loss = torch.tensor(0.0)
    for _ in range(steps):
        z = (torch.rand(n, d, device=dev) * 2 - 1) * zmax
        loss = ((drift_net.net(z) - z @ A.t()) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for p, r in zip(drift_net.parameters(), prev):
        p.requires_grad_(r)
    return float(loss.detach())
