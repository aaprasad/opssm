"""EKF-EM baseline worker (JAX / dynamax). Standalone: run by the jaxbaselines venv as
`python ekf_worker.py in.npz out.npz`. Needs jax + dynamax + optax (NOT opssm).

Learns a latent nonlinear SDE from obs, EKF-style: MLP drift f + linear-Gaussian emission Cz+d, discretized
as a nonlinear-Gaussian SSM (Euler: z_{t+1}=z_t+dt*f(z_t)+N(0,g^2 dt I), y=Cz+d+N(0,noise^2 I)), fit by optax
on dynamax's EKF marginal log-likelihood (fit-from-obs, no ground truth). z_hat = EKF filtered means (a causal
FILTER, like opssm). Also benchmarks JAX runtime vs our torch stack.
"""
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
from dynamax.nonlinear_gaussian_ssm import ParamsNLGSSM, extended_kalman_filter as ekf


def mlp_init(key, sizes):
    ps = []
    for i in range(len(sizes) - 1):
        key, k = jax.random.split(key)
        ps.append((jax.random.normal(k, (sizes[i], sizes[i + 1])) * 0.1, jnp.zeros(sizes[i + 1])))
    ps[-1] = (jnp.zeros_like(ps[-1][0]), ps[-1][1])              # zero last layer -> f~0 at start (like opssm)
    return ps


def mlp(ps, z):
    h = z
    for w, b in ps[:-1]:
        h = jnp.tanh(h @ w + b)
    w, b = ps[-1]
    return h @ w + b


def _params(theta, dt, d, N):
    g2, noise2 = jnp.exp(2 * theta['log_g']), jnp.exp(2 * theta['log_noise'])
    return ParamsNLGSSM(
        initial_mean=jnp.zeros(d), initial_covariance=jnp.eye(d),
        dynamics_function=lambda z: z + dt * mlp(theta['drift'], z),          # Euler step of dz = f dt
        dynamics_covariance=g2 * dt * jnp.eye(d),
        emission_function=lambda z: theta['C'] @ z + theta['d'],
        emission_covariance=noise2 * jnp.eye(N))


def main(fin, fout):
    z = np.load(fin)
    xf = np.asarray(z['obs_std_fit'], np.float32).transpose(1, 0, 2)          # (Bf,T,N) per-trajectory
    xe = np.asarray(z['obs_std_eval'], np.float32).transpose(1, 0, 2)         # (Be,T,N)
    dt, d, noise = float(z['dt']), int(z['latent_dim']), float(z['noise_std_eff'])
    Bf, T, N = xf.shape
    steps = int(z['steps']) if 'steps' in z.files else 800
    key = jax.random.PRNGKey(0)

    Yf = xf.reshape(-1, N)
    _, _, Vt = jnp.linalg.svd(Yf - Yf.mean(0), full_matrices=False)
    theta = dict(drift=mlp_init(key, [d, 64, 64, d]), C=Vt[:d].T, d=jnp.zeros(N),
                 log_g=jnp.array(0.0), log_noise=jnp.array(np.log(noise)))

    def neg_ll(th, ys):                                                       # ys (B,T,N) -> -sum marginal loglik
        return -jnp.sum(jax.vmap(lambda y: ekf(_params(th, dt, d, N), y).marginal_loglik)(ys))

    opt = optax.adam(3e-3)
    state = opt.init(theta)
    vg = jax.jit(jax.value_and_grad(lambda th: neg_ll(th, jnp.asarray(xf))))
    t0 = time.time()
    for it in range(steps):
        loss, grad = vg(theta)
        upd, state = opt.update(grad, state)
        theta = optax.apply_updates(theta, upd)
        if it % 200 == 0:
            print(f"  it {it:4d}  nll {float(loss):.1f}", flush=True)

    infer = jax.jit(jax.vmap(lambda y: ekf(_params(theta, dt, d, N), y).filtered_means))
    z_hat = np.asarray(infer(jnp.asarray(xe))).transpose(1, 0, 2)             # (T,Be,d)
    C, d_off = np.asarray(theta['C']), np.asarray(theta['d'])
    y_hat = z_hat @ C.T + d_off
    drift = np.asarray(jax.vmap(lambda zz: mlp(theta['drift'], zz))(
        jnp.asarray(z_hat.reshape(-1, d)))).reshape(z_hat.shape)
    g = float(np.exp(theta['log_g']))
    if int(z['window']) > 0:                                                  # Kato: whole-trace filter, squeeze B=1
        z_hat, y_hat, drift = z_hat[:, 0], y_hat[:, 0], drift[:, 0]
    np.savez(fout, z_hat=z_hat.astype(np.float32), y_hat=y_hat.astype(np.float32),
             drift_at_zhat=drift.astype(np.float32), g=np.float32(g),
             posterior_type="filter", window_mode="whole", runtime_s=np.float32(time.time() - t0))
    print(f"done: z_hat={z_hat.shape} runtime={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
