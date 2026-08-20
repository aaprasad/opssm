"""Linear dynamical system (LDS) baseline worker (JAX / dynamax). Standalone: run by the jaxbaselines
venv as `python lds_worker.py in.npz out.npz`. Needs jax + dynamax (NOT opssm).

The turnkey linear-Gaussian floor: PCA-init emission Cz+d, EM (dynamax LinearGaussianSSM.fit_em) to learn
{A,b,Q,C,d,R} from obs, z_hat = Kalman SMOOTHER means. Labeled "lds" (linear DS), NOT switching rSLDS —
dynamax's SLDS is generative-only (no turnkey posterior EM); this is the linear-dynamics reference point.
posterior_type='smoother'. Effective continuous drift f(z)=((A-I)z+b)/dt for the gauge-aligned drift metric.
"""
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from dynamax.linear_gaussian_ssm import LinearGaussianSSM


def main(fin, fout):
    z = np.load(fin)
    xf = np.asarray(z['obs_std_fit'], np.float32).transpose(1, 0, 2)           # (Bf,T,N)
    xe = np.asarray(z['obs_std_eval'], np.float32).transpose(1, 0, 2)          # (Be,T,N)
    dt, d, noise = float(z['dt']), int(z['latent_dim']), float(z['noise_std_eff'])
    Bf, T, N = xf.shape
    iters = int(z['steps']) if 'steps' in z.files else 60
    key = jax.random.PRNGKey(0)

    Yf = xf.reshape(-1, N)
    Ym = Yf.mean(0)
    _, _, Vt = jnp.linalg.svd(Yf - Ym, full_matrices=False)                    # PCA-init emission (top-d)
    C0 = Vt[:d].T

    lds = LinearGaussianSSM(state_dim=d, emission_dim=N)
    params, props = lds.initialize(
        key, emission_weights=C0, emission_bias=jnp.asarray(Ym),
        emission_covariance=(noise ** 2) * jnp.eye(N),
        dynamics_covariance=0.1 * jnp.eye(d))
    t0 = time.time()
    params, lls = lds.fit_em(params, props, jnp.asarray(xf), num_iters=iters)
    print(f"  EM done: ll {float(lls[0]):.0f} -> {float(lls[-1]):.0f}", flush=True)

    A, b = np.asarray(params.dynamics.weights), np.asarray(params.dynamics.bias)
    C, d_off = np.asarray(params.emissions.weights), np.asarray(params.emissions.bias)
    Q = np.asarray(params.dynamics.cov)

    smooth = jax.jit(jax.vmap(lambda y: lds.smoother(params, y).smoothed_means))
    z_hat = np.asarray(smooth(jnp.asarray(xe))).transpose(1, 0, 2)             # (T,Be,d)
    y_hat = z_hat @ C.T + d_off
    drift = (z_hat @ (A - np.eye(d)).T + b) / dt                              # continuous drift ((A-I)z+b)/dt
    g = float(np.sqrt(np.mean(np.diag(Q)) / dt))                              # Q = g^2 dt I  ->  g
    if int(z['window']) > 0:                                                   # Kato: whole-trace, squeeze B=1
        z_hat, y_hat, drift = z_hat[:, 0], y_hat[:, 0], drift[:, 0]
    np.savez(fout, z_hat=z_hat.astype(np.float32), y_hat=y_hat.astype(np.float32),
             drift_at_zhat=drift.astype(np.float32), g=np.float32(g),
             posterior_type="smoother", window_mode="whole", runtime_s=np.float32(time.time() - t0))
    print(f"done: z_hat={z_hat.shape} runtime={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
