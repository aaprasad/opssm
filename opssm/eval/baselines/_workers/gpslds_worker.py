"""gpSLDS / SING-GP baseline worker (JAX / SING). Standalone: run by the jaxbaselines venv as
`python gpslds_worker.py in.npz out.npz`. Needs jax + flax + tfp + the SING checkout (~/src/sing).

Follows the SING authors' own demos, branched by data type:
  - SYNTHETIC (window<=0): their `inference_and_learning` demo -> RBF-kernel GP drift (SING-GP). sigma is set
    to the TRUE diffusion, exactly as the demo does (it passes the data-generating sigma into the fit; SING
    fixes sigma while the paper's gpSLDS learns it). Learns C,d (so the latent gauge self-consistently matches
    sigma), GaussHermite quadrature for prior-drift expectations, small rho-schedule, 50 vEM iters, full-batch
    over a few trials.
  - KATO (window>0): their `neural_data_gpslds` demo -> SSL "smoothly switching linear" kernel (gpSLDS proper),
    sigma=1.0, frozen PCA emission, large rho-schedule, 30 iters, whole trace.
z_hat = variational posterior mean -> a transductive SMOOTHER. Drift via fn.get_posterior_f_mean (== demos).
"""
import sys
import time

sys.path.insert(0, "/home/aaprasad/src/sing")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random as jr, vmap
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from sing.likelihoods import Gaussian
from sing.sde import SparseGP
from sing.kernels import SSL, FullLinear, RBF
from sing.expectation import GaussHermiteQuadrature
from sing.initialization import initialize_params_pca
from sing.sing import fit_variational_em


def inducing_points(key, xs_pca, D, budget=200):
    """Grid over the PCA-latent bounding box for D<=3, else a random subsample of the path (avoids n^D)."""
    lo, hi = xs_pca.min(0), xs_pca.max(0)
    pad = 0.15 * (hi - lo + 1e-6)
    lo, hi = lo - pad, hi + pad
    n_per = {1: 25, 2: 12, 3: 6}.get(D)
    if n_per is not None:
        axes = [jnp.linspace(lo[i], hi[i], n_per) for i in range(D)]
        mesh = jnp.meshgrid(*axes, indexing='ij')
        return jnp.stack([m.ravel() for m in mesh], axis=1)
    idx = jr.choice(key, xs_pca.shape[0], (min(budget, xs_pca.shape[0]),), replace=False)
    return jnp.asarray(xs_pca[idx])


def build_ssl(key, D, num_states, zs, tau_init=0.5):
    """SSL(FullLinear) GP-SDE prior = gpSLDS (Hu 2024), for the Kato neural branch."""
    basis_set = lambda x: jnp.concatenate([jnp.ones(1), x])                    # linear decision boundaries
    kernel = SSL(FullLinear(D), basis_set, D)
    fn = SparseGP(zs, kernel)
    kW, kfp = jr.split(key)
    W = tfd.Normal(0., 1.).sample((D + 1, num_states - 1), seed=kW).astype(jnp.float64)
    fp = tfd.Normal(0., 1.).sample((num_states, D), seed=kfp).astype(jnp.float64)
    kparams = {'linear_params': [{'fixed_point': fp[i], 'log_M': jnp.zeros(D), 'log_noise_var': 0.}
                                 for i in range(num_states)], 'W': W, 'log_tau': jnp.log(tau_init)}
    return fn, kparams


def build_rbf(D, zs):
    """RBF-kernel GP-SDE = SING-GP, for the synthetic branch (their inference_and_learning demo)."""
    n_quad = 5 if D <= 2 else 3            # GaussHermite is n_quad^D points; 5^3=125 is impractically slow,
    fn = SparseGP(zs, RBF(D), GaussHermiteQuadrature(D, n_quad))   # 3^3=27 keeps D=3 tractable
    kparams = {'length_scales': jnp.ones(D), 'output_scale': 1.0}
    return fn, kparams


def main(fin, fout):
    z = np.load(fin)
    xe = jnp.asarray(np.asarray(z['obs_std_eval'], np.float64).transpose(1, 0, 2))  # (Be,T,N)
    dt, D = float(z['dt']), int(z['latent_dim'])
    Be, T, N = xe.shape
    kato = int(z['window']) > 0
    t_grid = jnp.arange(T) * dt
    key = jr.PRNGKey(0)

    if kato:                                                           # neural_data_gpslds demo (SSL / gpSLDS)
        num_states = int(z['num_states']) if 'num_states' in z.files else 4
        sigma = float(z['sigma']) if 'sigma' in z.files else 1.0
        n_iters, n_e, n_m = 30, 15, 50
        rho = jnp.concatenate([jnp.logspace(-1, 0, 10), jnp.ones(n_iters - 10)])
        learn_out, n_use = False, Be
    else:                                                             # inference_and_learning demo (RBF / SING-GP)
        n_iters, n_e, n_m = 50, 10, 50
        rho = jnp.concatenate([jnp.logspace(-3, -2, 10), jnp.ones(n_iters - 10)])
        n_use = min(Be, 8)
        if D >= 3:                                # large-scale (Lorenz): FREEZE the emission gauge so the latents
            learn_out, sigma = False, None        # stay inside the inducing grid (learn-C,d drifts them out -> RBF
        else:                                     # NaN); sigma data-driven below in that fixed gauge (no collapse).
            learn_out = True                      # dw/vdp: learn C,d + true diffusion, exactly as the demo does
            sigma = float(z['sigma']) if 'sigma' in z.files \
                else float(z['sigma_true']) if 'sigma_true' in z.files else 1.0
    xe = xe[:n_use]

    out0, x0 = initialize_params_pca(D, xe)
    C0, d0 = out0['C'], out0['d']
    xs_pca = (xe.reshape(-1, N) - d0) @ C0
    if not kato and sigma is None:                                    # data-driven diffusion in the frozen PCA gauge
        dz = jnp.diff((xe - d0) @ C0, axis=1)
        sigma = 0.5 * float(jnp.sqrt((dz ** 2).sum(-1).mean()) / jnp.sqrt(dt))
    zs = inducing_points(key, xs_pca, D)
    fn, kparams = build_ssl(key, D, num_states, zs) if kato else build_rbf(D, zs)
    init = {'mu0': x0, 'V0': jnp.eye(D)[None].repeat(n_use, 0)}
    lr = 1e-4 * jnp.ones(n_iters)

    t0 = time.time()
    print(f"  fitting {'gpSLDS-SSL(kato)' if kato else 'SING-GP-RBF(synthetic)'}: D={D} trials={n_use} "
          f"n_ind={fn.zs.shape[0]} sigma={sigma:.3f} iters={n_iters} learn_C={learn_out}", flush=True)
    res = fit_variational_em(key, fn, Gaussian(xe, jnp.ones((n_use, T))), t_grid, kparams, init, out0,
                             sigma=sigma, batch_size=None, rho_sched=rho, learning_rate=lr, n_iters=n_iters,
                             n_iters_e=n_e, n_iters_m=n_m, perform_m_step=True,
                             learn_output_params=learn_out, print_interval=max(1, n_iters // 5))
    marg, _, gp_post, kparams_l, _, out_l, _, elbos = res
    C, d_off = np.asarray(out_l['C']), np.asarray(out_l['d'])
    print(f"  fit done, ELBO {float(elbos[0]):.0f} -> {float(elbos[-1]):.0f}", flush=True)

    z_hat = np.asarray(marg['m']).transpose(1, 0, 2)                              # (T,n_use,D)
    z_cov = np.asarray(marg['S']).transpose(1, 0, 2, 3)                           # (T,n_use,D,D) posterior cov
    y_hat = z_hat @ C.T + d_off
    f_mean = lambda x: fn.get_posterior_f_mean(gp_post, kparams_l, x[None])[0]     # == the demos
    drift = np.asarray(vmap(f_mean)(jnp.asarray(z_hat.reshape(-1, D)))).reshape(z_hat.shape)
    if kato:                                                                      # whole-trace, squeeze B=1
        z_hat, y_hat, drift, z_cov = z_hat[:, 0], y_hat[:, 0], drift[:, 0], z_cov[:, 0]
    np.savez(fout, z_hat=z_hat.astype(np.float32), y_hat=y_hat.astype(np.float32),
             drift_at_zhat=drift.astype(np.float32), g=np.float32(sigma), z_cov=z_cov.astype(np.float32),
             posterior_type="smoother", window_mode="whole", runtime_s=np.float32(time.time() - t0))
    print(f"done: z_hat={z_hat.shape} runtime={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
