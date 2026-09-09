"""Controlled JAX ablation of Gaussian tails with known double-well dynamics and sensor.

All variants share training/validation trajectories, initialization seeds and collocation draws.
The M-step is excluded to isolate inference. Grid KL is against a discretized filtering oracle;
the legacy density remains globally improper even if finite-grid KL is small.

Run in the JAX environment:
    python scripts/compare_density_tails.py --steps 1000 --seeds 0 1 2 --tail-stds 0 2
"""
import argparse
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from opssm.data.synth_refs import _simulate, _grid_filter_highd
from opssm.models.jax.operator import OperatorFilter
from opssm.models.jax.losses import pinn_zakai_loss, sample_collocation
from opssm.models.jax.mstep import posterior_mean_mala


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--tail-stds', type=float, nargs='+', default=[0., 2.])
    parser.add_argument('--out', type=Path, default=Path('dump/density_tails'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dt, sigma, noise, T, B, nval = .1, .6, .2, 40, 24, 8
    rng = np.random.default_rng(123)
    latent = _simulate('doublewell', 1., B, T, dt, sigma, 10, 1., 0, None, rng)
    obs = (latent + noise * rng.standard_normal(latent.shape)).astype(np.float32)
    mask = np.ones((T, B, 1), np.float32)
    mask[15:20] = 0.                         # a short gap tests uncertain/multimodal predictions
    grid = np.linspace(-3., 3., 401, dtype=np.float32)
    target = _grid_filter_highd(obs[:, :nval], grid, 1., sigma, noise, dt, 10,
                               np.ones(1), np.zeros(1), mask[:, :nval, 0])
    x, m = jnp.asarray(obs[:, nval:]), jnp.asarray(mask[:, nval:])
    xv, mv = jnp.asarray(obs[:, :nval]), jnp.asarray(mask[:, :nval])
    opt = optax.adam(2e-3)
    s = jnp.linspace(0., 1., 3)

    @eqx.filter_jit
    def step(model, state, key):
        kc, kt = jax.random.split(key)
        z, logq = sample_collocation(kc, x, m, 64, .3, 1.6)
        ti = jax.random.permutation(kt, T)[:12]
        def loss(op):
            res, jump, ic, _ = pinn_zakai_loss(op, x, m, z, logq, s,
                lambda z: (z-z**3, (1.-3.*z*z).sum(-1)), sigma,
                lambda z: -.5*(z*z).sum(-1), noise, dt, ti=ti, res_mode='rel')
            return .2*res+jump+ic
        value, grad = eqx.filter_value_and_grad(loss)(model)
        updates, state = opt.update(grad, state, eqx.filter(model, eqx.is_inexact_array))
        return eqx.apply_updates(model, updates), state, value

    def evaluate(model, sample=False):
        logp = np.asarray(model.log_posterior(xv, mv, jnp.asarray(grid)))
        p = np.exp(logp)
        kl = (target*(np.log(np.maximum(target, 1e-30))-logp)).sum(-1)
        mean = (p*grid).sum(-1)
        refmean = (target*grid).sum(-1)
        sd = np.sqrt(np.maximum((p*grid**2).sum(-1)-mean**2, 0.))
        refsd = np.sqrt(np.maximum((target*grid**2).sum(-1)-refmean**2, 0.))
        out = dict(grid_kl=float(kl.mean()), gap_grid_kl=float(kl[15:20].mean()),
                   mean_rmse=float(np.sqrt(np.mean((mean-refmean)**2))),
                   mean_std=float(sd.mean()), oracle_mean_std=float(refsd.mean()))
        if sample:
            sampled, accept = posterior_mean_mala(model, xv, mv, xv, 1.6, jax.random.PRNGKey(99),
                                                  n_chains=64, n_steps=100, rng='crn')
            out['mala_grid_mean_rmse'] = float(np.sqrt(np.mean((np.asarray(sampled)[..., 0]-mean)**2)))
            out['mala_accept'] = accept
            # Fixed contexts: how much additional mass appears as the integration domain grows?
            ctx = model.context(xv, mv)[::10, :1]
            logmass = {}
            for radius in (3., 6., 24., 96.):
                z = np.linspace(-radius, radius, 8193, dtype=np.float32)
                ell = np.asarray(model.log_density(ctx, jnp.asarray(z)))
                peak = ell.max(-1, keepdims=True)
                logmass[str(radius)] = (np.log(np.trapezoid(np.exp(ell-peak), z, axis=-1))
                                        + peak[..., 0]).ravel().tolist()
            out['log_mass_by_radius'] = logmass
        return out

    report = dict(settings=dict(steps=args.steps, seeds=args.seeds, tail_stds=args.tail_stds,
                  data_seed=123, dt=dt, sigma=sigma, noise=noise, T=T, B=B, nval=nval,
                  w_res=.2, n_colloc=64, n_tcoll=12, n_scoll=3, width=32, lr=.002,
                  gap=[15, 20], oracle_grid_size=401, oracle_substeps=10), runs=[])
    for seed in args.seeds:
        for tail_std in args.tail_stds:
            model = OperatorFilter(gru_hidden=32, ctx_dim=32, p=32, branch_hidden=32,
                                   trunk_hidden=32, trunk_layers=2, tail_std=tail_std,
                                   key=jax.random.PRNGKey(seed))
            state = opt.init(eqx.filter(model, eqx.is_inexact_array))
            key = jax.random.PRNGKey(seed+1000)
            history = [dict(step=0, **evaluate(model))]
            start = time.monotonic()
            for i in range(1, args.steps+1):
                key, sk = jax.random.split(key)
                model, state, loss = step(model, state, sk)
                if i % 250 == 0 or i == args.steps:
                    metrics = evaluate(model)
                    history.append(dict(step=i, loss=float(loss), **metrics))
                    print(f'seed={seed} tail_std={tail_std:g} step={i} KL={metrics["grid_kl"]:.4f} '
                          f'gap_KL={metrics["gap_grid_kl"]:.4f} std={metrics["mean_std"]:.4f}', flush=True)
            final = evaluate(model, sample=True)
            row = dict(seed=seed, tail_std=tail_std, seconds=time.monotonic()-start,
                       history=history, final=final)
            report['runs'].append(row)
            eqx.tree_serialise_leaves(args.out / f'seed{seed}_tail{tail_std:g}.eqx', model)
            (args.out / 'results.json').write_text(json.dumps(report, indent=2))
            print(f'finished seed={seed} tail_std={tail_std:g} MALA/grid RMSE='
                  f'{final["mala_grid_mean_rmse"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
