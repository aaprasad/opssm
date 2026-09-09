"""Controlled JAX ablation of Gaussian tails with fixed or learned double-well dynamics.

All variants share training/validation trajectories, initialization seeds and collocation draws.
By default the M-step is excluded to isolate inference; --learn-dynamics enables the existing drift
and diffusion M-step while retaining the known direct sensor. Grid KL is against a filtering oracle;
the legacy density remains globally improper even if finite-grid KL is small.

Run in the JAX environment:
    python scripts/compare_density_tails.py --steps 1000 --seeds 0 1 2 --tail-stds 0 2
    python scripts/compare_density_tails.py --learn-dynamics --steps 6000 --out dump/density_tails_em
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
from opssm.models.jax.mstep import mstep
from opssm.models.jax.dynamics import DriftNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--tail-stds', type=float, nargs='+', default=[0., 2.])
    parser.add_argument('--trunk-activation', choices=['tanh', 'softplus'], default='tanh')
    parser.add_argument('--learn-dynamics', action='store_true')
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--m-every', type=int, default=500)
    parser.add_argument('--m-inner', type=int, default=200)
    parser.add_argument('--out', type=Path, default=Path('dump/density_tails'))
    args = parser.parse_args()
    if args.m_every <= 0 or args.m_inner <= 0 or args.warmup < 0:
        parser.error('m-every/m-inner must be positive and warmup must be nonnegative')
    first_mstep = max(args.m_every, ((args.warmup + args.m_every - 1)//args.m_every)*args.m_every)
    if args.learn_dynamics and args.steps <= first_mstep:
        parser.error('learned dynamics needs steps beyond the first M-step; try --steps 6000')
    if (args.out / 'results.json').exists():
        parser.error('output already contains results; choose a fresh --out directory')
    args.out.mkdir(parents=True, exist_ok=True)
    devices = [f'{device.platform}: {device.device_kind}' for device in jax.devices()]
    print(f'JAX backend={jax.default_backend()} devices={devices}', flush=True)
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
    drift_opt = optax.adam(2e-3)
    s = jnp.linspace(0., 1., 3)

    @eqx.filter_jit
    def step(model, state, key, drift_net, g):
        kc, kt = jax.random.split(key)
        z, logq = sample_collocation(kc, x, m, 64, .3, 1.6)
        ti = jax.random.permutation(kt, T)[:12]
        def loss(op):
            drift = drift_net.drift if args.learn_dynamics else lambda z: (z-z**3, (1.-3.*z*z).sum(-1))
            res, jump, ic, _ = pinn_zakai_loss(op, x, m, z, logq, s, drift,
                g if args.learn_dynamics else sigma,
                lambda z: -.5*(z*z).sum(-1), noise, dt, ti=ti, res_mode='rel')
            return .2*res+jump+ic
        value, grad = eqx.filter_value_and_grad(loss)(model)
        updates, state = opt.update(grad, state, eqx.filter(model, eqx.is_inexact_array))
        return eqx.apply_updates(model, updates), state, value

    def evaluate(model, drift_net, g, sample=False):
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
        if args.learn_dynamics:
            # Evaluate at true held-out states, in the known sensor's coordinates. No affine alignment
            # can hide a drift error here. Restrict to the well-sampled region for the main drift score.
            zv = latent[:, :nval].reshape(-1, 1)
            zv = zv[np.abs(zv[:, 0]) <= 1.5]
            ftrue = zv-zv**3
            fpred = np.asarray(drift_net.net(jnp.asarray(zv)))
            rmse = float(np.sqrt(np.mean((fpred-ftrue)**2)))
            out.update(drift_rmse=rmse, drift_rel=rmse/float(np.sqrt(np.mean(ftrue**2))),
                       g=float(g), g_abs_error=abs(float(g)-sigma), g_ratio=float(g)/sigma,
                       latent_rmse=float(np.sqrt(np.mean((mean-latent[:, :nval, 0])**2))))
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
                  trunk_activation=args.trunk_activation,
                  backend=jax.default_backend(), devices=devices,
                  data_seed=123, dt=dt, sigma=sigma, noise=noise, T=T, B=B, nval=nval,
                  w_res=.2, n_colloc=64, n_tcoll=12, n_scoll=3, width=32, lr=.002,
                  gap=[15, 20], oracle_grid_size=401, oracle_substeps=10,
                  learn_dynamics=args.learn_dynamics, learn_obs=False,
                  em=dict(warmup=args.warmup, m_every=args.m_every, m_inner=args.m_inner,
                          drift_width=32, drift_layers=2, drift_lr=.002, reg_lambda=.0003,
                          g_init=1., drift_init='zero', drift_target='det_mid',
                          mala_chains=64, mala_steps=30, mala_rng='stochastic',
                          skip_final_mstep=True) if args.learn_dynamics else None), runs=[])
    for seed in args.seeds:
        for tail_std in args.tail_stds:
            model = OperatorFilter(gru_hidden=32, ctx_dim=32, p=32, branch_hidden=32,
                                   trunk_hidden=32, trunk_layers=2, tail_std=tail_std,
                                   trunk_activation=args.trunk_activation,
                                   key=jax.random.PRNGKey(seed))
            state = opt.init(eqx.filter(model, eqx.is_inexact_array))
            drift_net = DriftNet(32, layers=2, key=jax.random.fold_in(jax.random.PRNGKey(seed), 17))
            drift_state = drift_opt.init(eqx.filter(drift_net.net, eqx.is_inexact_array))
            g = 1. if args.learn_dynamics else sigma
            key = jax.random.PRNGKey(seed+1000)
            history = [dict(step=0, **evaluate(model, drift_net, g))]
            m_history = []
            start = time.monotonic()
            for i in range(1, args.steps+1):
                key, sk = jax.random.split(key)
                model, state, loss = step(model, state, sk, drift_net, jnp.asarray(g))
                if args.learn_dynamics and i >= args.warmup and i % args.m_every == 0 and i < args.steps:
                    # Separate keys leave the E-step's random draws unchanged by M-step scheduling.
                    mk = jax.random.fold_in(jax.random.PRNGKey(seed+2000), i)
                    info, drift_net, drift_state = mstep(model, x, m, dt, drift_net, drift_opt,
                        drift_state, mk, learn_g=True, reg_lambda=.0003, m_inner=args.m_inner,
                        learn_obs=False, mean_method='mala', broad_std=1.6, drift_target='det_mid',
                        mala=dict(n_chains=64, n_steps=30, rng='stochastic'))
                    g = info['g_cur']
                    m_history.append(dict(step=i, g=g, accept=info['accept']))
                if i % 250 == 0 or i == args.steps:
                    metrics = evaluate(model, drift_net, g)
                    history.append(dict(step=i, loss=float(loss), **metrics))
                    print(f'seed={seed} tail_std={tail_std:g} step={i} KL={metrics["grid_kl"]:.4f} '
                          f'gap_KL={metrics["gap_grid_kl"]:.4f} std={metrics["mean_std"]:.4f}'
                          + (f' drift_rel={metrics["drift_rel"]:.3f} g={g:.3f}' if args.learn_dynamics else ''),
                          flush=True)
            final = evaluate(model, drift_net, g, sample=True)
            row = dict(seed=seed, tail_std=tail_std, seconds=time.monotonic()-start,
                       history=history, msteps=m_history, final=final)
            report['runs'].append(row)
            eqx.tree_serialise_leaves(args.out / f'seed{seed}_tail{tail_std:g}.eqx', model)
            if args.learn_dynamics:
                eqx.tree_serialise_leaves(args.out / f'seed{seed}_tail{tail_std:g}_drift.eqx', drift_net)
            (args.out / 'results.json').write_text(json.dumps(report, indent=2))
            print(f'finished seed={seed} tail_std={tail_std:g} MALA/grid RMSE='
                  f'{final["mala_grid_mean_rmse"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
