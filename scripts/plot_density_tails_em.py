"""Plot completed learned-dynamics tail comparisons, including the saved drift functions.

Run in the JAX environment:
    python scripts/plot_density_tails_em.py dump/density_tails_em/results.json
"""
import argparse
import csv
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from opssm.models.jax.dynamics import DriftNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--compare-results', type=Path, action='append', default=[],
                        help='Overlay another completed experiment with matching training settings')
    args = parser.parse_args()
    settings = json.loads(args.results.read_text())['settings']
    groups = {}
    comparable = lambda s: {k: v for k, v in s.items() if k not in ('trunk_activation', 'tail_stds')}
    for path in [args.results, *args.compare_results]:
        report = json.loads(path.read_text())
        current, runs = report['settings'], report['runs']
        if not current['learn_dynamics']:
            parser.error('this plot requires a --learn-dynamics experiment')
        if len(runs) != len(current['seeds'])*len(current['tail_stds']):
            parser.error('comparison is incomplete; wait for all runs')
        if comparable(current) != comparable(settings):
            parser.error('overlaid experiments must have matching data, training and device settings')
        for tail in current['tail_stds']:
            key = (current.get('trunk_activation', 'tanh'), tail)
            if key in groups:
                parser.error(f'duplicate activation/tail variant: {key}')
            groups[key] = (path.parent, [r for r in runs if r['tail_std'] == tail])
    out = args.results.parent
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    field_z = np.linspace(-1.5, 1.5, 301, dtype=np.float32)
    metrics = [('grid_kl', 'Filtering KL (finite grid)'), ('drift_rel', 'Relative drift error'),
               ('g', 'Learned diffusion g')]
    colors = plt.get_cmap('tab10')
    summary = []
    em = settings['em']
    skeleton = DriftNet(em['drift_width'], layers=em['drift_layers'], key=jax.random.PRNGKey(0))
    for idx, ((activation, tail), (source, subset)) in enumerate(sorted(groups.items())):
        label = activation + (' (no fixed tail)' if tail == 0 else f' + Gaussian (std={tail:g})')
        color = colors(idx)
        for ax, (key, ylabel) in zip(axes.flat, metrics):
            steps = np.array([h['step'] for h in subset[0]['history']])
            values = np.array([[h[key] for h in r['history']] for r in subset])
            for value in values:
                ax.plot(steps, value, color=color, alpha=.2, linewidth=.8)
            ax.plot(steps, values.mean(0), label=label, color=color, linewidth=2)
            ax.set(xlabel='Operator training step', ylabel=ylabel)
        fields = []
        for run in subset:
            name = f'seed{run["seed"]}_tail{tail:g}_drift.eqx'
            drift = eqx.tree_deserialise_leaves(source / name, skeleton)
            fields.append(np.asarray(drift.net(jnp.asarray(field_z[:, None])))[:, 0])
            summary.append(dict(seed=run['seed'], trunk_activation=activation, tail_std=tail,
                                **{k: v for k, v in run['final'].items() if isinstance(v, (int, float))}))
        fields = np.asarray(fields)
        axes[1, 1].plot(field_z, fields.mean(0), color=color, label=label, linewidth=2)
        axes[1, 1].fill_between(field_z, fields.min(0), fields.max(0), color=color, alpha=.15)
    axes[0, 0].set_yscale('log')
    axes[1, 0].axhline(settings['sigma'], color='black', linestyle='--',
                       label=f'True g={settings["sigma"]:g}')
    axes[1, 1].plot(field_z, field_z-field_z**3, color='black', linestyle='--', label='True drift')
    axes[1, 1].set(xlabel='Latent state z', ylabel='Drift f(z)')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Learned drift and diffusion: density representation comparison\n'
                 f'{len(settings["seeds"])} initialization seeds; shared data and fixed direct sensor')
    fig.savefig(out / 'learned_dynamics.png', dpi=180)
    fig.savefig(out / 'learned_dynamics.pdf')
    plt.close(fig)
    with (out / 'summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for activation, tail in sorted(groups):
        rows = [r for r in summary if r['tail_std'] == tail and r['trunk_activation'] == activation]
        print(f'activation={activation} tail_std={tail:g}: ' + ', '.join(
            f'{key}={np.mean([r[key] for r in rows]):.5f}'
            for key in ('grid_kl', 'gap_grid_kl', 'latent_rmse', 'drift_rel', 'g', 'g_abs_error')))
    print(f'Saved learned_dynamics.png/.pdf and summary.csv under {out}')


if __name__ == '__main__':
    main()
