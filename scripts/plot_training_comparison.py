"""Compare completed train_jax.py activation runs using their standard validation metrics."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads((path / 'result.json').read_text()) for path in args.runs]
    reference = None
    for report in reports:
        if report['status'] != 'done':
            parser.error('all runs must have completed successfully')
        comparable = {key: report[key] for key in ('seed', 'n_steps', 'data_eff')}
        for key in ('hparams', 'data'):
            comparable[key] = {k: v for k, v in report[key].items()
                               if k not in ('train_dir', 'trunk_activation')}
        if reference is not None and comparable != reference:
            parser.error('runs must match in data, seed and training settings apart from activation')
        reference = comparable

    args.out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    metrics = [('drift_rel', 'Relative drift error (aligned)'),
               ('g_aln', 'Diffusion (aligned)'), ('kl_aln', 'Filtering KL (aligned)'),
               ('lat_rel', 'Relative latent error (aligned)')]
    summary = []
    for path, report in zip(args.runs, reports):
        activation = report['hparams'].get('trunk_activation', 'tanh')
        with (path / 'metrics.csv').open() as file:
            rows = list(csv.DictReader(file))
        if int(rows[-1]['step']) != report['n_steps']:
            parser.error(f'{path}: final metrics do not match the requested training length')
        # Step zero has a random density; fit-derived alignment can be singular there.
        trained = [r for r in rows if int(r['step']) > 0]
        for ax, (key, label) in zip(axes.flat, metrics):
            ax.plot([int(r['step']) for r in trained], [float(r[key]) for r in trained],
                    marker='o', markersize=4, label=activation)
            ax.set(xlabel='Training step', ylabel=label)
        summary.append(dict(activation=activation, **report['final']))
    axes[0, 1].axhline(reports[0]['data']['sigma'], color='black', linestyle='--', label='True diffusion')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'train_jax.py / em_highd: activation comparison (seed {reports[0]["seed"]})')
    for extension in ('png', 'pdf'):
        fig.savefig(args.out / f'comparison.{extension}', dpi=180)
    plt.close(fig)
    with (args.out / 'summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
