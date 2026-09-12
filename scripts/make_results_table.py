"""Compile benchmark result.json files into a LaTeX table (dump/baselines style).

Regenerates from the results themselves, so rerunning it after a sweep lands picks up the new
numbers. Synthetic and real datasets carry different metrics, so the columns each dataset cannot
produce are rendered `--` rather than silently omitted.

    python scripts/make_results_table.py --out dump/benchmark_table/results.tex

Multiple --results roots are merged; later roots win on a (dataset, model) collision, which is how
a rerun of one method is folded in without disturbing the rest. Use --label to rename a model
within one root (e.g. to distinguish two checkpoint-selection rules).
"""
import argparse
from collections import OrderedDict
import glob
import json
from pathlib import Path

# (key, header, direction). "max"/"min" pick the starred best; "target" is scored by |value - goal|.
COLUMNS = [
    ("recon_r2", r"recon\,$R^2\!\uparrow$", "max"),
    ("latent_nrmse", r"lat$_{\rm nrmse}\!\downarrow$", "min"),
    ("dynamics_nrmse", r"drift$_{\rm nrmse}\!\downarrow$", "min"),
    ("forecast_rmse", r"fcst RMSE$\downarrow$", "min"),
    ("marginal_95_coverage", r"cov$_{95}\!\to\!0.95$", "target"),
    ("linear_decode_balanced_accuracy", r"decode$\uparrow$", "max"),
    ("best_step", "step", None),
]
COVERAGE_GOAL = .95
SMOOTHERS = {"latent_sde", "sde_matching"}
DISPLAY = {"kf": "KF", "ekf": "EKF", "ukf": "UKF", "slds": "SLDS", "rslds": "rSLDS",
           "latent_sde": "latent-SDE", "sde_matching": "SDE-match"}
DATASET_DISPLAY = {"doublewell": "double-well", "vanderpol": "van der Pol", "lorenz": "Lorenz",
                   "kato_WT_NoStim_worm0": "Kato no-stim w0"}
ORDER = ["kf", "ekf", "ukf", "slds", "rslds", "latent_sde", "sde_matching"]

# Trivial references a method must beat before "best in column" means anything. Without these a
# table happily stars a negative R^2, or a forecast worse than repeating the last observation.
REFERENCES = {
    "recon_r2": ("predict the channel mean", lambda row: 0.),
    "forecast_rmse": ("last observation", lambda row: row.get("forecast_persistence_rmse")),
    "linear_decode_balanced_accuracy": ("majority class", lambda row: row.get("linear_decode_majority_accuracy")),
}


def harmonize(row):
    """One metric name per quantity, since the synthetic and Kato scorers report different keys."""
    out = dict(row)
    out["recon_r2"] = row.get("clean_recon_r2", row.get("observation_recon_r2"))
    out["forecast_rmse"] = row.get("forecast_clean_rmse", row.get("forecast_observation_rmse"))
    return out


def collect(roots, labels):
    results = OrderedDict()
    for root in roots:
        rename = labels.get(str(root), {})
        for path in sorted(glob.glob(f"{root}/*/seed_*/*/result.json")):
            row = json.loads(Path(path).read_text())
            if row.get("status") != "ok" or row.get("smoke"):
                continue
            model = rename.get(row["model"], row["model"])
            results[(row["dataset"], model)] = harmonize(row) | {"model": model}
    return results


def fmt(value, key):
    if not isinstance(value, (int, float)):
        return "--"
    if key == "best_step":
        return f"{int(value)}"
    return f"{value:.3f}" if abs(value) < 100 else f"{value:.1f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", type=Path, required=True, help="Result roots to merge")
    ap.add_argument("--label", action="append", default=[], metavar="ROOT:MODEL=NEWNAME",
                    help="Rename a model within one root; repeatable")
    ap.add_argument("--out", type=Path, default=Path("dump/benchmark_table/results.tex"))
    ap.add_argument("--highlight", default="opssm", help="Substring marking the rows to bold")
    ap.add_argument("--caption", default="")
    args = ap.parse_args(argv)

    labels = {}
    for item in args.label:
        root, _, pair = item.partition(":")
        model, _, new = pair.partition("=")
        labels.setdefault(str(Path(root)), {})[model] = new
    results = collect(args.results, labels)
    if not results:
        raise SystemExit(f"No completed results under {[str(r) for r in args.results]}")

    datasets = list(OrderedDict.fromkeys(d for d, _ in results))
    datasets.sort(key=lambda d: (d.startswith("kato"), list(DATASET_DISPLAY).index(d)
                                 if d in DATASET_DISPLAY else 99))
    # Star the best value per (dataset, metric); coverage is scored by distance to its nominal level.
    # A column whose winner fails to beat its trivial reference gets no star at all -- see REFERENCES.
    best, references = {}, {}
    for dataset in datasets:
        rows = [results[(d, m)] for (d, m) in results if d == dataset]
        for key, source in ((k, v[1]) for k, v in REFERENCES.items()):
            values = [source(row) for row in rows]
            values = [v for v in values if isinstance(v, (int, float))]
            if values:
                references[(dataset, key)] = values[0]
        for key, _, direction in COLUMNS:
            if direction is None:
                continue
            values = [(m, results[(d, m)].get(key)) for (d, m) in results
                      if d == dataset and isinstance(results[(d, m)].get(key), (int, float))]
            if not values:
                continue
            if direction == "target":
                winner, value = min(values, key=lambda kv: abs(kv[1] - COVERAGE_GOAL))
            else:
                winner, value = (max if direction == "max" else min)(values, key=lambda kv: kv[1])
            reference = references.get((dataset, key))
            if reference is not None and (value <= reference if direction == "max" else value >= reference):
                continue          # nothing here beat the trivial baseline; starring it would mislead
            best[(dataset, key)] = winner

    lines = [r"\documentclass{article}", r"\usepackage{booktabs,multirow,amssymb,graphicx}",
             r"\usepackage[margin=0.6in]{geometry}", r"\begin{document}",
             r"\begin{table}[t]\centering", r"\caption{" + (args.caption or default_caption()) + "}",
             r"\label{tab:benchmarks}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{ll" + "c" * len(COLUMNS) + "}", r"\toprule",
             "Dataset & Model & " + " & ".join(h for _, h, _ in COLUMNS) + r" \\", r"\midrule"]

    for index, dataset in enumerate(datasets):
        models = [m for m in ORDER if (dataset, m) in results]
        models += [m for (d, m) in results if d == dataset and m not in models]
        if index:
            lines.append(r"\midrule")
        label = DATASET_DISPLAY.get(dataset, dataset.replace("_", r"\_"))
        for position, model in enumerate(models):
            row = results[(dataset, model)]
            head = rf"\multirow{{{len(models)}}}{{*}}{{{label}}}" if position == 0 else ""
            tag = "S" if model in SMOOTHERS else "F"
            name = DISPLAY.get(model, model).replace("_", r"\_")
            mine = args.highlight in model
            cells = []
            for key, _, direction in COLUMNS:
                text = fmt(row.get(key), key)
                if text != "--" and direction is not None and best.get((dataset, key)) == model:
                    text = rf"\textbf{{{text}}}$^\star$"
                elif text != "--" and mine:
                    text = rf"\textbf{{{text}}}"
                cells.append(text)
            shown = rf"\textbf{{{name}}} ({tag})" if mine else f"{name} ({tag})"
            lines.append(f"{head} & {shown} & " + " & ".join(cells) + r" \\")
        # Show the trivial baselines underneath, so a column of failures reads as failure.
        shown = {k: references.get((dataset, k)) for k, _, _ in COLUMNS}
        if any(isinstance(v, (int, float)) and k != "recon_r2" for k, v in shown.items()):
            cells = [rf"\textit{{{fmt(shown[k], k)}}}" if isinstance(shown.get(k), (int, float)) and k != "recon_r2"
                     else "--" for k, _, _ in COLUMNS]
            lines.append(r"\cmidrule(l){2-" + str(len(COLUMNS) + 2) + "}")
            lines.append(r" & \textit{trivial reference} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", r"\end{document}"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}  ({len(results)} cells, {len(datasets)} datasets)")
    return 0


def default_caption():
    return (r"Benchmark comparison scored by the \texttt{opssm.benchmarks} pipeline. "
            r"\textbf{Bold}=opssm; \textbf{bold}$^\star$=best per (dataset,\,metric). "
            r"Model tag (F)ilter/(S)moother. $\uparrow$/$\downarrow$=higher/lower better; "
            r"cov$_{95}$ is nominal 95\% marginal coverage. \texttt{--}=not run or not "
            r"applicable (real data has no ground-truth latents or drift; synthetic data has "
            r"no behavior labels). Normalized errors divide by the RMS true latent/drift, so "
            r"they are comparable across systems. "
            r"\textbf{Single seed; selection rules differ across methods (see text).}")


if __name__ == "__main__":
    raise SystemExit(main())
