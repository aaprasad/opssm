"""Plot the Kato whole-trace (kato.fold=-1) benchmark: one point per worm, per method.

Three panels: whole-trace reconstruction R2, forecast RMSE relative to persistence, and
behavior-decoding balanced accuracy. Everything lands in one self-contained directory
(--out, default figures/kato_whole_worm/): combined.* plus each panel standalone as
<panel>.*, in png/pdf/svg, with summary.csv and provenance.json beside them.

Panels carry axis labels and one legend only -- no titles, no in-panel annotation -- so
what the reference lines mean is recorded in provenance.json and belongs in the caption:

* Reconstruction is bounded above by rank-`latent_dim` PCA. The reconstruction is y = C z
  with z in R^d, hence rank d, so by Eckart-Young no such model can beat the rank-d SVD --
  which uses no dynamics at all. That ceiling differs per worm (0.846-0.910 on these
  recordings), so each point carries its own grey bar rather than sharing one band, which
  would imply a high scorer sits at its ceiling when it may not.
* Persistence RMSE ("repeat the last observed frame") differs per worm too, since it depends
  on how fast that recording moves. Raw forecast RMSE is therefore not comparable across
  worms; dividing by each worm's own persistence puts the floor at exactly 1.0 everywhere.
* Decoding chance is 1/K, and K -- the number of labeled behavior states -- differs by
  condition (NoStim 7, Stim 4), so that panel carries one line per condition.

Scores from this protocol are IN-SAMPLE: fold=-1 fits every frame it scores. Decoding is the
exception, since the probe carves out its own held-out frames, but its features still come
from a model that saw everything (linear_decode_status=ok_in_sample_latents).

Usage:
  python scripts/plot_kato_whole_worm.py
  python scripts/plot_kato_whole_worm.py 'dump/kato_*whole_worm.csv' --out figures/kato_whole_worm
"""
import argparse
import glob
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

EXTENSIONS = ("png", "pdf", "svg")
CONDITIONS = ("NoStim", "Stim")
PALETTE = {"NoStim": "#1864ab", "Stim": "#c1272d"}
REFERENCE = "#495057"
PANELS = {"reconstruction_r2": r"observation recon $R^2$ (in-sample)",
          "forecast_vs_persistence": "forecast RMSE / persistence RMSE",
          "behavior_decoding": "behavior decoding balanced accuracy"}


def pca_ceiling(worm, mat_dir, latent_dim, cache={}):
    """Rank-`latent_dim` PCA R2 on the mean-centred trace: the ceiling for y = C z.

    Mean-centred because observation_recon_r2 centres per neuron; obs_scale is a global
    scalar and R2 is invariant to it, so raw and standardized traces agree here.
    """
    if worm not in cache:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from opssm.data.kato.load import load_worm
        condition, index = worm.rsplit("_worm", 1)
        path = Path(mat_dir) / f"{condition}.mat"
        if not path.is_file():
            cache[worm] = np.nan
        else:
            y = np.asarray(load_worm(str(path), int(index))["traces"], dtype=float)
            centred = y - y.mean(0)
            u, s, vt = np.linalg.svd(centred, full_matrices=False)
            k = min(latent_dim, len(s))
            residual = centred - (u[:, :k] * s[:k]) @ vt[:k]
            cache[worm] = float(1 - (residual ** 2).sum() / (centred ** 2).sum())
    return cache[worm]


def load(pattern, mat_dir, latent_dim, max_forecast_ratio):
    """Tidy frame of usable runs, plus the failures and exclusions, neither of which is plotted."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No result CSV matched {pattern!r}")
    raw = pd.concat([pd.read_csv(p).assign(source=p) for p in paths], ignore_index=True)
    raw = raw[raw["dataset"].notna()].copy()
    raw["worm"] = raw["dataset"].str.replace("kato_", "", regex=False).str.replace("_full", "", regex=False)
    failures = raw[raw["status"] != "ok"][["worm", "model", "error", "source"]].to_dict("records")
    frame = raw[raw["status"] == "ok"].copy()
    if frame.empty:
        raise SystemExit(f"No successful runs in {pattern!r}")
    frame["condition"] = np.where(frame["worm"].str.contains("NoStim"), "NoStim", "Stim")
    frame["reconstruction_r2"] = frame["observation_recon_r2"].astype(float)
    frame["ceiling"] = [pca_ceiling(w, mat_dir, latent_dim) for w in frame["worm"]]
    frame["forecast_vs_persistence"] = (frame["forecast_observation_rmse"].astype(float)
                                        / frame["forecast_persistence_rmse"].astype(float))
    frame["behavior_decoding"] = frame["linear_decode_balanced_accuracy"].astype(float)
    frame["n_classes"] = frame["linear_decode_train_classes"].map(lambda v: len(json.loads(v)))
    frame["chance"] = 1.0 / frame["n_classes"]
    # A rollout that diverges is not a forecast score, it is a broken integration: rslds leaves
    # the ratio at 1e2-1e5 on some worms while reconstructing normally. Drop the whole run rather
    # than one metric, so every panel and summary.csv describe the same set of runs, and record
    # it -- an exclusion that only lives in the plotting code is an undocumented result.
    diverged = frame["forecast_vs_persistence"] > max_forecast_ratio
    excluded = [dict(worm=r.worm, model=r.model, forecast_vs_persistence=float(r.forecast_vs_persistence),
                     reason=f"forecast rollout diverged (ratio > {max_forecast_ratio})")
                for r in frame[diverged].itertuples()]
    return frame[~diverged].copy(), failures, excluded


def strip(ax, frame, order, metric):
    """One point per worm, coloured by condition; the bar is the per-method mean."""
    before = len(ax.collections)
    sns.stripplot(data=frame, x="model", y=metric, hue="condition", order=order,
                  hue_order=CONDITIONS, palette=PALETTE, jitter=.16, size=9, alpha=.95,
                  edgecolor="white", linewidth=1.1, dodge=False, ax=ax, legend=False)
    points = list(ax.collections[before:])        # only the stripplot's, not the mean hlines below
    means = frame.groupby("model")[metric].mean()
    for index, model in enumerate(order):
        if model in means.index and np.isfinite(means[model]):
            ax.hlines(means[model], index - .3, index + .3, color=REFERENCE, lw=2.2, zorder=4)
    ax.set_xlabel("")
    ax.set_ylabel(PANELS[metric], fontsize=10)
    ax.tick_params(axis="x", labelrotation=18)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.grid(axis="y", alpha=.25, lw=.6)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    return points


def draw_reconstruction_r2(ax, frame, order):
    # The ceiling is per worm, so draw each point's own: a shared band would suggest a high
    # scorer is at its ceiling when its worm's ceiling may be far above. seaborn jitters with
    # its own RNG, so read the plotted positions back rather than re-jittering, or the bars
    # land beside their points instead of on them.
    points = strip(ax, frame, order, "reconstruction_r2")
    plotted = np.concatenate([c.get_offsets() for c in points if len(c.get_offsets())])
    lookup = frame.set_index("reconstruction_r2")["ceiling"].to_dict()
    for x, y in plotted:
        ceiling = min(lookup, key=lambda v: abs(v - y))
        ceiling = lookup[ceiling]
        if np.isfinite(ceiling):
            ax.vlines(x, y, ceiling, color="#adb5bd", lw=1.1, zorder=1)
            ax.scatter(x, ceiling, s=44, marker="_", color=REFERENCE, zorder=2)


def draw_forecast_vs_persistence(ax, frame, order):
    strip(ax, frame, order, "forecast_vs_persistence")
    ax.axhline(1.0, ls="--", lw=1.6, color=REFERENCE, zorder=1)



def draw_behavior_decoding(ax, frame, order):
    strip(ax, frame, order, "behavior_decoding")
    for condition in CONDITIONS:
        for chance in sorted(set(frame.loc[frame["condition"] == condition, "chance"].dropna())):
            ax.axhline(chance, ls="--", lw=1.5, color=PALETTE[condition], zorder=1)
    ax.set_ylim(0, 1)


DRAW = {"reconstruction_r2": draw_reconstruction_r2,
        "forecast_vs_persistence": draw_forecast_vs_persistence,
        "behavior_decoding": draw_behavior_decoding}


def handles(frame, metric):
    """Every legend entry the panel needs, so each standalone asset reads on its own."""
    items = [plt.Line2D([], [], marker="o", ls="", color=PALETTE[c], markersize=8,
                        markeredgecolor="white", label=c) for c in CONDITIONS]
    items.append(plt.Line2D([], [], color=REFERENCE, lw=2.2, label="method mean"))
    if metric == "reconstruction_r2":
        items.append(plt.Line2D([], [], color="#adb5bd", lw=1.1, marker="_",
                                markeredgecolor=REFERENCE, label="rank-10 PCA ceiling (per worm)"))
    elif metric == "forecast_vs_persistence":
        items.append(plt.Line2D([], [], color=REFERENCE, ls="--", lw=1.6, label="persistence"))
    else:
        items += [plt.Line2D([], [], color=PALETTE[c], ls="--", lw=1.5,
                             label=f"chance {c} (1/{k})")
                  for c, k in zip(CONDITIONS, (7, 4))]
    return items


def save(figure, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for extension in EXTENSIONS:
        path = stem.with_suffix("." + extension)
        figure.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
        written.append(str(path))
    plt.close(figure)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="?", default="dump/kato_baselines_whole_worm.csv",
                        help="Result CSV, or a glob over several")
    parser.add_argument("--out", type=Path, default=Path("figures/kato_whole_worm"),
                        help="Output DIRECTORY; holds combined.*, <panel>.*, summary.csv, provenance.json")
    parser.add_argument("--mat-dir", default="/home/aaprasad/data/kato",
                        help="Directory holding WT_NoStim.mat / WT_Stim.mat, for the PCA ceiling")
    parser.add_argument("--latent-dim", type=int, default=10,
                        help="Latent dimension the runs used; sets the PCA ceiling rank")
    parser.add_argument("--max-forecast-ratio", type=float, default=10.0,
                        help="Drop runs whose forecast/persistence exceeds this: a diverged "
                             "rollout is a broken integration, not a forecast score. Recorded "
                             "in provenance.json as an exclusion.")
    args = parser.parse_args(argv)

    sns.set_theme(style="ticks", context="notebook")
    frame, failures, excluded = load(args.sources, args.mat_dir, args.latent_dim,
                                     args.max_forecast_ratio)
    order = sorted(frame["model"].unique(), key=lambda m: (m != "opssm", m))
    args.out.mkdir(parents=True, exist_ok=True)
    written = []

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, metric in zip(axes, PANELS):
        DRAW[metric](ax, frame, order)
    combined = handles(frame, "reconstruction_r2")[:3]
    combined += [h for m in PANELS for h in handles(frame, m)[3:]]
    figure.legend(handles=combined, loc="lower center", ncol=4, frameon=False, fontsize=9,
                  bbox_to_anchor=(.5, -.13))
    figure.tight_layout()
    written += save(figure, args.out / "combined")

    for metric in PANELS:                        # standalone, legend included so it stands alone
        single, ax = plt.subplots(figsize=(5.2, 4.4))
        DRAW[metric](ax, frame, order)
        single.legend(handles=handles(frame, metric), loc="lower center", ncol=2, frameon=False,
                      fontsize=8.5, bbox_to_anchor=(.5, -.22))
        single.tight_layout()
        written += save(single, args.out / metric)

    columns = ["worm", "condition", "model", "reconstruction_r2", "ceiling",
               "forecast_observation_rmse", "forecast_persistence_rmse",
               "forecast_vs_persistence", "behavior_decoding", "n_classes", "chance",
               "linear_decode_status"]
    summary_path = args.out / "summary.csv"
    frame.sort_values(["model", "worm"])[columns].to_csv(summary_path, index=False)
    written.append(str(summary_path))

    provenance = dict(
        sources=[dict(path=str(Path(p).resolve()),
                      sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
                 for p in sorted(glob.glob(args.sources))],
        outputs=written,
        protocol="kato.fold=-1 (whole trace; train == val == test)",
        in_sample=True,
        successful_runs=int(len(frame)),
        failed_runs=failures,
        worms_per_model={m: sorted(frame.loc[frame["model"] == m, "worm"]) for m in order},
        excluded_runs=sorted(excluded, key=lambda d: -d["forecast_vs_persistence"]),
        exclusion_rule=f"forecast_vs_persistence > {args.max_forecast_ratio}",
        points="one point per worm; bar is the unweighted per-method mean, no error bars "
               "(coverage is uneven, so methods are not measured on the same worms)",
        reference_lines=dict(
            reconstruction_r2=f"rank-{args.latent_dim} PCA R2 on the mean-centred trace, PER WORM; "
                              "y = C z is rank d so Eckart-Young bounds recon R2 by it",
            forecast_vs_persistence="each worm's own persistence RMSE (repeat the last observed "
                                    "frame); plotted as a ratio so the floor is 1.0 for every worm",
            behavior_decoding="1/K, K = labeled behavior states, per condition (NoStim 7, Stim 4)"),
        caveats=["fold=-1 fits every frame it scores, so reconstruction and forecast are in-sample",
                 "the decoding probe holds out its own frames but reads in-sample latents "
                 "(linear_decode_status=ok_in_sample_latents)",
                 "per-method means are over different worms wherever coverage is incomplete, "
                 "so they are not paired comparisons"],
        latent_dim=args.latent_dim)
    provenance_path = args.out / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    written.append(str(provenance_path))

    for path in written:
        print(path)
    for item in excluded:
        print(f"\nexcluded {item['worm']} {item['model']}: ratio={item['forecast_vs_persistence']:.4g} "
              f"({item['reason']})")
    if failures:
        print(f"\n{len(failures)} failed run(s) excluded:")
        for failure in failures:
            print(f"  {failure['worm']} {failure['model']}: {str(failure['error'])[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
