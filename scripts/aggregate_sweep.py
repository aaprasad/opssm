"""Aggregate a JAX sweep: collect every result.json under a multirun dir into one ranked table + CSV.

    python scripts/aggregate_sweep.py <multirun_dir> [--sort drift_rel] [--min] [--csv out.csv]

Ranked table shows the axes that VARY (model hparams AND data-config knobs AND seed) plus the sort metric.
The CSV dumps ALL metrics per run (every final.* validation metric + whole-trace/g_final if present +
the effective descriptors obs_scale / noise_std_eff / dt), so a characterization sweep can plot any of them.

--sort  metric to rank by; resolved from result.json top-level, else result['final'], else result['data_eff'].
--min   rank ascending (lower = better) -- use for drift_rel / lat_rel / kl / l2 (defaults to descending/max,
        correct for recon_r2 / c_cos).  Torch-free (json/glob); still-running tasks just have no result.json.
"""
import os
import csv
import glob
import json
import math
import argparse

_EFF = ("obs_scale", "noise_std_eff", "dt", "system", "latent_dim")   # per-run effective descriptors
_TOP = ("whole_trace_recon_r2", "g_final")                            # top-level scalar metrics (Kato)


def _axes(r):
    """Everything that could be a swept axis: model hparams + data.* config + seed."""
    ax = dict(r.get("hparams") or {})
    for k, v in (r.get("data") or {}).items():
        ax["data." + k] = v
    ax["seed"] = r.get("seed")
    return ax


def _get(r, key):
    """Resolve a metric: top-level, then final.*, then data_eff.*"""
    if key in r:
        return r[key]
    for sub in ("final", "data_eff"):
        d = r.get(sub) or {}
        if key in d:
            return d[key]
    return None


def _num(x, worst):
    try:
        v = float(x)
        return v if math.isfinite(v) else worst
    except (TypeError, ValueError):
        return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="multirun dir (e.g. multirun/em_lorenz)")
    ap.add_argument("--sort", default="whole_trace_recon_r2", help="metric to rank by")
    ap.add_argument("--min", action="store_true", help="rank ascending (lower is better: drift_rel/lat_rel/kl/...)")
    ap.add_argument("--csv", default=None, help="also write the FULL per-run metric table to this CSV")
    ap.add_argument("--keys", default=None, help="comma-sep axis columns to show (default: those that vary)")
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(a.root, "**", "result.json"), recursive=True)):
        try:
            rows.append(json.load(open(f)))
        except Exception:
            pass
    if not rows:
        print(f"no result.json found under {a.root} (runs may still be training/preempted)")
        return

    axsets = {}                                                     # which axes actually vary?
    for r in rows:
        for k, v in _axes(r).items():
            axsets.setdefault(k, set()).add(repr(v))
    varying = a.keys.split(",") if a.keys else [k for k, s in sorted(axsets.items()) if len(s) > 1]

    metric_keys = set(_EFF)                                        # all metrics for the CSV (union across runs)
    for r in rows:
        metric_keys |= set((r.get("final") or {}).keys())
        metric_keys |= {k for k in _TOP if k in r}
    metric_keys = sorted(metric_keys)

    worst = math.inf if a.min else -math.inf
    rows.sort(key=lambda r: _num(_get(r, a.sort), worst), reverse=not a.min)

    cols = ["run"] + varying + [a.sort, "status"]
    w = 14
    print(f"{len(rows)} runs under {a.root}; ranked by {a.sort} ({'min' if a.min else 'max'})")
    print("  ".join(c[:w].rjust(w) for c in cols))
    for r in rows:
        ax = _axes(r)
        rd = os.path.basename(r.get("run_dir", "")) or "?"
        sv = _get(r, a.sort)
        cells = [rd] + [str(ax.get(k)) for k in varying] + [f"{_num(sv, float('nan')):.4f}", str(r.get("status", "?"))]
        print("  ".join(s[:w].rjust(w) for s in cells))

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["run_dir"] + varying + metric_keys + ["status"])
            for r in rows:
                ax = _axes(r)
                wr.writerow([r.get("run_dir", "")] + [ax.get(k) for k in varying]
                            + [_get(r, m) for m in metric_keys] + [r.get("status")])
        print(f"wrote {a.csv}  ({len(metric_keys)} metrics x {len(rows)} runs)")


if __name__ == "__main__":
    main()
