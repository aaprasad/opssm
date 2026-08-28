"""Aggregate a JAX grid-search sweep: collect every result.json under a multirun dir into one ranked table.

    python scripts/aggregate_sweep.py <multirun_dir> [--sort whole_trace_recon_r2] [--csv out.csv]

Shows only the hyperparameters that VARY across runs (the swept axes) plus the sort metric, ranked best-first.
Torch-free (json/glob) -- runs in any venv. A still-running/preempted task simply has no result.json yet.
"""
import os
import csv
import glob
import json
import argparse


def _metric(r, key):
    return r.get(key, (r.get("final") or {}).get(key, float("-inf")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="multirun dir (e.g. multirun/2026-08-28/12-00-00)")
    ap.add_argument("--sort", default="whole_trace_recon_r2", help="metric to rank by (default: whole-trace recon)")
    ap.add_argument("--csv", default=None, help="also write the full table to this CSV")
    ap.add_argument("--keys", default=None, help="comma-sep hparam columns to show (default: those that vary)")
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

    hpsets = {}                                                      # which hyperparameters actually vary?
    for r in rows:
        for k, v in (r.get("hparams") or {}).items():
            hpsets.setdefault(k, set()).add(repr(v))
    varying = a.keys.split(",") if a.keys else [k for k, s in sorted(hpsets.items()) if len(s) > 1]
    rows.sort(key=lambda r: _metric(r, a.sort), reverse=True)

    cols = ["run"] + varying + [a.sort, "g_final", "status"]
    width = 14
    print(f"{len(rows)} runs under {a.root}; ranked by {a.sort}")
    print("  ".join(c[:width].rjust(width) for c in cols))
    for r in rows:
        hp = r.get("hparams") or {}
        rd = os.path.basename(r.get("run_dir", "")) or "?"
        cells = ([rd] + [str(hp.get(k)) for k in varying]
                 + [f"{_metric(r, a.sort):.4f}", f"{r.get('g_final', float('nan')):.4f}", str(r.get("status", "?"))])
        print("  ".join(s[:width].rjust(width) for s in cells))

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["run_dir"] + varying + [a.sort, "g_final", "status"])
            for r in rows:
                hp = r.get("hparams") or {}
                w.writerow([r.get("run_dir", "")] + [hp.get(k) for k in varying]
                           + [_metric(r, a.sort), r.get("g_final"), r.get("status")])
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
