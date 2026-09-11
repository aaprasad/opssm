"""Stage 2: gather a finished sweep into one table, and check it is actually comparable.

Beyond collecting rows this verifies the fairness invariant the whole comparison rests on:
within a (point, dataset, seed) cell every model must report the same dataset_hash. A
mismatch means some model trained on different data, so the cell is reported, not ranked.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from opssm.benchmarks.metrics import aggregate


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--sweep", type=Path, help="Sweep root from make_sweep.py (sbatch-array layout)")
    source.add_argument("--results-root", type=Path,
                        help="results_root of a Hydra multirun; points are its subdirectories. "
                             "Completeness cannot be checked without a manifest, so nothing is "
                             "reported as missing in this mode.")
    ap.add_argument("--metric", default="dynamics_nrmse", help="Metric for the printed per-point summary")
    args = ap.parse_args(argv)

    if args.sweep:
        root = args.sweep / "results"
        spec = json.loads((args.sweep / "sweep.json").read_text())
        points = list(spec["points"])
        expected = {(p, d, s, m) for p in spec["points"] for d in spec["datasets"]
                    for s in spec["seeds"] for m in spec["models"]}
        point_config = lambda p: json.loads((args.sweep / "points" / f"{p}.json").read_text())
        out_base = args.sweep
    else:
        root = args.results_root
        points = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
        expected = set()
        def point_config(point):
            found = list((root / point).glob("dataset_config_*.json"))
            return json.loads(found[0].read_text()) if found else {}
        out_base = root

    rows, found = [], set()
    for point in points:
        for path in sorted((root / point).glob("*/seed_*/*/result.json")):
            row = json.loads(path.read_text())
            row["point"] = point
            for dataset, fields in point_config(point).items():
                for field, value in fields.items():
                    row[f"cfg_{dataset}_{field}"] = value
            rows.append(row)
            found.add((point, row["dataset"], row["seed"], row["model"]))

    missing = sorted(expected - found)
    failed = [r for r in rows if r.get("status") != "ok"]

    # Fairness check: one dataset per (point, dataset, seed), shared by every model.
    inconsistent = []
    cells = {}
    for row in rows:
        cells.setdefault((row["point"], row["dataset"], row["seed"]), {})[row["model"]] = row.get("dataset_hash")
    for cell, hashes in cells.items():
        distinct = {h for h in hashes.values() if h}
        if len(distinct) > 1:
            inconsistent.append((cell, hashes))

    out = out_base / "collected"
    out.mkdir(exist_ok=True)
    if rows:
        fields = sorted(set().union(*(r.keys() for r in rows)) - {"settings"})
        with (out / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(aggregate([r for r in rows if r.get("status") == "ok"]),
                                                 indent=2, allow_nan=False) + "\n")
    (out / "status.json").write_text(json.dumps(dict(
        expected=len(expected), collected=len(rows), ok=len(rows) - len(failed),
        failed=[{k: r[k] for k in ("point", "dataset", "seed", "model")} | {"error": r.get("error")} for r in failed],
        missing=[dict(zip(("point", "dataset", "seed", "model"), m)) for m in missing],
        dataset_hash_mismatches=[dict(cell=list(c), hashes=h) for c, h in inconsistent]), indent=2) + "\n")

    total = f"/{len(expected)}" if expected else ""
    print(f"collected {len(rows)}{total} cells  ok={len(rows) - len(failed)} "
          f"failed={len(failed)}" + (f" missing={len(missing)}" if expected else ""))
    if inconsistent:
        print(f"!! {len(inconsistent)} cell(s) have models trained on DIFFERENT data -- not comparable:")
        for cell, hashes in inconsistent[:5]:
            print(f"   {cell}: {hashes}")
    ok = [r for r in rows if r.get("status") == "ok" and isinstance(r.get(args.metric), (int, float))]
    if ok:
        print(f"\nmean {args.metric} by point x dataset x model (over seeds):")
        table = {}
        for r in ok:
            table.setdefault((r["point"], r["dataset"]), {}).setdefault(r["model"], []).append(r[args.metric])
        for (point, dataset), per_model in sorted(table.items()):
            best = min(per_model, key=lambda m: sum(per_model[m]) / len(per_model[m]))
            cells_txt = "  ".join(f"{m}={sum(v)/len(v):.4f}{'*' if m == best else ''}"
                                  for m, v in sorted(per_model.items()))
            print(f"  {point:28}{dataset:12}{cells_txt}")
        print("  (* = best in row)")
    print(f"\nwrote {out}/metrics.csv, summary.json, status.json")
    return 1 if (failed or missing or inconsistent) else 0


if __name__ == "__main__":
    raise SystemExit(main())
