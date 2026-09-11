"""Stage 0: generate every dataset in a sweep once, on CPU, before the GPU array starts.

Array tasks then all read the same data.npz. save_dataset() fingerprint-checks rather than
rewriting, so this also removes any chance of two tasks racing to write the same file.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", type=Path, required=True, help="Sweep root from make_sweep.py")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args(argv)
    spec = json.loads((args.sweep / "sweep.json").read_text())
    repo = Path(__file__).resolve().parents[2]
    failures = []
    for point in spec["points"]:
        out = args.sweep / "results" / point
        command = [args.python, "-m", "opssm.benchmarks.runner", "--generate-only",
                   "--datasets", *spec["datasets"], "--seeds", *[str(s) for s in spec["seeds"]],
                   "--out", str(out)]
        config = args.sweep / "points" / f"{point}.json"
        if json.loads(config.read_text()):
            command.extend(["--dataset-config", str(config)])
        print(f"[{point}] {' '.join(command)}", flush=True)
        result = subprocess.run(command, cwd=repo, env={**__import__("os").environ,
            "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(repo)})
        if result.returncode:
            failures.append(point)
    if failures:
        print(f"FAILED to generate: {failures}", file=sys.stderr)
        return 1
    print(f"generated data for {len(spec['points'])} point(s) under {args.sweep / 'results'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
