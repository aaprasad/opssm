#!/usr/bin/env python
"""Scaling micro-benchmark for the pluggable context encoders (opssm/models/encoders.py).

Measures each encoder IN ISOLATION (no training) as we vary one axis at a time:
  - sequence length T   (O(T) sequential GRU vs O(T^2) attention vs parallel TCN/SSM)
  - depth  n_layers
  - input dim  in_dim   (= data_size + 1; the high-D-observation axis)
  - width  hidden
Records forward latency, forward+backward latency, param count, peak GPU mem. Emits a CSV and a
scaling-curve figure. This is the primary deliverable that justifies (or refutes) an encoder swap:
a swap is worth it where its advantage GROWS with problem size, which a single T=100 point can't show.

    python scripts/profile_encoder.py                       # all registered encoders, default sweeps
    python scripts/profile_encoder.py --encoders gru,tcn    # subset
    python scripts/profile_encoder.py --B 64 --reps 40 --out dump/encoder_scaling
"""
import argparse
import csv
import os
import time

import torch

from opssm.models.encoders import ENCODERS, make_encoder

# generic "depth" maps to each encoder's layer-count kwarg
DEPTH_KWARG = {"gru": "layers", "tcn": "n_layers", "transformer": "n_layers",
               "rno": "n_layers", "mamba": "n_layers", "deer_gru": "layers"}

# baseline point (held fixed while sweeping one axis)
BASE = dict(T=100, in_dim=11, hidden=64)
SWEEPS = dict(T=[50, 100, 200, 400, 800],
              depth=[1, 2, 4, 8],
              in_dim=[2, 11, 51, 101],
              hidden=[32, 64, 128, 256])


def build(name, in_dim, hidden, depth=None):
    kw = {}
    if depth is not None:
        kw[DEPTH_KWARG[name]] = depth
    return make_encoder(name, in_dim, ctx_dim=64, hidden=hidden, **kw)


def _median_ms(fn, device, reps, warmup):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    else:
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def bench_one(name, T, B, in_dim, hidden, depth, device, reps, warmup):
    """Returns dict(fwd_ms, fwdbwd_ms, params, peak_mb) or Nones on OOM/error."""
    try:
        enc = build(name, in_dim, hidden, depth).to(device)
        params = sum(p.numel() for p in enc.parameters())
        x = torch.randn(T, B, in_dim, device=device)

        def fwd():
            with torch.no_grad():
                enc(x)

        def fwdbwd():
            enc.zero_grad(set_to_none=True)
            enc(x).sum().backward()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        fwd_ms = _median_ms(fwd, device, reps, warmup)
        fwdbwd_ms = _median_ms(fwdbwd, device, reps, warmup)
        peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else float("nan")
        del enc, x
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return dict(fwd_ms=fwd_ms, fwdbwd_ms=fwdbwd_ms, params=params, peak_mb=peak_mb)
    except RuntimeError as ex:                                   # OOM etc.
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return dict(fwd_ms=None, fwdbwd_ms=None, params=None, peak_mb=None, error=str(ex)[:80])


def run(encoders, B, device, reps, warmup):
    rows = []
    for axis, values in SWEEPS.items():
        for v in values:
            T = v if axis == "T" else BASE["T"]
            in_dim = v if axis == "in_dim" else BASE["in_dim"]
            hidden = v if axis == "hidden" else BASE["hidden"]
            depth = v if axis == "depth" else None
            for name in encoders:
                r = bench_one(name, T, B, in_dim, hidden, depth, device, reps, warmup)
                rows.append(dict(axis=axis, value=v, encoder=name, T=T, B=B, in_dim=in_dim,
                                 hidden=hidden, depth=depth if depth is not None else "default", **r))
                tag = f"{r['fwd_ms']:.3f}ms fwd / {r['fwdbwd_ms']:.3f}ms fb" if r["fwd_ms"] else "FAIL"
                print(f"  [{axis:6s}={str(v):>4s}] {name:12s} {tag}  params={r['params']}")
    return rows


def plot(rows, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib unavailable; skipping figure)")
        return
    axes_order = ["T", "depth", "in_dim", "hidden"]
    encoders = sorted({r["encoder"] for r in rows})
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.2))
    for ax, axis in zip(axs, axes_order):
        for enc in encoders:
            pts = [(r["value"], r["fwdbwd_ms"]) for r in rows
                   if r["axis"] == axis and r["encoder"] == enc and r["fwdbwd_ms"] is not None]
            if pts:
                xs, ys = zip(*sorted(pts))
                ax.plot(xs, ys, "o-", label=enc)
        ax.set_title(f"fwd+bwd latency vs {axis}"); ax.set_xlabel(axis); ax.set_ylabel("ms")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Encoder scaling (isolated; other axes fixed at baseline)", y=1.02)
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight", dpi=110)
    print(f"figure -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default=",".join(sorted(ENCODERS)))
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default="dump/encoder_scaling")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    print(f"device={device}  encoders={encoders}  B={args.B}  reps={args.reps}")
    torch.manual_seed(0)

    rows = run(encoders, args.B, device, args.reps, args.warmup)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    csv_path = args.out + ".csv"
    keys = ["axis", "value", "encoder", "T", "B", "in_dim", "hidden", "depth",
            "params", "fwd_ms", "fwdbwd_ms", "peak_mb"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"csv -> {csv_path}")
    plot(rows, args.out + ".png")


if __name__ == "__main__":
    main()
