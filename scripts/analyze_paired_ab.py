"""Paired analysis of a fixed-data A/B sweep (run dirs named <prefix>_<arm>_m<seed>).

    python scripts/analyze_paired_ab.py fx_highd fx_vdp        # dirs under ROOT (edit ROOT or pass abs paths)

METHODOLOGY THIS ENCODES (learned the hard way on the learn-obsnoise-cov experiment):

1. FREEZE THE DATA. `seed` in train_jax also reseeds the DATASET (build_synthetic_data falls back to
   cfg.seed), so multi-seed runs otherwise confound data variance with run variance -- two baseline
   "seeds" differed by 0.08 in drift_rel (0.23 vs 0.13), far more than any effect being measured.
   Freeze one dataset to an npz and pass +data.npz_path=..., so `seed` moves only the model.

2. PAIR ON SEED, and judge the PAIRED delta -- not the across-seed spread. Those are different scales:
   kl_aln varies 0.019 -> 0.049 ACROSS model seeds on identical data, yet the WITHIN-seed paired delta is
   ~0.001, because the seed effect is common to both arms and cancels. Reporting arm means would hide a
   real effect behind a nuisance 30x larger.

3. REPORT BEST-OVER-TRAINING ALONGSIDE FINAL. drift_rel DEGRADES late in training on these benchmarks
   while kl_aln keeps improving (seed 0: best 0.2104 @12k -> 0.2171 @14k), so a final-step-only table
   mixes the treatment effect with where each run sits on its own degradation curve.


Reports, per arm, BOTH the final-step and the best-over-training value of each metric, because drift_rel
degrades late in training on this benchmark (seed 0 bottomed at 0.214 by step 12000 then rose to 0.230),
so a final-step-only table mixes the treatment effect with where each run sits on its own degradation curve.
Deltas are PAIRED within model seed -- arms share the frozen dataset AND the seed, so the pairing is exact.
"""
import csv, glob, os, statistics as st, sys

LOWER_IS_BETTER = {"drift_rel", "lat_rel", "kl_aln", "kl"}
SHOW = ["drift_rel", "lat_rel", "kl_aln", "g_rel", "recon_r2", "g_aniso", "obs_noise", "obs_noise_perp"]
ROOT = os.environ.get("AB_ROOT", "dump/learn_obsnoise_cov")   # override with AB_ROOT=<dir>


def load(run):
    f = os.path.join(run, "metrics.csv")
    if not os.path.exists(f):
        return None
    rows = [r for r in csv.DictReader(open(f)) if r["step"] != "0"]
    return rows or None


def series(rows, k):
    return [(int(r["step"]), float(r[k])) for r in rows if r.get(k) not in (None, "")]


def summarize(prefix):
    runs = {}
    for d in sorted(glob.glob(os.path.join(ROOT, f"{prefix}_*"))):
        if not os.path.isdir(d):
            continue
        rows = load(d)
        if rows:
            name = os.path.basename(d)[len(prefix) + 1:]              # "<arm>_m<seed>"
            arm, seed = name.rsplit("_m", 1)
            runs[(arm, seed)] = rows
    if not runs:
        return
    arms = sorted({a for a, _ in runs})
    seeds = sorted({s for _, s in runs})
    print(f"\n{'='*100}\n{prefix}   arms={arms}   model seeds={seeds}   (frozen dataset: identical across all)\n{'='*100}")
    for k in SHOW:
        table = {}
        for (arm, seed), rows in runs.items():
            s = series(rows, k)
            if not s:
                continue
            fin = s[-1][1]
            best = (min if k in LOWER_IS_BETTER else max)(s, key=lambda t: t[1])
            table[(arm, seed)] = (fin, best[1], best[0], s[-1][0])
        if not table:
            continue
        print(f"\n-- {k} --")
        print(f"  {'arm':6s} " + "".join(f"{'seed ' + s:>22s}" for s in seeds) + "     mean(final)")
        for arm in arms:
            cells, fins = [], []
            for s in seeds:
                v = table.get((arm, s))
                if v is None:
                    cells.append(f"{'-':>22s}")
                else:
                    cells.append(f"{v[0]:>10.4f} (best {v[1]:.4f})")
                    fins.append(v[0])
            m = f"{st.mean(fins):>10.4f}" if fins else f"{'-':>10s}"
            print(f"  {arm:6s} " + "".join(cells) + m)
        base = [a for a in arms if a == "base"]
        if base:
            for arm in arms:
                if arm == "base":
                    continue
                dl = [(table[(arm, s)][0] - table[("base", s)][0]) for s in seeds
                      if (arm, s) in table and ("base", s) in table]
                db = [(table[(arm, s)][1] - table[("base", s)][1]) for s in seeds
                      if (arm, s) in table and ("base", s) in table]
                if dl:
                    sign = "lower=better" if k in LOWER_IS_BETTER else "higher=better"
                    win = sum(1 for x in dl if (x < 0) == (k in LOWER_IS_BETTER))
                    print(f"     paired {arm}-base: final {st.mean(dl):+.4f} "
                          f"({'/'.join(f'{x:+.3f}' for x in dl)})   best {st.mean(db):+.4f}   "
                          f"[{win}/{len(dl)} seeds favor {arm}; {sign}]")


for p in (sys.argv[1:] or ["fx_highd", "fx_lorenz"]):
    summarize(p)
