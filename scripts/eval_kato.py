"""Post-hoc eval of a trained opssm model on Kato 2015 C. elegans data (thin driver over opssm.eval).

Loads <train_dir>/model.pt, filters the FULL worm trace to latents, then computes reconstruction R2 +
behavior decoding + the latent/manifold/dynamics figures -- all via the shared, model-agnostic eval-core
(opssm.eval.core), so opssm and every baseline in the baselining harness are scored by identical code.

Usage: python scripts/eval_kato.py --model dump/kato_stim0/model.pt
"""
import argparse
import os

import torch

from opssm.eval.data import context_from_kato_checkpoint
from opssm.eval.baselines.opssm_adapter import OpssmAdapter
from opssm.eval.core import kato_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to <train_dir>/model.pt")
    ap.add_argument("--out", default=None, help="output dir (default: alongside model.pt)")
    a = ap.parse_args()
    outdir = a.out or os.path.dirname(a.model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = context_from_kato_checkpoint(a.model, device=dev)
    res = OpssmAdapter().fit_predict(ctx, {"checkpoint": a.model}, device=dev)
    kato_report(res, ctx, outdir)


if __name__ == "__main__":
    main()
