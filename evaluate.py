"""Evaluate a finished run on all test sets (Block 1 PraNet sets + any external sets present).

Usage:
    python evaluate.py --run outputs/<ID>__<name>/seed0
Writes per_image.csv and summary.csv into the run folder (or --out). run.py calls this automatically.
"""
import argparse
from pathlib import Path

import torch

from polypssl.data import test_sets
from polypssl.engine.common import build_trained_model
from polypssl.evaluation import evaluate_model, save_results
from polypssl.paths import get_locations


def evaluate_run(run_dir, include_external: bool = True, out_dir=None):
    run_dir = Path(run_dir)
    loc = get_locations()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, state = build_trained_model(run_dir, device)
    print(f"[eval] {cfg['name']} seed{cfg['seed']} (iteration {state.get('iteration')})")
    per_image = evaluate_model(model, test_sets(loc.data, include_external), cfg["data"]["img_size"],
                               device, num_workers=cfg["data"]["num_workers"], amp=cfg["train"]["amp"])
    out_dir = Path(out_dir) if out_dir else run_dir
    summary = save_results(per_image, out_dir)
    print(summary[summary["size_bin"] == "all"][["dataset", "n", "dice", "iou", "recall", "hd95_pct", "mdice_thr"]]
          .to_string(index=False, float_format="%.4f"))
    print(f"[saved] {out_dir / 'summary.csv'}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None, help="results folder (default: the run folder)")
    ap.add_argument("--block1-only", action="store_true", help="only the 5 PraNet test sets")
    args = ap.parse_args()
    evaluate_run(args.run, include_external=not args.block1_only, out_dir=args.out)
