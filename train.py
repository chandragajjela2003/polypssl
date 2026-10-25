"""Train one run, then evaluate it on every test set.

Usage (from the polypssl/ folder):
    python train.py --config configs/stageA/fulnet_c_mscan_full.yaml --seed 0
    python train.py --config ... --seed 0 --iterations 200      # short run, e.g. speed or smoke test
Re-running the same command after a disconnect resumes from the last checkpoint.
"""
import argparse

import torch

from polypssl.config import load_config
from polypssl.data import test_sets
from polypssl.engine import train
from polypssl.paths import get_locations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=None, help="override (for smoke/speed tests)")
    ap.add_argument("--name-suffix", default="", help="appended to the run name, e.g. _smoke")
    ap.add_argument("--no-eval", action="store_true", help="skip evaluation after training")
    args = ap.parse_args()

    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.iterations is not None:
        overrides["train"] = {"iterations": args.iterations}
    cfg = load_config(args.config, overrides)
    cfg["name"] += args.name_suffix

    loc = get_locations()
    print(f"[env] {loc.env} | data: {loc.data} | outputs: {loc.out} | "
          f"device: {'cuda: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    test_sets(loc.data, include_external=False)
    print("[preflight] core PraNet test folders, pairings, and counts validated")
    run_dir = train(cfg, loc)

    if not args.no_eval:
        from evaluate import evaluate_run
        evaluate_run(run_dir)


if __name__ == "__main__":
    main()
