"""Create the random label-budget splits (stratified, validation inside the budget).

Usage:
    python make_splits.py --budgets 5 10 --seeds 0 1 2
Writes splits/random_<budget>pct_seed<k>.json INSIDE the code folder, so the files travel with the code
and every platform trains on identical labelled images. Existing files are never overwritten: the same
seed always means the same labelled images throughout the project.
"""
import argparse
from pathlib import Path

from polypssl.data import make_random_split, pranet_train, save_split, split_name
from polypssl.data.splits import check_split
from polypssl.paths import get_locations

CODE_SPLITS = Path(__file__).resolve().parent / "splits"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", nargs="+", default=["5"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    loc = get_locations()
    samples = pranet_train(loc.data)
    for b in args.budgets:
        for seed in args.seeds:
            path = CODE_SPLITS / f"{split_name('random', b, seed)}.json"
            if path.exists():
                print(f"[keep] {path.name} already exists")
                continue
            split = make_random_split(samples, b, seed)
            check_split(split, samples)
            save_split(split, CODE_SPLITS)
            src = {s["id"]: s["source"] for s in samples}
            lab = split["labelled_train"] + split["labelled_val"]
            print(f"[new] {path.name}: train {len(split['labelled_train'])}, val {len(split['labelled_val'])}, "
                  f"unlabelled {len(split['unlabelled'])}, Kvasir/ClinicDB labelled = "
                  f"{sum(src[i] == 'Kvasir' for i in lab)}/{sum(src[i] == 'ClinicDB' for i in lab)}")


if __name__ == "__main__":
    main()
