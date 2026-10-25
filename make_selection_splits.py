"""Create the Stage D label splits from the real training images (images only, never masks).

Usage:  python make_selection_splits.py --budgets 5 --seeds 0 1 2
Writes splits/acls_<b>pct_seed<k>.json and splits/compmatched_<b>pct_seed<k>.json inside the code folder
(the files travel with the code, so every platform trains on identical labelled images). Existing files
are never overwritten.
"""
import argparse
import json
from pathlib import Path

from polypssl.appearance import Standardiser, descriptors
from polypssl.data import pranet_train, save_split, split_name
from polypssl.data.transforms import load_image
from polypssl.paths import get_locations
from polypssl.selection import acls_split, compmatched_split

CODE_SPLITS = Path(__file__).resolve().parent / "splits"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", nargs="+", default=["5"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    samples = pranet_train(get_locations().data)
    print(f"[features] computing appearance descriptors for {len(samples)} training images ...")
    feats = descriptors([s["image"] for s in samples], load_image)
    z = Standardiser(feats)(feats)
    for b in args.budgets:
        for seed in args.seeds:
            acls_path = CODE_SPLITS / f"{split_name('acls', b, seed)}.json"
            if acls_path.exists():
                acls = json.loads(acls_path.read_text(encoding="utf-8"))
                print(f"[keep] {acls_path.name}")
            else:
                acls = acls_split(samples, z, b, seed)
                save_split(acls, CODE_SPLITS)
                print(f"[new] {acls_path.name}")
            cm_path = CODE_SPLITS / f"{split_name('compmatched', b, seed)}.json"
            if cm_path.exists():
                print(f"[keep] {cm_path.name}")
            else:
                save_split(compmatched_split(samples, acls, seed), CODE_SPLITS)
                print(f"[new] {cm_path.name}")
            src = {s['id']: s['source'] for s in samples}
            lab = acls["labelled_train"] + acls["labelled_val"]
            print(f"       ACLS seed {seed}: {len(lab)} labelled = "
                  f"{sum(src[i] == 'Kvasir' for i in lab)} Kvasir + {sum(src[i] == 'ClinicDB' for i in lab)} ClinicDB")


if __name__ == "__main__":
    main()
