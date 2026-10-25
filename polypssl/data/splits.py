"""Label-budget splits of the 1,450 training images.

Protocol (NOVELTY.md v6/v7):
- A budget is the TOTAL number of masks a doctor would draw (2.5% = 36, 5% = 72, 10% = 145, 20% = 290).
- The validation subset is INSIDE the budget (about 1/6 of it, e.g. 72 = 60 train + 12 val). It is used
  only to detect broken runs; no tuning and no checkpoint selection are done on it.
- Every other training image is unlabelled; its mask is never read during training.
- Random selection is stratified by source (Kvasir : ClinicDB = 900 : 550).

Each split is saved as JSON listing image ids, so every run can state exactly which images were labelled.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BUDGETS = {"2.5": 36, "5": 72, "10": 145, "20": 290}


def val_count(total: int) -> int:
    return max(1, round(total / 6))


def _stratified_pick(ids_by_source: dict[str, list[str]], n: int, rng: np.random.Generator) -> list[str]:
    """Pick n ids, keeping each source's share of the pool (largest-remainder rounding)."""
    sizes = {s: len(v) for s, v in ids_by_source.items()}
    pool = sum(sizes.values())
    exact = {s: n * sizes[s] / pool for s in sizes}
    quota = {s: int(np.floor(exact[s])) for s in sizes}
    for s in sorted(sizes, key=lambda s: exact[s] - quota[s], reverse=True)[: n - sum(quota.values())]:
        quota[s] += 1
    picked = []
    for s in sorted(ids_by_source):
        ids = sorted(ids_by_source[s])
        picked += [ids[i] for i in rng.choice(len(ids), size=quota[s], replace=False)]
    return picked


def make_random_split(samples: list[dict], budget_pct: str, seed: int) -> dict:
    total = BUDGETS[budget_pct]
    rng = np.random.default_rng(seed)
    by_source: dict[str, list[str]] = {}
    for s in samples:
        by_source.setdefault(s["source"], []).append(s["id"])
    labelled = _stratified_pick(by_source, total, rng)

    # validation subset: stratified again, from inside the labelled set
    lab_by_source: dict[str, list[str]] = {}
    source_of = {s["id"]: s["source"] for s in samples}
    for i in labelled:
        lab_by_source.setdefault(source_of[i], []).append(i)
    val = _stratified_pick(lab_by_source, val_count(total), np.random.default_rng(seed + 10_000))

    labelled_set, val_set = set(labelled), set(val)
    return {
        "method": "random",
        "budget_pct": budget_pct,
        "seed": seed,
        "labelled_train": sorted(labelled_set - val_set),
        "labelled_val": sorted(val_set),
        "unlabelled": sorted(s["id"] for s in samples if s["id"] not in labelled_set),
    }


def check_split(split: dict, samples: list[dict]) -> None:
    """Fail loudly on any protocol violation."""
    all_ids = {s["id"] for s in samples}
    tr, va, un = set(split["labelled_train"]), set(split["labelled_val"]), set(split["unlabelled"])
    assert not (tr & va) and not (tr & un) and not (va & un), "split subsets overlap"
    assert tr | va | un == all_ids, "split does not cover the training pool exactly"
    if split["budget_pct"] in BUDGETS:
        assert len(tr) + len(va) == BUDGETS[split["budget_pct"]], "labelled count != budget"


def split_name(method: str, budget_pct: str, seed: int) -> str:
    return f"{method}_{budget_pct}pct_seed{seed}"


def save_split(split: dict, folder: str | Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{split_name(split['method'], split['budget_pct'], split['seed'])}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=1)
    return path


def load_split(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def full_split(samples: list[dict]) -> dict:
    """100% labels (upper bound / reproduction runs): no validation subset, nothing unlabelled."""
    return {"method": "full", "budget_pct": "100", "seed": 0,
            "labelled_train": sorted(s["id"] for s in samples), "labelled_val": [], "unlabelled": []}
