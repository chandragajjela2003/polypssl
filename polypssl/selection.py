"""Label selection for Stage D (which images a doctor labels). All selections use images only.

acls        (run D1) ACLS = appearance-clustered label selection: k-means with k = budget on the
            standardised appearance descriptor, then label the MEDOID of each cluster (the real image
            nearest the centroid). Pilot P0 (NOVELTY.md 3.1) showed medoids, not k-center coverage, bring
            the labelled set closer to unseen hospitals. Closest prior art: USL (ECCV 2022), TypiClust.
compmatched (run D2) control: random selection with exactly the same Kvasir : ClinicDB counts as ACLS of
            the same seed, so any ACLS gain cannot be explained by dataset mix (review issue A5).
The validation subset (1/6 of the budget) is drawn INSIDE the selected set, stratified by source, exactly
as for random splits, so every method has the same 60 + 12 structure.
"""
from __future__ import annotations

import numpy as np

from .data.splits import BUDGETS, _stratified_pick, check_split, val_count


def _assemble(method: str, budget_pct: str, seed: int, labelled: list[str], samples: list[dict]) -> dict:
    source_of = {s["id"]: s["source"] for s in samples}
    by_source: dict[str, list[str]] = {}
    for i in labelled:
        by_source.setdefault(source_of[i], []).append(i)
    val = _stratified_pick(by_source, val_count(len(labelled)), np.random.default_rng(seed + 10_000))
    lab, va = set(labelled), set(val)
    split = {"method": method, "budget_pct": budget_pct, "seed": seed,
             "labelled_train": sorted(lab - va), "labelled_val": sorted(va),
             "unlabelled": sorted(s["id"] for s in samples if s["id"] not in lab)}
    check_split(split, samples)
    return split


def acls_split(samples: list[dict], features: np.ndarray, budget_pct: str, seed: int) -> dict:
    """features: standardised appearance descriptors, row i = samples[i]."""
    from sklearn.cluster import KMeans
    k = BUDGETS[budget_pct]
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(features)
    chosen = set()
    for c in range(k):
        members = np.where(km.labels_ == c)[0]
        d = np.linalg.norm(features[members] - km.cluster_centers_[c], axis=1)
        chosen.add(samples[int(members[np.argmin(d)])]["id"])
    return _assemble("acls", budget_pct, seed, sorted(chosen), samples)


def compmatched_split(samples: list[dict], reference: dict, seed: int) -> dict:
    """Random selection with the same per-source counts as `reference` (the ACLS split of this seed)."""
    source_of = {s["id"]: s["source"] for s in samples}
    ref = reference["labelled_train"] + reference["labelled_val"]
    counts = {}
    for i in ref:
        counts[source_of[i]] = counts.get(source_of[i], 0) + 1
    rng = np.random.default_rng(seed + 20_000)
    labelled = []
    for source in sorted(counts):
        pool = sorted(s["id"] for s in samples if s["source"] == source)
        labelled += [pool[i] for i in rng.choice(len(pool), counts[source], replace=False)]
    return _assemble("compmatched", reference["budget_pct"], seed, labelled, samples)
