"""Balance the selected hard fakes + reals to the paper's target counts.

Goals (defaults from config/default.yaml):
  * 25,697 fake images + 25,697 real images (frame-level, balanced).
  * >= 13 distinct manipulation methods among fakes (config: balance.min_methods).
  * Full manipulation-type coverage (AEGAN / Graphic / Unknown) where available.

Fakes are selected by stratified sampling across methods (so coverage is
maximized), then filled out to the target count. Reals are a plain seeded
random sample. Both use the provided ``rng`` for determinism.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np

from ..datasets.base import Sample


def _by_method(fakes: Sequence[Sample]) -> Dict[str, List[Sample]]:
    groups: Dict[str, List[Sample]] = defaultdict(list)
    for s in fakes:
        groups[s.method or "unknown"].append(s)
    return groups


def select_fakes(
    hard_fakes: Sequence[Sample],
    target: int,
    min_methods: int,
    rng: np.random.Generator,
) -> List[Sample]:
    """Stratified selection of *target* fakes maximizing method coverage."""
    groups = _by_method(hard_fakes)
    methods = sorted(groups.keys())

    if len(methods) < min_methods:
        print(
            f"[balance] WARNING: only {len(methods)} distinct methods available "
            f"(target >= {min_methods}). Proceeding anyway — obtain more source "
            f"datasets to reach the method-coverage goal."
        )

    # Shuffle within each method deterministically.
    for m in methods:
        rng.shuffle(groups[m])

    selected: List[Sample] = []
    # Round 1: guarantee every method is represented (one each).
    for m in methods:
        selected.append(groups[m].pop(0))

    # Round 2: distribute the remainder proportionally to group size.
    remaining = target - len(selected)
    while remaining > 0 and any(groups.values()):
        # Recompute pool sizes for proportional allocation.
        total = sum(len(g) for g in groups.values())
        if total == 0:
            break
        progressed = False
        for m in methods:
            if remaining <= 0:
                break
            if not groups[m]:
                continue
            quota = int(round(remaining_pool_alloc(len(groups[m]), total, remaining)))
            take = min(quota, len(groups[m]), remaining)
            if take <= 0:
                take = 1 if remaining > 0 and groups[m] else 0
            for _ in range(take):
                if groups[m] and remaining > 0:
                    selected.append(groups[m].pop(0))
                    remaining -= 1
                    progressed = True
        if not progressed:
            break

    if len(selected) < target:
        print(
            f"[balance] WARNING: only {len(selected)} hard fakes available "
            f"(target {target}). Lower detection thresholds or add datasets."
        )
    return selected[:target]


def remaining_pool_alloc(group_size: int, total: int, remaining: int) -> float:
    """Proportional share of *remaining* slots for a group of *group_size*."""
    return (group_size / total) * remaining if total else 0.0


def select_reals(
    reals: Sequence[Sample], target: int, rng: np.random.Generator
) -> List[Sample]:
    """Plain seeded random sample of reals (no per-method constraint)."""
    idx = np.arange(len(reals))
    rng.shuffle(idx)
    chosen = [reals[i] for i in idx[:target]]
    if len(chosen) < target:
        print(
            f"[balance] WARNING: only {len(chosen)} reals available "
            f"(target {target})."
        )
    return chosen
