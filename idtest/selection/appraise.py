"""Combine Phase-1 and Phase-2 into the final hard-sample decision.

A fake is retained as *hard* iff it survives BOTH:
  * Phase 1: falsely accepted as real by the detector ensemble (high P(real)).
  * Phase 2: judged real by the (proxied) human-perception step.

Real samples are never "hard" — the hard flag only applies to fakes used to
build the imperceptible portion of the ID test set.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


def combine(phase1_mask: Sequence[bool], phase2_mask: Sequence[bool]) -> np.ndarray:
    """Final hard mask = Phase-1 AND Phase-2 (both must pass)."""
    a = np.asarray(phase1_mask, dtype=bool)
    b = np.asarray(phase2_mask, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(
            f"Phase masks disagree: phase1={a.shape} vs phase2={b.shape}"
        )
    return a & b


def hard_indices(phase1_mask: Sequence[bool], phase2_mask: Sequence[bool]) -> List[int]:
    """Indices of fakes that are hard after combining both phases."""
    final = combine(phase1_mask, phase2_mask)
    return [int(i) for i in np.nonzero(final)[0]]


def run_selection(
    per_detector_probs: np.ndarray,
    phase2_scores_or_votes,
    *,
    tau: float = 0.5,
    ensemble: str = "all",
    human_threshold: int = 15,
    num_participants: int = 30,
    proxy_keep_quantile: float = 0.5,
    proxy_mode: bool = True,
) -> dict:
    """One-shot selection over fakes.

    *per_detector_probs*: (N, D) P(real) from Phase-1 detectors.
    *phase2_scores_or_votes*: either an (N,) array of proxy realism scores
        (proxy_mode=True) or an (N,) array of integer "real" vote counts
        (proxy_mode=False, from the manual UI).

    Returns a dict with per-phase masks, the final hard mask, and kept indices.
    """
    from .phase1_model import false_acceptance_mask
    from .phase2_human import ProxyRealismScorer, human_study_mask

    p1 = false_acceptance_mask(per_detector_probs, tau=tau, mode=ensemble)

    arr = np.asarray(phase2_scores_or_votes, dtype=np.float32)
    if proxy_mode:
        scorer = ProxyRealismScorer.__new__(ProxyRealismScorer)  # bypass ctor
        p2 = scorer.accept_mask(arr, keep_quantile=proxy_keep_quantile)
    else:
        p2 = human_study_mask(arr.astype(int), threshold=human_threshold, num_participants=num_participants)

    final = combine(p1, p2)
    return {
        "phase1_mask": p1,
        "phase2_mask": p2,
        "hard_mask": final,
        "hard_indices": [int(i) for i in np.nonzero(final)[0]],
    }
