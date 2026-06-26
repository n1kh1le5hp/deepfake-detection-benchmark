"""Phase 1: detection-model selection (false acceptance).

The paper "retains falsely accepted fake examples with high confidence". We
operationalize this as: a FAKE is *hard* iff the detector ensemble classifies
it as real with probability >= tau under the chosen aggregation rule.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

# Aggregation rules over the per-detector P(real) vector.
_ENSEMBLE = {
    "all": lambda probs: probs.min(axis=1),   # AND: every detector must accept
    "any": lambda probs: probs.max(axis=1),   # OR: at least one accepts
    "mean": lambda probs: probs.mean(axis=1),  # average confidence
}


def aggregate_real_prob(per_detector_probs: np.ndarray, mode: str = "all") -> np.ndarray:
    """Reduce an (N, D) P(real) matrix to an (N,) aggregated P(real)."""
    if per_detector_probs.ndim == 1:
        return per_detector_probs
    if mode not in _ENSEMBLE:
        raise ValueError(f"Unknown ensemble mode '{mode}'; choose from {list(_ENSEMBLE)}")
    return _ENSEMBLE[mode](per_detector_probs.astype(np.float32))


def false_acceptance_mask(
    per_detector_probs: np.ndarray, tau: float = 0.5, mode: str = "all"
) -> np.ndarray:
    """Boolean mask: True where the fake is falsely accepted as real.

    *per_detector_probs* is (N, D) P(real) for N fake samples across D detectors.
    """
    agg = aggregate_real_prob(per_detector_probs, mode)
    return agg >= float(tau)


def select_hard_fakes(
    per_detector_probs: np.ndarray, tau: float = 0.5, mode: str = "all"
) -> List[int]:
    """Indices of fakes that pass Phase-1 (falsely accepted with high confidence)."""
    mask = false_acceptance_mask(per_detector_probs, tau, mode)
    return [int(i) for i in np.nonzero(mask)[0]]
