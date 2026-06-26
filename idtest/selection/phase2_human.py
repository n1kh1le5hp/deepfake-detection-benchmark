"""Phase 2: user-perception selection.

The paper runs a blind study and keeps fakes judged real by >= 15 of 30
participants. True human annotation is out of scope for an automated pipeline,
so we provide two modes:

* ``proxy`` (default, APPROXIMATION): a *different* model scores perceived
  realism; a fake is accepted if its realism score exceeds a threshold.
* ``manual`` (faithful): apply the real 15/30 rule to vote tallies collected by
  ``annotation_ui/``.

Both paths share the :func:`passes_human_study` threshold check so the manual UI
and the proxy use the identical acceptance rule.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .detectors import Detector


def passes_human_study(votes_real: Sequence[int], threshold: int = 15, num_participants: int = 30) -> bool:
    """Paper rule: a fake is accepted iff >= *threshold* of *num_participants*
    judges mark it as real.

    *votes_real* may be a single int or a list of 0/1 votes; both reduce to the
    count of "real" judgements.
    """
    if isinstance(votes_real, int):
        n_real = votes_real
    else:
        n_real = int(np.sum(votes_real))
    return n_real >= int(threshold)


def human_study_mask(
    vote_counts: Sequence[int], threshold: int = 15, num_participants: int = 30
) -> np.ndarray:
    """Vectorized form of :func:`passes_human_study` over many samples."""
    return np.array(
        [passes_human_study(int(v), threshold, num_participants) for v in vote_counts],
        dtype=bool,
    )


class ProxyRealismScorer:
    """Model-proxy for the human-perception step.

    Wraps a detector that scores *perceived realism*. To avoid circularity with
    Phase-1, this should be a different model (e.g. a strong general classifier
    or a second, independent deepfake detector). The acceptance threshold is
    expressed as a quantile of the proxy scores so it is scale-invariant.
    """

    def __init__(self, detector: Detector):
        self.detector = detector

    def score(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Return per-image realism scores in [0,1] (higher = more real-looking)."""
        return self.detector.predict_real(images).astype(np.float32)

    def accept_mask(self, scores: np.ndarray, keep_quantile: float = 0.5) -> np.ndarray:
        """Accept fakes whose realism score is in the top (1-keep_quantile) fraction.

        With ``keep_quantile=0.5`` the proxy approximates "judged real by half
        the participants", mirroring the 15/30 threshold.
        """
        thr = float(np.quantile(scores, keep_quantile))
        return scores >= thr


def proxy_accept_mask(
    scorer: ProxyRealismScorer,
    images: Sequence[np.ndarray],
    keep_quantile: float = 0.5,
) -> np.ndarray:
    """Convenience: score images with the proxy and apply the acceptance rule."""
    scores = scorer.score(images)
    return scorer.accept_mask(scores, keep_quantile)
