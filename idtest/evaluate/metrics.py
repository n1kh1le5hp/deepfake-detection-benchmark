"""Detection metrics for the ID-test-set evaluation.

Convention (documented once, used everywhere):
    - Detectors expose ``predict_real`` -> ``P(real)`` in [0, 1].
    - Ground-truth ``label``: ``1`` = fake (positive class), ``0`` = real.
    - "fake score" used by the metrics is ``s = 1 - P(real)`` so that fakes get
      higher scores. AUC/EER/accuracy are computed on ``(label, s)``.

AUC here is the standard ROC-AUC (probability a random fake outranks a random
real). All functions accept anything array-like (lists / numpy / pandas Series)
and return plain Python floats. A group that contains only one class yields
``nan`` metrics (AUC is undefined) rather than raising.
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

METRIC_COLUMNS = ["n", "auc", "accuracy", "eer"]


def _fake_score(p_real: Iterable[float]) -> np.ndarray:
    return 1.0 - np.asarray(p_real, dtype=np.float64)


def _labels(labels: Iterable[int]) -> np.ndarray:
    return np.asarray(labels, dtype=np.int64)


def auc(labels: Sequence[int], p_real: Sequence[float]) -> float:
    """ROC-AUC. ``1.0`` = perfect ranking, ``0.5`` = chance, ``0.0`` = inverted."""
    y = _labels(labels)
    s = _fake_score(p_real)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def accuracy_at(labels: Sequence[int], p_real: Sequence[float], thr: float = 0.5) -> float:
    """Fraction correct when predicting *fake* iff fake-score >= ``thr``.

    With the default ``thr=0.5`` this is "predict fake iff P(real) < 0.5".
    """
    y = _labels(labels)
    s = _fake_score(p_real)
    pred = (s >= thr).astype(np.int64)
    if y.size == 0:
        return float("nan")
    return float(np.mean(pred == y))


def eer(labels: Sequence[int], p_real: Sequence[float]) -> Tuple[float, float]:
    """Equal-error rate and the threshold (on the fake score) where it occurs.

    EER = the error rate where false-acceptance == false-rejection. Lower is
    better. Returns ``(eer_value, threshold)``; ``threshold`` is on the fake-score
    scale (so predict fake iff fake-score >= threshold).
    """
    y = _labels(labels)
    s = _fake_score(p_real)
    if np.unique(y).size < 2:
        return float("nan"), float("nan")
    fpr, tpr, thresholds = roc_curve(y, s)
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    # average the two sides for a smoother estimate at the crossing point
    eer_val = float((fpr[idx] + fnr[idx]) / 2.0)
    thr = float(thresholds[idx]) if idx < len(thresholds) else float("nan")
    return eer_val, thr


def summarize(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Compute AUC / Accuracy / EER per group.

    ``df`` must contain ``label`` (1=fake,0=real) and ``p_real`` columns.
    Returns one row per unique combination of ``group_cols`` with the metric
    columns appended (see :data:`METRIC_COLUMNS`).
    """
    group_cols = list(group_cols)
    rows = []
    if group_cols:
        iterator = df.groupby(group_cols, sort=True, dropna=False)
    else:
        iterator = [((), df)]
    for keys, g in iterator:
        y = g["label"].to_numpy()
        s = g["p_real"].to_numpy()
        if isinstance(keys, tuple):
            row = dict(zip(group_cols, keys))
        else:
            row = {group_cols[0]: keys}
        eer_val, _ = eer(y, s)
        row.update(
            n=int(len(g)),
            auc=auc(y, s),
            accuracy=accuracy_at(y, s),
            eer=eer_val,
        )
        rows.append(row)
    cols = group_cols + METRIC_COLUMNS
    return pd.DataFrame(rows, columns=cols)
