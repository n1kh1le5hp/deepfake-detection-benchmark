"""Evaluation harness for the ID test set.

Scores deepfake detectors over the assembled test set (clean + perturbations)
and reports AUC / Accuracy / EER. Two submodules:

- :mod:`idtest.evaluate.metrics` — pure metric functions on ``(labels, p_real)``.
- :mod:`idtest.evaluate.harness`  — sample-table construction + batched scoring.

See ``scripts/08_evaluate.py`` for the command-line entry point.
"""
from idtest.evaluate.metrics import (  # noqa: F401
    auc,
    accuracy_at,
    eer,
    summarize,
    METRIC_COLUMNS,
)
