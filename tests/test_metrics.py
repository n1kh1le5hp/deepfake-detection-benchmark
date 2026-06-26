"""Tests for the evaluation metrics (hand-checked label/score arrays)."""
import math

import numpy as np
import pandas as pd

from idtest.evaluate.metrics import auc, accuracy_at, eer, summarize


def test_auc_perfect_separation():
    # fakes have low P(real) -> high fake score -> perfect ranking
    assert auc([0, 0, 1, 1], [0.9, 0.8, 0.1, 0.2]) == 1.0


def test_auc_known_partial():
    # fake scores = 1 - p_real = [0.6, 0.4, 0.55, 0.9]; 3 of 4 pairs concordant
    assert auc([0, 0, 1, 1], [0.4, 0.6, 0.45, 0.1]) == 0.75


def test_auc_chance_when_tied():
    assert auc([0, 1], [0.5, 0.5]) == 0.5


def test_auc_single_class_is_nan():
    assert math.isnan(auc([0, 0], [0.9, 0.1]))


def test_auc_accepts_numpy_and_series():
    import pandas as pd
    assert auc(np.array([0, 1]), pd.Series([0.9, 0.1])) == 1.0


def test_accuracy_default_threshold():
    # fake scores = [0.1, 0.2, 0.9, 0.8]; >=0.5 -> idx2,3 fake -> all correct
    assert accuracy_at([0, 0, 1, 1], [0.9, 0.8, 0.1, 0.2]) == 1.0


def test_accuracy_partial():
    # fake scores = [0.6, 0.4, 0.55, 0.9]; >=0.5 -> idx0,2,3 fake -> idx0 wrong -> 0.75
    assert accuracy_at([0, 0, 1, 1], [0.4, 0.6, 0.45, 0.1]) == 0.75


def test_eer_zero_for_perfect_separation():
    val, thr = eer([0, 0, 1, 1], [0.9, 0.8, 0.1, 0.2])
    assert val == 0.0


def test_eer_in_unit_interval():
    val, _ = eer([0, 0, 1, 1], [0.4, 0.6, 0.45, 0.1])
    assert 0.0 <= val <= 1.0


def test_summarize_groups_and_columns():
    df = pd.DataFrame(
        {
            "detector": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "label":    [0, 0, 1, 1, 0, 0, 1, 1],
            "p_real":   [0.9, 0.8, 0.1, 0.2, 0.4, 0.6, 0.45, 0.1],
        }
    )
    out = summarize(df, ["detector"])
    assert list(out["detector"]) == ["a", "b"]
    assert list(out.columns) == ["detector", "n", "auc", "accuracy", "eer"]
    assert out.loc[out.detector == "a", "auc"].iloc[0] == 1.0
    assert out.loc[out.detector == "b", "auc"].iloc[0] == 0.75
    assert (out["n"] == 4).all()


def test_summarize_multi_group():
    df = pd.DataFrame(
        {
            "detector":  ["a", "a", "a", "a"],
            "condition": ["clean", "clean", "blur", "blur"],
            "label":     [0, 1, 0, 1],
            "p_real":    [0.9, 0.1, 0.9, 0.1],
        }
    )
    out = summarize(df, ["detector", "condition"])
    assert len(out) == 2
    assert set(zip(out.detector, out.condition)) == {("a", "clean"), ("a", "blur")}
    assert (out["auc"] == 1.0).all()
