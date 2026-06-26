"""Tests for Phase-1 / Phase-2 selection logic and the detector scaffolding."""
import numpy as np

from idtest.selection.detectors import ConstantDetector, RandomDetector, ensemble_predict
from idtest.selection.phase1_model import false_acceptance_mask, select_hard_fakes
from idtest.selection.phase2_human import human_study_mask, passes_human_study
from idtest.selection.appraise import combine, hard_indices


# ---- Phase 1: false acceptance -----------------------------------------

def test_phase1_all_mode_requires_every_detector():
    # 3 fakes, 2 detectors. Only the middle fake is accepted by both.
    probs = np.array([
        [0.9, 0.1],   # detector 2 rejects -> NOT hard (mode='all')
        [0.8, 0.7],   # both accept -> hard
        [0.2, 0.3],   # neither -> not hard
    ], dtype=np.float32)
    mask = false_acceptance_mask(probs, tau=0.5, mode="all")
    assert mask.tolist() == [False, True, False]
    assert select_hard_fakes(probs, tau=0.5, mode="all") == [1]


def test_phase1_any_mode():
    probs = np.array([[0.9, 0.1], [0.2, 0.2]], dtype=np.float32)
    mask = false_acceptance_mask(probs, tau=0.5, mode="any")
    assert mask.tolist() == [True, False]


def test_phase1_mean_mode():
    probs = np.array([[0.9, 0.2], [0.6, 0.5]], dtype=np.float32)
    # row0 mean=0.55 -> hard ; row1 mean=0.55 -> hard
    mask = false_acceptance_mask(probs, tau=0.5, mode="mean")
    assert mask.tolist() == [True, True]


# ---- Phase 2: human-perception ----------------------------------------

def test_human_threshold_boundary():
    assert passes_human_study(15, threshold=15, num_participants=30) is True
    assert passes_human_study(14, threshold=15, num_participants=30) is False
    # list of votes reduces to count.
    assert passes_human_study([1] * 16 + [0] * 14, threshold=15, num_participants=30) is True


def test_human_study_mask_vectorized():
    counts = [10, 15, 16, 29, 0]
    mask = human_study_mask(counts, threshold=15, num_participants=30)
    assert mask.tolist() == [False, True, True, True, False]


# ---- Combine ------------------------------------------------------------

def test_combine_is_logical_and():
    p1 = np.array([True, True, False, True])
    p2 = np.array([True, False, True, True])
    assert combine(p1, p2).tolist() == [True, False, False, True]
    assert hard_indices(p1, p2) == [0, 3]


def test_combine_shape_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        combine([True, False], [True])


# ---- Detector scaffolding (no real weights) ----------------------------

def test_constant_detector():
    d = ConstantDetector(p_real=0.7)
    out = d.predict_real([np.zeros((8, 8, 3), np.uint8) for _ in range(3)])
    assert out.shape == (3,)
    assert np.allclose(out, 0.7)


def test_random_detector_is_deterministic():
    d1 = RandomDetector(seed=42)
    d2 = RandomDetector(seed=42)
    imgs = [np.zeros((8, 8, 3), np.uint8) for _ in range(5)]
    np.testing.assert_array_equal(d1.predict_real(imgs), d2.predict_real(imgs))


def test_ensemble_predict_shape():
    dets = [ConstantDetector(0.9, name="a"), ConstantDetector(0.1, name="b")]
    imgs = [np.zeros((8, 8, 3), np.uint8) for _ in range(4)]
    mat = ensemble_predict(dets, imgs)
    assert mat.shape == (4, 2)
