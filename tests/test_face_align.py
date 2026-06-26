"""Tests for face-alignment pure helpers.

The full FaceAligner needs the Dlib landmark model on disk; we skip those when
the model is absent and instead test the pure landmark/rect helpers.
"""
from types import SimpleNamespace

import numpy as np
import pytest

dlib = pytest.importorskip("dlib")  # skip whole module if dlib not installed

from idtest.preprocess.face_align import FaceAligner, _five_from_68, _rect_area


def test_five_from_68_shape():
    pts = np.arange(68 * 2, dtype=np.float32).reshape(68, 2)
    five = _five_from_68(pts)
    assert five.shape == (5, 2)
    # left eye = mean of points 36:42.
    expected_le = pts[36:42].mean(axis=0)
    np.testing.assert_allclose(five[0], expected_le)


def test_rect_area_plain_rectangle():
    r = SimpleNamespace(left=0, top=0, right=10, bottom=4)
    assert _rect_area(r) == 40


def test_rect_area_unwraps_mmod():
    inner = SimpleNamespace(left=0, top=0, right=8, bottom=8)
    mmod = SimpleNamespace(rect=inner)
    assert _rect_area(mmod) == 64


def test_template_is_5_points_normalized():
    from idtest.preprocess.face_align import _TEMPLATE_5
    assert _TEMPLATE_5.shape == (5, 2)
    assert _TEMPLATE_5.min() >= 0.0 and _TEMPLATE_5.max() <= 1.0


def test_facealigner_missing_model_raises(tmp_path):
    # external_root is empty -> model lookup should fail cleanly.
    with pytest.raises(FileNotFoundError):
        FaceAligner(size=256, landmark_model="does_not_exist.dat", external_root=tmp_path)
