"""Tests for the 5 perturbations: shape/dtype preservation + param bounds."""
import numpy as np
import pytest

from idtest.perturb.transforms import (
    PERTURBATIONS, apply_perturbation, color_contrast, color_saturation,
    gaussian_blur, jpeg_compression, perturb_all, sample_param,
    white_gaussian_noise,
)


@pytest.fixture
def face():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


def _all_transforms():
    return [
        (color_contrast, 1.2), (color_saturation, 0.8), (gaussian_blur, 1.0),
        (jpeg_compression, 40), (white_gaussian_noise, 0.02),
    ]


@pytest.mark.parametrize("fn,value", _all_transforms())
def test_shape_and_dtype_preserved(face, fn, value):
    out = fn(face, value)
    assert out.shape == face.shape
    assert out.dtype == np.uint8


def test_contrast_factor_one_is_identity(face):
    out = color_contrast(face, 1.0)
    np.testing.assert_array_equal(out, face)


def test_jpeg_quality_clamped(face):
    # quality out of range must not raise and must return uint8 image.
    out = jpeg_compression(face, 200)
    assert out.shape == face.shape and out.dtype == np.uint8


def test_sample_param_within_range():
    rng = np.random.default_rng(1)
    for name in PERTURBATIONS:
        is_int = PERTURBATIONS[name][1]
        # Integer params (e.g. JPEG quality) need an integer-valued range.
        lo, hi = (20, 70) if is_int else (0.1, 0.9)
        for _ in range(50):
            v = sample_param(name, lo, hi, rng)
            assert lo <= v <= hi
            if is_int:
                assert float(v).is_integer()


def test_sample_param_jpeg_is_int():
    rng = np.random.default_rng(2)
    for _ in range(20):
        assert float(sample_param("jpeg_compression", 20, 70, rng)).is_integer()


def test_apply_unknown_raises(face):
    with pytest.raises(KeyError):
        apply_perturbation("nope", face, 1.0)


def test_perturb_all_independent(face):
    rng = np.random.default_rng(3)
    params = {n: sample_param(n, 0.2, 0.8, rng) for n in PERTURBATIONS}
    out = perturb_all(face, params)
    assert set(out) == set(PERTURBATIONS)
    for img in out.values():
        assert img.shape == face.shape and img.dtype == np.uint8
