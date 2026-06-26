"""The 5 perturbations applied to the ID test set for the robustness evaluation.

All operate on BGR uint8 face images and return the same type/shape. Intensities
are randomized within the "rational ranges" declared in
``config/default.yaml`` so the perturbed set still looks visually pristine.

Paper: "we have applied five types of perturbation on our ID test set. The
videos are distorted by each type of perturbation with random intensity, which
is constrained within a rational range" — contrast change, saturation change,
Gaussian blur, JPEG compression, white Gaussian noise.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import cv2
import numpy as np


def color_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale contrast by *factor* around the per-image mean."""
    f = img.astype(np.float32)
    mean = f.mean()
    out = (f - mean) * factor + mean
    return np.clip(out, 0, 255).astype(np.uint8)


def color_saturation(img: np.ndarray, factor: float) -> np.ndarray:
    """Scale saturation by *factor* in HSV space."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= factor
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with given *sigma* (kernel size auto-computed)."""
    return cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma))


def jpeg_compression(img: np.ndarray, quality: float) -> np.ndarray:
    """Re-encode as JPEG at *quality* (0-100, int) then decode (block-artifact)."""
    q = int(round(float(quality)))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, q))])
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else img


def white_gaussian_noise(img: np.ndarray, std: float) -> np.ndarray:
    """Add i.i.d. Gaussian noise; *std* is relative to [0,1] (so 0.05 ~ 12.75/255)."""
    if std <= 0:
        return img
    noise = np.random.normal(0.0, float(std) * 255.0, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# Registry: name -> (transform_fn, is_int_param).
PERTURBATIONS: Dict[str, Tuple[Callable, bool]] = {
    "color_contrast": (color_contrast, False),
    "color_saturation": (color_saturation, False),
    "gaussian_blur": (gaussian_blur, False),
    "jpeg_compression": (jpeg_compression, True),
    "white_gaussian_noise": (white_gaussian_noise, False),
}


def sample_param(name: str, low: float, high: float, rng: np.random.Generator) -> float:
    """Uniformly sample a perturbation intensity in [low, high]."""
    val = float(rng.uniform(low, high))
    if PERTURBATIONS[name][1]:  # integer parameter (e.g. JPEG quality)
        val = float(round(val))
    return val


def apply_perturbation(name: str, img: np.ndarray, value: float) -> np.ndarray:
    """Apply a single named perturbation to *img*."""
    if name not in PERTURBATIONS:
        raise KeyError(f"Unknown perturbation '{name}'; choose from {list(PERTURBATIONS)}")
    fn = PERTURBATIONS[name][0]
    return fn(img, value)


def perturb_all(img: np.ndarray, params: Dict[str, float]) -> Dict[str, np.ndarray]:
    """Apply every named perturbation (each independently) to *img*.

    Returns ``{name: perturbed_image}``. Each copy is perturbed from the
    *original* image (not chained), matching the paper's per-type evaluation.
    """
    out = {}
    for name, value in params.items():
        out[name] = apply_perturbation(name, img, value)
    return out
