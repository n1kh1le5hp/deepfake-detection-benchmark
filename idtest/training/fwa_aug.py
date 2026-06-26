"""On-the-fly FWA-style synthetic-fake generation for training.

FWA ("Exposing DeepFake Videos by Detecting Face Warping Artifacts", Li & Lyu)
trains a detector on REAL faces plus synthetic fakes that simulate the
resolution inconsistency a face-swap pipeline leaves behind: **downscale ->
Gaussian blur -> upscale** reintroduces the warping artifact the detector learns
to spot. DeepFakeBench's full pipeline runs on raw images (detect -> align ->
warp the inner face -> blend back); since our training pool is **already-aligned
256x256 faces**, we apply the artifact directly to the aligned face.

This is a faithful-enough approximation for a small-data proof-of-concept — it
captures FWA's core signal (resolution/blur inconsistency) — not a byte-exact
reimplementation of FWA's raw-image blend pipeline.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def generate_fwa_negative(face_bgr: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Turn an aligned BGR uint8 face into a synthetic 'fake'.

    downscale (rand 0.2-0.8) -> Gaussian blur -> upscale (reintroduces the
    warping artifact), then blend with the original at a random ratio so the
    artifact is partial (mimicking a real blend boundary). Returns BGR uint8,
    same shape as input.
    """
    rng = rng or np.random.default_rng()
    h, w = face_bgr.shape[:2]
    scale = float(rng.uniform(0.2, 0.8))
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(face_bgr, (sw, sh), interpolation=cv2.INTER_LINEAR)
    small = cv2.GaussianBlur(small, (5, 5), 0)
    warped = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    alpha = float(rng.uniform(0.5, 1.0))          # how much of the warped face shows through
    return cv2.addWeighted(warped, alpha, face_bgr, 1.0 - alpha, 0)


def augment(face_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light augmentation: random horizontal flip + JPEG re-compression."""
    out = face_bgr
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    if rng.random() < 0.5:
        q = int(rng.integers(40, 101))
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                out = dec
    return out


def to_tensor(face_bgr: np.ndarray, size: int = 256):
    """BGR uint8 face -> normalized CHW float tensor (x-0.5)/0.5, RGB order."""
    import torch

    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size))
    arr = (rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
