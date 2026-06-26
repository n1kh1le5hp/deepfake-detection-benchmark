#!/usr/bin/env python
"""Generate SYNTHETIC source data so the pipeline scripts can be exercised
end-to-end without the real multi-TB datasets.

Produces small PNG "faces" (procedural blobs) in the on-disk layouts each reader
expects, with >= 13 manipulation methods across all 7 datasets. This is for a
PROOF-OF-CONCEPT run only — it is NOT the real ID test set.

Run:  python demo/make_synthetic_data.py [--per 12]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RNG = np.random.default_rng(2024)


def face_image(seed: int, size: int = 64) -> np.ndarray:
    """A cheap procedural 'face' so perturbations have something to act on."""
    r = np.random.default_rng(seed)
    img = np.full((size, size, 3), 0, np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = size / 2
    face = ((xx - cx) ** 2 / (0.34 * size) ** 2 + (yy - cy) ** 2 / (0.45 * size) ** 2) <= 1
    for c in range(3):
        img[..., c] = np.where(face, 120 + int(r.integers(-30, 90)), 20)
    # eyes + mouth
    for ex in (cx - 10, cx + 10):
        img[int(cy - 6):int(cy - 2), int(ex - 2):int(ex + 2)] = 30
    img[int(cy + 8):int(cy + 11), int(cx - 8):int(cx + 8)] = 60
    img = np.clip(img.astype(np.int16) + r.integers(-12, 12, img.shape), 0, 255).astype(np.uint8)
    return img


def write_png(path: Path, seed: int):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), face_image(seed))


def gen_simple(folder: str, n: int, *, real_sub: str, fake_sub: str, start: int = 0):
    for i in range(n):
        write_png(RAW / folder / real_sub / f"r{i:04d}.png", start + i)
        write_png(RAW / folder / fake_sub / f"f{i:04d}.png", start + 1000 + i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=12, help="images per real/fake bucket per dataset")
    args = ap.parse_args()
    n = args.per

    # UADFV
    gen_simple("UADFV", n, real_sub="real", fake_sub="fake")
    # DF-TIMIT (higher/lower quality)
    gen_simple("DF-TIMIT/higher_quality", n, real_sub="real", fake_sub="fake")
    gen_simple("DF-TIMIT/lower_quality", n, real_sub="real", fake_sub="fake")
    # Celeb-DF
    for i in range(n):
        write_png(RAW / "Celeb-DF" / "Celeb-real" / f"r{i:04d}.png", 2000 + i)
        write_png(RAW / "Celeb-DF" / "YouTube-real" / f"y{i:04d}.png", 2100 + i)
        write_png(RAW / "Celeb-DF" / "Celeb-synthesis" / f"s{i:04d}.png", 2200 + i)
    # DeeperForensics-1.0
    gen_simple("DeeperForensics-1.0", n, real_sub="source_videos", fake_sub="manipulated_videos")
    # FF++ (sequence dirs of frames): 3 methods
    for m, base_seed in (("deepfakes", 3000), ("faceswap", 4000), ("faceshifter", 5000)):
        for i in range(n):
            d = RAW / "FaceForensics++" / "manipulated_sequences" / m / "c23" / "sequences" / f"{i:04d}_{i+1:04d}"
            for f in range(3):
                write_png(d / f"{f:03d}.png", base_seed + i * 10 + f)
    for i in range(n):
        d = RAW / "FaceForensics++" / "original_sequences" / "youtube" / "c23" / "sequences" / f"{i:04d}"
        for f in range(3):
            write_png(d / f"{f:03d}.png", 6000 + i * 10 + f)
    # DFDC (per-part metadata.json)
    part = RAW / "DFDC" / "dfdc_train_part_0"
    meta = {}
    for i in range(n):
        write_png(part / "videos" / f"r{i:04d}.png", 7000 + i); meta[f"r{i:04d}.png"] = {"label": "REAL"}
        write_png(part / "videos" / f"f{i:04d}.png", 7100 + i); meta[f"f{i:04d}.png"] = {"label": "FAKE"}
    (part / "metadata.json").write_text(json.dumps(meta))
    # ForgeryNet (label CSV with multiple method categories)
    cats = ["FS", "DF", "F2F", "NT", "AEGAN", "Graphic"]
    rows = []
    k = 0
    for cat in cats:
        for i in range(max(2, n // 3)):
            rel = f"imgs/{cat}/{i:04d}.png"
            write_png(RAW / "ForgeryNet" / rel, 8000 + k); k += 1
            rows.append({"path": rel, "category": cat})
    with open(RAW / "ForgeryNet" / "ForgeryNet_v1_test_label.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "category"]); w.writeheader(); w.writerows(rows)

    # Count methods for the demo config.
    methods = {"UADFV", "DF-TIMIT-HQ", "DF-TIMIT-LQ", "Celeb-DF", "DF-1.0",
               "FF++-DF", "FF++-FS", "FF++-FShifter", "DFDC"} | set(cats)
    print(f"[demo] wrote synthetic data under {RAW}")
    print(f"[demo] distinct methods available: {len(methods)} -> {sorted(methods)}")


if __name__ == "__main__":
    sys.exit(main() or 0)
