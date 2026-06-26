#!/usr/bin/env python
"""00_setup_check.py — validate the environment without processing data.

Checks: ffmpeg/ffprobe, Dlib (+ landmark model), optional torch/CUDA, dataset
roots, optional FSGAN/MegaFS repos. Exits non-zero if anything required is
missing. Run this first: ``python scripts/00_setup_check.py``.
"""
from __future__ import annotations

import sys

from _common import ext, load

from idtest.io_utils import ffmpeg_available
from idtest.preprocess.face_align import locate_model
from idtest.datasets.registry import build_datasets


def _ok(msg): print(f"  [OK]   {msg}")
def _warn(msg): print(f"  [WARN] {msg}")
def _fail(msg): print(f"  [FAIL] {msg}"); return False


def main() -> int:
    cfg = load()
    all_good = True

    print("== binaries ==")
    if ffmpeg_available():
        _ok("ffmpeg + ffprobe found")
    else:
        all_good &= _fail("ffmpeg/ffprobe missing (needed for frame extraction)")

    print("== dlib + face alignment ==")
    try:
        import dlib  # noqa: F401
        _ok("dlib importable")
    except Exception:
        all_good &= _fail("dlib not importable (pip install dlib; needs cmake)")
    lm = locate_model(cfg["face"]["landmark_model"], ext(cfg))
    if lm.exists():
        _ok(f"landmark model: {lm}")
    else:
        _warn(f"landmark model missing at {lm} (download from dlib.net/files/)")

    print("== torch / cuda (needed for detectors + generation) ==")
    try:
        import torch
        _ok(f"torch {torch.__version__}; CUDA available={torch.cuda.is_available()}")
        if cfg.get("device") == "cuda" and not torch.cuda.is_available():
            _warn("device=cuda but no GPU detected; will fall back to CPU")
    except Exception:
        _warn("torch not installed (Phase-1/2 proxy and generation will be unavailable)")

    print("== source datasets ==")
    datasets = []
    try:
        datasets = build_datasets(cfg)
    except FileNotFoundError as e:
        all_good &= _fail(str(e))
    if datasets:
        for ds in datasets:
            _ok(f"{ds.name}: {ds.root}")
    else:
        _warn("no enabled datasets — set roots in config/datasets.yaml")

    print("== optional generators ==")
    for sub, label in (("fsgan", "FSGAN"), ("megafs", "MegaFS")):
        repo = ext(cfg) / sub
        if repo.is_dir():
            _ok(f"{label} repo present: {repo}")
        elif cfg["generation"][sub]["enabled"]:
            _warn(f"{label} enabled but repo missing at {repo}")
        else:
            _warn(f"{label} not set up (disabled) — private fakes will be skipped")

    print()
    print("RESULT:", "ready" if all_good else "issues found (see FAIL/WARN above)")
    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
