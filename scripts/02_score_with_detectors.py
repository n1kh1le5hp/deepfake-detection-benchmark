#!/usr/bin/env python
"""02_score_with_detectors.py — run the Phase-1 detector ensemble over fake faces.

Loads data/work/aligned_index.json, scores every FAKE face with each configured
detector, and saves per-detector P(real) to data/work/fake_scores.npz alongside
the fake UIDs. Reals are skipped here (Phase-1 only concerns false acceptance
of fakes).

Usage: python scripts/02_score_with_detectors.py [--batch 32]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from tqdm import tqdm

from _common import ext, load, read_json, work

from idtest.io_utils import read_image
from idtest.selection.detectors import build_detector


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args(argv)

    cfg = load()
    index = read_json(work(cfg) / "aligned_index.json") or []
    fakes = [e for e in index if e["split"] == "fake"]
    if not fakes:
        print("[02] no fake faces in index; nothing to score.")
        return 0

    names = cfg["phase1"]["detectors"]
    detectors = [build_detector(n, cfg, ext(cfg)) for n in names]
    print(f"[02] scoring {len(fakes)} fakes with detectors {names}")

    D = len(detectors)
    probs = np.zeros((len(fakes), D), dtype=np.float32)
    for i in tqdm(range(0, len(fakes), args.batch), desc="[02] scoring"):
        chunk = fakes[i:i + args.batch]
        imgs = []
        valid = []
        for e in chunk:
            try:
                imgs.append(read_image(e["path"]))
                valid.append(True)
            except Exception:
                imgs.append(None)
                valid.append(False)
        # Replace unreadable with a black image to keep batching simple, mask later.
        imgs = [x if x is not None else np.zeros((cfg["face"]["size"], cfg["face"]["size"], 3), np.uint8) for x in imgs]
        for d, det in enumerate(detectors):
            probs[i:i + len(chunk), d] = det.predict_real(imgs)

    np.savez(
        work(cfg) / "fake_scores.npz",
        probs=probs,
        uids=np.array([e["uid"] for e in fakes]),
    )
    print(f"[done] saved fake_scores.npz  shape={probs.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
