#!/usr/bin/env python
"""01_extract_and_align.py — extract frames + align faces for all source data.

For every enabled sample:
  * video        -> ffmpeg frame extraction (capped) -> per-frame face align
  * frames-dir   -> images in dir -> per-frame face align
  * single image -> face align
Aligned faces are written to data/work/aligned/<uid>.png and an index of all
extracted face-images is saved to data/work/aligned_index.json.

Usage: python scripts/01_extract_and_align.py [--limit N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from _common import ext, load, work, write_json

from idtest.datasets.base import IMAGE_EXTS
from idtest.datasets.registry import build_datasets
from idtest.io_utils import hash_id, read_image, write_image
from idtest.preprocess.face_align import FaceAligner
from idtest.preprocess.frame_extraction import extract_frames


def _sample_frames(sample, frames_dir: Path, cfg) -> list[Path]:
    """Return the list of source frame/image paths for *sample*."""
    pp = cfg["preprocess"]
    p = Path(sample.path)
    if sample.is_video:
        return extract_frames(
            p, frames_dir, fps=pp.get("fps"), stride=pp.get("stride"),
            max_frames=pp.get("max_frames_per_video"),
        )
    if p.is_dir():  # FF++ sequence directory of frames
        imgs = sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS)
        cap = pp.get("max_frames_per_video") or len(imgs)
        return imgs[:cap]
    return [p]  # single image


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap #samples (smoke test)")
    ap.add_argument("--max-faces-per-sample", type=int, default=8)
    ap.add_argument("--skip-align", action="store_true",
                    help="use input images as-is (no Dlib alignment) — for testing "
                         "on pre-aligned or synthetic data without dlib/ffmpeg")
    args = ap.parse_args(argv)

    cfg = load()
    if args.skip_align:
        aligner = None
    else:
        aligner = FaceAligner(
            size=cfg["face"]["size"],
            detector=cfg["face"]["detector"],
            landmark_model=cfg["face"]["landmark_model"],
            external_root=ext(cfg),
            min_face_size=cfg["face"]["min_face_size"],
        )
    aligned_dir = work(cfg) / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    index = []
    datasets = build_datasets(cfg)
    n = 0
    for ds in datasets:
        samples = list(ds)
        if args.limit:
            samples = samples[: max(0, args.limit - n)]
        for s in tqdm(samples, desc=f"[{ds.name}]"):
            frames_dir = work(cfg) / "frames" / hash_id(s.path)
            try:
                frames = _sample_frames(s, frames_dir, cfg)
            except Exception as e:
                tqdm.write(f"[skip] {s.path}: {e}")
                continue
            kept = 0
            for fp in frames:
                if kept >= args.max_faces_per_sample:
                    break
                try:
                    img = read_image(fp)
                except Exception:
                    continue
                if aligner is None:
                    face = img  # --skip-align: treat input as already aligned
                else:
                    face = aligner.align_largest(img)
                    if face is None:
                        continue
                uid = hash_id(s.path, fp.name, kept)
                face_path = aligned_dir / f"{uid}.png"
                write_image(face_path, face)
                index.append({
                    "uid": uid,
                    "split": s.split,
                    "source_dataset": s.source_dataset,
                    "method": s.method,
                    "manip_type": s.manip_type,
                    "video_id": s.video_id,
                    "path": str(face_path),
                    "raw_sample_path": s.path,
                    "hard": 0,
                })
                kept += 1
            n += 1
        if args.limit and n >= args.limit:
            break

    write_json(index, work(cfg) / "aligned_index.json")
    print(f"[done] {len(index)} aligned faces from {n} samples -> {work(cfg)}/aligned_index.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
