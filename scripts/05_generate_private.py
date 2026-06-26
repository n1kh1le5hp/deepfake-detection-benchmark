#!/usr/bin/env python
"""05_generate_private.py — reproduce the paper's self-generated (FSGAN/MegaFS) fakes.

If enabled in config, this:
  1. Generates FSGAN fake videos (CelebA identity -> FF++/real video) and
     MegaFS fake images (identity -> CelebA image).
  2. Extracts + aligns faces from the generated outputs.
  3. Appends them to data/work/aligned_index.json.
  4. Scores them with the Phase-1 ensemble + Phase-2 proxy and merges the
     passing (hard) UIDs into data/work/hard_uids.json.

Skipped silently if both generators are disabled. See generation/README.md for
external repo + weights setup.
"""
from __future__ import annotations

import sys

import numpy as np
from tqdm import tqdm

from _common import ext, load, read_json, work, write_json

from idtest.datasets.registry import build_datasets
from idtest.generation.fsgan_wrap import FSGANGenerator, build_pairs
from idtest.generation.megafs_wrap import MegaFSGenerator
from idtest.io_utils import hash_id, read_image, write_image
from idtest.preprocess.face_align import FaceAligner
from idtest.selection.detectors import build_detector
from idtest.selection.phase1_model import false_acceptance_mask
from idtest.selection.phase2_human import ProxyRealismScorer


def _gather(cfg, predicate):
    """Return raw sample paths matching *predicate(samples)* across datasets."""
    out = []
    for ds in build_datasets(cfg):
        for s in ds:
            if predicate(s):
                out.append(s.path)
    return out


def _align_outputs(cfg, aligner, paths, source_dataset, method, manip_type):
    """Extract the largest face from each generated image/video-frame; return index entries."""
    from idtest.preprocess.frame_extraction import extract_frames
    from idtest.datasets.base import IMAGE_EXTS

    aligned_dir = work(cfg) / "aligned"
    entries = []
    for i, p in enumerate(tqdm(paths, desc=f"[05] align {source_dataset}")):
        frames = []
        if Path(p).is_dir() or Path(p).suffix.lower() in IMAGE_EXTS:
            frames = [Path(p)] if Path(p).is_file() else sorted(q for q in Path(p).iterdir() if q.suffix.lower() in IMAGE_EXTS)
        else:  # generated video
            frames = extract_frames(p, work(cfg) / "gen_frames" / hash_id(p), fps=1.0, max_frames=4)
        for fp in frames[:4]:
            try:
                face = aligner.align_largest(read_image(fp))
            except Exception:
                continue
            if face is None:
                continue
            uid = hash_id(p, fp.name, source_dataset, i)
            face_path = aligned_dir / f"{uid}.png"
            write_image(face_path, face)
            entries.append({
                "uid": uid, "split": "fake", "source_dataset": source_dataset,
                "method": method, "manip_type": manip_type,
                "video_id": Path(p).stem, "path": str(face_path),
                "raw_sample_path": p, "hard": 0,
            })
    return entries


def _select_inline(cfg, entries):
    """Phase-1 + proxy Phase-2 over *entries*; return list of hard UIDs."""
    if not entries:
        return []
    p1 = cfg["phase1"]
    dets = [build_detector(n, cfg, ext(cfg)) for n in p1["detectors"]]
    imgs = [read_image(e["path"]) for e in entries]
    # Score in one batch (generated sets are small).
    probs = np.zeros((len(entries), len(dets)), dtype=np.float32)
    for j, d in enumerate(dets):
        probs[:, j] = d.predict_real(imgs)
    p1_mask = false_acceptance_mask(probs, tau=p1["tau"], mode=p1["ensemble"])

    scorer = ProxyRealismScorer(build_detector(cfg["phase2"]["proxy_model"] or "default", cfg, ext(cfg)))
    scores = scorer.score(imgs)
    p2_mask = scorer.accept_mask(scores, keep_quantile=0.5)

    hard = p1_mask & p2_mask
    return [e["uid"] for e, m in zip(entries, hard) if m]


def main() -> int:
    from pathlib import Path
    cfg = load()
    gen = cfg["generation"]
    if not (gen["fsgan"]["enabled"] or gen["megafs"]["enabled"]):
        print("[05] generation disabled; skipping private-fake stage.")
        return 0

    aligner = FaceAligner(
        size=cfg["face"]["size"], detector=cfg["face"]["detector"],
        landmark_model=cfg["face"]["landmark_model"],
        external_root=ext(cfg), min_face_size=cfg["face"]["min_face_size"],
    )

    # Identity sources: CelebA images. Targets: real videos (FSGAN) / images (MegaFS).
    identities = _gather(cfg, lambda s: s.source_dataset == "CelebA" and not s.is_video)
    real_videos = _gather(cfg, lambda s: s.is_video and s.split == "real")
    real_images = _gather(cfg, lambda s: (not s.is_video) and s.split == "real" and s.source_dataset != "CelebA")

    new_entries = []
    if gen["fsgan"]["enabled"]:
        if not identities or not real_videos:
            print("[05] FSGAN: need CelebA identities + real videos; skipping.")
        else:
            pairs = build_pairs(real_videos, identities, gen["fsgan"]["target_videos"])
            fsgan = FSGANGenerator(ext(cfg) / "fsgan", work(cfg) / "gen_fsgan",
                                   weights_dir=ext(cfg) / "fsgan" / "weights")
            produced = fsgan.generate(pairs)
            new_entries += _align_outputs(cfg, aligner, produced, "FSGAN", "FSGAN", "AEGAN")

    if gen["megafs"]["enabled"]:
        targets = real_images or identities
        if not identities or not targets:
            print("[05] MegaFS: need CelebA identities + target images; skipping.")
        else:
            pairs = build_pairs(targets, identities, gen["megafs"]["target_images"])
            megafs = MegaFSGenerator(ext(cfg) / "megafs", work(cfg) / "gen_megafs",
                                     weights_dir=ext(cfg) / "megafs" / "weights")
            produced = megafs.generate(pairs)
            new_entries += _align_outputs(cfg, aligner, produced, "MegaFS", "MegaFS", "AEGAN")

    # Append to aligned index.
    index = read_json(work(cfg) / "aligned_index.json") or []
    index.extend(new_entries)
    write_json(index, work(cfg) / "aligned_index.json")

    # Inline selection + merge hard UIDs.
    hard_new = _select_inline(cfg, new_entries)
    hard = read_json(work(cfg) / "hard_uids.json") or {"kept_uids": [], "n_kept": 0, "n_total": 0}
    kept = set(hard.get("kept_uids", [])) | set(hard_new)
    hard["kept_uids"] = sorted(kept)
    hard["n_kept"] = len(kept)
    write_json(hard, work(cfg) / "hard_uids.json")

    print(f"[done] generated {len(new_entries)} fake faces "
          f"({len(hard_new)} hard after selection); index now has {len(index)} faces.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
