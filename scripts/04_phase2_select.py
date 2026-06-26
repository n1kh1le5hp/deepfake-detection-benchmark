#!/usr/bin/env python
"""04_phase2_select.py — Phase-2 user-perception selection.

Two modes (config phase2.mode):
  * 'proxy'  : a model-proxy scores perceived realism; accept fakes in the top
               (1 - keep_quantile) fraction. APPROXIMATION of the human study.
  * 'manual' : apply the real 15/30 rule to vote tallies in
               data/work/votes.json (produced by annotation_ui/).

Combines with Phase-1 to produce the final hard-UID set at
data/work/hard_uids.json.

Usage: python scripts/04_phase2_select.py [--keep-quantile 0.5] [--batch 32]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from tqdm import tqdm

from _common import ext, load, read_json, work, write_json

from idtest.io_utils import read_image
from idtest.selection.appraise import combine
from idtest.selection.detectors import build_detector
from idtest.selection.phase2_human import ProxyRealismScorer, human_study_mask


def _load_phase1(cfg):
    p1 = np.load(work(cfg) / "phase1_mask.npy", allow_pickle=True)
    return p1.astype(bool)


def _proxy_scores(cfg, fakes, batch) -> np.ndarray:
    name = cfg["phase2"]["proxy_model"]
    # Reuse Phase-1 scores when the proxy model was already scored there (e.g.
    # both use Xception) — avoids a redundant ~8-min GPU re-scoring pass.
    if name in cfg["phase1"].get("detectors", []):
        sf = work(cfg) / "fake_scores.npz"
        if sf.exists():
            d = np.load(sf, allow_pickle=True)
            det_idx = cfg["phase1"]["detectors"].index(name)
            print(f"[04] reusing Phase-1 '{name}' scores (proxy == phase1 detector)")
            return d["probs"][:, det_idx].astype(np.float32)
    det = build_detector(name or "default", cfg, ext(cfg))
    scorer = ProxyRealismScorer(det)
    scores = np.zeros(len(fakes), dtype=np.float32)
    for i in tqdm(range(0, len(fakes), batch), desc="[04] proxy realism"):
        chunk = fakes[i:i + batch]
        imgs = []
        for e in chunk:
            try:
                imgs.append(read_image(e["path"]))
            except Exception:
                imgs.append(np.zeros((cfg["face"]["size"], cfg["face"]["size"], 3), np.uint8))
        scores[i:i + len(chunk)] = scorer.score(imgs)
    return scores


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-quantile", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args(argv)

    cfg = load()
    index = read_json(work(cfg) / "aligned_index.json") or []
    fakes = [e for e in index if e["split"] == "fake"]
    if not fakes:
        print("[04] no fakes; nothing to do.")
        return 0

    p1_mask = _load_phase1(cfg)

    mode = cfg["phase2"]["mode"]
    if mode == "proxy":
        scores = _proxy_scores(cfg, fakes, args.batch)
        scorer = ProxyRealismScorer.__new__(ProxyRealismScorer)
        p2_mask = scorer.accept_mask(scores, keep_quantile=args.keep_quantile)
    elif mode == "manual":
        votes = read_json(work(cfg) / "votes.json") or {}
        counts = np.array([int(votes.get(e["uid"], 0)) for e in fakes], dtype=int)
        p2_mask = human_study_mask(
            counts,
            threshold=cfg["phase2"]["human_threshold"],
            num_participants=cfg["phase2"]["num_participants"],
        )
    else:
        raise ValueError(f"Unknown phase2.mode '{mode}'")

    hard = combine(p1_mask, p2_mask)
    kept = [e["uid"] for e, m in zip(fakes, hard) if m]
    write_json({"kept_uids": kept, "n_kept": int(hard.sum()),
                "n_total": int(len(hard))}, work(cfg) / "hard_uids.json")

    print(f"[done] Phase-2 ({mode}) -> {int(hard.sum())}/{len(fakes)} fakes are hard "
          f"(phase1 kept {int(p1_mask.sum())}, phase2 kept {int(p2_mask.sum())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
