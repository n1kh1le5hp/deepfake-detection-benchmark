#!/usr/bin/env python
"""10_train_fwa.py — train FWA (Face Warping Artifacts) from a small real pool.

Builds a real-face training pool from the original extraction
(``data/work/aligned_index.json``), **excluding the ID-test uids** (no leakage),
generates FWA-style synthetic fakes on the fly, fine-tunes an Xception backbone,
and saves a checkpoint that ``08_evaluate.py`` can score on the ID test set.

NOTE (deviation): the Cadene ImageNet-Xception mirror is down, so the backbone is
initialized from ``xception_best.pth`` (the FF++-trained detector, same arch) and
fine-tuned on FWA synthetic fakes. This is a small-data proof-of-concept — expect
weak generalization to the hard ID test set.

Usage::
    python scripts/10_train_fwa.py                         # full (all-source real pool)
    python scripts/10_train_fwa.py --source FF++           # FF++ reals only
    python scripts/10_train_fwa.py --epochs 1 --limit 64   # smoke (<1 min)
"""
from __future__ import annotations

import argparse
import importlib
import sys

import numpy as np
import torch

from _common import ext, load, out, read_json, work

from idtest.io_utils import read_manifest
from idtest.training import train_model


def _build_real_pool(index, test_uids, source=None, limit=None, seed=0):
    """Aligned REAL faces not in the test set (optionally one source, optionally capped)."""
    pool = [
        e["path"] for e in index
        if e.get("split") == "real" and e.get("uid") not in test_uids
        and (source is None or e.get("source_dataset") == source)
    ]
    rng = np.random.default_rng(seed)
    rng.shuffle(pool)
    if limit:
        pool = pool[:limit]
    return pool


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    ap.add_argument("--batch", type=int, default=None, help="override training.batch_size")
    ap.add_argument("--lr", type=float, default=None, help="override training.lr")
    ap.add_argument("--source", type=str, default=None, help="restrict real pool to one source dataset")
    ap.add_argument("--limit", type=int, default=None, help="cap real-pool size (smoke / subsample)")
    args = ap.parse_args(argv)

    cfg = load()
    tr_cfg = cfg.get("training", {})
    epochs = args.epochs or int(tr_cfg.get("epochs", 15))
    batch_size = args.batch or int(tr_cfg.get("batch_size", 32))
    lr = args.lr or float(tr_cfg.get("lr", 1e-4))
    size = int(cfg.get("face", {}).get("size", 256))
    seed = int(cfg.get("seed", 2022))
    device = cfg.get("device", "cuda")

    # --- real pool: aligned reals MINUS test-set uids (no leakage) ---
    index = read_json(work(cfg) / "aligned_index.json") or []
    manifest = read_manifest(out(cfg) / "manifest.csv")
    test_uids = {r["uid"] for r in manifest}
    pool = _build_real_pool(index, test_uids, source=args.source, limit=args.limit, seed=seed)
    if len(pool) < 10:
        print(f"[10] only {len(pool)} real faces available — too few to train. "
              f"Loosen --source/--limit or check the extraction.")
        return 1
    n_val = max(1, len(pool) // 10)
    val_paths, train_paths = pool[:n_val], pool[n_val:]
    src_lbl = args.source or "all sources"
    print(f"[10] real pool: {len(pool)} ({src_label(args.source)}), excluding {len(test_uids)} test uids")
    print(f"[10] train={len(train_paths)} reals  val={len(val_paths)} reals  "
          f"(2x each as real+synthetic-fake)")

    # --- build Xception backbone, init from xception_best.pth ---
    vendored = ext(cfg) / "deepfakebench_models"
    if str(vendored) not in sys.path:
        sys.path.insert(0, str(vendored))
    xception = importlib.import_module("xception")
    model = xception.Xception({"num_classes": 2, "mode": "original", "inc": 3, "dropout": 0})
    init_w = ext(cfg) / "weights" / "xception" / "xception_best.pth"
    state = torch.load(init_w, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {(k[len("backbone."):] if k.startswith("backbone.") else k): v for k, v in state.items()}
    m, u = model.load_state_dict(state, strict=False)
    print(f"[10] init from {init_w.name}: missing={len(m)} unexpected={len(u)}")
    model = model.to(device)

    # --- train ---
    save_dir = ext(cfg) / "weights" / "fwa_trained"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "fwa_trained_best.pth"
    best = train_model(
        model, train_paths, val_paths,
        epochs=epochs, lr=lr, batch_size=batch_size,
        device=device, save_path=str(save_path), size=size, seed=seed,
    )
    print(f"[10] done. best val_auc={best:.4f}. checkpoint -> {save_path}")
    print(f"[10] evaluate with: python scripts/08_evaluate.py --detectors fwa_trained")
    return 0


def src_label(source):
    return f"source={source}" if source else "all sources"


if __name__ == "__main__":
    sys.exit(main())
