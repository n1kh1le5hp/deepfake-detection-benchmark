"""Scoring harness: turn the assembled test set into detector scores.

Two responsibilities, kept separate from the metrics (so metrics can be
recomputed without re-running inference):

1. :func:`build_sample_table` — expand the clean-image manifest into a tidy table
   of ``(uid, condition, path, label, method, source, hard)`` rows covering the
   clean image plus each perturbation type. Perturbed paths are derived from the
   clean UID by the on-disk convention ``out/perturbed/<uid>/<type>.png``.
2. :func:`score_detector` — stream a detector over a sample table in batches,
   returning per-sample ``P(real)`` (rows that fail to read are dropped, not
   silently zeroed).

The orchestrator (:mod:`scripts.08_evaluate`) wires these together, caches
per-detector scores to CSV, and feeds the cached scores to the metrics module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from idtest.io_utils import read_image

# All conditions the evaluation can cover. "clean" is the unperturbed image;
# the other five match the perturbation filenames written by stage 07.
ALL_CONDITIONS: List[str] = [
    "clean",
    "color_contrast",
    "color_saturation",
    "gaussian_blur",
    "jpeg_compression",
    "white_gaussian_noise",
]
PERTURBED_TYPES: List[str] = ALL_CONDITIONS[1:]

SCORE_COLUMNS: List[str] = [
    "uid", "condition", "p_real", "label", "method", "source", "hard",
]


def build_sample_table(
    manifest: Sequence[Dict],
    out_root,
    conditions: Sequence[str] = ALL_CONDITIONS,
    limit: int | None = None,
) -> Tuple[List[Dict[str, object]], int]:
    """Expand the manifest into per-condition sample rows.

    Each manifest row yields one row per requested condition (clean + each
    perturbation present on disk). Returns ``(rows, n_skipped)`` where
    ``n_skipped`` counts missing perturbation files (logged by the caller).

    ``limit`` caps the number of *manifest images* processed (not output rows),
    so ``--limit 500`` scores 500 clean faces plus their perturbations.
    """
    out_root = Path(out_root)
    perturbed_root = out_root / "perturbed"
    rows: List[Dict[str, object]] = []
    n_skipped = 0
    if limit is not None:
        # Stratified half-real / half-fake so AUC is defined and meaningful even
        # for small smoke runs (the manifest lists fakes and reals in blocks, so a
        # naive "first N" slice would be a single class).
        reals = [r for r in manifest if r["split"] == "real"][:limit // 2]
        fakes = [r for r in manifest if r["split"] == "fake"][:limit - len(reals)]
        selected = reals + fakes
    else:
        selected = manifest
    for r in selected:
        uid = r["uid"]
        label = 1 if r["split"] == "fake" else 0
        common = {
            "uid": uid,
            "label": label,
            "method": r.get("method", ""),
            "source": r.get("source_dataset", ""),
            "hard": int(r.get("hard", 0) or 0),
        }
        for cond in conditions:
            if cond == "clean":
                path = Path(r["path"])
            else:
                path = perturbed_root / uid / f"{cond}.png"
            if not path.is_file():
                n_skipped += 1
                continue
            rows.append({**common, "condition": cond, "path": str(path)})
    return rows, n_skipped


def score_detector(
    detector,
    samples: Sequence[Dict[str, object]],
    batch_size: int = 32,
    face_size: int = 256,
) -> List[Dict[str, object]]:
    """Score *detector* over *samples*, returning rows with ``p_real`` filled in.

    Unreadable images are dropped (their count is printed). Reuses the
    black-image-on-failure batching pattern from the Phase-1 scoring script so a
    single bad file can't abort a long run.
    """
    results: List[Dict[str, object]] = []
    n_failed = 0
    for start in tqdm(range(0, len(samples), batch_size), desc=f"[{detector.name}] scoring"):
        chunk = samples[start:start + batch_size]
        imgs, valid = [], []
        for s in chunk:
            try:
                imgs.append(read_image(s["path"]))
                valid.append(True)
            except Exception:
                imgs.append(None)
                valid.append(False)
        imgs = [
            x if x is not None else np.zeros((face_size, face_size, 3), np.uint8)
            for x in imgs
        ]
        probs = detector.predict_real(imgs)
        for s, p, ok in zip(chunk, probs, valid):
            if not ok:
                n_failed += 1
                continue
            row = {k: s[k] for k in ("uid", "condition", "label", "method", "source", "hard")}
            row["p_real"] = float(p)
            results.append(row)
    if n_failed:
        print(f"[{detector.name}] dropped {n_failed} unreadable images")
    return results
