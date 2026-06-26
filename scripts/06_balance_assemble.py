#!/usr/bin/env python
"""06_balance_assemble.py — balance hard fakes + reals and assemble the ID test set.

Reads:
  * data/work/aligned_index.json
  * data/work/hard_uids.json   (hard fake UIDs from Phase-1 + Phase-2 + generation)
Writes:
  * data/out/images/<uid>.png      (the assembled, balanced face images)
  * data/out/manifest.csv          (one row per image; the source of truth)
  * data/out/stats.json            (Table-2-style distribution + coverage)

Usage: python scripts/06_balance_assemble.py
"""
from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _common import load, out, read_json, work

from idtest.assemble.balance import select_fakes, select_reals


@dataclass
class _Entry:
    uid: str
    split: str
    source_dataset: str
    method: str
    manip_type: str
    video_id: str
    path: str


def _to_entry(d: dict) -> _Entry:
    return _Entry(d["uid"], d["split"], d["source_dataset"], d.get("method", ""),
                  d.get("manip_type", "Unknown"), d.get("video_id", ""), d["path"])


def main() -> int:
    cfg = load()
    bcfg = cfg["balance"]
    rng = np.random.default_rng(bcfg.get("seed", cfg.get("seed", 2022)))

    index = read_json(work(cfg) / "aligned_index.json") or []
    hard = read_json(work(cfg) / "hard_uids.json") or {"kept_uids": []}
    hard_uids = set(hard.get("kept_uids", []))

    reals = [_to_entry(e) for e in index if e["split"] == "real"]
    hard_fakes = [_to_entry(e) for e in index if e["split"] == "fake" and e["uid"] in hard_uids]

    if not hard_fakes:
        print("[06] WARNING: no hard fakes found. Did you run 02-05? Falling back "
              "to ALL fakes so the pipeline can still produce a manifest.")
        hard_fakes = [_to_entry(e) for e in index if e["split"] == "fake"]

    sel_fakes = select_fakes(hard_fakes, bcfg["target_fake_images"], bcfg["min_methods"], rng)
    sel_reals = select_reals(reals, bcfg["target_real_images"], rng)
    print(f"[06] selected {len(sel_fakes)} fakes + {len(sel_reals)} reals")

    # Copy assembled images + build manifest rows.
    img_dir = out(cfg) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in sel_fakes + sel_reals:
        dst = img_dir / f"{e.uid}.png"
        if not dst.exists() and Path(e.path).exists():
            shutil.copy2(e.path, dst)
        rows.append({
            "uid": e.uid, "split": e.split, "source_dataset": e.source_dataset,
            "method": e.method, "manip_type": e.manip_type, "video_id": e.video_id,
            "hard": 1 if e.split == "fake" else 0, "path": str(dst),
        })

    # Write manifest + stats.
    from idtest.assemble.manifest import write_report
    targets = {"fake_images": bcfg["target_fake_images"], "real_images": bcfg["target_real_images"]}
    stats = write_report(rows, str(out(cfg) / "manifest.csv"), str(out(cfg) / "stats.json"),
                         targets, bcfg["min_methods"])
    print(f"[done] wrote {len(rows)} rows to data/out/manifest.csv")
    print(f"[stats] methods={stats['method_coverage']} (target>={bcfg['min_methods']}), "
          f"by_split={stats['counts_by_split']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
