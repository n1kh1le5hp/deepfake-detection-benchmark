#!/usr/bin/env python
"""03_phase1_select.py — Phase-1 detection-model selection (false acceptance).

Loads data/work/fake_scores.npz, applies the false-acceptance rule
(P(real) >= tau under the ensemble mode), and writes the Phase-1 mask +
surviving UIDs to data/work/phase1_mask.npy and data/work/phase1_uids.json.

Usage: python scripts/03_phase1_select.py
"""
from __future__ import annotations

import sys

import numpy as np

from _common import load, work, write_json

from idtest.selection.phase1_model import false_acceptance_mask


def main() -> int:
    cfg = load()
    scores = np.load(work(cfg) / "fake_scores.npz", allow_pickle=True)
    probs, uids = scores["probs"], scores["uids"]

    p1 = cfg["phase1"]
    mask = false_acceptance_mask(probs, tau=p1["tau"], mode=p1["ensemble"])
    kept = [str(u) for u, m in zip(uids, mask) if m]

    np.save(work(cfg) / "phase1_mask.npy", mask)
    write_json({"kept_uids": kept, "n_kept": len(kept), "n_total": int(len(mask))},
               work(cfg) / "phase1_uids.json")

    print(f"[done] Phase-1 kept {len(kept)}/{len(mask)} fakes "
          f"(tau={p1['tau']}, ensemble={p1['ensemble']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
