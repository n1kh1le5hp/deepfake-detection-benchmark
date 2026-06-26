#!/usr/bin/env python
"""07_apply_perturbations.py — apply the 5 perturbations to the assembled set.

For every image in data/out/manifest.csv, sample randomized intensities from the
configured rational ranges and write one perturbed copy per type to
data/out/perturbed/<uid>/<type>.png.

Usage: python scripts/07_apply_perturbations.py
"""
from __future__ import annotations

import sys

import numpy as np
from tqdm import tqdm

from _common import load, out

from idtest.io_utils import read_manifest, write_image
from idtest.perturb.transforms import PERTURBATIONS, apply_perturbation, sample_param


def main() -> int:
    cfg = load()
    if not cfg["perturbations"]["enabled"]:
        print("[07] perturbations disabled; skipping.")
        return 0

    manifest = read_manifest(out(cfg) / "manifest.csv")
    if not manifest:
        print("[07] manifest is empty; run 06 first.")
        return 1

    ranges = cfg["perturbations"]["types"]
    seed = cfg.get("seed", 2022)
    pert_dir = out(cfg) / "perturbed"

    from idtest.io_utils import read_image

    for row in tqdm(manifest, desc="[07] perturbing"):
        uid = row["uid"]
        # Seed per-image so intensities are deterministic & reproducible.
        rng = np.random.default_rng(abs(hash((seed, uid))) % (2 ** 32))
        try:
            img = read_image(row["path"])
        except Exception:
            continue
        out_root = pert_dir / uid
        out_root.mkdir(parents=True, exist_ok=True)
        for name in PERTURBATIONS:
            lo, hi = ranges[name]["low"], ranges[name]["high"]
            value = sample_param(name, lo, hi, rng)
            perturbed = apply_perturbation(name, img, value)
            write_image(out_root / f"{name}.png", perturbed)

    print(f"[done] wrote {len(manifest) * len(PERTURBATIONS)} perturbed images "
          f"under {pert_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
