#!/usr/bin/env python
"""09_benchmark_datasets.py — per-source-dataset AUC on the ORIGINAL extraction.

Unlike 08_evaluate.py (which scores the curated/hard ID test set), this scores
each detector over the **full pre-selection extraction** (``data/work/aligned_index.json``,
~96.5k aligned faces), **clean-only**, and reports AUC / Accuracy / EER **per source
dataset** (Celeb-DF, FF++, ForgeryNet, DF-TIMIT, UADFV) — the standard per-dataset
benchmark format (detectors are DeepfakeBench FF++-trained weights).

Patch is excluded by default (45 models x 96k images is hours); add ``--detectors``
to override.

Usage::
    python scripts/09_benchmark_datasets.py                 # full
    python scripts/09_benchmark_datasets.py --limit 500      # smoke
    python scripts/09_benchmark_datasets.py --metrics-only   # recompute report from cached CSVs
    python scripts/09_benchmark_datasets.py --force          # re-score cached detectors
    python scripts/09_benchmark_datasets.py --detectors xception,ffd
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from _common import ext, load, out, read_json, work

from idtest.evaluate.harness import (
    ALL_CONDITIONS,
    SCORE_COLUMNS,
    build_sample_table,
    score_detector,
)
from idtest.evaluate.metrics import summarize
from idtest.selection.detectors import build_detector


def _build_report(results: pd.DataFrame, detectors: list[str], sources: list[str]) -> str:
    import numpy as np

    def _fmt(v):
        return "  -  " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"

    lines = [
        "# Per-Source-Dataset Benchmark (original extraction)",
        "",
        "Detectors are DeepfakeBench **FF++ c23-trained** weights, tested on each source dataset's "
        "real vs fake faces from the full extraction (clean only). This is the standard per-dataset "
        "format; compare against the in-domain AUCs in the paper's Table 4.",
        "",
        "## AUC by source dataset",
        "",
    ]
    auc_pivot = results.pivot(index="source", columns="detector", values="auc")
    header = "| Source dataset | " + " | ".join(detectors) + " |"
    sep = "|----------------|" + "|".join(["------"] * len(detectors)) + "|"
    lines += [header, sep]
    for s in sources:
        if s not in auc_pivot.index:
            continue
        row = auc_pivot.loc[s]
        cells = " | ".join(_fmt(row.get(d)) for d in detectors)
        # n for this source (real+fake), from any detector's row
        n = int(results[results.source == s]["n"].max()) if (results.source == s).any() else 0
        lines.append(f"| {s} ({n}) | {cells} |")
    lines += [
        "",
        "## Accuracy@0.5 by source dataset",
        "",
    ]
    acc_pivot = results.pivot(index="source", columns="detector", values="accuracy")
    lines += [header, sep]
    for s in sources:
        if s not in acc_pivot.index:
            continue
        row = acc_pivot.loc[s]
        cells = " | ".join(_fmt(row.get(d)) for d in detectors)
        lines.append(f"| {s} | {cells} |")
    lines += [
        "",
        "---",
        "",
        "*Convention: detector outputs P(real); label 1 = fake; AUC on fake-score = 1 − P(real).* "
        "*ForgeryNet's public test set is almost all REAL (fakes held back), so its per-source AUC is "
        "noisy/near-random — not a detector failure.*",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N manifest images (smoke test)")
    ap.add_argument("--batch", type=int, default=None, help="override evaluation.batch_size")
    ap.add_argument("--detectors", type=str, default=None,
                    help="comma-separated detector names (default: the 5 non-patch eval detectors)")
    ap.add_argument("--metrics-only", action="store_true",
                    help="recompute results.csv + report.md from cached score CSVs (no inference)")
    ap.add_argument("--force", action="store_true", help="re-score detectors whose CSV already exists")
    args = ap.parse_args(argv)

    cfg = load()
    eval_cfg = cfg.get("evaluation", {})
    if args.detectors:
        detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]
    else:
        detectors = [d for d in eval_cfg.get("detectors", []) if d != "patch"]
        if not detectors:
            detectors = ["xception", "meso4", "meso4inception", "ffd", "facexray"]
    batch_size = args.batch or int(eval_cfg.get("batch_size", 32))

    bdir = out(cfg) / "eval" / "per_dataset"
    bdir.mkdir(parents=True, exist_ok=True)

    if not args.metrics_only:
        index = read_json(work(cfg) / "aligned_index.json") or []
        if not index:
            print("[09] no aligned_index.json found; run the extraction pipeline first.")
            return 1
        samples, n_skipped = build_sample_table(index, out(cfg), conditions=["clean"], limit=args.limit)
        print(f"[09] {len(index)} extraction entries -> {len(samples)} clean samples "
              f"({len(samples)} imgs, 1 condition)")
        if n_skipped:
            print(f"[09] note: {n_skipped} clean paths missing on disk (skipped)")
        device = cfg.get("device", "cuda")
        for name in detectors:
            score_csv = bdir / f"scores_{name}.csv"
            if score_csv.exists() and not args.force:
                print(f"[09] {name}: cached at {score_csv.name} (skip; --force to overwrite)")
                continue
            print(f"[09] {name}: building detector on '{device}' ...")
            det = build_detector(name, {"device": device}, ext(cfg))
            rows = score_detector(det, samples, batch_size)
            del det
            if _torch_available():
                import torch
                torch.cuda.empty_cache()
            pd.DataFrame(rows, columns=SCORE_COLUMNS).to_csv(score_csv, index=False)
            print(f"[09] {name}: wrote {len(rows)} rows -> {score_csv}")

    frames = []
    for name in detectors:
        score_csv = bdir / f"scores_{name}.csv"
        if not score_csv.exists():
            print(f"[09] WARN: no scores for '{name}' at {score_csv}; skipping")
            continue
        df = pd.read_csv(score_csv)
        df["detector"] = name
        frames.append(df)
    if not frames:
        print("[09] no score files; run without --metrics-only first.")
        return 1

    all_scores = pd.concat(frames, ignore_index=True)
    results = summarize(all_scores, ["detector", "source"])
    results.to_csv(bdir / "results.csv", index=False)

    sources = sorted(all_scores.source.dropna().unique())
    report = _build_report(results, detectors, sources)
    (bdir / "report.md").write_text(report)
    print(f"[09] wrote results.csv + report.md -> {bdir}")
    return 0


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
