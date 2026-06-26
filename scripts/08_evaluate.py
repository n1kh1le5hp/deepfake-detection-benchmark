#!/usr/bin/env python
"""08_evaluate.py — evaluate detectors over the ID test set (clean + perturbations).

For each detector in ``config.evaluation.detectors``: score every clean image
and every perturbation copy, cache the per-sample ``P(real)`` to a CSV, then
compute AUC / Accuracy / EER and write ``results.csv`` + ``report.md``.

Usage::

    python scripts/08_evaluate.py                 # full run (clean + 5 perturbations)
    python scripts/08_evaluate.py --limit 500     # smoke test (seconds; isolated under eval/smoke/)
    python scripts/08_evaluate.py --metrics-only  # recompute report from cached score CSVs
    python scripts/08_evaluate.py --force         # re-score detectors whose CSV already exists
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from _common import ext, load, out, read_json

from idtest.io_utils import read_manifest
from idtest.selection.detectors import build_detector
from idtest.evaluate.harness import (
    ALL_CONDITIONS,
    SCORE_COLUMNS,
    build_sample_table,
    score_detector,
)
from idtest.evaluate.metrics import summarize


def _build_report(
    all_scores: pd.DataFrame,
    results: pd.DataFrame,
    detectors: list[str],
    conditions: list[str],
    per_method: bool,
    stats: dict | None,
) -> str:
    """Render the markdown report (3 tables + footer)."""
    import numpy as np

    pert_types = [c for c in conditions if c != "clean"]

    def _fmt(v):
        return "  -  " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"

    lines: list[str] = ["# ID Test Set — Detector Evaluation", ""]

    if stats:
        lines += [
            f"**Set:** {stats.get('total_images', '?')} images "
            f"({stats.get('counts_by_split', {}).get('fake', '?')} fake / "
            f"{stats.get('counts_by_split', {}).get('real', '?')} real), "
            f"{stats.get('method_coverage', '?')} manipulation methods.",
            f"**Detectors:** {', '.join(detectors)}.",
            f"**Conditions:** {', '.join(conditions)}.",
            "",
        ]

    # Table 1 — clean performance
    lines += ["## Table 1 — Clean performance", ""]
    clean = results[results.condition == "clean"].set_index("detector")
    lines.append("| Detector | AUC | Accuracy@0.5 | EER |")
    lines.append("|----------|-----|--------------|-----|")
    for d in detectors:
        if d in clean.index:
            r = clean.loc[d]
            lines.append(f"| {d} | {_fmt(r['auc'])} | {_fmt(r['accuracy'])} | {_fmt(r['eer'])} |")
    lines.append("")

    # Table 2 — robustness across perturbations (AUC)
    lines += ["## Table 2 — Robustness (AUC by condition)", ""]
    auc_pivot = results.pivot(index="detector", columns="condition", values="auc")
    header = "| Detector | " + " | ".join(conditions) + " | mean-pert | drop vs clean |"
    sep = "|----------|" + "|".join(["------"] * len(conditions)) + "|----------|---------------|"
    lines += [header, sep]
    for d in detectors:
        if d not in auc_pivot.index:
            continue
        row = auc_pivot.loc[d]
        clean_auc = row.get("clean")
        pert_vals = [row.get(c) for c in pert_types]
        pert_valid = [v for v in pert_vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        mean_pert = float(np.mean(pert_valid)) if pert_valid else float("nan")
        drop = (clean_auc - mean_pert) if (clean_auc == clean_auc and mean_pert == mean_pert) else float("nan")
        cells = " | ".join(_fmt(row.get(c)) for c in conditions)
        lines.append(f"| {d} | {cells} | {_fmt(mean_pert)} | {_fmt(drop)} |")
    lines.append("")

    # Table 3 — per-method AUC (clean): fakes of method vs all reals
    if per_method:
        lines += ["## Table 3 — Per-method AUC (clean; method fakes vs all reals)", ""]
        clean_df = all_scores[all_scores.condition == "clean"]
        reals = clean_df[clean_df.label == 0]
        # all_scores stacks one row per (image, detector); counts must come from a
        # single detector's view so they aren't inflated by the detector axis.
        clean_one = clean_df[clean_df.detector == detectors[0]]
        methods = sorted(
            m for m in clean_one.method.dropna().unique()
            if m and m.lower() not in ("pristine", "nan")
        )
        if methods:
            header = "| Method (n fake) | " + " | ".join(detectors) + " |"
            sep = "|----------------|" + "|".join(["------"] * len(detectors)) + "|"
            lines += [header, sep]
            for m in methods:
                fakes_m = clean_df[(clean_df.method == m) & (clean_df.label == 1)]
                n_fake = int((clean_one.method == m).sum())  # de-duplicated count
                if n_fake == 0:
                    continue
                sub = pd.concat([reals, fakes_m])[["detector", "label", "p_real"]]
                summ = summarize(sub, ["detector"]).set_index("detector")
                cells = " | ".join(_fmt(summ.loc[d, "auc"]) if d in summ.index else "  -  " for d in detectors)
                lines.append(f"| {m} ({n_fake}) | {cells} |")
            lines.append("")

    # Polarity sanity note
    low = []
    if "clean" in set(results.condition):
        for d in detectors:
            row = results[(results.detector == d) & (results.condition == "clean")]
            if not row.empty:
                a = row["auc"].iloc[0]
                if isinstance(a, float) and a == a and a < 0.5:
                    low.append((d, a))
    if low:
        lines += [
            "## ⚠ Polarity warning", "",
            "The following detectors scored **clean AUC < 0.5**, which usually means the "
            "real/fake class index is inverted. Check the `real_index` setting for these:",
            "",
            ", ".join(f"{d} ({a:.3f})" for d, a in low),
            "",
        ]

    # Footer caveats
    lines += [
        "---",
        "",
        "*Convention: detectors output P(real); label 1 = fake; AUC computed on "
        "fake-score = 1 − P(real). Accuracy uses the 0.5 threshold.*",
    ]
    if stats:
        cov = stats.get("method_coverage")
        tgt = stats.get("min_methods_target")
        if cov is not None and tgt is not None and cov < tgt:
            lines.append(
                f"*Caveat: this set currently covers {cov} manipulation methods "
                f"(paper target ≥ {tgt}); ForgeryNet-Graphic fakes = 0. "
                f"Treat as a checkpoint evaluation of the set as it stands today.*"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N manifest images (smoke test; isolated under eval/smoke/)")
    ap.add_argument("--batch", type=int, default=None, help="override evaluation.batch_size")
    ap.add_argument("--metrics-only", action="store_true",
                    help="recompute results.csv + report.md from cached score CSVs (no inference)")
    ap.add_argument("--force", action="store_true", help="re-score detectors whose CSV already exists")
    args = ap.parse_args(argv)

    cfg = load()
    eval_cfg = cfg.get("evaluation", {})
    detectors = list(eval_cfg.get("detectors", ["xception"]))
    conditions = list(eval_cfg.get("conditions", ALL_CONDITIONS))
    batch_size = args.batch or int(eval_cfg.get("batch_size", 32))
    per_method = bool(eval_cfg.get("per_method", True))

    base_eval = out(cfg) / eval_cfg.get("out_subdir", "eval")
    eval_dir = base_eval / ("smoke" if args.limit else "")
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ---- scoring phase ----
    if not args.metrics_only:
        manifest = read_manifest(out(cfg) / "manifest.csv")
        samples, n_skipped = build_sample_table(
            manifest, out(cfg), conditions, limit=args.limit,
        )
        if args.limit:
            print(f"[08] --limit {args.limit} manifest images -> {len(samples)} sample rows "
                  f"({len(samples) // max(1, len(conditions))} imgs x {len(conditions)} conds)")
        if n_skipped:
            print(f"[08] note: {n_skipped} requested perturbation files missing on disk (skipped)")
        device = cfg.get("device", "cuda")
        for name in detectors:
            score_csv = eval_dir / f"scores_{name}.csv"
            if score_csv.exists() and not args.force:
                print(f"[08] {name}: cached at {score_csv.name} (skip; use --force to overwrite)")
                continue
            print(f"[08] {name}: building detector on '{device}' ...")
            det = build_detector(name, {"device": device}, ext(cfg))
            rows = score_detector(det, samples, batch_size)
            del det  # free GPU memory before building the next detector
            if torch_available():
                import torch
                torch.cuda.empty_cache()
            pd.DataFrame(rows, columns=SCORE_COLUMNS).to_csv(score_csv, index=False)
            print(f"[08] {name}: wrote {len(rows)} rows -> {score_csv}")

    # ---- metrics phase (from cached score CSVs) ----
    frames = []
    for name in detectors:
        score_csv = eval_dir / f"scores_{name}.csv"
        if not score_csv.exists():
            print(f"[08] WARN: no scores for '{name}' at {score_csv}; skipping")
            continue
        df = pd.read_csv(score_csv)
        df["detector"] = name
        frames.append(df)
    if not frames:
        print("[08] no score files found; run without --metrics-only first.")
        return 1

    all_scores = pd.concat(frames, ignore_index=True)
    results = summarize(all_scores, ["detector", "condition"])
    results.to_csv(eval_dir / "results.csv", index=False)

    stats = read_json(out(cfg) / "stats.json")
    report = _build_report(all_scores, results, detectors, conditions, per_method, stats)
    (eval_dir / "report.md").write_text(report)
    print(f"[08] wrote results.csv + report.md -> {eval_dir}")

    # quick stdout summary
    clean = results[results.condition == "clean"]
    if not clean.empty:
        print("\nClean AUC by detector:")
        for _, r in clean.iterrows():
            print(f"  {r.detector:14s} AUC={r.auc:.4f}  acc={r.accuracy:.4f}  EER={r.eer:.4f}")
    return 0


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
