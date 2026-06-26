"""Build the final manifest + a Table-2-style distribution report.

The manifest is the single source of truth for the assembled ID test set: one
row per aligned face image. ``stats.json`` mirrors the structure of paper
Table 2 (image- and video-level manipulation-type distribution) plus method
coverage, so the result can be sanity-checked against the paper.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence

from ..datasets.base import Sample
from ..io_utils import hash_id, write_manifest, write_stats


def to_row(sample: Sample, face_path: str, hard: bool = False) -> dict:
    """Convert a Sample + its aligned-face path into a manifest row."""
    return {
        "uid": hash_id(sample.path, face_path),
        "split": sample.split,
        "source_dataset": sample.source_dataset,
        "method": sample.method,
        "manip_type": sample.manip_type,
        "video_id": sample.video_id,
        "hard": int(bool(hard)),
        "path": face_path,
    }


def build_manifest(rows: Iterable[dict], manifest_path: str) -> None:
    write_manifest(rows, manifest_path)


def compute_stats(rows: Sequence[dict], targets: Dict[str, int], min_methods: int) -> dict:
    """Aggregate manifest rows into a Table-2-style stats dict."""
    rows = list(rows)
    n = len(rows)

    split_counts = Counter(r["split"] for r in rows)
    method_counts = Counter(r["method"] for r in rows if r["split"] == "fake")
    method_coverage = len(method_counts)

    by_source = Counter(r["source_dataset"] for r in rows)
    by_manip = Counter(r["manip_type"] for r in rows)

    # Image-level manipulation-type distribution (one per row).
    img_by_manip = Counter(r["manip_type"] for r in rows if r["split"] == "fake")
    # Video-level: distinct video_ids per manipulation type (fakes only).
    vids_by_manip: Dict[str, set] = defaultdict(set)
    for r in rows:
        if r["split"] == "fake" and r.get("video_id"):
            vids_by_manip[r["manip_type"]].add(r["video_id"])
    vid_by_manip = {k: len(v) for k, v in vids_by_manip.items()}

    return {
        "total_images": n,
        "counts_by_split": dict(split_counts),
        "targets": targets,
        "method_coverage": method_coverage,
        "min_methods_target": min_methods,
        "methods": dict(method_counts),
        "by_source": dict(by_source),
        "by_manip_type": dict(by_manip),
        "table2_style": {
            "image": dict(img_by_manip),   # e.g. {"AEGAN": N, "Graphic": N, "Unknown": N}
            "video": vid_by_manip,
        },
    }


def write_report(rows: Sequence[dict], manifest_path: str, stats_path: str,
                 targets: Dict[str, int], min_methods: int) -> dict:
    """Write both manifest.csv and stats.json; return the stats dict."""
    build_manifest(rows, manifest_path)
    stats = compute_stats(rows, targets, min_methods)
    write_stats(stats, stats_path)
    return stats
