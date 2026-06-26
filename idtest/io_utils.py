"""I/O helpers: deterministic hashing, manifest read/write, and frame I/O."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

# Canonical manifest columns (the unit of the ID test set is an aligned face image).
MANIFEST_COLUMNS = [
    "uid",            # stable hash identifying this sample
    "split",          # "real" | "fake"
    "source_dataset", # e.g. "Celeb-DF", "FF++/DF", "FSGAN", "MegaFS"
    "method",         # specific manipulation method, or "pristine"
    "manip_type",     # AEGAN | Graphic | Unknown
    "video_id",       # originating video id (or "" for image-only sources)
    "hard",           # 1 if it survived Phase-1 + Phase-2 selection (fakes only)
    "path",           # path to the aligned face image (relative to project root)
]


def hash_id(*parts: Any) -> str:
    """Deterministic 16-char id from arbitrary parts (paths, strings, ints)."""
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()[:16]


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def video_frame_count(video_path: str | Path) -> Optional[int]:
    """Return frame count via ffprobe, or None if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-count_packets", "-show_entries", "stream=nb_read_packets",
                "-of", "csv=p=0", str(video_path),
            ],
            stderr=subprocess.DEVNULL,
        )
        return int(out.decode().strip().splitlines()[0])
    except Exception:
        return None


def read_image(path: str | Path) -> np.ndarray:
    """Read an image as a BGR uint8 ndarray (OpenCV convention)."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def write_image(path: str | Path, img: np.ndarray, jpeg_quality: Optional[int] = None) -> None:
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = []
    if jpeg_quality is not None and path.suffix.lower() in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    ok = cv2.imwrite(str(path), img, params)
    if not ok:
        raise IOError(f"Could not write image: {path}")


def write_manifest(rows: Iterable[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in MANIFEST_COLUMNS})


def read_manifest(path: str | Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_stats(stats: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
