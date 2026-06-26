"""Unified dataset abstraction.

Every source dataset (UADFV, DF-TIMIT, Celeb-DF, DF-1.0, FF++, DFDC,
ForgeryNet) is exposed as an iterable of :class:`Sample` records sharing the
same schema, regardless of its on-disk layout. Downstream stages never need to
know which dataset a sample came from beyond the metadata carried by Sample.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, List, Optional

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MEDIA_EXTS = VIDEO_EXTS | IMAGE_EXTS

# Substrings that mark a directory as holding real / fake data when a dataset
# has no machine-readable metadata. Kept conservative to avoid mislabeling.
_REAL_HINTS = ("real", "original", "pristine", "source", "youtube", "actor", " Celeb-real".strip())
_FAKE_HINTS = ("fake", "synthesis", "synth", "manipulat", "deepfake", "swap", "forged", "altered")


@dataclass(frozen=True)
class Sample:
    """A single unit of source data.

    *path* may point at a video file (is_video=True), a single image
    (is_video=False, has a video_id only if it belongs to a sequence), or a
    directory of extracted frames for a sequence (is_video=False, path is a dir).
    """

    path: str            # absolute path to video / image / frames-directory
    split: str           # "real" | "fake"
    source_dataset: str  # e.g. "Celeb-DF", "FF++"
    method: str          # specific method, or "pristine"
    manip_type: str      # AEGAN | Graphic | Unknown
    video_id: str        # originating video/sequence id ("" if none)
    is_video: bool       # True -> path is a video file

    def as_manifest_row(self) -> dict:
        d = asdict(self)
        # Manifest uses "hard" for fakes that survive selection; default unset.
        return d


class SourceDataset:
    """Base class. Subclasses implement :meth:`_iter_samples`."""

    name: str = "base"

    def __init__(self, root: str | Path, manip_type: str = "Unknown", enabled: bool = True, **opts):
        self.root = Path(root)
        self.manip_type = manip_type
        self.enabled = enabled
        self.opts = opts
        if self.enabled and not self.root.exists():
            raise FileNotFoundError(
                f"[{self.name}] expected root not found: {self.root}. "
                f"Set its 'root' in config/datasets.yaml or disable it."
            )

    def __iter__(self) -> Iterator[Sample]:
        if not self.enabled:
            return
        for s in self._iter_samples():
            yield s

    def _iter_samples(self) -> Iterator[Sample]:
        raise NotImplementedError

    # -- helpers shared by subclasses --------------------------------------

    def _scan_videos(self) -> List[Path]:
        """All video files under root (sorted)."""
        return sorted(
            p for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        )

    def _scan_media(self) -> List[Path]:
        """All video OR image files under root (sorted). Use this when a dataset
        may be distributed as images instead of (or alongside) videos."""
        return sorted(
            p for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in MEDIA_EXTS
        )

    @staticmethod
    def is_video_file(path: str | Path) -> bool:
        return Path(path).suffix.lower() in VIDEO_EXTS

    @staticmethod
    def _guess_split(path: Path, real_default: bool = True) -> str:
        """Infer real/fake from path components when no metadata exists."""
        loc = " ".join(part.lower() for part in path.parts).replace("_", " ").replace("-", " ")
        if any(h in loc for h in _FAKE_HINTS):
            return "fake"
        if any(h in loc for h in _REAL_HINTS):
            return "real"
        return "real" if real_default else "fake"

    @staticmethod
    def _vid(path: Path) -> str:
        return path.stem
