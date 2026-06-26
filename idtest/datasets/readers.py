"""Concrete readers for each of the 7 source datasets.

Each reader normalizes a dataset's idiosyncratic layout into :class:`Sample`
records. Layouts follow the canonical public releases; where a dataset is
distributed with metadata (DFDC, ForgeryNet), we parse it. Tolerant path-hint
heuristics cover minor layout variations. See DATASETS.md for expected roots.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator, List, Optional

from .base import IMAGE_EXTS, MEDIA_EXTS, Sample, SourceDataset, VIDEO_EXTS

# Map the config short method names -> FaceForensics++ folder names.
FFPP_METHOD_FOLDERS = {
    "DF": "deepfakes",
    "FS": "faceswap",
    "FShifter": "faceshifter",
    "F2F": "face2face",
    "NT": "neural_textures",
}


class UADFVDataset(SourceDataset):
    """UADFV. Expected: <root>/{real,fake}/*.mp4 (or similar).

    Some community mirrors also bundle derivative ``frames/`` and ``landmarks/``
    subdirs; we skip those so we don't double-count (our pipeline does its own
    extraction/alignment).
    """

    name = "UADFV"
    _DERIVATIVE_DIRS = {"frames", "landmarks", "faces", "cropped"}

    def _iter_samples(self) -> Iterator[Sample]:
        for v in self._scan_media():
            if any(part in self._DERIVATIVE_DIRS for part in v.parts):
                continue
            split = self._guess_split(v, real_default=False)
            yield Sample(
                path=str(v), split=split, source_dataset=self.name,
                method="pristine" if split == "real" else "UADFV",
                manip_type=self.manip_type, video_id=self._vid(v), is_video=self.is_video_file(v),
            )


class DFTIMITDataset(SourceDataset):
    """DF-TIMIT (DeepFakeTIMIT).

    Expected: <root>/{higher_quality,lower_quality}/{real,fake}/*.mp4 , or a flat
    <root>/{real,fake}/*.mp4. Quality level is encoded in the method name.
    """

    name = "DF-TIMIT"

    def _iter_samples(self) -> Iterator[Sample]:
        for v in self._scan_media():
            parts = {p.lower() for p in v.parts}
            # Only the canonical per-subject fakes live under higher_/lower_quality.
            # Top-level files (preview PNGs, "*-original.mov" samples) are skipped.
            if "higher_quality" not in parts and "lower_quality" not in parts:
                continue
            split = "fake"  # DeepFakeTIMIT ships only deepfakes; reals are in VidTIMIT
            quality = "HQ" if "higher_quality" in parts else "LQ"
            method = f"DF-TIMIT-{quality}"
            yield Sample(
                path=str(v), split=split, source_dataset=self.name,
                method=method, manip_type=self.manip_type,
                video_id=self._vid(v), is_video=self.is_video_file(v),
            )


class CelebDFDataset(SourceDataset):
    """Celeb-DF v2. Expected: <root>/{Celeb-real,YouTube-real,Celeb-synthesis}/*.mp4."""

    name = "Celeb-DF"

    def _iter_samples(self) -> Iterator[Sample]:
        for v in self._scan_media():
            split = "real" if any(h in v.parent.name.lower() for h in ("real", "youtube")) else "fake"
            yield Sample(
                path=str(v), split=split, source_dataset=self.name,
                method="pristine" if split == "real" else "Celeb-DF",
                manip_type=self.manip_type, video_id=self._vid(v), is_video=self.is_video_file(v),
            )


class DF10Dataset(SourceDataset):
    """DeeperForensics-1.0.

    Expected: <root>/{source_videos,manipulated_videos}/*.mp4 (the canonical
    deepfake_database layout).
    """

    name = "DF-1.0"

    def _iter_samples(self) -> Iterator[Sample]:
        for v in self._scan_media():
            split = "real" if "source" in v.parent.name.lower() else "fake"
            yield Sample(
                path=str(v), split=split, source_dataset=self.name,
                method="pristine" if split == "real" else "DF-1.0",
                manip_type=self.manip_type, video_id=self._vid(v), is_video=self.is_video_file(v),
            )


class FFPlusDataset(SourceDataset):
    """FaceForensics++.

    Expected (full release)::

        <root>/original_sequences/youtube/<quality>/sequences/<id>/...
        <root>/manipulated_sequences/<method>/<quality>/sequences/<src>_<dst>/...

    *quality* defaults to ``c23`` (configurable via opts['quality']). Only the
    methods listed in opts['methods'] (config short names: DF/FS/FShifter/F2F/NT)
    are emitted; originals are always emitted as reals.
    """

    name = "FF++"

    # Kaggle flat-mirror category folder -> (short_method, split). Real = pristine.
    _FLAT_CATS = {
        "original": (None, "real"),
        "deepfakes": ("DF", "fake"),
        "faceswap": ("FS", "fake"),
        "faceshifter": ("FShifter", "fake"),
        "face2face": ("F2F", "fake"),
        "neuraltextures": ("NT", "fake"),
        "deepfakedetection": ("DFD", "fake"),  # mixed real/fake; skipped unless in methods
    }

    def _iter_samples(self) -> Iterator[Sample]:
        flat = self._find_flat_root()
        if flat is not None:
            yield from self._iter_flat(flat)
            return
        yield from self._iter_standard()

    def _find_flat_root(self):
        """Return the ``<...>_C23`` dir if the Kaggle flat layout is present."""
        candidates = [self.root] + list(self.root.iterdir()) if self.root.is_dir() else [self.root]
        for d in candidates:
            if not d.is_dir():
                continue
            names = {p.name.lower() for p in d.iterdir() if p.is_dir()}
            if "original" in names and ("deepfakes" in names or "faceshifter" in names):
                return d
        return None

    def _iter_flat(self, c23_root) -> Iterator[Sample]:
        wanted = set(self.opts.get("methods", ["DF", "FS", "FShifter"]))
        for cat in sorted(p for p in c23_root.iterdir() if p.is_dir()):
            key = cat.name.lower()
            if key not in self._FLAT_CATS:
                continue  # skip 'csv' and unknown folders
            short, split = self._FLAT_CATS[key]
            if split == "fake" and short not in wanted:
                continue
            for v in sorted(p for p in cat.iterdir()
                            if p.is_file() and p.suffix.lower() in MEDIA_EXTS):
                yield Sample(
                    path=str(v), split=split, source_dataset=self.name,
                    method="pristine" if split == "real" else f"FF++-{short}",
                    manip_type=self.manip_type, video_id=v.stem, is_video=self.is_video_file(v),
                )

    def _iter_standard(self) -> Iterator[Sample]:
        """Official/DeepfakeBench frame-directory layout."""
        quality = self.opts.get("quality", "c23")
        wanted = set(self.opts.get("methods", ["DF", "FS", "FShifter"]))

        orig = self.root / "original_sequences" / "youtube" / quality / "sequences"
        if orig.exists():
            for seq in sorted(p for p in orig.iterdir() if p.is_dir()):
                yield Sample(
                    path=str(seq), split="real", source_dataset=self.name,
                    method="pristine", manip_type=self.manip_type,
                    video_id=seq.name, is_video=False,
                )

        manip = self.root / "manipulated_sequences"
        if manip.exists():
            for short, folder in FFPP_METHOD_FOLDERS.items():
                if short not in wanted:
                    continue
                mdir = manip / folder / quality / "sequences"
                if not mdir.exists():
                    continue
                for seq in sorted(p for p in mdir.iterdir() if p.is_dir()):
                    yield Sample(
                        path=str(seq), split="fake", source_dataset=f"{self.name}/{short}",
                        method=f"FF++-{short}", manip_type=self.manip_type,
                        video_id=seq.name, is_video=False,
                    )


class DFDCDataset(SourceDataset):
    """Deepfake Detection Challenge.

    Expected: <root>/dfdc_train_part_*/{metadata.json,videos/*.mp4}. Labels come
    from each part's ``metadata.json`` (filename -> {"label": "REAL"|"FAKE"}).
    Falls back to path hints if metadata is absent.
    """

    name = "DFDC"

    def _iter_samples(self) -> Iterator[Sample]:
        for v in self._scan_media():
            meta = self._read_part_metadata(v)
            if meta is not None:
                entry = meta.get(v.name) or meta.get(v.stem)
                split = "fake" if (entry and str(entry.get("label", "")).upper() == "FAKE") else "real"
            else:
                split = self._guess_split(v, real_default=True)
            yield Sample(
                path=str(v), split=split, source_dataset=self.name,
                method="pristine" if split == "real" else "DFDC",
                manip_type=self.manip_type, video_id=self._vid(v), is_video=self.is_video_file(v),
            )

    def _read_part_metadata(self, video: Path) -> Optional[dict]:
        for parent in video.parents:
            cand = parent / "metadata.json"
            if cand.exists():
                try:
                    return json.loads(cand.read_text())
                except Exception:
                    return None
        return None


class ForgeryNetDataset(SourceDataset):
    """ForgeryNet.

    ForgeryNet is large and image+video. We read the per-phase label CSV
    (``ForgeryNet_v1_<phase>_label.csv``) when present, yielding video samples.
    The CSV's exact columns vary by release; we look for a path-like column plus
    a category/label column. Falls back to directory-hint scanning otherwise.
    """

    name = "ForgeryNet"

    def _iter_samples(self) -> Iterator[Sample]:
        label_csv = self._find_label_csv()
        if label_csv is not None:
            yield from self._iter_from_csv(label_csv)
        else:
            for v in self._scan_media():
                cat = self._category_from_path(v)
                if cat is None:  # unknown layout -> generic heuristics
                    split = self._guess_split(v, real_default=True)
                    method = "pristine" if split == "real" else self._method_from_path(v)
                elif cat.lower() in self._REAL_CATS:
                    split, method = "real", "pristine"
                else:
                    split, method = "fake", cat
                yield Sample(
                    path=str(v), split=split, source_dataset=self.name,
                    method=method, manip_type=self.manip_type,
                    video_id=self._vid(v), is_video=self.is_video_file(v),
                )

    # ForgeryNet's public release is laid out as <root>/.../<CATEGORY>/<file>,
    # where CATEGORY is a forgery approach (DF, FS, F2F, NT, AEGAN, Graphic, ...)
    # or "public"/"real" for pristine images.
    _REAL_CATS = {"public", "real", "youtube", "pristine", "original"}
    _FAKE_CATS = {"df", "fs", "f2f", "nt", "aegan", "graphic", "attn", "style",
                  "deepfakes", "face2face", "faceswap", "faceshifter", "neural_textures"}

    def _category_from_path(self, p: Path) -> Optional[str]:
        """Return the ForgeryNet category folder component of *p*, if any."""
        for part in p.parts:
            low = part.lower()
            if low in self._REAL_CATS or low in self._FAKE_CATS:
                return part
        return None

    def _find_label_csv(self) -> Optional[Path]:
        cands = sorted(self.root.rglob("*label*.csv"))
        return cands[0] if cands else None

    def _iter_from_csv(self, csv_path: Path) -> Iterator[Sample]:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = self._guess_path_column(row)
                if not path:
                    continue
                p = (csv_path.parent / path) if not Path(path).is_absolute() else Path(path)
                # Resolve relative to dataset root if not found next to the CSV.
                if not p.exists():
                    p = self.root / path
                if not p.exists():
                    continue
                method = self._guess_method_column(row)
                split = "real" if (method.lower() in ("real", "pristine", "original", "")) else "fake"
                is_video = p.suffix.lower() in VIDEO_EXTS
                yield Sample(
                    path=str(p), split=split, source_dataset=self.name,
                    method=method or ("pristine" if split == "real" else "ForgeryNet"),
                    manip_type=self.manip_type, video_id=p.stem, is_video=is_video,
                )

    @staticmethod
    def _guess_path_column(row: dict) -> Optional[str]:
        for k, v in row.items():
            if v and any(ext in str(v).lower() for ext in VIDEO_EXTS | IMAGE_EXTS | {"/", "\\"}):
                return v
        return row.get("path") or row.get("Path")

    @staticmethod
    def _guess_method_column(row: dict) -> str:
        for key in ("category", "Category", "fake_type", "method", "label", "Label", "type"):
            if row.get(key):
                return str(row[key]).strip()
        return ""

    @staticmethod
    def _method_from_path(p: Path) -> str:
        # Use the immediate category folder name as the method if present.
        return p.parent.name or "ForgeryNet"


class CelebADataset(SourceDataset):
    """CelebA identity images — used as source identities for the FSGAN/MegaFS
    generation stage, not as a detection source. Yields real images only."""

    name = "CelebA"

    def _iter_samples(self) -> Iterator[Sample]:
        img_root = self.root / "Img" / "img_align_celeba"
        if not img_root.exists():
            img_root = self.root  # tolerate flat layout
        for img in sorted(p for p in img_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
            yield Sample(
                path=str(img), split="real", source_dataset=self.name,
                method="pristine", manip_type=self.manip_type,
                video_id=img.stem, is_video=False,
            )
