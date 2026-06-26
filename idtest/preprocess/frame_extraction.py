"""Frame extraction via ffmpeg.

The paper's pre-processing pipeline is "frame extraction, face cropping and
face alignment". We shell out to the ffmpeg/ffprobe binaries (faster and more
robust than decoding in Python) and sample frames at a configurable rate.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from ..io_utils import ffmpeg_available, video_frame_count


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    fps: Optional[float] = 1.0,
    stride: Optional[int] = None,
    max_frames: Optional[int] = None,
    quality: int = 2,
) -> List[Path]:
    """Sample frames from *video_path* into *out_dir*.

    Sampling rule:
      * if *fps* is set -> keep ~fps frames per second (``-vf fps=<fps>``).
      * elif *stride* is set -> keep every Nth frame (``select=not(mod(n\\,N))``).
      * else -> keep every frame.
    *max_frames* caps how many frames are finally retained (deterministic,
    earliest-first).

    Returns the list of saved frame paths (sorted). Raises RuntimeError if
    ffmpeg is unavailable; callers may fall back to imageio.
    """
    if not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg/ffprobe not found on PATH. Install ffmpeg, or pre-extract "
            "frames and place them in the dataset root."
        )

    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%06d.jpg")

    if fps is not None:
        vf = f"fps={fps}"
    elif stride is not None and stride > 1:
        vf = f"select=not(mod(n\\,{stride}))"  # one frame every N
    else:
        vf = None

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path)]
    if vf:
        cmd += ["-vf", vf, "-vsync", "vfr"]
    cmd += ["-q:v", str(quality), pattern]
    subprocess.run(cmd, check=True)

    frames = sorted(out_dir.glob("frame_*.jpg"))
    if max_frames is not None and len(frames) > max_frames:
        # Deterministic earliest-first subsample.
        for extra in frames[max_frames:]:
            extra.unlink()
        frames = frames[:max_frames]
    return frames


def expected_frame_count(video_path: str | Path, fps: Optional[float], stride: Optional[int]) -> Optional[int]:
    """Best-effort count of frames that extraction would yield (for planning)."""
    total = video_frame_count(video_path)
    if total is None:
        return None
    if fps is not None:
        dur = _duration(video_path)
        return int(dur * fps) if dur else None
    if stride is not None and stride > 1:
        return total // stride
    return total


def _duration(video_path: str | Path) -> Optional[float]:
    if shutil.which("ffprobe") is None:
        return None
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            stderr=subprocess.DEVNULL,
        )
        return float(out.decode().strip())
    except Exception:
        return None
