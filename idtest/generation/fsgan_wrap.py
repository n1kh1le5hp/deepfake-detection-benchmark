"""FSGAN wrapper — reproduces the paper's 40 self-generated FSGAN fake videos.

FSGAN (YuvalNirkin/fsgan, ICCV 2019) is subject-agnostic face swapping. The
paper generates private fakes by swapping CelebA identities into FF++ raw
videos, then runs them through the same two-phase selection.

This is a thin subprocess wrapper. Clone the official repo into
``external/fsgan`` and fetch its pretrained weights (see generation/README.md),
then point the script at source identities + target videos. The exact CLI
module name may differ between FSGAN versions; override via ``swap_module``.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


@dataclass
class SwapPair:
    target: str   # path to target VIDEO (whose face is replaced)
    source: str   # path to source identity IMAGE (the new identity)


class FSGANGenerator:
    """Drive FSGAN's video-swap command over a list of (target, source) pairs."""

    def __init__(
        self,
        repo_dir: str | Path,
        out_dir: str | Path,
        weights_dir: Optional[str | Path] = None,
        python: str = sys.executable,
        swap_module: str = "fsgan.bin.swap_vid_vid",
        extra_args: Optional[Sequence[str]] = None,
    ):
        self.repo = Path(repo_dir)
        self.out_dir = Path(out_dir)
        self.weights_dir = Path(weights_dir) if weights_dir else None
        self.python = python
        self.swap_module = swap_module
        self.extra_args = list(extra_args or [])
        self._validate()

    def _validate(self) -> None:
        if not (self.repo / "fsgan").is_dir():
            raise FileNotFoundError(
                f"FSGAN repo not found at {self.repo} (expected a 'fsgan/' subdir). "
                "See generation/README.md for setup."
            )
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _build_cmd(self, pair: SwapPair, out_path: Path) -> List[str]:
        # Default invocation; FSGAN v2's swap_vid_vid accepts positional args.
        cmd = [self.python, "-m", self.swap_module, pair.target, pair.source, str(out_path)]
        if self.weights_dir:
            cmd += ["--models_dir", str(self.weights_dir)]
        return cmd + self.extra_args

    def generate(self, pairs: Sequence[SwapPair]) -> List[Path]:
        """Run a swap for each pair; return the produced video paths."""
        produced: List[Path] = []
        for i, pair in enumerate(pairs):
            out_path = self.out_dir / f"fsgan_{i:04d}.mp4"
            cmd = self._build_cmd(pair, out_path)
            try:
                subprocess.run(cmd, cwd=str(self.repo), check=True)
            except subprocess.CalledProcessError as e:
                print(f"[FSGAN] swap {i} failed ({e}); skipping", file=sys.stderr)
                continue
            if out_path.exists():
                produced.append(out_path)
        return produced


def build_pairs(
    targets: Sequence[str | Path], sources: Sequence[str | Path], n: int
) -> List[SwapPair]:
    """Pair up to *n* targets with sources (cycling sources if fewer)."""
    pairs: List[SwapPair] = []
    for i in range(min(n, len(targets))):
        pairs.append(SwapPair(str(targets[i]), str(sources[i % len(sources)])))
    return pairs
