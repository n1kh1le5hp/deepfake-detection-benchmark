"""MegaFS wrapper — reproduces the paper's 2,937 self-generated MegaFS images.

MegaFS (zyainfal/One-Shot-Face-Swapping-on-Megapixels, CVPR 2021) is the first
megapixel one-shot face swapper. The paper generates private fake *images* by
swapping CelebA identities, then runs them through the same two-phase
selection, keeping 2,937.

This is a subprocess wrapper. Clone the repo into ``external/megafs`` and fetch
its StyleGAN2 weights (see generation/README.md), then point it at source
identities + target images. The exact inference entry point is configurable.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .fsgan_wrap import SwapPair, build_pairs


class MegaFSGenerator:
    """Drive MegaFS's image-swap inference over (target, source) pairs."""

    def __init__(
        self,
        repo_dir: str | Path,
        out_dir: str | Path,
        weights_dir: Optional[str | Path] = None,
        python: str = sys.executable,
        script: str = "inference/inference_image.py",
        extra_args: Optional[Sequence[str]] = None,
    ):
        self.repo = Path(repo_dir)
        self.out_dir = Path(out_dir)
        self.weights_dir = Path(weights_dir) if weights_dir else None
        self.python = python
        self.script = script
        self.extra_args = list(extra_args or [])
        self._validate()

    def _validate(self) -> None:
        if not (self.repo / "inference").is_dir() and not (self.repo / self.script).exists():
            raise FileNotFoundError(
                f"MegaFS repo not found at {self.repo} (expected 'inference/'). "
                "See generation/README.md for setup."
            )
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _build_cmd(self, pair: SwapPair, out_path: Path) -> List[str]:
        cmd = [self.python, self.script, "--target", pair.target, "--source", pair.source,
               "--out", str(out_path)]
        if self.weights_dir:
            cmd += ["--weights", str(self.weights_dir)]
        return cmd + self.extra_args

    def generate(self, pairs: Sequence[SwapPair]) -> List[Path]:
        produced: List[Path] = []
        for i, pair in enumerate(pairs):
            out_path = self.out_dir / f"megafs_{i:05d}.png"
            cmd = self._build_cmd(pair, out_path)
            try:
                subprocess.run(cmd, cwd=str(self.repo), check=True)
            except subprocess.CalledProcessError as e:
                print(f"[MegaFS] swap {i} failed ({e}); skipping", file=sys.stderr)
                continue
            if out_path.exists():
                produced.append(out_path)
        return produced
