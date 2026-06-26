"""Shared bootstrap for the numbered pipeline scripts.

Adds the project root to ``sys.path`` and provides a single ``load()`` helper
that reads config, seeds RNGs, and ensures output dirs exist. Intermediate
state lives under ``data/work/``; final output under ``data/out/``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from idtest.paths import (  # noqa: E402
    ensure_dirs, external_root, load_config, out_root, seed_everything, work_root,
)


def load(cfg_path: str | None = None):
    """Load + seed config and ensure dirs. Returns the merged config dict.

    If *cfg_path* is None, honors the ``IDTEST_CONFIG`` env var so a non-default
    config (e.g. ``config/demo.yaml``) can drive the numbered scripts without
    editing them.
    """
    if cfg_path is None:
        cfg_path = os.environ.get("IDTEST_CONFIG")
    cfg = load_config(cfg_path)
    seed_everything(cfg.get("seed", 2022))
    ensure_dirs(cfg)
    return cfg


def work(cfg) -> Path:
    return work_root(cfg)


def out(cfg) -> Path:
    return out_root(cfg)


def ext(cfg) -> Path:
    return external_root(cfg)


def read_json(path: Path):
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def write_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
