"""Centralized path resolution and config loading.

All stages read configuration through :func:`load_config`, which merges
``config/default.yaml`` and ``config/datasets.yaml`` into a single dict and
resolves every path against the project root.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import yaml

# Project root = parent of the `idtest/` package directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def load_config(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Load the main + datasets configs.

    Passing an explicit *path* overrides ``config/default.yaml`` (useful for
    experiments), while ``datasets.yaml`` is always read from ``CONFIG_DIR``.
    """
    main_path = Path(path) if path else CONFIG_DIR / "default.yaml"
    datasets_path = CONFIG_DIR / "datasets.yaml"

    with open(main_path) as f:
        cfg = yaml.safe_load(f)
    with open(datasets_path) as f:
        cfg["datasets"] = yaml.safe_load(f)

    # Resolve path roots to absolute locations.
    roots = cfg.setdefault("paths", {})
    for key in ("raw", "work", "out", "external"):
        roots[key] = str(_resolve(roots.get(key, key)))
    return cfg


def _resolve(path: str | os.PathLike) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def raw_root(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["paths"]["raw"])


def work_root(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["paths"]["work"])


def out_root(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["paths"]["out"])


def external_root(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["paths"]["external"])


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    """Create the work/ and out/ roots if missing (raw is expected to exist)."""
    for fn in (work_root, out_root, external_root):
        fn(cfg).mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch RNGs for determinism."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
