"""Registry: map config dataset names -> reader classes and build the active set."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Type

from ..paths import raw_root
from .base import Sample, SourceDataset
from .readers import (
    CelebADataset,
    CelebDFDataset,
    DF10Dataset,
    DFDCDataset,
    DFTIMITDataset,
    FFPlusDataset,
    ForgeryNetDataset,
    UADFVDataset,
)

# Config key -> reader class.
DATASET_REGISTRY: Dict[str, Type[SourceDataset]] = {
    "UADFV": UADFVDataset,
    "DF-TIMIT": DFTIMITDataset,
    "Celeb-DF": CelebDFDataset,
    "DF-1.0": DF10Dataset,
    "FF++": FFPlusDataset,
    "DFDC": DFDCDataset,
    "ForgeryNet": ForgeryNetDataset,
    "CelebA": CelebADataset,
}


def build_datasets(cfg: dict) -> List[SourceDataset]:
    """Instantiate every enabled dataset declared in config/datasets.yaml."""
    root = raw_root(cfg)
    datasets: List[SourceDataset] = []
    for name, spec in cfg["datasets"].items():
        if not spec.get("enabled", False):
            continue
        cls = DATASET_REGISTRY.get(name)
        if cls is None:
            raise KeyError(f"No reader registered for dataset '{name}'")
        ds_root = Path(spec["root"])
        if not ds_root.is_absolute():
            ds_root = root / ds_root
        datasets.append(
            cls(
                root=ds_root,
                manip_type=spec.get("manip_type", "Unknown"),
                enabled=True,
                methods=spec.get("methods", ["DF", "FS", "FShifter"]),
            )
        )
    return datasets


def iter_all_samples(cfg: dict) -> List[Sample]:
    """Flatten every enabled dataset into a list of Samples (preserves order)."""
    out: List[Sample] = []
    for ds in build_datasets(cfg):
        out.extend(ds)
    return out
