"""Pluggable deepfake detectors and ensemble scoring.

A detector exposes :meth:`predict_real`, returning P(real) for a batch of BGR
uint8 face images. The Phase-1 selection treats a FAKE as "hard" when detectors
falsely accept it as real with high confidence, i.e. P(real) is high.

The default :class:`TorchDetector` is fully functional once you point it at a
checkpoint and an architecture builder (see SETUP.md). For smoke-testing the
pipeline without trained weights, use :class:`ConstantDetector` /
:class:`RandomDetector` (clearly non-faithful, for plumbing only).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np


class Detector(ABC):
    """Interface: map BGR uint8 face images -> P(real) in [0,1]."""

    name: str = "base"

    @abstractmethod
    def predict_real(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Return an (N,) float array of P(real) for *images*."""

    def predict_real_one(self, image: np.ndarray) -> float:
        return float(self.predict_real([image])[0])


class ConstantDetector(Detector):
    """Returns a fixed P(real). TESTING/SMOKE-TEST ONLY (not a real detector)."""

    def __init__(self, p_real: float = 0.9, name: str = "constant"):
        self.p_real = float(p_real)
        self.name = name

    def predict_real(self, images: Sequence[np.ndarray]) -> np.ndarray:
        return np.full(len(images), self.p_real, dtype=np.float32)


class RandomDetector(Detector):
    """Deterministic pseudo-random P(real) for plumbing tests. NOT faithful."""

    def __init__(self, seed: int = 0, name: str = "random"):
        self.rng = np.random.default_rng(seed)
        self.name = name

    def predict_real(self, images: Sequence[np.ndarray]) -> np.ndarray:
        return self.rng.random(len(images)).astype(np.float32)


class TorchDetector(Detector):
    """Wrap a torchvision-style classifier checkpoint as a deepfake detector.

    Parameters
    ----------
    builder:
        Callable returning an ``nn.Module`` with a 2-class (or 1-logit) head.
    weights:
        Path to a ``.pth``/``.pt`` checkpoint (state_dict or full model).
    input_size:
        Square input size the model expects (e.g. 224 or 299).
    real_index:
        Output index that corresponds to "real" (for 2-logit heads). Set to
        ``None`` for single-logit (sigmoid) heads -> P(real)=1-sigmoid(logit).
    mean / std:
        Normalization stats the model was trained with.
    """

    def __init__(
        self,
        builder: Callable,
        weights: str,
        input_size: int = 224,
        real_index: Optional[int] = 0,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        device: str = "cuda",
        name: str = "torch",
    ):
        import torch

        self.name = name
        self.device = device
        self.input_size = int(input_size)
        self.real_index = real_index
        self._torch = torch
        self.model = builder().to(device).eval()
        state = torch.load(weights, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)
        self.mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, 3, 1, 1).to(device)

    def _preprocess(self, images: Sequence[np.ndarray]):
        import cv2

        t = self._torch
        tensors = []
        for img in images:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self.input_size, self.input_size))
            arr = rgb.astype(np.float32) / 255.0  # HWC
            tensors.append(t.from_numpy(arr).permute(2, 0, 1))  # CHW
        batch = t.stack(tensors).to(self.device)
        batch = (batch - self.mean) / self.std
        return batch

    def predict_real(self, images: Sequence[np.ndarray]) -> np.ndarray:
        t = self._torch
        if not images:
            return np.zeros(0, dtype=np.float32)
        with t.no_grad():
            batch = self._preprocess(images)
            logits = self.model(batch)
            if logits.shape[-1] == 1 or self.real_index is None:
                # Single-logit head: assume logit = P(fake); P(real)=1-sigmoid.
                p_fake = t.sigmoid(logits).squeeze(-1)
                return (1.0 - p_fake).cpu().numpy().astype(np.float32)
            probs = t.softmax(logits, dim=-1)
            return probs[:, self.real_index].cpu().numpy().astype(np.float32)


def build_detector(name: str, cfg: dict, external_root) -> Detector:
    """Construct a detector by name from the registry.

    ``default`` resolves to :class:`ConstantDetector` so the pipeline runs
    end-to-end out of the box; ``xception`` loads the real DeepfakeBench
    Xception checkpoint; ``torch:<path>`` is a drop-in custom detector.
    """
    name = name or "default"
    if name == "default":
        return ConstantDetector(p_real=0.9, name="default")
    if name == "random":
        return RandomDetector(seed=cfg.get("seed", 0))
    if name == "ffd":
        return FFDDetector(external_root, device=cfg.get("device", "cuda"))
    if name == "facexray":
        return FaceXrayDetector(external_root, device=cfg.get("device", "cuda"))
    if name == "patch":
        return PatchDetector(external_root, device=cfg.get("device", "cuda"))
    if name == "fwa_trained":
        # Self-trained FWA: Xception fine-tuned on FWA synthetic fakes (scripts/10_train_fwa.py).
        # Checkpoint is a raw Xception state_dict (no backbone. prefix) -> DeepfakeBenchDetector's
        # prefix-strip is a no-op and the keys load directly. Same arch/preproc as xception.
        return DeepfakeBenchDetector(
            {"name": "fwa_trained", "module": "xception", "class": "Xception",
             "cfg": {"num_classes": 2, "mode": "original", "inc": 3, "dropout": 0},
             "weights": "weights/fwa_trained/fwa_trained_best.pth"},
            external_root, device=cfg.get("device", "cuda"),
        )
    if name in DB_DETECTORS:
        return DeepfakeBenchDetector(
            DB_DETECTORS[name], external_root, device=cfg.get("device", "cuda"),
        )
    # 'torch:<path>' -> TorchDetector with a user builder (see SETUP.md).
    if name.startswith("torch:"):
        weights = name.split(":", 1)[1]
        builder = cfg.get("_detector_builder")
        if builder is None:
            raise ValueError(
                "TorchDetector requires a builder. Register one via "
                "cfg['_detector_builder'] = lambda: <nn.Module>."
            )
        return TorchDetector(builder=builder, weights=weights, device=cfg.get("device", "cuda"))
    raise KeyError(f"Unknown detector '{name}'")


class DeepfakeBenchDetector(Detector):
    """Generic loader for uniform DeepfakeBench PyTorch detectors.

    Loads a vendored network from ``external/deepfakebench_models/``, strips the
    ``backbone.`` prefix DeepfakeBench adds, and runs on 256x256 RGB faces
    normalized (x-0.5)/0.5. Works for any DB detector whose network class takes
    a single config dict (Xception, Meso4, MesoInception4).
    """

    def __init__(self, spec, external_root, device="cuda", real_index=0):
        import importlib
        import sys
        import cv2
        import torch

        self._torch = torch
        self._cv2 = cv2
        self.device = device
        self.real_index = real_index
        self.input_size = 256
        self.name = spec["name"]

        vendored = Path(external_root) / "deepfakebench_models"
        if str(vendored) not in sys.path:
            sys.path.insert(0, str(vendored))
        mod = importlib.import_module(spec["module"])
        cls = getattr(mod, spec["class"])
        self.model = cls(spec["cfg"]).to(device).eval()

        weights = Path(external_root) / spec["weights"]
        state = torch.load(weights, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # DeepfakeBench wraps networks as self.backbone -> strip that prefix.
        state = {(k[len("backbone."):] if k.startswith("backbone.") else k): v
                 for k, v in state.items()}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[{self.name}] missing keys: {len(missing)} (e.g. {missing[:3]})")
        if unexpected:
            print(f"[{self.name}] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")

    def _preprocess(self, images):
        t = self._torch
        tensors = []
        for img in images:
            rgb = self._cv2.cvtColor(img, self._cv2.COLOR_BGR2RGB)
            rgb = self._cv2.resize(rgb, (self.input_size, self.input_size))
            arr = rgb.astype(np.float32) / 255.0
            arr = (arr - 0.5) / 0.5  # (x-0.5)/0.5
            tensors.append(t.from_numpy(arr).permute(2, 0, 1))
        return t.stack(tensors).to(self.device)

    def predict_real(self, images):
        t = self._torch
        if not images:
            return np.zeros(0, dtype=np.float32)
        with t.no_grad():
            batch = self._preprocess(images)
            out = self.model(batch)
            logits = out[0] if isinstance(out, tuple) else out
            if logits.shape[-1] == 1:
                p_fake = t.sigmoid(logits).squeeze(-1)
                return (1.0 - p_fake).cpu().numpy().astype(np.float32)
            probs = t.softmax(logits, dim=-1)
            return probs[:, self.real_index].cpu().numpy().astype(np.float32)


class FFDDetector(Detector):
    """DeepfakeBench FFD (Feature Layered Mapping) detector.

    FFD is a *detector* wrapper (not a single network): the unified Xception
    backbone is split into ``fea_part1..5``; a :class:`RegressionMap` computes a
    soft attention mask after ``block7``; the masked features continue through
    ``block8..12`` and the classifier. See DeepfakeBench ``ffd_detector.py``.

    The released ``ffd_best.pth`` stores ``backbone.*`` (the unified Xception,
    incl. the unused-in-inference ``adjust_channel``) plus ``map.*`` (the
    RegressionMap); these match ``self.backbone`` + ``self.map`` exactly, so no
    key remapping is needed. Convention: ``softmax(logits)[:, 1]`` = P(fake),
    hence ``real_index = 0``.
    """

    def __init__(self, external_root, device="cuda", real_index=0):
        import importlib
        import sys
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import cv2

        self._torch = torch
        self._cv2 = cv2
        self.device = device
        self.real_index = real_index
        self.input_size = 256
        self.name = "ffd"

        vendored = Path(external_root) / "deepfakebench_models"
        if str(vendored) not in sys.path:
            sys.path.insert(0, str(vendored))
        xception_mod = importlib.import_module("xception")
        Xception = getattr(xception_mod, "Xception")
        SeparableConv2d = getattr(xception_mod, "SeparableConv2d")

        class RegressionMap(nn.Module):
            def __init__(self_, c_in):
                super().__init__()
                self_.c = SeparableConv2d(c_in, 1, 3, stride=1, padding=1, bias=False)
                self_.s = nn.Sigmoid()

            def forward(self_, x):
                mask = self_.s(self_.c(x))
                return mask, None

        class _FFD(nn.Module):
            def __init__(self_):
                super().__init__()
                self_.backbone = Xception(
                    {"num_classes": 2, "mode": "adjust_channel", "inc": 3, "dropout": 0}
                )
                self_.map = RegressionMap(728)

            def forward(self_, x):
                b = self_.backbone
                x = b.fea_part1(x)
                x = b.fea_part2(x)
                x = b.fea_part3(x)
                mask, _ = self_.map(x)
                x = x * mask
                x = b.fea_part4(x)
                x = b.fea_part5(x)
                return b.classifier(x)

        self.model = _FFD().to(device).eval()
        weights = Path(external_root) / "weights" / "ffd" / "ffd_best.pth"
        state = torch.load(weights, map_location=device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[ffd] missing keys: {len(missing)} (e.g. {missing[:3]})")
        if unexpected:
            print(f"[ffd] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")

    def _preprocess(self, images):
        t = self._torch
        tensors = []
        for img in images:
            rgb = self._cv2.cvtColor(img, self._cv2.COLOR_BGR2RGB)
            rgb = self._cv2.resize(rgb, (self.input_size, self.input_size))
            arr = rgb.astype(np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            tensors.append(t.from_numpy(arr).permute(2, 0, 1))
        return t.stack(tensors).to(self.device)

    def predict_real(self, images):
        t = self._torch
        if not images:
            return np.zeros(0, dtype=np.float32)
        with t.no_grad():
            logits = self.model(self._preprocess(images))
            if logits.shape[-1] == 1:
                p_fake = t.sigmoid(logits).squeeze(-1)
                return (1.0 - p_fake).cpu().numpy().astype(np.float32)
            probs = t.softmax(logits, dim=-1)
            return probs[:, self.real_index].cpu().numpy().astype(np.float32)


# HRNet-w18 architecture config (widths 18/36/72/144) for Face X-Ray. Identical
# structure to DeepfakeBench's cls_hrnet_w48.yaml with w18 channel widths.
class FaceXrayDetector(Detector):
    """Face X-ray detector (Li et al., CVPR 2020) — HRNet-w18 + X-ray mask.

    Uses the wkq-wukaiqi/Face-X-Ray unofficial implementation vendored as
    ``wkq_hrnet.py``. ``HRDetectNet`` wraps an HRNet-w18 backbone
    (``HRNet_layer``, returns the 4 multi-resolution branches) + bilinear
    upsamples + a 1x1 ``one_channel_conv`` (270 -> 1, the blending-boundary
    "X-ray" mask) + sigmoid + ``cls_layer`` (AdaptiveAvgPool -> Linear(1, 2)).
    The checkpoint keys match ``HRDetectNet`` exactly, so no remapping is needed.

    Preprocessing matches the original repo: ``transforms.ToTensor()`` (RGB in
    [0, 1], no mean/std subtraction), input resized to 317x317 (the training
    resolution). forward returns ``(mask, cls)``; ``softmax(cls)[:, 1]`` = P(fake).
    """

    def __init__(self, external_root, device="cuda", real_index=0):
        import importlib
        import sys
        import torch
        import cv2
        import yaml

        self._torch = torch
        self._cv2 = cv2
        self.device = device
        self.real_index = real_index
        self.input_size = 317          # training resolution (dataset.py resizes to 317x317)
        self.name = "facexray"

        vendored = Path(external_root) / "deepfakebench_models"
        if str(vendored) not in sys.path:
            sys.path.insert(0, str(vendored))
        wkq = importlib.import_module("wkq_hrnet")
        cfg = yaml.safe_load((Path(vendored) / "wkq_hrnet_w18.yaml").read_text())
        model = wkq.HRDetectNet(wkq.HighResolutionNet(cfg))

        weights = Path(external_root) / "weights" / "facexray" / "best_model.pth.tar"
        state = torch.load(weights, map_location=device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[facexray] missing keys: {len(missing)} (e.g. {missing[:3]})")
        if unexpected:
            print(f"[facexray] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
        self.model = model.to(device).eval()

    def _preprocess(self, images):
        # ToTensor-equivalent: RGB in [0, 1], no mean/std normalization.
        t = self._torch
        tensors = []
        for img in images:
            rgb = self._cv2.cvtColor(img, self._cv2.COLOR_BGR2RGB)
            rgb = self._cv2.resize(rgb, (self.input_size, self.input_size))
            arr = rgb.astype(np.float32) / 255.0
            tensors.append(t.from_numpy(arr).permute(2, 0, 1))
        return t.stack(tensors).to(self.device)

    def predict_real(self, images):
        t = self._torch
        if not images:
            return np.zeros(0, dtype=np.float32)
        with t.no_grad():
            _mask, cls = self.model(self._preprocess(images))   # HRDetectNet returns (mask, cls)
            probs = t.softmax(cls, dim=-1)
            return probs[:, self.real_index].cpu().numpy().astype(np.float32)


class PatchDetector(Detector):
    """Patch-Forensics ensemble (Chai et al., ECCV 2020).

    45 patch classifiers: for each Xception block N in 1..5, 9 models (different
    synthetic-training-data subsets). Each model is an Xception prefix (entry
    convs + blocks up to blockN) capped by a 1x1 ``convout`` head emitting a
    per-pixel 2-class map -- every pixel is one "patch" with a limited receptive
    field. Backbones are NOT shared (each model trained separately), so each runs
    its own forward. Per-image score = mean over the patch-map of
    ``softmax(convout)`` averaged across all 45 models. ``fake_class_id=0`` ->
    ``P(real) = softmax[:, 1]``. Vendored ``patch_forensics/`` from chail/patch-forensics.

    Note: this is ~45x a single forward, so scoring the full set takes several
    hours; runs are resumable per-detector via the cached score CSV.
    """

    def __init__(self, external_root, device="cuda", real_index=1):
        import glob
        import importlib
        import re
        import sys
        import torch
        import cv2

        self._torch = torch
        self._cv2 = cv2
        self.device = device
        self.real_index = real_index          # fake_class_id=0 -> real is index 1
        self.input_size = 299
        self.name = "patch"
        self._micro = 16                       # images per model-forward (memory bound)

        vendored = Path(external_root) / "deepfakebench_models"
        if str(vendored) not in sys.path:
            sys.path.insert(0, str(vendored))
        pf = importlib.import_module("patch_forensics")
        root = Path(external_root) / "weights" / "patch" / "patch_forensics_eccv"
        ckpts = sorted(glob.glob(str(root / "*_xception_block[1-5]_*" / "bestval_net_D.pth")))
        if not ckpts:
            raise FileNotFoundError(f"no patch-forensics checkpoints under {root}")
        self.models = []
        for p in ckpts:
            block = re.search(r"xception_(block[1-5])", p).group(1)
            net = pf.make_patch_xceptionnet(block)
            sd = torch.load(p, map_location=device, weights_only=False)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            net.load_state_dict(sd, strict=False)
            self.models.append(net.to(device).eval())
        print(f"[patch] loaded {len(self.models)} patch models")

    def _preprocess(self, images):
        t = self._torch
        tensors = []
        for img in images:
            rgb = self._cv2.cvtColor(img, self._cv2.COLOR_BGR2RGB)
            rgb = self._cv2.resize(rgb, (self.input_size, self.input_size))
            arr = rgb.astype(np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            tensors.append(t.from_numpy(arr).permute(2, 0, 1))
        return t.stack(tensors).to(self.device)

    def predict_real(self, images):
        t = self._torch
        if not images:
            return np.zeros(0, dtype=np.float32)
        n = len(images)
        accum = np.zeros(n, dtype=np.float32)
        with t.no_grad():
            for start in range(0, n, self._micro):
                batch = self._preprocess(images[start:start + self._micro])
                per_model = []
                for net in self.models:
                    out = net(batch)                                   # [b, 2, H, W]
                    p_real = t.softmax(out, 1)[:, self.real_index].mean(dim=(1, 2))
                    per_model.append(p_real)
                accum[start:start + batch.shape[0]] = t.stack(per_model, 0).mean(0).cpu().numpy()
        return accum.astype(np.float32)


# Registry of uniform DeepfakeBench PyTorch detectors (256x256, (x-0.5)/0.5 norm).
DB_DETECTORS = {
    "xception": {"name": "xception", "module": "xception", "class": "Xception",
                 "cfg": {"num_classes": 2, "mode": "original", "inc": 3, "dropout": 0},
                 "weights": "weights/xception/xception_best.pth"},
    "meso4": {"name": "meso4", "module": "mesonet", "class": "Meso4",
              "cfg": {"num_classes": 2, "inc": 3},
              "weights": "weights/meso4/meso4_best.pth"},
    "meso4inception": {"name": "meso4inception", "module": "mesonet", "class": "MesoInception4",
                       "cfg": {"num_classes": 2, "inc": 3},
                       "weights": "weights/meso4Incep/meso4Incep_best.pth"},
}


def ensemble_predict(detectors: Sequence[Detector], images: Sequence[np.ndarray]) -> np.ndarray:
    """Return per-detector P(real) as an (N, D) array (caller applies the rule)."""
    return np.stack([d.predict_real(images) for d in detectors], axis=1)
