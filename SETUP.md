# Setup

Recommended: Python 3.10–3.11 on a CUDA GPU machine.

## 1. System packages

```bash
# Ubuntu/Debian
sudo apt-get install -y ffmpeg cmake build-essential python3-dev
# macOS (Homebrew)
brew install ffmpeg cmake
```

`ffmpeg`/`ffprobe` drive frame extraction; `cmake` is needed to build `dlib`.

## 2. Python environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`dlib` compiles from source — if the wheel isn't available, ensure `cmake` and a
C++ compiler are installed (step 1).

## 3. Dlib landmark model (required for face alignment)

```bash
mkdir -p external/models
cd external/models
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
cd ../..
```

Optional (better face detection): also fetch
`http://dlib.net/files/mmod_human_face_detector.dat.bz2` and set
`face.detector: cnn` in `config/default.yaml`.

## 4. (Faithful) deepfake detectors for Phase-1 / Phase-2

The pipeline ships a `ConstantDetector`/`RandomDetector` so it runs end-to-end
without weights, but those are **plumbing only**. For faithful hard-sample
selection, plug in real detectors, e.g. trained Xception / EfficientNet /
MesoNet checkpoints. Register a builder and a name:

```python
# in your own small script, or a config extension:
import torch.nn as nn
from idtest.selection.detectors import TorchDetector

def build_xception() -> nn.Module:
    ...  # return your model with a 2-class head

detector = TorchDetector(
    builder=build_xception,
    weights="external/weights/xception_ff.pth",
    input_size=299, real_index=0,
    device="cuda", name="xception",
)
```

Then list its name in `config/default.yaml` under `phase1.detectors` (and use it
as `phase2.proxy_model` for the perception proxy). The cleaner route is to extend
`build_detector()` in `idtest/selection/detectors.py` with a registry entry that
maps a config name to your builder+weights.

## 5. (Optional) FSGAN / MegaFS for the private fakes

See `idtest/generation/README.md`. Clone both repos under `external/` and fetch
their pretrained weights. Enable them in `config/default.yaml`:
```yaml
generation:
  fsgan:  {enabled: true, target_videos: 40}
  megafs: {enabled: true, target_images: 2937}
```

## 6. Verify

```bash
python scripts/00_setup_check.py
```
It checks ffmpeg, dlib + the landmark model, torch/CUDA, dataset roots, and the
optional generators. Resolve any `[FAIL]` before running the pipeline.

## 7. Unit tests

```bash
pytest tests/
```
