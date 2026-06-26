"""Face crop + alignment with Dlib + OpenCV (as in the paper).

Pipeline per frame:
  1. Detect faces (Dlib HOG frontal detector by default; CNN optional).
  2. Predict 68 facial landmarks.
  3. Collapse to 5 keypoints (eye centres, nose tip, mouth corners) and fit a
     similarity transform to a canonical template -> aligned *size*x*size* crop.

The canonical 5-point template is the standard ArcFace/FFHQ template expressed
in normalized [0,1] coordinates and scaled to the requested output size.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

# dlib is imported lazily inside FaceAligner.__init__ so the rest of the module
# (and projects that only use the helpers) works without dlib installed.

# Canonical 5-point template (normalized); scaled by output size at runtime.
_TEMPLATE_5 = np.array(
    [
        [0.34191607, 0.46157411],  # left eye
        [0.65651693, 0.45983393],  # right eye
        [0.50022500, 0.64050554],  # nose tip
        [0.37097593, 0.82461670],  # mouth left
        [0.62957604, 0.82362019],  # mouth right
    ],
    dtype=np.float32,
)


def locate_model(name: str, external_root: Path) -> Path:
    """Find a Dlib model file under *external_root* or next to the package."""
    for base in (external_root, external_root / "models"):
        cand = base / name
        if cand.exists():
            return cand
    # Last-resort: assume it's on PATH-adjacent locations.
    return external_root / name


class FaceAligner:
    """Detect + align faces to a fixed square size using Dlib landmarks."""

    def __init__(
        self,
        size: int = 256,
        detector: str = "frontal",
        landmark_model: str = "shape_predictor_68_face_landmarks.dat",
        external_root: Optional[Path] = None,
        min_face_size: int = 80,
    ):
        import cv2

        self._cv2 = cv2
        import dlib  # imported here so module import does not require dlib
        self._dlib = dlib
        self.size = int(size)
        self.min_face_size = int(min_face_size)
        external_root = Path(external_root) if external_root else Path("external")
        model_path = locate_model(landmark_model, external_root)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Dlib landmark model not found: {model_path}. Download "
                f"http://dlib.net/files/{landmark_model}.bz2, decompress, and "
                f"place it under external/models/."
            )

        if detector == "cnn":
            mmod = locate_model("mmod_human_face_detector.dat", external_root)
            self._detector = self._dlib.cnn_face_detection_model_v1(str(mmod))
            self._cnn = True
        else:
            self._detector = self._dlib.get_frontal_face_detector()
            self._cnn = False
        self._predictor = self._dlib.shape_predictor(str(model_path))

    # -- public API ---------------------------------------------------------

    def align_largest(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Align the largest detected face in *image* (BGR uint8).

        Returns an ``size x size x 3`` BGR uint8 array, or None if no face that
        satisfies ``min_face_size`` is found.
        """
        gray = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2GRAY)
        rects = self._detect(gray)
        if not rects:
            return None
        # Pick the largest detection.
        rect = max(rects, key=lambda r: _rect_area(r))
        if _rect_area(rect) < self.min_face_size ** 2:
            return None
        shape = self._predictor(gray, rect)
        pts = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
        return self._warp(image, pts)

    def align_all(self, image: np.ndarray) -> List[np.ndarray]:
        """Align every qualifying face in *image* (largest-first ordering)."""
        gray = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2GRAY)
        out = []
        for rect in sorted(self._detect(gray), key=_rect_area, reverse=True):
            if _rect_area(rect) < self.min_face_size ** 2:
                continue
            shape = self._predictor(gray, rect)
            pts = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
            warped = self._warp(image, pts)
            if warped is not None:
                out.append(warped)
        return out

    # -- internals ----------------------------------------------------------

    def _detect(self, gray):
        """Return dlib rectangles (unwrap CNN detections)."""
        if self._cnn:
            return [r.rect for r in self._detector(gray, 1)]
        return self._detector(gray, 1)

    def _warp(self, image: np.ndarray, pts68: np.ndarray) -> np.ndarray:
        src5 = _five_from_68(pts68)
        dst = _TEMPLATE_5 * float(self.size)
        # estimateAffinePartial2D -> 4 DOF similarity transform (rot+scale+t).
        M, _ = self._cv2.estimateAffinePartial2D(src5.reshape(-1, 1, 2), dst.reshape(-1, 1, 2))
        if M is None:
            return None
        return self._cv2.warpAffine(
            image, M, (self.size, self.size),
            flags=self._cv2.INTER_LINEAR,
            borderMode=self._cv2.BORDER_REPLICATE,
        )


def _five_from_68(pts68: np.ndarray) -> np.ndarray:
    """Collapse 68 landmarks to the 5 canonical keypoints."""
    left_eye = pts68[36:42].mean(axis=0)
    right_eye = pts68[42:48].mean(axis=0)
    nose_tip = pts68[30]
    mouth_left = pts68[48]
    mouth_right = pts68[54]
    return np.array([left_eye, right_eye, nose_tip, mouth_left, mouth_right], dtype=np.float32)


def _rect_area(rect) -> int:
    """Works for both dlib.rectangle and dlib.mmod_rectangle.rect.

    dlib rectangle bounds are exposed as *methods* (.left(), .right(), .top(),
    .bottom()), so call them when needed.
    """
    r = getattr(rect, "rect", rect)

    def _v(x):
        return x() if callable(x) else x

    return (_v(r.right) - _v(r.left)) * (_v(r.bottom) - _v(r.top))
