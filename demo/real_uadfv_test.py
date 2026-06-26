#!/usr/bin/env python
"""Real end-to-end test on a single dataset (UADFV): ffmpeg frame extraction +
Dlib face alignment. Isolated — does not touch the main config or other datasets.
"""
import itertools
import sys
from pathlib import Path

ROOT = Path("/home/nikhi/id_test_reconstruction")
sys.path.insert(0, str(ROOT))

from idtest.datasets.readers import UADFVDataset
from idtest.io_utils import hash_id, read_image, write_image
from idtest.preprocess.face_align import FaceAligner
from idtest.preprocess.frame_extraction import extract_frames

OUT = ROOT / "data" / "work" / "uadfv_test"
(OUT / "aligned").mkdir(parents=True, exist_ok=True)

aligner = FaceAligner(
    size=256, detector="frontal",
    landmark_model="shape_predictor_68_face_landmarks.dat",
    external_root=ROOT / "external", min_face_size=40,
)

ds = UADFVDataset(root=ROOT / "data" / "raw" / "UADFV", manip_type="AEGAN",
                  enabled=True, methods=[])
samples = list(ds)
print(f"UADFV samples: {len(samples)}  (running REAL ffmpeg+dlib pipeline)")

n_frames = n_faces = no_face = errors = 0
for i, s in enumerate(samples):
    fdir = OUT / "frames" / hash_id(s.path)
    try:
        frames = extract_frames(s.path, fdir, fps=1.0, max_frames=4)
    except Exception as e:
        errors += 1
        continue
    n_frames += len(frames)
    for fp in frames:
        try:
            img = read_image(fp)
        except Exception:
            continue
        face = aligner.align_largest(img)
        if face is None:
            no_face += 1
            continue
        write_image(OUT / "aligned" / f"{hash_id(s.path, fp.name)}.png", face)
        n_faces += 1
    if (i + 1) % 20 == 0:
        print(f"  ...{i+1}/{len(samples)} videos processed")

print()
print(f"videos processed : {len(samples) - errors}/{len(samples)}")
print(f"frames extracted : {n_frames}")
print(f"faces aligned    : {n_faces}")
print(f"frames w/o face  : {no_face}")
print(f"errors           : {errors}")
print("sample aligned outputs:")
for p in itertools.islice(sorted((OUT / "aligned").glob("*.png")), 5):
    print("  ", p.relative_to(ROOT))
