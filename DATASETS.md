# Source datasets — acquisition guide

The ID test set is built from **7 public deepfake datasets** (+ CelebA as an
identity source for the generation stage). Each is gated differently: some are
open, several require agreeing to a license, and two are very large. Set each
dataset's `root` in `config/datasets.yaml` (relative to `paths.raw`, i.e.
`data/raw/<name>`) and keep `enabled: true` only for the ones you obtain.

Expected sizes (approximate, video form):

| Dataset        | Size      | Access            |
|----------------|-----------|-------------------|
| UADFV          | ~2 GB     | open              |
| DF-TIMIT       | ~6 GB     | license agreement |
| Celeb-DF       | ~12 GB    | open (Google Drive)|
| DF-1.0         | ~50 GB    | license agreement |
| FF++           | ~400 GB+  | license agreement |
| DFDC           | ~500 GB   | open (Kaggle)     |
| ForgeryNet     | ~7 TB     | license agreement |
| CelebA (ids)   | ~1.5 GB   | open              |

You do **not** need all of them — disable any you skip. More sources → better
method coverage (target ≥ 13 manipulation methods).

---

## UADFV  (`manip_type: AEGAN`)
Tiny set of 49 real / 44 fake videos.
- Open. Various community mirrors; e.g. search "UADFV dataset deepfake".
- Expected layout: `data/raw/UADFV/{real,fake}/*.mp4` (path hints classify).

## DF-TIMIT  (`manip_type: AEGAN`)
DeepFakeTIMIT, higher/lower quality swaps.
- License: request via the TIMIT/DF-TIMIT distribution (academic use).
- Expected layout: `data/raw/DF-TIMIT/{higher_quality,lower_quality}/{real,fake}/*.mp4`
  (or a flat `{real,fake}` layout — both handled).

## Celeb-DF  (`manip_type: AEGAN`)
Celeb-DF v2: 590 real + 5,639 fake videos.
- Open: [Google Drive](https://google.github.io/celeb-deepfake-faceset/) /
  [project page](https://github.com/yuezunli/celeb-deepfake).
- Expected layout: `data/raw/Celeb-DF/{Celeb-real,YouTube-real,Celeb-synthesis}/*.mp4`.

## DF-1.0 / DeeperForensics-1.0  (`manip_type: AEGAN`)
50,000 fake + 11,000 real videos at multiple perturbation depths.
- License: [project page](https://github.com/EndlessSama/DeeperForensics-1.0)
  (accept agreement to download).
- Expected layout: `data/raw/DeeperForensics-1.0/{source_videos,manipulated_videos}/*.mp4`.

## FaceForensics++ (FF++)  (`manip_type: AEGAN`)
The benchmark trains/tests on **DF, FS, FShifter** (configure via
`datasets.yaml` `methods`). Distribu­ted as per-sequence frame folders.
- License: request at the [FF++ download form](https://github.com/ondyari/FaceForensics).
- Expected layout:
  ```
  data/raw/FaceForensics++/original_sequences/youtube/<quality>/sequences/<id>/
  data/raw/FaceForensics++/manipulated_sequences/<deepfakes|faceswap|faceshifter>/<quality>/sequences/<src>_<dst>/
  ```
  `<quality>` defaults to `c23` (set `FF++.quality` in `datasets.yaml`).

## DFDC  (`manip_type: AEGAN`)
Deepfake Detection Challenge — 113,000+ videos with per-part `metadata.json`.
- Open: [Kaggle](https://www.kaggle.com/competitions/deepfake-detection-challenge/data)
  and the [DFDC preview](https://ai.meta.com/datasets/dfdc/).
- Expected layout: `data/raw/DFDC/dfdc_train_part_*/{metadata.json,videos/*.mp4}`.
  Labels are read automatically from each part's `metadata.json`.

## ForgeryNet  (`manip_type: Graphic`)
Very large image+video forgery dataset with per-phase label CSVs.
- License: [project page](https://shahrukhx01.github.io/ForgeryNet/)
  (academic; multi-part download).
- Expected layout: `data/raw/ForgeryNet/...` plus a `*label*.csv`. The reader auto-
  detects the CSV and parses path/category columns; if your release's columns differ,
  tweak `ForgeryNetDataset._guess_path_column` / `_guess_method_column`.

## CelebA (identity source, `enabled: false` by default)
Used **only** by the FSGAN/MegaFS generation stage as source identities.
- Open: [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebData.html) or Kaggle.
- Expected layout: `data/raw/CelebA/Img/img_align_celeba/*.jpg`.
- Enable only if you run script `05_generate_private.py`.

---

## Tips

- Start small: enable **UADFV + Celeb-DF + a slice of DFDC**, run the pipeline with
  `--limit`, and confirm `manifest.csv` / `stats.json` look right before scaling.
- For very large datasets (FF++, DFDC, ForgeryNet), you can point `root` at a
  subset directory containing just the videos you've downloaded.
- Real/fake labels are taken from metadata (DFDC) or path hints (the rest). If your
  layout differs, adjust the matching reader in `idtest/datasets/readers.py`.
