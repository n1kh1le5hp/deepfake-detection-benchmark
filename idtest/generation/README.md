# Generation stage — reproducing the "private" fakes

The paper's ID test set includes self-generated fakes produced by two
**open-source** methods, which makes the "private" portion reproducible:

| Method   | Target count | Repo                                                         | Inputs (per paper)            |
|----------|-------------|--------------------------------------------------------------|-------------------------------|
| FSGAN    | 40 videos   | https://github.com/YuvalNirkin/fsgan                         | CelebA identities + FF++ vids |
| MegaFS   | 2,937 images| https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels | CelebA identities + FF++ imgs |

These generated fakes are then put through the **same** two-phase selection
(Phase-1 false-acceptance + Phase-2 perception) before being added to the set.

## Setup

```bash
cd <project_root>/external

# 1. FSGAN (ICCV 2019 / v2)
git clone https://github.com/YuvalNirkin/fsgan.git
cd fsgan && pip install -r requirements.txt
# Download FSGAN pretrained weights (see that repo's README) into external/fsgan/weights
cd ..

# 2. MegaFS (CVPR 2021)
git clone https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels.git megafs
# Requires StyleGAN2 weights + the repo's pretrained encoders (see inference/README.md)
```

Then enable in `config/default.yaml`:

```yaml
generation:
  fsgan:  {enabled: true, target_videos: 40}
  megafs: {enabled: true, target_images: 2937}
```

## Notes / caveats

- The wrappers call each repo's CLI via `subprocess`. The exact module/script
  names (`fsgan.bin.swap_vid_vid`, `inference/inference_image.py`) may differ by
  release — override via the wrapper constructor (`swap_module=` / `script=`) or
  in `scripts/05_generate_private.py`.
- FSGAN/MegaFS have heavy, finicky dependencies (older PyTorch, StyleGAN2 CUDA
  ops). Run them in a dedicated environment if needed; the wrapper only needs
  the produced files on disk afterwards.
- If you cannot set these up, leave `enabled: false`; the pipeline still builds
  the **public-portion** ID test set from the 7 datasets without the private
  fakes (the result will be missing the 40 FSGAN / 2,937 MegaFS samples).
