# ID Test Set — Reconstruction Pipeline

A faithful, reproducible reimplementation of the construction methodology behind the
**Imperceptible & Diverse (ID) test set** from
*"Towards Benchmarking and Evaluating Deepfake Detection"* (Lin et al., XJTU —
IEEE TIFS 2024, [arXiv:2203.02115](https://arxiv.org/abs/2203.02115)). This is the
hard-sample evaluation set behind the
[`aisec-xjtu-group/deepfake_benchmark`](https://github.com/aisec-xjtu-group/deepfake_benchmark)
leaderboard, which is **not publicly downloadable** (it contains a self-generated
portion). This pipeline rebuilds it from public sources on your own GPU.

> ⚠️ **Scope:** This project reproduces the *construction method*, not the exact
> bytes. See **Known limits** below.

---

## What it builds

The ID test set, per the paper:

| Property            | Target (paper)                |
|---------------------|-------------------------------|
| Fake / real images  | 25,697 / 25,697 (balanced)    |
| Source datasets     | UADFV, DF-TIMIT, Celeb-DF, DF-1.0, FF++ (DF/FS/FShifter), DFDC, ForgeryNet |
| Selection           | Two-phase: detection-model (false acceptance) **+** user-perception (≥15/30 humans) |
| Private fakes       | 40 FSGAN videos + 2,937 MegaFS images (self-generated; **reproducible** — methods are open-source) |
| Method coverage     | ≥ 13 manipulation methods     |
| Perturbations       | 5 (contrast, saturation, Gaussian blur, JPEG, white Gaussian noise) |

## Pipeline

```
01 extract + align faces (Dlib + OpenCV, ffmpeg)
02 score fakes with detector ensemble            ── Phase 1 ──
03 keep fakes falsely accepted as real (P(real)≥τ)
04 user-perception step (model-proxy OR manual UI)── Phase 2 ──
05 (optional) generate FSGAN/MegaFS fakes → same selection
06 balance to 25,697/25,697 + ≥13 methods → manifest + stats
07 apply the 5 perturbations
```

Outputs live in `data/out/`:
- `images/` — assembled balanced face images
- `manifest.csv` — one row per image (source of truth)
- `stats.json` — Table-2-style distribution + method coverage
- `perturbed/` — perturbed variants

## Quickstart

```bash
pip install -r requirements.txt               # see SETUP.md for dlib/ffmpeg
python scripts/00_setup_check.py             # validate env + dataset roots

# Point config/datasets.yaml at your data, then:
python scripts/01_extract_and_align.py
python scripts/02_score_with_detectors.py
python scripts/03_phase1_select.py
python scripts/04_phase2_select.py
python scripts/05_generate_private.py        # only if FSGAN/MegaFS enabled
python scripts/06_balance_assemble.py
python scripts/07_apply_perturbations.py
```

Smoke-test on a tiny subset:
```bash
python scripts/01_extract_and_align.py --limit 20
```

## Configuration

Everything is driven by `config/`:
- `default.yaml` — paths, thresholds (τ, ensemble mode), target counts, perturbation ranges, seed.
- `datasets.yaml` — enable/locate each of the 7 sources.

Plug in **real** deepfake-detector weights for faithful Phase-1/2 (the default
`ConstantDetector` only makes the plumbing runnable). See `SETUP.md`.

## Faithful Phase-2 (humans)

The paper's user-perception step needs human annotators (≥15 of 30 per image).
We provide **both** paths:

- **`phase2.mode: proxy`** (default): a model scores perceived realism. *Approximation.*
- **`phase2.mode: manual`**: run `python annotation_ui/app.py`, have ~30 people
  judge blind, then `/export`. `scripts/04_phase2_select.py` applies the real 15/30 rule.

## Known limits (please read)

1. **Phase-2 proxy ≠ humans.** The default `proxy` mode is an automated
   approximation. For a faithful ID test set, use `manual` mode with real annotators.
2. **Detector weights are your responsibility.** Faithful selection requires real
   deepfake detectors; ship your own checkpoint via the `TorchDetector` builder.
3. **Exact counts vary** with which datasets you obtain. Targets are parameterized;
   `stats.json` reports actual coverage so you can compare to paper Table 2.
4. **FSGAN/MegaFS are finicky** (old PyTorch, StyleGAN2 CUDA ops). They're optional;
   without them the set omits the 40 videos / 2,937 images private portion.

## Testing

```bash
pytest tests/        # pure-logic tests (selection, balance, perturbations, align helpers)
```

## Project layout

See `SETUP.md` (environment), `DATASETS.md` (where to get the 7 source datasets),
and `idtest/generation/README.md` (FSGAN/MegaFS setup). Key modules:
- `idtest/preprocess/` — frame extraction + Dlib face alignment
- `idtest/datasets/` — unified readers for the 7 sources
- `idtest/selection/` — Phase-1/2 logic + pluggable detectors
- `idtest/generation/` — FSGAN/MegaFS wrappers
- `idtest/assemble/` — balance + manifest/stats
- `idtest/perturb/` — the 5 perturbations

## Reference

Bojia Lin, Jingyi Deng, Pengbin Hu, Chao Shen, Qian Wang, Qi Li.
*"Towards Benchmarking and Evaluating Deepfake Detection."* IEEE TIFS, 2024.
[arXiv:2203.02115](https://arxiv.org/abs/2203.02115)
