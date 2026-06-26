# ID Test Set — Detector Evaluation

**Set:** 51394 images (25697 fake / 25697 real), 7 manipulation methods.
**Detectors:** xception, meso4, meso4inception, ffd, facexray, patch, fwa_trained.
**Conditions:** clean, color_contrast, color_saturation, gaussian_blur, jpeg_compression, white_gaussian_noise.

## Table 1 — Clean performance

| Detector | AUC | Accuracy@0.5 | EER |
|----------|-----|--------------|-----|
| xception | 0.5639 | 0.5863 | 0.4181 |
| meso4 | 0.5045 | 0.5000 | 0.5000 |
| meso4inception | 0.6031 | 0.5440 | 0.4151 |
| ffd | 0.5464 | 0.4966 | 0.4540 |
| facexray | 0.6382 | 0.4992 | 0.3917 |
| patch | 0.6701 | 0.4998 | 0.3561 |
| fwa_trained | 0.5081 | 0.4754 | 0.4877 |

## Table 2 — Robustness (AUC by condition)

| Detector | clean | color_contrast | color_saturation | gaussian_blur | jpeg_compression | white_gaussian_noise | mean-pert | drop vs clean |
|----------|------|------|------|------|------|------|----------|---------------|
| xception | 0.5639 | 0.5703 | 0.5724 | 0.5809 | 0.6090 | 0.6108 | 0.5887 | -0.0248 |
| meso4 | 0.5045 | 0.5094 | 0.5044 | 0.4990 | 0.5060 | 0.5147 | 0.5067 | -0.0022 |
| meso4inception | 0.6031 | 0.5958 | 0.5958 | 0.5959 | 0.6142 | 0.6129 | 0.6029 | 0.0002 |
| ffd | 0.5464 | 0.5552 | 0.5428 | 0.5872 | 0.5277 | 0.5131 | 0.5452 | 0.0012 |
| facexray | 0.6382 | 0.6063 | 0.5750 | 0.6380 | 0.5004 | 0.5454 | 0.5730 | 0.0651 |
| patch | 0.6701 | 0.6599 | 0.6475 | 0.6732 | 0.4485 | 0.4358 | 0.5730 | 0.0971 |
| fwa_trained | 0.5081 | 0.4980 | 0.5084 | 0.5414 | 0.7090 | 0.5941 | 0.5702 | -0.0621 |

## Table 3 — Per-method AUC (clean; method fakes vs all reals)

| Method (n fake) | xception | meso4 | meso4inception | ffd | facexray | patch | fwa_trained |
|----------------|------|------|------|------|------|------|------|
| Celeb-DF (19434) | 0.6224 | 0.4851 | 0.6552 | 0.5919 | 0.6816 | 0.7409 | 0.5049 |
| DF-TIMIT-HQ (254) | 0.6527 | 0.6875 | 0.6855 | 0.6037 | 0.8056 | 0.8326 | 0.7631 |
| DF-TIMIT-LQ (39) | 0.7042 | 0.6188 | 0.8800 | 0.7236 | 0.8885 | 0.9397 | 0.8205 |
| FF++-DF (288) | 0.6416 | 0.5814 | 0.5285 | 0.6748 | 0.7654 | 0.7397 | 0.5473 |
| FF++-FS (290) | 0.6031 | 0.4884 | 0.3583 | 0.7885 | 0.6197 | 0.5036 | 0.2836 |
| FF++-FShifter (5374) | 0.3406 | 0.5622 | 0.4261 | 0.3571 | 0.4657 | 0.4103 | 0.5154 |
| UADFV (18) | 0.6744 | 0.3810 | 0.6345 | 0.7277 | 0.6342 | 0.4583 | 0.5615 |

---

*Convention: detectors output P(real); label 1 = fake; AUC computed on fake-score = 1 − P(real). Accuracy uses the 0.5 threshold.*
*Caveat: this set currently covers 7 manipulation methods (paper target ≥ 13); ForgeryNet-Graphic fakes = 0. Treat as a checkpoint evaluation of the set as it stands today.*
