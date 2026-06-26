# Per-Source-Dataset Benchmark (original extraction)

Detectors are DeepfakeBench **FF++ c23-trained** weights, tested on each source dataset's real vs fake faces from the full extraction (clean only). This is the standard per-dataset format; compare against the in-domain AUCs in the paper's Table 4.

## AUC by source dataset

| Source dataset | xception | meso4 | meso4inception | ffd | facexray |
|----------------|------|------|------|------|------|
| Celeb-DF (50860) | 0.7579 | 0.5015 | 0.6729 | 0.6999 | 0.6491 |
| DF-TIMIT (2769) |   -   |   -   |   -   |   -   |   -   |
| FF++ (30381) | 0.8350 | 0.5291 | 0.7032 | 0.7978 | 0.8111 |
| ForgeryNet (11929) |   -   |   -   |   -   |   -   |   -   |
| UADFV (580) | 0.9390 | 0.5484 | 0.8816 | 0.8872 | 0.6659 |

## Accuracy@0.5 by source dataset

| Source dataset | xception | meso4 | meso4inception | ffd | facexray |
|----------------|------|------|------|------|------|
| Celeb-DF | 0.8219 | 0.8661 | 0.8689 | 0.1353 | 0.1935 |
| DF-TIMIT | 0.9733 | 1.0000 | 1.0000 | 0.0878 | 0.3608 |
| FF++ | 0.7431 | 0.7494 | 0.7582 | 0.3302 | 0.4045 |
| ForgeryNet | 0.5037 | 0.0000 | 0.0858 | 0.9850 | 0.8895 |
| UADFV | 0.7879 | 0.5017 | 0.5741 | 0.5121 | 0.5534 |

---

*Convention: detector outputs P(real); label 1 = fake; AUC on fake-score = 1 − P(real).* *ForgeryNet's public test set is almost all REAL (fakes held back), so its per-source AUC is noisy/near-random — not a detector failure.*
