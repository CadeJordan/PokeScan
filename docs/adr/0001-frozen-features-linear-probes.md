# 1. Grade with frozen DINOv2 region features and linear ordinal probes

Date: 2026-10-23. Status: accepted.

## Context

The served model was a dual-input ConvNeXt-Tiny fine-tuned end to end with a
CORN head. On the test split it scored MAE 0.84, exact 42%, QWK 0.32, worse
than always predicting 10 (MAE 0.64, exact 59%). The two latest training runs
(`mvp_8gb_sp05`, `mt_sp05`) collapsed to a constant prediction (val QWK 0.0
every epoch).

The data explains why end-to-end fine-tuning struggles: ~2.6k PSA certs, 87%
grade 9-10, ~100 certs graded 1-7 (mostly vintage cards), photographed through
slab plastic at ~470 px card width.

The classical-CV sub-grades didn't carry signal either. Against the real
grade: corners Spearman -0.05 (AUC 0.47 for <=8 vs >=9), edges -0.12 (0.44),
surface 0.02 (0.50). Centering put 44% of PSA 10 fronts outside 60/40, which a
real PSA 10 can't be.

The per-cert split also leaks card identity: 302 printed cards appear in more
than one of train/val/test, and certs of the same card share grades (76%
majority-grade purity).

## Decision

- Features: frozen DINOv2 ViT-B/14 with registers, per face, pooled into 11
  region vectors (CLS, mean, 4 corners, 4 edge strips, centre).
- Head: four proportional-odds probes (corners, edges, surface, global; both
  faces each) on standardised, whitened-PCA(128) features, stacked by a
  proportional-odds meta probe trained on their out-of-fold soft grades.
- The region probes' soft grades are the corners/edges/surface sub-grades.
- Centering: colour-run border measurement on both faces that abstains when
  unsure, with PSA's back tolerances (75/25 for a 10).
- Evaluation: card-grouped CV (StratifiedGroupKFold on brand|subject|number).

## Alternatives measured (card-grouped 5-fold CV on train+val)

| Model | MAE | exact | QWK | AUC <=8 vs >=9 | AUC 10 vs 9 |
|---|---|---|---|---|---|
| Always 10 | 0.654 | 59.2% | 0.00 | - | - |
| DINOv2-B, global CLS only, ordinal logistic | 0.518 | 60.4% | 0.48 | 0.84 | 0.72 |
| DINOv2-B, all regions concatenated | 0.476 | 61.9% | 0.61 | 0.85 | 0.73 |
| DINOv2-L, all regions | 0.490 | 62.3% | 0.58 | 0.85 | 0.72 |
| Attention-pooling (MIL) head over patch tokens | 0.519 | 61.4% | 0.55 | 0.84 | 0.70 |
| 10 per-region probes, stacked | 0.446 | 65.0% | 0.61 | 0.86 | 0.73 |
| **4 sub-grade-aligned probes, stacked (chosen)** | **0.447** | **65.1%** | **0.62** | **0.85** | **0.74** |

Single-fold check: partially fine-tuning DINOv2-B (last 4 blocks, 6 epochs)
peaked at MAE 0.48 / QWK 0.61 against the probe's 0.46 / 0.64 on the same
fold, at roughly 100x the training cost.

Region probes alone (QWK): corners 0.60, edges 0.59, surface 0.53, global
0.54. Back-only features beat front-only (0.57 vs 0.50): backs share one
design, so what varies is condition rather than artwork.

## Consequences

- Held-out test (388 certs): MAE 0.49, exact 62%, QWK 0.52; the old model was
  MAE 0.84, exact 42%, QWK 0.32 on the same split.
- On the 137 test certs whose printed card never appears in train+val,
  accuracy drops (MAE 0.63, exact 54%; always-10 there is MAE 0.88, exact
  51%), so part of the signal is still card or era identity. More data across
  eras is the main lever.
- The region sub-grades track condition (test means: 8.3 for grades <=8, 9.2
  for 9s, 9.6 for 10s) but move closely together. They indicate overall wear
  more than they pinpoint which region is damaged.
- 9 vs 10 remains hard (AUC ~0.74). At this photo resolution most of what
  separates them isn't visible.
- Training takes about a minute after features are cached, so retraining as
  the harvest grows is cheap.
- Inference: ~1.5 s per face for DINOv2 on CPU (ONNX Runtime); the head is numpy.
- The CORN `cummin` vs `cumprod` question (`intent/corn-decode-verification`)
  only affects the legacy model; proportional odds is monotone by construction.
- All training data is slab photos. Accuracy on raw phone photos is unmeasured;
  `python -m ml.eval.evaluate --manifest` can measure it once a labelled set exists.
- Front centering is measured for only ~17% of cards (mostly regular-bordered
  English cards) and backs for ~81%; full-art, silver-bordered and most Japanese
  fronts are reported as unmeasured. A side is measured only when its line
  chunks agree, which drops the least reliable faces: of the PSA 10 fronts this
  excluded, 47% had read outside 60/40, against 6% of those still measured.
