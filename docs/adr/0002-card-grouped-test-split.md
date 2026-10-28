# 2. Card-grouped test split

Date: 2026-10-27. Status: accepted. Supersedes the test numbers in ADR 0001.

## Context

The PSA split was assigned per cert (a per-grade shuffle), so certs of one
printed card landed in different splits. 251 of the 388 test certs that
`train_probe` scores shared a printed card with train+val. Certs of a card
share grades, so the headline test number mixed seen-card performance
(MAE 0.414) with unseen-card performance (MAE 0.628, n 137). The shuffle also
re-dealt a whole grade whenever a PSA cert was added, so test certs drifted
into train between harvests (fixed by the sticky split, phase 1 of
`intent/stable-grouped-split`).

## Decision

- One-time re-split, `python -m ml.data.split --mode grouped --reassign`, run
  on 2026-10-27 after a DB and `data/models` backup.
- Every PSA cert goes to the split given by a hash of its card key
  (`brand|subject|number`, `ml/data/groups.py`, salt `pokescan-split-v1`):
  15% test, 15% val, the rest train. No printed card spans train/val/test.
- eBay certs go to train, except an eBay cert of a card with a PSA test cert,
  which gets `split='excluded'` (no loader reads it). There are 0 eBay rows
  today.
- The salt was not tuned. The 47 test certs graded 8 or below (simulated range
  49-61) were accepted rather than trying salts until the test set looked nicer.
- The old assignment is kept in `split_legacy`, filled on this first reassign
  only. From now on the default sticky mode places only new certs, so the
  test set stays fixed across harvests.
- Test-set policy (CLAUDE.md): the test split gives one final number per ADR.
  Model selection uses card-grouped CV on train+val.

## The new split

1,218 of 2,630 PSA certs changed split. Test straddle (test certs sharing a
card with train/val) went from 255 of 393 to 0 of 375 over all PSA certs
(251 of 388 to 0 of 370 over `train_probe`'s rows). 49 of the 388 old test
certs that `train_probe` scored are still in test.

Test certs scored by `train_probe` (usable precrops), by grade:

| Test split | n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | <=8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Old (per cert) | 388 | | | 1 | 1 | 4 | 5 | 9 | 30 | 110 | 228 | 50 |
| New (per card) | 370 | | | 1 | 1 | 3 | 6 | 7 | 28 | 115 | 209 | 46 |

The fit set (train+val) went from 2,223 to 2,241 certs. All 375 new test certs
are in `ml.eval.evaluate`'s raw-photo run (5 have no usable precrop).

## Metrics

Same model and features (`feature_spec_hash` `428e7a366e2e`, card features
ONNX unchanged), refit on the new train+val. Old: the 2026-10-23/27 head on the
per-cert split. New: the head trained on 2026-10-27 at commit `f44f72f`.

| | Old | New |
|---|---|---|
| Card-grouped 5-fold CV, n | 2,223 | 2,241 |
| CV MAE / exact / QWK (seed 0, `train_probe`) | 0.447 / 65.1% / 0.620 | 0.469 / 64.0% / 0.583 |
| CV MAE / exact / QWK (mean of seeds 0, 1, 2) | 0.452 / 64.7% / 0.616 | 0.466 / 64.3% / 0.581 |
| CV ECE / AUC <=8 vs >=9 / AUC 10 vs 9 (seed 0) | 0.036 / 0.853 / 0.739 | 0.031 / 0.853 / 0.721 |
| CV always-10 MAE / exact | 0.654 / 59.2% | 0.651 / 59.6% |
| Test (precrop features), n | 388 (251 seen cards) | 370 (0 seen cards) |
| Test MAE / exact / QWK | 0.490 / 62.1% / 0.524 | 0.441 / 66.5% / 0.485 |
| Test ECE / AUC <=8 vs >=9 / AUC 10 vs 9 | 0.057 / 0.856 / 0.732 | 0.054 / 0.834 / 0.789 |
| Test bias | -0.02 | +0.08 |
| Test always-10 MAE / exact | 0.644 / 58.8% | 0.660 / 56.5% |
| Served path (`ml.eval.evaluate`, raw photos), n | 393 | 375 |
| Served MAE [95% CI] / exact / QWK [95% CI] | 0.494 [0.42, 0.58] / 62.1% / 0.511 [0.35, 0.66] | 0.435 [0.36, 0.51] / 66.9% / 0.491 [0.35, 0.62] |
| Served ECE / bias | 0.067 / -0.02 | 0.053 / +0.07 |
| Served always-10 MAE / exact | 0.636 / 59.3% | 0.656 / 56.8% |
| Served fallback crops (label visible) | 5 | 5 |

The served "Old" column is the 2026-10-23 head (same features and CV numbers)
from ADR 0001's evaluation. The served "New" column is
`ml/eval/baseline.json`.

CV seeds 0/1/2 by MAE: old 0.4467 / 0.4552 / 0.4534, new 0.4694 / 0.4668 /
0.4610. The sanity check for this re-split was that the new 3-seed CV mean
(0.4657) lies within 0.02 of the old one (0.4518). It does (+0.014). CV was
card-grouped before and after, so only the fit population changed.

## Consequences

- The test number now measures unseen printed cards only. Test-number
  comparisons across this re-split are not like-for-like: the test set and the
  fit set both changed, and only 49 test certs are shared. Compare a later
  model with the "New" column, never with ADR 0001's test numbers.
- The new test MAE (0.44) is lower than the old test's unseen-card subset
  (0.63, n 137). That subset is not comparable: it held only the cards none of
  whose certs happened to land in train+val. With 46 test certs graded 8 or
  below, the test number is noisy (served MAE 95% CI 0.36-0.51).
  QWK fell (0.52 to 0.49) and the model over-predicts slightly (bias +0.08).
  The CV numbers, not the test number, are the basis for model selection.
- `ml/eval/baseline.json` is now tracked. It pins the served-path numbers
  above for `test_model_performance.py` and `evaluate`'s regression check.
- This ADR is the baseline that `intent/face-side-orientation-guard` measures
  against.
- The old assignment is in `split_legacy`. To reproduce the old numbers, use
  the backups (`data/backups/pre-grouped-split.sqlite`,
  `data/models/backup/pre-grouped-split/`) or copy `split_legacy` into `split`
  on a copy of the DB. No command reads `split_legacy` directly.
