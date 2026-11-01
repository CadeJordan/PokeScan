# 3. A 10 needs P(10) above a tuned cut-off

Date: 2026-10-31. Status: accepted. Spec: `intent/grade-decision-rule/intent.md`.

## Context

The probe head outputs a grade distribution (cumulative P(grade > k)). Until
now `grade_int` was its median, 1 + the number of P(grade > k) above 0.5. On
card-grouped CV over train+val (2,241 certs: 1,336 10s, 613 9s), the median
called 10 for 76% of certs (1,712 of 2,241, mean over 5 seeds) against 60% true
10s, and called only 28% of true 9s a 9. That miss, a 9 told it's a 10, is the
most common error.

The rule only decides `grade_int`. The soft grade (`grade`), `rank_probs`, the
probes and every AUC are unchanged, so a rule can only trade 10s called 9
against 9s called 10.

## Decision

- `grade_int` is the median, except at the 9/10 boundary: 10 only if
  P(10) = P(grade > 9) is **above t**, otherwise the median of the rest (9).
  Grades 8 and below keep the median. `confidence` is still P(grade =
  `grade_int`), the highlighted bar on the result card.
- t lives in the head's metadata (`grade_head.json`, `ten_cutoff`), so a
  retrained head carries its own cut-off. A head without the key uses 0.5,
  which is exactly the old median (strict `>`, also at P(10) = 0.5). A present
  value that isn't a finite number in [0.5, 1) fails closed (503,
  `probe_load_error`).
- `train_probe` tunes t on its own out-of-fold predictions: `cross_validate`
  for seeds 0–4, then the **highest t in {0.50, 0.52, …, 0.70} whose mean exact
  accuracy is within 1 pp of the median's and whose mean MAE is no worse than
  the median's by more than the median's seed-to-seed SD** (the owner's
  "don't lose accuracy" objective). It ships only if t > 0.5. Seed 0 stays the
  reported `cv_grouped`. `--cv-folds 0` gives t = 0.5.
- The API, `train_probe.metrics` (CV and test numbers) and `ml.eval` all use
  the head's t, so reported numbers describe what the API returns. The legacy
  CORN path keeps its median.
- The result card's caption no longer calls the highlighted grade "the middle
  of this distribution": "The highlighted grade is the prediction; on a close
  call it can sit next to a taller bar."
- Mode and rounding the expected grade were measured and not chosen (below).
  Cost-weighted rules and cut-offs at other boundaries are out of scope.

## CV evidence (train+val only)

Out-of-fold predictions from `cross_validate`, 5 card-grouped folds, seeds 0–4,
the 2026-10-31 head's code and features. Mean ± SD over seeds. Output:
`test-artifacts/eval/grade_decision_rule/sweep.md`.

| Rule | Exact | MAE | QWK | 9 recall | 10 recall | Predicted 9 / 10 |
|---|---|---|---|---|---|---|
| Median (old) | 0.643 ± 0.005 | 0.464 ± 0.006 | 0.579 ± 0.002 | 0.281 | 0.911 | 379 / 1,712 |
| Mode | 0.645 ± 0.005 | 0.474 ± 0.004 | 0.555 ± 0.010 | 0.218 | 0.948 | 290 / 1,821 |
| Round the expected grade | 0.626 ± 0.005 | 0.470 ± 0.005 | 0.599 ± 0.003 | 0.428 | 0.812 | 612 / 1,453 |
| P(10) > 0.54 | 0.643 ± 0.005 | 0.461 ± 0.005 | 0.585 ± 0.003 | 0.342 | 0.882 | 463 / 1,627 |
| P(10) > 0.58 (chosen t, in-sample) | 0.634 ± 0.006 | 0.465 ± 0.006 | 0.587 ± 0.003 | 0.403 | 0.840 | 567 / 1,523 |
| P(10) > 0.60 | 0.629 ± 0.005 | 0.468 ± 0.006 | 0.589 ± 0.006 | 0.439 | 0.814 | 629 / 1,461 |
| **Tune t, cross-fitted** | **0.634 ± 0.005** | **0.467 ± 0.005** | **0.585 ± 0.003** | **0.391** | **0.845** | **551 / 1,539** |

- **The criterion picks t = 0.58.** The floor is exact ≥ 0.6333 (median
  0.6433 − 1 pp) and the ceiling MAE ≤ 0.4703 (median 0.4640 + SD 0.0063).
  t = 0.58 has exact 0.6341 and MAE 0.4654. At 0.60 exact falls to 0.6287,
  below the floor. The exact floor binds, not the MAE ceiling.
- **The cross-fitted number is the honest one.** For each seed and outer
  fold, t was chosen by the same criterion on the other four folds' certs (all
  five seeds' predictions there) and applied to that fold. The per-fold choices
  were 0.56 or 0.58 (18 of 25 folds chose 0.58). Paired against the median on the same
  predictions, per cert, averaged over seeds, with a 95% bootstrap over 1,100
  card groups:

  | Cross-fitted tuned t vs median | Δ | 95% CI |
  |---|---|---|
  | Exact | −0.0096 | [−0.0181, −0.0009] |
  | MAE | +0.0025 | [−0.0071, +0.0113] |
  | 9 recall | +0.110 | [+0.092, +0.131] |
  | 10 recall | −0.067 | [−0.079, −0.055] |
  | Certs whose grade changes | 172 of 2,241 (mean over seeds) | |

  The rule costs about 1 pp of exact accuracy (a real cost: the CI excludes 0),
  costs no MAE beyond noise, and calls 11 pp more of the true 9s a 9.
- The mode isn't worth it: +0.15 pp exact [−0.5, +0.9], but MAE +0.010
  [+0.002, +0.019] and 9 recall −6 pp. Rounding the expected grade costs
  1.7 pp exact [−2.9, −0.6] for 9 recall +15 pp. A fixed cut-off of 0.54 is
  free (exact −0.04 pp, MAE −0.003) but buys only +6 pp 9 recall. The owner's
  criterion allows up to 1 pp of exact accuracy, and 0.58 is the highest
  cut-off within it.
- `train_probe` reproduced this: t = 0.58 from the same seeds, and its tuning
  (per-seed exact / MAE, the grid means) is in `grade_head.json`
  (`ten_cutoff_tuning`). `cv_grouped` (seed 0) under the new rule: MAE 0.473,
  exact 62.6%, QWK 0.581, ECE 0.029, against 0.468 / 64.1% / 0.577 / 0.031
  under the median. These are in-sample for t; the cross-fitted row above is the
  number to quote.

## Test split (recorded once; nothing chosen on it)

`python -m ml.eval.evaluate --split test --robustness-limit 100 --perturbations
rotate_180,swap_faces --out-dir test-artifacts/eval/decision_rule`: the served
pipeline on raw photos, CPU, the same face guard. It's compared with
`test-artifacts/eval/cv_folds`, the same head under the median. The probes are
identical, so every soft grade is identical (max |Δ| 0.000). Only `grade_int`
moves.

| Test split, live pipeline (n 375) | Median (cv_folds) | P(10) > 0.58 (this ADR) |
|---|---|---|
| MAE [95% CI] | 0.435 [0.363, 0.509] | 0.456 [0.384, 0.533] |
| Exact [95% CI] | 66.7% [62.1, 71.7] | 64.0% [59.5, 69.1] |
| QWK [95% CI] | 0.506 [0.352, 0.647] | 0.499 [0.340, 0.637] |
| ECE | 0.049 | 0.051 |
| 9 recall (n 115) | 36.5% | 47.8% [38.9, 56.9] |
| 10 recall (n 213) | 94.8% | 84.0% [78.5, 88.3] |
| Predicted 9 / 10 | 76 / 284 | 114 / 246 |
| AUC ≤8 vs ≥9 / AUC 10 vs 9 | 0.832 / 0.778 | 0.832 / 0.778 |
| `rotate_180` MAE / grade flip rate (n 100) | 0.32 / 3% | 0.31 / 9% |
| `swap_faces` MAE / grade flip rate (n 100) | 0.33 / 0% | 0.32 / 0% |
| Always 10 MAE / exact | 0.656 / 56.8% | 0.656 / 56.8% |

- 38 of 375 grades change, all from 10 to 9: 13 true 9s now right, 23 true 10s
  now wrong, plus a PSA 8 and a PSA 5 that move one grade closer. Paired
  difference (per-cert bootstrap, 2,000): exact −2.7 pp [−5.9, +0.5], MAE
  +0.021 [−0.008, +0.053].
- That costs more than the cross-fitted CV estimate (−1.0 pp, +0.003). It's one
  draw of 375 certs, and its CIs cover the CV estimate. Nothing was changed
  because of it, and t was not re-chosen.
- The `rotate_180` flip rate (grade changed by rotating both photos) rises from 3
  to 9 of 100. The soft shift is identical (0.109). More test certs now sit near
  the 9/10 boundary, so the same shift crosses it more often. The MAE under
  rotation didn't get worse.
- `train_probe`'s own test metrics (precrop features, n 370) under the new rule:
  MAE 0.451, exact 64.6%, QWK 0.498, ECE 0.056, against 0.438 / 66.5% / 0.499 /
  0.052 under the median.

## Consequences

- The head's probes are byte-identical to the 2026-10-31 head (`grade_head.npz`
  sha256 `0f62a686…`, unchanged). Only `grade_head.json` changed: `ten_cutoff`
  0.58, `ten_cutoff_tuning`, and `cv_grouped` / `test` now under the served
  rule. The old head is in
  `data/backups/grade_head.pre-decision-rule-2026-10-31.npz` / `.json`. Restoring
  that `.json` (or deleting the key) restores the median.
- Headline exact accuracy is about 1 pp lower than under the median in CV
  (2.7 pp on the one test draw), and 9 recall is about 11 pp higher. Compare old and new numbers with this in mind:
  ADR 0002's exact and recall are under the median.
- `train_probe` runs CV five times (~7 min instead of ~1.5) and may pick a
  different t after a retrain. That's intended: the cut-off belongs to the
  probabilities it was tuned on.
- On a close call the highlighted grade can sit next to a taller bar more often
  than before: a 9 with P(10) between 0.5 and 0.58 is shown as a 9 next to a
  taller 10.
