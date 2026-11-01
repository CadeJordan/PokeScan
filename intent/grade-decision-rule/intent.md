# How the predicted grade is picked from the distribution

## Intent

Choose, on card-grouped CV over train+val, the rule that turns the probe's grade distribution into `grade_int`, and serve it. Today the rule is the median, and it calls 10 far more often than 10s occur. This continues the 9-vs-10 work in [[cv-ml-improvement-plan]] item 12, after [[full-resolution-pilot]] ruled out resolution, and follows the note in [[cv-fold-stratification]]: compare rules on pooled, paired out-of-fold predictions, not single folds.

## Why

`grade_head.decode` (`backend/app/ml/grade_head.py:155`) sets `grade_int` = 1 + the number of thresholds with P(grade > k) > 0.5, which is the median. `confidence` is P(grade = `grade_int`). The same `decode` feeds the API (`inference.py:502`), `train_probe.metrics` and so `ml.eval`.

Measured 2026-10-31 on train+val: `cross_validate`, 5 folds, seed 0, the served head's code, 2,241 certs (613 9s, 1,336 10s). Nothing was written under `data/`.

| Rule | Exact | MAE | QWK | 8 recall | 9 recall | 10 recall | Predicted 9 / 10 | Mean confidence |
|---|---|---|---|---|---|---|---|---|
| Median (today) | 0.641 | 0.468 | 0.577 | 0.22 | 0.27 | 0.91 | 372 / 1,719 | 0.634 |
| Mode (most likely grade) | 0.643 | 0.476 | 0.562 | 0.20 | 0.22 | 0.94 | 297 / 1,814 | 0.638 |
| Round the expected grade | 0.622 | 0.471 | 0.599 | 0.23 | 0.43 | 0.80 | 620 / 1,441 | 0.606 |
| 10 only if P(10) ≥ 0.55 | 0.636 | 0.466 | 0.583 | 0.22 | 0.35 | 0.87 | 492 / 1,599 | 0.626 |
| 10 only if P(10) ≥ 0.60 | 0.622 | 0.475 | 0.582 | 0.22 | 0.43 | 0.81 | 637 / 1,454 | 0.611 |
| 10 only if P(10) ≥ 0.65 | 0.610 | 0.481 | 0.584 | 0.22 | 0.56 | 0.73 | 833 / 1,258 | 0.583 |

What this shows:
- **The median calls 10 too often:** 77% of predictions are 10, but 60% of the certs are 10s. Only 27% of true 9s are called 9.
- **No rule is free.** The soft grade and its AUCs don't change, so a rule only moves certs between 9 and 10. Calling more 9s correctly means calling some 10s 9.
- **The mode isn't worth it.** It differs from the median on 145 certs, makes exact accuracy only 0.2 pp better, and makes MAE, QWK and 9 recall worse.
- **A small move is nearly free.** At a P(10) cut-off of 0.55, MAE and QWK are slightly better and 9 recall goes from 27% to 35%, for 0.5 pp of exact accuracy.
- **Matching the base rate costs more.** At 0.60 the predicted 9s and 10s roughly match the true counts (637 / 1,454 against 613 / 1,336). That gives 9 recall of 43% for 1.9 pp of exact accuracy.

These numbers are from one seed, and the cut-offs were picked by looking at the same predictions, so they are indicative only. The differences are small: 0.5 pp is about 11 certs. The work below re-measures them properly.

## Scope

1. **Measurement on train+val only.** Out-of-fold cumulative probabilities from `cross_validate` for seeds 0–4 (features read with `save=False`; nothing written under `data/`). For each seed and the mean ± SD across seeds, report the candidate rules' exact, MAE, QWK, per-grade recall for 8, 9 and 10, the predicted 9 / 10 counts, mean confidence and ECE. The candidates are:
   - the median (today);
   - the mode;
   - the rounded expected grade;
   - a P(10) cut-off t ∈ {0.50, 0.52, …, 0.70}.

   Report a tuned cut-off **cross-fitted**: for each outer fold, pick t on the other four folds' OOF predictions with the owner's objective (decision 1), then apply it to that fold. That gives an honest CV number for "tune t". Compare every rule with the median as **paired** differences per cert, with a bootstrap CI over card groups.
2. **The rule in code.**
   - Today the 9/10 boundary is "10 if P(10) > 0.5". The new rule changes only that boundary: `grade_int` = 10 only if P(10) > t (strict; see Implementation), otherwise the median of the rest, which is 9 in that case. The other grades keep the median.
   - `confidence` stays P(grade = `grade_int`).
   - Where t lives follows decision 2. With the default t = 0.5, `decode` behaves exactly as today. Old heads keep loading.
3. **Ship it** if decision 1's criterion is met. Today's `grade_head.*` is backed up to `data/backups/` first. Then `train_probe` is rerun, or `grade_head.json` is rewritten, depending on decision 2. The API is load-checked.
4. **Consistent reporting.** `train_probe.metrics` and `ml.eval` use the served rule, so CV and test numbers describe what the API returns.
5. **Copy.** The result card's caption says the highlighted grade is "the middle of this distribution" (`frontend/src/components/GradeResult.tsx:154`). With a cut-off above 0.5 that's no longer true for 9/10 calls, so reword it (draft in decision 3).
6. **Test number.** Record the test-split number once (decision 4).

## Out of scope

- The probes, the features, the folds and the soft grade (`grade`). The AUCs don't change.
- Cut-offs at other grade boundaries. Grades 8 and below are rare (8 recall is 22% under every rule) and too few to tune on.
- Cost-weighted rules, for example weighting a 9 called 10 as worse than a 10 called 9. That can come later if the owner wants it.
- The legacy model's hard rank (`inference.py:446-458`). It's a fallback, and it keeps the median.
- The API schema. `grade_int`, `confidence` and `rank_probs` keep their meaning and types.

## Acceptance criteria

- [x] The seed sweep (seeds 0–4, train+val only) is in `test-artifacts/eval/grade_decision_rule/` and summarised in this spec, including the cross-fitted cut-off. Nothing under `data/` was written by the measurement.
- [x] Rules are compared with the median as paired per-cert differences, with card-grouped bootstrap CIs.
- [x] Unit tests:
  - with the default t, `decode` matches today's output on random inputs;
  - each boundary case for t gives the right grade (P(10) just below, at and above t);
  - `confidence` is P(grade = `grade_int`) under the new rule;
  - an old `grade_head.json` without the new field loads and decodes as today (if decision 2 stores t in the head).
- [x] The served rule, `train_probe.metrics` and `ml.eval` all use the same `grade_int`; a test pins this.
- [x] If shipped: the old head is in `data/backups/`, the API loads the new head, and one real pair grades.
  - The result card's highlighted bar still equals the confidence.
  - The caption is true for the new rule.
- [x] The test-split number is recorded per decision 4, and nothing is chosen on it.
- [x] The fast suite and `ruff check .` pass. If the caption changes, `npx eslint src`, `npx tsc --noEmit`, `npm run build` and `npm test` also pass.

## Owner questions

1. **What should the rule aim for?** This sets how t is chosen and when to ship.
   - (a) **Don't lose accuracy.** Take the highest cut-off whose mean exact accuracy across seeds is within 1 pp of the median's, and whose MAE is no worse than the median's by more than the seed SD. From the table that's about 0.55: 9 recall about 35%.
   - (b) **Match the base rate.** Predict 9s and 10s about as often as they occur. From the table that's about 0.60: 9 recall about 43%, for about 2 pp of exact accuracy.
   - (c) **Keep the median.** Record the measurement and change nothing.
   - Recommended: (a). It's nearly free and makes the most common miss less frequent. (b) is the honest choice if users care more about not being told "10" for a 9 than about exact accuracy.
2. **Where t lives.**
   - (a) **In `grade_head.json`.** `train_probe` picks it from its own OOF predictions with the decision 1 objective and stores it, so a retrained head carries its own cut-off. `decode` reads it; a head without it uses 0.5.
   - (b) **A constant in `grade_head.py`,** set from this measurement. It's simpler, but a retrained head may want a different value and nothing re-checks it.
   - Recommended: (a). A cut-off belongs with the probabilities it was tuned for.
3. **The result card caption.** Draft: "The highlighted grade is the prediction. A 10 needs at least N% probability, so on a close call the highlighted grade can sit next to a taller bar." N comes from t. Or keep the caption more general: "…can sit next to a taller bar on a close call." Recommended: the general one, so the copy doesn't change when t is retrained.
4. **The test number.**
   - (a) Record it once in this spec next to cv_folds, as in [[cv-fold-stratification]].
   - (b) Write ADR 0003, since this changes what `grade_int` means.
   - Recommended: (b). Exact accuracy and recall on the headline numbers move, and an ADR is where that's explained.

## Owner decisions (2026-10-31)

1. **(a) Don't lose accuracy.** t is the highest cut-off on the grid whose mean exact accuracy across seeds 0–4 is within 1 pp of the median's and whose mean MAE is no worse than the median's by more than the median's seed-to-seed SD. Ship it if t > 0.5. The cross-fitted number is what's reported for "tune t"; if t comes out at 0.5, nothing changes in what's served.
2. **(a) In `grade_head.json`.** `train_probe` tunes t on its own OOF predictions with the decision 1 criterion and stores it. `decode` reads it, and a head without it uses 0.5. The criterion needs the seed SD, so `train_probe` runs `cross_validate` for seeds 0–4 when it tunes. Seed 0 is still the run reported as `cv_grouped`. With `--cv-folds 0`, t is 0.5. Shipping means backing up the current `grade_head.*` to `data/backups/`, then rerunning `train_probe`. Those are the only writes under `data/` that are approved.
3. **General caption:** "The highlighted grade is the prediction; on a close call it can sit next to a taller bar."
4. **ADR 0003** (`docs/adr/0003-…`) records the rule, the CV evidence and the test-split number once. Nothing is chosen on the test split.

## Facts vs assumptions

- **Fact:** `decode` is the only place the probe's `grade_int` is decided. `inference.py:502` serves it, `train_probe.metrics` reports it, and `ml.eval` reads it from the service.
- **Fact:** the table above. It was measured 2026-10-31 on train+val OOF predictions (`cross_validate`, seed 0, current code and features). The predictions are saved outside the repo at `C:\Users\User\AppData\Local\Temp\claude\C--Users-User-dev-poke-scan\0e1dd37f-3e40-4abe-bf3c-e778af635956\scratchpad\oof_seed0.npz`.
- **Fact:** the soft grade is the expected grade (1 + ΣP(grade > k)) and doesn't depend on the rule, so the AUCs, `grade` and `rank_probs` don't change.
- **Assumption:** the one-seed, in-sample numbers hold up across seeds and under cross-fitting. Their differences are about 11–40 certs out of 2,241, so this has to be checked.
  - **Checked 2026-10-31:** mostly. Seed 0 reproduces the table exactly. Across seeds, a cut-off of 0.54 is free and 0.60 costs ~1.5 pp exact. Under decision 1's 1 pp floor the criterion picks 0.58, not the ~0.55 the draft expected. Cross-fitted, tuning t costs 1.0 pp exact [−1.8, −0.1] and gains 11 pp 9 recall.
- **Assumption:** users prefer fewer "10" calls on 9s at a small cost in exact accuracy. That's decision 1.
  - Decided 2026-10-31 (decision 1, (a)).

## Implementation (2026-10-31)

- **The rule** (`backend/app/ml/grade_head.py`): `decode(cum_probs, cutoff=0.5)` sets the hard grade to `1 + #(P(grade > k) > 0.5 for k = 1..8) + (P(10) > cutoff)`. So a 10 needs P(10) **strictly above** t, and otherwise the grade is the median of the rest (9 whenever P(10) > 0.5 and the input is monotone, which the proportional-odds head guarantees). This deviates from the spec's "P(10) ≥ t" on purpose: with `>` the default t = 0.5 is today's `1 + #(P > 0.5)` for every input, including P(10) at exactly 0.5 and non-monotone inputs. With `≥` it would need a special case at 0.5. For t above 0.5, `>` and `≥` differ only when P(10) equals a grid value exactly. `confidence` is still P(grade = `grade_int`). The soft grade, `rank_probs` and the API schema are unchanged.
- **Where t lives** (decision 2): `grade_head.json` key `ten_cutoff` (`TEN_CUTOFF_KEY`). `grade_head.ten_cutoff(metadata)` / `GradeHead.ten_cutoff` returns it: 0.5 when the key is absent (old heads), and `ValueError` unless a present value is a finite number in [0.5, 1) (bools, strings, null, NaN and inf are refused).
- **Served path** (`backend/app/ml/inference.py`): `_load_probe` reads `head.ten_cutoff` right after the feature-spec check, so a bad value fails closed like the other head-metadata checks (503, `probe_load_error` "ValueError: ten_cutoff …"; with `ALLOW_LEGACY_MODEL=1` it falls back to legacy, as for any probe load error). `_predict_probe` decodes with it. `/api/health` shows `ten_cutoff` because it returns the head's metadata. The legacy CORN path keeps its own median rank, which is unchanged. CLAUDE.md's fail-closed list and probe-model paragraph say so.
- **`train_probe`** (`ml/training/train_probe.py`): with `--cv-folds N` (> 0) it runs `cross_validate` for seeds 0–4 (`CUTOFF_SEEDS`), and `tune_ten_cutoff` applies decision 1 over `CUTOFF_GRID` (0.50, 0.52, …, 0.70). The criterion: the highest t whose mean exact accuracy is at least the median's minus 0.01 and whose mean MAE is at most the median's plus the median's seed SD (ddof 1). 0.5 always qualifies. Seed 0 is still `cv_grouped`. `cv_grouped`, `test`, `test_seen_cards` / `test_unseen_cards` all use the tuned t (`metrics(y, cum, ten_cutoff)`), so they describe what the API returns. Because `cv_grouped` uses a t tuned on those same predictions, a comment points to the cross-fitted number. The head's metadata stores `ten_cutoff` and `ten_cutoff_tuning`: the criterion text, n, the number of seeds, per-seed exact / MAE for the median and the chosen t, the median's MAE SD, and the mean exact / MAE at every grid t. The same two keys go into `probe_report.json`. With `--cv-folds 0`, t is 0.5 and no tuning is stored. `train_probe` now runs CV five times (~7 min instead of ~1.5).
- **`ml.eval`** reads `grade_int` from the service, so it uses the head's t with no change. `ml/eval/resolution_pilot.py` (a past pilot's arms) keeps the median.
- **Frontend**: `GradeResult.tsx`'s caption is now "The highlighted grade is the prediction; on a close call it can sit next to a taller bar." (decision 3). `gradeProbs.test.ts` models `grade_int` with the cut-off (`apiGrade`) and adds a case where a cut-off of 0.6 calls a 9 next to a taller 10, with the highlighted value still equal to the confidence.
- **Tests** (+55 in the fast suite):
  - `backend/tests/test_grade_head.py`:
    - default t matches today's decode exactly (expected grade, grade, confidence, per-grade probabilities) on 2,000 random monotone and non-monotone inputs, some with values at exactly 0.5;
    - P(10) just below, at and just above t for t ∈ {0.5, 0.56, 0.6, 0.7} (at t is a 9);
    - grades below the 9/10 boundary keep the median;
    - `confidence` is P(`grade_int`) under the cut-off;
    - valid and invalid `ten_cutoff` values;
    - an old head without the key loads with t = 0.5 and decodes as today;
    - the key round-trips.
  - `backend/tests/test_inference.py`:
    - a bad `ten_cutoff` fails closed (8 values, and in the existing fail-closed / legacy-fallback matrix);
    - `test_served_grade_uses_the_heads_cutoff_like_metrics_and_eval` pins that the served `grade_int`, `decode` with the head's t, `train_probe.metrics` and `ml.eval.predict_all` agree, on a close call that the median calls 10 (t absent / 0.5 / 0.54 → 10, 0.56 / 0.6 → 9), and that `/api/health` metadata shows t.
  - `ml/tests/test_ten_cutoff.py`:
    - the criterion: the exact floor binds; the MAE ceiling binds with no seed spread; seed spread loosens it; never below 0.5;
    - `main()` tunes on seeds 0–4, stores t and the tuning, and reports `cv_grouped` / `test` with t;
    - `--cv-folds 0` keeps 0.5.
- **Measurement scripts** (`test-artifacts/eval/grade_decision_rule/`, gitignored):
  - `decision_rule_sweep.py`: CV seeds 0–4 → `oof_seeds.npz`, then the rules, the paired card-group bootstrap (2,000 resamples over 1,100 card groups) and the cross-fitting → `sweep.json` / `sweep.md`;
  - `code_check.py`: the shipped `decode` and `tune_ten_cutoff` against the script;
  - `compare_heads.py`;
  - `integrity.py` (copied from cv-fold-stratification).

## Results (2026-10-31)

### Seed sweep (train+val, n 2,241: 613 9s, 1,336 10s, 171 8s, 121 ≤7; 1,100 card groups)

`cross_validate`, 5 folds, seeds 0–4, the 2026-10-31 head's code. Features came from `region_feats.npz`, read with `save=False` (2,611 cached, 0 extracted). Seed 0's OOF predictions equal the preliminary `oof_seed0.npz` exactly (max |Δ| 0). Nothing under `data/` changed (`integrity_before.json` / `integrity_after_sweep.json`). Mean ± SD across seeds (ddof 1). The full table, including every grid t, per-seed rows and ECE, is in `test-artifacts/eval/grade_decision_rule/sweep.md` / `sweep.json`.

| Rule | Exact | MAE | QWK | ECE | 8 / 9 / 10 recall | Pred 9 / 10 | Mean conf |
|---|---|---|---|---|---|---|---|
| Median (today) | 0.6433 ± 0.0051 | 0.4640 ± 0.0063 | 0.5790 ± 0.0018 | 0.0375 | 0.22 / 0.281 / 0.911 | 379 / 1,712 | 0.633 |
| Mode | 0.6448 ± 0.0048 | 0.4742 ± 0.0036 | 0.5552 ± 0.0096 | 0.0385 | 0.21 / 0.218 / 0.948 | 290 / 1,821 | 0.637 |
| Round expected | 0.6259 ± 0.0045 | 0.4702 ± 0.0052 | 0.5986 ± 0.0030 | 0.0369 | 0.22 / 0.428 / 0.812 | 612 / 1,453 | 0.607 |
| P(10) > 0.52 | 0.6437 ± 0.0048 | 0.4616 ± 0.0056 | 0.5816 ± 0.0034 | 0.0372 | 0.22 / 0.310 / 0.899 | 418 / 1,672 | 0.630 |
| P(10) > 0.54 | 0.6428 ± 0.0048 | 0.4607 ± 0.0053 | 0.5853 ± 0.0027 | 0.0376 | 0.22 / 0.342 / 0.882 | 463 / 1,627 | 0.627 |
| P(10) > 0.56 | 0.6386 ± 0.0045 | 0.4631 ± 0.0055 | 0.5857 ± 0.0028 | 0.0372 | 0.22 / 0.369 / 0.863 | 510 / 1,581 | 0.623 |
| **P(10) > 0.58 (t)** | 0.6341 ± 0.0056 | 0.4654 ± 0.0059 | 0.5866 ± 0.0034 | 0.0373 | 0.22 / 0.403 / 0.840 | 567 / 1,523 | 0.618 |
| P(10) > 0.60 | 0.6287 ± 0.0052 | 0.4679 ± 0.0061 | 0.5888 ± 0.0064 | 0.0365 | 0.22 / 0.439 / 0.814 | 629 / 1,461 | 0.611 |
| P(10) > 0.62 | 0.6270 ± 0.0042 | 0.4671 ± 0.0055 | 0.5920 ± 0.0065 | 0.0362 | 0.22 / 0.487 / 0.789 | 697 / 1,393 | 0.602 |
| P(10) > 0.66 | 0.6080 ± 0.0074 | 0.4799 ± 0.0100 | 0.5913 ± 0.0083 | 0.0345 | 0.22 / 0.583 / 0.713 | 871 / 1,219 | 0.575 |
| P(10) > 0.70 | 0.5664 ± 0.0082 | 0.5144 ± 0.0110 | 0.5788 ± 0.0073 | 0.0334 | 0.22 / 0.671 / 0.603 | 1,089 / 1,001 | 0.535 |
| **Tune t, cross-fitted** | **0.6336 ± 0.0048** | **0.4665 ± 0.0052** | **0.5853 ± 0.0030** | 0.0372 | 0.22 / **0.391** / **0.845** | 551 / 1,539 | 0.619 |

- **Decision 1 → t = 0.58.** Floor: exact ≥ 0.6333 (median 0.6433 − 1 pp). Ceiling: MAE ≤ 0.4703 (median 0.4640 + SD 0.0063). t = 0.58 meets both (0.6341, 0.4654). 0.60 fails the exact floor (0.6287), and so does every higher t. The exact floor binds, not the MAE ceiling. t > 0.5, so it shipped.
- **Cross-fitted** (each seed's outer fold gets the t chosen by the same criterion on the other four folds' certs, using all five seeds' predictions there): the per-fold t was 0.56 or 0.58 (18 of 25 folds chose 0.58). Paired against the median per cert, averaged over seeds, with a 95% bootstrap over card groups (2,000 resamples):

| vs median | Certs changed | Δ exact | Δ MAE | Δ 9 recall | Δ 10 recall |
|---|---|---|---|---|---|
| Mode | 166 | +0.0015 [−0.0052, +0.0085] | +0.0102 [+0.0016, +0.0189] | −0.063 [−0.081, −0.045] | +0.036 [+0.028, +0.046] |
| Round expected | 306 | −0.0174 [−0.0291, −0.0055] | +0.0062 [−0.0069, +0.0189] | +0.148 [+0.122, +0.175] | −0.100 [−0.116, −0.084] |
| P(10) > 0.54 | 85 | −0.0004 [−0.0051, +0.0044] | −0.0033 [−0.0090, +0.0018] | +0.061 [+0.048, +0.076] | −0.029 [−0.036, −0.023] |
| P(10) > 0.58 | 188 | −0.0092 [−0.0184, +0.0005] | +0.0014 [−0.0088, +0.0110] | +0.123 [+0.103, +0.145] | −0.072 [−0.085, −0.059] |
| P(10) > 0.60 | 250 | −0.0145 [−0.0255, −0.0033] | +0.0039 [−0.0080, +0.0156] | +0.158 [+0.136, +0.184] | −0.097 [−0.114, −0.081] |
| **Tune t, cross-fitted** | 172 | **−0.0096 [−0.0181, −0.0009]** | **+0.0025 [−0.0071, +0.0113]** | **+0.110 [+0.092, +0.131]** | **−0.067 [−0.079, −0.055]** |

- Reading: the cut-off rule does what decision 1 asked. It costs about 1 pp of exact accuracy (the CI just excludes 0), costs no MAE beyond noise, and calls 11 pp more of the true 9s a 9, at 7 pp of 10 recall. The mode remains not worth it.
- `code_check.py`: the shipped `decode(cum, t)` equals the script's rule on every cert, seed and grid t (0 mismatches). `train_probe.tune_ten_cutoff` picks 0.58 on all of train+val and the same t as the script in all 25 cross-fitting folds.

### Ship (decisions 1 and 2)

- No other Python job was running. Backups: `data/backups/grade_head.pre-decision-rule-2026-10-31.npz` (sha256 `0f62a686996b5e40c7a58683256a599583ed5d19d914db4b9817af505617ba76`) and `.json` (`fb382fa147ebc1d1c018b1f6439ab9351e616cd6a32bf5448ae22cdd60e91c5d`). Both match the served head byte for byte, which is the cv-fold-stratification head.
- `python -m ml.training.train_probe` (log: `test-artifacts/eval/grade_decision_rule/train_probe.log`): 2,611 cached features, 0 extracted, no ONNX export line. CV seeds 0–4 took ~85 s each. It logged "P(10) cut-off: 0.58", with per-seed exact / MAE matching the sweep.
- **Only the metadata changed.** `grade_head.npz` is byte-identical (sha256 still `0f62a686…`). Every probe and meta-probe array is identical (`head_compare.json`). `grade_head.json` (now `84407d7e…`) gained `ten_cutoff` 0.58 and `ten_cutoff_tuning`. Its `cv_grouped` and `test` are now under the served rule, and `trained_at` / `git_commit` are new (2026-11-01T01:14:23Z, `ee34f3b`: the branch head; this work is uncommitted).
- **Load check** (`health.json`, `grade_76000010.txt`):
  - `/api/health`: `status ok`, `model_loaded` true, `ten_cutoff` 0.58, no `probe_load_error`; face guard loaded (side + front_upright), no `face_guard_load_error`.
  - `/api/grade` on cert 76000010 (a PSA 9, test split, `data/images`): HTTP 200, grade 9.46, **`grade_int` 9** (10 under the median), confidence 0.3423 = P(9) = 0.9186 − 0.5763, next to a taller P(10) 0.576. So the highlighted bar is the confidence, and the new caption describes this case.
  - The server was stopped and port 8000 is free.
- Integrity (`integrity_final.json`): `pokescan.sqlite` (sha256 `185221d6…`), `data/images` (5,275 files) and `data/crops` (5,260) are unchanged. In `data/models/` only `grade_head.json` changed. `data/backups/` gained only the two head backups.

| `train_probe` (`grade_head.json`) | n | MAE | exact | QWK | ECE | AUC ≤8 vs ≥9 | AUC 10 vs 9 |
|---|---|---|---|---|---|---|---|
| CV seed 0, median (cv_folds head) | 2241 | 0.4676 | 64.1% | 0.5767 | 0.0305 | 0.8514 | 0.7243 |
| CV seed 0, t 0.58 (in-sample for t) | 2241 | 0.4726 | 62.6% | 0.5812 | 0.0294 | 0.8514 | 0.7243 |
| Test (precrop features), median | 370 | 0.4378 | 66.5% | 0.4991 | 0.0524 | 0.8292 | 0.7823 |
| Test (precrop features), t 0.58 | 370 | 0.4514 | 64.6% | 0.4980 | 0.0559 | 0.8292 | 0.7823 |

### Test split end to end (decision 4: recorded once in ADR 0003; nothing chosen on it)

`python -m ml.eval.evaluate --split test --robustness-limit 100 --perturbations rotate_180,swap_faces --out-dir test-artifacts/eval/decision_rule`, CPU, the same face guard: "regressions vs baseline: none". Comparison: `test-artifacts/eval/grade_decision_rule/evaluate_vs_cv_folds.json`. ADR: `docs/adr/0003-grade-decision-rule.md`.

| n 375 | cv_folds (median) | decision_rule (t 0.58) |
|---|---|---|
| MAE [95% CI] | 0.435 [0.363, 0.509] | 0.456 [0.384, 0.533] |
| Exact [95% CI] | 66.7% [62.1, 71.7] | 64.0% [59.5, 69.1] |
| QWK [95% CI] | 0.506 [0.352, 0.647] | 0.499 [0.340, 0.637] |
| ECE | 0.049 | 0.051 |
| 9 / 10 recall | 36.5% / 94.8% | 47.8% / 84.0% |
| Predicted 9 / 10 | 76 / 284 | 114 / 246 |
| AUC ≤8 vs ≥9 / 10 vs 9 | 0.832 / 0.778 | 0.832 / 0.778 (soft grades identical) |
| `rotate_180` MAE / flip rate | 0.32 / 3% | 0.31 / 9% |
| `swap_faces` MAE / flip rate | 0.33 / 0% | 0.32 / 0% |
| `n_would_reject` card_not_found / same_side | 5 / 0 | 5 / 0 |

- All 38 changed grades go from 10 to 9: 13 true 9s now right, 23 true 10s now wrong, plus a PSA 8 and a PSA 5 one grade closer. Paired (per-cert bootstrap): exact −2.7 pp [−5.9, +0.5], MAE +0.021 [−0.008, +0.053]. That's more than the cross-fitted CV cost (−1.0 pp, +0.003), but the CIs cover it. It's recorded, not acted on.
- The rotation flip rate rose (3 → 9 of 100) with an identical soft shift (0.109): more certs sit near the new boundary.

### Checks

- Fast suite: 1613 passed, 1 skipped, 2 xfailed (was 1558; +55 new tests), 253 s.
- `ruff check .`: clean.
- Frontend: `npx eslint src`, `npx tsc --noEmit` and `npm run build` are clean; `npm test` 10 / 10 passed.
