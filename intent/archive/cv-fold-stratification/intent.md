# CV folds stratified on 9 vs 10

## Intent

Make the probe's card-grouped CV folds balance PSA 9 against PSA 10, as `probe._folds`' comment says it should. Then re-measure, so that later 9-vs-10 work (a decision rule, more data) is chosen on folds that don't swing from one to the next. Found by the [[full-resolution-pilot]] re-test (2026-10-31, `smallfix.md`).

## Why

`ml/training/probe.py:103-106`:

```python
def _folds(y, groups, n, seed):
    # Stratify on grade with <=8 pooled: the low grades are too rare to spread.
    skf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), np.minimum(y, 8), groups))
```

`np.minimum(y, 8)` does the opposite of the comment: it keeps grades 1–7 apart and pools 8, 9 and 10 into one class. So the folds are balanced on the rare low grades and not on 9 vs 10, the split the model is weakest at.

Measured on the 2,241 train+val certs with a usable precrop (613 9s, 1,336 10s, 171 8s, 121 ≤7), `StratifiedGroupKFold` on `card_groups`, seeds 0–4:

| Stratify on | Outer 5 folds: 9s / (9s+10s) | 8s per fold | ≤7 per fold | Inner 4 folds: 9s / (9s+10s) | 8s per fold |
|---|---|---|---|---|---|
| today, `np.minimum(y, 8)` | 0.25–0.36 | 25–44 | 23–25 | 0.28–0.36 | 31–61 |
| `np.maximum(y, 8)` (≤8 pooled, as the comment says) | 0.31–0.32 | 27–40 | 18–32 | 0.31–0.32 | 35–54 |
| `np.maximum(y, 7)` (≤7 pooled) | 0.31–0.32 | 34–35 | 24–25 | 0.31–0.32 | 42–43 |

`_folds` is used in two places, so the fix changes both what CV reports and the served head:
- **`cross_validate`** (5 folds, seed 0): every card-grouped CV number. That includes `train_probe`'s `cv_grouped` and the resolution pilot's arms.
- **`oof_region_grades`** (`INNER_FOLDS` = 4, seed 1): the out-of-fold region grades the meta probe is fitted on. So `fit_grade_head`, and the `grade_head.npz` that is served, change too.

The face guard is not affected: `train_face_guard.py` builds its own `StratifiedGroupKFold` on its side / flip labels.

## Scope

- Change `_folds` to the stratification key chosen in decision 1, and fix its comment to match.
- A fast test that the folds balance 9 vs 10 (and the other strata) on synthetic grouped data.
- A seed-sweep measurement of CV under the old and new key, run on train+val only.
- Retraining and shipping the head per decision 2, and evaluating it on the test split per decision 3.
- Results in this spec; the baseline table in [[cv-ml-improvement-plan]]; the `smallfix.md` entry ticked.

## Out of scope

- Any change to the probes, the features, `INNER_FOLDS`, the seeds or the number of folds.
- The face guard, its folds and its thresholds.
- Re-running past reports (the resolution pilot, guard_noquad). Their arm comparisons used the same folds for every arm, so the fold imbalance doesn't bias them.
- A 9-vs-10 decision rule. That is the follow-up this spec prepares for.

## Acceptance criteria

- [x] `_folds` stratifies on the key from decision 1, and its comment says what it does.
- [x] A fast test on synthetic grouped data (about 30% 9s among the 9s and 10s, plus some 8s and lower grades) shows every outer and inner fold's 9-share within ±0.03 of the overall share. The existing fold tests still pass: deterministic, out-of-fold, a rare grade in one group, and the spy in `test_review_adversarial.py`.
- [x] **Seed sweep, train+val only, nothing written under `data/`:** `cross_validate` with seeds 0–4 under the old and the new key. For each, report the mean and SD across seeds of MAE, QWK, ECE, AUC ≤8 vs ≥9, AUC 10 vs 9 and 9-recall, plus the per-fold spread of AUC 10 vs 9. Output goes to `test-artifacts/eval/cv_fold_stratification/`.
- [x] Per decision 2: the current `grade_head.npz` / `grade_head.json` are backed up to `data/backups/`, `python -m ml.training.train_probe` is run (it reads the feature cache, with no ONNX re-export because the feature spec is unchanged), and the new head loads: `/api/health` ready, and the face guard still loads.
- [x] Per decision 3: the test-split numbers for the new head, next to the current head's, with nothing chosen on them.
- [x] The fast suite is green and `ruff check .` is clean. The [[cv-ml-improvement-plan]] baseline table is updated, the `smallfix.md` entry is ticked, and this spec's Results section is filled in.

## Owner questions

1. **The stratification key.**
   - (a) `np.maximum(y, 7)`: strata ≤7, 8, 9 and 10. It balances 9 vs 10 and also keeps 8s (34–35 per outer fold) and ≤7s (24–25) even, which matters for AUC ≤8 vs ≥9. There are enough 8s (171) and ≤7s (121) to spread across 5 folds.
   - (b) `np.maximum(y, 8)`: strata ≤8, 9 and 10, which is what the comment literally says. It balances 9 vs 10 but lets the ≤7s swing from 18 to 32 per fold.
   - Recommended: **(a)**.
2. **Ship the retrained head?** The served head changes slightly, because its meta probe is fitted on inner-fold OOF grades.
   - (a) Yes: back up the current head to `data/backups/`, retrain, and serve the new head. It's held back only if the seed sweep shows a worse mean AUC 10 vs 9 or MAE beyond the seed-to-seed SD.
   - (b) Fix the code only and keep serving the old head until the next retrain. The served head then no longer matches what the code would train.
   - Recommended: **(a)**. It writes to `data/models/` and `data/backups/`, which needs your approval.
3. **The test split.** `train_probe` writes test metrics into `grade_head.json` on every run, and the end-to-end number comes from `ml.eval.evaluate`.
   - (a) Record both once in this spec: the `train_probe` test metrics, and `ml.eval.evaluate --split test --robustness-limit 100 --perturbations rotate_180,swap_faces` as `test-artifacts/eval/cv_folds`, compared with guard_noquad. Nothing is chosen on them, and there's no new ADR, since this is a bug fix and not a modelling decision.
   - (b) The same, plus an ADR 0003 that records it as the new baseline.
   - Recommended: **(a)**.

## Owner decisions (2026-10-31)

1. **(a) `np.maximum(y, 7)`:** strata ≤7, 8, 9 and 10.
2. **(a) Back up, retrain, ship.** The current `grade_head.npz` / `grade_head.json` go to `data/backups/` first, and the writes to `data/models/` and `data/backups/` are approved. The new head is held back only if the seed sweep shows mean AUC 10 vs 9 or MAE worse by more than the seed-to-seed SD.
3. **(a) Record in this spec:** `train_probe`'s test metrics and one `ml.eval.evaluate` run (`test-artifacts/eval/cv_folds`) next to guard_noquad. Nothing is chosen on them, and there's no ADR.

## Implementation (2026-10-31)

- `ml/training/probe.py:103-107`: `_folds` stratifies on `np.maximum(y, 7)` (strata ≤7, 8, 9, 10), and the comment says so. Nothing else in the probe changed (`INNER_FOLDS` 4, seeds 0 / 1, 5 outer folds). Side effect: sklearn's "least populated class has only 1 member" warning is gone, since grades below 7 are pooled.
- `ml/tests/test_probe.py:99-137`: `test_folds_balance_9_vs_10_and_spread_8s_and_low_grades` (fast, 3 synthetic datasets of ~1,800 certs on 900 cards, 1–4 certs a card, ~30% 9s among 9s+10s). It checks the outer 5 folds (seed from `cross_validate`'s default), the inner 4 folds inside every outer training set and the inner folds over all the data (seed from `oof_region_grades`' default): every fold's 9-share is within ±0.03 of its parent's, and its 8s and ≤7s within ±2 of an even share. Under the old key it fails on all 3 datasets (9-share off by up to 0.08). No existing test pinned the old key; the existing fold tests pass unchanged.
- The seed sweep is a measurement script, not kept in `ml/`: `test-artifacts/eval/cv_fold_stratification/cv_seed_sweep.py` (with `integrity.py` and `compare_eval.py`). It monkeypatches the old key into `probe._folds`, so both the outer and the inner folds change.

## Results (2026-10-31)

### Seed sweep (train+val, n 2,241: 613 9s, 1,336 10s, 171 8s, 121 ≤7)

`cross_validate`, 5 folds, seeds 0–4 (the seed moves the outer folds; the inner folds stay at seed 1, as in production). Features from `region_feats.npz` read with `save=False` (2,611 cached, 0 to extract, no missing or extra entries); nothing written under `data/`. SD is across the 5 seeds (ddof 1). Old seed 0 reproduces the 2026-10-28 head's `cv_grouped` exactly, and new seed 0 is the `cv_grouped` that `train_probe` recorded below. Output: `test-artifacts/eval/cv_fold_stratification/sweep.json` / `sweep.md`.

| | old `np.minimum(y, 8)` | new `np.maximum(y, 7)` |
|---|---|---|
| MAE | 0.4659 ± 0.0031 | 0.4640 ± 0.0063 |
| QWK | 0.5810 ± 0.0046 | 0.5790 ± 0.0018 |
| ECE | 0.0391 ± 0.0068 | 0.0375 ± 0.0086 |
| AUC ≤8 vs ≥9 | 0.8534 ± 0.0016 | 0.8554 ± 0.0036 |
| AUC 10 vs 9 | 0.7226 ± 0.0076 | 0.7254 ± 0.0060 |
| 9-recall | 0.278 ± 0.011 | 0.281 ± 0.013 |
| per-fold AUC 10 vs 9, min–max over 25 folds | 0.663–0.788 | 0.668–0.780 |
| per-fold AUC 10 vs 9, mean within-seed SD / range | 0.026 / 0.070 | 0.030 / 0.076 |
| outer-fold 9-share, min–max | 0.249–0.363 | 0.313–0.315 |
| outer-fold 8s / ≤7s, min–max | 25–44 / 23–25 | 34–35 / 24–25 |

- **Gate (decision 2): ship.** Mean AUC 10 vs 9 is 0.0028 higher under the new key (the limit was a drop of 0.0076, the old key's seed SD) and mean MAE 0.0019 lower (limit: a rise of 0.0031). Decision 2 doesn't say which key's SD to use; both differences go the good way, so the call is the same with either.
- The folds are now balanced as intended, but **the fold-to-fold swing of AUC 10 vs 9 didn't shrink**: the within-seed SD is 0.030 vs 0.026, and the seed-to-seed SDs are mixed (AUC 10 vs 9 and QWK lower, MAE and ECE higher). So the swing comes from ~120 9s per fold, not from the 9-share imbalance. For 9-vs-10 work, compare rules on pooled OOF predictions over the same folds (paired), not on single folds.
- Per-seed rows and per-fold AUCs: `sweep.md`.

### Ship (decision 2)

- Backups: `data/backups/grade_head.pre-cv-folds-2026-10-31.npz` (sha256 `d0443cb59996a5d10951bd5848c1fc60054bf19d48c23538c9859f7187187ab3`) and `data/backups/grade_head.pre-cv-folds-2026-10-31.json` (`4179ab928e5ac826a5053c4187cd6687ea15c6e0f9ebe0cf5aa6f8db2f5d9787`), matching the originals byte for byte.
- `python -m ml.training.train_probe` (log: `test-artifacts/eval/cv_fold_stratification/train_probe.log`): 2,611 cached features, 0 extracted, no ONNX export line (the sidecar's `feature_spec_hash` matches, `428e7a366e2e`). It wrote `grade_head.npz` (sha256 `0f62a686…`), `grade_head.json` (`fb382fa1…`), `trained_at` 2026-10-31T21:21:45+00:00, `git_commit` 8862bbb (the branch head; the fix was uncommitted at training time), and `test-artifacts/eval/probe_report.json`.
- Load check: `/api/health` `status ok`, `model_loaded` true, the new `trained_at`, no `probe_load_error`; face guard `loaded` true, `enabled_checks` side + front_upright, no `face_guard_load_error` (`health.json`). `/api/grade` on cert 76000010 (a PSA 9, test split): HTTP 200, grade 9.46, `grade_int` 10, confidence 0.576, corners 9.6 / edges 9.2 / surface 9.3, centering not graded (note). The server was stopped by PID and port 8000 is free.

| `train_probe` (`grade_head.json`) | n | MAE | exact | QWK | ECE | AUC ≤8 vs ≥9 | AUC 10 vs 9 |
|---|---|---|---|---|---|---|---|
| CV, 2026-10-28 head (old key) | 2241 | 0.4694 | 64.0% | 0.5829 | 0.0305 | 0.8529 | 0.7209 |
| CV, new head | 2241 | 0.4676 | 64.1% | 0.5767 | 0.0305 | 0.8514 | 0.7243 |
| Test (precrop features), 2026-10-28 head | 370 | 0.4405 | 66.5% | 0.4852 | 0.0543 | 0.8344 | 0.7888 |
| Test (precrop features), new head | 370 | 0.4378 | 66.5% | 0.4991 | 0.0524 | 0.8292 | 0.7823 |

### Test split end to end (decision 3, recorded only; nothing chosen on it)

`python -m ml.eval.evaluate --split test --robustness-limit 100 --perturbations rotate_180,swap_faces --out-dir test-artifacts/eval/cv_folds`, same face guard as guard_noquad, CPU provider, "regressions vs baseline: none". Comparison: `test-artifacts/eval/cv_fold_stratification/evaluate_vs_guard_noquad.json` (AUCs and 9-recall computed from the reports' per-cert predictions).

| | guard_noquad (2026-10-28 head) | cv_folds (new head) |
|---|---|---|
| clean n | 375 | 375 |
| clean MAE / QWK / exact | 0.435 / 0.491 / 66.9% | 0.435 / 0.506 / 66.7% |
| clean MAE 95% CI / QWK 95% CI | [0.360, 0.509] / [0.350, 0.617] | [0.363, 0.509] / [0.352, 0.647] |
| clean ECE | 0.053 | 0.049 |
| clean AUC ≤8 vs ≥9 / AUC 10 vs 9 | 0.838 / 0.785 | 0.832 / 0.778 |
| recall 9 / recall 10 | 36.5% / 94.8% | 36.5% / 94.8% |
| clean `pred_int` changed vs guard_noquad | | 6 (2 up, 4 down) |
| `rotate_180` MAE (Δ) / flip rate / soft shift / with notes (n 100) | 0.31 (−0.02) / 4.0% / 0.0990 / 93 | 0.32 (−0.01) / 3.0% / 0.1092 / 93 |
| `swap_faces` MAE (Δ) / flip rate / soft shift / with notes (n 100) | 0.33 (0.00) / 0.0% / 0.000 / 100 | 0.33 (0.00) / 0.0% / 0.000 / 100 |
| `n_would_reject` card_not_found / same_side | 5 / 0 | 5 / 0 |
| latency p50 | 2734 ms | 2392 ms |

- The 6 changed grades (true: old → new): 78000624 10: 10 → 9, 82000563 10: 9 → 10, 82000605 9: 10 → 9, 88000944 3: 5 → 4, 88001436 8: 8 → 7, 88001644 9: 9 → 10. Three moved closer, three further, so MAE is unchanged. Mean |Δ soft grade| 0.030, max 0.578.
- The face guard's outputs are identical (every clean and perturbed P(back) / P(flip) and note), as expected: the guard doesn't use the head. `rotate_180` has 1 `pred_int` change, `swap_faces` none.

### Integrity

Snapshots before the sweep and after training (`integrity_before.json`, `integrity_after_train.json`, `integrity_final.json` in `test-artifacts/eval/cv_fold_stratification/`): `pokescan.sqlite` sha256 `185221d6a61faa930ebaee502bae63eacbe34691f67e4ba5a237d16bf1399ffb` (size and mtime unchanged), `data/images` 5,275 files, `data/crops` 5,260 files, all unchanged. In `data/models/` only `grade_head.npz` / `.json` changed; `card_features.onnx` / `.json`, `face_guard.npz` / `.json` and every cache file have the same sha256. `data/backups/` gained only the two head backups.

### Checks

Fast suite: 1554 passed, 1 skipped, 2 xfailed (was 1551; +3 for the new parametrized test), 248 s. `ruff check .`: clean.

## Facts vs assumptions

- **Fact:** the fold balance table above, measured 2026-10-31 on the current DB (read-only) with `card_groups` and `StratifiedGroupKFold(shuffle=True)`, seeds 0–4.
- **Fact:** the current head (`grade_head.json`, trained 2026-10-28) records `cv_grouped` AUC 10 vs 9 0.7209, MAE 0.4694, QWK 0.5829 and ECE 0.0305 over 2,241 certs, plus `test` on n = 370 precrop features.
- **Fact:** the face guard's folds don't use `_folds`.
- **Assumption:** the fix mostly lowers seed-to-seed and fold-to-fold variance and barely moves mean CV numbers. The seed sweep measures this.
  - **Checked 2026-10-31:** half right. Mean CV numbers moved by less than one seed SD. Variance didn't drop: the per-fold AUC 10 vs 9 SD is 0.030 vs 0.026, and the seed-to-seed SDs are mixed.
- **Assumption:** the retrained head grades within noise of the current one. The test-split comparison (decision 3) shows how far it moves.
  - **Checked 2026-10-31:** yes. 6 of 375 served grades change, clean MAE is identical (0.435), and every other metric moves well inside its bootstrap CI.
