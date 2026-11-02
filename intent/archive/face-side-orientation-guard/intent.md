# Face side and orientation guard

## Intent

Before the grade head runs, the probe model checks two things, using the DINOv2 features it already computes:
1. Is the "front" upload a card front and the "back" upload a card back?
2. Is each face upright?

Swapped uploads are swapped back, and upside-down faces are rotated, each with a note. This keeps the grade from shifting silently because of how the user took or uploaded the photos.

## Why

Measured 2026-10-23 with the served probe model (see [[cv-ml-improvement-plan]] item 6):

- **Swapped uploads** (8 test certs) move the soft grade by 0.7–2.9: e.g. 9.66 → 7.72, 9.62 → 8.01, 2.91 → 5.86. Nothing flags it.
- **Both faces upside down** (16 test certs, photos rotated 180°; detection still finds 98.3% of quads) shift the soft grade by −0.19 on average, max 0.63.
  - [[serve-input-normalization]]'s sideways retry lands about half of sideways photos upside down, so this case becomes more common once that spec ships.
- **The probes depend on position.** The corner probe concatenates TL/TR/BR/BL and the head concatenates [front, back] (`backend/app/ml/grade_head.py:53-56`, `features.py:61-78`), so neither swapping nor rotating is neutral.
- **Front/back is separable from features we already have.** A logistic probe on the cached CLS features separates fronts from backs with card-grouped 5-fold AUC 0.99995: 6 errors in 5,222 faces.
  - The HSV side classifier from ingest (`ml/data/side_classifier.py::back_score_image`) only reaches AUC 0.84 on dewarped crops, so don't reuse it.
- **No orientation data exists yet.** The feature cache holds upright crops only, so an upright/flipped probe needs features of 180°-rotated crops.

## Scope

### Guard module: `backend/app/ml/face_guard.py` (new, numpy only)

- `FaceGuard` holds linear logistic probes over one face's global features (CLS + mean, `GLOBAL_SLICE`, 2×D):
  - `side`: P(back), trained on fronts and backs in both orientations so it doesn't depend on orientation;
  - `front_upright`: P(flipped) for fronts;
  - `back_upright`: P(flipped) for backs.

  Each probe is a `(weight, bias)` pair with the scaler folded in, as in `grade_head.py`.
- Thresholds `side_threshold` and `flip_threshold` are stored with the probes.
- The metadata JSON sidecar lists which checks are enabled. A check is disabled if training didn't meet its gate (see Acceptance).
- Saved as `data/models/face_guard.npz` (no pickle) plus `face_guard.json`, holding the CV metrics and `feature_spec_hash` if [[probe-artifact-consistency]] has landed.
- `FaceGuard.load(path)` mirrors `GradeHead.load`.

### Training: `ml/training/train_face_guard.py` (new CLI)

1. Use the same rows as `train_probe.load_rows`, and the upright features from the existing cache. Read them; don't rewrite the existing cache.
2. Extract features of every crop rotated 180° (`np.rot90(rgb, 2)` of the stored precrop) with `build_region_model`. Cache them in `data/models/cache/region_feats_rot180.npz`, keyed and invalidated like the main cache (reuse [[probe-artifact-consistency]]'s keying if it has landed).
3. Fit each probe with `sklearn` `LogisticRegression` on standardised features, then fold the scaler into `(weight, bias)`.
4. Report card-grouped 5-fold CV (`StratifiedGroupKFold` on `card_groups`) on train+val: AUC, error rate at 0.5, and the false-positive rate on correctly placed, upright faces at the chosen threshold.
5. Choose each threshold as the lowest value in {0.9, 0.95, 0.99} whose out-of-fold FPR on clean faces is ≤ 0.2%.
6. Fit on train+val, report on test, and write the artifacts plus `test-artifacts/eval/face_guard_report.json`.

### Inference (`backend/app/ml/inference.py`, probe path only)

- Load `FaceGuard` if `face_guard.npz` exists next to the head. If it's absent, behaviour is unchanged.
- In `_predict_probe`, after extracting `feats` (2, n_regions, D):
  1. **Side check.** Compute P(back) for both rows.
     - If the front row reads as a back and the back row as a front (both beyond `side_threshold`), swap the feature rows and the crops, and add the note `"front and back photos looked swapped; graded them the other way round"`.
     - If both rows read as the same side with confidence, add the note `"both photos look like card backs"` (or fronts) and don't swap.
  2. **Orientation check.** On the possibly swapped rows, compute P(flipped) for each face with its probe. For each face over `flip_threshold`:
     - 2026-10-28 (tester finding): skipped for a same-side pair (step 1's "both photos look like" note, which stays): which probe applies to which face is unknown, and the front probe fires on 5.5% of upright backs.
     - rotate that face's crop image 180°;
     - re-run the extractor once on the (possibly rotated) pair, after the same `_jpeg_roundtrip` as today;
     - add the note `"front photo was upside down; rotated it"` (or back).
  3. Run the head on the final features.
- `GradePrediction` gains `notes: list[str] = []`. Its `crops` (from [[serve-input-normalization]]) reflect the swap and rotation, so centering applies front tolerances to the real front.
- `backend/app/api/grade.py` appends `pred.notes` to the response notes. The schema doesn't change.

### Evaluation (`ml/eval/evaluate.py`)

- Add a pair perturbation `swap_faces` (front ↔ back bytes), alongside the per-image `PERTURBATIONS`. `robustness()` runs it like the others. `rotate_180` already exists.
- Add a CLI option `--perturbations rotate_180,swap_faces`, a comma list passed to `run_evaluation(perturbations=...)`. Today the CLI always runs all 6 perturbations, about 30 min at 100 certs.

## Out of scope

- 90° orientation: [[serve-input-normalization]] guarantees portrait crops.
- Rejecting uploads, or adding schema fields: notes only. Rejection is [[cv-ml-improvement-plan]] open question 2.
- Detecting non-card images or other out-of-distribution input beyond the side check.
- The legacy CNN path.
- Retraining or changing the grade head.

## Acceptance criteria

- [x] **Before any code change**, record the baseline here:
  - run `python -m ml.eval.evaluate --split test --robustness-limit 100 --perturbations rotate_180,swap_faces --out-dir test-artifacts/eval/guard_before`, with `swap_faces` and `--perturbations` added first (that part is evaluation code only);
  - write down clean MAE and QWK, and the `rotate_180` and `swap_faces` values of `grade_flip_rate` and `mean_abs_soft_shift`.
  - 2026-10-28: evaluation code first: `PAIR_PERTURBATIONS["swap_faces"]` (exchanges the raw upload bytes, no re-encode), `ALL_PERTURBATIONS`, `parse_perturbations` (unknown names → `ValueError` naming them and the known ones; the CLI turns it into a usage error, exit 2) and `--perturbations`. The default still runs every perturbation, now including `swap_faces`. To allow the cert-by-cert comparison below, the report also gains `predictions` (cert, grade, `pred_int`, `pred_soft`, guard notes) and, per robustness run, the per-cert `pred_int` / `pred_soft` and `n_with_notes`. Tests: 9 new in `ml/tests/test_eval_metrics.py`.
  - Run 2026-10-28 on HEAD `d59e840` plus the evaluation change only (the served head is ADR 0002's, `feature_spec_hash` 428e7a366e2e, CPU provider). Report kept in `test-artifacts/eval/guard_before/report.json`; `ml/eval/baseline.json` not touched ("regressions vs baseline: none").

    | | n | MAE | QWK | exact | `grade_flip_rate` | `mean_abs_soft_shift` |
    |---|---|---|---|---|---|---|
    | clean | 375 | 0.435 | 0.491 | 66.9% | | |
    | `rotate_180` | 100 | 0.340 | | | 13.0% | 0.155 |
    | `swap_faces` | 100 | 1.130 | | | 85.0% | 1.050 |

    Clean latency p50 2,496 ms, p95 2,823 ms.
- [x] Fast suite: `python -m pytest -m "not slow and not visual and not model"` → 0 failed, 2 xfailed, passed = baseline + the new tests.
  - 2026-10-28: 561 passed (520 + 41 new: 23 in `backend/tests/test_face_guard.py`, 9 in `ml/tests/test_face_guard_training.py`, 9 in `ml/tests/test_eval_metrics.py`), 2 xfailed, 184 deselected, 131 s. The first run failed 2 existing tests that pin the exact probe metadata (`test_retest_a92911c`, `test_review_adversarial`): `face_guard` was reported even with no guard file. Now `face_guard` / `face_guard_load_error` appear only when a `face_guard.npz` exists, so with no guard the metadata is unchanged too. The existing tests were not edited. CLAUDE.md baseline updated to 561.
- [x] Unit tests (synthetic features and a hand-built `FaceGuard`, `tmp_path` only):
  - [x] Swapped rows get swapped back, with the note; the crops are swapped too.
    - `test_swapped_rows_are_swapped_back_with_crops`. `test_swap_then_flip_uses_the_swapped_rows`: the orientation check runs on the swapped rows, and the real front is the one rotated.
  - [x] Two fronts get a note and no swap.
    - `test_same_side_pair_gets_a_note_and_no_swap` (two fronts and two backs). `test_disabled_check_does_nothing`.
  - [x] A flipped face is rotated and the extractor is called exactly twice; an upright pair calls it once.
    - `test_flipped_face_is_rotated_and_extracted_again` (front and back): the second call sees `_jpeg_roundtrip` of the rotated crop, the head sees the second features, and the quad is re-ordered. `test_both_faces_flipped_extract_once_more`, `test_upright_pair_extracts_once`.
  - [x] With `face_guard.npz` absent, predictions are bit-identical to today's.
    - `test_no_guard_file_is_bit_identical_to_the_head`: loaded through the real probe loader, features that would trigger every check, every field equal to decoding the head directly, and no `face_guard` metadata.
  - [x] Probabilities just below the thresholds change nothing.
    - `test_just_below_thresholds_changes_nothing`: 1e-4 below in logit space, for each side condition and both flip probes. Every field equals the unguarded prediction; just above, the check acts.
  - [x] `FaceGuard` save/load round-trips exactly, with no pickle.
    - `test_save_load_round_trips_exactly_without_pickle` (`allow_pickle=False`, no object arrays).
  - Also (owner decision, artifact mismatch): `test_untrusted_guard_is_disabled_and_api_stays_ready`, with the sidecar missing, the hash missing, empty or mismatched, the sidecar unreadable or not a dict, a corrupt npz, or a wrong width. Each leaves the probe ready, logs a WARNING, sets `face_guard_load_error` and grades as with no guard. `test_health_reports_untrusted_guard_while_grading` checks the same through `/api/health` and `/api/grade` (200). `test_grade_response_includes_guard_notes`. Training: `ml/tests/test_face_guard_training.py` covers scaler folding, gates and disabled reasons, test never moving thresholds or CV, card-grouped folds, and the rot180 cache variant with `save=False`.
- [x] `python -m ml.training.train_face_guard`, card-grouped 5-fold CV on train+val, recorded in `face_guard_report.json`:
  - side probe error ≤ 0.3% and AUC ≥ 0.999 (the upright-only measurement was 0.11% / 0.99995);
  - each upright probe AUC ≥ 0.99, with clean-face FPR ≤ 0.2% at its threshold;
  - any upright probe that misses this gate is disabled in `face_guard.json`, and the reason is recorded in this spec.
  - 2026-10-28: `data/models` backed up first to `data/models/backup/pre-face-guard/`: the 8 top-level files (`card_features.*`, `grade_head.*`, `grade_model.*`, `training.log`), all SHA-256 identical to the originals (`checkpoints/` and `cache/` not copied). The run read all 2,611 upright feature rows from `cache/region_feats.npz` (0 extracted) and never wrote it: its SHA-256 is unchanged (`ba3f0eed…02261`). It extracted 2,611 × 2 rotated faces on the GPU into `cache/region_feats_rot180.npz` (keyed by `train_probe.extract_features`, variant `rot180`). Fit on 2,241 train+val certs, test 370. `LogisticRegression(C=1)` on standardised CLS + mean (1,536 dims); `StratifiedGroupKFold` on card groups, seed 0. Each check has its own threshold (the spec's single `flip_threshold` became `thresholds[check]`, since step 5 picks one per probe).

    | check | CV AUC | CV error at 0.5 | clean FPR at 0.9 / 0.95 / 0.99 | threshold | status |
    |---|---|---|---|---|---|
    | side (8,964 faces, both orientations) | 0.99999 | 0.19% (17) | 0.067% / 0.022% / 0.022% | 0.9 | enabled |
    | front_upright (4,482) | 0.99969 | 0.69% (31) | 0.357% / 0.223% / 0.179% | 0.99 | enabled |
    | back_upright (4,482) | 0.99804 | 0.22% (10) | 0.268% / 0.223% / 0.223% | (0.99 stored) | **disabled** |

    - Side, upright faces only: AUC 0.99999, error 0.18% (the 2026-10-23 measurement was 0.11%). As pairs at 0.9: 0 false swaps in 2,241, 99.5% of swapped pairs caught, 0.13% of clean pairs get a "both look like" note.
    - front_upright at 0.99 catches 94.4% of flipped fronts out of fold.
    - **back_upright disabled**, recorded in `face_guard.json` `disabled_reasons`: "no threshold in (0.9, 0.95, 0.99) has clean-face FPR <= 0.2% (0.9: 0.268%, 0.95: 0.223%, 0.99: 0.223%)". Its AUC passes (0.998 ≥ 0.99), but 5 of 2,241 upright backs are called flipped even at 0.99, one over the limit. The near point-symmetric back design is the suspected reason; the probe is stored but unused.
    - Test, reported once (thresholds as above): side AUC 1.0, error 0.14%, clean FPR 0, 99.2% of swapped pairs caught, 0 false swaps. front_upright AUC 0.9999, clean FPR 0 at 0.99, detection 95.1%. (back_upright: AUC 1.0, FPR 0, for information only; it stays disabled, because test doesn't choose.)
- [x] `python -m ml.eval.evaluate --split test --robustness-limit 100 --perturbations rotate_180,swap_faces --out-dir test-artifacts/eval/guard_after`: **met under the restated `rotate_180` gate, with the shift miss accepted (owner decision 2026-10-28, below)**
  - `rotate_180`: ~~`grade_flip_rate` ≤ 3% and `mean_abs_soft_shift` ≤ 0.05~~ restated 2026-10-28: `grade_flip_rate` no worse than a perfect check (the oracle run below, 5.0%): **4.0%, pass**. `mean_abs_soft_shift` 0.100 against ≤ 0.05: **missed and accepted**, because the back check is disabled; the oracle reaches 0.046;
  - `swap_faces`: `grade_flip_rate` ≤ 3%;
  - clean run: `pred_int` identical to `guard_before` on ≥ 99% of test certs, MAE within ±0.005, latency p50 up by ≤ 5%.
  - 2026-10-28, guard loaded with `side` + `front_upright` (back_upright disabled, see above). Report in `test-artifacts/eval/guard_after/report.json`. `rotate_180` missed both original gates; see the owner decision below. Nothing was tuned on test.

    | | before | after | gate | |
    |---|---|---|---|---|
    | clean MAE / QWK / exact (n 375) | 0.435 / 0.491 / 66.9% | 0.437 / 0.492 / 66.7% | MAE ±0.005 | pass (+0.003) |
    | clean `pred_int` identical | | 374 / 375 (99.7%) | ≥ 99% | pass |
    | clean latency p50 / p95 | 2,496 / 2,823 ms | 2,343 / 2,500 ms | p50 ≤ +5% | pass (−6%) |
    | `rotate_180` flip rate / soft shift (n 100) | 13.0% / 0.155 | **4.0% / 0.100** | ≤ 3% / ≤ 0.05 | **fail** |
    | `swap_faces` flip rate / soft shift (n 100) | 85.0% / 1.050 | 0.0% / 0.000 | ≤ 3% | pass |

    - Clean run: the side check fired on no clean pair. front_upright rotated 3 of 375 upright fronts (0.8%, against 0.18% out-of-fold on precrops and 0 on test precrops). One of them, cert 88001436 (grade 8), went 8 → 7: that is the 1 changed cert.
    - `swap_faces`: all 100 pairs swapped back, and the grades are identical to clean.
    - `rotate_180` breakdown, per cert: `test-artifacts/eval/guard_after/rot180_breakdown.json`, from a read-only diagnostic that reruns the 100 certs. Its predictions match the eval run on 100 / 100. The "oracle" also turns the back upright, i.e. a perfect back check.

      | certs | n | shift with guard | flips with guard | shift, oracle | flips, oracle |
      |---|---|---|---|---|---|
      | all | 100 | 0.100 | 4.0% | 0.046 | 5.0% |
      | front rotated by the guard | 94 | 0.086 | 3.2% | 0.035 | 4.3% |
      | front not rotated (P(flip) 0.21–0.989 < 0.99) | 6 | 0.324 | 16.7% | 0.226 | 16.7% |

    - Diagnosis:
      - The front check fixed 94 of 100 fronts.
      - The back left upside down is most of the remaining shift: on the 94 fixed certs it goes from 0.086 to 0.035 once the back is turned upright too. The disabled back probe would have caught every one of these backs: P(flip) ≥ 0.998 on all 100. It is disabled because of false positives on upright backs (CV FPR 0.223% at 0.99, against a 0.2% limit), not because it misses flipped ones.
      - Even with a perfect back check, the flip rate is 5.0%. Re-detecting a 180°-rotated photo crops the card a pixel or two differently (the spec's second assumption), and certs sitting near a grade boundary flip on that. The median shift with the oracle is 0.016. So the 3% flip gate looks unreachable from the guard alone, while the 0.05 shift gate would pass with a working back check.
- [x] `python -m pytest -m model` passes.
  - 2026-10-28: 13 passed, 734 deselected, 958 s, with the guard loaded (incl. `test_model_performance` against `ml/eval/baseline.json`).
- [x] `ruff check .` exits 0.
  - 2026-10-28.

## Facts vs assumptions

- **Fact:** the swap and rotation shifts, and the front/back separability from cached CLS features (2026-10-23). The rotation numbers are for both faces rotated together.
- **Assumption:** DINOv2 global features separate upright from flipped faces well, at AUC ≥ 0.99. This is likely for fronts (text, set symbol, HP line). It's less certain for backs, whose design is close to point-symmetric apart from the logo; the gate above covers that.
- **Assumption:** the crop of a 180°-rotated photo is close enough to the 180° rotation of the upright crop that features from rotated precrops are representative. Precrops are byte-identical to the detector output, but the rotated detection can snap differently by a pixel or two.
- **Dependency:** [[serve-input-normalization]] (`GradePrediction.crops`). [[probe-artifact-consistency]] is optional; if it has landed, reuse its cache keying and fingerprint.
- **Compute:** one GPU feature extraction of about 5.2k rotated faces, plus about 1.4 s extra CPU latency only for requests where a face is flipped ([[cv-ml-improvement-plan]] open question 6).

## Owner decisions (2026-10-27)

- **Compute (Q6):** approved: one GPU extraction of the ~5.2k 180°-rotated crops into `data/models/cache/region_feats_rot180.npz`, and a new `data/models/face_guard.npz` + `.json`, after backing up `data/models`.
- **Notes (Q7):** approved as written in Scope; the schema doesn't change.
- **Artifact mismatch:** `face_guard.json` records `feature_spec_hash`. If it is missing, unreadable or differs from the code's `feature_spec_hash()`, the guard is **disabled, not the API**: log a WARNING, report `face_guard_load_error` in `/api/health` metadata, and grade as if no guard existed. A stale guard making wrong swaps is the risk; an absent guard changes nothing.
- **Gates:** a check that misses its gate (likely the back upright probe, as card backs are nearly point-symmetric) is disabled and recorded; ship the checks that pass.
- **Latency:** the extra extraction (~1.4 s CPU) on requests with a flipped face is accepted.
- **"Both photos look like backs/fronts":** note only, no rejection. Rejection is decided with [[cv-ml-improvement-plan]] open question 2 in a later round.
- **Baseline:** measured on the card-grouped test split from [[stable-grouped-split]] phase 2 (ADR 0002), after it lands; start only then.
- **Delivery:** one PR with phase 2 and `smallfix.md` §11 / §12.

## Owner decisions (2026-10-28, after guard_after)

- **Ship as is:** `side` and `front_upright` enabled, `back_upright` disabled. Swapped uploads go from 85% to 0% changed grades, and the `rotate_180` shift drops from 0.155 to 0.100.
- **The back-check gate stays at 0.2%.** It misses on CV (0.223%), and on served photos the front check's false rotations ran about 4× its CV rate (3 of 375, 0.8%, against 0.18%), so a loosened back check would likely misfire on about 1% of real backs.
- **`rotate_180` gates restated.** The original 3% flip gate mixed orientation with re-detection noise: a perfect check (both faces turned upright) still changes 5.0% of grades, because the rotated photo is cropped a pixel or two differently. The flip gate is now "no worse than the oracle run". The 0.05 shift gate is kept as the target and recorded as missed (0.100), owing to the disabled back check.
- **Follow-up:** [[face-guard-serve-path-features]] retrains the guard on features from the served path, so served false-rotation rates match CV, and reconsiders the back check there.
