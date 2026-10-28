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

- [ ] **Before any code change**, record the baseline here:
  - run `python -m ml.eval.evaluate --split test --robustness-limit 100 --perturbations rotate_180,swap_faces --out-dir test-artifacts/eval/guard_before`, with `swap_faces` and `--perturbations` added first (that part is evaluation code only);
  - write down clean MAE and QWK, and the `rotate_180` and `swap_faces` values of `grade_flip_rate` and `mean_abs_soft_shift`.
- [ ] Fast suite: `python -m pytest -m "not slow and not visual and not model"` → 0 failed, 2 xfailed, passed = baseline + the new tests.
- [ ] Unit tests (synthetic features and a hand-built `FaceGuard`, `tmp_path` only):
  - [ ] Swapped rows get swapped back, with the note; the crops are swapped too.
  - [ ] Two fronts get a note and no swap.
  - [ ] A flipped face is rotated and the extractor is called exactly twice; an upright pair calls it once.
  - [ ] With `face_guard.npz` absent, predictions are bit-identical to today's.
  - [ ] Probabilities just below the thresholds change nothing.
  - [ ] `FaceGuard` save/load round-trips exactly, with no pickle.
- [ ] `python -m ml.training.train_face_guard`, card-grouped 5-fold CV on train+val, recorded in `face_guard_report.json`:
  - side probe error ≤ 0.3% and AUC ≥ 0.999 (the upright-only measurement was 0.11% / 0.99995);
  - each upright probe AUC ≥ 0.99, with clean-face FPR ≤ 0.2% at its threshold;
  - any upright probe that misses this gate is disabled in `face_guard.json`, and the reason is recorded in this spec.
- [ ] `python -m ml.eval.evaluate --split test --robustness-limit 100 --perturbations rotate_180,swap_faces --out-dir test-artifacts/eval/guard_after`:
  - `rotate_180`: `grade_flip_rate` ≤ 3% and `mean_abs_soft_shift` ≤ 0.05;
  - `swap_faces`: `grade_flip_rate` ≤ 3%;
  - clean run: `pred_int` identical to `guard_before` on ≥ 99% of test certs, MAE within ±0.005, latency p50 up by ≤ 5%.
- [ ] `python -m pytest -m model` passes.
- [ ] `ruff check .` exits 0.

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
