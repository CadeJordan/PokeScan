# Face guard on served-path features

## Intent

Train the [[face-side-orientation-guard]] probes on features computed the way `/api/grade` computes them, so the guard's false-rotation rate on real uploads matches what cross-validation predicts. Then reconsider the disabled back upright check.

## Why

Measured 2026-10-28 (guard_after on the card-grouped test split, ADR 0002):

- **CV under-predicts served false rotations.** `front_upright` at 0.99 rotated 3 of 375 upright served fronts (0.8%). Out-of-fold on precrops it was 0.18%, and on test precrops 0. One of the 3 changed a grade (cert 88001436, 8 → 7), and all 3 got a wrong "front photo was upside down" note.
- **The probes are trained on precrop features, not served ones.** The served path decodes the upload, applies the EXIF transpose and the 1024 px cap, re-detects, and runs `_jpeg_roundtrip` before extraction. The precrop cache skips the re-detection and the cap.
- **The back check is off.** `back_upright` has CV AUC 0.998 but clean-face FPR 0.223% at 0.99, against a 0.2% limit, so it was disabled. On the 100 `rotate_180` test certs, P(flip) was ≥ 0.998 for every rotated back. A perfect back check would bring the `rotate_180` shift from 0.100 to 0.046.

## Scope

- Extract guard training features through the served preprocessing path, upright and 180°-rotated, for train+val. Cache them beside the existing caches, keyed the same way.
- Refit all three probes and re-run the gates on those features. Thresholds are chosen on card-grouped CV only.
- Re-measure guard_after, comparing against the guard_before and guard_after reports from 2026-10-28.

## Out of scope

- Changing the grade head or its features.
- Rejecting uploads ([[cv-ml-improvement-plan]] open question 2).
- 90° orientation.

## Acceptance criteria

- [x] Guard training features for train+val (upright and 180°-rotated) come from the same preprocessing `/api/grade` applies to an upload (decode, EXIF transpose, 1024 px cap, detection, `_jpeg_roundtrip`), through shared code, not a copy. The source images are the original DB photos the eval harness feeds the service.
  - 2026-10-29: `inference.py` now names the two steps `predict` runs on each upload, `detect_face` (was `_detect_face`) and `extractor_input` (`_jpeg_roundtrip` of the crop), and `_predict_probe` calls them. `served_extractor_input(bytes)` chains them, and `train_face_guard --features served` (the new default) reads `front_path` / `back_path` from the DB (read-only) through it. The rotated face is the photo rotated before decoding with `evaluate.perturb_image("rotate_180", ...)`, which `apply_perturbation` now also uses: cv2 decode, `cv2.rotate` 180°, re-encode as JPEG q95, exactly as the eval's `rotate_180`. The crop is never rotated. The rows are `train_probe.load_rows`' 2,611 certs (2,241 train+val, 370 test). The 14 train+val certs and 5 test certs with no usable precrop are not in it (see Results).
- [x] The served-path features are cached beside the existing caches, with the same keying and invalidation as `train_probe.extract_features`. `region_feats.npz` and `region_feats_rot180.npz` are never rewritten.
  - `data/models/cache/region_feats_served.npz` (variant `served`) and `region_feats_served_rot180.npz` (`served_rot180`), written by `train_probe.extract_features` itself. It gained `path_keys` (which DB columns the key and the file signature use; default the precrop columns), `load` (path → RGB) and `workers`. The default call is unchanged. The served path reads both precrop caches with `save=False` and never writes them. Their SHA-256 is unchanged after both runs (`ba3f0eed…02261`, `5a657bdc…4c7c8`). A served cache is not invalidated by a change to the detection code, which is documented in the module docstring and CLAUDE.md.
- [x] A test (on synthetic images, `tmp_path`) shows that the training-side served-path features of an upload equal what `GradeInferenceService` extracts for that upload, within float tolerance.
  - `test_served_features_equal_what_the_service_extracts`: the images fed to the extractor are bit-identical, upright and `rotate_180`, and the features agree to float16 tolerance. One cert's photos are 1,500 px with EXIF orientation 6, so the transpose and the cap both act. `model`-marked `test_served_features_match_the_real_service_on_db_photos`: torch training features vs the running ONNX service on 2 real DB certs, upright and rotated, relative error < 2e-3.
- [x] Before refitting: out-of-fold served-path clean-face FPR of the current probes, on train+val, is reported, which tests the spec's assumption that the CV gap comes from the feature difference.
  - Reported in `transfer_from_precrop` (`test-artifacts/eval/guard_served/pre_refit_report.json`). **The assumption is false**; see Results.
- [x] All three probes are refit on served-path features. Thresholds and gates are chosen on card-grouped CV only, with `FPR_LIMIT` still 0.2% and `THRESHOLD_GRID` unchanged. The test split is used for one final number.
- [x] `back_upright` is enabled only if it passes that gate; otherwise it stays disabled with its reason in the sidecar.
  - Still disabled: 5 of 2,241 upright backs at 0.99 (0.223%).
- [x] `face_guard.json` records which feature source the probes were trained on.
  - `feature_source: "served"`, plus `feature_source_note` and `feature_caches`. `--features precrop` still trains the 2026-10-28 way and records `"precrop"`.
- [x] guard_after is re-measured on the test split (clean false rotations among the 375 fronts and backs, `rotate_180` and `swap_faces` robustness) and compared with the 2026-10-28 guard_before / guard_after reports. The report includes per-cert `p_flip` / `p_back` for every clean cert the guard changed.
  - `test-artifacts/eval/guard_served/report.json`, plus `comparison.json` beside it. `GradePrediction.guard_decision` (internal, `compare=False`, not in the API) carries the guard's probabilities. `ml.eval.evaluate` now writes `p_back` / `p_flip` for every prediction the guard scored, and per cert in each robustness run (`guard`, `notes`).
- [x] A rotation stays a note only; `confidence` and the API schema are unchanged.
  - No change to `confidence`, the head, `GradeResponse` or the frontend.
- [x] The fast suite passes, with no new failures against the current baseline.
  - 1,401 passed, 1 skipped, 2 xfailed, 0 failed. HEAD collects 1,396 fast tests, this branch 1,404 (8 new). CLAUDE.md's "663" was stale and is updated.
  - Re-verified 2026-10-30 by an independent check: two more fast tests (`test_guard_probabilities_stay_out_of_the_response`, `test_extract_features_defaults_and_workers_are_unchanged`), now 1,403 passed, 1 skipped, 2 xfailed. The slow `test_stored_precrops_are_reproduced_byte_identical` and the `model` test `test_served_features_match_the_real_service_on_db_photos` both pass. The README face-guard and evaluate sections now describe the served source and the per-cert guard probabilities.

## Owner decisions (2026-10-29)

- **Q1 compute:** approved. One GPU extraction of ~10.4k faces through the served path, after copying `face_guard.*` and the caches in `data/models/cache/` to `data/models/backup/`.
- **Q2 back gate:** keep 0.2% clean-face FPR, measured out of fold on served-path features, with the threshold grid unchanged.
- **Q3 confidence:** note only, as today.

## Results (2026-10-29)

Run on branch `CadeJordan/FaceGuardServedFeatures` (HEAD `a0c064d` plus this change, uncommitted). Backup first: `face_guard.*` and both caches went to `data/models/backup/pre-guard-served-2026-10-29/`, SHA-256 identical. The GPU extraction covered 2,611 certs × 2 faces × upright/rotated = 10,444 faces, about 22 min, with 8 preprocessing threads. `data/pokescan.sqlite` is unchanged (SHA-256 and mtime), images and crops untouched.

**Before refitting: the assumption is false.** These are the 2026-10-28 (precrop-trained) probes, out of fold on train+val (2,241 certs, same card-grouped folds), scored on served-path faces (`transfer_from_precrop`):

| check | threshold (precrop CV) | clean FPR, precrop faces | clean FPR, served faces |
|---|---|---|---|
| side | 0.9 | 0.067% (3 of 4,482) | 0.067% (3 of 4,482), 0 false swaps |
| front_upright | 0.99 | 0.178% (4 of 2,241) | 0.178% (4 of 2,241) |
| back_upright | none passes (0.99 shown) | 0.223% (5) | 0.223% (5) |

- **Why the numbers are identical.** For every one of the 2,611 certs, the served-path upright features equal the precrop features: maximum relative difference 0.000 in the float16 cache. The stored photos are already upright and capped at 1,024 px, precrop ran the same detector, and the stored precrop JPEG decodes to the same pixels as `_jpeg_roundtrip`. So for upright faces, "precrop vs served" makes no difference. Only the rotated set differs: rotating the photo before detection, against `np.rot90` of the precrop, changes the global features by a median of 9% (fronts) and 6% (backs), relative.
- **Where the test split's 3 of 375 came from.** All 3 front rotations in guard_after (76000413, 88000652, 88001436) are among the 5 test certs with no usable precrop. Their front has no card outline, so the served path grades the padded whole photo (`found_quad` False). That kind of face isn't in `load_rows`, so the guard never trained on it. The old guard's P(flip) on those fronts was 0.991, 0.999 and 1.000. On the 370 test certs with a detected card it rotated 0 fronts, so the served rate there is 0, in line with CV. Since #8, `/api/grade` refuses all 5 of these inputs with 422 `card_not_found`, so the false rotations show only in `ml.eval`, which still grades refused inputs, and never in a response.

**Refit on served-path features** (card-grouped 5-fold CV on train+val, seed 0; `test-artifacts/eval/face_guard_report.json`):

| check | CV AUC | CV error at 0.5 | clean FPR at 0.9 / 0.95 / 0.99 | threshold | status |
|---|---|---|---|---|---|
| side (8,964 faces) | 0.99999 | 0.18% (16) | 0.067% / 0.022% / 0.022% | 0.9 | enabled |
| front_upright (4,482) | 0.99978 | 0.54% (24) | 0.357% / 0.223% / 0.134% (3) | 0.99 | enabled |
| back_upright (4,482) | 0.99813 | 0.29% (13) | 0.312% / 0.312% / 0.223% (5) | (0.99 stored) | **disabled** |

- Side as pairs at 0.9: 0 false swaps in 2,241, 99.5% of swapped pairs caught, 0.13% of clean pairs get a "both look like" note. front_upright at 0.99 catches 95.1% of flipped fronts out of fold (94.4% before).
- **back_upright still misses the 0.2% gate** (Q2): 5 of 2,241 upright backs read as flipped at 0.99, against a limit of 4.48. The sidecar's `disabled_reasons` holds the reason: "no threshold in (0.9, 0.95, 0.99) has clean-face FPR <= 0.2% (0.9: 0.312%, 0.95: 0.312%, 0.99: 0.223%)".
- Test, reported once: side AUC 1.0, clean FPR 0, 0 false swaps, 99.5% of swapped pairs caught. front_upright AUC 0.99976, clean FPR 0 at 0.99, 95.7% of flipped fronts caught. back_upright, for information only: AUC 1.0, FPR 0, 99.2% caught.

**guard_served** (`python -m ml.eval.evaluate --split test --robustness-limit 100 --perturbations rotate_180,swap_faces --out-dir test-artifacts/eval/guard_served`, CPU provider, "regressions vs baseline: none"):

| | guard_before | guard_after (10-28) | guard_served (10-29) |
|---|---|---|---|
| clean MAE / QWK / exact (n 375) | 0.435 / 0.491 / 66.9% | 0.437 / 0.492 / 66.7% | 0.437 / 0.492 / 66.7% |
| clean `pred_int` identical to guard_before | 375 | 374 | 374 |
| clean fronts rotated / backs rotated / swaps | 0 / 0 / 0 | 3 / 0 / 0 | 3 / 0 / 0 (the same 3 certs) |
| clean latency p50 / p95 | 2,496 / 2,823 ms | 2,343 / 2,500 ms | 2,289 / 2,491 ms |
| `rotate_180` flip rate / soft shift / fronts rotated (n 100) | 13.0% / 0.155 / 0 | 4.0% / 0.100 / 94 | 4.0% / 0.099 / 93 |
| `swap_faces` flip rate / soft shift / swapped back (n 100) | 85.0% / 1.050 / 0 | 0.0% / 0.000 / 100 | 0.0% / 0.000 / 100 |

Clean certs the guard changed (identical in guard_after and guard_served). P(back) is per uploaded row and P(flip) per face, as [front, back]. The guard_served values come from the eval report, and a separate ONNX run reproduces them exactly:

| cert | grade | before → guarded | quad found (front, back) | old guard P(flip) | new guard P(flip) | new guard P(back) |
|---|---|---|---|---|---|---|
| 76000413 | 10 | 10 → 10 (9.520 → 9.509) | no, yes | 0.991, 0.000 | 0.999, 0.000 | 0.000, 1.000 |
| 88000652 | 10 | 10 → 10 (9.658 → 9.679) | no, yes | 0.999, 0.000 | 1.000, 0.000 | 0.000, 1.000 |
| 88001436 | 8 | **8 → 7** (7.357 → 6.962) | no, no | 1.000, 0.994 | 1.000, 0.771 | 0.321, 0.984 |

- `rotate_180`: 7 rotated fronts not caught (P(flip) 0.13–0.988; 6 before). The one front rotation in `swap_faces` is 76000413 again. The disabled back probe read every one of the 100 rotated backs as flipped (P ≥ 0.9988), and no clean test back above 0.99 (max 0.771). That is test data, so it changes nothing.
- **Outcome.** Retraining on served features changed the served behaviour within noise and didn't enable the back check. That is expected now: the only training data that changed is the rotated set. The one real failure mode measured, front rotations on fallback (no-quad) faces, is outside the training rows. Whether to act on it is an owner decision, outside this spec. Options: skip the orientation check when a face has no quad, since `/api/grade` refuses it anyway; train the guard on fallback faces too; or accept it because they are refused.

## Facts vs assumptions

- **Fact:** the 2026-10-28 numbers above. Per-cert breakdown: `test-artifacts/eval/guard_after/rot180_breakdown.json`.
- **Assumption:** the gap between CV and served false rotations comes from the precrop vs served-path feature difference, not from test-set noise. 3 of 375 is a small count, so a first step is to measure the served FPR on train+val out of fold.
  - **Refuted 2026-10-29** (Results): upright served and precrop features are identical, and the precrop probes' out-of-fold served FPR equals their precrop FPR. The 3 test rotations are all on fronts with no card outline (fallback crops), which the guard never trained on and `/api/grade` refuses with 422.
