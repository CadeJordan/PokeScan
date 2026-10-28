# CV/ML improvement plan (review of 2026-10-23)

Review of the card detector, centering, the probe grade model, its training and evaluation, and the legacy CNN path, on branch `CadeJordan/ScoreImprovements` with the uncommitted probe work treated as current. No source was changed. Every number below was measured on 2026-10-23 with read-only scripts against `data/` (DB opened `mode=ro`), unless it's quoted from a report.

Specs written from this plan:
- [[serve-input-normalization]]: items 1 and 2
- [[probe-artifact-consistency]]: item 3
- [[stable-grouped-split]]: items 4 and 9
- [[face-side-orientation-guard]]: item 6

Small fixes found in the review are in `intent/smallfix.md` §10.

## How the system performs now

Probe model (`data/models/grade_head.json`, trained 2026-10-23, commit `77a34ea`, 2,223 fit certs, all PSA):

| | n | MAE | exact | QWK | ECE | AUC ≤8 vs ≥9 | AUC 10 vs 9 |
|---|---|---|---|---|---|---|---|
| Card-grouped 5-fold CV (train+val) | 2223 | 0.447 | 65.1% | 0.620 | 0.036 | 0.853 | 0.739 |
| Test, precrop features | 388 | 0.490 | 62.1% | 0.524 | 0.057 | 0.856 | 0.732 |
| Test, cards also in train+val | 251 | 0.414 | 66.5% | 0.516 | | | |
| Test, unseen cards | 137 | 0.628 | 54.0% | 0.510 | 0.073 | 0.833 | 0.638 |
| Test, live pipeline (`ml.eval.evaluate`, raw photos) | 393 | 0.494 | 62.1% | 0.511 | 0.067 | | |
| Always 10, test | 393 | 0.636 | 59.3% | 0 | | | |

- I reproduced the CV and test numbers exactly from the cached features (CV takes 85 s).
- Per-grade recall on test (live pipeline): 10: 90%, 9: 20%, 8: 37%, ≤7: 2 of 20. Grade-10 bias is −0.41, grade-9 bias is +0.31.
- Latency (live pipeline, CPU): p50 2.5 s per front+back pair (`test-artifacts/eval/probe/report.json`), ~2.7 s in my runs. Detection takes ~110 ms per 1024-px face. The rest is DINOv2.
- Detection on PSA photos: a quad is found on 98.9% of fronts and 99.7% of backs. 19 of 2,630 certs miss on at least one face. Crop kinds in the fit set: card 93%, `well` 5%, `small` 2%.
- The fast suite is 381 passed, 2 xfailed, 182 deselected (120 s). The `smallfix.md` header still says 248/3.

Calibration and the decode rule are fine as they are. I checked:
- On the CV out-of-fold probabilities, the NLL-optimal temperature is exactly 1.00. The meta probe trains on out-of-fold soft grades, so it comes out calibrated.
- Median decoding (current) has the best MAE. Argmax gives 0.462 MAE / 0.590 QWK, round(soft) gives 0.467 / 0.618. Moving the P(>9) cutoff between 0.45 and 0.55 changes MAE by ≤0.005 and QWK by ≤0.002.
- So I propose no calibration or decode work. ECE does rise on unseen cards (0.073), which is part of item 9, not a calibration problem.

## Urgent and correct (bugs)

Ranked by impact over effort. Effort: S < ½ day, M = 1–2 days, L > 3 days.

### 1. Phone uploads break detection: EXIF is ignored, full-res photos aren't downscaled, sideways cards are never found (spec: [[serve-input-normalization]])

- **Problem:** `detect_and_crop` decodes upload bytes with `Image.open(...).convert("RGB")` (`backend/app/ml/preprocess.py:512-514`). It doesn't apply EXIF orientation or limit resolution. Ingest does both (`ml/data/image_io.py:46` `exif_transpose`, long edge capped at 1024). All 5,260 training images are exactly 1024 px on the long edge with no EXIF orientation, so serving sees a different input distribution than training. The aspect filter only accepts portrait quads (`_AR_MIN, _AR_MAX = 0.65, 0.82`, `preprocess.py:73`), so a sideways card can never be found.
- **Evidence** (120 real test faces, `detect_and_crop` on re-encoded bytes):

  | Input | quad found | detect p50 |
  |---|---|---|
  | as stored (1024 px) | 99.2% | 111 ms |
  | phone-style EXIF Orientation=6 (pixels stored sideways) | **14.2%** | 60 ms |
  | sideways, no EXIF (landscape shot) | **18.3%** | |
  | upscaled 3× (3072 px, like a 12 MP phone) | **70.0%** | 503 ms (p95 813) |
  | EXIF-6 + `exif_transpose` + 1024 cap (the proposed fix) | 97.5% | |
  | 3× + 1024 cap (the proposed fix) | 99.2% | |
  | tilted 4/8/15° on an expanded canvas | 100% (n=80 each) | |

  Tilt is not a problem.
- **Impact:** high. Most phone photos carry EXIF rotation or are 12 MP. Today most of them fall back to the padded whole photo and get a grade from out-of-distribution features. The only signal is a note.
- **Effort:** S. **Dependencies:** none. The new crops are byte-identical on existing data (see the spec), so no retrain is needed.

### 2. Centering is measured on fallback canvases, and every face is detected twice per request (spec: [[serve-input-normalization]])

- **Problem:** `compute_centering_subgrade` re-runs `detect_and_crop` on the raw bytes (`backend/app/ml/centering.py:314-317`) and measures whatever comes back, including the padded fallback canvas. `/api/grade` runs detection once in `predict` (`inference.py:184-185`) and again for centering (`grade.py:108`). The legacy path does it a third time in `compute_condition_subgrades` (`condition.py:393-394`).
- **Evidence:** on the 21 real faces whose detection falls back, centering returned a measurement for 3, all `50/50 · 48/52` → **10.0**. Those are confident PSA-10 centering grades for faces where no card was found. The duplicate detection costs ~110 ms per face per request.
- **Impact:** medium. The bogus value is rare on PSA photos, but it would be common on user photos until item 1 lands. **Effort:** S. **Dependencies:** none.

### 3. The feature cache and the probe artifacts can silently go stale (spec: [[probe-artifact-consistency]])

- **Problem:**
  - `extract_features` keys the cache on `(cert, cropped_front_path, cropped_back_path)` (`ml/training/train_probe.py:90`). Crop paths are deterministic (`crops/<original path>`, `ml/data/precrop.py:44-46`), so after `precrop --force` with a changed detector, training reuses features of the old crops while serving computes features from the new ones.
  - The cache is only validated against `BACKBONE` (`train_probe.py:87`). Changing `FEAT_H/FEAT_W`, `CORNER_PATCHES` or `EDGE_PATCHES` (`backend/app/ml/features.py:26-30`) leaves stale features in use.
  - `card_features.onnx` is only exported when the file is missing (`train_probe.py:205`), so a retrained head can be paired with an old feature graph.
  - At load time, the only check is that feature dim D matches (`grade_head.py:94-105`, `inference.py:128-130`).
  - The head is saved before the ONNX export (`train_probe.py:201-207`), so a failed export leaves a mismatched pair.
- **Evidence:** this comes from reading the code. Today the artifacts are consistent: 80/80 sampled precrops are byte-identical to the current detector's output, and `test_served_features_match_training_cache` covers 6 certs. The risk is latent but silent, and it would show up exactly when someone improves the detector (items 1 and 8).
- **Impact:** medium (it prevents a silent train/serve skew). **Effort:** S. **Dependencies:** none. Item 1 doesn't change existing crops, but any later detector change does.

### 4. `split.py` reshuffles the PSA test set whenever PSA certs are added (spec: [[stable-grouped-split]])

- **Problem:** `_stratified_assign` shuffles each grade's full cert list with a fixed seed (`ml/data/split.py:29-52`). Adding a single PSA cert therefore re-deals every cert of that grade. The README says "the eval numbers stay comparable across harvests", but that only holds for eBay rows. The harvest is ongoing (`HANDOFF.md` §5).
- **Evidence** (simulated on the real rows; re-running on the current rows reproduces the DB 2630/2630):

  | Added PSA certs | Existing certs that change split | Leave test | Enter test |
  |---|---|---|---|
  | +1 | 56 | 12 | 12 |
  | +50 | 831 | 253 | 258 |
  | +200 | 987 | 292 | 293 |

  After +50, about 65% of the test set has been replaced, mostly by former training certs. Every earlier model was trained on those certs.
- **Impact:** high for evaluation validity. Before/after comparisons and `--save-baseline` silently become invalid, and test certs leak into the next model's train set. **Effort:** S (phase 1: sticky). **Dependencies:** none. Phase 2 is item 9.

### 5. If the probe can't load, the service silently serves a collapsed model

- **Problem:** if the probe artifacts fail to load, `GradeInferenceService` logs and falls back to the legacy CORN model (`inference.py:105-120`). The same happens if they're missing. That legacy model "collapsed towards a constant grade … worse than always predicting 10" (ADR 0001: test MAE 0.84 vs always-10 0.64). `/api/health` only shows the legacy metadata.
- **Evidence:** this is a pinned behaviour (`test_unloadable_probe_falls_back_to_legacy`, `test_selection_*`).
- **Impact:** medium. An artifact mismatch (item 3) or a corrupt file turns into worse-than-trivial predictions with HTTP 200. **Effort:** S. **Dependencies:** owner decision (open question 3). [[probe-artifact-consistency]] adds a `probe_load_error` to `/api/health` either way.

## Improvements

### 6. Swapped or upside-down photos change the grade silently (spec: [[face-side-orientation-guard]])

- **Problem:** nothing checks which photo is the front, or which way up it is. The probes are position-specific (corner TL/TR/BR/BL and the front/back rows are concatenated, `grade_head.py:53-56`).
- **Evidence:**
  - Swapped front/back (8 test certs): the soft grade moves by 0.7–2.9, e.g. 9.66→7.72 and 9.62→8.01.
  - Both faces rotated 180° (16 test certs): mean shift −0.19, max 0.63.
  - The HSV side classifier from ingest (`ml/data/side_classifier.py`) separates fronts from backs on crops with AUC only 0.84.
  - A logistic probe on the cached DINOv2 CLS features separates them with grouped-CV AUC 0.99995 (6 errors in 5,222 faces).
- **Impact:** medium–high for real users (swapping the uploads is an easy mistake, and item 1's landscape retry lands half of sideways photos upside down). **Effort:** M (one extra GPU feature extraction of ~5.2k rotated crops). **Dependencies:** [[serve-input-normalization]] (crops in `GradePrediction`). [[probe-artifact-consistency]] is optional.

### 7. Fallback crops are graded as if they were cards (needs decision)

- **Problem:** with no quad, the whole photo is padded to 600×840 (`preprocess.py:536-538`). For a slab, that includes the grade label. The probe was trained only on certs where both faces had a quad (`train_probe.py:63`), yet `/api/grade` returns a normal grade and confidence plus a note (`grade.py:99-102`).
- **Evidence:** on the 19 PSA certs with a fallback face, live-pipeline MAE is 0.84, against 0.49 overall. Predictions include 9→6 and 10→7. On user photos (not slabs) the input is further out of distribution, and that's unmeasured.
- **Impact:** medium. **Effort:** S once the contract is decided (open question 2). **Dependencies:** item 1 first, which removes most fallbacks.

### 8. Crop quality: `well` and `small` crops

- **Problem:** about 5% of fit-set fronts are cropped at the slab well (`crop_kind='well'`), and 2% at an inner frame or small card (`small`). Corner and edge features then see the gasket or art instead of the card edge.
- **Evidence** (out-of-fold, MAE vs always-10 MAE on the same rows): card 0.441 vs 0.653, well 0.482 vs 0.562, small 0.622 vs 0.911. On test (live): card 0.476, well 0.545, small 1.11 (n=9).
- **Impact:** low–medium (7% of certs, a partial gain). **Effort:** M (detector work, then `precrop --force` and a feature re-extraction; depends on item 3). Not specced.

### 9. The test split isn't card-grouped (phase 2 of [[stable-grouped-split]], needs decision)

- **Problem:** 251 of the 388 test certs share a printed card with train+val, so the headline test number mixes seen-card performance (MAE 0.414) with unseen-card performance (MAE 0.628).
- **Evidence:** I simulated a hash-on-card-group assignment with four salts. It gives 341–446 test certs with 49–61 graded ≤8 (the current split has 50), so stratification is roughly preserved. Only 54–63 of today's 393 test certs would stay in test.
- **Impact:** medium. The headline number would become honest, and it moves to a new baseline. **Effort:** S code + a retrain of about a minute, plus a re-baseline. **Dependencies:** item 4 phase 1, a DB backup ([[data-backup-before-destructive-runs]]), and owner approval (open question 1).

### 10. Sub-grades don't localise anything (needs decision)

- **Problem:** the corners, edges and surface sub-grades are the region probes' soft grades (`grade.py:118-123`). They are trained on the overall grade.
- **Evidence** (out-of-fold, n=2223):
  - The four region grades correlate with each other at 0.83–0.96.
  - Each correlates with the grade at about the same level (Spearman 0.49–0.52).
  - A region's deviation from the mean of the four correlates with the grade at |ρ| ≤ 0.08.
  - Mean region grades by true grade are nearly flat across regions (grade 10: 9.64/9.62/9.61/9.61; grade 8: 8.68/8.69/8.67/8.63).
  - So the UI shows three independent-looking sub-grades that are really one number. ADR 0001 says this ("move closely together"), but the UI doesn't.
- **Impact:** medium (user trust), low (accuracy). **Effort:** S for a copy change, L for real sub-grades, which need labels. **Dependencies:** open question 4.

### 11. Retire the legacy CNN path and the Phase C hooks

- **Problem:**
  - `model.py`, `multitask.py`, `calibrate.py`, `export.py`, `train.py`, `lightning_module.py` and the classical `condition.py` sub-grades (Spearman −0.05 to 0.02 with grade, ADR 0001) only serve a model that is worse than always-10.
  - `defects.py` is imported nowhere, and its docstring says the API "returns an empty defect list", which has no API field.
  - [[corn-decode-verification]] only matters for this path.
- **Impact:** low on accuracy, medium on maintenance and confusion. **Effort:** M. **Dependencies:** open question 3. Not specced.

### 12. Resolution ceiling on 9 vs 10

- **Problem:** the card is ~660 px tall in the stored 1024-px photo, and features are computed at 602 px (`features.py:26`). That's below the source resolution, which is itself downscaled from the 3–4 MB PSA originals at ingest (`image_io.py`, originals kept in `front_url`/`back_url`).
- **Evidence:** 9-vs-10 AUC is 0.73–0.74 in CV and on test, 0.64 on unseen cards. ADR 0001 attributes this to photo resolution. Whether resolution would help is unmeasured.
- **Impact:** possibly the largest accuracy lever after more low-grade data. **Effort:** L (a re-download decision, higher-resolution tiles for the corner and edge regions, re-extraction). **Dependencies:** open question 5. Not specced. The first step would be a 200-cert pilot.

### 13. No phone-photo evaluation set

- **Problem:** all labelled photos are PSA catalogue slab shots. `--manifest` evaluation exists (`README.md` "Raw (unslabbed) cards") but has no data. Items 1, 6 and 7 are measured here only with simulated perturbations.
- **Impact:** high for knowing real-world accuracy. **Effort:** S code (none needed), M owner time. **Dependencies:** open question 8.

## Ranked list

| Rank | Item | Kind | Impact | Effort | Spec |
|---|---|---|---|---|---|
| 1 | Normalise uploads: EXIF, 1024 cap, sideways retry (1) | bug | high | S | [[serve-input-normalization]] |
| 2 | One detection per face; no centering on fallback (2) | bug | medium | S | [[serve-input-normalization]] |
| 3 | Sticky split, so harvesting doesn't reshuffle test (4) | bug | high | S | [[stable-grouped-split]] |
| 4 | Feature/artifact fingerprinting, cache invalidation (3) | bug | medium | S | [[probe-artifact-consistency]] |
| 5 | Async endpoint blocks the event loop ~2.7 s per request | bug | medium | S | `smallfix.md` §10 |
| 6 | Don't silently serve the collapsed legacy model (5) | bug | medium | S | decision (Q3) |
| 7 | Face side + orientation guard (6) | improvement | medium–high | M | [[face-side-orientation-guard]] |
| 8 | Fallback-crop policy (7) | improvement | medium | S | decision (Q2) |
| 9 | Card-grouped test split + re-baseline (9) | improvement | medium | S + retrain | [[stable-grouped-split]] phase 2 (Q1) |
| 10 | Sub-grade presentation (10) | improvement | medium (trust) | S–L | decision (Q4) |
| 11 | Phone-photo evaluation set (13) | improvement | high (knowledge) | M owner time | decision (Q8) |
| 12 | Retire legacy path (11) | cleanup | low–medium | M | decision (Q3) |
| 13 | Tighter crops for well/small (8) | improvement | low–medium | M | none yet |
| 14 | Higher-resolution region features (12) | improvement | unknown, possibly large | L | decision (Q5) |

Suggested order: 1–2 and 4 in parallel (they touch different files), then 3, then 7 (which needs 1). Items 6, 8, 9 and 10 wait on answers.

## Missing tests (covered by the specs)

- EXIF-rotated, high-resolution and sideways uploads ([[serve-input-normalization]]).
- Centering abstains on a fallback crop; `/api/grade` detects each face once ([[serve-input-normalization]]).
- Cache invalidation on a crop rewrite or a feature-spec change; head/ONNX fingerprint mismatch ([[probe-artifact-consistency]]).
- Split stability under additions, group integrity, row-order independence ([[stable-grouped-split]]).
- Swapped and upside-down faces, end to end ([[face-side-orientation-guard]]). `ml.eval.evaluate` robustness has never been run on the probe model (`robustness: null` in `test-artifacts/eval/probe/report.json`).

## Open questions for the owner

1. **Re-split and re-baseline now?** Phase 2 of [[stable-grouped-split]] replaces ~85% of the test set and resets every number in ADR 0001 and the sidecars. *Recommendation:* yes, now, while the dataset is small. Keep the old assignment in a `split_legacy` column so the old numbers stay reproducible, and record the change in an ADR.
2. **What should `/api/grade` do when no card outline is found?** Options:
   - (a) 422 "card not found, retake on a plain background";
   - (b) 200 with an additive `reliable: false` field (needs the pydantic schema and `api-client.ts` changed in sync);
   - (c) keep today's note only.

   *Recommendation:* (a) for the probe model, after [[serve-input-normalization]] lands and fallbacks are rare. Grades from padded whole photos are out of distribution, and today they look exactly like real grades.
3. **Legacy model:** fail closed (503) when the probe can't load, instead of serving the collapsed CNN? And retire the legacy path (item 11), closing [[corn-decode-verification]] as won't-do? *Recommendation:* fail closed now, with an `ALLOW_LEGACY_MODEL=1` env opt-in. Retire the code in a follow-up spec.
4. **Sub-grades:** keep three numbers that measure the same thing, show one "condition" estimate, or keep the fields and reword the UI ("what this region suggests about the overall grade")? Real sub-grades need labels, e.g. BGS-graded listings with sub-grades from eBay. *Recommendation:* keep the API fields (no contract change), reword the UI hints now, and decide about BGS data separately.
5. **Re-fetch full-resolution PSA originals** (from the `front_url`/`back_url` already in the DB) for a resolution experiment? It's ~20 GB for all certs. I assume it doesn't consume PSA API quota, but that's unverified. *Recommendation:* a 200-cert pilot first.
6. **Compute for [[face-side-orientation-guard]]:** one GPU feature extraction of ~5.2k 180°-rotated crops (minutes) and a new `data/models/face_guard.npz`. OK? *Recommendation:* yes.
7. **Can user-facing notes change?** The specs add free-text `notes`, such as "front photo was upside down; rotated it" and "front and back looked swapped". The schema doesn't change. *Recommendation:* yes.
8. **Phone-photo evaluation set:** will you photograph ~50–100 of your own slabbed cards (known grades) with a phone, in varied light and angles, as a `--manifest` CSV? It's the only way to measure items 1, 6 and 7 for real. *Recommendation:* yes, with slabs first, then raw cards that are later graded.
9. **Test-set policy:** use the test split only for a final number per ADR, and do all selection on grouped CV? *Recommendation:* yes. Write it into CLAUDE.md with phase 2 of [[stable-grouped-split]].
10. **Scope:** is the frontend in scope for these specs? None of the specs needs a frontend change; only Q2(b) and Q4 would. *Recommendation:* keep the specs backend-only.

## Owner decisions (2026-10-23)

- **Round 1 scope:** [[serve-input-normalization]] plus `smallfix.md` §10 only. [[probe-artifact-consistency]], [[stable-grouped-split]] and [[face-side-orientation-guard]] wait for a later round.
- **Q1:** no re-split this round. Land sticky mode (phase 1) first; phase 2 later, after a DB backup and with an ADR.
- **Q3:** fail closed. If the probe can't load, return 503 and report why in `/api/health`; `ALLOW_LEGACY_MODEL=1` opts back into the legacy CNN. Implement this with [[probe-artifact-consistency]], replacing its "keep today's fallback" scope line.
- **Real runs:** agents may run real jobs that rewrite `data/models` (train_probe, GPU extraction), after copying the current artifacts to `data/models/backup/`.
