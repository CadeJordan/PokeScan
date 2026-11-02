# Tighter crops for slab-well and small-outline photos

## Intent

Make card detection find the card itself, not the slab well around it or a frame inside it, on the PSA photos it gets wrong today. Measure the effect on card-grouped CV before anything under `data/` is rewritten, then re-crop and retrain if it helps. This is [[cv-ml-improvement-plan]] item 8.

## Why

`precrop.crop_kind` sorts each PSA detection by how tall the quad is relative to the photo (PSA framing is fixed: the card is ~0.64–0.66 of the height, the well ~0.73–0.80):
- 'card': normal;
- 'well': taller than 0.70;
- 'small': shorter than 0.60, which is either a small-format card or an outline inside the card.

Counts on the current DB (read-only, 2026-10-31):

| Split | Front card / well / small / none | Back card / well / small / none |
|---|---|---|
| train | 1,757 / 73 / 43 / 8 | 1,845 / 15 / 17 / 4 |
| val | 349 / 20 / 2 / 3 | 374 / 0 / 0 / 0 |
| test | 346 / 16 / 8 / 5 | 364 / 5 / 5 / 1 |

The corner and edge regions are fixed fractions of the 600×840 crop. On a 'well' crop they see the slab gasket instead of the card's edge, and on an inner-frame 'small' crop they see the artwork.

The plan's evidence (out-of-fold MAE, with always-10 MAE on the same rows in brackets):

| Crop kind | MAE | Always 10 |
|---|---|---|
| card | 0.441 | 0.653 |
| well | 0.482 | 0.562 |
| small | 0.622 | 0.911 |

On the live test pipeline it's card 0.476, well 0.545, and small 1.11 (n = 9).

The served path runs the same `preprocess.detect_and_crop`, so a detector fix also changes what the API grades. That matters for slab photos users upload.

## Scope

**Phase 1: diagnose and measure. Nothing under `data/` is written.**
1. **Look at the failures.** Render overlays of the found quad for every 'well' and 'small' face to `test-artifacts/overlays/tighter_slab_crops/`. Sort them into:
   - a correct small-format card;
   - an inner frame or partial outline;
   - the slab well with the card visible inside it;
   - other.
   Read `preprocess._find_card_quad`, `_candidates`, `_is_holder` and `_contains` to find why the card's own outline loses to the well or the frame.
2. **Fix the detector** in `backend/app/ml/preprocess.py`, so serving and training both get it (decision 1). Add tests on synthetic images for each failure type, using `tmp_path` and no real data.
3. **No collateral damage.** Re-run detection in memory on every stored PSA photo (front and back, all splits). Report how each face's kind moves (card/well/small/none, before → after). For faces that were 'card' and stay 'card', report the quad change: the max corner shift in px and the IoU. Any 'card' face whose quad moves by more than 2 px is listed with an overlay. The existing byte-identical preprocessing tests must still pass, or any change to them is explained.
4. **CV estimate.**
   - Crop the changed faces in memory and extract their features. Write them to a scratch cache under `test-artifacts/`, never `data/models/cache/`.
   - Run `cross_validate` for seeds 0–4 on train+val with those faces' features substituted.
   - Compare with today's features as paired, per-cert differences, with card-group bootstrap CIs: overall, on the certs whose crops changed, and per crop kind.
   - Report MAE, exact accuracy, QWK, AUC 10 vs 9, 9 recall and 10 recall, using today's `ten_cutoff` (0.58) for the hard grades.

**Phase 2: ship, only if decision 2's gate is met.** These writes under `data/` are approved by decision 2:
5. **Back up first.**
   - Copy `data/pokescan.sqlite` to `data/backups/pokescan.pre-tighter-crops-2026-10-31.sqlite`, using SQLite's backup API so the copy is consistent.
   - Copy `data/models/grade_head.*`, `face_guard.*` and `data/models/cache/*.npz` to `data/backups/`.
   - Record sha256 checksums for all of them.
   - `data/crops/` isn't backed up: it's 726 MB and can be regenerated from `data/images` with the old code.
6. **Re-crop and retrain.**
   - Run `python -m ml.data.precrop --force`, after checking no other job is writing to the DB.
   - Run `python -m ml.training.train_probe`. It re-extracts the features of changed crops: the cache is keyed on crop size and mtime, and `--force` touches every crop, so expect a full re-extraction. It also retunes `ten_cutoff`.
7. **Face guard (decision 3).** The face guard's served-path feature caches aren't invalidated by detection changes (CLAUDE.md), so delete `region_feats_served*.npz` and retrain the guard. Ship the new guard only if its gates hold; otherwise keep the backup.
8. **Check and record.**
   - Load-check the API and grade one real pair.
   - Run `ml.eval.evaluate --split test --robustness-limit 100 --perturbations rotate_180,swap_faces --out-dir test-artifacts/eval/tighter_crops` once.
   - Record the result per decision 4, next to the decision_rule run. Nothing is chosen on the test split.

## Out of scope

- The crop-kind thresholds and `crop_kind` itself, except where the fix needs a new kind name. The kinds stay a PSA-only diagnostic.
- Non-PSA framing, phone photos and the sideways retry (no labelled data; [[serve-input-normalization]]).
- Training-set filtering, i.e. dropping 'well' or 'small' certs from the fit. That's decision 1(c), not the default.
- The probes, the features, `FEATURE_SPEC`, the folds and the decision rule's criterion. `ten_cutoff` is retuned by `train_probe` as usual.
- [[data-backup-before-destructive-runs]]. Step 5 does the backups by hand.

## Acceptance criteria

- [x] Overlays and the failure breakdown for every 'well' and 'small' face are in `test-artifacts/overlays/tighter_slab_crops/` and summarised here.
- [x] The detector fix has synthetic tests for each failure type it addresses. The fast suite and `ruff check .` pass.
- [x] The kind-transition table over all stored PSA faces, and the quad-shift report for faces that stay 'card', are in this spec. No 'card' face's quad moves by more than 2 px unless it's listed and explained.
- [x] The phase 1 CV comparison (seeds 0–4, paired, card-group CIs; overall, changed certs, per kind) is in this spec. Nothing under `data/` was written in phase 1, checked with before/after integrity snapshots.
- [x] The decision 2 gate is applied and stated. If it's met:
  - the backups exist with sha256 checksums;
  - `precrop --force`, `train_probe` and, per decision 3, the face guard retrain have run;
  - the API loads and grades a real pair;
  - the test split is evaluated once and recorded per decision 4.
  - 2026-10-31: gate met (Phase 1 results). Backups in `data/backups/*pre-tighter-crops-2026-10-31*` (sha256 in `backups.json`). `precrop --force`, `train_probe` and the face guard retrain ran; the new guard passed its gates and shipped. `/api/health` and one real pair checked. Test split evaluated once (`test-artifacts/eval/tighter_crops/`). See Phase 2 results.
- [x] CLAUDE.md is updated if the detector's behaviour, the fast-suite baseline or the artifacts change.

## Owner questions

1. **Where to fix it.**
   - (a) **In the detector** (`preprocess.py`), so the API benefits too. That's riskier: it touches every crop and needs the no-collateral-damage check.
   - (b) **In `precrop` only.** A PSA-framing rule re-picks the quad for training. The API keeps today's detector, so training and serving crops differ.
   - (c) **No detector work.** Drop 'well' and 'small' certs from the fit set. Cheap, but it doesn't help the API, and it removes 7% of the training data.
   - Recommended: (a). Training and serving should crop the same way, and users upload slab photos too.
2. **When to ship (phase 2).**
   - (a) Approve phase 2 now, gated on phase 1: ship if the paired CV MAE on the changed certs improves, with a CI excluding 0, and the overall mean MAE and AUC 10 vs 9 aren't worse by more than the seed-to-seed SD.
   - (b) Stop after phase 1 and ask again with the numbers.
   - Recommended: (a). It's the same pattern as [[cv-fold-stratification]], and the backups make it reversible.
3. **The face guard.**
   - (a) Delete the served-feature caches and retrain the guard on the new crops. Ship it only if its gates hold; otherwise keep the old guard.
   - (b) Keep today's guard. Its features came from the old detector, so it would see slightly different crops at serve time.
   - Recommended: (a).
4. **The test number.**
   - (a) Record it once in this spec, next to the decision_rule run. This is a fix, not a modelling decision.
   - (b) Write ADR 0004.
   - Recommended: (a).

## Owner decisions (2026-10-31)

1. **(a) Fix it in the detector** (`preprocess.py`), so serving and training crop the same way. The no-collateral-damage check (phase 1 step 3) is required.
2. **(a) Phase 2 is approved, gated on phase 1.**
   - **Gate:** ship if the paired CV MAE on the changed certs improves with a CI excluding 0, and overall mean MAE and AUC 10 vs 9 aren't worse than today's by more than the seed-to-seed SD.
   - **Approved writes:** only the writes in steps 5–7:
     - the backups in `data/backups/`;
     - `precrop --force`, which rewrites `data/crops/` and the DB's crop columns;
     - `train_probe`'s `data/models/grade_head.*` and `cache/region_feats*.npz`;
     - deleting `cache/region_feats_served*.npz` and retraining the face guard.
   - **If the gate fails:** nothing under `data/` is written. The detector change stays in the branch for the owner to decide.
3. **(a) Retrain the face guard** on the new crops. Ship it only if its gates hold; otherwise restore the backed-up guard and caches.
4. **(a)** Record the test number once in this spec, next to the decision_rule run. No ADR.

## Phase 1 results (2026-10-31)

Scripts and outputs: `test-artifacts/eval/tighter_slab_crops/` (all read-only on `data/`). The detection dumps (`detect_before.npz`, `detect_v4.npz`) re-run `detect_and_crop` on the stored bytes of all 5,260 PSA faces (2,630 certs, all splits); `detect_before.npz` reproduces the DB's stored `crop_kind` on 5,260 / 5,260 faces.

### Failure breakdown (step 1)

Every 'well' and 'small' face has an overlay with all detector candidates (green strong / orange weak, with height fraction and edge support), the chosen quad and the crop: `test-artifacts/overlays/tighter_slab_crops/before/` (204 faces; contact sheets in `sheets_before/`). Classes are from looking at every overlay; `trace_before.json` replays the ranking for each face, and `classes.json` lists each face's class and its kind after the fix.

| Kind (faces) | Class | Faces (front / back) | After the fix |
|---|---|---|---|
| well (129) | slab well, card visible; the card's outline **is** a nested candidate but ranks out | 95 (86 / 9) | 61 → card, 34 still well |
| | slab well, card visible; the card's outline is **never** a closed candidate (e.g. dark-blue backs on a dark well, soft edges) | 19 (14 / 5) | unchanged |
| | other: Pikachu Award die-cut stickers, a Hanafuda card in an insert, a sealed pack | 11 (7 / 4) | 6 quads move, all still well |
| | correct: oversized (jumbo VMAX) card fills the well | 4 (2 / 2) | unchanged |
| small (75) | inner frame or partial outline inside the card | 34 (32 / 2) | 13 → card, 21 unchanged |
| | correct small-format card (MY151 stickers 26, die-cut stickers 3) | 29 (21 / 8) | unchanged |
| | partial outline inside a small-format card (MY151 backs) | 12 (0 / 12) | unchanged |

Why the card's outline loses (`_find_card_quad` / `_is_holder`):
- **Well, outline ranked out.** The largest strong candidate is the well. `_is_holder` steps inward only when the band inside the nested outline is brighter or more colourful than the ring between the outlines by `_HOLDER_CONTRAST`. Silver, grey and pale full-art borders are the same brightness and colour as the grey gasket (ring ≈ band luma, low chroma), so the step is refused. The two differ in texture: the moulded gasket ring is flat (luma median absolute deviation ~6–16 grey levels) while the printed card border is textured (MAD 25–76) (`ring_stats.json`).
- **Small, inner frame.** The detector steps correctly from the well to the card (strong outline, ~0.66 of the photo height), then takes a second step into a weak inner frame or a partial outline in the artwork, which passes the brightness/colour test because the art is brighter than the card's border.

### The detector change (step 2)

`backend/app/ml/preprocess.py`, generic to the image (no PSA framing):
- **Inner frames.** Past the first step, a *strong* outline steps into a *weak* nested one only if the ring between them is a flat gasket around a textured band (`_is_holder(..., require_flat=True)`), not on brightness/colour contrast alone. A card's own printed border around an inner frame fails this; a flat gasket around a weak card outline passes.
- **Same-colour gaskets.** When no other step applies, a last-resort step accepts a strong nested outline whose ring is flat (MAD ≤ `_GASKET_MAX_MAD` 20) and whose band is textured (MAD ≥ max(`_GASKET_MIN_TEXTURE` 15, 2 × ring MAD)), largest first. The search stops there, because inside the card a flat ring is its plain border.
- Synthetic tests (`test_slab_photo_finds_the_card` in `backend/tests/test_preprocess.py`, `tmp_path`, 4 seeds each): a textured silver full-art card in a same-luma grey well, and a card with a card-shaped art panel whose outline fades out halfway. All 8 fail on the old detector and pass on the new one.
- Not addressed: wells where the card's outline is never a closed contour (19); wells where the nested outline is weak or the ring isn't flat (34); the non-card items. 'well' fronts go from 109 to 49 and 'small' fronts from 53 to 40.

### No collateral damage (step 3)

Kind transitions over all 5,260 stored PSA faces (before → after):

| before \ after | card | well | small | none |
|---|---|---|---|---|
| card (5,035) | 5,035 | 0 | 0 | 0 |
| well (129) | 61 | 68 | 0 | 0 |
| small (75) | 13 | 0 | 62 | 0 |
| none (21) | 0 | 0 | 0 | 21 |

| Split | Front card / well / small / none, after (before) | Back card / well / small / none, after (before) |
|---|---|---|
| train | 1,812 / 29 / 32 / 8 (1,757 / 73 / 43 / 8) | 1,846 / 14 / 17 / 4 (1,845 / 15 / 17 / 4) |
| val | 356 / 13 / 2 / 3 (349 / 20 / 2 / 3) | 374 / 0 / 0 / 0 (unchanged) |
| test | 357 / 7 / 6 / 5 (346 / 16 / 8 / 5) | 364 / 5 / 5 / 1 (unchanged) |

- 'card' → 'card': 5,035 faces. 5,008 quads are identical and **27 move by more than 2 px** (max corner shift 76.1 px, min IoU 0.808); none moves by a smaller nonzero amount.
- The 27 (overlays in `test-artifacts/overlays/tighter_slab_crops/after/card-card_*.jpg`, before blue / after red, with both crops; sheets 00–04 in `sheets_after/`): 76000585, 76000857, 76001542, 76001673, 76001713, 76032736, 76032738, 76037119, 76037148, 76047046, 76047055, 76047077 (back), 76057060, 80000417, 80000429, 80000437, 82000406, 85000585, 88000774, 88000968, 88001855, 88001865, 88002024, 88002041, 88002042, 88002043, 88600741 (fronts unless noted). Checked on every overlay: in 26 of them the old quad was an inner frame or a partial art outline whose height still fell in the 'card' band (it cut off the card's top or sides), and the new one is the card's outline. 82000406 is a die-cut sticker; both quads are a box around it and the new one is slightly larger.
- 107 faces change in total (61 well → card, 13 small → card, 27 card moved, 6 well quads moved), one per cert; 90 of the certs are in train+val.
- Preprocess tests including slow (`-m "slow or not slow"`): 102 passed, 1 xfailed (the existing `cut_off` xfail). `test_stored_precrops_are_reproduced_byte_identical` passed before the re-crop because its seeded sample (50 test faces plus the fallback faces) contains no changed face.

### CV estimate (step 4)

- **Data.** Train+val, n 2,241 (613 9s, 1,336 10s; 1,100 card groups). `cross_validate`, 5 folds, seeds 0–4.
- **Features.** Today's features come from `region_feats.npz` (`save=False`), with only the changed faces substituted. Those faces were re-cropped from the photo with the new detector, then `encode_jpeg` + decode as `precrop` stores them, and extracted on the GPU into the scratch cache `test-artifacts/eval/tighter_slab_crops/region_feats_new.npz`.
- **Sanity check.** The unchanged face of each changed cert, re-extracted through the same loader, matches its cached features exactly (max |diff| 0.0 over 107 faces).
- **Metrics.** Hard grades use `ten_cutoff` 0.58 from the current `grade_head.json`. Values are mean ± seed SD (ddof 1) over the 5 seeds.
- **Paired diffs.** Δ = new − old per cert, averaged over seeds, with a card-group bootstrap 95% CI (2,000 draws).
- **Output:** `cv.json`, `cv.md`, `oof.npz`.

| Subset (n; 9s / 10s) | MAE old → new | exact | QWK | AUC 10 vs 9 | 9-recall | 10-recall | paired ΔMAE [95% CI] | paired Δexact [CI] |
|---|---|---|---|---|---|---|---|---|
| overall (2,241; 613 / 1,336) | 0.4654±0.0059 → 0.4666±0.0060 | 63.4% → 63.3% | 0.587 → 0.588 | 0.7254±0.0060 → 0.7255±0.0066 | 40.3% → 40.6% | 84.0% → 83.8% | +0.0012 [−0.0023, +0.0048] | −0.0013 [−0.0044, +0.0015] |
| changed certs (90; 17 / 68) | 0.3089 → 0.2800 | 73.6% → 76.0% | 0.314 → 0.360 | 0.693 → 0.717 | 31.8% → 32.9% | 87.4% → 90.3% | **−0.0289 [−0.0588, −0.0022]** | +0.0244 [−0.0022, +0.0545] |
| changed, was well (59; 12 / 44) | 0.2780 → 0.2712 | 75.6% → 75.6% | 0.348 → 0.400 | 0.751 → 0.752 | 33.3% → 35.0% | 90.9% → 90.5% | −0.0068 [−0.0308, +0.0189] | 0.0000 [−0.0246, +0.0226] |
| changed, was small (11; 0 / 9) | 0.4909 → 0.3636 | 69.1% → 81.8% | 0.292 → 0.322 | – | – | 75.6% → 91.1% | −0.1273 [−0.2727, 0.0000] | +0.1273 [0.0000, +0.2727] |
| changed, card moved (20; 5 / 15) | 0.3000 → 0.2600 | 70.0% → 74.0% | 0.128 → 0.198 | 0.579 → 0.656 | 28.0% → 28.0% | 84.0% → 89.3% | −0.0400 [−0.1043, +0.0111] | +0.0400 [−0.0111, +0.1043] |
| today's front kind card (2,105) | 0.4637 → 0.4656 | 63.4% → 63.2% | 0.597 → 0.597 | 0.7254 → 0.7254 | 40.2% → 40.4% | 84.0% → 83.8% | +0.0019 [−0.0017, +0.0054] | −0.0021 [−0.0051, +0.0010] |
| today's front kind well (92) | 0.3761 → 0.3717 | 68.9% → 68.7% | 0.274 → 0.307 | 0.695 → 0.697 | 33.0% → 35.0% | 87.8% → 86.9% | −0.0043 [−0.0237, +0.0163] | −0.0022 [−0.0228, +0.0160] |
| today's front kind small (44) | 0.7364 → 0.7136 | 53.6% → 57.3% | 0.301 → 0.326 | 0.815 → 0.828 | 60.0% → 64.0% | 68.7% → 73.9% | −0.0227 [−0.0732, +0.0227] | +0.0364 [0.0000, +0.0800] |

- **Paired Δ recall.** Changed certs: 9-recall +0.012 [0.000, +0.035], 10-recall +0.029 [−0.006, +0.068]. Overall: 9-recall +0.003 [−0.004, +0.009], 10-recall −0.002 [−0.005, +0.002].
- **Hard grades that differ between old and new features**, per seed: 71, 63, 70, 58, 71 overall, of which 9, 4, 8, 3, 7 are on changed certs. Most of the overall movement is refit noise on unchanged certs (the probes are refitted on slightly different training features). That's why the overall paired ΔMAE is +0.0012 with a CI that straddles 0.
- **Per-seed Δ AUC 10 vs 9**, overall: −0.0021, +0.0011, −0.0016, +0.0021, +0.0008.
- **Where the gain comes from.** It comes mostly from the formerly 'small' certs and the moved 'card' faces. The well → card certs barely move: their crops already held the whole card plus a gasket margin, so the regions saw the card's edge, only smaller.

### Gate (decision 2): met, ship

- Changed certs: paired ΔMAE −0.0289, 95% CI [−0.0588, −0.0022], which excludes 0. Met.
- Overall mean MAE 0.4654 → 0.4666 (+0.0012); the limit was a rise of 0.0059. Met.
- Overall mean AUC 10 vs 9 0.7254 → 0.7255 (+0.0001); the limit was a drop of 0.0060. Met.
- SD used: the seed-to-seed SD (ddof 1, seeds 0–4) of **today's** features (MAE 0.0059, AUC 0.0060). The new features' SDs (0.0060 / 0.0066) give the same verdict.

### Integrity (phase 1)

`integrity_before.json` and `integrity_after_phase1.json` (in `test-artifacts/eval/tighter_slab_crops/`) are identical:
- `pokescan.sqlite`: sha256 `185221d6a61faa930ebaee502bae63eacbe34691f67e4ba5a237d16bf1399ffb`, 7,086,080 bytes, mtime unchanged;
- `data/images`: 5,275 files / 954,527,815 bytes;
- `data/crops`: 5,260 files / 748,007,793 bytes;
- every file in `data/models/`, `data/models/cache/` and `data/backups/` has the same sha256.

Nothing under `data/` was written in phase 1.

## Phase 2 results (2026-10-31)

Branch `CadeJordan/TighterSlabCrops` (HEAD `50ecf08` plus this change, uncommitted). Scripts, logs and outputs: `test-artifacts/eval/tighter_slab_crops/`. Every long job ran alone, after checking that no other Python process was running.

### Implementation

- **Detector:** `backend/app/ml/preprocess.py` (`_is_holder(..., require_flat=True)`, the flat-gasket last-resort step, `_GASKET_MAX_MAD` / `_GASKET_MIN_TEXTURE` / `_GASKET_TEXTURE_RATIO`). Tests: `test_slab_photo_finds_the_card` in `backend/tests/test_preprocess.py`.
- **No other code changed.** `FEATURE_SPEC`, the probes, the folds, the decision rule's criterion and the face guard's training code are as before.

### Backups (step 5)

`backup.py` → `backups.json`. The DB went through SQLite's backup API (`data/backups/pokescan.pre-tighter-crops-2026-10-31.sqlite`, 7,429 certs; the source sha256 `185221d6…` is recorded beside it). `grade_head.*`, `face_guard.*` and the four `cache/*.npz` are byte copies with the same sha256 as their sources. `data/crops/` isn't backed up (decision 2).

### Re-crop and probe retrain (step 6)

- `python -m ml.data.precrop --force` (`precrop.log`): ok 2,630, skipped 0, failed 0, quad_fallback 19 (same as before).
- The DB, against the backup: same 7,429 rows. Only `precrop_at` (2,630 rows), `crop_kind_front` (73) and `crop_kind_back` (1) changed. That's the 74 well/small → card faces of the Phase 1 table. Crop paths and `crop_found_*` are unchanged.
- `python -m ml.training.train_probe` (`train_probe.log`): every feature re-extracted (`region_feats.npz`: 0 cached, 2,611 to extract), as expected after `--force`. `ten_cutoff` was retuned to **0.56** (was 0.58), by the decision-rule criterion on CV seeds 0–4. At t 0.58 the new features' mean exact is 63.3% and MAE 0.4666, the same as Phase 1's estimate. At 0.56 they're 63.8% / 0.4634, which now passes the criterion.
- `grade_head.npz` sha256 `2c9c89e7…` (was `0f62a686…`), `grade_head.json` `dc63e057…`. `feature_spec_hash` is unchanged (`428e7a366e2e`). The new `git_commit` is null: `train_probe` got no answer from `git rev-parse` in that run.

| `train_probe` (`grade_head.json`) | n | MAE | exact | QWK | ECE | AUC ≤8 vs ≥9 | AUC 10 vs 9 |
|---|---|---|---|---|---|---|---|
| CV seed 0, old crops, t 0.58 | 2,241 | 0.4726 | 62.6% | 0.5812 | 0.0294 | 0.8514 | 0.7243 |
| CV seed 0, new crops, t 0.56 | 2,241 | 0.4645 | 63.4% | 0.5849 | 0.0288 | 0.8522 | 0.7222 |
| Test (precrop features), old, t 0.58 | 370 | 0.4514 | 64.6% | 0.4980 | 0.0559 | 0.8292 | 0.7823 |
| Test (precrop features), new, t 0.56 | 370 | 0.4378 | 66.0% | 0.5111 | 0.0552 | 0.8350 | 0.7788 |

CV seed 0 is in-sample for t, so the "old vs new" rows mix the crop change with the t change. Phase 1 is the like-for-like crop comparison.

### Face guard (step 7, decision 3)

- Deleted `cache/region_feats_served.npz` and `region_feats_served_rot180.npz`. Then `python -m ml.training.train_face_guard --out test-artifacts/eval/tighter_slab_crops/face_guard_new/face_guard.npz --report test-artifacts/eval/tighter_slab_crops/face_guard_report_new.json`, with the defaults (`--features served`, 5 folds, 8 workers), log `train_face_guard.log`. It took ~28 min on the GPU: served upright 11.5 min, served rotated 11.5 min, plus the in-memory precrop-rotated set for `transfer_from_precrop`.
- The guard was written to a staging path first, so `data/models/face_guard.*` stayed untouched until it passed. Both served caches were regenerated (`923784cc…`, `e8bb13cc…`).
- `region_feats_rot180.npz` (the `--features precrop` source) is stale against the new crops and wasn't rewritten. Nothing at serve time or in the default training path reads it. The transfer step reads it with `save=False`, so it re-extracts in memory. A `--features precrop` run would refresh it.

Card-grouped 5-fold CV on train+val (2,241 certs, seed 0). Old = the 2026-10-29 guard (backup), new = this retrain:

| check | CV AUC old → new | CV error at 0.5 | clean FPR at 0.9 / 0.95 / 0.99, new (old) | threshold | status |
|---|---|---|---|---|---|
| side (8,964 faces) | 0.99999 → 0.99999 | 0.18% (16) → 0.18% (16) | 0.067% / 0.022% / 0.022% (same) | 0.9 | enabled |
| front_upright (4,482) | 0.99978 → 0.99977 | 0.54% (24) → 0.51% (23) | 0.312% / 0.223% / 0.134% (0.357 / 0.223 / 0.134) | 0.99 | enabled |
| back_upright (4,482) | 0.99813 → 0.99810 | 0.29% (13) → 0.22% (10) | 0.312% / 0.268% / 0.223% (0.312 / 0.312 / 0.223) | (0.99) | disabled |

- **Gates hold.** Side: AUC ≥ 0.999 and error ≤ 0.3%. front_upright: AUC ≥ 0.99, and 0.99 meets the 0.2% FPR limit (3 of 2,241). The enabled checks (side, front_upright) and thresholds are identical. back_upright still misses the 0.2% limit (5 of 2,241 at 0.99), with its reason in `disabled_reasons`.
- Side pairs at 0.9: 0 false swaps, 99.51% of swapped pairs caught (99.46% before). front_upright catches 94.7% of flipped fronts at 0.99 (95.1% before).
- **Shipped:** staged files copied to `data/models/face_guard.npz` (`d3313ff8…`) and `.json` (`690e34ed…`). The report was copied to `test-artifacts/eval/face_guard_report.json`; the previous one is kept as `face_guard_report_before.json`.
- Test, once: front_upright clean FPR at 0.99 0 → 1 of 370 (76047084, below). Detection 95.9% (95.7%). Side unchanged.

### Load check and one real pair (step 8)

API: `uvicorn backend.app.main:app --host 127.0.0.1 --port 8000`. Output in `health.json` and `grade_76001278.txt`.
- `/api/health`: `status ok`, `model_loaded` true, `ten_cutoff` 0.56, no `probe_load_error`. Face guard loaded (side + front_upright), no `face_guard_load_error`.
- `/api/grade` on cert 76001278 (PSA 9, test). Its front was 'well' and is now 'card'. HTTP 200: grade 9.71, `grade_int` 10, confidence 0.763. The decision_rule eval had 9.73 / 10.
- The server was stopped and port 8000 is free.

### Test split end to end (decision 4: recorded once; nothing chosen on it)

`python -m ml.eval.evaluate --split test --robustness-limit 100 --perturbations rotate_180,swap_faces --out-dir test-artifacts/eval/tighter_crops`: CPU, "regressions vs baseline: none" (`evaluate.log`). The comparison is in `eval_compare.json` (script `compare_eval.py`). AUCs are computed from the soft grades in each report's predictions, as `train_probe.metrics` does. This was the one end-to-end test run. Test numbers were also seen, but nothing was chosen on them: `train_probe` prints its test metrics, the face guard report includes test, and the 0.58 re-decode below was done after the fact to split the cut-off effect from the crop effect.

| n 375 | decision_rule (old crops, t 0.58) | tighter_crops (new crops, t 0.56) |
|---|---|---|
| MAE [95% CI] | 0.456 [0.384, 0.533] | 0.443 [0.368, 0.515] |
| Exact [95% CI] | 64.0% [59.5, 69.1] | 65.3% [60.5, 70.1] |
| QWK [95% CI] | 0.499 [0.340, 0.637] | 0.512 [0.358, 0.645] |
| Soft MAE / bias | 0.553 / +0.067 | 0.552 / +0.066 |
| ECE | 0.051 | 0.052 |
| 9 / 10 recall | 47.8% / 84.0% | 45.2% / 87.8% |
| Predicted 9 / 10 | 114 / 246 | 103 / 256 |
| AUC ≤8 vs ≥9 / 10 vs 9 | 0.832 / 0.778 | 0.839 / 0.774 |
| Crop kind card / well / small / none: n, MAE | 343 0.40 / 19 1.11 / 8 1.00 / 5 0.80 | 354 0.40 / 10 1.40 / 6 0.83 / 5 0.80 |
| `n_would_reject` card_not_found / same_side | 5 / 0 | 5 / 0 |
| MAE both quads found / fallback (n 5) | 0.451 / 0.80 | 0.438 / 0.80 |
| `rotate_180` MAE / flip rate / soft shift / with notes | 0.31 / 9% / 0.109 / 93 | 0.31 / 9% / 0.091 / 97 |
| `swap_faces` MAE / flip rate / with notes | 0.32 / 0% / 100 | 0.32 / 0% / 100 |
| Latency p50 / p95 | 2,394 / 2,565 ms | 2,376 / 2,491 ms |

- **Crop kinds** come from the DB, so the groups differ between runs. The 9 certs that left 'well' and the 2 that left 'small' are now in 'card'. The remaining 10 'well' certs (MAE 1.40) are the hard ones. These are small groups, so this isn't evidence either way.
- **Changed grades: 13 of 375.** Eleven go 9 → 10: 8 true 10s now right, 3 true 9s (76000010, 76001220, 76033529) now wrong. 76001564 (PSA 9) goes 10 → 9, now right. 76001464 (PSA 9) goes 9 → 8, now wrong. Net +5 exact.
- **None of the 13 is a cert whose crop changed.** On the 17 test certs with a changed crop, every hard grade is the same as before (MAE 0.647, 10 of 17 exact, both runs). Their soft grades moved by a mean 0.046, against 0.020 over all certs.
- **Cutoff vs crops+retrain, separated approximately.** The report has no cumulative probabilities, so P(10) comes from the new head on the new precrop test features (`region_feats.npz`, read only). That number matches the served soft grade within 1e-3 on 369 of 370 certs. The exception is 76047084, which the guard rotated. Re-decoding the new run at 0.58 gives:
  - **Cutoff (0.58 → 0.56):** 8 test certs have P(10) in (0.56, 0.58], and all are 10 at 0.56. Seven of them are among the 13 changed grades (5 true 10s and 2 true 9s now 10). 76052018 (PSA 10) would have dropped to 9 at 0.58 under the new head, and the cutoff keeps it at 10. Effect: MAE 0.453 → 0.443, exact 64.3% → 65.3%.
  - **Crops + retrain (at the old t 0.58):** 7 grades differ from decision_rule (76000010, 76001464, 76001564, 76033531, 80000430, 88002170 and 76052018), with net +1 exact. Effect: MAE 0.456 → 0.453, exact 64.0% → 64.3%. All within refit noise, like Phase 1's unchanged certs.
- **Guard on clean test certs: fronts rotated 3 → 4.** The 3 fallback fronts are still rotated (76000413, 88000652, 88001436). 88001436 is still 7 (PSA 8), soft 6.78 → 6.98. The new rotation is 76047084 (PSA 8, front quad found, crop unchanged): its P(flip) went from 0.9876 under the old guard to 0.9903, just over 0.99. It now carries a "front photo was upside down" note, and its grade stays 9 (soft 9.02 → 8.91). This is the test-split false positive in the guard's own report (front_upright clean FPR 1 of 370). It's recorded, not acted on: thresholds are chosen on CV only.
- `rotate_180`: 97 of 100 now carry a note (93 before), and the soft shift fell from 0.109 to 0.091. The flip rate is unchanged at 9%; 2 per-cert grades differ from the old run. `swap_faces`: 100 of 100 swapped back, 0 flips.

### Integrity (phase 2)

`integrity_final.json`, against `integrity_after_phase1.json` (same script). The DB diff against the backup is described above.
- `data/images`: 5,275 files / 954,527,815 bytes, unchanged. Newest mtime 2026-05-15.
- `data/crops`: 5,260 files (bytes 748,007,793 → 747,773,136, re-cropped).
- `pokescan.sqlite`: sha256 `185221d6…` → `c9cc5526…`, same size; only the crop columns listed above changed.
- `data/models/`: only `grade_head.npz/.json`, `face_guard.npz/.json`, `cache/region_feats.npz` and `cache/region_feats_served*.npz` changed. `card_features.*`, `grade_model.*` and `cache/region_feats_rot180.npz` are byte-identical.
- `data/backups/`: only the 9 `*pre-tighter-crops-2026-10-31*` files were added, and nothing was removed.

### Checks

- Fast suite (`-m "not slow and not visual and not model"`, run alone): **1,631 passed, 1 skipped, 2 xfailed**, 268 s (CLAUDE.md said 1,623; it's updated). The first run had one failure, `test_retest_grade_decision_rule.py::test_saved_oof_predictions_reproduce_the_shipped_cutoff`. It checked that the shipped head's `ten_cutoff` equals the t tuned on the decision-rule run's saved OOF predictions (gitignored, old crops). Those predictions still give 0.58, but the retrained head is tuned on new predictions and ships 0.56.
  - **Fix:** the head comparison now applies only when the head's stored `ten_cutoff_tuning.median` equals the one tuned from those predictions, i.e. the head was tuned on them. That condition holds for the backed-up decision-rule head, which the test now also checks directly (`data/backups/grade_head.json.pre-tighter-crops-2026-10-31`). The `t == 0.58` pin on the saved predictions stays.
- `ruff check .`: clean.
- `python -m pytest backend/tests/test_preprocess.py -m "slow or not slow"`: 102 passed, 1 xfailed. That includes `test_stored_precrops_are_reproduced_byte_identical`, now against the re-cropped `data/crops`.

## Facts vs assumptions

- **Fact:** the crop-kind counts above (DB read-only, 2026-10-31). `data/crops` is 726 MB and the DB is 6.8 MB.
- **Fact:** serving and `precrop` both call `preprocess.detect_and_crop`. The face guard's served caches (`region_feats_served*.npz`) aren't invalidated by detection changes. The probe feature cache is keyed on crop size and mtime.
- **Fact:** the MAE-by-kind evidence is from the 2026-10-23 review (plan item 8), before the card-grouped re-split and the decision rule. Phase 1 re-measures it.
- **Assumption:** most 'well' faces have a visible card outline inside the well that the detector ranks lower, so a ranking fix can recover it. Phase 1 step 1 checks this.
- **Assumption:** the gain is small overall (about 7% of certs; maybe 0.01–0.02 MAE) and concentrated on the changed certs. That's why the gate in decision 2 looks at the changed certs.
