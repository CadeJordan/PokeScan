# Retire the legacy CNN model path

## Intent

Remove the end-to-end ConvNeXt CORN model (`grade_model.onnx`) and everything that exists only to train, export, calibrate or serve it, so the codebase describes the one model that's served: the probe model. Users see no change: the `/api/grade` response schema, the served grades and the frontend stay exactly the same. This is [[cv-ml-improvement-plan]] item 11 and the second half of open question 3. The first half, failing closed instead of silently serving the CNN, shipped earlier.

## Why

- The CNN is worse than always answering 10 (ADR 0001). The probe model has served by default since then, and the CNN loads only with an explicit `model_path` or `ALLOW_LEGACY_MODEL=1`.
- Every recent spec had to keep this path working or say why it didn't apply. Examples: `GradePrediction.rejection()` (probe only), TTA (legacy only, raises with the probe), the guard and decision-rule wording, and the fallback sub-grade branch in `api/grade.py`.
- The classical `condition.py` corner/edge/surface sub-grades are reached only on the legacy branch of `api/grade.py`. They don't track the grade (Spearman −0.05 to 0.02, ADR 0001).
- `defects.py` is imported nowhere. Its docstring describes an API field that doesn't exist.
- [[corn-decode-verification]] only concerns this model.

## What goes (to be confirmed by the import sweep, step 1)

**Serving (`backend/app/`)**
- `ml/model.py`, `ml/multitask.py`, `ml/defects.py`, `ml/condition.py`.
- In `ml/inference.py`:
  - the legacy load and predict path, the `model_path` argument, TTA (`tta`, `TTA_VARIANTS`, `DEFAULT_TTA`), and `kind == "legacy"`;
  - the `ALLOW_LEGACY_MODEL` fallback.

  The service becomes probe-only and still fails closed (503 with `probe_load_error`).
- In `api/grade.py`: the `compute_condition_subgrades` fallback branch. Corners/edges/surface always come from the probe's region grades, which is already the only path when the service is ready.
- In `core/config.py`: `active_model_path` and `allow_legacy_model`.

**Training and tools (`ml/`)**
- `training/train.py`, `lightning_module.py`, `calibrate.py`, `export.py`, and `configs/*.yaml` (if only these use them).
- `data/precompute_factors.py`: pseudo-labels for the CNN's factor heads.
- `data/export_for_annotation.py`: it imports `multitask`. Keep it only if it's still useful without that; see Q3.
- Anything in `ml/data/dataset.py` and augmentation code used only by the CNN's training loop. `load_records` / `precrop_coverage` stay, because `train_probe`, `precrop` and `ml.eval` use them.
- In `ml/eval/evaluate.py`: `--model-path`, `--tta` and the matching report fields. `--save-baseline`'s "(no --manifest / --tta)" guard loses its TTA half.

**Tests:** tests that only exercise the removed code are deleted, for example `test_model.py`, `test_training_setup.py`, `test_onnx_parity.py`, `test_condition.py`, and the legacy cases in `test_inference.py` / `test_api.py`. A shared test that covers probe and legacy keeps its probe case. Tests that pin a contract which stays get rewritten against the probe, not deleted; for example `check_training_inputs` in `test_precrop_filtering.py` moves, or is dropped with `train.py` only if `train_probe` has its own equivalent guard.

**Docs:**
- `CLAUDE.md`, `README.md` (training, ONNX export, TTA, `ALLOW_LEGACY_MODEL`) and `.env.example` if present.
- The plan: item 11 done, open question 3 closed.
- `intent/corn-decode-verification`: closed as won't-do, with a one-line reason at the top.
- ADR 0001 is history; it's left as written.

## What stays

- `backend/app/ml/centering.py`: the centering sub-grade the app shows is classical image measurement and isn't legacy.
- The `/api/grade` request and response schemas, field for field: `SubGrades.corners/edges/surface` with `grade`/`hint`/`flags`. `frontend/src/app/api-client.ts` is unchanged.
- `features.py`'s `export_onnx` (it exports the DINOv2 feature graph, not the CNN), `train_probe`, `train_face_guard`, `ml.eval`, `precrop`, `split`, `collect`.
- DB columns written by `precompute_factors` (factor sub-grades): left in place. No DB write, no migration.
- Dependencies: no package is uninstalled. Removing now-unused entries from `pyproject.toml` is Q4.

## Steps

1. **Import sweep (read-only).** For every candidate file, list its importers (static imports, `python -m` entry points named in docs, tests, CI, and string references like `"ml.training.train"`). Produce `test-artifacts/retire_legacy/sweep.md`: go / stay / move, each with a reason. Anything with a non-legacy importer stays or has the needed piece moved, never broken.
2. **Remove** per the sweep. Keep `inference.py`'s probe path byte-for-byte where possible; simplify only what the removal makes dead.
3. **Prove nothing visible changed:**
   - Before removing anything, grade 20 test-split pairs through `/api/grade` (TestClient, real model, CPU) and save the JSON responses. Include at least one reject (422) and one with notes.
   - After removal, the same 20 pairs must give identical JSON apart from timing fields.
   - `/api/health` keeps its keys, except the legacy-only metadata, which is listed in the PR.
   - One `ml.eval.evaluate --split val --limit 200` run before and after gives identical predictions. Use val, not test, so the test split isn't touched.
4. **Checks:** the fast suite, `ruff check .`, the slow tests that still exist, the frontend `npm test` / `npm run build` (should be untouched), and a CI-equivalent run if CI references a removed file.
5. **Docs** as above, plus the fast-suite baseline in CLAUDE.md.

## Out of scope

- Any model, feature or threshold change. The served model and its outputs must be identical.
- Real corner/edge/surface sub-grades (plan item 10, needs labels).
- Deleting anything under `data/` unless the owner approves it in Q2.
- Rewriting ADR 0001 or old specs, apart from closing [[corn-decode-verification]].

## Acceptance criteria

- [x] `sweep.md` lists every removed and kept candidate with its reason; no remaining code imports a removed module (`python -m compileall` / `ruff` / a grep for each removed module name are clean).
- [x] `GradeInferenceService` has no legacy path, `model_path`, TTA or `ALLOW_LEGACY_MODEL`; a missing or broken probe still gives 503 with `probe_load_error` (existing tests pass).
- [x] The 20 `/api/grade` responses and the val eval predictions are identical before and after.
- [x] The API schema and `api-client.ts` are unchanged; the frontend tests and build pass.
- [x] The fast suite and `ruff` pass; CLAUDE.md, README and the plan are updated; [[corn-decode-verification]] is closed.

## Owner questions

1. **Scope:** remove the legacy *training* code (train/lightning/calibrate/export/configs/precompute_factors) as well as the serving path? *Recommendation:* yes, both. Retraining the CNN isn't planned, and git history keeps it.
2. **Legacy files under `data/models/`:** `grade_model.onnx` / `.pt` / `.json` (~214 MB) and `checkpoints/` (5.3 GB). Options:
   - (a) leave them;
   - (b) move them to `data/backups/legacy-cnn-2026-11-01/`;
   - (c) delete them.

   *Recommendation:* (b). Nothing loads them after this change, and moving keeps them recoverable while making `data/models/` honest. It's a single move with a listing before and after.
3. **`export_for_annotation.py`** (samples cards for hand-labelling in CVAT/Label Studio): delete it with the rest, or keep it, minus the `multitask` import, for future sub-grade labels (plan item 10)? *Recommendation:* delete it. It's easy to rewrite against the current schema if item 10 happens.
4. **Dependencies:** drop now-unused entries from `pyproject.toml`, likely `pytorch-lightning`, `albumentations` and `wandb`, if the sweep confirms nothing else uses them? Nothing is uninstalled from the venv. *Recommendation:* yes, in the same PR, with a fresh-install check left to CI.

## Owner decisions (2026-11-01)

1. **Remove both** the serving path and the legacy training code.
2. **Move** `data/models/grade_model.{onnx,pt,json}` and `data/models/checkpoints/` to `data/backups/legacy-cnn-2026-11-01/`, with a file listing (names, sizes) before and after, saved to `test-artifacts/retire_legacy/`. This is the only approved write under `data/`: no deletes, no DB writes. Check no other Python job is running first.
   - If the sweep finds something non-legacy that reads `checkpoints/` or `logs/`, don't move it; report instead.
   - `data/models/logs/` and `training.log` are left in place unless the sweep shows they're CNN-only.
3. **Delete** `export_for_annotation.py`.
4. **Drop unused dependencies** from `pyproject.toml` in the same PR, only those the sweep shows nothing else imports. Nothing is uninstalled.

## Facts vs assumptions

- **Fact:** `api/grade.py` reaches `condition.py` only when `pred.region_grades is None`, i.e. the legacy model. The centering sub-grade comes from `centering.py`.
- **Fact:** `defects.py` has no importers. `precompute_factors.py` exists to supervise the CNN's factor heads.
- **Fact:** the `export` hits in `features.py` / `inference.py` are `features.export_onnx` / wording, not imports of `ml.training.export`.
- **Assumption:** nothing outside this repo imports these modules.
- **Assumption:** CI installs from `pyproject.toml`, so a dropped dependency that's still needed would fail CI rather than production.

## Implementation (2026-11-01)

Artifacts in `test-artifacts/retire_legacy/`: `sweep.md`, `compare.md`, `before/` / `after/`, `eval_before/` / `eval_after/`, `move_before.txt` / `move_after.txt`, `fast_suite.log`.

- **Removed:** `backend/app/ml/{model,multitask,defects,condition}.py`; `ml/training/{train,lightning_module,calibrate,export,metrics}.py` and `configs/`; `ml/data/{precompute_factors,export_for_annotation}.py`; `db.factor_label_coverage`; the CNN-only parts of `dataset.py` (Dataset/DataModule, transforms, sampler, factor fields); in `inference.py` the legacy load/predict path, `model_path`, `kind`, TTA and the `ALLOW_LEGACY_MODEL` fallback; in `grade.py` the condition fallback and the fallback-crop notes (only reachable with the legacy model: with the probe a face with no quad is a 422 first); `Settings.active_model_path`, `allow_legacy_model`, `wandb_*`; `evaluate --model-path` / `--tta` and `meta.tta`. `ml/training/metrics.py` wasn't listed in the spec but was used only by the Lightning module.
- **Kept:** the probe path in `inference.py` (unchanged apart from the selection code around it), `centering.py`, `load_records` / `precrop_coverage`, `db.update_factor_subgrades` and the `*_subgrade` columns, `features.export_onnx`, `train_probe`, `train_face_guard`, `ml.eval`. `GradePrediction.region_grades` is now a required field.
- **Proof:** the 20 `/api/grade` responses (5 × 422, 12 with notes) and `/api/health` are byte-identical before and after; the val eval (`--limit 200`) gives identical predictions and an identical report apart from latency and the removed `meta.tta`. `/api/health` lost no key (the served probe never showed legacy metadata).
- **Q2 move:** `grade_model.{onnx,pt,json}` + `checkpoints/` renamed (same volume) to `data/backups/legacy-cnn-2026-11-01/`: 20 files, 5,904,147,827 bytes before and after; the API is ready afterwards with the same health. `data/models/logs/` and `training.log` are CNN-only (written by `train.py`) but not named by decision 2, so they stay.
- **Q4 dependencies:** dropped `pytorch-lightning`, `albumentations`, `wandb`, `coral-pytorch` (none imported). Kept `torchvision` (`import timm` imports it; CI pins its CPU wheel) and `onnx` (not made unused by this change). Nothing uninstalled. CI's `WANDB_MODE` env removed.
- **Tests:** deleted `test_model.py`, `test_training_setup.py`, `test_onnx_parity.py`, `test_condition.py`; legacy cases removed from the others, probe cases kept. `check_training_inputs`: its raw-photo contract is now pinned against `train_probe.load_rows`; its 90% precrop-coverage check had no probe equivalent and went with `train.py` (adding one to `train_probe` would change probe training). `ml/tests/conftest.py::models_dir` now gates on the probe files, so the probe regression gate doesn't skip forever.
- **Checks:** fast suite 1560 passed, 1 skipped, 1 xfailed (was 1681 / 1 / 2; the xfail was in `test_condition.py`); `ruff check .` and `compileall` clean; slow/model tests in `test_inference`, `test_review_adversarial`, `test_api`, `test_feature_export`, `test_face_guard_training` pass (150); frontend `npm test` (10) and `npm run build` pass, frontend untouched. A grep for every removed module/setting finds only history docs (ADR 0001, old specs, `TESTING.md`, `HANDOFF.md`) and the "Retired CNN" note in CLAUDE.md.
