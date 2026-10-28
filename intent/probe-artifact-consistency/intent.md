# Probe artifact consistency

## Intent

The probe model's three artifacts each record which feature definition they were built with:
- the feature cache (`data/models/cache/region_feats.npz`);
- the feature graph (`card_features.onnx`);
- the head (`grade_head.npz`).

Training never reuses features of a crop that has since been rewritten, and never pairs a new head with an old graph. Serving never silently combines artifacts that disagree.

## Why

From reading the code on 2026-10-23 (see [[cv-ml-improvement-plan]] item 3):

- **Stale cache after re-cropping.** `extract_features` keys the cache on `(cert_number, cropped_front_path, cropped_back_path)` (`ml/training/train_probe.py:90`). Crop paths are deterministic (`ml/data/precrop.py:44-46`: `crops/<original relative path>`). After `precrop --force` with an improved detector, the paths are unchanged, so training reuses features of the old crops while the API computes features from new detections.
- **Feature definition not checked.** The cache is validated only against `BACKBONE` (`train_probe.py:87`). Changing `FEAT_H`, `FEAT_W`, `PATCH`, `CORNER_PATCHES`, `EDGE_PATCHES` or `REGION_NAMES` (`backend/app/ml/features.py:25-44`) leaves the cache, the ONNX graph or both stale.
- **ONNX only exported when missing.** `card_features.onnx` is re-exported only if the file doesn't exist or `--force-onnx` is passed (`train_probe.py:205-207`).
- **Head saved first.** The head is written before the ONNX export (`train_probe.py:201` vs `:207`), so an export failure leaves a new head beside an old graph.
- **Weak load check.** At load, only the feature width D is checked (`grade_head.py:94-105`, called from `inference.py:128-130`). A mismatch in region layout or input size passes.
- **Silent fallback.** If the probe does fail to load, the service quietly serves the legacy CNN (`inference.py:105-120`). That model is worse than always predicting 10 (ADR 0001), and `/api/health` doesn't say why.

Today the artifacts happen to agree:
- sampled precrops are byte-identical to the current detector (80/80);
- `test_served_features_match_training_cache` (marked `model`) checks 6 cached certs against the ONNX graph.

Neither check would catch the cases above.

## Scope

### Feature spec fingerprint (`backend/app/ml/features.py`)

- `FEATURE_SPEC: dict` holds `backbone`, `feat_h`, `feat_w`, `patch`, `corner_patches`, `edge_patches`, `regions` (list) and `version` (int, start at 1). Bump `version` whenever `to_input` or `region_means` changes behaviour.
- `feature_spec_hash() -> str`: the first 12 hex chars of the sha256 of `json.dumps(FEATURE_SPEC, sort_keys=True)`.
- `export_onnx(path)` also writes `path.with_suffix(".json")` containing `{"feature_spec": FEATURE_SPEC, "feature_spec_hash": ..., "exported_at": ISO-8601}`.
  - 2026-10-27 (tester review): it deletes the old sidecar before exporting the graph. Before, a crash between overwriting `card_features.onnx` and writing the sidecar left a new graph beside an old sidecar that still matched the old head. Now a missing sidecar fails closed at load (see Load check), and `train_probe` re-exports when it's missing. Test: `ml/tests/test_review_split_artifacts.py::test_failed_sidecar_write_leaves_no_stale_matching_sidecar`.
  - 2026-10-27 (smallfix 12, owner decision): deleting the sidecar first made an export that failed before writing anything (e.g. CUDA OOM on `--force-onnx`) leave a consistent old pair that no longer loaded. Now graph and sidecar are written to `card_features.onnx.tmp` / `card_features.json.tmp` first; only then is the old sidecar deleted, the graph renamed into place (`os.replace`), and the sidecar last. A failed export leaves the old pair untouched and no temp files; a crash between the renames leaves a missing sidecar, which fails closed. Tests: `backend/tests/test_feature_export.py`.

### Cache (`ml/training/train_probe.py::extract_features`)

- Each entry's key becomes `(cert, front_rel, back_rel, front_sig, back_sig)`, where `sig = f"{st_size}:{st_mtime_ns}"` of the crop file.
  - A changed file means a cache miss and re-extraction.
  - Entries whose key isn't in the current rows are dropped on save, so the cache doesn't grow without bound.
- The cache file stores `feature_spec_hash`. If it's missing or different, discard the whole cache with a log line saying why.
  - The existing cache has no hash, so the first run after this change re-extracts every cert. That's expected and takes minutes on the GPU.
- Add `extract_features(..., model_factory=build_region_model)` so tests can inject a tiny torch module and avoid the DINOv2 download.

### Train order and metadata (`train_probe.main`)

1. Export the ONNX graph if `--force-onnx`, or if its sidecar is missing, or if the sidecar's hash differs from `feature_spec_hash()`.
2. Then save the head.

Head metadata gains `feature_spec_hash` and `feature_spec`. Keep every existing key; `/api/health` already returns this metadata.

### Load check (`backend/app/ml/inference.py::_load_probe`)

- Read the `card_features.json` sidecar, and compare with `feature_spec_hash()` of the running code (which runs `to_input`).
- Head metadata has a `feature_spec_hash`: it must equal the code's hash, and the sidecar must exist, parse as a dict and have the same hash. Any violation, including a missing or unparseable sidecar, raises `ValueError` naming what disagreed (e.g. `feature_spec_hash: head 0123456789ab (grade_head.npz) != code 428e7a366e2e`, `card_features.json missing`), so the service fails closed.
- Head has no hash (no `feature_spec_hash` key; artifacts from before this spec): log a WARNING and continue, whatever the sidecar says, so existing artifacts keep loading. Exceptions: a sidecar whose hash differs from the code's raises too, and so does a sidecar that can't be read at all (OS error; see the smallfix 12 note below).
- 2026-10-27 (smallfix 12, owner decision): only a head *without* the `feature_spec_hash` key counts as pre-spec. A key whose value isn't a non-empty string (`""`, `null`, `0`) raises `ValueError` (`head '' (grade_head.npz) is not a hash`). An OS-level error reading the sidecar (`OSError`, e.g. a directory at that path) fails closed for every head, including a head without a hash, with the error in `probe_load_error`. Tests: `backend/tests/test_retest_a92911c.py::test_falsy_or_non_string_head_hash_fails_closed`, `::test_unreadable_sidecar_os_error_fails_closed_for_every_head`, `::test_sidecar_read_os_error_fails_closed_for_no_hash_head`.
- 2026-10-27 (tester review): the first version compared only the head with the sidecar, never with the code, so after a `FEATURE_SPEC` bump without retraining the API served an old head + graph on new preprocessing. Tests: `backend/tests/test_review_split_artifacts.py::test_artifacts_from_another_feature_spec_than_the_code_are_refused` and `backend/tests/test_inference.py::test_feature_spec_disagreement_fails_closed`.

### Fail closed (owner decision, [[cv-ml-improvement-plan]] open question 3)

Model selection in `GradeInferenceService.__init__`:

- An explicit `model_path` loads the legacy model, as today (`ml.eval.evaluate --model-path`, tests). It's a deliberate choice, so it's not gated.
- Otherwise the probe is the only model served. If either probe file is missing, or loading raises (corrupt file, D mismatch, `feature_spec_hash` mismatch), the service is **not ready**: `/api/grade` returns 503 (the existing `service.ready` check) and the legacy model is not loaded.
- `ALLOW_LEGACY_MODEL=1` (a `Settings` field, `allow_legacy_model: bool = False`) restores today's behaviour: when the probe is missing or fails, load the legacy `active_model_path`, logging a WARNING.
- Either way, store the error and have `metadata` include `"probe_load_error": "<type>: <message>"`, e.g. `"FileNotFoundError: grade_head.npz"` or `"ValueError: feature_spec_hash a1b2… != c3d4…"`. A missing file counts as an error here. `/api/health` then shows `model_loaded: false` (or `true` with the legacy opt-in) plus the error. It's additive.
- Update the 503 detail in `api/grade.py` to point at `/api/health` for the reason.

### Docs

- README and CLAUDE.md: the API needs the probe files; `ALLOW_LEGACY_MODEL=1` opts into the legacy CNN, next to the `smoke.yaml` end-to-end check. The opt-in only applies when no probe model loads; to serve a freshly exported legacy model, pass its `model_path` or move the probe files aside (2026-10-27 re-test). `smoke.yaml` itself is unchanged.
- The served model directory needs `card_features.json` beside `card_features.onnx` after the real run. Artifacts without it still load, with a WARNING, as long as the head has no `feature_spec_hash` either (2026-10-27: a head with a hash and no sidecar fails closed).

## Out of scope

- Retiring the legacy code path ([[cv-ml-improvement-plan]] item 11; a follow-up spec).
- Any change to features, regions, probes or hyperparameters. Metrics must not move beyond extraction noise.
- Content hashing of crop files (size + mtime is enough here). Model provenance for the legacy export ([[model-provenance-metadata]]).
- Precrop or detector changes.

## Acceptance criteria

All new tests use `tmp_path` and synthetic data. None of them reads or writes `data/`.

- [x] Fast suite: `python -m pytest -m "not slow and not visual and not model"` → 0 failed, 2 xfailed, passed = baseline + the new tests. The baseline is 398 on 2026-10-27 (after PR #4), plus whatever [[stable-grouped-split]] adds.
  - The only existing tests allowed to change are those pinning the silent fallback: `test_unloadable_probe_falls_back_to_legacy` (`backend/tests/test_inference.py`), `test_selection_head_only_is_legacy` and `test_selection_features_only_is_legacy` (`backend/tests/test_review_adversarial.py`). Rewrite them to expect "not ready + `probe_load_error`", and add a variant of each with `ALLOW_LEGACY_MODEL=1` expecting `kind == "legacy"`. Tests that need the legacy model pass `model_path` or set the opt-in.
  - 2026-10-27: 457 passed (421 + 36 new), 2 xfailed, 184 deselected, 117 s. The three tests are now `test_unloadable_probe_fails_closed` (parametrized: corrupt head, D mismatch, and a new `feature_spec_hash` mismatch case), `test_selection_head_only_is_not_ready` and `test_selection_features_only_is_not_ready`, each with a `..._legacy_opt_in...` variant. Their legacy model is a tiny ONNX graph (`tiny_legacy_onnx` in `backend/tests/conftest.py`) and the opt-in is set through the env (`serve_env` fixture), so they read nothing under `data/`. No other assertion changed: `_probe_service` still asserts `kind == "legacy"` (the default `kind` until a probe loads; only its comment changed), and the `model`-marked `test_served_features_match_training_cache` now unpacks the first 3 of the 5 key fields.
- [x] `ml/tests/test_probe_cache.py` (new), with an injected `model_factory` and tiny JPEG crops written to `tmp_path`:
  - [x] A second call with unchanged files extracts nothing: the factory isn't called, or a call counter stays at 0.
  - [x] Rewriting one crop file with different pixels re-extracts exactly that cert and returns new features for it.
    - The test also moves the file's mtime 2 s forward, so it doesn't depend on the filesystem's timestamp granularity. `test_touched_crop_reextracts` pins the mtime half of the signature on its own.
  - [x] Monkeypatching `FEATURE_SPEC["version"]` discards the whole cache and re-extracts everything.
    - Plus `test_cache_without_spec_hash_is_discarded`: the pre-spec cache format (`backbone`, 3-field keys) is discarded, as on the real run.
  - [x] Certs that are no longer in `rows` are absent from the saved cache.
  - Also `test_export_onnx_writes_feature_spec_sidecar` (a tiny module in place of DINOv2).
- [x] `backend/tests/test_grade_head.py` or a new test: `feature_spec_hash()` is stable across calls and changes when any `FEATURE_SPEC` field changes.
  - `test_feature_spec_hash_is_stable`, `test_feature_spec_hash_changes_with_every_field` (one case per field, 8).
- [x] Inference tests, using the existing `probe_files` fixture pattern:
  - [x] Matching hashes load as `kind == "probe"`.
    - And log no `feature_spec_hash` warning.
  - [x] Mismatched hashes give `ready == False`, and `metadata["probe_load_error"]` contains "feature_spec_hash". With `ALLOW_LEGACY_MODEL=1` and a legacy model present, `kind == "legacy"` and the error is still in `metadata`.
    - Error text: `ValueError: feature_spec_hash aaaaaaaaaaaa (h.npz) != bbbbbbbbbbbb (f.json): retrain with ...`.
    - 2026-10-27, after the tester review's load-check fix: `ValueError: feature_spec_hash: head aaaaaaaaaaaa (h.npz) != code 428e7a366e2e; f.json bbbbbbbbbbbb != head aaaaaaaaaaaa: retrain with ...` (every disagreement, `; `-separated).
  - [x] Missing probe files give `ready == False` and a `probe_load_error`, even when a legacy model exists. An explicit `model_path` still loads it.
    - Head, features or both missing, e.g. `FileNotFoundError: grade_head.npz, card_features.onnx` (file names, not paths). With the opt-in the legacy model loads and predicts.
  - [x] API: with the service not ready, `/api/grade` returns 503 and `/api/health` shows `model_loaded: false` and the `probe_load_error`.
    - `test_unloadable_probe_gives_503_and_health_says_why` (`backend/tests/test_api.py`) uses a real service with no probe files; the 503 detail mentions `/api/health`.
  - [x] A head without a hash and features without a sidecar still load as probe, with a WARNING logged (use `caplog`).
    - Parametrized: neither has a hash, head only, sidecar only.
    - 2026-10-27 (tester review): now a head without a hash and no sidecar, a sidecar matching the code, or an unparseable sidecar. "Head only" and a sidecar from another spec than the code now fail closed (`test_feature_spec_disagreement_fails_closed`).
- [x] `train_probe` order test: with `export_onnx` monkeypatched to raise, `grade_head.npz` is not written (using a `tmp_path` models dir via a monkeypatched `get_settings`).
  - `test_failed_export_writes_no_head`. `test_head_records_feature_spec_and_export_follows_the_sidecar` covers the export rule (exported when the graph is missing; not when the sidecar matches; again with `--force-onnx`, another sidecar hash or no sidecar) and the head metadata's `feature_spec_hash` and `feature_spec`.
- [x] Real run, once, after both this spec and [[stable-grouped-split]] are implemented, and after `copy data\models\grade_head.* data\models\backup\` and the same for `card_features.onnx` and `cache\region_feats.npz`:
  - `python -m ml.training.train_probe` re-extracts all certs once, writes `card_features.json`, and the head metadata contains `feature_spec_hash`, `test_seen_cards` and `n_test_seen`;
  - CV MAE is 0.4467 ± 0.003, CV QWK 0.6201 ± 0.005, test MAE 0.4897 ± 0.005 (2026-10-23 values);
  - a second run extracts 0 certs.
  - 2026-10-27: no python/sqlite process running beforehand. Backup in `data/models/backup/` (sizes and SHA-256 match the originals): `grade_head.npz` 139,506 B, `grade_head.json` 925 B, `card_features.onnx` 346,365,289 B, `region_feats.npz` 89,203,100 B (plus the old `probe_report.json`).
  - Run 1: "feature_spec_hash None, not 428e7a366e2e: re-extracting every cert", 0 cached / **2,611** extracted (GPU, 5.4 min); graph re-exported ("card_features.json missing"). `feature_spec_hash` **428e7a366e2e**.
  - Metrics identical to the 2026-10-23 values to 4 decimals: CV MAE **0.4467**, CV QWK **0.6201** (n 2,223), test MAE **0.4897** (n 388, QWK 0.5241). `n_test_seen` 251; `test_seen_cards` MAE 0.4143 (n 251), `test_unseen_cards` MAE 0.6277 (n 137).
  - Run 2: 2,611 cached, **0** extracted, no re-export (cache and graph files untouched), same metrics.
- [x] `python -m pytest -m model backend/tests/test_review_adversarial.py` still passes against the new artifacts, in particular `test_served_features_match_training_cache`.
  - 2026-10-27, after the real run, the whole `-m model` suite: 13 passed, 0 skipped (17 min; 4 in `test_review_adversarial.py` including `test_served_features_match_training_cache` and `test_onnx_matches_torch_region_model` against the re-exported graph, 2 in `test_inference.py`, 4 in `test_model_performance.py`, 3 in `test_onnx_parity.py`). The legacy-model tests pass `model_path`, so fail-closed doesn't affect them.
- [x] `ruff check .` exits 0.

## Facts vs assumptions

- **Fact:** the code paths above, and that current artifacts are consistent (80/80 precrops byte-identical to the detector; the CV and test metrics reproduce exactly from the cache).
- **Fact:** the cache stores features as float16 and the served path uses float32. The relative difference is below 2e-3 (`test_served_features_match_training_cache`), so it's left as is.
- **Assumption:** `st_mtime_ns` changes whenever precrop rewrites a crop. It does for `Path.write_bytes` on NTFS. Copying `data/` to another machine can change mtimes, which only causes a safe re-extraction.
- **Assumption:** GPU re-extraction reproduces the cached features to within float16 rounding, so the metrics stay within the tolerances above.

## Owner decisions (2026-10-27)

- **Fail closed** (open question 3, decided 2026-10-23): replaces this spec's original "keep today's fallback" scope; see "Fail closed" above. Missing probe files count as a failure too, so deleting a file can't silently bring back the legacy CNN. An explicit `model_path` still loads it, and `ALLOW_LEGACY_MODEL=1` opts back in.
- **Existing fallback tests** are rewritten as listed under the fast-suite criterion; no other existing assertion changes.
- **Real run:** approved, once, at the end of the round, after backing up the current artifacts to `data/models/backup/`.
- **Delivery:** one PR together with [[stable-grouped-split]], which lands first (both edit `train_probe.py`'s report and metadata).
