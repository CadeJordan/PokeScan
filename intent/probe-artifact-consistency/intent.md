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

- Read the `card_features.json` sidecar.
- If both the head metadata and the sidecar have a `feature_spec_hash` and they differ, raise `ValueError` naming both hashes.
- If either side has no hash (artifacts from before this spec), log a WARNING and continue, so existing artifacts keep loading.

### Visible failure

When probe loading raises, keep today's fallback behaviour; changing it is [[cv-ml-improvement-plan]] open question 3. Also:
- store the error string;
- have `metadata` include `"probe_load_error": "<type>: <message>"`. It's additive and appears in `/api/health`.

## Out of scope

- Changing what happens after a probe load failure (fail closed vs legacy). That's an owner decision.
- Any change to features, regions, probes or hyperparameters. Metrics must not move beyond extraction noise.
- Content hashing of crop files (size + mtime is enough here). Model provenance for the legacy export ([[model-provenance-metadata]]).
- Precrop or detector changes.

## Acceptance criteria

All new tests use `tmp_path` and synthetic data. None of them reads or writes `data/`.

- [ ] Fast suite: `python -m pytest -m "not slow and not visual and not model"` → 0 failed, 2 xfailed, passed = baseline + the new tests. The baseline is 381 on 2026-10-23, plus whatever earlier specs added.
- [ ] `ml/tests/test_probe_cache.py` (new), with an injected `model_factory` and tiny JPEG crops written to `tmp_path`:
  - [ ] A second call with unchanged files extracts nothing: the factory isn't called, or a call counter stays at 0.
  - [ ] Rewriting one crop file with different pixels re-extracts exactly that cert and returns new features for it.
  - [ ] Monkeypatching `FEATURE_SPEC["version"]` discards the whole cache and re-extracts everything.
  - [ ] Certs that are no longer in `rows` are absent from the saved cache.
- [ ] `backend/tests/test_grade_head.py` or a new test: `feature_spec_hash()` is stable across calls and changes when any `FEATURE_SPEC` field changes.
- [ ] Inference tests, using the existing `probe_files` fixture pattern:
  - [ ] Matching hashes load as `kind == "probe"`.
  - [ ] Mismatched hashes give `kind == "legacy"`, and `metadata["probe_load_error"]` contains "feature_spec_hash".
  - [ ] A head without a hash and features without a sidecar still load as probe, with a WARNING logged (use `caplog`).
- [ ] `train_probe` order test: with `export_onnx` monkeypatched to raise, `grade_head.npz` is not written (using a `tmp_path` models dir via a monkeypatched `get_settings`).
- [ ] Real run, after `copy data\models\grade_head.* data\models\backup\` and the same for `cache\region_feats.npz`:
  - `python -m ml.training.train_probe` re-extracts all certs once, writes `card_features.json`, and the head metadata contains `feature_spec_hash`;
  - CV MAE is 0.4467 ± 0.003, CV QWK 0.6201 ± 0.005, test MAE 0.4897 ± 0.005 (2026-10-23 values);
  - a second run extracts 0 certs.
- [ ] `python -m pytest -m model backend/tests/test_review_adversarial.py` still passes against the new artifacts, in particular `test_served_features_match_training_cache`.
- [ ] `ruff check .` exits 0.

## Facts vs assumptions

- **Fact:** the code paths above, and that current artifacts are consistent (80/80 precrops byte-identical to the detector; the CV and test metrics reproduce exactly from the cache).
- **Fact:** the cache stores features as float16 and the served path uses float32. The relative difference is below 2e-3 (`test_served_features_match_training_cache`), so it's left as is.
- **Assumption:** `st_mtime_ns` changes whenever precrop rewrites a crop. It does for `Path.write_bytes` on NTFS. Copying `data/` to another machine can change mtimes, which only causes a safe re-extraction.
- **Assumption:** GPU re-extraction reproduces the cached features to within float16 rounding, so the metrics stay within the tolerances above.
