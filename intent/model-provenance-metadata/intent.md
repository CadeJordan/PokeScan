# Model provenance metadata

## Intent

Given any exported `grade_model.onnx`, you can tell from its `grade_model.json` sidecar exactly which code, config and data split produced it, and whether that's reproducible.

## Why

- The active model (`data/models/grade_model.json`, Jun 14) records only shapes, `front_weight`, temperature and calibration NLL. It has no checkpoint path, commit, config or dataset fingerprint.
- `ml/training/export.py` now also writes `checkpoint` and `threshold_offsets`, but still nothing about code or data.
- Eval results (`--save-baseline`, `test-artifacts/eval/*`) are compared across models. Without provenance, a "regression" could be a data-split change rather than a model change.

## Scope

Additive fields in the export metadata dict (`ml/training/export.py`, around line 201):

- `git_commit` (and `git_dirty: bool`), from `git rev-parse HEAD` / `git status --porcelain`; `null` if git is unavailable.
- `training_config`: the hyperparameters already loaded from the checkpoint (`hyper_parameters`), or the config path plus a sha256 of its contents.
- `split_fingerprint`: counts per `split` × `source` from the `certs` table at export time, plus a sha256 of the sorted `(cert_number, split)` pairs.
- `exported_at`: ISO timestamp.
- Key versions: torch, timm, onnxruntime.

`/api/health` already returns the sidecar as `metadata`, so no API change is needed.

## Out of scope

- Setting `deterministic=True` in training. It changes speed and results; that's a separate decision.
- Re-exporting or replacing the current active model.
- Checkpoint-side metadata. Lightning hparams already cover most of it.

## Acceptance criteria

- [ ] Exporting the `smoke.yaml` checkpoint writes a sidecar containing all fields above.
- [ ] Existing keys are unchanged in name and meaning; `/api/health` and `test_api` still pass.
- [ ] Export still succeeds outside a git checkout and with no DB present (fields become `null`, with a warning).
- [ ] A test in `ml/tests/test_training_setup.py` asserts the new keys on its existing tiny export.
- [ ] Two exports from the same commit and DB state give identical `split_fingerprint`.

## Facts vs assumptions

- **Fact:** current sidecar contents and the export code path (read 2026-10-17).
- **Assumption:** reading the DB at export time is acceptable. Use read-only mode from [[test-data-isolation]].
