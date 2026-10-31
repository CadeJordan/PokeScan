# Small fixes

Changes too small for their own spec. Each one should leave behaviour unchanged and keep the fast suite at its baseline: 520 passed / 2 xfailed (2026-10-27, after [[stable-grouped-split]], [[probe-artifact-consistency]], their tester-review fixes, the re-test suites and smallfix 11/12; raise it as tests are added):
`python -m pytest -m "not slow and not visual and not model"`

## 1. Lint baseline (ruff: 25 errors → 0)

Needed before [[ci-fast-suite]] can gate on `ruff check .`.

Config (`pyproject.toml`): these patterns are idiomatic here, so allow them rather than rewriting code.
- [x] `[tool.ruff.lint.flake8-bugbear] extend-immutable-calls = ["typer.Option", "typer.Argument"]` clears all 14 B008 errors (typer CLI defaults).
- [x] Allow N812 for the conventional aliases `import albumentations as A` (`ml/data/dataset.py:16`) and `import torch.nn.functional as F` (`ml/training/calibrate.py:25`), via `ignore` or `# noqa: N812`.
  - Done with `# noqa: N812` on the two import lines (narrower than a global ignore).
- [x] `backend/app/ml/preprocess.py:491` `M` (OpenCV transform-matrix convention): `# noqa: N806`, or rename to `warp_m`.
  - Done with `# noqa: N806` (kept the OpenCV `M` name).

Code (auto-fixable with `ruff check --fix`; review the diff):
- [x] UP035 in `backend/app/main.py:11`, `ml/data/db.py:8`, `ml/data/ebay_client.py:20`: import from `collections.abc`.
- [x] UP037 in `ml/data/ebay_client.py:99`: unquote the annotation.
- [x] I001 + F401 in `backend/app/ml/multitask.py:18,25`: sort imports, drop the unused `NUM_THRESHOLDS`.

Code (manual):
- [x] E741 in `backend/app/ml/centering.py:40`: rename `l` to `left` (or similar).
- [x] SIM105 in `backend/app/core/config.py:65`: use `contextlib.suppress(OSError)`.

Done when `ruff check .` exits 0.

## 2. Deprecated 413 status constant

- [x] `backend/app/api/grade.py:67`: `HTTP_413_REQUEST_ENTITY_TOO_LARGE` → `HTTP_413_CONTENT_TOO_LARGE`. It's the only warning from our own code in the test run. Check that the installed Starlette has the new name (fastapi 0.136.1 does).

## 3. Misleading comment in the model

- [x] `backend/app/ml/model.py`, in `GradeModel.__init__`: the comment says outputs are "guaranteed monotonic via softplus", but there's no softplus. Monotonicity comes from `cummin` in `decode_corn_logits`. Fix the comment only; the decoding question itself is [[corn-decode-verification]].

## 4. Commit the agent-skills setup

- [x] Commit the already-written `docs/agents/` (issue tracker, triage labels, domain docs), the `CLAUDE.md` updates and this `intent/` folder on `CadeJordan/ScoreImprovements`. The draft `.github/workflows/ci.yml` is covered by [[ci-fast-suite]].
  - Committed as e57d5bf, before the other fixes, so these ticks and notes landed in a later commit. `.github/workflows/ci.yml` left untracked.

## 5. GitHub CLI auth (user action)

- [x] Run `! gh auth login`. `gh` is currently unauthenticated, so the issue-tracker workflow in `docs/agents/issue-tracker.md` can't work, and repo visibility (relevant to [[git-history-image-purge]]) can't be checked.
  - Already authenticated (account CadeJordan, scopes gist/read:org/repo/workflow) as of 2026-10-18; nothing to do.
- [x] Then create the five triage labels if they're missing: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`.
  - `wontfix` already existed; created `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human` on CadeJordan/PokeScan.

## 6. README venv instructions

- [x] `README.md` setup tells you to create and activate `.venv`. Until [[pinned-dev-environment]] lands, add a one-line note that the current working env is the global pyenv 3.11.6, or fold this into that spec.

## 7. Stop tracking graphify cache

- [x] `graphify-out/cache/` (dozens of AST/semantic JSON blobs) is committed. Add `graphify-out/cache/` to `.gitignore` and `git rm -r --cached graphify-out/cache`. Keep `graph.json` / `GRAPH_REPORT.md` tracked if they're wanted. Ask before removing them.
  - Done. `graph.json`, `GRAPH_REPORT.md` and the other top-level `graphify-out/` files stay tracked.

## 8. Frontend lint script

- [x] `frontend/package.json` still has `"lint": "next lint"`, but the draft CI notes `next lint` was removed in Next 16 and calls `npx eslint src` instead. If the installed Next is 16+, change the script to `eslint src` so `npm run lint` (documented in `CLAUDE.md`) works locally.
  - Installed Next is 16.3.5, so the script is now `eslint src`; `npm run lint` exits 0 with no findings. Note: `eslint-config-next` is still pinned at 15.4.6 (left as is).

## 9. Global env sanity check

- [x] Run `python -m pip check` on the global pyenv env and record the result in [[pinned-dev-environment]] before freezing constraints from it.
  - One conflict: `opencv-contrib-python 4.13.0.92` (pulled in by global `mediapipe`) wants `numpy>=2`, but numpy is 1.26.4. It's outside `pokescan[dev]`. Recorded in `intent/pinned-dev-environment/intent.md`.

## 10. CV/ML review follow-ups (2026-10-23)

From [[cv-ml-improvement-plan]]. The fast-suite baseline is now **398 passed / 2 xfailed / 183 deselected** (2026-10-23, 103 s), as in this file's header.

- [x] Update the header of this file to the new baseline.
- [x] `backend/app/api/grade.py:94,108,128`: `grade_card` is `async def` but calls `service.predict` (~2.7 s of CPU with the probe model), `compute_centering_subgrade` and `compute_condition_subgrades` synchronously. That blocks the event loop, so `/api/health` and every other request stall during a grade. Wrap the three calls in `await starlette.concurrency.run_in_threadpool(...)`. `GradeInferenceService` already serialises the ONNX session with its `_lock`. Done when the existing API tests pass unchanged.
  - Done; all existing API tests pass unchanged. With [[serve-input-normalization]], sub-grades get the model's crops, so the three calls are `run_in_threadpool(service.predict, ...)` and `run_in_threadpool(compute_*_subgrade, *faces)`.
- [x] `backend/app/ml/defects.py`: imported nowhere in `backend/` or `ml/`. Its docstring says "the API returns an empty defect list", but the API has no defects field. Change the docstring to "not wired into the API (Phase C hook for the legacy CNN)". Deleting the file is part of retiring the legacy path ([[cv-ml-improvement-plan]] item 11, open question 3).
- [x] `CLAUDE.md` Commands: "API (needs data/models/grade_model.onnx, else /api/grade returns 503)" is out of date. The API needs `grade_head.npz` + `card_features.onnx` (probe) or `grade_model.onnx` (legacy). In Architecture, "**Export** … fits temperature scaling" applies to the legacy model only; the probe has no post-hoc calibration, and needs none (out-of-fold NLL-optimal T = 1.00).
- [x] `backend/app/ml/centering.py:45` imports the pydantic `CenteringSubGrade` from `backend.app.api.grade`, so the ML module depends on the API layer. Import it lazily inside `compute_centering_subgrade`, or move the schema to a module both can import. Behaviour unchanged.
  - Imported lazily, with a `TYPE_CHECKING` import for the annotation. `import backend.app.ml.centering` no longer loads `backend.app.api.grade`.

Follow-ups from the [[serve-input-normalization]] review (2026-10-23; not done):

- [x] `up3x` detect-time criterion is borderline: p50 is 1.48–1.55× `orig` between runs (1.50× and 1.51× after the landscape fix), against a ≤ 1.5× bar.
  - Retest: this is noise, not a regression. Interleaved against HEAD on 250 faces: up3x p50 is 157 ms now vs 435 ms at HEAD, found rate 99.6% vs 76%; the ratio's 95% CI is [1.465, 1.523]. Restate the bar as "up3x p50 ≤ HEAD's up3x p50 and ≤ orig + 75 ms", and untick or reword `serve-input-normalization` intent.md's detection-report criterion until then.
  - Done: the spec's criterion is restated to that bar, which the recorded run meets (161 ms vs `orig` + 75 = 182 ms and HEAD's 435 ms).
- [ ] `run_in_threadpool` in `backend/app/api/grade.py` gives unbounded concurrency (anyio's default 40 threads, ~36 MB per 12 MP decode); consider a `CapacityLimiter`.
- [ ] Sideways crops that land upside down report swapped centering sides (left/right, top/bottom); needs the [[face-side-orientation-guard]].
- [x] Corrupt or truncated uploads return a plain 500, unchanged from HEAD; they should be 400/422.
  - Done in [[reject-unusable-uploads]]: 400 `"<face>: not a readable image (corrupt or truncated file)"`.
- [ ] Serving drops alpha with `convert("RGB")` (`preprocess.decode_image`), whereas ingest composites onto white.
- [ ] Square photos (`w == h`) are never retried: `preprocess.py:575` uses a strict `>`. Either list square photos under the spec's accepted cost, or use `>=` (no stored photo is square) and add a test pinning it.
- [ ] The slow byte-identical test (`backend/tests/test_preprocess.py:393-400`) doesn't assert how many fallback faces it covers. Add `assert len(fallbacks) == 21` (or at least non-empty) so missing files can't make it check fewer, down to zero.
- [ ] Cosmetic: 3 blank lines before `test_sideways_card_in_portrait_photo_is_not_retried` (`test_preprocess.py:210`).

## 11. Split follow-ups (2026-10-27)

From the [[stable-grouped-split]] implementation.

- [x] `ml/data/collect.py:67` upserts `"split": None`, so re-scanning an already-collected PSA cert range resets those certs' `split` to NULL, and the next sticky split re-places them (a test cert can move). `ebay_collect` skips existing certs, so it isn't affected. Drop `split` from the row dict (and check `db.upsert` then leaves the column alone on conflict); add a `tmp_path` test that re-upserting a cert keeps its split.
  - Done 2026-10-27. The base row is now built by a pure `collect._details_row(details)`, without `split`. A new cert gets the schema default (NULL), and `upsert_cert` only updates the keys in the row, so a re-scan keeps the split. Tests: `ml/tests/test_collect_rescan.py::test_rescan_keeps_assigned_split` (fresh cert NULL; `test`, crop and sub-grade columns survive a re-upsert) and `::test_details_row_omits_pipeline_owned_columns`. `ebay_collect` checks `cert_exists` before every upsert, so it still sends `"split": None` but only for new rows; left as is.
  - [x] Was still open after the split fix: a PSA re-scan always rewrites `front_url`/`back_url`/`front_path`/`back_path`/`has_images`. If the images call returns nothing or a CDN download fails, an already-collected cert loses its image paths and drops to `has_images=0` (the files stay on disk). Its `cropped_*`/`crop_*`/`*_subgrade` survive but point at a cert the dataset no longer reads. When the downloads succeed, the images are re-downloaded and overwritten, so the crops and sub-grades can be stale relative to the new files. `collect` doesn't skip existing certs at all, so a re-scan also spends PSA quota on certs it already has. Fix: skip certs that are already collected with images, or on a failed fetch leave the image columns out of the row.
    - Done 2026-10-27 (owner decision below). `_run` checks `collect._collected_with_images(conn, cert)` (a row with `has_images=1`) before queueing a cert, so such a cert gets no PSA call and its row isn't touched; the run logs how many it skipped. Certs stored without images (non-Pokémon, or no images last time) are fetched as before. `--refresh` (on `collect` and `collect_all`, which forwards it) re-fetches them. Skipped certs count as evidence of an issued range, so they also block the "zero hits" 404 abort. README and HANDOFF mention it. Tests in `ml/tests/test_collect_rescan.py` (tmp_path DB, stubbed PSA client): `::test_rescan_skips_cert_collected_with_images` (no PSA call; image paths, split, crop and sub-grade columns unchanged), `::test_rescan_with_refresh_refetches_collected_cert`, `::test_rescan_refetches_cert_without_images`, `::test_skipped_certs_do_not_count_towards_the_404_abort`, `::test_collect_all_forwards_refresh`.

Owner decision (2026-10-27): **skip PSA certs already collected with images**, with a `--refresh` flag for a deliberate re-fetch. In this round's PR.

## 12. Probe-artifact follow-ups (2026-10-27 re-test)

From the re-test of [[probe-artifact-consistency]]; none is a bug, all fail in the safe direction or need hand-edited metadata.

- [x] `backend/app/ml/features.py` `export_onnx` deletes the old `card_features.json` before exporting. If `torch.onnx.export` then fails before writing anything (e.g. CUDA OOM on `--force-onnx`), the old graph, head and hash are still consistent, but the next API start fails closed on the missing sidecar. Write graph and sidecar to temp names and rename instead.
  - Done 2026-10-27. The graph goes to `card_features.onnx.tmp` and the sidecar to `card_features.json.tmp`; only once both are written is the old sidecar deleted, then the graph `os.replace`d, then the sidecar. A `finally` removes any temp file left by a failure. Every crash window leaves either the old consistent pair or a missing sidecar (fails closed; `train_probe` re-exports). Tests: `backend/tests/test_feature_export.py::test_export_failing_before_writing_leaves_old_pair_loadable`, `::test_export_failing_after_graph_temp_leaves_old_pair_and_no_temp_files` (both byte-compare the old pair and load it through `GradeInferenceService`), `::test_export_replaces_both_files_and_leaves_no_temp_files`. Two re-test tests that pinned the old delete-first behaviour now assert the old pair survives: `ml/tests/test_retest_a92911c.py::test_export_failing_in_torch_export_keeps_old_pair_and_train_probe_skips_reexport` (renamed from `..._removes_sidecar_and_train_probe_reexports`) and `::test_train_probe_with_real_export_failing_writes_no_head`.
- [x] `backend/app/ml/inference.py` `_check_feature_spec` treats a falsy head hash (`""`, `null`) as a pre-spec head (WARNING and load) rather than failing closed. Only reachable with hand-edited or corrupt `grade_head.json`.
  - Done 2026-10-27. Only an absent `feature_spec_hash` key means pre-spec; a present value that isn't a non-empty string raises `ValueError` (`head '' (grade_head.npz) is not a hash`). Tests: `backend/tests/test_retest_a92911c.py::test_falsy_or_non_string_head_hash_fails_closed` (replaces `test_falsy_head_hash_counts_as_no_hash`; `""`, `null`, `0`, `false`, `[]`, `428`, with and without a matching sidecar) and `::test_absent_head_hash_key_still_counts_as_pre_spec`.
- [x] `_check_feature_spec` catches only JSON/Unicode errors when reading the sidecar; an OS-level read error fails closed even for a no-hash head, where the spec says such heads load "whatever the sidecar says". Safe direction; decide which the spec should say.
  - Done 2026-10-27. Behaviour kept (the `OSError` propagates, so it fails closed for every head, named in `probe_load_error`); the docstring and the Load check bullets of [[probe-artifact-consistency]] now say so. Tests: `backend/tests/test_retest_a92911c.py::test_unreadable_sidecar_os_error_fails_closed_for_every_head` (a directory at the sidecar path; with and without a head hash) and `::test_sidecar_read_os_error_fails_closed_for_no_hash_head` (a plain `OSError` from the read).

Owner decision (2026-10-27), in this round's PR: write the graph and sidecar to temp names and rename (an export that fails before writing anything must leave the old, consistent pair loadable); a falsy head hash is an error (fail closed); an OS-level sidecar read error fails closed, and the spec says so.

## Result card follow-ups ([[result-copy-matches-probe-model]] re-test, 2026-10-30)

- [ ] `frontend/src/components/GradeResult.tsx`: the distribution chart highlights the bar ">grade_int−1" (`isPick`: `i + 2 === grade_int`), which is P(grade ≥ grade_int), shown right under the confidence caption "probability of this exact grade". The two only agree for a 10. Either show per-grade probabilities (differences of `rank_probs`) with the predicted grade highlighted, or relabel. The heading "Distribution over grades" names a cumulative chart.
- [ ] Backend region hints "clean / slight wear / wear / heavy wear" (`backend/app/api/grade.py` `_REGION_HINTS`) read as physical wear, but they are thresholds on a region's soft overall-grade signal. Needs an owner call: they are part of the API response.
- [ ] At 320 px, "generated · <date>" wraps into "PREDICTION" in the result header strip.

## Full-resolution pilot follow-ups ([[full-resolution-pilot]], 2026-10-31)

- [ ] `ml/data/image_io.py` docstring says PSA originals are "3-4 MB". Measured on 1,200 originals: mostly 1430×2400 at ~1 MB (median 1.01 MB); only the rare ~4,740 px ones are 3–4 MB.
- [x] `ml/training/probe.py:105` stratifies folds on `np.minimum(y, 8)`, which pools 9s and 10s into one class and keeps grades below 8 apart. The comment says "≤8 pooled", which is `np.maximum`. So CV folds aren't stratified on 9 vs 10 (on a 9/10-only set, not at all). Fixing it changes every CV number, so re-baseline after. Specced in [[cv-fold-stratification]].
  - Done 2026-10-31: `_folds` stratifies on `np.maximum(y, 7)` (strata ≤7, 8, 9, 10), head retrained and shipped; re-baseline in [[cv-fold-stratification]] Results.
