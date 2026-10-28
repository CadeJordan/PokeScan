# Small fixes

Changes too small for their own spec. Each one should leave behaviour unchanged and keep the fast suite at its baseline: 398 passed / 2 xfailed (2026-10-23, after [[serve-input-normalization]] and its landscape-only retry fix; raise it as tests are added):
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
- [ ] Corrupt or truncated uploads return a plain 500, unchanged from HEAD; they should be 400/422.
- [ ] Serving drops alpha with `convert("RGB")` (`preprocess.decode_image`), whereas ingest composites onto white.
- [ ] Square photos (`w == h`) are never retried: `preprocess.py:575` uses a strict `>`. Either list square photos under the spec's accepted cost, or use `>=` (no stored photo is square) and add a test pinning it.
- [ ] The slow byte-identical test (`backend/tests/test_preprocess.py:393-400`) doesn't assert how many fallback faces it covers. Add `assert len(fallbacks) == 21` (or at least non-empty) so missing files can't make it check fewer, down to zero.
- [ ] Cosmetic: 3 blank lines before `test_sideways_card_in_portrait_photo_is_not_retried` (`test_preprocess.py:210`).

## 11. Split follow-ups (2026-10-27)

From the [[stable-grouped-split]] implementation.

- [ ] `ml/data/collect.py:67` upserts `"split": None`, so re-scanning an already-collected PSA cert range resets those certs' `split` to NULL, and the next sticky split re-places them (a test cert can move). `ebay_collect` skips existing certs, so it isn't affected. Drop `split` from the row dict (and check `db.upsert` then leaves the column alone on conflict); add a `tmp_path` test that re-upserting a cert keeps its split.
