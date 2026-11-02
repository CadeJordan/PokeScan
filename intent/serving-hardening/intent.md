# Serving hardening: bounded grading, CI test guard, raw-card crop check

## Intent

Close the open follow-ups that carry real risk, in one small PR:
1. Cap how many grades run at once, so a burst of uploads can't run the server out of memory.
2. Make CI fail when a test step finds no tests.
3. Pin the raw-card crop case left open by [[tighter-slab-crops]] with a test, and fix the detector only if the test fails.
4. Tidy: the plan's stale performance table, a test-only helper, model provenance.

No model, threshold or schema change. Served grades stay identical.

## 1. Bounded grading

**Problem:** `/api/grade` sends `service.predict` and `compute_centering_subgrade` to `run_in_threadpool` (`backend/app/api/grade.py:99,125`). The only limit is anyio's default thread limiter (40 tokens), which `/api/health`, file reads and other sync work share. Each predict holds the decoded photos plus DINOv2 activations for two 602×434 faces. That's a few hundred MB, unmeasured. ONNX Runtime already uses all cores per call (`features.py:182`, default intra-op threads), so running more than one or two at once adds memory and latency but no throughput. On a 15.7 GB machine a handful of concurrent uploads can push it into swap or OOM.

**Change:**
- A module-level `anyio.CapacityLimiter(settings.grade_concurrency)` (new setting, default per Q1) wraps the predict and centering calls of one request. They run under one token, so a request never holds the model while waiting for centering.
- A request that can't get a token within `settings.grade_queue_timeout_s` (Q2) gets **503** `"server busy, try again"` with a `Retry-After` header. That's the same status the frontend already handles for "model not loaded", so the API contract and `api-client.ts` don't change. The wait itself is `anyio.fail_after` around acquiring the token, so a timed-out request has done no model work.
- `/api/health` doesn't take a token and stays instant under load. Optionally it reports `grading_in_flight` / `grading_waiting` in its metadata, as additive keys.
- Upload reading and size checks (`_read_image`) happen before the wait, as today, so an oversized upload is still refused at once.
- The limit is per process. Running uvicorn with `--workers N` multiplies it, and the README says so.

**Measure, don't assume** (write-up in this spec):
- Peak RSS of one predict on CPU.
- With N concurrent requests (N = 1, 4, 8, 16) through a real uvicorn on the served model: peak RSS, p50/p95 latency and count of 503s, before and after.
- Synthetic photos or a handful of test-split pairs, read-only. Stop at 16, or earlier if RSS passes ~10 GB, so the measurement can't itself exhaust the machine.

**Tests** (fast, no real model): a fake slow service shows that with the limit at 1, a second concurrent request waits and a third times out to 503 with `Retry-After`. Health answers while a grade is in flight. The token is released after a predict error (400/422/500 paths) and after a client disconnect. The default settings load.

## 2. CI fails on zero tests

**Fact:** `node --test --experimental-strip-types "src/**/*.test.ts"` with no matching files exits **0** (checked on Node 22.14). If the frontend tests move or the glob breaks, CI goes green with no tests run. Pytest already exits 5 on zero collected tests, so the backend step is safe; a test pins that assumption.

**Change:** make `npm test` fail when it runs zero tests. Use a tiny wrapper script `frontend/scripts/test.mjs` that runs `node --test` and fails if the TAP summary's `# tests` is 0, or an equivalent that adds no dependency. Keep the same command name, so CI and docs don't change. Verify both ways: today's suite passes, and a glob with no matches fails.

## 3. Raw-card crop check

**Open question from [[tighter-slab-crops]]:** the last-resort flat-ring step (`preprocess._find_card_quad`, `allow_flat`) can also fire once `best` is already the card. Then a raw card with a plain, flat, dark border and a strongly textured art window of similar brightness could be cropped to its art window. It caused no regressions on the 5,260 PSA faces, but raw uploads weren't measured.

**Change:**
- Add synthetic negative tests to `backend/tests/test_preprocess.py`, next to `test_slab_photo_finds_the_card`: a raw card (no slab) on a plain background with a flat dark border and a textured window. Cover several border/window brightnesses, and one at a slight angle. The card's own outline must be kept (IoU ≥ 0.9 with the card).
- **If they pass:** done; record it.
- **If any fail:** limit the last-resort step (e.g. to the first step, or to a `best` that isn't card-height). Then re-run the 8 slab scenes and the read-only check over all 5,260 PSA faces from tighter-slab-crops: every stored quad must stay identical, using the same comparison script as before. **Any PSA quad that changes stops the work** and goes to the owner. A re-crop would need a new approval. Nothing under `data/` is written either way.

## 4. Tidy

- **Plan table:** replace `intent/cv-ml-improvement-plan.md`'s "How the system performs now" with a short current table and history pointers:
  - head 2026-10-31, `ten_cutoff` 0.56;
  - CV seed 0 MAE 0.4645;
  - test once: MAE 0.443, exact 65.3%;
  - pointers to ADRs 0002/0003, [[tighter-slab-crops]] and the old tables' location in git history.

  Drop the stale "fast suite is 381 passed" line.
- **`dataset.precrop_coverage`:** its only caller is a test, since `train.py` was retired. Remove it and its test; `db.precrop_coverage*` stays, because `precrop` uses it.
- **Provenance:** `train_probe._git_commit` returned null in the last retrain while `train_face_guard`, 30 minutes later, got a hash. Find out why, e.g. a missing `git` on that shell's PATH or a different cwd, and fix it so a missing commit is logged as a warning, not stored silently as null. Don't edit `data/models/grade_head.json` (no write under `data/`); the next retrain records it.

## Out of scope

- Any model, detector or threshold change, except the conditional fix in 3.
- Rate limiting per client, auth, request queuing across processes.
- The phone-photo run (waits for photos) and real sub-grades (plan item 10).

## Acceptance criteria

- [x] Grading is bounded by `grade_concurrency`. A request over the limit waits up to `grade_queue_timeout_s`, then gets 503 + `Retry-After`. Health stays responsive. Tests cover the wait, the timeout, release on every error path, and health under load.
- [x] The load measurement before and after is recorded here; peak RSS under 16 concurrent requests is bounded after the change.
- [x] `npm test` fails on zero tests and passes today; CI unchanged otherwise.
- [ ] Raw-card negative tests exist. Either they pass, or the detector fix keeps all 5,260 PSA quads identical. **Not met — deferred:** shipped as xfail (5 of 6 cases); the owner chose to wait for the phone-photo set before any detector work. Follow-up under item 8 of `intent/cv-ml-improvement-plan.md`; see Implementation, section 3.
- [x] The plan table is current, `dataset.precrop_coverage` is gone, `_git_commit` logs a warning on failure.
- [x] Served responses are identical on the 20-pair set from [[retire-legacy-model]] (`test-artifacts/retire_legacy/pairs.json`) when unloaded. The fast suite, `ruff`, frontend test and build all pass.

## Owner questions

1. **How many grades at once (`grade_concurrency`)?** *Recommendation:* 1 by default. ONNX already uses every core, so 2 mostly adds memory. It's a setting, so a bigger host can raise it.
2. **How long should a waiting request queue before it gets 503 (`grade_queue_timeout_s`)?** One grade takes ~2.5 s on CPU. *Recommendation:* 30 s, about 10 queued grades. The frontend shows the 503's message; the frontend isn't changed in this PR.
3. **Load test:** OK to run up to 16 concurrent requests against a local uvicorn on the real model (CPU, read-only, stopping early if memory passes ~10 GB)? *Recommendation:* yes.
4. **If the raw-card test fails**, fix the detector in this PR, provided all 5,260 PSA quads stay identical, or split it into its own PR? *Recommendation:* same PR under that condition; otherwise stop and ask.

## Owner decisions (2026-11-02)

1. `grade_concurrency` defaults to **1**.
2. `grade_queue_timeout_s` defaults to **30**.
3. **Load test approved:** up to 16 concurrent requests on a local uvicorn with the real model. It runs on CPU, reads data only, and stops early if RSS passes ~10 GB.
4. **Detector fix in this PR** only if all 5,260 PSA quads stay identical; otherwise stop and ask.

## Owner decisions (2026-11-02, after measurement)

Supersedes decision 1.

1. `grade_concurrency` defaults to **4**. Keep the limiter. The load test showed each extra concurrent grade adds only ~20–25 MB: peak RSS at 16 concurrent grades was 1,674 MB unbounded vs 1,323 MB for one. Running one at a time cost ~15% throughput (8 grades: 25.1 s vs 21.0 s). With 4, peak RSS at N = 16 is 1,402 MB, and latency and throughput are back to the unbounded level (table in Implementation).
2. Raw-card tests ship as non-strict xfail. No detector work until the phone-photo set exists; the follow-up is under item 8 of the plan.

## Facts vs assumptions

- **Fact:** `node --test` with an empty glob exits 0 (Node 22.14, checked 2026-11-02).
- **Fact:** grading runs in anyio's shared thread pool with no limit of its own (`grade.py:99,125`).
- **Fact:** `grade_head.json` has `git_commit: null` and `face_guard.json` has `50ecf08…`; both came from the same `_git_commit` helper.
- **Assumption:** memory per concurrent predict is large enough to matter. The measurement in 1 checks this before the limit is tuned.

## Implementation (2026-11-02)

Branch `CadeJordan/ServingHardening` (from main `26b0c59`), uncommitted. Scripts, logs and outputs: `test-artifacts/serving_hardening/`. Every heavy job ran alone, after checking that no other Python process was running.

### 1. Bounded grading

- `Settings.grade_concurrency` (default 4 since the post-measurement decision; first shipped as 1; ≥ 1) and `grade_queue_timeout_s` (default 30, > 0); env `GRADE_CONCURRENCY` / `GRADE_QUEUE_TIMEOUT_S`.
- `api/grade.py`: a module-level `anyio.CapacityLimiter`, created on first use from the settings (`grade_limiter()`; `reset_grade_limiter()` for tests). It keeps no event-loop state between requests. `_grading_slot()` acquires a token under `anyio.fail_after(grade_queue_timeout_s)`. On timeout it raises 503 `"server busy, try again"` with `Retry-After: 5` (`BUSY_RETRY_AFTER_S`, about two grades). Predict and centering (`_grade`, the old body moved unchanged) run under that one token, which is released in a `finally`. `run_in_threadpool` isn't abandoned on cancel, so a request cancelled mid-predict keeps its token until its thread is done. The ready check and `_read_image` (type, empty, size) run before the wait.
- `/api/health` takes no token. The optional `grading_in_flight` / `grading_waiting` keys were **not** added. They would change the health response, which the 20-pair check compares byte for byte, and nothing reads them yet.
- `main.py`: CORS `expose_headers=["Retry-After"]`, so a browser client can read the header. This is additive; the frontend is unchanged.
- README ("Start the API" and the API status table) and CLAUDE.md describe the limit and say that `--workers N` multiplies it.
- Tests: `backend/tests/test_grade_limit.py` has 15 fast tests. A fake gated service runs through httpx's ASGI transport. They check that:
  - a second request waits (`tasks_waiting == 1`, no model call) and gets the token when the first finishes;
  - a third times out to 503 + `Retry-After` after at least the queue timeout, with no model work;
  - `grade_concurrency=2` runs two grades at once;
  - health and a 413 answer within 1 s while a grade holds the token;
  - the token is released after a 400, a 422, a 500 (`RuntimeError`), an unhandled-exception 500 and a centering failure, and the next request gets it at once;
  - a request cancelled mid-predict keeps the token until predict ends, then releases it;
  - a request cancelled while queued leaves no waiter;
  - default settings are 4 / 30, and non-positive values are rejected;
  - the limiter takes its size from the settings;
  - `Retry-After` is exposed through CORS.

  The conftest `client` fixture resets the limiter for each test.

### Load measurement (real uvicorn, served model, CPU)

`loadtest.py` starts `uvicorn backend.app.main:app` on a free 127.0.0.1 port and sends one warm-up grade. Then it sends N concurrent `/api/grade` requests, for N = 1, 4, 8 and 16. The requests cycle through the 15 test-split pairs from `pairs.json` that grade 200, read only. It samples peak RSS of the server process tree every 20 ms and polls `/api/health` every 250 ms during each level.

The machine has 16 cores and 15.7 GB. RSS never came near the 10 GB stop. The server was terminated after each run, and its port was free afterwards (`port_free_after: true`). Output: `load_before.json` (unmodified tree) and `load_after.json`.

Idle RSS after the warm-up grade was 918 MB in both runs. One predict on CPU peaks at ~1,320 MB, i.e. **~400 MB over idle**.

| N | peak RSS before → after (MB) | p50 / p95 latency before (s) | p50 / p95 latency after (s) | status before | status after | health max before / after (s) |
|---|---|---|---|---|---|---|
| 1 | 1,323 → 1,324 | 3.27 / 3.27 | 3.43 / 3.43 | 1×200 | 1×200 | 0.45 / 0.45 |
| 4 | 1,395 → 1,333 | 8.53 / 10.50 | 9.35 / 12.46 | 4×200 | 4×200 | 0.45 / 0.60 |
| 8 | 1,485 → 1,342 | 14.23 / 21.02 | 16.00 / 25.12 | 8×200 | 8×200 | 0.62 / 0.60 |
| 16 | 1,674 → 1,348 | 26.27 / 41.48 | 28.04 / 30.95 (200s only: 16.13 / 30.94) | 16×200 | 10×200, 6×503 | 0.93 / 0.96 |

**After, with the default raised to `grade_concurrency` 4** (`load_after_c4.json`, same script and constraints; no other Python job running; port free afterwards):

| N | peak RSS (MB) | p50 / p95 latency (s) | status | health max (s) |
|---|---|---|---|---|
| 1 | 1,322 | 3.39 / 3.39 | 1×200 | 0.57 |
| 4 | 1,418 | 9.06 / 11.15 | 4×200 | 0.45 |
| 8 | 1,394 | 14.00 / 20.93 | 8×200 | 0.57 |
| 16 | 1,402 | 24.52 / 37.18 (200s only: 22.17 / 37.18) | 15×200, 1×503 | 0.89 |

With 4, wall time for 8 grades is 20.9 s, the same as unbounded (21.0 s), and peak RSS at 16 stays at 1,402 MB (unbounded: 1,674 MB). At N = 16, one request waited past 30 s and got a 503.

- **The memory assumption was mostly wrong.** On these ~1024-px PSA photos, concurrent predicts share ONNX Runtime's memory, so each extra grade in flight added only ~20–25 MB. Sixteen at once used +350 MB over one. Memory never came near the risk this spec feared, at least up to N = 16. After the change, peak RSS stays at one grade's ~1.35 GB whatever N is.
- **Throughput cost.** Running grades one at a time is ~15% slower than letting ONNX overlap them. Eight grades took 25.1 s wall after vs 21.0 s before; at N = 16 a grade took ~3.1 s vs ~2.7 s. At N = 16, the six requests that would have waited past 30 s now get a fast 503 instead of a 30–41 s wait. On this host, `grade_concurrency` 2 would recover most of the throughput for little memory. After this measurement the owner raised the default to 4; see the c = 4 table below.
- Health stayed under 1 s throughout, before and after. What slows it is the GIL during preprocessing, not the limiter.

### 20-pair check

`capture.py` posts `pairs.json` through FastAPI `TestClient` on the served model. `before/` was captured on the unmodified tree and `after/` with the change. **All 22 files are byte-identical** (`cmp`): the 20 responses (5 × 422 card not found, 15 × 200), `health.json` and `summary.json`. The response has no timing fields. Re-run with the default of 4 (`after_c4/`): again 22 / 22 byte-identical.

### 2. CI fails on zero tests

- `frontend/scripts/test.mjs` adds no dependency. It runs `node --test --experimental-strip-types` on `src/**/*.test.ts`, or on globs given as arguments. It keeps node's usual console reporter and also writes TAP to a temp file. It passes on a non-zero exit, and exits 1 when the TAP summary says `# tests 0`. `package.json` now has `"test": "node scripts/test.mjs"`, so CI still runs `npm test`.
- Checked:
  - `npm test` passes (10 tests, exit 0);
  - `npm test -- "nomatch/**/*.test.ts"` exits 1 ("no tests ran"), where plain `node --test` on that glob exits 0;
  - a failing test file exits 1.
- `backend/tests/test_ci_guards.py` pins that pytest exits 5 when it collects zero tests. It runs pytest in a subprocess on an empty `tmp_path` project (~2 s).

### 3. Raw-card crop check: tests fail, detector unchanged

- `test_raw_card_with_flat_border_keeps_its_outline` has 6 cases, next to `test_slab_photo_finds_the_card`. Each is a raw card on a plain light background with a flat dark border (luma 40–120) and a textured art window of similar brightness; it needs IoU ≥ 0.9 with the card.
  - With no frame line around the window (`b70_w70_no_frame`), it **passes**: the window has no strong outline.
  - With a thin frame line, light or dark (printed art frames have one), all 5 cases **fail**. IoU is ~0.81: the crop is the art window.
  - Traced: the last-resort flat-ring step does it, at the **first** step, with `best` = the card.
- **No constraint fixes it.** The raw scene and `silver_full_art_in_grey_well` look the same to the detector: a strong card-shaped outline, a flat dark ring, and a strong textured nested outline at ~0.81 of the area, with the same aspect. Tried in memory (`variants.py`), as slab scenes passing / raw cases passing:

  | last-resort variant | slab scenes | raw cases |
  |---|---|---|
  | current | 8/8 | 1/6 |
  | only at the first step | 8/8 | 1/6 (it already fires there) |
  | only past the first step | 4/8 (breaks the silver slab scene) | 4/6 |
  | only when `best` isn't card-height (0.62–0.68 of the photo) | 8/8 | 1/6 |

- On real PSA photos (`flat_diag.py`, read only, all 5,260 faces), the last-resort step decides **84 faces, 65 of them at the first step**, with `best` at 0.649–0.768 of the photo height. So "past the first step" or a height rule would change PSA quads too. No candidate reached the PSA gate, because none keeps the slab scenes.
- Per the stop rule:
  - the detector is unchanged (`preprocess.py` untouched);
  - the 5 failing cases are non-strict `xfail`, with the reason;
  - this goes back to the owner.

  Nothing under `data/` was written. The PSA quad comparison wasn't needed, because no detector change was kept.

### 4. Tidy

- **Plan table.** In `intent/cv-ml-improvement-plan.md`, "How the system performs now" is now a short current table:
  - head 2026-10-31, `ten_cutoff` 0.56;
  - CV seed 0: MAE 0.4645, exact 63.4% (from `grade_head.json`);
  - test once: MAE 0.443, exact 65.3% (from [[tighter-slab-crops]] Phase 2).

  It points to ADRs 0002/0003, [[tighter-slab-crops]] and `git show 26b0c59:intent/cv-ml-improvement-plan.md` for the old tables. The "381 passed" line is gone.
- **`precrop_coverage`.** `ml.data.dataset.precrop_coverage` is removed, and so is its assertion in the test. The test's DB-level assertion stays, renamed `test_db_precrop_coverage_by_split`: it's the only test of `db.precrop_coverage_by_split`, which `precrop` uses.
- **Why `_git_commit` was null: not reproducible.**
  - Now `_git_commit()` returns `26b0c59…` from both Bash (`mingw64\bin\git.exe`) and PowerShell (`Git\cmd\git.exe`).
  - Its `cwd` is `PROJECT_ROOT`, an absolute path from `config.py`, so the caller's cwd doesn't matter.
  - The reflog shows HEAD at `50ecf08` on `CadeJordan/TighterSlabCrops` from 20:53 to 23:29 local on 2026-10-31. So the repo was the same for `train_probe` (finished 22:09) and for `train_face_guard` (22:38), which got `50ecf08`.
  - `.git` is owned by `BUILTIN\Administrators`, the user is `user-pc\user`, and no `safe.directory` is set. Git accepts that today, but a process started with a different token could hit "dubious ownership". The other possibility is a launching shell without git on its PATH.
  - The old helper swallowed `OSError` / `CalledProcessError` and dropped stderr, so that run left no trace of which.

  Root cause: the environment of the process that launched that `train_probe` run, most likely its PATH or git's ownership check. The code's defect was that the failure was silent.
- **Fix.** `_git_commit` now logs a WARNING with the reason (the exception, or the exit code + git's stderr) and still returns None. It also catches a 30 s timeout. `train_face_guard` imports the same function, so it warns too. `ml/tests/test_git_commit.py` has 7 tests:
  - HEAD is returned with no warning;
  - git missing, a timeout, a refusal ("dubious ownership") and empty output each warn once, with the reason;
  - `train_face_guard` uses the same helper;
  - the real checkout answers.

  `data/models/grade_head.json` is untouched; its null stays until the next retrain.

### Checks

- Fast suite (`python -m pytest -m "not slow and not visual and not model"`): **1584 passed, 1 skipped, 6 xfailed** (was 1560 / 1 / 1). The +24 passing tests are 15 grade-limit, 1 CI guard, 7 `_git_commit` and 1 raw-card case. The +5 xfails are the framed raw-card cases. One existing PIL warning. The baseline in CLAUDE.md is updated.
- `ruff check .`: all checks passed.
- `python -m pytest backend/tests/test_preprocess.py -m "slow or not slow"`: 103 passed, 6 xfailed (was 102 / 1: +1 raw-card pass, +5 raw-card xfails). That includes `test_stored_precrops_are_reproduced_byte_identical`.
- Frontend: `npm test` 10 passed (exit 0), `npm run lint` clean, `npx tsc --noEmit` clean, `npm run build` succeeded.
- Nothing under `data/` was written. No package was installed, and no test-split evaluation was run.
