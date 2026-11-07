# Rate limits, phone photo picker, tab title

## Intent

Three small changes to the live site (Cloud Run API + Vercel frontend, [[cloud-run-deploy]]):

1. **Photo picker on phones.** Tapping an upload box on an iPhone opens the camera directly, so you can't choose a photo you already took. Drop `capture` so iOS (and Android) offer the camera *or* the photo library.
2. **Per-IP rate limits in the API:**
   - at most **5 wrong access codes per minute per IP**;
   - at most **30 grades per hour per IP**.

   Guessing the code becomes impractical, and no one visitor can tie up the single instance or run up the bill.
3. **Tab title.** The browser tab reads just **PokeScan**.

## Facts (2026-11-07)

- **The picker.** `frontend/src/components/ImageDropzone.tsx:128` sets `capture="environment"` on the hidden file input (`accept="image/*"`).
- **iPhone photos already work.** The owner took photos with an iPhone camera on the live site and they graded, so Safari hands the API a format it accepts. Library picks go through the same `accept="image/*"` path. The plan is to confirm that on the device after this ships, not to build a HEIC conversion.
- **Access code.**
  - `backend/app/core/access.py` (`AccessCodeMiddleware`, pure ASGI, outside CORS) answers 401 for a missing or wrong `X-Access-Code` on `/api/grade`, before the body is read.
  - There's no attempt limit.
- **Grading.**
  - The only limit is the concurrency limiter (`grade_concurrency`; 2 on Cloud Run), with a 30 s queue and then 503 + `Retry-After`. That protects memory, not abuse.
  - Cloud Run runs at most **1 instance** with **1 uvicorn worker**, so a per-process, in-memory counter sees every request.
- **Client IP.** Behind Cloud Run, the TCP peer (`scope["client"]`) is Google's front end, not the visitor. The visitor's IP is in `X-Forwarded-For`.
- **Frontend error handling.**
  - `api-client.ts` turns a 401 into `AccessCodeError`, and any other non-OK response into an `Error` carrying the API's `detail`. So a 429 with a clear `detail` already shows sensibly.
  - CORS exposes `Retry-After`.
- **Tab title.** `frontend/src/app/layout.tsx:17` has `title: "PokeScan — predict your PSA grade"`.

## Assumptions (to verify)

- **`X-Forwarded-For` on Cloud Run:** Google's front end *appends* the real client IP. So with a spoofed header from the client, the value is `<spoofed>, <real>`, and the **right-most** entry is the one to trust. Not verifiable locally. The deploy guide gets a post-deploy check (below). If the check fails, stop and fix before relying on the limit.
- **Shared IPs:** friends on the same Wi-Fi (or a carrier NAT) share an IP, and so share the limits. That's acceptable at 30 grades per hour; note it in the docs.

## Scope

### 1. Photo picker (`ImageDropzone.tsx`)
- Remove `capture="environment"`; keep `accept="image/*"`.
- Drag-and-drop and click-to-choose on desktop are unchanged.

### 2. Rate limits (API)
- **A small limiter module,** e.g. `backend/app/core/rate_limit.py`:
  - a thread-safe sliding-window counter per key (`(bucket, ip)`), in memory;
  - prunes expired entries as it goes, so memory stays bounded;
  - an injectable clock for tests.
- **Client IP:**
  - With `settings.trust_forwarded_for` true, use the right-most `X-Forwarded-For` entry. Otherwise use `scope["client"]` (local dev, tests).
  - The setting defaults to false; the Cloud Run deploy sets `TRUST_FORWARDED_FOR=true`.
  - Never trust the header when the setting is off. If there's no IP at all, use a single shared `"unknown"` key.
- **Wrong codes:** `settings.code_attempts_per_minute`, default 5. Lives in `AccessCodeMiddleware`.
  - Each missing or wrong code on `/api/grade` counts as one failure for that IP.
  - Once an IP has 5 failures inside the last 60 s, **every** `/api/grade` request from it gets **429** `"too many attempts, wait a minute"` with `Retry-After` (seconds until the oldest failure expires), even with the right code, until the window clears. That blocks guessing past the limit.
  - It's checked before the body is read, like the 401.
  - A missing header counts too. A page that hasn't asked for the code yet sends one request without it, so a normal visitor uses 1 of the 5.
  - Off when `ACCESS_CODE` is unset.
- **Grades:** `settings.grades_per_hour`, default 30; 0 means off, so local dev and the existing tests are unaffected unless they set it.
  - It counts each request that gets past the access check and is about to take a grading slot. Requests refused for size or format, or with a 401 or 429, don't count.
  - Over the limit: **429** `"grade limit reached (30 per hour), try again later"` with `Retry-After`.
  - It's checked **before** the upload is read and before waiting for a slot.
  - Lives in the same middleware layer or in `grade.py` ahead of `_grading_slot()`, whichever gives "before the body is read". Pure ASGI is preferred, for the reason in `access.py`'s docstring.
- **Response details:**
  - CORS stays outermost, so every 429 carries `Access-Control-Allow-Origin`, and `Retry-After` is readable.
  - `/api/health` is never limited.
- **Tests (fast, no real model):**
  - wrong codes: the 6th within a minute is 429 even with the right code; it clears after the window (fake clock);
  - IPs are counted separately; with the setting on, a spoofed left-most `X-Forwarded-For` entry doesn't escape the limit; with it off, the header is ignored;
  - grades: the 31st within an hour is 429 with `Retry-After`, and the 1st after the window is 200;
  - 413, 400, 401 and 429 responses don't use up grade allowance;
  - a 429 doesn't read the body or take a slot (as in the existing 401 test);
  - health is unaffected; defaults are off, so behaviour is unchanged;
  - memory stays bounded (entries pruned).

### 3. Frontend
- `api-client.ts`: a 429 becomes an `Error` with the API's `detail`, as today. Make sure the access-code flow doesn't treat a 429 as "wrong code" or clear the stored code.
- The message shown is the `detail` text, and should read naturally. No new UI beyond that.
- Tests in `src/lib/accessCode.test.ts` (or a sibling) for "429 doesn't clear the stored code or re-prompt".

### 4. Tab title
- `layout.tsx`: `title: "PokeScan"`. Keep the meta `description` as is; it's for search results and link previews, not the tab.

### 5. Docs and deploy
- **`docs/deploy.md`:**
  - add `TRUST_FORWARDED_FOR=true` to the step 2.2 deploy command, and an `--update-env-vars` one-liner for the running service;
  - a table row for each of the three new settings;
  - the shared-IP caveat;
  - a **post-deploy check**: from PowerShell, send 6 requests with a wrong code, each with a different fake `X-Forwarded-For`. The 6th must be 429. If it isn't, the trusted entry is wrong.
- **`owner-steps.md`:** a short "after this PR merges" section. It needs:
  - `git pull`;
  - re-running the deploy command (the API code changed);
  - adding `TRUST_FORWARDED_FOR=true`;
  - the check above.

  Vercel redeploys the frontend on push to `main` by itself.
- **README / CLAUDE.md:** the new settings, in the API paragraph.

## Out of scope

- HEIC conversion or browser-side resizing (iPhone photos already work), and separate camera/library buttons.
- Cloud Armor, API Gateway, Redis or any shared store (one instance is enough), and per-user accounts.
- Changing grading, models or responses.

## Acceptance criteria

- [x] `capture` is removed. Lint, typecheck, build and frontend tests pass.
- [x] The wrong-code limit: 5 per minute per IP, then 429 with `Retry-After` even for the right code, cleared after the window. Tested.
- [x] The grade limit: 30 per hour per IP, then 429 with `Retry-After`. Only requests that would grade count. Tested.
- [x] The client IP is the right-most `X-Forwarded-For` entry only when `TRUST_FORWARDED_FOR` is on; spoofing the left-most entry doesn't help. Tested.
- [x] A 429 reads no body and takes no grading slot; health is never limited. Defaults (`ACCESS_CODE` unset, `GRADES_PER_HOUR` 0, `TRUST_FORWARDED_FOR` false) leave behaviour unchanged.
- [x] The frontend shows the 429 detail and keeps the stored code. Tested.
- [x] The tab title is "PokeScan".
- [x] The fast suite and ruff pass; the CLAUDE.md baseline is updated.
- [x] `docs/deploy.md` and `owner-steps.md` updated, including the post-deploy spoof check.
- [x] Results written below.

## Owner decisions (2026-11-07)

1. **Photo picker:** drop `capture` (one button; iOS offers camera or library). iPhone camera photos already grade on the live site.
2. **Wrong codes:** 5 per minute per IP.
3. **Grades:** 30 per hour per IP.
4. **Tab title:** "PokeScan".

## Defaults chosen in this spec (the owner can override)

- A locked-out IP is refused even with the right code until its minute clears.
- A missing code counts as a failed attempt (a normal first visit uses 1 of the 5).
- `GRADES_PER_HOUR` defaults to 0 (off) in code and is set to 30 in the deploy command, so local dev and tests stay unlimited. The same goes for `TRUST_FORWARDED_FOR`.
- Counts are in memory and reset when the instance restarts (e.g. after scaling to zero). That's acceptable here: a restart takes a cold start, which slows a guesser further.

## Results

Done 2026-11-07, branch `CadeJordan/RateLimitsPhotoPicker` (uncommitted).

- **Photo picker / title:** `capture="environment"` removed from `ImageDropzone.tsx` (`accept="image/*"` kept); `layout.tsx` title is `"PokeScan"`, description unchanged. To confirm on the iPhone after the deploy (owner-steps step 5).
- **Limiter** (`backend/app/core/rate_limit.py`): `SlidingWindowLimiter` (lock-guarded deque of timestamps per key, one instance per bucket so keys are IPs; expired stamps dropped on every touch plus a sweep of all keys once per window; a key never holds more than `limit` stamps; injectable clock, `create_app(clock=...)`). `client_ip()`: right-most non-empty `X-Forwarded-For` entry (header lines joined in order) only with `TRUST_FORWARDED_FOR`, else `scope["client"]`, else `"unknown"`. Settings `code_attempts_per_minute` (5; 0 = off), `grades_per_hour` (0 = off), `trust_forwarded_for` (false).
- **Wrong codes:** in `AccessCodeMiddleware` (only when `ACCESS_CODE` is set). A locked IP gets 429 `"too many attempts, wait a minute"` before the code is compared; otherwise a missing/wrong code records one failure and gets the 401. 429s don't record, so a lockout doesn't extend itself. `Retry-After` = whole seconds (≥ 1) until the oldest counted failure leaves the window.
- **Grades — design: reserve, then keep or release.** `GradeRateLimitMiddleware` (pure ASGI, inside the access middleware, so 401/429-for-code never reach it) atomically reserves one grade for the IP before calling the app, or answers 429 at once (`"grade limit reached (30 per hour), try again later"`, `Retry-After`) without reading the body. It puts a `GradeTicket` in the ASGI scope; `grade.py` calls `ticket.count()` once the request **holds a grading slot**. When the request ends, a reservation whose ticket isn't counted is released. So 413/415/empty-400, FastAPI's 422 for a missing field, a 503 (model not loaded, or busy after waiting 30 s), 401 and 429 cost nothing; a request that held a slot counts whatever its outcome (200, the unreadable-image 400, the 422 rejections, 500). The unreadable-image 400 counts (changed after testing): it holds a slot while the other face is decoded, so a free one could keep slots busy at no cost. Reserving rather than "check now, count later" keeps the limit exact under concurrent requests (tested). Deviation from the letter of the spec: a 503 "server busy" (no slot within the queue timeout) doesn't count either, since it did no work.
- **Order:** CORS → access code (+ wrong-code limit) → grade limit → app, so every 429 carries `Access-Control-Allow-Origin` and the exposed `Retry-After`. `/api/health` and preflights are never limited.
- **Frontend:** `gradeError(status, detail)` in `src/lib/accessCode.ts` (used by `api-client.ts`): 401 → `AccessCodeError`, anything else (incl. 429) → `Error(detail)`. `withAccessCode` already rethrows non-401 errors without clearing the code; tests pin that for 429.
- **Tests:** `backend/tests/test_rate_limit.py` (48 tests; centering stubbed so it runs in seconds): wrong codes (`test_sixth_wrong_code_attempt_is_429_even_with_the_right_code`, `test_wrong_code_lockout_clears_after_the_window`, `test_lockout_does_not_extend_itself`, `test_right_codes_do_not_count`, `test_wrong_code_limit_is_per_ip`), IPs (`test_client_ip[...]`, `test_spoofed_left_most_entry_does_not_escape_the_limit`, `test_forwarded_for_ignored_when_not_trusted`, `test_trusted_header_separates_ips`, `test_client_ip_ipv6_peer_is_keyed_by_its_64`, `test_ipv6_addresses_in_one_64_share_the_limits`), grades (`test_31st_grade_in_an_hour_is_429_and_clears_after_the_window`, `test_refused_requests_do_not_use_grade_allowance`, `test_model_not_loaded_503_does_not_use_grade_allowance`, `test_server_busy_503_does_not_use_grade_allowance`, `test_graded_failures_use_grade_allowance`, `test_concurrent_requests_cannot_exceed_the_grade_limit`), no body / no slot (`test_wrong_code_429_reads_no_body`, `test_grade_429_reads_no_body_and_takes_no_slot`: the only slot is held elsewhere and the over-limit request still gets an immediate 429 with zero `receive` calls), CORS (`test_wrong_code_429_carries_cors_headers`, `test_grade_429_carries_cors_headers`), health (`test_health_is_never_limited`, `test_health_stays_open_when_the_grade_limit_is_used_up`), defaults (`test_settings_defaults`, `test_defaults_add_no_limit`), memory (`test_limiter_memory_is_bounded`, `test_limiter_is_thread_safe`). Frontend: 4 new tests in `accessCode.test.ts` (`gradeError: ...`, `a 429 (...) keeps the stored code and doesn't ask for one` ×2, `a 429 on a just-entered code keeps that code`).
- **Checks:** `ruff check .` clean. Fast suite **1691 passed, 1 skipped, 6 xfailed** (was 1643; +48, including 8 added after the test pass: IPv6 /64 keying, the unreadable-image 400 counting, busy-503 and health tests), data guard no change; CLAUDE.md baseline updated. Frontend `npm run lint`, `npx tsc --noEmit`, `npm run build` pass; `npm test` 28/28.
- **Container** (root `Dockerfile`, `ACCESS_CODE=test TRUST_FORWARDED_FOR=true GRADES_PER_HOUR=30`, port 18080): 6 wrong codes with different forged left-most entries and one right-most IP → `401 ×5, 429`; the right code from that IP → 429 `retry-after: 60`; another right-most IP → 422 (code accepted); health 200. With `GRADES_PER_HOUR=2` (no code): two grades of a synthetic pair (422 same-side rejections, which ran the model) then 429; another IP still graded; the 429 carried `retry-after: 3581` and `access-control-allow-origin`. Container and image removed.
- **Docs:** `docs/deploy.md` (two env vars in the step 2.2 command and YAML, an `--update-env-vars` redeploy for an existing service, table rows for the three settings, the shared-IP caveat, step 3.1 spoof check in PowerShell 5.1 with `curl.exe`, plus a phone-on-mobile-data check that the right-most entry isn't a shared Google IP; step 6 no longer suggests re-running `--set-env-vars`), `owner-steps.md` ("After the rate-limit PR merges", with `pkmn-scan`), README (settings table, 429 row) and CLAUDE.md.
- **Open:** the `X-Forwarded-For` assumption is verified only by the owner's post-deploy check (step 3.1). Counts reset on a cold start (accepted default).
