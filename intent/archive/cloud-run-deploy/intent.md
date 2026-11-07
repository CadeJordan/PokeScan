# Deploy: API on Cloud Run, frontend on Vercel

## Intent

Put PokeScan online for the owner and a few friends, at about $0 a month:
- **The API** (FastAPI + the probe model) runs as a container on Google Cloud Run. It scales to zero, with at most one instance.
- **The frontend** (Next.js) runs on Vercel's free Hobby plan and calls the API directly from the browser.
- **A shared access code** keeps strangers who find the URL from using it.

The repo gets the container, the config and a step-by-step deploy guide. The owner runs the account steps (Google Cloud project, `gcloud`, Vercel import), because they need the owner's accounts and billing.

## Facts (2026-11-07)

- **The model.**
  - Serving needs these files in `data/models/`: `card_features.onnx` (346 MB), `card_features.json`, `grade_head.npz` + `.json`, and `face_guard.npz` + `.json` (the face guard fails open without them).
  - All of them are gitignored.
  - The model loads once at startup (`main._lifespan` → `get_inference_service()`).
- **No torch.** `backend/app` never imports `ml.*`. `features.py` imports torch/timm only inside the export helpers. The serving imports are fastapi, pydantic(-settings), python-dotenv, anyio, numpy, opencv, pillow and onnxruntime.
- **Known-good versions** (global env, the suite passes): fastapi 0.136.1, starlette 1.0.0, uvicorn 0.46.0, pydantic 2.13.3, pydantic-settings 2.14.0, python-multipart 0.0.26, python-dotenv 1.2.1, anyio 4.13.0, onnxruntime 1.25.0, opencv-python 4.11.0.86 (use `-headless` in the container), numpy 1.26.4, pillow 12.1.1.
- **Memory.** In the [[serving-hardening]] load test, the whole process peaked at 1,402 MB with 16 concurrent requests at concurrency 4. A grade takes about 3 s on this machine's CPU.
- **Uploads and errors.** Uploads are capped at 12 MB per image (`api/grade.py` `MAX_BYTES`). A busy server returns a 503 with `Retry-After`, and CORS already exposes `Retry-After`.
- **CORS** is `CORS_ORIGINS`, an exact comma-separated list. There's no pattern support, but Vercel preview deployments get a new URL each time.
- **Frontend.**
  - `src/app/api-client.ts` calls `${NEXT_PUBLIC_API_BASE}/health` and `/grade`, and `NEXT_PUBLIC_API_BASE` includes `/api`.
  - The frontend is set up for Cloudflare Workers via OpenNext: `wrangler.jsonc`, `open-next.config.ts`, `cloudflare-env.d.ts`, the `deploy`/`preview`/`upload`/`cf-typegen` scripts, the `initOpenNextCloudflareForDev()` call in `next.config.ts`, the `@opennextjs/cloudflare` dependency, and an OpenNext starter `README.md`.
- **Tools.** Docker Desktop is installed on this machine; `gcloud` and the Vercel CLI are not.

## Assumptions (unverified; to check during development or deployment)

- **Cloud Run's free tier** is about 180k vCPU-seconds, 360k GiB-seconds and 2M requests a month, in Tier 1 regions such as `us-central1`. That covers thousands of grades. The deploy guide links the pricing page instead of promising numbers.
- **Cloud Run's request body limit** (32 MiB on HTTP/1) is above the largest legal upload: 2 × 12 MB plus multipart overhead.
- **Cold start** (pulling the ~1 GB image and loading a 346 MB ONNX model) takes about 10–20 s with startup CPU boost. To be measured.
- **Cloud Run CPU is slower per core** than this desktop, so a grade may take 4–6 s.

## Scope

### 1. Container
- **`deploy/requirements-serve.txt`:** the serving packages above, pinned with `==`. Use `opencv-python-headless` and `uvicorn[standard]`. Don't install `pokescan` itself or torch.
- **`deploy/Dockerfile`:**
  - `python:3.11-slim` base, plus the system libraries that headless OpenCV needs.
  - Copies `backend/` (without `backend/tests`) and the model files into `/app/data/models/`.
  - Runs as a non-root user.
  - Starts with `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT --workers 1`.
  - No `.env` and no PSA/eBay secrets in the image.
- **Where the Dockerfile goes.** `gcloud run deploy --source .` (Q4) builds from a `Dockerfile` at the root of the uploaded source. So the Dockerfile lives at the repo root (or `deploy/` plus a `cloudbuild.yaml`; the dev picks one and the guide matches it).
- **`.gcloudignore`.** Without one, `gcloud` falls back to `.gitignore`, which would leave the gitignored model files out of the upload. It must mirror `.dockerignore`'s allow-list.
- **`.dockerignore`:** an allow-list. The build context holds only `backend/` and the model files; never `data/images`, crops, the DB, backups, `frontend/` or `.git`.
- **Model files come in by Q2.** Either they're copied into the build context from `data/models/` (read only), or they're downloaded in the build from a private bucket.

### 2. API changes (small; defaults keep today's behaviour)
- **`CORS_ORIGIN_REGEX`:** an optional setting passed to `CORSMiddleware(allow_origin_regex=...)`, e.g. `https://pokescan[-a-z0-9]*\.vercel\.app`. Unset means exact `CORS_ORIGINS` only, as today.
- **`ACCESS_CODE`** (Q3), optional:
  - When it's set, `/api/grade` requires the header `X-Access-Code`, compared with `hmac.compare_digest`.
  - A missing or wrong code is a **401** `"access code required"`, returned before the upload is read or a grading slot is taken.
  - `/api/health` stays open, so the hosting platform's health checks and the frontend's model check work, and it doesn't reveal the code.
  - When it isn't set, nothing changes, which covers local dev and tests.
- **Tests:**
  - the regex allows a preview origin and rejects others;
  - with a code set: no code → 401, wrong → 401, right → 200; health stays 200; the 401 happens before any model work (the fake service isn't called);
  - with no code set, behaviour is unchanged.

### 3. Frontend
- **Access code:**
  - If the API returns 401, the page asks for the access code (a small inline field, not `window.prompt`).
  - The code is stored in `localStorage`, wrapped in try/catch, and sent as `X-Access-Code` on `/grade`. Then the grade is retried once.
  - A wrong code shows "That code didn't work" and clears the stored one.
  - `api-client.ts` still mirrors the schema; the 401 is the only new case.
- **Replace Cloudflare with Vercel** per Q1: remove the OpenNext/wrangler files, scripts and dependency and the `next.config.ts` hook, and replace the starter README with a short frontend README.
- **`.env.example`:** documents `NEXT_PUBLIC_API_BASE` (with `/api`).
- **Gates:** `npm run lint`, `npx tsc --noEmit`, `npm run build` and `npm test` pass. CI's frontend job must not depend on anything removed.

### 4. Deploy guide: `docs/deploy.md`
Copy-paste steps for Windows PowerShell, with a note on what each one does:
1. **Google Cloud:**
   - Create a project and link billing.
   - Set a **budget alert** (e.g. $5).
   - Install the `gcloud` CLI, run `gcloud auth login`, and enable the Run, Artifact Registry and Cloud Build APIs.
2. **Build and deploy:** `gcloud run deploy pokescan-api --source . ...` (Cloud Build from the Dockerfile) or a local `docker build` + push (Q4), with:
   - `--region us-central1`, `--cpu 2`, `--memory 2Gi` (or 4Gi if measured peak memory needs it);
   - `--min-instances 0`, `--max-instances 1`, `--cpu-boost`, `--timeout 60`, `--allow-unauthenticated` (the access code is the lock);
   - env vars `GRADE_CONCURRENCY=2`, `CORS_ORIGINS`, `CORS_ORIGIN_REGEX` and `ACCESS_CODE`. The code goes in through Secret Manager or `--set-env-vars`, per Q3.
3. **Check:** `curl <url>/api/health` shows `model_loaded: true`.
4. **Vercel:**
   - Import the GitHub repo with root directory `frontend`.
   - Set the env var `NEXT_PUBLIC_API_BASE=<cloud run url>/api` for Production and Preview.
   - Deploy.
5. **Smoke test:** open the site, enter the code, grade a real pair.
6. **Updating:** redeploy the API after retraining (the model is in the image); Vercel redeploys on push to `main`.
7. **Costs and limits:** what's free, what costs a few cents (image storage), how to see usage, and how to take it down.

### 5. Verification (by the dev and test agents, locally; no cloud account needed)
- `docker build` succeeds. Report the image size.
- `docker run` with `PORT=8080` on the 2 GiB / 2 CPU limits Cloud Run would use (`--memory 2g --cpus 2`):
  - `/api/health` reports the model loaded;
  - a real front/back pair from `data/images` (read only) grades;
  - the response is **identical** to the local uvicorn's for the same pair, on 5 pairs;
  - with `ACCESS_CODE` set, a request without it is 401.
- Measure startup time to healthy, peak container memory during 4 concurrent grades, and time per grade. Record them in Results, and choose `--memory` from them.
- Confirm the image contains no `.env`, DB, images or crops (`docker run ... ls`, and the `.dockerignore` test).

## Out of scope

- Creating the cloud accounts, running the actual deploy, or spending money. Those are the owner's steps; the guide covers them.
- User accounts or real auth, per-IP rate limiting, custom domains, CI/CD auto-deploy of the API, GPU, and model storage outside the image (unless Q2 picks it).
- Any change to grading, models or thresholds. Responses must be identical.

## Acceptance criteria

- [x] `Dockerfile` (repo root, see Results), `deploy/requirements-serve.txt` and `.dockerignore` exist. The image builds, runs as non-root, and holds only the backend code and the model files.
- [x] The containerised API's responses are identical to local uvicorn's on 5 real pairs.
- [x] Startup time, peak memory under 4 concurrent grades, and per-grade latency are measured in the container and recorded. `--memory` in the guide follows from them.
- [x] `CORS_ORIGIN_REGEX` and `ACCESS_CODE` work as specified, with tests. Both are off by default with no behaviour change, and the fast suite and ruff pass.
- [x] The frontend asks for the code on a 401, remembers it, and sends it. Lint, typecheck, build and tests pass. Cloudflare is handled per Q1.
- [x] `docs/deploy.md` takes a new reader from nothing to a working site, including a budget alert and a max-instances cap.
- [x] README and CLAUDE.md point to the guide and the two new settings.
- [x] Results written below.

## Questions for the owner

1. **Cloudflare setup:** remove the OpenNext/wrangler config, scripts and dependency, and make Vercel the only frontend target? *Recommendation:* yes. Keeping both means two build paths to maintain, and `initOpenNextCloudflareForDev()` in `next.config.ts` is Cloudflare-only.
2. **How the model gets into the image:** copy `data/models/` into the build context (simple; each deploy uploads ~350 MB to Cloud Build), or put the files in a private Cloud Storage bucket the build downloads from (smaller uploads, one more setup step)? *Recommendation:* copy them in. Deploys are rare, and it's one less thing to set up.
3. **Access code:** add it? If so, store it as an env var or in Secret Manager? *Recommendation:* yes, as a plain Cloud Run env var. It's a shared code for friends, not a real secret, and Secret Manager adds setup for no real gain here.
4. **Build route:** `gcloud run deploy --source .` (Cloud Build does the build; nothing to run locally) or a local `docker build` + push (needs Docker Desktop running and a registry login)? *Recommendation:* `--source .`. Cloud Build's free monthly minutes cover occasional deploys. The local Docker build is still used to test before deploying.
5. **Where results go:** *Recommendation:* in this spec, as before.

## Owner decisions (2026-11-07)

1. **Cloudflare:** remove the OpenNext/wrangler setup; Vercel is the only frontend target.
2. **Model files:** copied into the image from `data/models/` (read only).
3. **Access code:** yes, as a plain Cloud Run env var (`ACCESS_CODE`), not Secret Manager.
4. **Build route:** `gcloud run deploy --source .` (Cloud Build). The local Docker build is for testing.
5. **Results:** in this spec.

## Results

Dev, 2026-11-07, branch `CadeJordan/CloudRunDeploy` (uncommitted). Measured locally with Docker Desktop 29.8.2 (Linux VM: 2 CPUs, 7.7 GiB); no cloud resources were created.

**Container.**
- The `Dockerfile` is at the **repo root** (not `deploy/Dockerfile`): `gcloud run deploy --source .` builds a root `Dockerfile`, so no `cloudbuild.yaml` is needed. `deploy/requirements-serve.txt` pins the 12 serving packages plus their 16 transitive dependencies with `==` (as the image resolved them). Headless OpenCV and ONNX Runtime import and grade on `python:3.11-slim` with no extra system libraries.
- `.dockerignore` and `.gcloudignore` are matching allow-lists. The `.gcloudignore` was checked with git's own ignore matching on a mirror of the tree: it lets through exactly `Dockerfile`, `deploy/requirements-serve.txt`, `backend/` (no tests) and the six model files. `backend/tests/test_deploy_files.py` keeps the two lists, the Dockerfile's model list, the non-root `USER` and the pins in step.
- **Image size: 809 MB** (808,611,632 bytes; /app is 331 MB, of which the ONNX graph is 346 MB on disk ≈ 330 MiB). Runs as uid 10001 (`app`). `find` over the image: no `.env`, no `*.sqlite*`, no card images, no `crops`/`images_full`/backups, no `backend/tests`, no torch.
- `ensure_dirs` creates empty `data/images` and `data/crops` at startup, so the app user owns `/app/data` (not the model files, which stay root-owned and read only).

**Startup.** `docker run` (`--memory 2g --cpus 2 -e PORT=8080`) to `/api/health` with `model_loaded: true`: **2.2 s** (default settings) and **3.0 s** (deploy settings), image already local. No `probe_load_error`, no `face_guard_load_error`. Cloud Run adds the image pull on a cold start; not measured here.

**Identical responses.** 5 PSA test-split pairs (certs 88001086, 88002126, 78000563, 82000599, 76000681; picked at random with a read-only DB query): the container's `/api/grade` bodies are **byte-for-byte identical** to local `uvicorn backend.app.main:app` (Windows, onnxruntime 1.25.0 CPU, cv2 4.11.0) on all 5, across two container runs with default settings and one with the deploy settings (`GRADE_CONCURRENCY=2`, `ACCESS_CODE`, CORS vars).

**Latency and memory** (2 CPUs, 2 GiB limit; per grade = one request's wall time):
| Config | Sequential, per grade | 4 at once (wall) | Peak memory (cgroup `memory.peak`) |
|---|---|---|---|
| Default (`GRADE_CONCURRENCY=4`) | 8.2 s and 9.0 s mean (two runs) | 28.8 s / 31.0 s | 1.51 GiB (1,626 MB) |
| Deploy (`GRADE_CONCURRENCY=2`) | 7.1 s mean (6.8–7.5) | 31.5 s (last request queued ~24 s) | 1.46 GiB (1,572 MB; anon 1.42 GiB) |
| Local uvicorn on the host (all cores) | 4.0 s mean | 13.7 s | — |

- Memory hardly depends on concurrency (the ONNX session and arenas dominate). **`--memory 2Gi`** leaves ~0.5 GiB (25%) headroom over the measured peak, so the guide uses 2Gi; switch to 4Gi only if Cloud Run logs an out-of-memory restart.
- With 2 vCPUs a grade takes ~7 s (vs ~3–4 s on this desktop), and 4 simultaneous grades finish in ~30 s. At `GRADE_CONCURRENCY=2` a 5th simultaneous request would wait past the 30 s queue timeout and get the 503 "server busy" (with `Retry-After`); fine for a few friends.

**Access code and CORS in the container** (deploy settings): `/api/grade` without the header → 401 `{"detail":"access code required"}`, wrong code → 401, right code → 200; all three carry `Access-Control-Allow-Origin` for a preview origin matching the regex. Preflight with `Access-Control-Request-Headers: x-access-code` → 200 and allows the header; a non-matching origin's preflight → 400. `/api/health` → 200 with no code.

**API changes.** `ACCESS_CODE` is a plain ASGI middleware (`backend/app/core/access.py`) rather than a FastAPI dependency, because FastAPI parses the multipart body before it resolves dependencies; it's added only when the code is set, inside `CORSMiddleware`. `CORS_ORIGIN_REGEX` goes to `allow_origin_regex` (Starlette full-matches it). Tests: `backend/tests/test_access_code.py` (29: defaults off, no code → unchanged, missing/wrong/right code, header case, health open, no grading slot taken, the 401 sent before `receive` is ever called, regex allows previews and rejects lookalikes, preflight allows `X-Access-Code`, the 401 is readable cross-origin) and `test_deploy_files.py` (4).

**Frontend.** `src/lib/accessCode.ts` (plain TS, tested by `accessCode.test.ts`, 13 tests) + `api-client.ts` (`gradeCard(front, back, accessCode)`, a 401 throws `AccessCodeError`) + an inline code form in `Uploader.tsx`. Removed: `wrangler.jsonc`, `open-next.config.ts`, `cloudflare-env.d.ts`, the `deploy`/`upload`/`preview`/`cf-typegen` scripts, `@opennextjs/cloudflare` and `wrangler` (`npm uninstall`: 355 lockfile entries removed, no other version changed), the `next.config.ts` hook, the tsconfig type entry, the OpenNext/wrangler ignore lines, `public/_headers` (Cloudflare static-asset headers; Vercel caches `/_next/static` itself) and `.vscode/settings.json` (only a wrangler file association). New short `frontend/README.md`.

**Gates.** `ruff check .` clean. Fast suite: **1641 passed, 1 skipped, 6 xfailed** (baseline 1608 + the 33 new tests), data guard: no change. Frontend: `npm run lint`, `npx tsc --noEmit`, `npm run build`, `npm test` (23 tests) pass; `npm ci --dry-run` is consistent with the new lockfile.

**Surprises.** Something on this machine already listens on host port 8080 (an Apache 404), so local runs map the container to 18080 (the guide does too). Grades are ~2× slower on 2 vCPUs than on the desktop, so the 10–20 s cold-start assumption is now mostly image pull, and per-grade latency on Cloud Run is likely ≥ 7 s. Not verified (needs the cloud): Cloud Run cold start, Cloud Build behaviour with `.gcloudignore`, and the `^DELIM^` escaping through `gcloud.cmd` (the guide avoids commas and offers `--env-vars-file`).

### After the test agent (PASS)

Fixed before the PR:
- **`docs/deploy.md` §2.2:** the `--env-vars-file` fallback now uses single-quoted YAML values, with a sample file. In double quotes, `\.` in the regex is invalid YAML, so gcloud would reject the file.
- **Access-code characters:** the guide and `owner-steps.md` say letters, digits and hyphens only, because `&`, `^`, `%` and `!` get mangled by `gcloud.cmd`, and non-ASCII can't be sent in a header. `$CODE` is now single-quoted. The frontend refuses a code that isn't printable ASCII (`isValidCode`, tested).
- **`ACCESS_CODE` is stripped** at startup, and a blank code means no lock (`test_configured_code_is_stripped`, `test_whitespace_only_code_means_no_code`).
- **Docs:**
  - the source bucket is found with `gcloud storage ls`, since its name varies;
  - §3 says to set `$CODE` again in a new window;
  - §4.2 and `owner-steps.md` say to check that the Vercel project name is `pokescan` and to pick `frontend`, not FastAPI.

Gates after the fixes:
- Fast suite: 1643 passed, 1 skipped, 6 xfailed; data guard no change. `ruff check .` clean.
- Frontend: 24 tests, lint, typecheck and build pass.
