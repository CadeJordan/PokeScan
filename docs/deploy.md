# Deploy PokeScan: API on Cloud Run, frontend on Vercel

This takes you from nothing to a working site that you and a few friends can use, at about $0 a month:

- **The API** (FastAPI + the probe model) runs as a container on **Google Cloud Run**. It scales to zero when idle and never runs more than one instance.
- **The frontend** (Next.js) runs on **Vercel**'s free Hobby plan. Browsers call the API directly.
- **A shared access code** (`ACCESS_CODE`) keeps strangers who find the URL from grading cards. The site asks for it once and remembers it.
- **Per-IP rate limits** make guessing the code impractical (5 wrong codes a minute) and stop one visitor from tying up the instance (30 grades an hour).

Every command is for **Windows PowerShell**, run from `C:\Users\User\dev\poke-scan\PokeScan` (the folder with the `Dockerfile`) unless it says otherwise. Anything in `<angle brackets>` is yours to fill in. A backtick (`` ` ``) at the end of a line continues the command on the next line, so copy the whole block.

The short owner checklist is `intent/cloud-run-deploy/owner-steps.md`; this guide is the full version.

---

## What gets deployed

| Piece | Where it comes from |
|---|---|
| API image | The root `Dockerfile`: `python:3.11-slim`, the serving packages in `deploy/requirements-serve.txt` (no torch), `backend/` (no tests) and six model files copied from `data/models/`: `card_features.onnx` + `.json`, `grade_head.npz` + `.json`, `face_guard.npz` + `.json`. It runs as a non-root user, one uvicorn worker on `$PORT`. |
| What is uploaded | Only what `.gcloudignore` allows: the same allow-list as `.dockerignore`. Never `data/images`, crops, the DB, backups, `.env` or `frontend/`. |
| Frontend | The `frontend/` folder, built by Vercel on every push to `main`. |

The model is baked into the image, so the API always serves the model files that were in `data/models/` when you last deployed.

---

## 1. Google Cloud: project, billing, budget alert, CLI

### 1.1 Project and billing
1. Go to https://console.cloud.google.com and sign in. New accounts are offered free trial credit; accepting it is fine.
2. Create a **billing account** and add a card. Cloud Run needs billing even when usage stays inside the free tier.
3. Create a project, e.g. **PokeScan**, and note its **Project ID** (globally unique, e.g. `pokescan-123456`).

### 1.2 Budget alert (do this before deploying)
1. In the console open **Billing → Budgets & alerts → Create budget**.
2. Scope it to the PokeScan project, set the amount to **$5**, and keep the email alerts at 50%, 90% and 100%.

A budget alert only **emails** you; it doesn't stop spending. The hard cap is `--max-instances 1` in the deploy command below: at most one copy of the API ever runs, however much traffic arrives.

### 1.3 Install and set up the gcloud CLI
```powershell
winget install Google.CloudSDK
```
(or use the installer from https://cloud.google.com/sdk/docs/install). Then **open a new PowerShell window** so `gcloud` is on your PATH, and run:

```powershell
gcloud --version
gcloud auth login                                   # opens a browser to sign in
gcloud config set project <your-project-id>         # every later command uses this project
gcloud config set run/region us-central1            # default region for Cloud Run commands
```

`us-central1` is one of the regions where Cloud Run's free tier applies.

### 1.4 Turn on the APIs the deploy uses
```powershell
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
```
- **Cloud Run** runs the container.
- **Cloud Build** builds the image from the `Dockerfile` in Google's cloud, so you don't need Docker running locally.
- **Artifact Registry** stores the built image.

Enabling them is free; you pay only for usage. It can take a minute.

---

## 2. Build and deploy the API

### 2.1 Pick an access code
Choose a code to share with friends, e.g. three random words like `pikachu-teal-river`. Use **only letters, digits and hyphens**: other characters can be mangled by the Windows command line on the way into gcloud (`&`, `^`, `%`, `!`), and spaces or accented letters can't travel in the header the site sends. Keep it somewhere private (a password manager). **Don't** put it in the repo or any file under `PokeScan/`: it only goes into the deploy command. It's a shared lock for friends, not a real secret, so a plain Cloud Run environment variable is enough.

### 2.2 Deploy
Check that the six model files exist first (the build copies them in):
```powershell
Get-ChildItem data\models\card_features.onnx, data\models\card_features.json, data\models\grade_head.npz, data\models\grade_head.json, data\models\face_guard.npz, data\models\face_guard.json
```

Then, from `PokeScan/`:
```powershell
$CODE = '<your-access-code>'   # single quotes, so PowerShell leaves it as typed
gcloud run deploy pokescan-api `
  --source . `
  --region us-central1 `
  --cpu 2 `
  --memory 2Gi `
  --min-instances 0 `
  --max-instances 1 `
  --cpu-boost `
  --timeout 60 `
  --allow-unauthenticated `
  --set-env-vars "GRADE_CONCURRENCY=2,CORS_ORIGINS=http://localhost:3000,CORS_ORIGIN_REGEX=https://pokescan[-a-z0-9]*\.vercel\.app,ACCESS_CODE=$CODE,TRUST_FORWARDED_FOR=true,GRADES_PER_HOUR=30"
```

**Already deployed** (the service exists and you've set CORS in step 4.3)? Don't re-run the command above: `--set-env-vars` replaces **every** variable, which would reset your CORS values. Redeploy with `--update-env-vars`, which changes only the variables it names and keeps the rest (`ACCESS_CODE`, `CORS_ORIGINS`, `CORS_ORIGIN_REGEX`, `GRADE_CONCURRENCY`) and the CPU, memory and instance settings:
```powershell
gcloud run deploy pokescan-api --source . --region us-central1 --update-env-vars "TRUST_FORWARDED_FOR=true,GRADES_PER_HOUR=30"
```

What each part does:

| Flag | Why |
|---|---|
| `--source .` | Uploads the folder (filtered by `.gcloudignore`: backend code, requirements, `Dockerfile`, the six model files, about 350 MB) and has Cloud Build build the `Dockerfile`. The first run takes several minutes. |
| `--region us-central1` | A free-tier region. |
| `--cpu 2 --memory 2Gi` | 2 vCPUs and 2 GiB. Measured in a local container with the same limits: peak memory 1.46–1.51 GiB under 4 simultaneous grades, about 7 s per grade (see `intent/cloud-run-deploy/intent.md`, Results). |
| `--min-instances 0` | Scales to zero when idle, so idle time costs nothing. The first request after idling waits for a cold start. |
| `--max-instances 1` | **The cost cap.** Never more than one instance. |
| `--cpu-boost` | Extra CPU while the instance starts, so the model loads faster on a cold start. |
| `--timeout 60` | A request is cut off after 60 s. A grade that waits for a free slot gives up after 30 s with a 503 "server busy, try again", well inside this. |
| `--allow-unauthenticated` | Anyone can reach the URL; the access code is the lock on grading. `/api/health` stays open (it reveals nothing about the code). |
| `GRADE_CONCURRENCY=2` | Grades run at once inside the instance. With 2 vCPUs, more doesn't finish sooner. |
| `CORS_ORIGINS` | Exact browser origins allowed to call the API. `http://localhost:3000` for now; you'll set the Vercel URL in step 4.3. |
| `CORS_ORIGIN_REGEX` | Extra origins as a pattern, for Vercel's preview deployments (each gets a new URL). It must match the whole origin. You'll tighten it in step 4.3. |
| `ACCESS_CODE` | The shared code. `/api/grade` answers 401 without it. |
| `TRUST_FORWARDED_FOR=true` | Count rate limits per visitor IP. Behind Cloud Run the connection comes from Google's front end, which puts the visitor's IP at the right-most end of the `X-Forwarded-For` header; the API trusts only that entry (anything a client forges sits to its left). Default `false`: the TCP peer is used and the header is ignored. Turn it on only behind a proxy that appends the header, and run the check in step 3.1. |
| `GRADES_PER_HOUR=30` | At most 30 grades per IP in any hour; the 31st gets 429 "grade limit reached (30 per hour), try again later" with `Retry-After`. Only requests that reach the model count: a refused upload (too large, wrong format, empty), a wrong code or a "server busy" doesn't; an unreadable photo does, since it held a grading slot. IPv6 visitors are counted per /64 (one connection's block of addresses) for both limits. Default `0` = no limit. |
| `CODE_ATTEMPTS_PER_MINUTE` (not set: default `5`) | Missing or wrong access codes per IP per minute. Past it, every `/api/grade` request from that IP gets 429 "too many attempts, wait a minute" with `Retry-After`, **even with the right code**, until its oldest failure is a minute old. A first visit uses 1 (the page tries once without a code). Only applies with `ACCESS_CODE` set; `0` turns it off. |

The rate limits are counted in the instance's memory: they reset when it restarts (e.g. after scaling to zero), and they're exact because there's only ever one instance with one worker. `/api/health` is never limited.

> **Shared IPs.** Everyone behind the same public IP (friends on the same Wi-Fi, a phone carrier's NAT) shares one set of limits: one person typing the code wrong 5 times locks the others out for a minute, and together they get 30 grades an hour. Fine for a few friends; raise `GRADES_PER_HOUR` with `--update-env-vars` if it bites.

- The first time, gcloud asks to create an Artifact Registry repository (`cloud-run-source-deploy`). Answer **Y**.
- When it finishes it prints the **Service URL**, like `https://pokescan-api-xxxxxxxxxx-uc.a.run.app`.

> **Values with commas.** `--set-env-vars` splits on commas, so a value that contains one (e.g. two exact origins in `CORS_ORIGINS`) needs gcloud's `^DELIM^` syntax, which switches the separator: `"^@^CORS_ORIGINS=https://a.vercel.app,https://b.vercel.app@ACCESS_CODE=..."` (see `gcloud topic escaping`). On Windows `gcloud` runs through a `.cmd` wrapper, where `^` is cmd's escape character and can be swallowed. If that happens, put the variables in a YAML file **outside the repo** (e.g. `$HOME\pokescan-env.yaml`) and pass `--env-vars-file $HOME\pokescan-env.yaml` instead of `--set-env-vars`. Use **single quotes** around every value: in YAML double quotes, `\.` in the regex is an error and gcloud rejects the file.
>
> ```yaml
> GRADE_CONCURRENCY: '2'
> CORS_ORIGINS: 'https://pokescan.vercel.app,https://pokescan-git-main-<your-scope>.vercel.app'
> CORS_ORIGIN_REGEX: 'https://pokescan[-a-z0-9]*\.vercel\.app'
> ACCESS_CODE: '<your-access-code>'
> TRUST_FORWARDED_FOR: 'true'
> GRADES_PER_HOUR: '30'
> ```
> With the regex covering previews, one exact origin is usually enough, so you shouldn't need this.

### 2.3 If the build fails with a permissions error
Some new projects don't give the build's service account the roles it needs. Grant it the Cloud Run Builder role and deploy again:
```powershell
$PROJECT = gcloud config get-value project
$NUMBER = gcloud projects describe $PROJECT --format "value(projectNumber)"
gcloud projects add-iam-policy-binding $PROJECT `
  --member "serviceAccount:$NUMBER-compute@developer.gserviceaccount.com" `
  --role roles/run.builder
```

---

## 3. Check the API

```powershell
$API = gcloud run services describe pokescan-api --region us-central1 --format "value(status.url)"
$API
curl.exe "$API/api/health"
```
Use `curl.exe`, not `curl`: in Windows PowerShell `curl` is an alias for `Invoke-WebRequest`.

Look for `"model_loaded":true`. If it's `false`, `metadata.probe_load_error` says why (usually a model file missing from `data/models/` at deploy time). The first call after idling takes a few seconds longer: the instance starts and loads the model before it answers (about 2–3 s in the local container, plus Cloud Run pulling the ~0.8 GB image).

Check the lock, too. If this is a new PowerShell window, set `$CODE` again first (`$CODE = '<your-access-code>'`); otherwise the second line sends an empty code and shows 401.
```powershell
curl.exe -s -o NUL -w "%{http_code}`n" -X POST "$API/api/grade"                              # 401: no code
curl.exe -s -o NUL -w "%{http_code}`n" -X POST -H "X-Access-Code: $CODE" "$API/api/grade"     # 422: code accepted, no images sent
```

### 3.1 Check that the rate limit can't be dodged (after every deploy that sets `TRUST_FORWARDED_FOR=true`)
The limit relies on Google's front end *appending* your real IP to `X-Forwarded-For`, so a forged entry ends up on the left where it's ignored. Check it: send 6 wrong codes, each with a different fake `X-Forwarded-For`. **Wait a minute after the lock check above first** (its first request was a failure too), then:
```powershell
1..6 | ForEach-Object { curl.exe -s -o NUL -w "%{http_code}`n" -X POST -H "X-Access-Code: wrong-$_" -H "X-Forwarded-For: 198.51.100.$_" "$API/api/grade" }
```
- **Expected:** `401` five times, then **`429`**. The forged IPs didn't help: all six were counted against your real IP.
- **If the 6th is `401`:** the API is trusting the forged entry, so anyone can dodge both limits. Stop relying on the limits and fix it before sharing the site: set `TRUST_FORWARDED_FOR=false` with `gcloud run services update pokescan-api --region us-central1 --update-env-vars "TRUST_FORWARDED_FOR=false"` and look into it (with it off, every visitor shares the front end's IP, so the limits are shared too).
- Your IP is now locked out for a minute (even with the right code). Wait a minute before grading.

One more check, that the limit is per visitor and not shared by everyone: while your PC is still locked out (within the minute), open the site on your phone **on mobile data, not your Wi-Fi**, and grade a pair. It should work. If it says "too many attempts, wait a minute", the API sees the same IP for every visitor; set `TRUST_FORWARDED_FOR=false` as above and look into it.

---

## 4. Frontend on Vercel

### 4.1 Account
Sign up at https://vercel.com/signup **with GitHub**, on the free **Hobby** plan, and let the Vercel GitHub app access the **CadeJordan/PokeScan** repo.

### 4.2 Import the project
1. **Add New → Project**, pick **PokeScan**.
2. Set **Root Directory** to `frontend`. Vercel detects Next.js; keep the default build settings.
3. Under **Environment Variables** add `NEXT_PUBLIC_API_BASE` = `<Service URL>/api` (e.g. `https://pokescan-api-xxxxxxxxxx-uc.a.run.app/api`), for **Production** and **Preview**. It's built into the page, so after changing it, redeploy.
4. **Check the project name** Vercel suggests before deploying. The starting pattern in step 2.2 assumes it begins with `pokescan`. If Vercel suggests something else (e.g. `poke-scan` or `frontend`), rename it to `pokescan` here, or the site gets CORS errors until step 4.3.
5. **Deploy.** Vercel prints the production URL, e.g. `https://pokescan.vercel.app` (or `https://pokescan-<something>.vercel.app` if that name is taken).

### 4.3 Allow the Vercel site to call the API
The browser only lets the page call the API if the API lists its origin. Set the production URL (no trailing slash) and, once you've seen a preview URL, a pattern that matches only your previews. Vercel preview URLs look like `https://pokescan-<hash>-<your-scope>.vercel.app` and `https://pokescan-git-<branch>-<your-scope>.vercel.app`, where `<your-scope>` is your Vercel account slug (e.g. `cadejordans-projects`; check the URL of any deployment in the Vercel dashboard).

```powershell
gcloud run services update pokescan-api --region us-central1 `
  --update-env-vars "CORS_ORIGINS=https://<your-app>.vercel.app,CORS_ORIGIN_REGEX=https://pokescan-[-a-z0-9]+-<your-scope>\.vercel\.app"
```
- `--update-env-vars` changes only the named variables and keeps the rest (including `ACCESS_CODE`). It makes a new revision in under a minute; nothing is rebuilt.
- The starting pattern `https://pokescan[-a-z0-9]*\.vercel\.app` also matches other people's Vercel sites whose names start with `pokescan`. The access code still protects grading, but the tighter pattern is better.
- Before you know a preview URL, the production URL alone is fine: leave `CORS_ORIGIN_REGEX` out of the command.

### 4.4 Smoke test
1. Open the Vercel URL. The navbar shows the model status (it calls `/api/health`).
2. Pick a real front and back photo and press **Predict grade**. The page asks for the access code; enter it and press **Unlock and grade**. The grade appears after a few seconds (longer on a cold start).
3. Grade a second pair: no code is asked for (it's remembered in this browser).
4. If you see "That code didn't work", the code you typed doesn't match `ACCESS_CODE`; the stored one has been cleared.

Then share the Vercel link and the code with your friends.

If the page says it can't reach the API, open the browser's developer tools (F12) → Console: a CORS error means the page's origin isn't in `CORS_ORIGINS`/`CORS_ORIGIN_REGEX` (step 4.3).

---

## 5. Optional: test the image locally first

Docker Desktop must be running. This is what the deploy was verified with; it doesn't touch the cloud.
```powershell
docker build -t pokescan-api:local .
docker run --rm -d --name pokescan-local --memory 2g --cpus 2 -e PORT=8080 -e ACCESS_CODE=test -e TRUST_FORWARDED_FOR=true -e GRADES_PER_HOUR=30 -p 127.0.0.1:18080:8080 pokescan-api:local
curl.exe http://127.0.0.1:18080/api/health
docker stats --no-stream pokescan-local
docker stop pokescan-local
```
(Port 18080 on your side, in case something else already uses 8080.)

---

## 6. Updating

- **New model** (after `train_probe` or `train_face_guard`): the model is inside the image, so redeploy: `gcloud run deploy pokescan-api --source . --region us-central1` (the "already deployed" form in step 2.2, without an env-vars flag, or with `--update-env-vars` to change some). Never re-run the first-time command with `--set-env-vars`: it resets the CORS settings from step 4.3.
- **Backend code change:** same: redeploy with `--source .`.
- **Frontend:** Vercel redeploys by itself on every push to `main`, and builds a preview for every other branch and pull request.
- **Change the access code:** `gcloud run services update pokescan-api --region us-central1 --update-env-vars "ACCESS_CODE=<new-code>"` (letters, digits and hyphens only). Everyone's stored code stops working; the page asks again.
- **Change a rate limit:** e.g. `gcloud run services update pokescan-api --region us-central1 --update-env-vars "GRADES_PER_HOUR=60"` (`0` turns it off). No rebuild.
- **Rollback:** Cloud Run keeps old revisions. In the console, **Cloud Run → pokescan-api → Revisions**, pick one and **Manage traffic** to send 100% to it.

---

## 7. Costs and limits

Don't take numbers from this page as fact; the pricing pages are the source of truth:
- Cloud Run: https://cloud.google.com/run/pricing (the free tier is a monthly allowance of vCPU-seconds, GiB-seconds and requests, in Tier 1 regions such as `us-central1`).
- Artifact Registry: https://cloud.google.com/artifact-registry/pricing
- Cloud Build: https://cloud.google.com/build/pricing
- Vercel: https://vercel.com/pricing

What to expect:
- **Cloud Run:** you're billed only while a request is being handled (and while the instance starts). One grade uses about 7 s of 2 vCPUs and 2 GiB. With `--min-instances 0` idle time is free; `--max-instances 1` caps the worst case.
- **Image storage** (Artifact Registry) is the one thing likely to cost a few cents a month: each deploy stores a new ~0.8 GB image. Delete old ones now and then (below).
- **Cloud Build** has a free monthly allowance of build minutes; occasional deploys stay inside it. The uploaded source (~350 MB per deploy) is kept in a Cloud Storage bucket that gcloud creates (named like `run-sources-<project-id>-us-central1`, or `<project-id>_cloudbuild` on older setups; `gcloud storage ls` lists it), which you can empty.
- **Vercel Hobby** is free for personal, non-commercial use.

See usage in the console under **Billing → Reports**, and per service under **Cloud Run → pokescan-api → Metrics**.

Delete old images (keep the newest):
```powershell
$PROJECT = gcloud config get-value project
gcloud artifacts docker images list "us-central1-docker.pkg.dev/$PROJECT/cloud-run-source-deploy" --include-tags
gcloud artifacts docker images delete "us-central1-docker.pkg.dev/$PROJECT/cloud-run-source-deploy/pokescan-api@<digest>" --delete-tags
```

Limits worth knowing:
- Uploads are capped at 12 MB per photo by the API, well under Cloud Run's request size limit.
- One instance grades 2 photo pairs at once; more simultaneous requests queue, and one that waits over 30 s gets "server busy, try again". Fine for a few friends.
- Each IP gets 30 grades an hour and 5 wrong codes a minute (step 2.2); people on one Wi-Fi share them.

---

## 8. Take it down

```powershell
gcloud run services delete pokescan-api --region us-central1
gcloud artifacts repositories delete cloud-run-source-deploy --location us-central1
gcloud storage ls    # find the source bucket: run-sources-<project-id>-us-central1 or <project-id>_cloudbuild
gcloud storage rm --recursive "gs://<that-bucket-name>"
```
Or delete the whole project: `gcloud projects delete <your-project-id>` (stops all billing for it after a 30-day grace period).

On Vercel: **Project → Settings → Advanced → Delete Project**.
