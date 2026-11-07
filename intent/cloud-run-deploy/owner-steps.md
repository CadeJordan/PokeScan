# Your steps to get PokeScan live

These are the account steps only you can do. Part A can be done **now**, while the container, API changes and guide are being built. Part B waits until that PR is merged. The full guide is `docs/deploy.md`; this page is the short version.

Commands are for **Windows PowerShell**. Anything in `<angle brackets>` is yours to fill in.

---

## Part A: do now (about 20–30 minutes)

### 1. Create a Google Cloud account and project
1. Go to https://console.cloud.google.com and sign in with your Google account. New accounts are offered free trial credit, which is fine to accept.
2. Create a **billing account** and add a card. Cloud Run needs one even when you stay inside the free tier.
3. Create a project, e.g. **PokeScan**. Note its **Project ID**: it's globally unique, so it may come out as something like `pokescan-123456`.

### 2. Set a budget alert (the safety net)
1. In the console, open **Billing → Budgets & alerts → Create budget**.
2. Scope it to the PokeScan project, set the amount to **$5**, and keep the email alerts at 50%, 90% and 100%.

A budget alert only emails you; it doesn't stop spending. The real cap is `--max-instances 1` in the deploy command. Together they make a surprise bill very unlikely.

### 3. Install the gcloud CLI
Either:
```powershell
winget install Google.CloudSDK
```
or download the installer from https://cloud.google.com/sdk/docs/install.

Then **open a new PowerShell window** and run:
```powershell
gcloud --version
```

### 4. Log in and point gcloud at the project
```powershell
gcloud auth login
gcloud config set project <your-project-id>
gcloud config set run/region us-central1
```
`us-central1` is one of the regions covered by Cloud Run's free tier.

### 5. Turn on the services the deploy uses
```powershell
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
```
This can take a minute. It's free; you're only billed for usage.

### 6. Create a Vercel account
1. Go to https://vercel.com/signup and sign up **with GitHub**, on the free **Hobby** plan.
2. When asked, give the Vercel GitHub app access to the **CadeJordan/PokeScan** repo, or to all repos.
3. **Don't import the project yet.** The frontend still has the Cloudflare setup until the PR merges.

### 7. Pick an access code
Choose a code to share with your friends, e.g. three random words like `pikachu-teal-river`. Use **only letters, digits and hyphens**: other characters can get mangled on the Windows command line or can't be sent by the browser.
- Keep it somewhere private (a password manager or a note).
- **Don't** put it in the repo or in any file under `PokeScan/`. It only ever goes into the Cloud Run deploy command.

### 8. Tell me when Part A is done
Mention any step that errored, with the message, and I'll sort it out.

---

## Part B: after the deploy PR is merged

The exact commands are in `docs/deploy.md`. In outline:

1. **Deploy the API.** From `C:\Users\User\dev\poke-scan\PokeScan`, run one `gcloud run deploy ... --source .` command.
   - It uploads the code plus the ~350 MB model and builds the image in the cloud, so the first run takes several minutes.
   - It prints the service URL, like `https://pokescan-api-xxxxx-uc.a.run.app`.
   - The first time, gcloud may ask to create an Artifact Registry repository. Answer **Y**.
   - If the build stops with a permissions error, run the one-off fix in the guide's step 2.3 and deploy again.
2. **Check it.** Open `<service URL>/api/health` in a browser and look for `"model_loaded": true`. The first load after idling takes a few seconds longer (the model loads in about 2–3 s locally, plus Cloud Run fetching the image).
3. **Import the frontend on Vercel.**
   - Choose **Add New → Project**, pick PokeScan, and set **Root Directory** to `frontend`.
   - Choose **`frontend`** (Next.js), not the FastAPI option: the API runs on Cloud Run.
   - Check the project name is `pokescan` (rename it if Vercel suggests something else, e.g. `frontend` or `poke-scan`); the API's starting CORS pattern expects that name.
   - Add the environment variable `NEXT_PUBLIC_API_BASE` = `<service URL>/api`, for Production and Preview.
   - Deploy.
4. **Allow the Vercel site to call the API.** Put the Vercel URL into the API's allowed origins with the one-line `gcloud run services update` command in the guide (step 4.3).
5. **Try it.** Open the Vercel site, enter the access code, and grade a real card pair.
6. **Share** the Vercel link and the access code with your friends.

## Later

- **After retraining the model,** re-run the deploy command so the API gets the new model. The frontend redeploys by itself on every push to `main`.
- **To check costs,** look at **Billing → Reports** in the console.
- **To take it all down,** delete the Cloud Run service and the Artifact Registry repository, and remove the project in Vercel.
