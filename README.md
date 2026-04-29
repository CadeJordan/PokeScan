# PokeScan

Predicts the PSA grade (1-10) a Pokemon card will receive from a photo, plus a calibrated confidence score.

## Architecture

End-to-end CNN trained on PSA cert images. Phased toward an interpretable multi-task model:

- **Phase A (MVP)**: dual-input ConvNeXt-Tiny + CORN ordinal regression head -> overall grade.
- **Phase B**: classical-CV centering module (no labels needed).
- **Phase C**: hand-annotated corner/edge/surface defect heads, fine-tuned jointly with the grade head.

```
Front/back image -> card detector -> ConvNeXt backbone -> ordinal head -> {grade, confidence}
                                                       \-> centering CV -> centering subgrade
                                                       \-> defect heads -> per-factor subgrades (Phase C)
```

## Repo layout

```
PokeScan/
  backend/        FastAPI inference service
    app/
      api/        HTTP endpoints (POST /grade)
      ml/         Inference, preprocessing, classical-CV centering
      core/       Config (pydantic-settings)
  ml/             Offline training package
    data/         PSA API client, cert harvester, PyTorch Dataset
    training/     Lightning loop, CORN loss, metrics
    notebooks/    EDA, error analysis
  frontend/       Next.js 16 app (upload UI, grade display)
  data/           Local images, SQLite DB, model artifacts (gitignored)
  pyproject.toml
```

## Setup

### 1. Get a PSA API token (gold-standard validation source)
Sign in at https://www.psacard.com/publicapi/documentation and generate a bearer token. The free tier allows **100 calls/day** which is far too low for primary training data — use it for a clean held-out validation set, then ingest the bulk of training data from eBay (next step).

### 2. Get eBay Browse API credentials (primary training source)
1. Sign up as a developer at https://developer.ebay.com/my/keys.
2. Generate a **Production** keyset; you need the **App ID (Client ID)** and **Cert ID (Client Secret)**.
3. Default OAuth scope (`https://api.ebay.com/oauth/api_scope`) is enough for `Browse` search.

Copy `.env.example` to `.env` and fill in:
```
PSA_API_TOKEN=your_psa_bearer_token
EBAY_APP_ID=your_ebay_app_id
EBAY_CERT_ID=your_ebay_cert_id
```

### 3. Backend (Python 3.10+)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

For GPU PyTorch on Windows you may want to install torch from the official index first:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Frontend (Node 20+)
```powershell
cd frontend
npm install
npm run dev
```

## Workflow

Both data sources write to the same SQLite table (`certs`) tagged by a `source` column (`'psa'` | `'ebay'`). The split, dataset and trainer all consume them transparently.

```powershell
# 1a. Harvest the bulk of training data from eBay (free; no daily cap as
#     restrictive as PSA's). Aims for ~500 listings per grade by default.
python -m ml.data.ebay_collect --grades 6 7 8 9 10 --per-grade 500

# 1b. (Optional) Harvest a small high-fidelity sample from PSA's API.
#     Use this for validation/test rather than burning quota on training.
python -m ml.data.collect --start-cert 90000000 --count 90

# 2. Stratified train/val/test split across all sources.
python -m ml.data.split

# 3. Train the MVP grade model
python -m ml.training.train --config ml/training/configs/mvp.yaml

# 4. Export to ONNX for inference
python -m ml.training.export --checkpoint data/models/best.ckpt

# 5. Run the API
uvicorn backend.app.main:app --reload

# 6. Run the frontend
cd frontend; npm run dev
```

## Data sources

| Source | What it gives you | Caveats |
|---|---|---|
| **eBay Browse API** | Active listings of PSA-graded Pokemon cards. Title parsed for grade; lot/multi-card listings auto-skipped. Image URLs upscaled to `s-l1600` before download. Front and back are picked by a color-histogram side classifier (Pokemon backs have a uniquely tight blue+yellow signature in HSV space), then both images run through a Laplacian-variance blur check + min-resolution gate. | Photos are user-supplied (varied lighting/angles) — *good* for generalisation, but noisier than PSA's catalogue. The collector's classifier rejects ambiguous titles to keep label noise low. |
| **PSA Public API** | PSA's official catalogue images + verified grades. | 100 calls/day on the free tier; only certs from Oct 2021+ have images. |

You can also run additional `ebay_collect` invocations targeting harder-to-find low grades (e.g. `--grades 1 2 3 4 5 --per-grade 200`) — those classes are scarce on eBay so expect lower yield.

## Caveats

- PSA only began storing cert images in October 2021 - older certs return metadata only.
- PSA grades are imbalanced (9 and 10 dominate); training uses stratified sampling and a CORN ordinal loss.
- Augmentations that change the apparent grade (added scratches, large rotations, edge crops) are NOT applied.
- This tool is independent and not affiliated with PSA or eBay. Respect each platform's API terms.
