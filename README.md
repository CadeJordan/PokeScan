# PokeScan

Predict the PSA grade (1–10) a Pokémon card will receive from front and back photos, plus a calibrated confidence score and per-factor sub-grades (centering, corners, edges, surface).

This project is independent and not affiliated with PSA or eBay. Respect each platform's API terms.

## What it does

1. **Detect & dewarp** — finds the card in the photo (including slab shots) and warps it to a 600×840 canvas.
2. **Overall grade** — dual-input ConvNeXt-Tiny + CORN ordinal regression over front and back images.
3. **Sub-grades** — classical computer vision (no extra labels required):
   - **Centering** — inner art-frame ratios mapped to PSA tolerances.
   - **Corners** — factory-cut sharpness and fraying from the original card contour.
   - **Edges** — boundary roughness (wear) and rim whitening.
   - **Surface** — strong linear defects (scratches / creases) isolated from the artwork.

Both sides are analyzed; the **worse side** sets each condition sub-grade (PSA-style).

## Architecture

```
Front/back image
  -> card detector (classical CV)
  -> ConvNeXt backbone + CORN head -> {grade, confidence}
  -> centering CV        -> centering subgrade
  -> condition CV         -> corners / edges / surface subgrades
```

Longer-term, the condition sub-grades can be replaced by trained multi-task heads (`backend/app/ml/multitask.py`) without changing the API.

## Repo layout

```
PokeScan/
  backend/              FastAPI inference service
    app/
      api/              HTTP endpoints (POST /grade, GET /health)
      ml/               Inference, preprocessing, centering, condition analysis
      core/             Config (pydantic-settings)
  ml/                   Offline training
    data/               PSA / eBay collectors, SQLite, PyTorch Dataset
    training/           Lightning loop, CORN loss, export to ONNX
  frontend/             Next.js app (upload UI, grade display)
  data/                 Local images, SQLite, model artifacts (gitignored by default)
  pyproject.toml
  .env                  Secrets and paths (copy from .env.example)
```

## Prerequisites

- **Python 3.10+**
- **Node.js 20+** (for the frontend)
- **Optional:** NVIDIA GPU + CUDA for training (CPU works for inference and data collection)

## Setup

### 1. Clone and enter the project

```powershell
cd PokeScan
```

### 2. Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

On Windows with an NVIDIA GPU, install PyTorch from the official index first if the default wheel is CPU-only:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
```

### 3. Environment variables

```powershell
copy .env.example .env
```

Edit `.env` with your credentials and paths:

| Variable | Purpose |
|---|---|
| `PSA_API_TOKEN` or `PSA_API_TOKEN0`…`PSA_API_TOKEN5` | PSA Public API bearer token(s). Use numbered tokens for multi-token daily rotation. |
| `EBAY_APP_ID`, `EBAY_CERT_ID` | eBay Browse API (primary bulk training source). |
| `DATA_DIR`, `IMAGES_DIR`, `DB_PATH` | Where harvested images and SQLite metadata live. |
| `ACTIVE_MODEL_PATH` | ONNX file loaded by the API (default `./data/models/grade_model.onnx`). |
| `CORS_ORIGINS` | Frontend origin(s), default `http://localhost:3000`. |

**PSA token:** https://www.psacard.com/publicapi/documentation  
**eBay keys:** https://developer.ebay.com/my/keys (Production App ID + Cert ID)

If your data directory is the parent folder (`../data` relative to `PokeScan/`), set:

```env
DATA_DIR=../data
IMAGES_DIR=../data/images
DB_PATH=../data/pokescan.sqlite
```

### 4. Frontend

```powershell
cd frontend
copy .env.example .env.local
npm install
```

`frontend/.env.local` should point at the API:

```env
NEXT_PUBLIC_API_BASE=http://localhost:8000/api
```

## Run the app

You need an exported ONNX model before `/grade` works. If none exists yet, skip to [Train a model](#train-a-model) or the health check will report `model_loaded: false`.

### 1. Start the API

From `PokeScan/` with the venv activated:

```powershell
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

- Health: http://localhost:8000/api/health  
- OpenAPI docs: http://localhost:8000/docs  

### 2. Start the frontend

In a second terminal:

```powershell
cd frontend
npm run dev
```

Open http://localhost:3000, upload front and back images, and view the predicted grade and sub-grades.

### 3. Grade via API (curl)

```powershell
curl -X POST http://localhost:8000/api/grade `
  -F "front=@path\to\front.jpg" `
  -F "back=@path\to\back.jpg"
```

Example response shape:

```json
{
  "grade": 9.2,
  "grade_int": 9,
  "confidence": 0.74,
  "rank_probs": [ ... ],
  "sub_grades": {
    "centering": { "grade": 9.0, "left_right": "52/48", "top_bottom": "51/49" },
    "corners": { "grade": 10.0, "hint": "clean", "flags": [] },
    "edges": { "grade": 10.0, "hint": "clean", "flags": [] },
    "surface": { "grade": 9.0, "hint": "clean", "flags": [] }
  },
  "notes": []
}
```

If the model is missing, `/grade` returns **503** with a message to run export first.

## Collect training data

Both sources write to the same SQLite `certs` table with a `source` column (`psa` | `ebay`).

### eBay (bulk training)

```powershell
python -m ml.data.ebay_collect --grades 6 7 8 9 10 --per-grade 500
```

Low grades are scarce on eBay; run additional passes if needed:

```powershell
python -m ml.data.ebay_collect --grades 1 2 3 4 5 --per-grade 200
```

### PSA (high-fidelity validation sample)

Free tier is ~100 calls/day per token. Prefer this for val/test, not bulk training.

Single token:

```powershell
python -m ml.data.collect --start-cert 90000000 --count 90
```

Multiple tokens in parallel windows (each token scans its own cert range):

```powershell
python -m ml.data.collect_all `
  --starts 76032000,76037000,76042000,76047000,76052000,76057000 `
  --count 5000 --daily-limit 200 --daily-limit-token0 1000
```

Resume from where a token stopped (use the logged cert number as the next start for that slot).

## Train a model

### MVP (overall grade only)

```powershell
python -m ml.data.split
python -m ml.data.precrop
python -m ml.training.train --config ml/training/configs/mvp.yaml
python -m ml.training.export --checkpoint data/models/checkpoints/best-XX.ckpt
```

### Multi-task (grade + corners/edges/surface)

Classical-CV pseudo-labels from the condition module supervise auxiliary factor
heads alongside the CORN grade head. The API still serves factor sub-grades from
classical CV at inference; the multitask heads improve the shared backbone.

```powershell
python -m ml.data.split
python -m ml.data.precrop              # one-time dewarp to data/crops/ (speeds training)
python -m ml.data.precompute_factors
python -m ml.training.train --config ml/training/configs/multitask.yaml
python -m ml.training.export --checkpoint data/models/checkpoints/best-XX.ckpt
```

Run `precrop` again after harvesting new certs (`--force` to rebuild all).
Training configs default to `use_precrop: true` and skip live `detect_and_crop`.

The export step writes `data/models/grade_model.onnx` (or the path in
`ACTIVE_MODEL_PATH`). Restart the API to load the new weights.

## Data sources

| Source | What you get | Caveats |
|---|---|---|
| **eBay Browse API** | PSA-graded Pokémon listings; titles parsed for grade; front/back picked by HSV side classifier; blur and resolution gates applied. | User photos — good for generalization, noisier than PSA catalogue shots. |
| **PSA Public API** | Official catalogue images + verified grades. | Low daily quota; cert images only from Oct 2021 onward. |

## Caveats

- PSA grades are imbalanced (9 and 10 dominate); training uses stratified sampling and CORN ordinal loss.
- Augmentations that change apparent condition (scratches, heavy rotation, edge crops) are **not** applied.
- Classical condition sub-grades are conservative heuristics calibrated on slab photos; they ship interpretable scores today and can be swapped for learned heads later.
- Condition **whitening** on corners is limited at slab-photo resolution; corner grades lean on geometry (rounding / jaggedness).

## Development

```powershell
# Lint
ruff check .

# Tests (when present)
pytest
```

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Service status and whether the ONNX model is loaded |
| `POST` | `/api/grade` | Multipart upload: `front` + `back` images (JPEG/PNG/WebP, max 12 MB each) |

## License / disclaimer

Not affiliated with PSA, The Pokémon Company, or eBay. Grades are predictions only — not official PSA assessments.
