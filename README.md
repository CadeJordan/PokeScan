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

Low grades are scarce (the PSA test split has only ~20 cards graded 8 or
below), so run a dedicated low-grade pass. `--per-grade` is a target; the
collector stops early when eBay runs out of matching listings. Then re-split
and re-crop so the new certs are usable:

```powershell
python -m ml.data.ebay_collect --grades 1 2 3 4 5 6 7 8 --per-grade 300
python -m ml.data.split          # eBay certs always go to train; PSA keeps val/test
python -m ml.data.precrop        # crops only the new certs (add --force to rebuild all)
python -m ml.data.precompute_factors   # only if you train multitask.yaml
```

`split.py` stratifies **only PSA certs** into val/test; every `source='ebay'`
cert goes to train (`--ebay-split stratify` restores the old behaviour). Adding
eBay rows therefore never reshuffles the PSA val/test assignment, and the eval
numbers stay comparable across harvests.

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

### Training knobs (all configs)

| Key | Default | Meaning |
|---|---|---|
| `sampler_power` | `0.5` | Sampler weights `count(grade) ** -power`. `0` = natural distribution, `0.5` = `1/sqrt(count)`, `1` = full inverse frequency (the old behaviour; draws the single grade-1 card ~1000x more than a 10 and biases the model low). Only used when `balanced_sampler: true`. |
| `aug_strength` | `1.0` | Scales the lighting/colour augmentation (brightness, contrast, gamma, white balance, small hue shift, rare mild CLAHE). `0` turns it off. No blur, noise, scratches or crops, ever. |
| `monitor` / `monitor_mode` | `val/qwk` / `max` | Metric for checkpointing and early stopping. MAE rewards always predicting 10; QWK does not. Checkpoints are named `best-EE-qwk0.XXX.ckpt`. |
| `run_name` | unset | Sub-folder for checkpoints (`data/models/checkpoints/<run_name>/`) and CSV logs. |

`train.py` also takes `--sampler-power`, `--aug-strength` and `--run-name` to
override the config from the command line.

### Calibration: temperature + threshold offsets

`export.py` fits two things on the val split and bakes both into the graph as
`corn_logits = logits / T + b`, so the API and `/api/health` need no changes:

1. a temperature `T` (minimises val NLL, so confidences are calibrated), and
2. nine per-threshold offsets `b` (coordinate search that minimises val
   `MAE - QWK`; `--offset-objective mae|qwk` to change). This removes
   systematic grade bias without retraining. `T` is then re-fitted on the
   offset logits; that never changes a hard grade.

Both values and the val metrics before/after go into `grade_model.json`
(`temperature`, `threshold_offsets`, `val_metrics_temperature_only`,
`val_metrics_calibrated`). Flags: `--no-fit-offsets` (temperature only),
`--offsets-json <file>` (re-use offsets from an earlier `grade_model.json` or
calibration JSON), `--calibration-only` (fit and report, write nothing under
`data/models/`), `--no-activate` (export to `--out-onnx` without replacing the
served model).

**Quick win: re-export the current checkpoint with offsets** (no retraining):

```powershell
# 1. Preview: fit on val, print MAE/QWK/bias before -> after, write nothing under data/models/
python -m ml.training.export --checkpoint data/models/checkpoints/best-14-0.863.ckpt --calibration-only --calibration-out test-artifacts/calibration_v4.json

# 2. Export side by side and evaluate it on test before promoting it
python -m ml.training.export --checkpoint data/models/checkpoints/best-14-0.863.ckpt `
  --out-onnx data/models/candidates/v4_offsets.onnx --out-torchscript data/models/candidates/v4_offsets.pt --no-activate
python -m ml.eval.evaluate --split test --model-path data/models/candidates/v4_offsets.onnx --out-dir test-artifacts/eval/v4_offsets

# 3. Promote it if it wins (or re-run step 2 without --no-activate), then restart the API
python -m ml.training.export --checkpoint data/models/checkpoints/best-14-0.863.ckpt --offsets-json data/models/candidates/v4_offsets.json
```

Do the same with the single-task v3 checkpoint (`best-03-0.668.ckpt`).

### Experiment matrix

Run each configuration on the same clean crops (Step 1 done, `precrop --force`
run). Pick the winner by **val** QWK, then confirm it on **test**. It must beat
always-predict-10 on MAE (0.64 on the current test split).

| Run | Config | `sampler_power` |
|---|---|---|
| `mvp_sp0` | `mvp.yaml` | 0 |
| `mvp_sp05` | `mvp.yaml` | 0.5 |
| `mvp_sp1` | `mvp.yaml` | 1 |
| `mt_sp0` | `multitask.yaml` | 0 |
| `mt_sp05` | `multitask.yaml` | 0.5 |
| `mt_sp1` | `multitask.yaml` | 1 |

```powershell
# Train (use mvp_8gb.yaml instead of mvp.yaml on an 8 GB GPU)
foreach ($p in 0, 0.5, 1) {
  $tag = "$p".Replace(".", "")
  python -m ml.training.train --config ml/training/configs/mvp.yaml       --sampler-power $p --run-name "mvp_sp$tag"
  python -m ml.training.train --config ml/training/configs/multitask.yaml --sampler-power $p --run-name "mt_sp$tag"
}

# Export each run's best checkpoint (highest qwk in the file name) side by side
foreach ($run in "mvp_sp0", "mvp_sp05", "mvp_sp1", "mt_sp0", "mt_sp05", "mt_sp1") {
  $ckpt = Get-ChildItem "data/models/checkpoints/$run/best-*.ckpt" |
    Sort-Object { [double][regex]::Match($_.Name, 'qwk(-?\d+\.\d+)').Groups[1].Value } |
    Select-Object -Last 1
  python -m ml.training.export --checkpoint $ckpt.FullName `
    --out-onnx "data/models/candidates/$run.onnx" --out-torchscript "data/models/candidates/$run.pt" --no-activate
  python -m ml.eval.evaluate --split val --no-robustness --model-path "data/models/candidates/$run.onnx" --out-dir "test-artifacts/eval/$run-val"
}

# Winner: full test evaluation, promote it, and pin it as the regression baseline
python -m ml.eval.evaluate --split test --model-path data/models/candidates/<winner>.onnx --out-dir test-artifacts/eval/<winner>
python -m ml.training.export --checkpoint <winner checkpoint> --offsets-json data/models/candidates/<winner>.json
python -m ml.eval.evaluate --split test --save-baseline
```

Each run's CSV log (`data/models/logs/<run>/version_*/metrics.csv`) has
`val/qwk`, `val/mae` and `test/qwk` per epoch if you want to plot the runs.

## Evaluate

```powershell
python -m ml.eval.evaluate --split test                 # full report + charts in test-artifacts/eval/
python -m ml.eval.evaluate --split test --tta           # average over hflip + dim/bright copies
python -m ml.eval.evaluate --split test --n-boot 2000   # tighter CI estimates (default 1000; 0 = off)
```

The report includes 95% bootstrap CIs for MAE, QWK, exact and bias, plus a
per-grade table with a Wilson CI on recall and a bootstrap CI on MAE. Low
grades with a handful of test cards show very wide intervals; don't read much
into them. `--tta` only affects the evaluation; the API default is unchanged
(`GradeInferenceService(tta=True)` turns it on in code).

**Raw (unslabbed) cards.** Every training photo is through PSA plastic. Before
trusting the model on raw cards, build a separate labelled set (e.g. from eBay
"ungraded" listings with a known later grade, or your own photos) and evaluate
it with a CSV manifest:

```csv
front,back,grade,id
raw/charizard_front.jpg,raw/charizard_back.jpg,8,charizard-1
```

```powershell
python -m ml.eval.evaluate --manifest data/raw_eval/manifest.csv --out-dir test-artifacts/eval/raw
```

Relative paths resolve against the CSV's folder; `id` is optional.

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
