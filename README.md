# PokeScan

Predict the PSA grade (1–10) a Pokémon card will receive from front and back photos, plus a calibrated confidence score and per-factor sub-grades (centering, corners, edges, surface).

This project is independent and not affiliated with PSA or eBay. Respect each platform's API terms.

## What it does

1. **Detect & dewarp** — normalises the upload like the training photos (upright per its EXIF orientation, at most 1024 px on the long edge), finds the card in the photo (including slab shots) and warps it to a 600×840 canvas. If no card outline is found in a landscape photo, it retries once with the photo turned 90°, so a sideways shot is still found (possibly upside down). Portrait photos aren't retried, since the turned search finds false outlines on them, so a sideways card inside a portrait photo isn't found. Failing that, the whole photo is padded onto the canvas and the response notes it.
2. **Region features** — a frozen DINOv2 ViT-B (self-supervised, not fine-tuned) summarises each face as 11 vectors: CLS, mean, 4 corners, 4 edge strips, centre.
3. **Overall grade** — linear proportional-odds (ordinal) probes over the corner, edge, centre and global features of both faces, stacked into one ordinal model.
4. **Sub-grades**:
   - **Corners / edges / surface** — the soft grade each region probe gives on its own. They are learned from overall PSA grades (no sub-grade labels exist), so read them as "what this region suggests", not PSA's own sub-grade.
   - **Centering** — classical CV: border widths from colour runs, mapped to PSA tolerances (front 60/40 for a 10; back 75/25). Reports nothing for a face whose borders it can't find reliably rather than guessing.

## Architecture

```
Front/back image
  -> card detector (classical CV)
  -> DINOv2-B region features (ONNX)       -> 2 x 11 x 768
  -> corner / edge / surface / global probes -> region soft grades -> sub-grades
  -> meta ordinal probe                     -> {grade, confidence, rank_probs}
  -> centering CV (both faces)              -> centering subgrade
```

Why not the end-to-end CNN any more: with ~2.6k labelled certs (87% grade 9–10),
fine-tuning ConvNeXt end to end collapsed towards a constant grade and scored
below "always predict 10". Frozen features + linear probes can't collapse that
way, refit in seconds, and the per-region probes give sub-grades that actually
track the grade. The API serves only the probe model; the legacy CNN
(`grade_model.onnx`) loads only with `ALLOW_LEGACY_MODEL=1` or an explicit
`--model-path`. See `docs/adr/0001-frozen-features-linear-probes.md`.

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

> **Note:** `.venv` is currently near-empty; the working environment is the global pyenv Python 3.11.6 (with `pokescan` installed editable). Use plain `python` there until `intent/pinned-dev-environment` lands.

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

You need a trained probe model (`python -m ml.training.train_probe`) before `/grade` works: `data/models/grade_head.npz` (+ `grade_head.json`) and `card_features.onnx` (+ `card_features.json`). If none exists yet, skip to [Train a model](#train-a-model) or the health check will report `model_loaded: false`.

The API fails closed: if a probe file is missing or doesn't load (corrupt, or head, feature graph and the running code on different feature specs), `/api/grade` returns 503 and `/api/health` shows `model_loaded: false` with the reason in `metadata.probe_load_error`. It never silently serves the legacy CNN, which scores worse than always predicting 10. Set `ALLOW_LEGACY_MODEL=1` (env or `.env`) to fall back to the legacy model at `ACTIVE_MODEL_PATH` instead, e.g. to serve a `smoke.yaml` end-to-end run; the error stays in the health metadata. Artifacts trained before the feature-spec check (no `card_features.json`, no `feature_spec_hash` in `grade_head.json`) still load, with a WARNING.

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

If no model is loaded, `/grade` returns **503**; `/api/health` says why (`metadata.probe_load_error`).

## Collect training data

Both sources write to the same SQLite `certs` table with a `source` column (`psa` | `ebay`).

### eBay (bulk training)

```powershell
python -m ml.data.ebay_collect --grades 6 7 8 9 10 --per-grade 500
```

Low grades are scarce (the PSA test split has 47 certs graded 8 or below, 18 of
them 7 or below), so run a dedicated low-grade pass. `--per-grade` is a target; the
collector stops early when eBay runs out of matching listings. Then re-split
and re-crop so the new certs are usable:

```powershell
python -m ml.data.ebay_collect --grades 1 2 3 4 5 6 7 8 --per-grade 300
python -m ml.data.split          # sticky: only new certs are placed; eBay goes to train
python -m ml.data.precrop        # crops only the new certs (add --force to rebuild all)
python -m ml.data.precompute_factors   # only if you train multitask.yaml
```

`split.py` is **sticky** by default: a cert that already has a split keeps it,
so re-running after any harvest (eBay or PSA) never moves a val/test cert and
the eval numbers stay comparable. A new PSA cert of a printed card that already
has PSA certs joins that card's split (the most common one; ties go to test,
then val). A new card is placed by a hash of its card key (brand|subject|number,
`ml/data/groups.py`): 15% test, 15% val, the rest train. Only PSA certs go to
val/test. Every `source='ebay'` cert goes to train, except an eBay cert of a
card that has a PSA test cert, which gets `split='excluded'` (read by no loader)
so the test card isn't trained on. `--dry-run` prints the per-split grade
histogram, the number of test certs that share a card with train/val (the
"test straddle") and the rows that would change, and writes nothing.

Changing existing assignments needs `--reassign`, a destructive step that
resets every test number: only with the owner's approval and after a DB
backup. `--mode stratified --reassign` is the old per-grade shuffle
(`--ebay-split stratify` works only there). `--mode grouped --reassign` is the
one-time re-split (phase 2 of `intent/stable-grouped-split`, run on 2026-10-27,
`docs/adr/0002-card-grouped-test-split.md`): every PSA cert is re-placed by its
card's hash, so no printed card spans train/val/test. The current DB is in that
state; since then the default sticky mode keeps it.
The first reassign copies the old split into `split_legacy` (NULL for certs
that had none); later reassigns leave that column alone, so the old numbers stay
reproducible.

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

Both skip certs already collected with images, without any PSA call (the log says how many). Certs stored without images are fetched again. `--refresh` re-fetches every cert in the range; it spends quota and re-downloads the images.

## Train a model

### Probe model (default)

```powershell
python -m ml.data.split
python -m ml.data.precrop                 # one-time dewarp to data/crops/
python -m ml.training.train_probe         # features -> grouped CV -> fit -> test -> export
```

`train_probe` extracts DINOv2 region features for every cert whose front and
back outlines were found (cached in `data/models/cache/region_feats.npz`, so
refits after the first run take a minute), reports card-grouped 5-fold CV on
train+val (certs of the same printed card never straddle folds), fits the head
on train+val, evaluates once on test, and writes:

- `data/models/card_features.onnx` + `card_features.json` (the frozen feature
  extractor and its feature spec). Exported when missing, when the sidecar is
  missing or records another `feature_spec_hash`, or with `--force-onnx`.
  Written before the head, so a failed export never pairs a new head with an
  old graph.
- `data/models/grade_head.npz` + `grade_head.json` (weights, metrics,
  `feature_spec_hash`; served as `/api/health` metadata)
- `test-artifacts/eval/probe_report.json`

The cache is keyed on each crop file's size and mtime, so certs re-cropped by
`precrop --force` are re-extracted. Changing the feature definition
(`FEATURE_SPEC` in `backend/app/ml/features.py`: input size, patch, region
layout, or its `version` when `to_input`/`region_means` change) discards the
whole cache and re-exports the graph; the API refuses a head and graph whose
hashes differ.

Restart the API afterwards. The rest of this section is the legacy end-to-end
CNN, which the API serves only with `ALLOW_LEGACY_MODEL=1` (or through
`ml.eval.evaluate --model-path`).

### Legacy CNN: MVP (overall grade only)

```powershell
python -m ml.data.split
python -m ml.data.precrop
python -m ml.training.train --config ml/training/configs/mvp.yaml
python -m ml.training.export --checkpoint data/models/checkpoints/best-XX.ckpt
```

### Legacy CNN: multi-task (grade + corners/edges/surface)

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
`ACTIVE_MODEL_PATH`). Restart the API with `ALLOW_LEGACY_MODEL=1` to serve it
(only when no probe model loads); this is also how to serve the `smoke.yaml`
end-to-end check (train -> checkpoint -> export -> serve).

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
always-predict-10 on MAE (0.66 on the current test split).

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
(`GradeInferenceService(tta=True)` turns it on in code). TTA applies to the
legacy CNN only: with the probe model it raises an error rather than being
silently ignored, so pass `--model-path` to evaluate the legacy model with it.

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
| `GET` | `/api/health` | Service status, whether a model is loaded, its metadata (and `probe_load_error` when the probe didn't load) |
| `POST` | `/api/grade` | Multipart upload: `front` + `back` images (JPEG/PNG/WebP, max 12 MB each) |

## License / disclaimer

Not affiliated with PSA, The Pokémon Company, or eBay. Grades are predictions only — not official PSA assessments.
