# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Layout

This top-level folder is a workspace, not a git repo. All source is in `PokeScan/`; the top-level `data/` holds an older copy of harvested images + SQLite. `PokeScan/README.md` is the full setup/run guide and `PokeScan/HANDOFF.md` records the state of the PSA data harvest (resume points per token).

PokeScan predicts a PSA grade (1–10) for a Pokémon card from front + back photos, with calibrated confidence and centering/corners/edges/surface sub-grades.

## Commands

Run all Python commands from `PokeScan/` with the venv active (`.\.venv\Scripts\Activate.ps1`). Modules are invoked as `python -m ...` because `backend` and `ml` import each other as top-level packages (`pip install -e ".[dev]"`).

```powershell
# API (needs data/models/grade_model.onnx, else /api/grade returns 503)
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (PokeScan/frontend; reads NEXT_PUBLIC_API_BASE from .env.local)
npm run dev        # also: npm run build, npm run lint
npm run preview    # Cloudflare runtime via OpenNext; npm run deploy to ship

# Lint / tests (pytest testpaths are backend/tests and ml/tests; none exist yet)
ruff check .
pytest path/to/test_file.py::test_name

# Data collection
python -m ml.data.ebay_collect --grades 6 7 8 9 10 --per-grade 500
python -m ml.data.collect --start-cert 90000000 --count 90          # single PSA token
python -m ml.data.collect_all --starts <6 comma-separated certs> --count 5000 --daily-limit 200 --daily-limit-token0 1000

# Training pipeline (order matters)
python -m ml.data.split                 # assigns certs.split = train/val/test (stratified by grade)
python -m ml.data.precrop               # dewarp to data/crops/; rerun after new harvests (--force to rebuild)
python -m ml.data.precompute_factors    # only for multitask: CV pseudo-labels into certs table
python -m ml.training.train --config ml/training/configs/mvp.yaml   # or multitask.yaml / mvp_8gb.yaml / smoke.yaml
python -m ml.training.export --checkpoint data/models/checkpoints/best-XX.ckpt
```

`smoke.yaml` is the fast end-to-end check (train → checkpoint → export → serve). Restart the API after exporting — the model is loaded once at startup.

## Architecture

**Inference path** (`backend/app/`): `api/grade.py` → `ml/inference.py` (`GradeInferenceService` singleton wrapping one ONNX Runtime session) → `ml/preprocess.py::detect_and_crop` (classical-CV quad detection + perspective warp to 600×840, with a padded fallback when no quad is found). The ONNX graph takes inputs `front`/`back` and outputs `corn_logits` (9 values); inference turns them into cumulative probs, hard rank `1 + sum(p>0.5)`, soft grade `1 + sum(p)`, and per-class confidence.

**Sub-grades are classical CV, not the network**: `ml/centering.py` (front only) and `ml/condition.py` (corners/edges/surface; worse side wins). The endpoint computes them independently and swallows their failures into `notes`, so a CV failure never fails the request. `ml/defects.py` (YOLO) and `ml/multitask.py` are Phase C hooks; `defects.py` is inert until weights exist.

**Model** (`backend/app/ml/model.py`): shared-weight timm ConvNeXt backbone over front and back, features blended with `front_weight`, single linear CORN head. The training code in `ml/` imports this model from `backend`, so the backend package is the source of truth for the architecture. Multitask training adds corners/edges/surface heads supervised by pseudo-labels from `condition.py`, but the served API still uses classical CV for those sub-grades.

**Export** (`ml/training/export.py`) fits temperature scaling on val (`calibrate.py`) and bakes it into the ONNX graph, writing `grade_model.onnx` plus a `grade_model.json` sidecar that `/api/health` returns as `metadata`.

**Data** (`ml/data/`): one SQLite `certs` table (`db.py`) is the hub — collectors upsert into it (`cert_number` is the PSA number or `ebay:<itemId>`, `source` is `psa`|`ebay`), `split.py` writes `split`, `precrop.py` writes cropped paths, `precompute_factors.py` writes `*_subgrade` columns, and `dataset.py` reads it all. `db.py` migrates columns on connect, so add new columns there. PSA API quota is tight (~100–200 calls/day/token, ~2 calls per Pokémon hit); `psa_client.py` tracks quota and honors `Retry-After`. eBay is the bulk source; PSA is the high-fidelity val/test source.

**Frontend** (`frontend/`): Next.js App Router + React Query + Tailwind v4, deployed to Cloudflare Workers via OpenNext. `src/app/api-client.ts` mirrors the `GradeResponse` pydantic schema — keep the two in sync.

## Gotchas

- `Settings` (`backend/app/core/config.py`) reads `PokeScan/.env`, but relative paths there (`./data`, `../data`) resolve against the current working directory — always run from `PokeScan/`. PSA tokens are `PSA_API_TOKEN0..N` (indexed, for rotation) or legacy `PSA_API_TOKEN`.
- Training augmentations are deliberately conservative: never add anything that changes apparent condition (blur, scratches, edge crops, heavy rotation).
- Use `num_workers: 0` on Windows if DataLoader workers misbehave (see `smoke.yaml`).
- Don't put data under `frontend/data/` (gitignored; an easy mistake after `cd frontend`).
