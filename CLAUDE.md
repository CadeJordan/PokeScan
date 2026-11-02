# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Layout

This top-level folder is a workspace, not a git repo. All source is in `PokeScan/`; the top-level `data/` holds an older copy of harvested images + SQLite. `PokeScan/README.md` is the full setup/run guide and `PokeScan/HANDOFF.md` records the state of the PSA data harvest (resume points per token).

PokeScan predicts a PSA grade (1–10) for a Pokémon card from front + back photos, with calibrated confidence and centering/corners/edges/surface sub-grades.

## Commands

Run all Python commands from `PokeScan/`. Modules are invoked as `python -m ...` because `backend` and `ml` import each other as top-level packages (`pip install -e ".[dev]"`).

Interpreter: `.venv` is currently near-empty. The working environment is the global pyenv Python 3.11.6, where `pokescan` is installed editable (torch 2.5.1+cu121). Use plain `python` and don't activate `.venv` until `intent/pinned-dev-environment` lands. Don't install or upgrade packages in the global env without asking.

```powershell
# API (needs data/models/grade_head.npz + card_features.onnx (probe), else /api/grade returns 503 and /api/health says why)
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (PokeScan/frontend; reads NEXT_PUBLIC_API_BASE from .env.local)
npm run dev        # also: npm run build, npm run lint
npm run preview    # Cloudflare runtime via OpenNext; npm run deploy to ship

# Lint / tests (testpaths: backend/tests, ml/tests)
ruff check .
python -m pytest -m "not slow and not visual and not model"   # fast suite, ~4 min; baseline 1681 passed, 1 skipped, 2 xfailed
python -m pytest path/to/test_file.py::test_name
# markers: slow (sweeps, full test-split eval), model (needs data/models/grade_model.onnx, else skips),
#          visual (writes overlays to test-artifacts/overlays/)
```

Data collection and the training pipeline (run in order: split → precrop → `python -m ml.training.train_probe`; the legacy CNN path is precompute_factors → train → export): see `README.md`. Evaluate models with card-grouped CV (`train_probe` does): a printed card has several certs, so per-cert folds leak card identity. The test split itself is card-grouped since 2026-10-27 (`--mode grouped --reassign`, `docs/adr/0002-card-grouped-test-split.md`): no printed card spans train/val/test, and the pre-re-split assignment is kept in `split_legacy`.

`split` is sticky: `python -m ml.data.split` only places unassigned certs and never moves an assigned one (check with `--dry-run`, which opens the DB read-only). `--reassign` (`--mode stratified` or `grouped`) rewrites existing assignments and resets every test number: it's a destructive action that needs the owner's explicit approval and a DB backup first.

Test-set policy (decided 2026-10-27): the test split gives one final number per ADR. All model selection (hyperparameters, features, thresholds) uses card-grouped CV on train+val, never the test split.

`smoke.yaml` is the fast end-to-end check (train → checkpoint → export → serve). It exports a legacy model; `ALLOW_LEGACY_MODEL=1` serves it only when no probe model loads, so to serve a freshly exported legacy model next to a working probe, pass its `model_path` or move the probe files aside. Restart the API after exporting — the model is loaded once at startup.

## Architecture

**Inference path** (`backend/app/`): `api/grade.py` → `ml/inference.py` (`GradeInferenceService` singleton) → `ml/preprocess.py::detect_and_crop` (classical-CV quad detection + perspective warp to 600×840, with a padded fallback when no quad is found). On slab photos `_find_card_quad` steps inward from a holder outline to the nested card outline (`_is_holder`: a brighter/more colourful band, or, since 2026-10-31, a flat gasket ring around a textured card border); past the first step it enters a weak inner outline only on that flat-ring test, so inner art frames aren't taken for the card (`intent/tighter-slab-crops`). Any detection change can alter crops: re-run `precrop --force` + `train_probe`, and delete the face guard's served caches and retrain it. The service serves only the **probe model** (`data/models/grade_head.npz` + `card_features.onnx`) and fails closed: a missing file or a load error (corrupt, D mismatch, a `feature_spec_hash` in `grade_head.json` that differs from the running code's or from the `card_features.json` sidecar, a missing sidecar beside such a head, a `feature_spec_hash` key that is empty or not a string, a sidecar that can't be read, or a `ten_cutoff` in `grade_head.json` that isn't a finite number in [0.5, 1)) leaves it not ready (503), with `probe_load_error` in `/api/health` metadata. The **legacy** CORN `grade_model.onnx` loads only with an explicit `model_path` or, as a fallback, with `ALLOW_LEGACY_MODEL=1`. `/api/grade` refuses unusable input: an undecodable image, or bytes that aren't JPEG/PNG/WebP (`preprocess.UPLOAD_FORMATS`), is 400 (`preprocess.ImageDecodeError`, labelled per face in `predict`), and with the probe model `GradePrediction.rejection()` turns a face with no quad, or a pair the face guard reads as two of one side, into a 422. `predict` itself still grades those, so `ml.eval` covers them and reports `pipeline.n_would_reject`.

**Probe model** (default, `ml/features.py` + `ml/grade_head.py`): frozen DINOv2-B (`card_features.onnx`, input 602×434) → per face 11 region vectors (CLS, mean, 4 corners, 4 edge strips, centre) → four proportional-odds probes (corners / edges / surface / global, both faces) → meta proportional-odds probe over their soft grades. Scaler+PCA are folded into each probe's weight vector, so the head is plain numpy. Trained by `python -m ml.training.train_probe` (`ml/training/probe.py` holds the fitting; card-grouped CV, nested OOF for the meta probe). The hard grade (`grade_head.decode`, used by the API, `train_probe.metrics` and so `ml.eval`) is the distribution's median, except that a 10 needs P(10) > the head's `ten_cutoff` (in `grade_head.json`; absent means 0.5, the plain median). `train_probe` tunes it on CV seeds 0–4 (`docs/adr/0003-grade-decision-rule.md`), so it runs CV five times (~7 min). `features.FEATURE_SPEC` fingerprints the feature definition: bump its `version` whenever `to_input` or `region_means` changes, which invalidates the feature cache (also keyed on crop file size+mtime) and re-exports the graph. Proportional odds is monotone by construction, so the CORN `cummin` question below doesn't apply to it.

**Face guard** (`backend/app/ml/face_guard.py`, probe path only): three numpy logistic probes over one face's CLS + mean features (side: P(back); front/back upright: P(flipped 180°)), with per-check thresholds and an enabled list (a check is disabled when it misses its gate or no threshold in {0.9, 0.95, 0.99} meets clean-face FPR ≤ 0.2%; the shipped guards of 2026-10-28, 2026-10-29 and 2026-10-31 (retrained on the tighter-slab crops) have `back_upright` disabled, so only swaps and upside-down fronts are corrected). `_predict_probe` runs it on the extracted features: a pair whose front reads as a back and back as a front is swapped (features and crops), an upside-down face's crop is rotated and both faces re-extracted once (skipped when both faces read as the same side, and never done to a face with no quad: `decide` takes `found_quad`, which follows the swap), and `GradePrediction.notes` says so (appended to the response notes; schema unchanged). Artifacts: `data/models/face_guard.npz` + `face_guard.json` (enabled checks, CV metrics, `feature_spec_hash`), trained by `python -m ml.training.train_face_guard` (thresholds from card-grouped CV on train+val). Since 2026-10-29 it trains on served-path features by default: the DB photos through `inference.served_extractor_input` (the probe path's own decode/EXIF/cap/detection/`_jpeg_roundtrip`), the rotated set being the photo rotated first with `evaluate.perturb_image("rotate_180")`, cached in `data/models/cache/region_feats_served.npz` / `region_feats_served_rot180.npz` (not invalidated by preprocessing code changes: delete them after changing detection). `--features precrop` keeps the old source (`region_feats.npz`, read only, + `region_feats_rot180.npz`); `face_guard.json` records `feature_source`. Unlike the head it **fails open**: missing `face_guard.npz` means no guard; a missing/unreadable sidecar, a missing or mismatched `feature_spec_hash`, or a bad file (incl. a weight that isn't 1-D of width 2×D, a non-finite bias, a threshold outside (0.5, 1), or any error stat-ing it) drops the guard with a WARNING and `face_guard_load_error` in `/api/health` metadata, and the API stays ready; an error in the guard's decision on a request grades that request unguarded with a WARNING.

**Sub-grades**: with the probe model, corners/edges/surface are the region probes' soft grades (learned from overall grade, not PSA sub-grades); otherwise classical CV from `ml/condition.py`, whose factors were measured to have ~zero correlation with grade. `ml/centering.py` measures border widths on both faces from colour runs and returns None for a face it can't measure (most full-art/silver/Japanese fronts) rather than guessing; the API notes it. The endpoint swallows sub-grade failures into `notes`, so a CV failure never fails the request. `ml/defects.py` (YOLO) and `ml/multitask.py` are Phase C hooks for the legacy CNN.

**Legacy model** (`backend/app/ml/model.py`): shared-weight timm ConvNeXt over front and back, single linear CORN head; trained by `ml/training/train.py` + `export.py`. It collapsed towards a constant grade on this data (worse than always predicting 10); kept only as a fallback.

**Export** (legacy model only; `ml/training/export.py`) fits temperature scaling on val (`calibrate.py`) and bakes it into the ONNX graph, writing `grade_model.onnx` plus a `grade_model.json` sidecar that `/api/health` returns as `metadata`. The probe model has no post-hoc calibration and needs none: its out-of-fold NLL-optimal temperature is 1.00.

**Data** (`ml/data/`): one SQLite `certs` table (`db.py`) is the hub — collectors upsert into it (`cert_number` is the PSA number or `ebay:<itemId>`, `source` is `psa`|`ebay`), `split.py` writes `split`, `precrop.py` writes cropped paths, `precompute_factors.py` writes `*_subgrade` columns, and `dataset.py` reads it all. `db.py` migrates columns on connect, so add new columns there. PSA API quota is tight (~100–200 calls/day/token, ~2 calls per Pokémon hit); `psa_client.py` tracks quota and honors `Retry-After`. eBay is the bulk source; PSA is the high-fidelity val/test source.

**Frontend** (`frontend/`): Next.js App Router + React Query + Tailwind v4, deployed to Cloudflare Workers via OpenNext. `src/app/api-client.ts` mirrors the `GradeResponse` pydantic schema — keep the two in sync. It's Next 16, which breaks older APIs: read `frontend/AGENTS.md` (regenerated by `next dev`; keep it committed) and check `frontend/node_modules/next/dist/docs/` before using Next APIs.

## Gotchas

- `Settings` (`backend/app/core/config.py`) reads `PokeScan/.env`, but relative paths there (`./data`, `../data`) resolve against the current working directory — always run from `PokeScan/`. PSA tokens are `PSA_API_TOKEN0..N` (indexed, for rotation) or legacy `PSA_API_TOKEN`.
- Training augmentations are deliberately conservative: never add anything that changes apparent condition (blur, scratches, edge crops, heavy rotation).
- Use `num_workers: 0` on Windows if DataLoader workers misbehave (see `smoke.yaml`).
- Don't put data under `frontend/data/` (gitignored; an easy mistake after `cd frontend`).
- `data/` (images, crops, `pokescan.sqlite`, models) is the only copy of quota-limited harvest data. Never download, move, or delete images, or rewrite the DB, unless the task explicitly asks. Tests must use `tmp_path` and only *read* real data.
- Long jobs (`precompute_factors --force`, `precrop --force`, collectors) may already be running and writing the DB. Check before starting another DB-writing command, and don't run two at once.
- Card images exist in git history (commit `608c429`, ~271 MiB). Don't `git add` anything under `data/`. History rewriting is pending a decision (`intent/git-history-image-purge`).
- Known open question: `decode_corn_logits` uses `cummin(sigmoid)` while `corn_loss` trains conditional probabilities. See `intent/corn-decode-verification` before touching decoding, calibration, or the soft grade.

## Intent specs

Planned changes live in `intent/<spec-name>/intent.md` (intent, scope, out of scope, acceptance criteria, facts vs assumptions). Fixes too small for a spec are listed in `intent/smallfix.md`. Before working on one, read its spec. Check off acceptance criteria as they're met, and don't widen scope beyond "Out of scope" without asking.

## Agent skills

### Issue tracker

Issues live in GitHub Issues on CadeJordan/PokeScan, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the five default labels: needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one root `GLOSSARY.md` plus `docs/adr/`, both created when first needed. See `docs/agents/domain.md`.
