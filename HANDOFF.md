# PokeScan — Session Handoff

## 1. PROJECT GOALS

Build **PokeScan**: a system that predicts PSA grades (1–10) for Pokémon cards from photos, with calibrated confidence. The immediate work in this session was **harvesting training/validation data** from the **PSA Public API** using multiple bearer tokens (`PSA_API_TOKEN0`–`PSA_API_TOKEN5` in `.env`), saving cert metadata and front/back images into local SQLite + `data/images/`.

Longer-term pipeline (from README): eBay bulk collection → PSA high-fidelity sample → stratified split → train CORN ordinal model → export ONNX → FastAPI + Next.js UI.

---

## 2. CURRENT STATE

### PSA multi-token collection (`collect_all`)

The user ran two `collect_all` sessions with six tokens, each scanning **5,000 cert attempts** per token in non-overlapping 5k-wide windows.

**Yesterday (planned windows):**

```text
--starts 76002000,76007000,76012000,76017000,76022000,76027000
--count 5000 --daily-limit 200 --daily-limit-token0 1000
```

**Today’s run (continuation windows):**

```powershell
python -m ml.data.collect_all --starts 76032000,76037000,76042000,76047000,76052000,76057000 --count 5000 --daily-limit 200 --daily-limit-token0 1000
```

**Today’s results (all tokens quota-limited before finishing their 5k blocks):**

| Token | Started at | Stopped before | Progress | Notes |
|-------|------------|----------------|----------|-------|
| 0 | 76,032,000 | **76,032,860** | 849/5000 | Local 1000-call cap hit; 838 hits, 148 Pokémon |
| 1 | 76,037,000 | **76,037,104** | 88/5000 | PSA daily quota (`Retry-After` ~86k s) |
| 2 | 76,042,000 | **76,042,112** | 102/5000 | PSA daily quota |
| 3 | 76,047,000 | **76,047,073** | 58/5000 | PSA daily quota |
| 4 | 76,052,000 | **76,052,064** | 48/5000 | PSA daily quota |
| 5 | 76,057,000 | **76,057,097** | 81/5000 | PSA daily quota |

**DB state after today’s run:**

- Total certs: **6,352**
- Pokémon rows: **2,213**
- Pokémon with images: **2,208**
- Grade histogram: `{1:1, 2:2, 3:7, 4:7, 5:25, 6:31, 7:50, 8:158, 9:597, 10:1330}`

**Earlier DB snapshot (before today):** numeric PSA certs spanned **70,000,001 → 89,000,797** (5,152 rows); highest Pokémon-with-images cert in **76M** band was **76,001,804**.

---

## 3. WHAT WORKED

- **`python -m ml.data.collect_all`** — correct entry point for running tokens **0–5 in series**, each with its own `--starts` entry and per-token daily limits.
- **Resume strategy** — use the log line `local quota exhausted before reaching <N>` as the next `--start-cert` for that token (do **not** advance to the next 5k block until the current block is finished).
- **Quota flags:**
  - `--daily-limit 200` for tokens 1–5
  - `--daily-limit-token0 1000` for token 0
- **Non-overlapping windows** — space starts by **5,000** when `--count 5000` (e.g. 76,032,000 / 76,037,000 / …).
- **DB is source of truth** for what was stored; `upsert_cert` means re-scanning an already-seen cert is safe but wastes API quota.

---

## 4. WHAT FAILED

- **Tokens 1–5 exhausted PSA’s real daily quota** almost immediately (`Retry-After` ~86,000s). Local `--daily-limit 200` did not matter once upstream quota was gone.
- **Token 0 hit local cap (1000 calls)** after only ~849 cert attempts (~17% of its 5k window). Pokémon hits use ~2 API calls each (details + images), so call budget burns faster than cert count.
- **ReadTimeouts** occurred during token 0’s run (retried automatically; run continued).
- **No dedicated “resume” subcommand** — must manually derive `--starts` from logs or DB.
- **Agent-side SQLite query** failed on PowerShell due to quoting/`&&` issues; ad-hoc Python script worked instead.

---

## 5. NEXT 3 STEPS

1. **After PSA quota resets** (tokens 1–5 showed ~24h `Retry-After`), resume the **same six unfinished windows** from today’s stop points:

   ```powershell
   cd PokeScan
   python -m ml.data.collect_all --starts 76032860,76037104,76042112,76047073,76052064,76057097 --count 5000 --daily-limit 200 --daily-limit-token0 1000
   ```

2. **Verify token 0 quota** before running — if token 0 also reset, it can finish its remaining ~4,151 certs in the 76,032,000–76,036,999 block; if not, consider `--tokens 1,2,3,4,5` with the five corresponding starts to avoid burning token 0 first.

3. **After all six windows complete** (through ~76,061,999), advance to the next non-overlapping slice:

   ```text
   --starts 76062000,76067000,76072000,76077000,76082000,76087000
   ```

   (Same `--count 5000 --daily-limit 200 --daily-limit-token0 1000`.)

---

## 6. OPEN QUESTIONS

- **Has token 0’s PSA upstream quota also reset**, or only the local 1000-call counter? Logs suggest token 0 stopped on local cap, not `Retry-After`.
- **Cert range strategy:** DB already contains certs up to **~89M** from earlier runs. Should collection continue filling **76M** (current plan) or shift to under-sampled ranges / higher bands?
- **Quota math:** With ~2 calls per Pokémon hit, is `--daily-limit-token0 1000` the right cap, or should `--count` for token 0 be lowered to match realistic daily yield?
- **Partial token 2 run** yielded 94 hits but **0 Pokémon** — is that range sparse, or worth investigating?
- **Whether yesterday’s windows** (76,002,000–76,031,999) fully completed before today’s run started.

---

## 7. REFERENCED FILES

| Path | Relevance |
|------|-----------|
| `PokeScan/ml/data/collect_all.py` | Multi-token fan-out orchestrator; `--starts`, `--tokens`, `--daily-limit-token0` |
| `PokeScan/ml/data/collect.py` | Single-token harvester; quota-aware scan loop; resume semantics |
| `PokeScan/ml/data/psa_client.py` | PSA API client, daily quota tracking, `Retry-After` / quota-exhausted handling |
| `PokeScan/backend/app/core/config.py` | Loads `PSA_API_TOKEN0..N` from `.env`; `get_psa_token()` |
| `PokeScan/ml/data/db.py` | SQLite schema, `upsert_cert`, cert queries |
| `PokeScan/data/pokescan.sqlite` | Harvested cert metadata (6,352 rows after latest run) |
| `PokeScan/data/images/` | Downloaded front/back JPEGs (e.g. `76000046_front.jpg`) |
| `PokeScan/.env` | PSA API tokens 0–5 (user-managed; not committed) |
| `PokeScan/README.md` | Project setup, workflow, data-source docs |
| `c:\Users\User\.cursor\projects\c-Users-User-dev-poke-scan\terminals\1.txt` | Terminal log of today’s `collect_all` run |

---

*Generated from session focused on PSA cert harvesting, multi-token rotation, and resume planning.*
