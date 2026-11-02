# Full-resolution pilot: does more resolution separate 9 from 10?

## Intent

Measure, before committing ~20 GB and a re-extraction, whether higher-resolution corner and edge features improve how well the probe model tells PSA 9 from PSA 10. This is [[cv-ml-improvement-plan]] item 12 / Q5 ("a 200-cert full-resolution pilot first, after checking whether image downloads cost PSA API quota", owner decision 2026-10-27).

This is a measurement spec. It changes nothing that's served and ends in a go / no-go for a full download.

## Why

- **9 vs 10 is the weak spot.** Test split (guard_noquad, 2026-10-30): grade-9 recall 36.5%, and 9s are mostly predicted 10. AUC 10 vs 9 is 0.73–0.74 in CV and on test, and 0.64 on unseen cards (ADR 0001 / 0002). The difference between a 9 and a 10 is small corner, edge and surface flaws.
- **Resolution is capped twice.**
  - Ingest stores photos at a 1024 px long edge (`ml/data/image_io.py`, `IMAGE_MAX_LONG_EDGE`). For example, `76000010_front.jpg` is 609×1024, so the slabbed card is ~660 px tall.
  - The features are then computed on a 602×434 resize of the 600×840 crop (`features.py`). A corner region is 5×5 patches, ~70 px of a ~6.3 cm card side.
  - The PSA originals are 3–4 MB (`image_io.py` docstring), and their URLs are in the DB.
- **Nobody has measured it.** ADR 0001 attributes the 9/10 ceiling to resolution, but that is a hypothesis.

## Facts checked for this spec (2026-10-30)

- **The downloads don't use PSA API quota.** `PSAClient.download_image_bytes` (`ml/data/psa_client.py:248`) does a plain `httpx` GET on the CDN URL with a fresh client and no auth header, and it doesn't call `_bump_quota`. The URLs are on `d1htnxwo4o0jhw.cloudfront.net`, not the API host. Step 0 still checks this on 5 images.
- **Available originals** (PSA certs with both URLs and a precrop):
  - train: 9s 502, 10s 1,137, ≤8 241
  - val: 9s 118, 10s 205, ≤8 51
  - test: 9s 115, 10s 213, ≤8 47
- **Disk:** C: has 537 GB free.
- **Gitignore:** `.gitignore` covers `data/images/`, `data/crops/`, `data/models/` and `data/backups/`, but no new folder. A new download folder needs its own line.
- **Coordinates:** `CardCrop` already carries `source_bgr` and `scale` (`preprocess.py`), so a quad found at 1024 px can be mapped back onto a larger source.

## Design

Three arms, compared on the same certs with the same card-grouped folds. Only the corner and edge region features differ between arms. The CLS, mean and centre features stay as today.

| Arm | Source photo | Corner / edge features | Needs download |
|---|---|---|---|
| A, baseline | stored 1024 px | today's: region means of the 602×434 pass | no |
| B, feature resolution only | stored 1024 px | high-res tiles: the region cropped from a larger warp (e.g. 1200×1680) of the stored photo, each tile run through the same frozen DINOv2 on its own | no |
| C, full resolution | the PSA original | the same high-res tiles, warped from the original | yes |

- **B separates the two caps.** If B ≈ C > A, the gain comes from feature resolution, and no download is needed. If C > B ≈ A, the source pixels matter. If A ≈ B ≈ C, resolution isn't the lever.
- **B is cheap.** It needs no download, so it also runs on all 2,241 train+val certs, not just the pilot set.
- **Tile definition** (fixed before looking at results):
  - Corners are a square tile at each warped corner, covering the same card area as today's 5×5 patch block.
  - Edges are the same 3-patch-deep strips, cut into tiles along their length.
  - Tiles run at DINOv2's native 14 px patch at a fixed tile size (e.g. 224 or 448 px). A tile's feature is its mean patch token, and a region's feature is the mean over its tiles. The final tile size is set in step 1 from the measured card size in the originals, and recorded before CV runs.
- **Detection:** reuse the stored precrop's quad, which was found on the 1024 px photo, scaled to the original's size. Re-detecting on the original only if the scaled quad misses would give C a different crop than A and B, so cases like that are dropped and counted instead.
- **Evaluation:**
  - Card-grouped 5-fold CV on the pilot set (`probe.cross_validate`, same seed and folds for every arm). The whole head is refit per arm.
  - Primary metric: OOF AUC 10 vs 9. Secondary: MAE, QWK, 9-recall, and AUC 10 vs 9 of the corners probe and the edges probe on their own.
  - Paired bootstrap over cards (1,000 resamples) for each arm difference.
- **The test split isn't used** (test-set policy, CLAUDE.md). The pilot's certs are from train+val only.

## Tile definition (fixed 2026-10-31, before any CV run)

Written after step 0 and before any features were extracted or any CV was run. Code: `ml/eval/resolution_pilot.py` (`region_cells`, `region_boxes`, `TileSpec`, `region_tiles`, `tile_to_source`).

- **Card area per region = today's patch blocks, exactly.** Today a face is warped to 600×840 and resized to 434×602, a 31×43 grid of 14 px patches. `region_means` takes, in patch rows × columns (half-open):
  - corners: rows 0–5 / 38–43 × columns 0–5 / 26–31 (5×5 patches: 5/31 of the card's width × 5/43 of its height, ≈10.2 × 10.2 mm);
  - top / bottom strips: rows 0–3 / 40–43 × columns 5–26 (3 patches deep, 21 long);
  - left / right strips: rows 5–38 × columns 0–3 / 28–31 (3 deep, 33 long).

  Each block becomes a rectangle in normalised card coordinates (u = column / 31, v = row / 43). A test checks that these blocks are the ones `region_means` averages.
- **Tile size T** (the corner tile's side, 224 or 448) sets the scale: T / 70 tile px per px of today's 434×602 input (3.2 for 224, 6.4 for 448).
  - **Corners:** one T×T tile per corner.
  - **Edges:** each strip is cut along its length into n = round(length at that scale / T) equal pieces (4 along top and bottom, 7 along the sides, for either T). Each piece is resampled to T along the strip × D deep, where D = 14 × round(3 × T / 70): 140 px for T = 224, 266 px for T = 448. The strip's full 3-patch depth is kept, and every tile is a whole number of 14 px patches. The pieces partition the strip with no gaps or overlaps. The resampling scales them by 0.95–1.06 along the strip and 1.04 (T = 224) or 0.99 (T = 448) across it.
  - That makes 26 tiles per face and 52 per cert.
- **Sampling.** Each tile is warped straight from the source photo through the precrop quad's homography. This is one resampling, equivalent to cropping a (434 × T/70) × (602 × T/70) warp of the card. It uses `cv2.INTER_LINEAR`, as the precrop warp does, and the same pixel-centre convention, so u = 0 is the left edge of crop pixel 0. When the source is denser than that canvas (card height in px > 602 × T/70), the whole photo is first shrunk with `INTER_AREA` to the canvas density, so the warp never aliases. Arm B upsamples the stored photo, whose card is ~660 px tall against an 1,926 px canvas for T = 224.
- **Features.** Each tile goes on its own through the same frozen DINOv2 (`vit_base_patch14_reg4_dinov2.lvd142m`, timm `dynamic_img_size=True`, so its position embedding is resampled to the tile's grid) with the same ImageNet normalisation as `features.to_input`. A tile's feature is its mean patch token (registers and CLS excluded). A region's feature is the mean over its tiles. CLS, mean and centre stay as arm A's in every arm.
- **Quad.** The precrop's quad isn't stored, so it is re-detected with today's `detect_and_crop` on the stored 1024 px photo (the precrop's own input). The re-warp must match the stored precrop JPEG: a mean absolute difference over 8 grey levels drops the cert as `precrop_mismatch`. A face with no quad or a 90° retry is dropped too. For arm C the quad is scaled per axis to the original's EXIF-upright size (`scale_quad`: (x + 0.5) × W/w − 0.5).
- **Original check (arm C).** The original's size must round to the stored size at a 1024 px long edge (within 1 px), and the original shrunk to the stored size must match the stored photo within 8 grey levels (mean absolute difference). Otherwise the cert is dropped as `size_mismatch`, `original_mismatch` or `not_larger`. Dropped certs leave every arm, so A, B and C stay on the same certs.
- **Choosing T (rule fixed before measuring the full set):** whichever of 224 / 448 is closer, on a log scale, to the median corner-block size (card height × 5/43) over all downloaded pilot originals. The crossover is a 317 px corner block, i.e. a 2,724 px card.
- **T = 224 (measured 2026-10-31 on all 1,200 downloaded originals, before any tile extraction or CV;** `python -m ml.eval.resolution_pilot sizes` → `data/images_full/sizes.json`):
  - Long edges: 2400 px for 994 faces, 3000 px for 182, ~2170 px for 12 and ~4740 px for 12. Step 0's sample of 5 happened to include two of the rare 4,740 px ones.
  - The card is 1,549 px tall at the median (p5–p95 1,518–1,969), 2.35× the stored ~660 px.
  - A corner block is 180 px at the median (p5–p95 176–229), so a 224 px corner tile holds all of a typical original's detail (≈1.0–1.25× upsampling). In arm B the same tile is a ~2.9× upsampling of the stored photo's ~77 px block.
  - Final geometry: corner tiles 224×224 (16×16 patches); edge tiles 224×140 along top/bottom (4 per strip) and 140×224 along the sides (7 per strip); 26 tiles per face. The whole card at tile scale is a 1,389 × 1,926 px canvas. Of the originals, only those taller than that (the ~4,740 px ones) get the `INTER_AREA` pre-shrink.
- **Evaluation, fixed in advance:**
  - Arm A comes from `data/models/cache/region_feats.npz`, read only.
  - `probe.cross_validate` runs with 5 folds and seed 0 per arm, on the same certs.
  - Paired bootstrap over card groups: 1,000 resamples, seed 0, 95% percentile CI.
  - **"B already gets most of that gain"** means B − A ≥ 0.5 × (C − A) on AUC 10 vs 9 (whole head), applied only when the go condition on C − A holds.

## Steps

0. **Quota check.** Download 5 originals into the pilot folder. Confirm that no request carries a token. If PSA's API responses expose remaining quota (rate-limit headers), compare it with one API call before and one after on an idle token. That costs 2 calls, so it's only done if the headers exist. Otherwise the code inspection above stands. Record image sizes and the card's height in pixels.
1. **Download the pilot set** (size per owner decision 1): train+val certs, stratified by grade, cards spread over many `card_groups`. Write the files and a `manifest.csv` (cert, URL, bytes, sha256, size in pixels) into the pilot folder. Rate-limit to a few requests per second. Never touch `data/images/` or the DB.
2. **Extract** the A, B and C features on the GPU into caches inside the pilot folder (B also on all train+val).
3. **CV and report:** `test-artifacts/eval/resolution_pilot/report.json`, plus a results section here.
4. **Go / no-go.** The default rule, open to owner change:
   - **Go** for a full download if C − A AUC 10 vs 9 ≥ +0.03 and the paired bootstrap's 95% CI excludes 0.
   - **Go with no download** if B already gets most of that gain.
   - **Stop** otherwise.

## Out of scope

- Changing the served model, `FEATURE_SPEC`, the serve path's 1024 px cap, or anything in `data/models/`.
- The full ~20 GB download and a new production feature definition. That's a follow-up spec if the result is "go".
- New DB columns. The manifest is a CSV in the pilot folder.
- Phone uploads at higher resolution. The serve path caps them at 1024 px too, so serving would need its own change after a "go".

## Acceptance criteria

- [x] Step 0 shows the downloads send no token and don't change PSA API quota (headers before and after recorded, where PSA exposes them), and it records the originals' pixel size. *(No token sent. Neither the CDN nor, per the repo, the API exposes quota headers, so 0 API calls; sizes in Results.)*
- [x] The pilot folder is gitignored before the first download, and `git status` stays clean.
- [x] `data/images/`, `data/crops/`, `data/models/` and `data/pokescan.sqlite` are unchanged (sha256 or mtime of the DB, file counts of the folders before and after). *(Identical before the first download and after `evaluate`; see Integrity.)*
- [x] The manifest lists every downloaded file with its sha256, and failed downloads are listed with a reason. *(1,200 ok, 0 failed.)*
- [x] The tile definition and size are written into this spec before any CV runs.
- [x] The report gives, for A, B and C on the same certs and folds: AUC 10 vs 9 (whole head, corners probe, edges probe), MAE, QWK and 9-recall, plus the paired bootstrap CIs for B − A, C − A and C − B. B is also reported on all train+val.
- [x] No test-split cert is downloaded or evaluated. *(The selection and downloads are train+val only; `evaluate` reads only train+val rows.)*
- [x] New code (download script, tile extractor) has fast tests on synthetic images in `tmp_path`, and the fast suite stays green.
- [x] Results and the go / no-go call are in this spec, and the [[cv-ml-improvement-plan]] item 12 links here. *(Call: stop.)*

## Owner questions

1. **Pilot size.** 200 certs (as approved) can't measure a +0.03 AUC change. With ~100 9s against ~100 10s, the AUC's standard error is ~0.035.
   - Recommended: **600** (300 9s, 300 10s): ~4 GB, ~30 min of downloading at a polite rate. The paired difference is then measurable at about ±0.03.
   - Alternative: 200 as approved, as a feasibility run only (sizes, timings, a first look), with a larger pilot decided after.
2. **Where the originals go.**
   - Recommended: a new `data/images_full/` with its own `.gitignore` line, written only by the pilot script and never touching existing folders.
   - Alternative: outside the repo, e.g. `C:\Users\User\dev\poke-scan\data_full\`.
3. **Run arm B on all train+val too?** It costs GPU minutes and no download, and it answers "is feature resolution alone enough?" with 2,241 certs instead of 600. Recommended: yes.
4. **Compute and real runs:** one GPU extraction pass for the pilot (plus B on train+val), with caches written in the pilot folder and nothing written in `data/models/`. Recommended: approve.

## Owner decisions (2026-10-31)

1. **600 certs:** 300 PSA 9s and 300 PSA 10s from train+val, spread over card groups.
2. **`data/images_full/`**, with its own `.gitignore` line added before the first download, and a `manifest.csv` instead of DB columns.
3. **Arm B also runs on all 2,241 train+val certs.**
4. **Real runs approved:** the CDN downloads and one GPU extraction pass, with caches in `data/images_full/` and the report in `test-artifacts/eval/resolution_pilot/`. Nothing goes to `data/models/` or the DB.

## Implementation

- **`ml/eval/resolution_pilot.py`** is a typer CLI. Every step resumes, and everything it writes is under `data/images_full/` (gitignored) except the report. `check_output_dir` refuses any path inside `data/` except under `data/images_full/` (resolved first), and every output path goes through it, `evaluate --out` included.
  - `integrity --out F`: sha256 + size + mtime of `pokescan.sqlite`, and file counts + bytes of `data/images`, `data/crops`, `data/models`.
  - `select`: `pilot_certs.csv`. The DB is opened read-only (`mode=ro`), the SQL and Python both filter on `split IN ('train','val')`, grades are 9 and 10, both `*_url` must be set, and the precrop must be usable (as in `load_rows`). Card groups are visited round-robin in a seeded order, one cert per group before any repeats.
  - `step0`: the first 5 pilot originals, with the request headers sent, the response headers and the card size in px (`step0.json`).
  - `download`: both faces of each pilot cert, at most 2 requests/s, through a bare `httpx.Client` (no token, no headers; `fetch` builds each request and raises before sending it if it carries an `Authorization`, `Cookie` or `X-API-Key` header). It retries 429 / 5xx / transport errors 3 times. Files are written atomically. `manifest.csv` has cert, face, url, file, bytes, sha256, width, height, EXIF orientation, status, reason and fetched_at. An ok row whose file still hashes the same is skipped; a file without a row is re-hashed and recorded; failures are retried on the next run.
  - `quads`: re-detects the precrop quad on every train+val stored photo and compares the re-warp with the stored precrop (`cache/quads.json`).
  - `extract --arm b|c`: tile features (`cache/tiles_{arm}_{T}.npz`, keyed by cert + source paths + file signatures + tile-spec hash). B covers all train+val certs, C the pilot originals, after the original check (`cache/original_checks.json`).
  - `evaluate`: arm A from `data/models/cache/region_feats.npz` through `train_probe.extract_features(..., save=False)`, so nothing is written, plus a from-scratch re-extraction of 8 pilot certs to check that A reproduces. B and C tiles are used only when their cache key matches the current source files, and C only when both originals passed the check on those same files; a pilot cert dropped from any arm (quad, download, original check, missing or stale tiles) leaves them all and is listed in `pilot_dropped`. It then runs `probe.cross_validate` (5 folds, seed 0) per arm on the pilot, A and B on all train+val, and the paired card-group bootstrap, and writes `test-artifacts/eval/resolution_pilot/report.json`.
  - `redetect-sample`: the "scaled quad lands on the card" check on 20 originals.
- **Tests:** `ml/tests/test_resolution_pilot.py`, 46 fast tests on synthetic images in `tmp_path`, with HTTP mocked by `httpx.MockTransport`. They cover:
  - tile blocks equal to `region_means`' blocks, and tiles sampling exactly their card area;
  - edge tiles partitioning their strip;
  - quad scaling 1024 px → original, including detection on an ingest-like LANCZOS shrink of a synthetic 1430×2400 photo;
  - the manifest, rate limiting, retry, resume / recover / re-fetch, and the refusal to send a token;
  - test-split exclusion in SQL, the selection and the plan, with a read-only DB check;
  - the bootstrap, AUC and the decision rule.
- **Tile model:** timm `dynamic_img_size=True` on the same weights. On a full 602×434 input its mean patch token matches the production region model's `mean` region (cosine 1.0, max abs diff 1.3e-6).
- **Fixes after re-test (2026-10-31).** Three defects from the test pass; none changed a result.
  - `check_output_dir` now refuses every path inside `data/` except under `data/images_full/`, after resolving it (`..`, relative paths, case on Windows), so `integrity --out data/pokescan.sqlite` or a file in `data/backups/` is refused. `evaluate --out` goes through it before any work.
  - `evaluate` uses B / C tiles only when their cache key matches the current source files (path + size + mtime) and C only when both originals passed the check on those files. Certs with a quad, download or original-check problem, or with missing or stale tiles, leave every arm and are listed in `pilot_dropped` (the report also counts `stale_tiles_skipped`). Before, a stale C entry for a since-replaced original that failed the check was still evaluated.
  - `fetch` builds each request (`client.build_request`), audits its headers and only then sends it, so a request with `Authorization`, `Cookie` or `X-API-Key` is never sent. Retries are unchanged.
  - Re-running `evaluate` on a scratch copy of the pilot folder gives the same `pilot`, `trainval` and `decision` sections as `report.json` (to 1e-12; 600 pilot certs, 2,241 train+val, 0 dropped, 0 stale tiles; call: stop). The published report wasn't rewritten.
  - Tests: 52 in `test_resolution_pilot.py` and 17 in `test_retest_resolution_pilot.py` (the two strict xfails now pass; the end-to-end fixture writes real tile keys, originals and a manifest). Fast suite 1542 passed, 1 skipped, 2 xfailed; `ruff check .` clean.
- **Fixes after the second re-test (2026-10-31).** Two gaps in the fixes above, neither reachable through the pilot's own use; no result changed.
  - The guard compared resolved path strings, so `\?\` verbatim and UNC spellings of `data/`, or a trailing-space folder name (`data/models /x.json`), got past it. `_within` now also compares each existing ancestor by file identity (`os.path.samefile`).
  - `fetch` audited the built request, but client auth, request hooks and cookies set on a redirect add headers inside `client.send`. It now refuses a client with auth, request hooks or cookies up front, sends each hop with `auth=None, follow_redirects=False` after auditing it, follows redirects itself (at most 5), and drops cookies the CDN sets after every hop, so a CDN cookie can't abort a run either.
  - Tests: the six strict xfails the second re-test added in `test_retest_resolution_pilot.py` now pass (52 + 26 pilot tests). Fast suite 1551 passed, 1 skipped, 2 xfailed; `ruff check .` clean.

## Results (2026-10-31)

**Decision: stop.** Full resolution doesn't measurably help frozen-DINOv2 corner and edge features tell a 9 from a 10, so no full download.

- **C − A** AUC 10 vs 9 is **+0.017** (95% CI −0.010 to +0.045, P(≤0) 0.11). That is below the +0.03 bar, and the CI includes 0.
- **B − A on the pilot** is **+0.027** (CI +0.004 to +0.052). It doesn't replicate on the 3.7× larger train+val set (2,241 certs, 1,100 card groups): there it is **+0.006** (CI −0.009 to +0.021), and 9-recall *drops* by 3.4 points (CI −7.1 to +0.5). Read the pilot's B − A as noise on a 600-cert subset, not a feature-resolution gain.
- **C − B** is **−0.009** (CI −0.033 to +0.011). The PSA originals add nothing over upsampling the stored photo.
- **Of the three outcomes in Design**, this is "A ≈ B ≈ C: resolution isn't the lever", at least for a frozen DINOv2-B at 224 px tiles.

Report: `test-artifacts/eval/resolution_pilot/report.json` (`python -m ml.eval.resolution_pilot evaluate`, 318 s). Arm C was extracted with `--workers 1` after the first attempt was stopped for low memory (see Extraction).

### CV (5 card-grouped folds, seed 0, the whole head refit per arm)

Pilot, 600 certs (300 9s, 300 10s, 555 card groups), the same certs and folds in every arm:

| Arm | AUC 10v9 | corners probe | edges probe | 9-recall | 10-recall | MAE | QWK |
|---|---|---|---|---|---|---|---|
| A, baseline | 0.651 | 0.658 | 0.650 | 61.0% | 64.7% | 0.372 | 0.257 |
| B, stored photo, 224 px tiles | 0.678 | 0.679 | 0.641 | 59.0% | 64.3% | 0.383 | 0.233 |
| C, PSA original, 224 px tiles | 0.669 | 0.668 | 0.654 | 58.3% | 62.0% | 0.398 | 0.203 |

Paired bootstrap over card groups (1,000 resamples, seed 0, 95% percentile CI):

| Pair | AUC 10v9 | corners | edges | 9-recall |
|---|---|---|---|---|
| B − A | +0.027 [+0.004, +0.052] | +0.021 [−0.015, +0.061] | −0.010 [−0.048, +0.028] | −0.020 [−0.065, +0.023] |
| C − A | +0.017 [−0.010, +0.045] | +0.010 [−0.029, +0.050] | +0.004 [−0.034, +0.043] | −0.027 [−0.084, +0.024] |
| C − B | −0.009 [−0.033, +0.011] | −0.011 [−0.046, +0.020] | +0.014 [−0.014, +0.042] | −0.007 [−0.051, +0.035] |

All train+val, arms A and B (2,241 certs: 613 9s, 1,336 10s, plus ≤8s; 1,100 card groups):

| Arm | AUC 10v9 | corners probe | edges probe | 9-recall | MAE | QWK |
|---|---|---|---|---|---|---|
| A | 0.721 | 0.717 | 0.715 | 27.1% | 0.469 | 0.583 |
| B | 0.727 | 0.727 | 0.709 | 23.7% | 0.478 | 0.559 |
| B − A | +0.006 [−0.009, +0.021] | +0.010 [−0.012, +0.031] | −0.006 [−0.026, +0.015] | −0.034 [−0.071, +0.005] | +0.009 [−0.007, +0.026] | −0.024 [−0.056, +0.008] |

- The pilot holds only 9s and 10s, so its MAE is roughly the 9/10 error rate and its QWK is close to a two-class kappa. Neither compares with the all-grade numbers. AUC 10 vs 9 is the primary metric.
- The pilot's AUC (0.65) sits below train+val's (0.72) probably because its 9s and 10s are spread one per card group, and because it holds no ≤8s (which train with the head). That is a guess, not measured. The arms are compared on the same certs, so this doesn't bias the differences.
- **Arm A reproduces:** re-extracting 8 pilot certs from scratch matches the cached `region_feats.npz` exactly (max abs diff 0.0, cosine ≥ 0.99999988).

### What this does and doesn't rule out

- **Ruled out:** that the 9/10 ceiling comes from the stored photo's 1024 px cap or the 602×434 feature resize, *for frozen DINOv2-B region features*. ADR 0001's resolution hypothesis isn't supported for this feature extractor.
- **Not tested:**
  - a backbone that is fine-tuned or trained for defects at high resolution;
  - surface features (only corners and edges were tiled);
  - 448 px tiles. The rule picked 224, and a typical original's corner holds only ~180 px of real detail.
- **The downloaded originals stay in `data/images_full/`** (1.32 GB, gitignored) for any later experiment of that kind. Deleting them is the owner's call.

### Step 0 (5 originals, `data/images_full/step0.json`)

- **No token is sent.** The request headers are only `host`, `accept`, `accept-encoding`, `connection` and `user-agent: python-httpx/0.28.1`, and the URL has no query string. The host is `d1htnxwo4o0jhw.cloudfront.net`, served by `AmazonS3` behind CloudFront (`x-cache`, `via`, `x-amz-*`).
- **No rate-limit or quota headers on the CDN.** The repo has no evidence that PSA's API exposes them either: the client only reads `Retry-After` on a 429. So the header condition doesn't apply, **0 PSA API calls were made**, and the code inspection stands.
- **Sizes:** 1430–1438×2400 (0.75–1.17 MB, card 1,434–1,590 px tall) and 2807×4742 (3.6–3.8 MB, card ~3,050 px tall). The stored photos are 605–614×1024, with the card 612–679 px tall. The originals are well above 1024 px, so the pilot continues.

### Download (step 1)

- **Selection:** 600 certs, 300 PSA 9 + 300 PSA 10. 499 train and 101 val, **0 test**; 555 card groups (300 distinct among the 9s and 300 among the 10s), from 1,949 train+val candidates.
- **Result:** 1,200 / 1,200 originals ok and **0 failed**, 1.32 GB in total. Fetched sequentially at ≤ 2 requests/s; the 1,195 after step 0 took 602 s. `manifest.csv` lists each file's sha256, bytes, width, height and EXIF orientation.
- **Original sizes:** long edge 2400 px for 994 faces, 3000 px for 182, ~2170 px for 12 and ~4740 px for 12. The card is 1,549 px tall at the median (p5–p95 1,518–1,969), 2.35× the stored photo. A corner block is 180 px at the median. The "3–4 MB" in `image_io.py` holds only for the rare 4,740 px ones; most originals are about 1 MB.

### Checks

- **Precrop quad re-detection, all 2,241 train+val certs (4,482 faces):** every quad is found, none is rotated, and the re-warp against the stored precrop has a MAD median of 1.65, p99 2.51 and max 2.96 grey levels. **0 certs dropped.** It took 335 s on 8 threads.
- **Original vs stored photo, all 1,200 pilot faces:** every size rounds to the stored size, and the MAD median is 2.61 (p99 3.95, max 4.45). **0 dropped**, so the scaled quad covers exactly the precrop's card area on every pilot original.
- **Re-detection sample** (the "Facts vs assumptions" check, 20 faces, `redetect_sample.json`):
  - Running `detect_and_crop` on the original's bytes (shrunk to 1024 px without ingest's JPEG pass) reproduces the stored quad within ~1 stored px on 13 / 20 faces. On 7 it moves 5–164 original px to another nested outline (card / slab well / inner frame).
  - Detection on the full-size original, which the detector isn't tuned for, differs by 10.8 px at the median and 165 px at most.
  - So the scaled quad itself is exact (the check above), but **detection is not stable to "a pixel or two" under a re-encode**. A full-resolution pipeline should carry the 1024 px quad over rather than re-detect, as this spec does.
- **Tile model:** the same weights with `dynamic_img_size`. On a full 602×434 input it reproduces the production region model's `mean` region (cosine 1.0, max abs diff 1.3e-6).

### Extraction (step 2)

| Arm | Certs | Dropped | Time | GPU throughput |
|---|---|---|---|---|
| B, all train+val | 2,241 | 0 | 538 s | 224² corner tiles ~137/s, 224×140 edge tiles ~239/s (RTX 3070 laptop, fp32) |
| C, pilot | 600 | 0 | 148 s | (`--workers 1`) |

The first arm C attempt (`--workers 2`) was stopped by Claude Code for critically low system memory (15.7 GB RAM) before any C tile was cached. With the owner's go-ahead it was rerun with `--workers 1`, and it finished with no memory trouble.

### Integrity (before the first download vs. after `evaluate`)

| | Before (07:19 UTC) | After (08:55 UTC) |
|---|---|---|
| `pokescan.sqlite` sha256 | `185221d6…1399ffb` | `185221d6…1399ffb` |
| size / mtime_ns | 7,086,080 / 1793168145879782500 | 7,086,080 / 1793168145879782500 |
| `data/images` files / bytes | 5,275 / 954,527,815 | 5,275 / 954,527,815 |
| `data/crops` files / bytes | 5,260 / 748,007,793 | 5,260 / 748,007,793 |
| `data/models` files / bytes | 72 / 8,364,610,165 | 72 / 8,364,610,165 |

`git status` lists only the code, the specs, `.gitignore` and `CLAUDE.md`. `data/images_full/` is ignored, and `TESTING.md` was untracked before this work.

## Facts vs assumptions

- **Fact:** the downloads are unauthenticated CDN GETs that don't touch the local quota counter (code above). Step 0 checks PSA's side where it can.
- **Fact:** the AUC numbers and per-grade recall above (guard_noquad report, ADR 0002).
- **Assumption:** the originals are about 3–4 MB and well above 1024 px (`image_io.py` docstring, written at ingest). Step 0 measures this. If they are barely larger, arm C can't differ from B and the pilot stops early.
  - **Checked:** well above 1024 px (median card 2.35× the stored size), but mostly ~1 MB; only the rare ~4,740 px originals are 3–4 MB.
- **Assumption:** a quad found at 1024 px, scaled up, lands on the card in the original to within a pixel or two. Step 2 checks this on a sample, by re-detecting on a few originals and comparing corners.
  - **Checked:** the scaled quad matches on all 1,200 pilot faces (the original check), but re-detecting on the original's bytes isn't stable (Re-detection sample).
- **Assumption:** a frozen DINOv2 at 14 px patches can pick up print-line-scale flaws in a 224–448 px tile. That's what the pilot tests.
  - **Checked at 224 px:** no measurable gain (Results).
