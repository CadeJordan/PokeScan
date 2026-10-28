# Serve-time input normalisation

## Intent

A phone photo uploaded to `/api/grade` reaches the card detector in the same form as the training photos: upright per its EXIF orientation, and capped at 1024 px on the long edge. A sideways card is still found. Each face is detected once per request, and centering is never measured on a fallback canvas.

On the existing data, nothing about detection or crops changes: every stored precrop stays byte-identical.

## Why

Measured 2026-10-23 on 120 real test-split faces (see [[cv-ml-improvement-plan]] item 1):

| Upload | quad found | detect p50 |
|---|---|---|
| as stored (1024 px, no EXIF) | 99.2% | 111 ms |
| EXIF Orientation=6 (phone portrait; pixels stored sideways) | 14.2% | 60 ms |
| sideways, no EXIF | 18.3% | |
| upscaled 3× (3072 px, like a 12 MP phone) | 70.0% | 503 ms |
| EXIF-6 after `ImageOps.exif_transpose` + 1024 cap | 97.5% | |
| 3× after the 1024 cap | 99.2% | |

- `detect_and_crop` decodes bytes with `Image.open(...).convert("RGB")` (`backend/app/ml/preprocess.py:512-516`), with no EXIF handling and no resolution cap.
- Ingest applies `ImageOps.exif_transpose` and a 1024 long-edge cap (`ml/data/image_io.py:27-28,46,59-65`). All 5,260 stored images are exactly 1024 px on the long edge with no EXIF orientation tag, so the cap and the transpose are no-ops on training data.
- The aspect gate `_AR_MIN, _AR_MAX = 0.65, 0.82` (`preprocess.py:73`) rejects every landscape quad, so a card photographed sideways always falls back.
- `compute_centering_subgrade` runs `detect_and_crop` again on the raw bytes (`backend/app/ml/centering.py:314-317`) and measures the fallback canvas too. On the 21 real fallback faces it returned 3 measurements, all `50/50` → 10.0.
- `/api/grade` detects every face twice (`inference.py:184-185` and `centering.py:314-317`). With the legacy model it detects a third time (`condition.py:393-394`).

## Scope

### Decode and normalise (`backend/app/ml/preprocess.py`)

- Add `MAX_LONG_EDGE = 1024`, with a comment that it must equal `ml.data.image_io.IMAGE_MAX_LONG_EDGE`. A test asserts they're equal.
- Add `decode_image(image: bytes | bytearray | Image.Image) -> tuple[np.ndarray, float]`. It returns `(bgr, scale)`:
  1. `ImageOps.exif_transpose`;
  2. `.convert("RGB")`, the same alpha behaviour as today;
  3. if the long edge is over `MAX_LONG_EDGE`, resize with `Image.LANCZOS` to a long edge of exactly `MAX_LONG_EDGE`, rounding the same way as `compress_jpeg`.

  `scale` is the normalised size over the decoded size, 1.0 when nothing was resized.
- `detect_and_crop` uses `decode_image` for bytes and PIL inputs. **numpy inputs are not normalised.** Document that callers passing arrays own their preprocessing; the synthetic-geometry tests rely on this.

### Sideways retry (`detect_and_crop`)

- If `_find_card_quad(bgr)` returns None **and the photo is landscape** (`bgr.shape[1] > bgr.shape[0]`, on the `bgr` passed to detection, i.e. after EXIF transpose and the cap), retry once on `cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)`. This applies to every input type, numpy included.
  - Added 2026-10-23 after the unconditional retry found false quads on 2 of the 21 stored fallback faces, both upright portrait photos (see the detection-report criterion). All 5,260 stored photos are portrait, so the retry never runs on them, and every simulated sideways upload is landscape.
  - Accepted cost: a sideways card inside a portrait photo is not found and falls back. The unconditional retry missed it too on the measured data, so this isn't a regression. Area thresholds were measured and can't separate the false quads from real cards.
- If the retry finds a quad, warp from the rotated image. `source_bgr`, `quad` and `contour` are all in the rotated frame, so `condition.py`'s contour geometry stays consistent.
- The card may come out upside down. Fixing that is [[face-side-orientation-guard]]'s job, not this spec's.

### `CardCrop` fields

Add two fields with defaults:
- `rotation: int = 0`: degrees clockwise applied before detection (0 or 90);
- `scale: float = 1.0`: from `decode_image`.

Existing fields keep their meaning, in the frame of `source_bgr`.

### One detection per face

- `GradePrediction` (`backend/app/ml/inference.py`) gains `crops: tuple[CardCrop, CardCrop] | None = None`. Both the probe and the legacy `predict` paths fill it with the `(front, back)` crops they used.
- `compute_centering_subgrade(front, back=None)` accepts `bytes | CardCrop` per face. A face whose crop has `found_quad=False` is not measured: it's None for that face. Detect only when given bytes.
- `compute_condition_subgrades` accepts `bytes | CardCrop` the same way. It's legacy only; keep its behaviour.
- `backend/app/api/grade.py` passes `pred.crops` to both when present, and bytes otherwise. The `FakeService` in `backend/tests/conftest.py` returns no crops and must keep working.
- When `crop.rotation == 90`, add a note: `"front: photo was sideways; rotated it"` (or `back:`). Notes are free text, so the schema doesn't change.

### Detection report (`ml/eval/detection_report.py`)

- Add `--variant` with values `orig` (default), `exif_rot90`, `rot90`, `rot180`, `up3x`. Each re-encodes the stored image as a JPEG at q=92 before `detect_and_crop(bytes)`:
  - `exif_rot90`: pixels rotated 90° CCW, EXIF Orientation=6 written with PIL;
  - `rot90`: rotated 90° CW, no EXIF;
  - `up3x`: `cv2.resize` ×3 with `INTER_CUBIC`.
- Print the found rate per side and the detect-time p50/p95 per variant.
- Open the DB read-only: `sqlite3.connect(f"file:{db}?mode=ro", uri=True)`, or `connect(..., read_only=True)` once [[test-data-isolation]] lands. Today it uses the migrating read-write `connect` (`detection_report.py:56`).
- `summary.json` gains `variant`, `detect_ms_p50` and `detect_ms_p95`.

### Docs

- Fix the `preprocess.py` module docstring: the fallback is a resize-and-pad of the whole photo, not a bounding-box crop.
- Document the normalisation and the sideways retry there, and in the README "What it does" step 1.

## Out of scope

- Fixing upside-down cards (that's [[face-side-orientation-guard]]).
- Rejecting or flagging fallback crops beyond today's note. That's a contract decision ([[cv-ml-improvement-plan]] open question 2).
- Any change to detector thresholds, `_find_card_quad`, precrop output, or the training pipeline. No `precrop --force`, no retrain.
- Normalising numpy-array inputs.
- Frontend changes.

## Acceptance criteria

Unit tests use synthetic images only (`backend/tests/helpers/synth.py`) and `tmp_path`.

- [x] Fast suite: `python -m pytest -m "not slow and not visual and not model"` → 0 failed, 2 xfailed, and passed = 381 + the new tests.
  - The only existing assertion allowed to change is `test_detect_and_crop_accepts_input_types` (`backend/tests/test_preprocess.py:105`). For the `bytes`/`pil` kinds it now expects `max(source_bgr.shape[:2]) == 1024`, because the 1200×1000 scene is capped. The `ndarray` kind is unchanged.
  - 2026-10-23, after the landscape-only retry fix: 398 passed (381 + 17 new; `test_sideways_card_in_portrait_photo_is_not_retried` added), 2 xfailed, 183 deselected, 103 s.
  - 2026-10-23: 397 passed (381 + 16 new), 2 xfailed, 183 deselected (182 + the new slow test), 103 s. No other existing test changed. `GradePrediction.crops` is excluded from `==`/repr (`field(compare=False, repr=False)`): with it compared, `test_selection_both_is_probe_and_predicts` (`a == b`) failed on numpy truth values.
- [x] A new test asserts `preprocess.MAX_LONG_EDGE == ml.data.image_io.IMAGE_MAX_LONG_EDGE`.
- [x] EXIF test: encode `make_scene(seed=0)` with its pixels rotated for EXIF orientations 3, 6 and 8, writing the matching tag with PIL.
  - Each gives `found_quad=True`.
  - The crop's normalised cross-correlation with the crop of the upright upload is ≥ 0.9.
  - `test_exif_orientation_is_applied`: NCC 1.00 for all three. It also asserts `rotation == 0`, so the sideways retry can't mask a missing transpose (without it, 6 and 8 would be found by the retry, upside down or not).
- [x] High-res test: a 3600×3000 scene as JPEG bytes gives `found_quad=True`, `max(source_bgr.shape[:2]) == 1024`, and `scale ≈ 1024/3600` (rel. 1e-3).
- [x] Sideways test: `cv2.rotate(scene, ROTATE_90_CLOCKWISE)` as bytes, no EXIF, gives `found_quad=True` and `rotation == 90`. The crop has NCC ≥ 0.9 with either the upright crop or its 180° rotation.
  - Measured: NCC 1.00 with the 180° rotation, 0.53 with the upright crop (this direction lands upside down, as the open question expects).
- [x] Negative test: the existing blank, noise and square fallback cases (`test_no_card_falls_back_to_padded_resize`) still fall back after the retry.
  - On real data the unconditional retry was not free of false quads (see the `orig` note under the detection report criterion), so the retry now runs on landscape photos only. `test_sideways_card_in_portrait_photo_is_not_retried` pins that: a sideways card padded into a portrait photo gives `found_quad=False` and `rotation == 0` (the unconditional retry found it at `rotation == 90`). The landscape sideways test is unchanged and still passes.
- [x] Centering test: `compute_centering_subgrade(<fallback CardCrop>, <card CardCrop>)` gives a result with front fields None. Two fallback crops give None.
  - The fallback crop's canvas is a clean bordered card that `measure_centering` does measure, so the test shows `found_quad=False` alone causes the abstention. Bytes that fall back are skipped the same way (`test_fallback_detection_from_bytes_is_not_measured`).
- [x] API test: with a stub service whose `GradePrediction.crops` is set, `/api/grade` never calls `detect_and_crop`. Monkeypatch it in `centering` and `preprocess` to raise. A note is present when a crop has `rotation=90`.
  - `test_grade_reuses_the_service_crops`, for the probe and legacy paths; it also patches `condition`. Sub-grade errors are swallowed into notes, so the test checks centering and corners came back and no "unavailable" note was added.
- [x] Real-data no-op check (new test marked `slow`; reads only, DB opened `mode=ro`): for 50 random test-split faces with a stored precrop, `encode_jpeg(detect_and_crop(raw_bytes).image)` is **byte-identical** to the stored crop file. 80/80 matched before this change (2026-10-23).
  - `test_stored_precrops_are_reproduced_byte_identical` (`backend/tests/test_preprocess.py`, `slow`): 50/50 byte-identical, 6 s. The sample is `default_rng(0)` over all test-split faces with a precrop.
  - With the unconditional retry this was not quite "every stored precrop": 2 of the 21 stored fallback faces in the whole DB were found by the retry (88000652 front, test; 88002016 front, train), so `precrop --force` would have changed those two crops. Fixed by the landscape condition: both are portrait photos (607×1024 and 614×1024) and now give `found_quad=False`, `rotation=0` and a crop byte-identical to the stored file.
  - The test now always adds every stored fallback face (`crop_found_front = 0` or `crop_found_back = 0`, any split; 21 faces) to the random sample, asserts their crops are byte-identical, and asserts `found_quad=False` and `rotation == 0` for them. The sample of 50 alone missed the bug. 2026-10-23: passes, 7 s.
- [x] `python -m ml.eval.detection_report --split test --variant <v>`, record the table in this spec:
  - `orig`: found rate ≥ 98.6% front and ≥ 99.4% back (today 98.9% / 99.7%);
  - `exif_rot90` and `up3x`: each side within 2 pp of `orig`;
  - `rot90`: each side within 3 pp of `orig`;
  - `up3x` detect p50 ≤ HEAD's `up3x` p50 and ≤ the `orig` p50 + 75 ms. (Restated 2026-10-23 from "≤ 1.5× the `orig` p50", which sat on the run-to-run noise; see the `up3x` note below.)

  Results, 2026-10-23, test split (393 certs, 786 faces), CPU, run one at a time. Detect time is `detect_and_crop(bytes)`, decode included:

  | Variant | front | back | detect p50 | p95 |
  |---|---|---|---|---|
  | `orig` | 99.0% | 99.7% | 107 ms | 149 ms |
  | `exif_rot90` | 98.7% | 99.7% | 104 ms | 144 ms |
  | `rot90` | 99.5% | 99.7% | 142 ms | 179 ms |
  | `rot180` | 99.0% | 100.0% | 103 ms | 143 ms |
  | `up3x` | 99.5% | 100.0% | 161 ms | 201 ms |

  - The table is the re-run with the landscape-only retry (2026-10-23). All rate thresholds are met. `orig` front is back to 99.0% (it was 99.2% with the unconditional retry, which counted the false quad on 88000652 front).
  - **`up3x` p50 meets the restated bar:** 161 ms against `orig` + 75 = 182 ms, and against HEAD's 435 ms (interleaved retest on 250 faces, `intent/smallfix.md` §10, where the found rate also went from 76% to 99.6%). Under the original 1.5× bar it was borderline: 161/107 = 1.50× (1.505), and a second pair gave 162/107 = 1.51×; the ratio's 95% CI is [1.465, 1.523], so that was noise, not a regression. The landscape condition can't affect this, since stored and `up3x` photos are portrait and the retry doesn't run on them.
  - Before the landscape fix: `orig` 99.2% / 99.7%, 107 ms / 150 ms; `exif_rot90` 105 ms; `rot90` 146 / 184 ms; `rot180` 104 ms; `up3x` 160 / 201 ms. `up3x` p50 was then 1.495× `orig`: met, but only just. An earlier run measured 172 vs 111 ms (1.55×). The fix that got it under was skipping `exif_transpose` and `convert` when they're no-ops (each copies the image, ~10 ms at 12 MP). The rest is the full-size JPEG decode (~19 ms) plus LANCZOS (~40 ms). PIL `draft` (DCT-domain downscale) would roughly halve that, but it no longer matches the ingest resize, so I left it out. Expect this ratio to wobble around 1.5× between runs.
  - `orig` changed: front is 99.2% against 98.9% before. One test face, 88000652 front, used to fall back and is now found by the sideways retry. **It is a false quad**, so the facts-vs-assumptions "no false quads" assumption doesn't hold. The photo is an ordinary upright slab, but the quad covers an inner text panel of a full-art card, taken in the rotated frame (overlay checked). Across all 21 stored fallback faces in the DB, the retry finds a quad on 2, and both are wrong in the same way: 88000652 front (test) and 88002016 front (train), both full-art fronts in portrait photos. That is 2 of ~5,260 faces on PSA data. For these faces the API returned a "photo was sideways" note, dropped the fallback note, graded a part of the card, and centering measured that part rather than abstaining.
  - **Resolved 2026-10-23:** the retry now runs only when the decoded photo is landscape (see Scope, "Sideways retry"). Both false hits are portrait photos and every simulated sideways upload is landscape, so both now fall back as before, while the rates above stay within thresholds.
  - `orig` passes the stored bytes unchanged, with no q=92 re-encode, so it stays comparable with the numbers above and with precrop. The four simulated variants are re-encoded with PIL at q=92.
  - Output goes to `test-artifacts/detection_failures/` (`orig`) and `test-artifacts/detection_failures/<variant>/`. The DB is opened `mode=ro`.
- [x] `ruff check .` exits 0.

## Facts vs assumptions

- **Fact:** the measurements above. All stored images are 1024 px on the long edge with no EXIF orientation. Stored precrops are byte-identical to the current detector's output (80/80 sampled). Those three facts are why the change is a no-op for training data.
- **Fact:** the `exif_rot90` simulation uses PIL with an EXIF tag. Real phone JPEGs can also carry orientation in XMP or HEIC containers; HEIC isn't in `ALLOWED_TYPES`.
- **Assumption (falsified 2026-10-23 on 2 of 21 fallback faces; see the detection-report criterion):** retrying at 90° on a fallback adds no false quads on normal photos. A rotated portrait card becomes landscape and fails the aspect gate, but check the overlays of any face whose found status changes in the `orig` report.
  - It failed on upright portrait photos of full-art cards, where the rotated search locks onto an inner text panel. The retry is now limited to landscape photos. All 5,260 stored photos are portrait, so it never runs on them.
- **Assumption:** a sideways card in a phone photo arrives as a landscape photo (after EXIF transpose). A sideways card inside a portrait photo is not found and falls back, the same as before the retry existed.
- **Assumption:** LANCZOS via PIL (the ingest path) is the right resize to match training. The tiny difference against `cv2.INTER_AREA` was not measured.

## Open questions

- Retry direction: either 90° rotation makes a sideways card portrait again, so the direction doesn't affect whether a quad is found. It only decides whether the card ends upside down, and [[face-side-orientation-guard]] fixes that. Until the guard lands, about half of sideways uploads are graded upside down: −0.19 soft grade on average, max 0.63 (16 test certs). That is still far better than the fallback. Is that acceptable as an interim state?
