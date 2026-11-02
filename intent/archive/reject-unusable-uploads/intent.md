# Reject unusable uploads

## Intent

With the probe model, `/api/grade` refuses to grade an upload it can't grade meaningfully. It tells the user what to retake instead of returning a grade that looks like any other. There are three cases:

- a photo that isn't a readable image (corrupt or truncated): **400**, instead of today's 500;
- a face where no card outline is found: **422** "card not found, retake on a plain background", instead of grading the padded whole photo;
- a pair the face guard reads as two backs or two fronts: **422**, instead of grading it with a note.

Every other request is graded exactly as today.

Decided by the owner on 2026-10-27 ([[cv-ml-improvement-plan]] open question 2, option (a), "next round, together with the guard's 'both photos look like backs' case"). The corrupt-upload case is `smallfix.md` §10.

## Why

- **Fallback grades look like real grades.** With no quad, `detect_and_crop` pads the whole photo onto the 600×840 canvas (`backend/app/ml/preprocess.py`). The probe never trained on such canvases: `train_probe` fits only certs where both faces had a quad. Yet `/api/grade` returns a normal grade and confidence, plus a note (`backend/app/api/grade.py:105-108`).
- **Same-side pairs are graded anyway.** When both faces read as the same side, the guard adds `"both photos look like card backs"` (or fronts) and changes nothing else (`backend/app/ml/face_guard.py:162-163`). The head then grades two backs as front + back.
- **Corrupt uploads are a 500.** `decode_image` lets PIL errors through (`preprocess.py:533`), and `grade_card` catches only `RuntimeError` (`grade.py:97-100`). Measured 2026-10-28 through `detect_and_crop`: garbage bytes raise `PIL.UnidentifiedImageError`, and a JPEG cut at 50%, a JPEG cut after 200 bytes and a PNG cut at 50% each raise a plain `OSError` ("image file is truncated", "Truncated File Read"). All are `OSError` subclasses, so all become a 500.

How often each case fires on PSA data (2026-10-28, current model and guard):

| Case | Rate | Source |
|---|---|---|
| Fallback face, test split | 5 of 375 certs (1.3%); front 98.7% found, back 99.7% | `test-artifacts/eval/guard_after/report.json` `pipeline` |
| Fallback face, whole DB | 21 stored faces | `crop_found_front/back = 0` |
| Same-side note, genuine pairs, test | 0 of 370 | `face_guard_report.json` `test.side.pairs` |
| Same-side note, genuine pairs, card-grouped CV | 3 of 2,241 (0.13%) at the 0.9 side threshold | `face_guard_report.json` `cv_grouped.side.pairs` |

## Scope

### Undecodable images (`backend/app/ml/preprocess.py`, `inference.py`, `api/grade.py`)

- Add `class ImageDecodeError(ValueError)` in `preprocess.py`. `decode_image` raises it for bytes PIL can't decode, chained with `from exc`: `OSError` (including `UnidentifiedImageError` and truncation), `Image.DecompressionBombError`, and the `SyntaxError`, `ValueError`, `IndexError`, `TypeError`, `KeyError`, `OverflowError` and `struct.error` PIL's plugins raise for malformed headers and chunks. Wrap the whole decode (open, load, EXIF, convert, resize, `np.array`), because `Image.open` is lazy and truncation surfaces later, and nothing outside it, so no non-decode error becomes a 400. An `ImageDecodeError` is never wrapped twice.
  - 2026-10-28, after the third re-test (D4): `KeyError` and `OverflowError` added as a safety net. PIL's TIFF loader raised them for a malformed IFD, and the allowed formats' decoders share that code.
- Bytes are opened only as JPEG, PNG or WebP (`Image.open(..., formats=preprocess.UPLOAD_FORMATS)`), the formats `/api/grade`'s `ALLOWED_TYPES` accepts. The content type is client-declared, so bytes of any other format PIL knows (TIFF, GIF, BMP, PPM, ICO, QOI, SGI, …) sent as `image/jpeg` are an `ImageDecodeError` (400) instead of reaching that format's decoder. A PIL `Image.Image` passed to `decode_image` / `detect_and_crop` is already opened by the caller and isn't format-checked. Every bytes caller (`/api/grade`, `ml.data.precrop`, `ml.eval.evaluate`, `ml.eval.detection_report`) feeds stored photos, which are all JPEG (`ml.data.image_io`; 5,275 photos and 5,260 crops checked 2026-10-28), or JPEGs it re-encodes; `ml.data.dataset` passes numpy arrays.
  - 2026-10-28, after the third re-test (D4): added (owner-approved).
- Once an image opens, a malformed EXIF block doesn't reject it: the pixels are decoded first, then the EXIF step catches any exception. If the orientation can't be read, or isn't 2–8, the image is taken as upright (orientation 1). If it reads but `ImageOps.exif_transpose` fails, that orientation is applied directly with the same transpose (pixel-identical to `exif_transpose` on clean EXIF). Only the EXIF step catches any exception; the pixel decode still wraps `_DECODE_ERRORS` only. The exception is EXIF that PIL reads while opening: a JPEG whose EXIF makes PIL's JPEG `open` fail (e.g. a one-element XResolution with ResolutionUnit set raises `IndexError` in its dpi read, which PIL reports as unidentified) is refused as unreadable (400). It is rare (0.34% of an EXIF-only JPEG fuzz). Stored training photos carry no orientation tag, so they are unaffected.
  - 2026-10-28, after the re-test: widened from `OSError` and `DecompressionBombError` only.
  - 2026-10-28, after the second re-test (D2/N1): the EXIF step widened from `_DECODE_ERRORS` to any exception, and a failing `exif_transpose` applies the readable orientation instead of grading the photo unrotated.
  - 2026-10-28, after the third re-test (D3): narrowed from "a malformed EXIF block alone doesn't reject a photo". The owner accepted the 400 for EXIF that fails PIL's open; no EXIF strip-and-retry.
- `GradeInferenceService.predict` labels it per face: `ImageDecodeError("front: …")` / `("back: …")`, front first. Both the probe and the legacy paths do this.
- `grade_card` maps it to **400** with the detail `"<face>: not a readable image (corrupt or truncated file)"`.
- Never catch a bare `OSError` in `grade.py`.

### Card not found (`api/grade.py`, probe model only)

- After `service.predict`, if `pred.region_grades is not None` (the probe path) and a face has `found_quad_* == False`, return **422** with the detail:
  - `"front: card not found; retake the photo on a plain background"`;
  - `"back: …"`, or `"front and back: …"` when both faces fail.
- The legacy model keeps today's behaviour: 200 plus the `"used center-padded fallback"` note. It's served only through an explicit `model_path` or `ALLOW_LEGACY_MODEL=1`.
- `found_quad_*` describe the faces after a guard swap. That doesn't change whether a face failed, but the message must name the face as **uploaded**. The side check also runs on a padded canvas, so a pair with a fallback face can be swapped. `GradePrediction` gains `swapped: bool = False`, and the check reverses the flags when it's set. A test pins this.
  - 2026-10-28, during development: corrected from "a swap goes with both faces found", which nothing guarantees.
- The rule lives in `GradePrediction.rejection() -> (kind, message) | None`, so the API and `ml.eval` share it.

### Same-side pair (`inference.py`, `api/grade.py`)

- `GradePrediction` gains `same_side: str | None = None`. `_predict_probe` sets it to `"back"` or `"front"` when the guard's side check reads both faces as that side (the `decide` short-circuit at `face_guard.py:162`). `GuardDecision` carries it as a `same_side` field set there, so it isn't inferred from the notes. The note stays in `pred.notes`, so `ml.eval` and the existing tests are unchanged.
- `grade_card` returns **422** when `pred.same_side` is set, with the detail `"both photos look like card backs; upload one photo of the front and one of the back"` (or `fronts`).
- No guard loaded, or a guard error on this request: no check, and the request is graded as today. The guard fails open.

### Order

The decode error comes first (400; front checked before back), then card not found, then same side. Card not found takes precedence over same side, because the side check on a padded canvas is out of distribution.

### Evaluation (`ml/eval/evaluate.py`)

- `predict` keeps returning a prediction for every input, so `evaluate` and the robustness runs still grade fallback and same-side pairs, and their metrics don't change.
- The report's `pipeline` block gains `n_would_reject: {"card_not_found": n, "same_side": n}`, so a run shows how many test certs the API would refuse.

### Docs

- README "What it does" step 1: replace "the whole photo is padded onto the canvas and the response notes it" with the 422, keeping the padded fallback as the legacy-model behaviour.
- README "API reference" (`README.md:508`): add the `/api/grade` error responses (400 empty/unreadable, 413, 415, 422 card not found / same side, 503) under the table. None of them is documented today.
- `CLAUDE.md` inference-path paragraph: one sentence on the rejections.
- `smallfix.md` §10: tick the corrupt-upload item, pointing here.

## Out of scope

- Frontend changes. `src/app/api-client.ts:51-56` already throws `detail` as the error message, and `Uploader.tsx:79-86` shows it, so the user sees the retake text unchanged. Per-face highlighting or a structured error code is its own small PR, as the owner decided for Q10.
- A structured error body (`{code, faces}`). The detail stays a string, the same as the existing 400/413/415.
- Rejecting earlier to save compute. A card-not-found request still runs DINOv2 (~2.3 s p50) before the 422, because `predict` must keep grading for `ml.eval`. It's rare (1.3% of PSA test certs).
- Changing detection, the face guard's thresholds or probes ([[face-guard-serve-path-features]]), or the legacy path.
- The other §10 items (`CapacityLimiter`, alpha compositing, square photos).

## Acceptance criteria

Tests use synthetic images (`backend/tests/helpers/synth.py`) and stub services (`backend/tests/conftest.py` `FakeService`), with `tmp_path` only.

- [x] Fast suite: `python -m pytest -m "not slow and not visual and not model"` → 0 failed, 2 xfailed, passed = 663 + the new tests. No existing assertion changes, except API tests that post a fallback face to a probe-path stub, if any. List each one here with its reason.
  - 2026-10-28: 698 passed (663 + 35 in `backend/tests/test_reject_uploads.py`), 2 xfailed, 184 deselected. No existing test changed: `test_grade_reports_missing_quads_in_notes` posts a fallback through the legacy-path stub (`region_grades=None`), which still returns 200.
- [x] `ruff check .` exits 0.
- [x] Decode: garbage bytes, a JPEG truncated at 50%, a JPEG cut after 200 bytes and a PNG truncated at 50% each give `ImageDecodeError` from `decode_image`. Posted as `front` (valid `back`), each gives 400 with the detail starting `"front: "`. As `back`, the detail starts `"back: "`. None gives a 500.
  - 2026-10-28, re-test (`backend/tests/test_retest_reject_uploads.py`): three inputs still gave a 500, because PIL raised something other than `OSError`. A multi-IDAT PNG with a garbled second chunk type (`SyntaxError: broken PNG file` in `load()`), a readable WebP whose EXIF has a garbled TIFF header (`SyntaxError: not a TIFF file` in `getexif()`), and non-JPEG bytes sent as `image/jpeg` (a bad PPM header: `ValueError`; fuzzed QOI/SGI/TIFF: `IndexError`, `TypeError`). Fixed as in Scope: the PNG and PPM give 400, and the WebP is graded (200) with its EXIF ignored. The re-test's 5 strict-xfail tests now pass as regression tests; `test_reject_uploads.py` adds one per wrapped exception type, the EXIF fallback, and non-decode errors that stay 500. `test_preprocess.py::test_stored_precrops_are_reproduced_byte_identical` still passes.
  - 2026-10-28, second re-test (`backend/tests/test_retest_reject_uploads_fix.py`), D2: a readable photo whose EXIF holds a valid orientation and a tag PIL types as ASCII but that holds another type (e.g. XResolution renamed to InkNames) gave a 500. `exif_transpose` rotates, then re-serialises the EXIF, which raised `AttributeError` (`'IFDRational' object has no attribute 'encode'`), outside `_DECODE_ERRORS`; ~4.5% of fuzzed EXIF blocks with a valid orientation, JPEG/PNG/WebP. N1: when `exif_transpose` raised a wrapped error instead (`TypeError`), the photo was graded unrotated although its orientation was readable. Fixed as in Scope (`preprocess._apply_exif_orientation`): the photo is graded (200) with the readable orientation applied, pixel-identical to `exif_transpose` of a clean-EXIF copy for orientations 2–8. The re-test's 5 strict-xfail tests now pass. Changed to the new rule (owner-approved): `test_retest_reject_uploads_fix.py::test_exif_transpose_type_error_means_upright` and `test_exif_transpose_attribute_error_means_upright` (now `…_still_applies_the_orientation`), and `test_reject_uploads.py::test_a_failing_exif_transpose_means_upright` (now `test_a_failing_exif_transpose_still_applies_the_orientation`, all of orientations 2–8). Added: `AttributeError`/`KeyError`/`Exception` from the EXIF read or transpose, an unhashable or out-of-range orientation (upright), the fallback per orientation and format, and truncated pixels with broken EXIF (still `ImageDecodeError`). `test_preprocess.py::test_stored_precrops_are_reproduced_byte_identical` still passes.
  - 2026-10-28, third re-test (`backend/tests/test_retest_reject_uploads_exif.py`). D4: TIFF bytes sent as `image/jpeg` whose IFD PIL can't use gave a 500. PIL's TIFF `load()` applies EXIF itself and raised `KeyError` (an Interoperability IFD pointer in IFD0) or `OverflowError` (a LONG8 strip offset past `ssize_t`), both outside `_DECODE_ERRORS`. D3: a JPEG whose EXIF has ResolutionUnit and a one-element XResolution fails PIL's JPEG `open` (`IndexError` in its dpi read, reported as `UnidentifiedImageError`), so it gave 400 although its pixels are intact (0.34% of an EXIF-only JPEG fuzz). Fixed as in Scope (owner-approved): bytes are opened only as JPEG/PNG/WebP (`preprocess.UPLOAD_FORMATS`), so other formats' bytes are a 400, and `KeyError`/`OverflowError` are in `_DECODE_ERRORS`. D3 is accepted as a 400, and the EXIF rule and the README 400 row now say so. The re-test's 4 D4 strict xfails now pass, tightened within their own contract to `ImageDecodeError` and 400 naming the face (the API one per face). Its 4 D3 strict xfails became tests pinning the 400 (`test_d3_jpeg_with_an_unindexable_xresolution_is_a_decode_error`, `…_is_a_400_naming_the_face`). Changed to the new rule, since their bytes are TIFF/GIF/BMP: in `test_retest_reject_uploads_fix.py`, `test_valid_images_decode_exactly_as_before_the_fix` (6 TIFF/GIF/BMP cases), `test_animated_uploads_are_graded_on_their_first_frame[animated_gif]` and `test_every_valid_orientation_is_still_applied[*-TIFF]`; in `test_retest_reject_uploads_exif.py`, `test_forced_fallback_is_pixel_identical_to_exif_transpose[*-TIFF-*]`. Those formats now go in as an opened PIL image, which isn't format-checked, so the pixel checks still hold. Added: valid bytes of 9 other formats give `ImageDecodeError` and a 400 per face, JPEG/progressive JPEG/PNG/WebP/MPO bytes still decode, a PIL image of a TIFF/GIF/BMP still decodes, `KeyError`/`OverflowError` from the load give `ImageDecodeError`/400 (synthetic, and PIL's own from the D4 TIFFs opened by the caller), and the same two from outside the decode stay 500. Fast suite: 1207 passed, 2 xfailed (was 1145 passed, 10 xfailed). `test_preprocess.py::test_stored_precrops_are_reproduced_byte_identical` still passes.
  - 2026-10-28, fourth re-test (`backend/tests/test_retest_reject_uploads_formats.py`). D5: a palette PNG (colour type 3) with no PLTE chunk and a one-entry tRNS gave a 500. PIL loads it as mode `"P"` with no palette, and `convert("RGB")` fails `assert self.palette is not None` (`AssertionError`, outside `_DECODE_ERRORS`). It was the only failure in a 1,440-PNG header matrix and 8 of 192,000 header-fuzzed inputs; same on main. Fixed (owner-approved, without a blanket `AssertionError` catch, which could hide bugs as 400s): `_decode_image` refuses a `"P"`/`"PA"` image whose palette is None as an `ImageDecodeError` right after the load, so any palette-less palette PNG is a 400. Its 3 strict xfails became tests pinning the 400 (`test_d5_palette_png_without_plte_is_a_decode_error`, `…_is_a_400_naming_the_face`), and the matrix and fuzz tests no longer skip D5. The re-test found no other defect: all 20 other formats PIL reads are a 400, every JPEG/PNG/WebP variant tried decodes as on main, and 288,000 fuzzed inputs gave only a decode or `ImageDecodeError`.
- [x] Card not found, probe stub: front fallback → 422 with the front detail; back → back detail; both → `"front and back: …"`.
- [x] Card not found, legacy stub (`region_grades=None`): front fallback → 200 with the existing fallback note.
- [x] Same side, probe stub: `same_side="back"` → 422 with the backs detail; `"front"` → fronts.
- [x] Same side through the real `_predict_probe` with a stub guard (as `test_retest_419a2d4.py:365` does): `pred.same_side` is set and `pred.notes` is unchanged. A swap or flip decision leaves `same_side` None.
- [x] Order: a fallback face plus `same_side` → card-not-found detail; an undecodable front plus an undecodable back → 400 naming the front.
- [x] The swap-naming criterion from Scope is pinned by a test.
  - `test_card_not_found_names_the_uploaded_face_after_a_swap`: a blank uploaded front, swapped by the guard, gives `"front: card not found; …"`.
- [x] Evaluation: `python -m ml.eval.evaluate --split test` on the current model gives the same per-cert predictions as `test-artifacts/eval/guard_after/report.json`, and `pipeline.n_would_reject` is recorded here (expected card_not_found 5, same_side 0).
  - 2026-10-28 (`--no-robustness --n-boot 0`, report in the session scratchpad): all 375 certs match `guard_after` exactly (0 `pred_int` differences, max `pred_soft` difference 0.0, 0 note differences, identical `overall`). `n_would_reject`: card_not_found 5, same_side 0.
- [x] Docs updated as listed.

## Owner decisions (2026-10-28)

1. **Fallback faces: reject as decided (422).** The 2026-10-23 evidence was MAE 0.84 on 19 fallback certs (old split, before [[serve-input-normalization]]). Today's 5 fallback test certs have MAE 0.2, but n=5 is too few to read. The probe never trained on padded canvases, and user photos are unmeasured until the phone-photo set exists ([[cv-ml-improvement-plan]] Q8).
2. **Same side: reject both backs and both fronts.** Two fronts are as ungradeable as two backs, since half of the input is wrong.
3. **Unreadable image: 400**, like the empty upload. 422 is for a readable photo whose content can't be graded.
4. **Wording: the drafts in Scope are final.** They're shown to users verbatim.

## Facts vs assumptions

- **Fact:** the rates in the table and the decode exceptions above, measured 2026-10-28.
- **Fact:** the frontend shows `detail` as-is (`api-client.ts:51-56`).
- **Assumption:** user uploads miss a quad more often than PSA's studio photos do, so the 422 fires more often in production than the 1.3% here. Unmeasured until the phone-photo set exists.
- **Assumption:** a same-side note on a genuine pair (0.13% in CV) is rare enough to refuse rather than grade. The user can retake.
