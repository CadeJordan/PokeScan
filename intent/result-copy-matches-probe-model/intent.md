# Frontend copy matches the probe model

## Intent

Make every claim the web UI makes about how a grade is produced true for the model `/api/grade` actually serves: the frozen DINOv2 probe model ([[probe-artifact-consistency]]). Implements [[cv-ml-improvement-plan]] Q4 ("reword the UI hints", owner decision 2026-10-27). Frontend only, its own small PR (Q10).

## Why

The UI copy was written for the legacy ConvNeXt + CORN model, which the API no longer serves unless `ALLOW_LEGACY_MODEL=1` is set (fail closed since 2026-10-27). What the user sees today:

| Where | Says | Reality with the probe model |
|---|---|---|
| `GradeResult.tsx`, sub-grade header | "Sub-grades": Centering / Corners / Edges / Surface | Corners / edges / surface are each region probe's guess at the **overall** grade, learned from overall PSA grades. No sub-grade labels exist. Centering is a real measurement from border widths. README already says "what this region suggests, not PSA's own sub-grade". |
| `GradeResult.tsx`, confidence caption | "temperature-calibrated on validation" | False. The probe has no temperature step. Its out-of-fold NLL-optimal temperature is 1.00, so it needs none. Test-split ECE is 0.053 (guard_noquad, 2026-10-30). |
| `GradeResult.tsx`, distribution caption | "P(grade > N): cumulative output of the CORN ordinal head" | The head is a proportional-odds meta probe. `rank_probs` is still P(grade > N), so only the model name is wrong. |
| `page.tsx`, pipeline step 02 | "Score with ConvNeXt … shared-weight ConvNeXt-Tiny backbone with a CORN ordinal head" | Frozen DINOv2-B features (11 regions per face), four region probes and a meta probe. |
| `page.tsx`, pipeline step 03 | "Calibrate: logits pass through a temperature scalar fit on held-out PSA data" | No such step. Also missing: the face guard (swap / upside-down front) and the 422 when no card is found. |
| `page.tsx`, intro, and `layout.tsx` meta description | "scores it on centering, corners, edges and surface … with a calibrated confidence" | "Scores it on" corners/edges/surface overstates it, as above. "Calibrated" is defensible only as "checked against held-out grades". |
| `page.tsx`, tip "Both sides" | "Back contributes ~30% of the grade." | Legacy `front_weight`. The probe head has no such weight, and nothing measures a 30% share. |

The [[cv-ml-improvement-plan]] calls this "medium (trust)": the numbers are fine, but the page tells users they mean something they don't.

## Scope

- Copy changes in `frontend/src/components/GradeResult.tsx`, `frontend/src/app/page.tsx` and the `layout.tsx` metadata description.
- Small layout changes if the owner picks them (e.g. a caption under the region cells, or splitting centering from the regions).

## Out of scope

- The API, `GradeResponse` schema and `api-client.ts` types. The backend `hint` strings ("clean", "slight wear", "wear", "heavy wear") stay as they are.
- Copy for the legacy model. The UI describes the served default; the legacy path is an opt-in fallback.
- Error display. 400/422 `detail` strings already show under "Couldn't grade that pair".
- Visual redesign.

## Acceptance criteria

- [x] No remaining claim of ConvNeXt, CORN or temperature calibration in `frontend/src`: `grep -rniE "convnext|corn|temperature|calibrat|30%"` finds nothing (a plain `corn` also matches "corners").
- [x] The corners / edges / surface cells say, on screen without hovering, that they are the region's read on the overall grade and not PSA sub-grades. Wording per owner decision 1.
- [x] The confidence caption and the distribution caption are true for the probe model (decisions 2 and 3).
- [x] The landing-page pipeline describes detect → features + region probes → face guard / refusal, with the owner's chosen wording (decision 3). No step claims a calibration the model doesn't do.
- [x] The "~30%" back claim is gone or replaced (decision 4).
- [x] `npx eslint src`, `npx tsc --noEmit` and `npm run build` pass. No change outside `frontend/src` except this spec.
- [x] Checked in the running app, once with a real grade at 375 px and desktop width: nothing overflows and the new captions are readable.

## Owner questions

1. **The region cells' heading and caption.** Drafts:
   - (a) Heading "Region read-outs", caption "What each region suggests about the overall grade. Not PSA sub-grades." Centering stays in the same row.
   - (b) Keep "Sub-grades" and add that caption.
   - (c) Split: Centering under "Measured", corners / edges / surface under "Region read-outs" with the caption.
   - Recommended: (a). It's the smallest change that stops the PSA reading.
2. **Confidence caption.** Drafts:
   - (a) "probability of this exact grade". This is what the number is.
   - (b) "matches accuracy on held-out PSA certs (±5%)". This is true today, but drifts as the model changes.
   - Recommended: (a).
3. **Landing-page pipeline (3 steps).** Drafts:
   - 01 "Detect & dewarp": "Finds the card outline (slabs too), corrects the perspective and warps it to 600 × 840. No outline, no grade: you're asked to retake the photo."
   - 02 "Read the regions": "A frozen DINOv2 vision model describes 11 regions per face: corners, edges, centre and the whole card. Small probes trained on PSA-graded cards turn each into a grade signal."
   - 03 "Check and combine": "A guard checks the front is a front, the back a back, and the front is the right way up. A final probe combines the region signals into the grade distribution. Centering is measured separately from the borders." It doesn't promise a check on the back's orientation, because `back_upright` is disabled.
   - Recommended: these three, edited as you like.
4. **The "Back contributes ~30%" tip.** Recommended: replace it with "Front and back are both required; the model reads both." Or drop the sentence.

## Owner decisions (2026-10-30)

1. **(a)** Heading "Region read-outs", caption "What each region suggests about the overall grade. Not PSA sub-grades." Centering stays in the same row.
2. **(a)** Confidence caption "probability of this exact grade".
3. **The drafts** for the three pipeline steps, which may be tightened for length but keep their claims.
4. **Replace** the back tip with "Front and back are both required; the model reads both."
The intro and the meta description follow from these: no "scores it on corners, edges and surface", and "calibrated" is dropped.

## Implementation (2026-10-30)

- `GradeResult.tsx`: heading "Region read-outs" with the caption under it, in the same row of cells. Confidence caption "probability of this exact grade". Distribution caption "cumulative output of the ordinal grade head".
- `page.tsx`: the intro ("reads its corners, edges and surface, measures centering … with its confidence"), the three pipeline steps as drafted, and the "Both sides" tip.
- `layout.tsx`: meta description "… a predicted PSA grade with its confidence."
- Checks: `npx eslint src`, `npx tsc --noEmit` and `npm run build` pass.
- Real app: the probe API on :8000 and `npm run dev` on :3000, driven by headless Edge over CDP. Uploaded cert 76000010 (PSA 9, read from `data/images`), which graded 10, soft 9.51, confidence 60.7%. Screenshots of the landing page and the result at 375 px and 1280 px show no horizontal overflow, and every caption reads on screen.

## Re-test (2026-10-30)

Every acceptance criterion passed. A fact check of each new claim against the backend found them true for the probe path. The tester also found copy that is still false, mostly outside the original list. The owner chose to fix the copy in this PR:

- **D1, framing tip** (`page.tsx`): "Fill the frame with the card" led straight to a 422. `preprocess._candidates` discards outlines within 1.2% (at least 3 px) of the image edge, so a card that fills the frame isn't found. Now: "Show the whole card with a little background around it; cropping is automatic."
- **D2, navbar** (`Navbar.tsx`): "warming up" for `model_loaded: false` was wrong. The model loads in the lifespan and fails closed, so waiting never makes it ready. Now: "model unavailable".
- **D3, intro:** it implied centering feeds the grade. Now: "… reads its corners, edges and surface and returns a predicted grade with its confidence; centering is measured separately."
- **D4, step 02:** "each" read as one probe per region. Now: "Four small probes … turn the regions into grade signals" (four region-group probes over both faces).

After these edits, `npx eslint src`, `npx tsc --noEmit` and `npm run build` pass again.

Left for later (`smallfix.md`):
- **D5:** the distribution chart highlights the bar ">grade_int−1", i.e. P(grade ≥ grade_int), right under "probability of this exact grade". The two numbers match only for a 10. Its heading "Distribution over grades" also names a cumulative chart.
- **D6:** the backend hint strings ("clean" … "heavy wear") still read as physical wear. This spec keeps them out of scope.
- At 320 px, "generated · date" wraps into "PREDICTION" in the result header. This predates the change.

## Facts vs assumptions

- **Fact:** the probe head has no temperature step and no front/back weight (`grade_head.json`, CLAUDE.md "Export").
- **Fact:** `rank_probs` from the probe is P(grade > N) for N = 1..9, the same shape the distribution chart already plots.
- **Fact:** the test-split ECE is 0.053, with mean confidence 0.628 against accuracy 0.669 (`test-artifacts/eval/guard_noquad/report.json`).
- **Assumption:** nobody serves the legacy model to real users, so its copy can go. If `ALLOW_LEGACY_MODEL=1` is used in production, the new copy will be wrong for it.
