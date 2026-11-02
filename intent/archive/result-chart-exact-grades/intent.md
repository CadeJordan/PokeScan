# Result chart shows the chance of each grade

## Intent

Make the result card's distribution chart agree with the confidence number above it. Today the chart plots the chance of getting *more than* each grade and highlights P(grade ≥ predicted grade), right under a confidence captioned "probability of this exact grade". Also fix the 320 px header wrap on the same card. Frontend only, its own small PR (owner decision Q10, [[cv-ml-improvement-plan]]). Follows up D5 and the header item from [[result-copy-matches-probe-model]] (`smallfix.md`, "Result card follow-ups").

## Why

`GradeResponse.rank_probs` is the cumulative P(grade > k) for k = 1..9 (`backend/app/api/grade.py:61`). `frontend/src/components/GradeResult.tsx:116-143` draws one bar per k, labelled ">k". It highlights the bar where `i + 2 === grade_int`, which is ">grade_int−1" = P(grade ≥ grade_int).

`confidence` is P(grade = grade_int) (`backend/app/ml/grade_head.py::decode`). So the highlighted bar and the confidence above it are two different numbers, and they agree only for a predicted 10. A worked example, a borderline 9/10 card:

| | 8 | 9 | 10 |
|---|---|---|---|
| P(exact) | 0.15 | 0.45 | 0.40 |
| What the chart shows | ">7": 1.00 | ">8": **0.85** (highlighted) | ">9": 0.40 |

The card says "9, confidence 45%", and the highlighted bar beneath it says 85. The heading "Distribution over grades" reads as per-grade probabilities, which the bars aren't.

A second, smaller issue: at 320 px, "generated · <date>" in the header strip wraps into "PREDICTION". The date is `new Date()` at render time, i.e. the date in the viewer's browser when the result was shown.

## Scope

- `GradeResult.tsx`: the distribution chart, its heading and caption (owner decision 1), and the header strip (decision 2).
- The per-grade conversion is a pure function in its own module, e.g. `frontend/src/lib/gradeProbs.ts`. From `rank_probs` (length 9) it returns 10 probabilities: P(1) = 1 − P(>1), P(k) = P(>k−1) − P(>k), P(10) = P(>9). It clips negatives to 0 and renormalises, the same as `decode`. The highlighted bar is then the same number as `confidence`, up to the API's 4-decimal rounding.
- A test for that function, in whatever form decision 3 picks.

## Out of scope

- The API, the `GradeResponse` schema and `api-client.ts`. `rank_probs` stays cumulative; the frontend derives per-grade probabilities.
- How `grade_int` is chosen. It is the median of the distribution (1 + the number of P(>k) > 0.5), not the most likely grade. The chart keeps highlighting `grade_int`; see Facts vs assumptions.
- The backend region hint strings (`_REGION_HINTS`), which need a separate owner call (`smallfix.md`).
- The legacy model. Its `rank_probs` has the same cumulative meaning (after `cummin`), so the chart is correct for it too.
- Any other copy or visual redesign.

## Acceptance criteria

- [x] The chart has 10 bars labelled 1–10. Each shows P(grade = k) as a whole percent, and the bars sum to 100% (± rounding).
- [x] The highlighted bar is `grade_int`. Its value matches `confidence` to within 0.1 percentage points for the probe model's responses, so the two numbers on the card agree.
- [x] The chart heading and caption are the owner's chosen wording (decision 1). Nothing on the card describes the chart as cumulative.
- [x] At 320 px, 375 px and desktop width, the header strip stays on one line and the 10-bar chart doesn't overflow (decision 2).
- [x] The conversion function is tested (decision 3): a typical distribution; a predicted 10; a predicted 1; a non-monotone input (a negative difference is clipped and the rest renormalised); and a close 9/10 call where the highlighted bar isn't the tallest (P(9) = 0.40, P(10) = 0.45).
- [x] `npx eslint src`, `npx tsc --noEmit` and `npm run build` pass. No change outside `frontend/` except this spec, `smallfix.md` (tick the two items) and, if decision 3 picks it, the CI workflow.
- [x] Checked in the running app with a real graded pair (read from `data/images`) at 320 px, 375 px and desktop width, with the chart expanded: the highlighted bar equals the confidence, and nothing wraps or overflows.

## Owner questions

1. **What the chart shows.**
   - (a) **Exact grades.** 10 bars, one per grade, with the predicted grade highlighted. Heading "Chance of each grade". Caption: "The highlighted grade is the prediction: the middle of this distribution, so on a close call it can sit next to a slightly taller bar."
   - (b) **Keep the cumulative chart and relabel it.** Heading "Chance of at least each grade", bars labelled "≥2".."≥10", highlight "≥grade_int", and the caption says it isn't the confidence.
   - Recommended: (a). It's what "Distribution over grades" promised, and it makes the two numbers on the card the same number. (b) keeps a chart most users read wrong.
2. **The header strip at 320 px.**
   - (a) Hide the "generated ·" prefix below the `sm` breakpoint and keep the date.
   - (b) Drop the date. It's the browser's date at render time, so it never says anything the viewer doesn't know.
   - Recommended: (a), the smallest change. (b) is also fine if you'd rather lose it.
3. **How to test the conversion.** The frontend has no test runner today (only `eslint`).
   - (a) **Node's built-in runner, no new dependency.** A `gradeProbs.test.ts` run by `node --test --experimental-strip-types` (Node 22.14 here, 22 in CI), plus an `npm test` script and a CI step after lint. The module has to stay plain TypeScript with no `@/` imports.
   - (b) Add Vitest as a dev dependency, plus an `npm test` script and a CI step.
   - (c) No unit test. Rely on lint, build and the screenshot check.
   - Recommended: (a). It's a real test with no new packages, and it's cheap to switch to Vitest if the frontend grows more logic.

## Owner decisions (2026-10-31)

1. **(a) Exact grades.** 10 bars, the predicted grade highlighted, heading "Chance of each grade", with the drafted caption (it may be tightened for length but keeps its claim).
2. **(a)** Hide the "generated ·" prefix below `sm` and keep the date.
3. **(a)** Node's built-in runner: `node --test --experimental-strip-types`, an `npm test` script and a CI step after lint. No new dependency.

## Facts vs assumptions

- **Fact:** `rank_probs` is P(grade > k) for k = 1..9 on both the probe and legacy paths (`grade.py:61`, `inference.py:84`). `confidence` is the per-grade probability of `grade_int`, after clip and renormalise (`grade_head.decode`).
- **Fact:** `grade_int` is the median grade, so on a close 9/10 call the predicted grade's bar can be shorter than its neighbour's (the worked example above). The eval reports don't save per-cert probabilities, so how often this happens isn't measured. If it matters for the caption, count it on train+val out-of-fold predictions, not the test split (test-set policy).
- **Fact:** the frontend has no test runner (`frontend/package.json`); CI runs `npm ci`, `npm run lint` and `npm run build` on Node 22 (`.github/workflows/ci.yml`).
- **Assumption:** 10 columns fit at 320 px. The chart's content width is about 280 px, so each column is about 22 px after the gaps, and "100" at 10 px monospace is about 18 px wide. The screenshot check confirms it.

## Implementation (2026-10-31)

**What changed**

- `frontend/src/lib/gradeProbs.ts` (new): `perGradeProbs(rankProbs)` turns the 9 cumulative P(grade > k) into 10 P(grade = k): differences of `[1, ...p]` and `[...p, 0]`, negatives clipped to 0, renormalised (same as `grade_head.decode`, checked against it in Python on the non-monotone and close-call inputs). Plain TypeScript, no `@/` imports.
- `frontend/src/lib/gradeProbs.test.ts` (new): `node:test` + `node:assert`, 7 tests: a typical distribution, a predicted 10, a predicted 1, a non-monotone input (clipped and renormalised by 1.05), the worked borderline 9/10 example (highlighted 0.45, where the old chart showed 0.85), a close 9/10 call where the median 9 (0.40) sits next to a taller 10 (0.45), and a real `/api/grade` response (cert 76000632) where the highlighted value is within 0.1 pp of `confidence`.
- `frontend/src/components/GradeResult.tsx`: 10 bars labelled 1–10, each P(grade = k) as a whole percent (bar height on the same absolute 0–100% scale as before), `grade_int` highlighted. Heading "Chance of each grade"; caption is the drafted one with "slightly" dropped (see Notes): "The highlighted grade is the prediction: the middle of this distribution, so on a close call it can sit next to a taller bar." The cumulative caption and ">k" labels are gone. Header strip: "generated ·" is wrapped in `hidden sm:inline`, the date stays. The bar grid's gap is `gap-1` below `sm` (`sm:gap-1.5` as before), so a "100" (18 px at 10 px monospace) fits a column at 320 px (19.4 px; 17.6 px with the old gap).
- `frontend/package.json`: `"test": "node --test --experimental-strip-types \"src/**/*.test.ts\""` (Node expands the quoted glob itself, on cmd.exe and sh alike).
- `frontend/tsconfig.json`: `"allowImportingTsExtensions": true`. Node's type stripping needs the explicit `./gradeProbs.ts` import, which `tsc --noEmit` otherwise rejects (TS5097); allowed because `noEmit` is set.
- `.github/workflows/ci.yml`: a "Unit tests" step (`npm test`) after ESLint in the frontend job. The job name is unchanged, in case it is a required check.

**Checks** (from `frontend/`, Node 22.14): `npx eslint src` clean; `npx tsc --noEmit` clean; `npm run build` passes; `npm test` 7 passed, 0 failed (Node prints its ExperimentalWarning for type stripping and a MODULE_TYPELESS_PACKAGE_JSON warning, since `package.json` has no `"type"`; both harmless).

**Running app** (API `uvicorn backend.app.main:app --host 127.0.0.1 --port 8000`, probe model; `npm run dev`; no `frontend/.env.local`, so the default `http://localhost:8000/api`; headless Edge over CDP, uploading through the page's file inputs; screenshots in `test-artifacts/screens/result-chart-exact-grades/`):

- Cert 76000620 (Heracross holo, PSA 8), at 320, 375 and 1280 px: `grade_int` 9, `confidence` 0.4149, `rank_probs` [0.9998, 0.9994, 0.9971, 0.9942, 0.9705, 0.9294, 0.8412, 0.5379, 0.123]. Bars 1–10: 0 0 0 0 2 4 9 30 **41** 12, 9 highlighted; P(9) = 0.5379 − 0.123 = 0.4149, exactly the confidence (shown as 41.5%). The old chart would have highlighted 54 (P(grade ≥ 9)). Header "PREDICTION … 2026-10-31" on one line at 320 and 375 px ("generated · 2026-10-31" at 1280 px); page `scrollWidth` equals the viewport at every width; no element of the card extends past its right edge.
- Cert 76000632 (Dark Charizard holo, PSA 7), at 375 px, a real close call: `grade_int` 6, `confidence` 0.2227, bars 1 1 5 6 31 **22** 17 13 4 1. The highlighted P(6) is 0.2226 before rounding (0.01 pp from the 4-decimal confidence), next to a taller 5 (31%).

**Notes for the owner**

- The whole-percent bars don't always sum to exactly 100 (98 for 76000620, 101 for 76000632): plain rounding, kept so the highlighted bar is the confidence rounded, not nudged.
- The drafted caption said the neighbour can be "slightly" taller. In 76000632 it is 31% against 22%, so "slightly" was dropped (the owner decision allows tightening as long as the claim stays).
- The worked example in Why had ">7" as 0.85; it is 1.00 (P(8) + P(9) + P(10) = 1), now corrected. In that example the highlighted 9 is also the tallest bar, so the "isn't the tallest" case is tested with P(9) = 0.40, P(10) = 0.45 and the real 76000632 response.

## Re-test (2026-10-31)

Pass, no functional defects. `perGradeProbs` matches `grade_head.decode` to 2.2e-16 on 6,006 inputs (monotone, non-monotone, out-of-range, legacy-style TTA means, 4-dp rounded). Over 200k simulated responses the highlighted bar is within 0.01 pp of `confidence`. Mutants that drop the clip, the renormalise or shift the index all fail the suite. The tester added two tests (out-of-range inputs differenced as given; all-zeros / all-ones), so `npm test` runs 9. Checked live with cert 76000011 (PSA 10) and stubbed grade-10 / grade-1 responses at 320, 375 and 1280 px: no overflow, header on one line.

Nits, left as they are:
- The bar is a whole percent and the confidence one decimal, so 41.5% shows a bar of 41 (as specified).
- `node --test` with a glob that matches nothing exits 0, so moving the test files out of `src/**/*.test.ts` would make the CI step pass with zero tests.
- At 320 px a "100" label has about 1.4 px to spare in its column; a larger font or padding would overflow it.
- For the served head the tallest bar is never more than one grade from the median, so "next to a taller bar" holds today; it isn't guaranteed by construction.
