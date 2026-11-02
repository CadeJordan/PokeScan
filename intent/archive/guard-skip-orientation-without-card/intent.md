# Face guard: skip the orientation check on faces with no card

## Intent

Make the [[face-side-orientation-guard]] leave the orientation alone on a face where no card outline was found (`found_quad=False`). Those faces are graded on the padded whole photo, which the upright probes never trained on, and that's where every false rotation measured so far comes from. Follow-up to [[face-guard-serve-path-features]].

## Why

Measured 2026-10-29 (guard_served on the card-grouped test split, see [[face-guard-serve-path-features]] Results):

- **All 3 false front rotations on clean test certs are no-quad fronts**: 76000413, 88000652 and 88001436, with P(flip) 0.999, 1.000 and 1.000. On the 370 test certs where a card was found, the guard rotated 0 upright fronts, as CV predicts.
- **One of them changes a grade.** 88001436 (PSA 8) goes 8 → 7 (7.357 → 6.962) with a wrong "front photo was upside down" note.
- **The probes never saw this input.** Guard training rows come from `train_probe.load_rows`, which only holds certs with a usable precrop, so no fallback faces.
- **`/api/grade` already refuses these uploads** with 422 `card_not_found` ([[reject-unusable-uploads]]). So the wrong rotation never reaches a response today. It does show up in `ml.eval` (which grades refused inputs too), in direct `GradeInferenceService.predict` callers, and it would reach users if the 422 were ever relaxed.

## Scope

- In `_predict_probe` / `FaceGuard.decide`, don't rotate a face whose `CardCrop.found_quad` is False. The check applies to the face as graded, after any swap.
- Still compute and record P(flip) for that face in `GuardDecision.p_flip`, so `ml.eval` keeps reporting it.
- Don't add a "photo was upside down" note for a face that wasn't rotated.
- Re-run guard_served's eval as `guard_noquad` and compare.

## Out of scope

- Training the guard on fallback faces.
- Changing the side/swap check (owner decision 2).
- Changing the 422 rejection or the card detector.
- Enabling `back_upright`.

## Acceptance criteria

- [x] A face with `found_quad=False` is never rotated by the guard. When the guard swaps the faces, the check uses the swapped face's `found_quad`, not the uploaded slot's.
- [x] A face with `found_quad=True` is decided exactly as before: same `flip`, notes and P(flip) for the same features.
- [x] Without a guard, or with the orientation checks disabled, predictions are byte-for-byte unchanged.
- [x] `/api/grade` responses are unchanged. Every input this affects is already a 422.
- [x] Tests: a unit test on `decide` for each face × quad combination, including a swap where the quad flags cross over. A service-level test that a no-quad front with P(flip) above the threshold isn't rotated and gets no note.
- [x] `guard_noquad` on the test split (same command as guard_served, out dir `test-artifacts/eval/guard_noquad`): clean fronts rotated 3 → 0, 88001436 back to 8, and no other clean `pred_int` changes against guard_served. Report the `rotate_180` and `swap_faces` numbers next to guard_served's, including how many perturbed faces had no quad and so went uncorrected.
- [x] README face-guard section, CLAUDE.md inference path and this spec's Results updated.

## Owner decisions (2026-10-30)

1. **Both faces.** The rule applies to the front and the back, so it still holds if `back_upright` is enabled later.
2. **Keep the side check.** No change to the swap. `guard_noquad` reports P(back) on the 5 no-quad test certs.
3. **No note.** The skip is recorded in `GuardDecision` only, and `notes` doesn't mention it. These uploads are refused with 422 anyway.

## Implementation (2026-10-30)

- `FaceGuard.decide(feats, found_quad=(True, True))` takes `found_quad` per uploaded row and swaps it with the rows. `check_orientation` never flips a face whose check is enabled but whose `found_quad` is False, adds no note for it, and still returns its P(flipped). `GuardDecision.no_card` is (front, back) after the swap: True where an enabled orientation check was skipped for lack of a card. The default keeps every existing caller unchanged.
- `_predict_probe` passes `(front_crop.found_quad, back_crop.found_quad)`.
- Tests (`backend/tests/test_face_guard.py`, new section "No card outline"): `decide` for each face × quad, quad flags crossing over on a swap, `no_card` only for enabled checks, the default equals (True, True). Service level: a no-card front read as flipped isn't rotated (one extraction, no note, P(flip) kept), a no-card back doesn't stop the front being rotated, and a no-card back swapped into the front slot isn't rotated.
- Two older tests pinned the rotation of a no-card face and were updated: `test_retest_reject_uploads.py::test_swap_then_flip_still_names_the_uploaded_face` and `test_retest_reject_uploads_exif.py::test_a_rotated_fallback_face_is_still_card_not_found` (renamed `test_a_fallback_face_read_as_flipped_is_still_card_not_found`). Both still check the 422. Four `decide` mocks that raise were given the new signature, so they raise their own exception rather than a TypeError.
- Fast suite: 1413 passed, 1 skipped, 2 xfailed (1403 before, plus 10 new tests). Ruff clean.
- Re-test 2026-10-30: no defects. `backend/tests/test_retest_guard_noquad.py` (60 tests) covers: the full service matrix (which face has a card × swap × which faces read as flipped); decisions on faces with a card are identical to `df2a561`'s `face_guard.py` for random guards and features; `/api/grade` returns the same 422 with or without the guard; and `found_quad` given as a list, numpy bools or ints. Fast suite 1473 passed, 1 skipped, 2 xfailed.

## Results (2026-10-30)

`python -m ml.eval.evaluate --split test --robustness-limit 100 --perturbations rotate_180,swap_faces --out-dir test-artifacts/eval/guard_noquad`, same guard and head as guard_served, CPU provider, "regressions vs baseline: none":

| | guard_served | guard_noquad |
|---|---|---|
| clean MAE / QWK / exact (n 375) | 0.437 / 0.492 / 66.7% | 0.435 / 0.491 / 66.9% |
| clean fronts rotated | 3 | **0** |
| clean `pred_int` changed vs guard_served | | 1 (88001436, 7 → 8) |
| clean `pred_soft` changed vs guard_served | | 3 (the three no-card certs, back to their unguarded values) |
| `rotate_180` flip rate / soft shift / fronts rotated (n 100) | 4.0% / 0.0991 / 93 | 4.0% / 0.0990 / 93 |
| `swap_faces` flip rate / soft shift / swapped back / fronts rotated (n 100) | 0.0% / 0.000 / 100 / 1 | 0.0% / 0.000 / 100 / 0 |
| `n_would_reject` card_not_found / same_side | 5 / 0 | 5 / 0 |

- **Clean grades match guard_before on the no-card certs.** 76000413 9.520, 88000652 9.658, 88001436 7.357: the unguarded values from the guard_served table.
- **No `rotate_180` correction was lost.** The per-cert notes and grades are identical to guard_served's. The soft-shift change comes only from the clean reference of 76000413, whose rotated photo does get a card outline and is still corrected. No-card perturbed faces left uncorrected: `rotate_180` 0 (the 7 uncorrected fronts all have P(flip) 0.13–0.988, below 0.99, so they weren't rotated before either); `swap_faces` 1, 76000413, and that was a false rotation, not a lost correction.
- **`swap_faces`:** the only change is 76000413, whose false front rotation after the swap is gone (9.5085 → 9.5198).
- **Side check on the 5 no-card test certs (decision 2).** P(back) per uploaded row: 76000413 (0.000, 1.000), 88000652 (0.000, 1.000), 88001641 (0.000, 1.000), 76033476 (0.000, 1.000), 88001436 (0.321, 0.984). The four with one no-card face read correctly. 88001436 (both faces no card) has an uncertain front, below the 0.9 threshold, so no swap or note. 0 false swaps, 0 same-side, so keeping the side check is supported.
- P(flip) on the no-card fronts is still reported: 76000413 0.999, 88000652 1.000, 88001436 1.000, 88001641 0.978, 76033476 0.980. The last two are below 0.99 and weren't rotated before either.

## Facts vs assumptions

- **Fact:** the 3 clean false rotations and their P(flip) values (guard_served report, reproduced in a separate ONNX run).
- **Fact:** `CardCrop.found_quad` is set in `preprocess.py` and reaches `_predict_probe` for both faces.
- **Assumption:** some `rotate_180` perturbations also land on no-quad faces, so the rule may cost a few true corrections there. `guard_noquad` measures it.
  - **Checked 2026-10-30:** no correction lost; the `rotate_180` notes and grades are identical to guard_served's.
