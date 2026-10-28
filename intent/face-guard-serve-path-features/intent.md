# Face guard on served-path features

## Intent

Train the [[face-side-orientation-guard]] probes on features computed the way `/api/grade` computes them, so the guard's false-rotation rate on real uploads matches what cross-validation predicts. Then reconsider the disabled back upright check.

## Why

Measured 2026-10-28 (guard_after on the card-grouped test split, ADR 0002):

- **CV under-predicts served false rotations.** `front_upright` at 0.99 rotated 3 of 375 upright served fronts (0.8%). Out-of-fold on precrops it was 0.18%, and on test precrops 0. One of the 3 changed a grade (cert 88001436, 8 → 7), and all 3 got a wrong "front photo was upside down" note.
- **The probes are trained on precrop features, not served ones.** The served path decodes the upload, applies the EXIF transpose and the 1024 px cap, re-detects, and runs `_jpeg_roundtrip` before extraction. The precrop cache skips the re-detection and the cap.
- **The back check is off.** `back_upright` has CV AUC 0.998 but clean-face FPR 0.223% at 0.99, against a 0.2% limit, so it was disabled. On the 100 `rotate_180` test certs, P(flip) was ≥ 0.998 for every rotated back. A perfect back check would bring the `rotate_180` shift from 0.100 to 0.046.

## Scope (to be refined with the owner)

- Extract guard training features through the served preprocessing path, upright and 180°-rotated, for train+val. Cache them beside the existing caches, keyed the same way.
- Refit all three probes and re-run the gates on those features. Thresholds are chosen on card-grouped CV only.
- Re-measure guard_after, comparing against the guard_before and guard_after reports from 2026-10-28.

## Out of scope

- Changing the grade head or its features.
- Rejecting uploads ([[cv-ml-improvement-plan]] open question 2).
- 90° orientation.

## Open questions

1. Compute: one more GPU extraction, about 10.4k faces (upright and rotated, served path). Approve?
2. Should the back check's gate stay at 0.2% FPR measured on served-path features, or change?
3. Should a guard rotation be recorded as a note only, or also lower the grade's confidence?

## Facts vs assumptions

- **Fact:** the 2026-10-28 numbers above. Per-cert breakdown: `test-artifacts/eval/guard_after/rot180_breakdown.json`.
- **Assumption:** the gap between CV and served false rotations comes from the precrop vs served-path feature difference, not from test-set noise. 3 of 375 is a small count, so a first step is to measure the served FPR on train+val out of fold.
