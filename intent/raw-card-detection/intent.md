# Raw-card detection: keep a raw card's outline, not its art window

## Intent

Make `preprocess._find_card_quad` keep the outline of a raw (unslabbed) card whose plain, flat, dark border surrounds a framed, textured art window. Today the last-resort flat-ring step takes that border for a slab gasket and crops to the window (IoU ~0.81 with the card). Do it without changing how PSA slab photos are cropped, so the served grades on slab photos, the precrops, the probe and the face guard all stay as they are.

This is the follow-up deferred in [[serving-hardening]] section 3 and item 8 of `intent/cv-ml-improvement-plan.md`. The owner chose (2026-11-02) to start it before the phone-photo set exists.

## Background (facts, from [[serving-hardening]] section 3)

- `test_raw_card_with_flat_border_keeps_its_outline` (`backend/tests/test_preprocess.py`) has 6 synthetic raw cards: a flat dark border (luma 40–120), a textured window of similar brightness, on a plain light background. The 5 with a thin frame line around the window fail (non-strict xfail). The one with no frame passes.
- The failure happens at the **first** step: `best` is the card, the framed window is a strong nested outline at ~0.81 of its area with the card's aspect, and `_is_holder(..., allow_flat=True)` accepts the card's own flat border as a gasket.
- `silver_full_art_in_grey_well` (`test_slab_photo_finds_the_card`) needs exactly that step: `best` is the grey well, the ring is a flat gasket, the band is the card's textured silver border.
- Constraints on the step itself don't separate the two (only at the first step, only past it, a height rule on `best`): each either keeps the slab scenes and fails the raw cases, or the reverse.
- On all 5,260 PSA faces, the last-resort step decides 84 faces, 65 of them at the first step, with `best` at 0.649–0.768 of the photo height.

## Approach

The local pattern (outline, flat ring, textured nested outline) is the same in both scenes, so the fix has to look at **context the slab has and a raw card doesn't**. Candidates to measure, in this order:

1. **Slab evidence outside `best`.** A PSA photo has the slab around the well: the label (a bright, mostly white rectangle with red/colour, above the card on the front, smaller on the back), the slab's own edges, the bright lip around the well. A raw phone photo has a table or a hand. Allow the last-resort step only when there's slab evidence (e.g. a strong outline or a label-like region outside `best` and above or below it).
2. **Corner shape.** A card's outer corners are rounded; a printed art window's corners are square. If `best` has rounded corners and the nested outline has square ones, `best` is the card. Unknown whether PSA wells' corners are round too: measure it.
3. Anything else the developer finds, measured the same way.

Pick the simplest rule that meets the gates. Don't add a learned model.

## Gates (in order; stop at the first that fails and report)

1. **Synthetic:** the 5 framed raw-card cases pass (drop their xfail), the no-frame case still passes, and all of `test_slab_photo_finds_the_card` and the rest of `test_preprocess.py` pass, including the existing xfails' status.
2. **New negative tests** for the rule itself, so it can't pass by accident: a raw card next to a white sticky note or paper (a label-like region, if rule 1 is used), and a slab scene with the label cut off by the photo edge.
3. **PSA quads:** run the detector over all 5,260 PSA faces (read only, like `flat_diag.py` in [[serving-hardening]]) before and after. Gate per Q1. `test_stored_precrops_are_reproduced_byte_identical` passes.
4. **Real raw photos,** if the owner supplies them (Q2): each one's crop keeps the card's outline (overlay checked by eye, saved to `test-artifacts/raw_card_detection/`).
5. Fast suite passes; `ruff check .` clean; the baseline in CLAUDE.md updated.

## Scope

- `backend/app/ml/preprocess.py` (`_find_card_quad` and helpers).
- `backend/tests/test_preprocess.py`.
- A read-only before/after diff script (scratch, or under `ml/eval/` if worth keeping).
- This spec's Results section; the plan's item 8 note; CLAUDE.md's inference-path paragraph if the detector's described behaviour changes.

## Out of scope

- Retraining the probe or the face guard, `precrop --force`, or any write under `data/` (unless Q1's answer allows it and the owner approves the run).
- Other raw-card failure modes (glare, sleeves, top-loaders, hands, busy backgrounds). Those wait for the phone-photo set.
- Frontend or API changes.

## Assumptions (unverified)

- How often this happens on real raw uploads is unknown. The window must be card-shaped (aspect ≥ 0.65, `_AR_MIN`) and at least half the card's area (`_NEST_MIN`) to be a candidate, and most Pokémon art boxes are landscape, so it may be rare (full-art and some trainer layouts are the likely cases). The phone-photo set will tell.
- PSA photos always include some slab context around the well. To be checked on the 84 faces the last-resort step decides.

## Acceptance criteria

- [ ] The 5 framed raw-card cases pass without xfail; the no-frame case passes.
- [ ] The new negative tests exist and pass.
- [ ] PSA quads: per Q1, with the before/after counts in Results.
- [ ] Real raw photos (if supplied): crops keep the card outline; overlays saved.
- [ ] Fast suite and lint pass; CLAUDE.md baseline updated.
- [ ] Results written below; plan item 8 updated.

## Questions for the owner

1. **PSA gate:** must all 5,260 PSA quads stay identical? Or may a few change if each is checked by eye and is no worse (which then needs `precrop --force`, a probe retrain, a face-guard retrain and a new test number)? *Recommendation:* identical. If no rule manages that, stop and ask.
2. **Real raw photos:** can you take ~10 phone photos of raw cards (any you have, ideally silver/grey or dark-bordered ones and full-arts) on a plain table and put them in `data/raw_card_check/`? *Recommendation:* yes; it's the only real check, and 10 photos is a few minutes. Optional: without them, synthetic gates only.
3. **If the slab-evidence rule (1) works but needs a new tuned threshold,** tune on the synthetic scenes plus the PSA faces (detection only, no grades)? *Recommendation:* yes; it isn't model selection on the test split.
4. **Where do results go?** *Recommendation:* in this spec, as for the earlier ones.

## Owner decisions (2026-11-02)

1. **PSA gate: identical.** All 5,260 PSA quads must stay the same. If no rule manages that, stop and ask.
2. **Real raw photos: yes.** The owner will add ~10 to `data/raw_card_check/` (owner-written; the code only reads it). The synthetic work starts now; gate 4 runs once the photos are there.
3. **Tuning:** allowed on the synthetic scenes plus detection-only runs over the PSA photos.
4. **Results:** in this spec.

## Results

(pending)
