# CORN decode verification

## Intent

Settle, with a test, whether `decode_corn_logits` interprets the head's outputs the same way `corn_loss` trains them. If it doesn't, document the impact on the soft grade and confidence before anyone decides to change it.

This is a verification spec, not an architecture change.

## Why

- `corn_loss` (`backend/app/ml/model.py`) masks threshold *k* to samples whose rank already exceeds *k−1*. That's the CORN formulation, where `sigmoid(logit_k)` is the **conditional** P(y > k | y > k−1), and the unconditional P(y > k) is the cumulative product.
- `decode_corn_logits`, `calibrate.py` and the API instead treat `sigmoid(logit_k)` as unconditional and apply `cummin`.
- The **hard rank agrees** either way (`calibrate.py:133` notes the equivalence via `cumprod` of `p > 0.5`).
- The **soft expected grade** (`1 + sum(p)`), **per-class confidence** and **temperature calibration** can differ. The suspected direction: `cummin` overstates P(y > k) for high *k*, which inflates the expected grade and skews confidence. That matters because the API returns the soft grade as `grade`.
- The model.py comment also says outputs are "guaranteed monotonic via softplus", but there is no softplus in the code.

## Scope

- Add a test using fixed logits that compares the `cummin(sigmoid)` and `cumprod(sigmoid)` decodings and documents where they diverge.
- Run both decodings over the val split's saved logits (via `ml/eval/evaluate.py`) and report MAE, QWK and ECE for each, with no change to the model or the API.
- Write the conclusion as an ADR in `docs/adr/`, either "keep cummin, here's why" or "switch, here's the expected effect".

## Out of scope

- Changing the decode in served code, retraining or recalibrating. Any of those would be a follow-up spec gated on this one's ADR.

## Acceptance criteria

- [ ] A unit test shows both decodings give the same hard rank on random logits, and quantifies the soft-grade gap.
- [ ] An eval report (under `test-artifacts/eval/`) compares the two decodings on val, using the current active model.
- [ ] An ADR records the decision and its reasoning.
- [ ] No change to `backend/app/ml/inference.py` output in this spec.

## Facts vs assumptions

- **Fact:** the loss masking and the `cummin` decoding as read in code on 2026-10-17.
- **Assumption:** the network has learned conditional probabilities because of the masked loss. This should hold, but the size of the effect is unknown until measured.
