# Stable, card-grouped split

## Intent

Two phases:

1. **Sticky (do now).** Re-running `python -m ml.data.split` after a harvest never moves an already-assigned cert. New certs of a known printed card join that card's split; new cards are assigned by a deterministic hash of their card key.
2. **Grouped (only with owner approval).** A one-time re-assignment puts every printed card entirely in one split, so the test number measures generalisation to unseen cards. The old assignment is kept for reference.

## Why

Measured 2026-10-23 on the real rows (read-only; see [[cv-ml-improvement-plan]] items 4 and 9):

- `_stratified_assign` shuffles each grade's full cert list with a fixed seed (`ml/data/split.py:29-52`), so adding one cert re-deals the whole grade. Re-running on the current rows reproduces the DB (2630/2630), but adding PSA certs moves existing ones:

  | Added PSA certs | Existing certs that change split | Leave test | Enter test |
  |---|---|---|---|
  | +1 | 56 | 12 | 12 |
  | +50 | 831 | 253 | 258 |
  | +200 | 987 | 292 | 293 |

- The README says "eval numbers stay comparable across harvests". That's true for eBay rows only. The PSA harvest is ongoing (`HANDOFF.md`).
- Certs that leave test enter train, so a re-split followed by a retrain trains on the previous model's test certs. `--save-baseline` comparisons then compare different test sets.
- The split is per cert: 251 of 388 test certs share a printed card with train+val. Test MAE is 0.414 on seen cards and 0.628 on unseen cards (CV is already card-grouped in `ml/training/probe.py`).
- Simulated hash-on-card-group assignment (15% test, four salts) gives 341–446 test certs, 49–61 of them graded ≤8 (currently 50), so pure hashing keeps enough low grades. Only 54–63 of today's 393 test certs would stay in test.

## Scope

### Shared group key: `ml/data/groups.py` (new)

- `card_group_key(brand, subject, number) -> str` returns exactly what `ml.training.probe.card_groups` builds today: `f"{brand or ''}|{subject or ''}|{number or ''}"`.
- `card_groups` in `probe.py` calls it, so CV groups are unchanged.
- No normalisation of subject variants in this spec. 5 brand+number keys have variant subject spellings, e.g. `CHARIZARD-HOLO` / `CHARIZARD-REV.FOIL`. They're left as they are.
- `group_bucket(key, salt="pokescan-split-v1") -> float` in [0, 1): `int(sha1(f"{salt}|{key}")[:8], 16) / 2**32`.

### `ml/data/split.py`: modes

`assign_splits(rows, ...)` keeps its signature and behaviour as the pure function behind `--mode stratified`. Add pure functions for the new modes:
- `assign_sticky(rows: list[SplitRow], val_frac, test_frac, salt) -> dict[str, str]`
- `assign_grouped(rows: list[SplitRow], val_frac, test_frac, salt) -> dict[str, str]`

`SplitRow` is a dataclass `(cert, grade, source, group, current_split)`. `main` reads `brand_title, subject, card_number, split` to build the rows. Each mode:

- `--mode sticky` (new default):
  - Rows with a non-null `split` keep it.
  - Unassigned PSA rows whose group has assigned PSA rows take the group's most common split. On a tie, test beats val beats train, so a straddling card doesn't add new test-to-train leakage.
  - Other unassigned PSA rows go to test if `group_bucket < test_frac`, to val if `< test_frac + val_frac`, else train.
  - eBay rows go to train, except that an eBay row whose group has a PSA test cert gets the new value `split='excluded'`. `excluded` is never read by `train_probe`, `dataset.load_records` or `evaluate`, because they all filter on train/val/test. Verify that.
- `--mode stratified --reassign`: today's behaviour, explicit.
- `--mode grouped --reassign` (phase 2):
  - Every PSA row is re-assigned by `group_bucket` of its group, with eBay handled as above.
  - Before writing, copy the current `split` into a new column `split_legacy`. Add the column via the migration in `ml/data/db.py`. Only fill it if it's null, so a second re-split doesn't overwrite the original.
- `--reassign` is required to change any existing assignment; without it, `stratified` and `grouped` refuse to run.
- `--dry-run` computes and prints, and writes nothing. Open the DB read-only in that mode.
- `--ebay-split stratify` keeps working only with `--mode stratified`.

### Logging and report

- After assignment, log: per-split grade histogram, the number of test certs whose group also appears in train/val (the "test straddle" count), and the number of rows that changed split.
- `train_probe` adds `test_seen_cards` (metrics on test certs whose group is in train+val, like `test_unseen_cards`) and `n_test_seen` to its report and to the head metadata.

### Docs

- Update the README "Collect training data" paragraph: sticky by default, and how phase 2 works.
- Add a line to CLAUDE.md that `split` is sticky and `--reassign` is a destructive, owner-approved action.
- Phase 2 only: a new ADR recording the new test set and the new baseline numbers.

## Out of scope

- Running phase 2 on the real DB without the owner's explicit approval ([[cv-ml-improvement-plan]] open question 1).
- Normalising card keys across variant spellings or languages.
- Any change to `train_probe`'s CV (already card-grouped) beyond the extra report fields.
- DB backups in general ([[data-backup-before-destructive-runs]]). Phase 2 must not run until that spec has landed or a manual backup has been taken (`sqlite3 data/pokescan.sqlite ".backup data/backups/pre-grouped-split.sqlite"`).

## Acceptance criteria

All tests build their own DB in `tmp_path` via `ml.data.db.connect`. None of them touches `data/`.

- [ ] Fast suite: `python -m pytest -m "not slow and not visual and not model"` → 0 failed, 2 xfailed, passed = baseline + the new tests. `ml/tests/test_split.py` passes **unchanged** (`assign_splits` is untouched).
- [ ] `ml/tests/test_split_sticky.py` (new):
  - [ ] Split 500 synthetic PSA certs (sticky), then add 1 and then 50 new certs and split again: **0** existing certs change split.
  - [ ] A new cert of a card already in test lands in test. Same for val and train.
  - [ ] Over 3,000 synthetic single-cert cards, the new-card test share is 15% ± 2 pp and the val share is 15% ± 2 pp.
  - [ ] An eBay cert of a card with a PSA test cert gets `excluded`. Other eBay certs get `train`.
  - [ ] Shuffling the input row order gives identical assignments.
  - [ ] `--mode grouped --reassign`: no group spans two of train/val/test; `split_legacy` equals the pre-run `split` for every row; a second run leaves `split_legacy` unchanged.
  - [ ] `--mode stratified` without `--reassign` exits non-zero and writes nothing.
  - [ ] `--dry-run` leaves the DB file's bytes unchanged.
- [ ] `card_groups` output is identical before and after the refactor: a test over synthetic rows, plus one read-only check over the real DB (`mode=ro`), marked `slow`.
- [ ] Real DB, phase 1 (read-only): `python -m ml.data.split --dry-run` prints 0 changed rows and a test straddle count of 251 (or the current value). Running it for real changes 0 rows.
- [ ] Phase 2, only after approval and a backup:
  - `python -m ml.data.split --mode grouped --reassign`, then `python -m ml.training.train_probe`;
  - the report shows `n_test_seen == 0`;
  - a new ADR records the new test n, grade histogram and metrics next to the old ones;
  - `python -m ml.eval.evaluate --split test --save-baseline` is re-run.
- [ ] `ruff check .` exits 0.

## Facts vs assumptions

- **Fact:** the instability numbers, the 251/388 seen-card test certs and the simulated grouped-split sizes above (2026-10-23). There are currently 0 eBay rows.
- **Fact:** `train_probe` fits on train+val together, so after phase 2 only the test assignment matters for the probe. Val still matters for the legacy CNN.
- **Assumption:** `split='excluded'` breaks no reader. Grep for `split` filters in `ml/` and `backend/`. `ml/data/db.py` helpers that count per split will show a new key.
- **Assumption:** a hash split without grade stratification keeps enough low grades in test. Simulated 49–61 ≤8 vs 50 today; check with `--dry-run` before phase 2.
