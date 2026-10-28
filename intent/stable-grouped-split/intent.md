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
- Next to it in CLAUDE.md, the test-set policy (open question 9, decided 2026-10-27): the test split gives one final number per ADR; all model selection (hyperparameters, features, thresholds) uses card-grouped CV on train+val.
- Phase 2 only: a new ADR recording the new test set and the new baseline numbers.

## Out of scope

- Running phase 2 on the real DB without the owner's explicit approval ([[cv-ml-improvement-plan]] open question 1).
- Normalising card keys across variant spellings or languages.
- Any change to `train_probe`'s CV (already card-grouped) beyond the extra report fields.
- DB backups in general ([[data-backup-before-destructive-runs]]). Phase 2 must not run until that spec has landed or a manual backup has been taken (`sqlite3 data/pokescan.sqlite ".backup data/backups/pre-grouped-split.sqlite"`).

## Acceptance criteria

All tests build their own DB in `tmp_path` via `ml.data.db.connect`. None of them touches `data/`.

- [x] Fast suite: `python -m pytest -m "not slow and not visual and not model"` → 0 failed, 2 xfailed, passed = baseline + the new tests. `ml/tests/test_split.py` passes **unchanged** (`assign_splits` is untouched).
  - 2026-10-27: 421 passed (398 + 23 new: 20 in `test_split_sticky.py`, 3 in `test_card_groups.py`), 2 xfailed, 184 deselected (183 + the new slow test), 110 s. `test_split.py` and `assign_splits` / `_stratified_assign` are unchanged.
- [x] `ml/tests/test_split_sticky.py` (new):
  - [x] Split 500 synthetic PSA certs (sticky), then add 1 and then 50 new certs and split again: **0** existing certs change split.
    - `test_adding_certs_never_moves_existing_ones` (pure function; the 50 mix known and new cards). `test_cli_sticky_rerun_changes_nothing` does the same through the CLI on a `tmp_path` DB (300 PSA + 40 eBay, then +50).
  - [x] A new cert of a card already in test lands in test. Same for val and train.
    - `test_new_cert_joins_its_cards_split`; `test_straddling_card_takes_majority_then_test_first` pins the majority rule and the test > val > train tie-break.
  - [x] Over 3,000 synthetic single-cert cards, the new-card test share is 15% ± 2 pp and the val share is 15% ± 2 pp.
  - [x] An eBay cert of a card with a PSA test cert gets `excluded`. Other eBay certs get `train`.
    - For both sticky and grouped. An eBay cert already in `train` whose card later gets a PSA test cert moves to `excluded` in sticky mode too (see Facts vs assumptions); eBay certs in any other split keep it (`test_assigned_ebay_keeps_its_split_in_sticky`).
  - [x] Shuffling the input row order gives identical assignments.
    - Sticky and grouped, 3 shuffles each, over assigned + unassigned PSA rows and eBay rows.
  - [x] `--mode grouped --reassign`: no group spans two of train/val/test; `split_legacy` equals the pre-run `split` for every row; a second run leaves `split_legacy` unchanged.
    - `test_cli_grouped_reassign_keeps_legacy` starts from a random per-cert split. "No group spans" holds for PSA rows. eBay rows go to train by the rule above, so a *val* card can have eBay train certs; the test asserts a test card has nothing in train or val. The second run uses another salt so it really moves certs, and `split_legacy` is still the original.
  - [x] `--mode stratified` without `--reassign` exits non-zero and writes nothing.
    - Exit 1, DB file hash unchanged; same for `--mode grouped`. `--ebay-split stratify` without `--mode stratified` is refused too (exit 2).
  - [x] `--dry-run` leaves the DB file's bytes unchanged.
    - Sticky and grouped dry runs over 340 unassigned rows: hash unchanged, all splits still NULL, log says "changed split: 340 of 340 rows".
- [x] `card_groups` output is identical before and after the refactor: a test over synthetic rows, plus one read-only check over the real DB (`mode=ro`), marked `slow`.
  - `ml/tests/test_card_groups.py`. The slow test (2026-10-27: passes, 3 s, all 7,429 rows) compares against the old f-string and also checks that `card_group_key` on the raw (NULL-able) columns, as `split.py` calls it, equals the CV groups cert for cert.
- [x] Real DB, phase 1:
  - [x] `python -m ml.data.split --dry-run` prints 0 changed rows and a test straddle count of 251 (or the current value). Stop here if it reports any changed row.
    - 2026-10-27: "changed split: 0 of 2630 rows (mode=sticky)", test straddle **255** of 393 test certs. The 251 of 388 was over `train_probe`'s rows (usable precrop only); recomputed on those rows it is still 251/388. DB file hash identical before and after. Per-split grades:

      | split | n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
      |---|---|---|---|---|---|---|---|---|---|---|---|
      | train | 1844 | 1 | 2 | 6 | 5 | 20 | 26 | 40 | 140 | 515 | 1089 |
      | val | 393 | | | 1 | 1 | 4 | 5 | 9 | 30 | 110 | 233 |
      | test | 393 | | | 1 | 1 | 4 | 5 | 9 | 30 | 110 | 233 |
  - [x] Check that no DB-writing job is running (collectors, `precrop`, `precompute_factors`), then back up: `sqlite3 data/pokescan.sqlite ".backup data/backups/pre-sticky-split.sqlite"` (or Python's `sqlite3.Connection.backup` if the CLI is missing). Confirm the backup opens and has the same row count. Check `data/backups/` is gitignored (`data/*` subfolders are listed one by one in `.gitignore`; add it if not).
    - 2026-10-27: no `python.exe`/`sqlite3.exe` process running. Backup with `Connection.backup` from a `mode=ro` source: `data/backups/pre-sticky-split.sqlite`, 6,762,496 bytes, `integrity_check` ok, 7,429 rows (1,844 train / 393 val / 393 test / 4,799 NULL), same as the source. `data/backups/` was not ignored; added to `.gitignore`.
  - [x] Run `python -m ml.data.split` for real: it changes 0 rows.
    - 2026-10-27: "changed split: 0 of 2630 rows". Compared cert by cert with the backup: 0 of 7,429 `split` values differ. The only change is the migration's new `split_legacy` column (all NULL).
- [ ] Phase 2, only after approval and a backup:
  - `python -m ml.data.split --mode grouped --reassign`, then `python -m ml.training.train_probe`;
  - the report shows `n_test_seen == 0`;
  - a new ADR records the new test n, grade histogram and metrics next to the old ones;
  - `python -m ml.eval.evaluate --split test --save-baseline` is re-run.
- [x] `ruff check .` exits 0.

## Facts vs assumptions

- **Fact:** the instability numbers, the 251/388 seen-card test certs and the simulated grouped-split sizes above (2026-10-23). There are currently 0 eBay rows.
- **Fact:** `train_probe` fits on train+val together, so after phase 2 only the test assignment matters for the probe. Val still matters for the legacy CNN.
- **Fact (checked 2026-10-27):** `split='excluded'` breaks no reader. Every `split` read in `ml/` and `backend/`:
  - `train_probe.load_rows` loads `split IS NOT NULL`, so excluded rows get features extracted (and cached), but they are in neither `fit_mask` (train/val), `test_mask` (test) nor the majority baseline. Wasted extraction only, no effect on any number.
  - `dataset.load_records(split)` filters `split=?`; its callers (`PSACardDataModule`, `precrop_coverage` in `train.py`'s guard, `evaluate.load_samples`) only ask for train/val/test.
  - `precrop`, `precompute_factors` and `detection_report` take `--split` as an exact match; without it they process every row, excluded included (harmless). `precrop_coverage_by_split` and `detection_report`'s table show an `excluded` key.
  - `backend/app` reads no `split`; backend tests read `split='test'` only.
- **Fact:** an eBay cert already in `train` moves to `excluded` in sticky mode when its card gets a PSA test cert. That's the one move sticky mode makes; it only takes a test card out of train, and keeping it would add test-card leakage. eBay certs in val/test (an old `--ebay-split stratify` run) or `excluded` keep their split. 0 eBay rows today, so it didn't affect the real run.
- **Fact:** any `--reassign` (stratified too, not only grouped) first copies `split` into `split_legacy` where that is NULL, so the original assignment survives whichever reassign runs first.
- **Open (not fixed, out of scope):** `ml/data/collect.py` upserts `"split": None`, so re-scanning a PSA cert range that was already collected resets those certs' split to NULL, and sticky mode then re-places them (by card majority or hash), which can move a test cert. `ebay_collect` skips existing certs, so it's unaffected. Dropping `split` from the collector's row dict would close this; candidate for `intent/smallfix.md`.
- **Assumption:** a hash split without grade stratification keeps enough low grades in test. Simulated 49–61 ≤8 vs 50 today; check with `--dry-run` before phase 2.

## Owner decisions (2026-10-27)

- **Phase 1 only** on real data (open question 1, decided 2026-10-23). Phase 2 is built and tested on `tmp_path` DBs in this round, but not run on the real DB.
- **Phase 1 real run:** approved, after a 0-changed-rows dry run, a check that no DB-writing job is running, and a manual backup (see the acceptance criterion).
- **`split='excluded'`:** accepted as written. Grep every split filter to confirm no reader breaks (there are 0 eBay rows today).
- **Test-set policy** (open question 9): yes; written into CLAUDE.md in this round (see Docs).
- **Delivery:** one PR together with [[probe-artifact-consistency]], this spec first. Its `train_probe` report fields (`test_seen_cards`, `n_test_seen`) are checked by the single real `train_probe` run at the end of that spec.
