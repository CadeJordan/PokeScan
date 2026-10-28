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
    - 2026-10-27 (tester review): "only if it's null" per row let a second reassign fill in certs that had no split at the first one (NULL), or arrived later, with a post-first-reassign split. Now `split` is copied only on the first reassign ever: if any row already has a non-NULL `split_legacy`, the copy is skipped. Certs with no split at that first reassign stay NULL ("had no split"). Edge case: if every cert was unassigned at the first reassign, the column stays all NULL and the next reassign counts as the first (there was no old assignment to keep). Test: `ml/tests/test_review_split_artifacts.py::test_split_legacy_not_filled_by_a_later_reassign`.
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
- [x] Phase 2, only after approval and a backup:
  - `python -m ml.data.split --mode grouped --reassign`, then `python -m ml.training.train_probe`;
  - the report shows `n_test_seen == 0`;
  - a new ADR records the new test n, grade histogram and metrics next to the old ones;
  - `python -m ml.eval.evaluate --split test --save-baseline` is re-run.
  - 2026-10-27 (second attempt, after the sanity check was re-referenced): read-only 3-seed CV first: grouped fit set 0.4694 / 0.4668 / 0.4610, mean **0.4657**, +0.014 over 0.4518, inside ±0.02. Old fit set reproduced 0.4467 / 0.4552 / 0.4534.
  - Backups: the live DB was first checked equal to the earlier backups (0 split diffs, `split_legacy` all NULL), then backed up with `Connection.backup` from a `mode=ro` source to `data/backups/pre-grouped-split.sqlite` (integrity ok, 7,429 rows, 1,844 / 393 / 393 / 4,799 NULL, per-cert identical). `grade_head.*`, `card_features.*` and `grade_model.*` copied to `data/models/backup/pre-grouped-split/`, SHA-256 identical. No `python.exe`/`sqlite3.exe` was running.
  - Dry run and real run print the same: test 375 (47 graded ≤8), val 374, train 1,881, test straddle 0, "changed split: 1218 of 2630 rows (mode=grouped)". Cert by cert against the dry-run assignment: 0 mismatches, 0 non-eligible rows touched, 0 of 1,310 cards span splits. This was the first reassign, so `split_legacy` was filled and equals the pre-run split for every row.
  - `train_probe`: `n_test_seen` **0**, test n 370, `feature_spec_hash` 428e7a366e2e unchanged, ONNX not re-exported. CV MAE 0.4694 (n 2,241), test MAE 0.4405 / QWK 0.485 / exact 66.5%.
  - `evaluate --split test --save-baseline` (served probe, n 375): MAE 0.435, exact 66.9%, QWK 0.491, ECE 0.053, always-10 MAE 0.656. Written to `ml/eval/baseline.json`, which is tracked.
  - ADR: `docs/adr/0002-card-grouped-test-split.md`.
- [x] `ruff check .` exits 0.

## Facts vs assumptions

- **Fact:** the instability numbers, the 251/388 seen-card test certs and the simulated grouped-split sizes above (2026-10-23). There are currently 0 eBay rows.
- **Fact:** `train_probe` fits on train+val together, so after phase 2 only the test assignment matters for the probe. Val still matters for the legacy CNN.
- **Fact (checked 2026-10-27):** `split='excluded'` breaks no reader. Every `split` read in `ml/` and `backend/`:
  - `train_probe.load_rows` filters `split IN ('train','val','test')` (it loaded `split IS NOT NULL` when this was checked, so excluded rows got features extracted but were in neither `fit_mask`, `test_mask` nor the majority baseline), so excluded rows are no longer even extracted.
  - `dataset.load_records(split)` filters `split=?`; its callers (`PSACardDataModule`, `precrop_coverage` in `train.py`'s guard, `evaluate.load_samples`) only ask for train/val/test.
  - `precrop`, `precompute_factors` and `detection_report` take `--split` as an exact match; without it they process every row, excluded included (harmless). `precrop_coverage_by_split` and `detection_report`'s table show an `excluded` key.
  - `backend/app` reads no `split`; backend tests read `split='test'` only.
- **Fact:** an eBay cert already in `train` moves to `excluded` in sticky mode when its card gets a PSA test cert. That's the one move sticky mode makes; it only takes a test card out of train, and keeping it would add test-card leakage. eBay certs in val/test (an old `--ebay-split stratify` run) or `excluded` keep their split. 0 eBay rows today, so it didn't affect the real run.
- **Fact:** the first `--reassign` (stratified too, not only grouped) copies `split` into `split_legacy`, so the original assignment survives whichever reassign runs first. Later reassigns leave the column alone (2026-10-27, see Scope).
- **Fixed (commit 15bce27, 2026-10-27, [[smallfix]]):** `ml/data/collect.py` upserted `"split": None`, so re-scanning a PSA cert range that was already collected reset those certs' split to NULL, and sticky mode then re-placed them (by card majority or hash), which could move a test cert. The collector's row dict no longer has `split`. `ebay_collect` skips existing certs, so it was unaffected.
- **Assumption:** a hash split without grade stratification keeps enough low grades in test. Simulated 49–61 ≤8 vs 50 today; check with `--dry-run` before phase 2.
- **Fact (for phase 2 planning):** a read-only `--mode grouped --reassign --dry-run` on the real DB on 2026-10-27 gave 375 test certs, 47 of them graded ≤8 (below the simulated 49–61), test straddle 0, and 1,218 rows changed.

## Owner decisions (2026-10-27)

- **Phase 1 only** on real data (open question 1, decided 2026-10-23). Phase 2 is built and tested on `tmp_path` DBs in this round, but not run on the real DB.
- **Phase 1 real run:** approved, after a 0-changed-rows dry run, a check that no DB-writing job is running, and a manual backup (see the acceptance criterion).
- **`split='excluded'`:** accepted as written. Grep every split filter to confirm no reader breaks (there are 0 eBay rows today).
- **Test-set policy** (open question 9): yes; written into CLAUDE.md in this round (see Docs).
- **Delivery:** one PR together with [[probe-artifact-consistency]], this spec first. Its `train_probe` report fields (`test_seen_cards`, `n_test_seen`) are checked by the single real `train_probe` run at the end of that spec.
  - 2026-10-27, checked: the real run wrote `n_test_seen` 251 and `test_seen_cards` (n 251, MAE 0.4143) to the report and the head metadata, beside `test_unseen_cards` (n 137, MAE 0.6277). 251 of 388 matches the straddle count above.

## Owner decisions (2026-10-27, round 3)

- **Phase 2 approved** (open question 1): run `--mode grouped --reassign` on the real DB this round, after a fresh DB backup (`data/backups/pre-grouped-split.sqlite`) and a `data/models` backup.
- **Salt stays `pokescan-split-v1`.** The 47 test certs graded ≤8 (simulated 49–61) are accepted; trying other salts until the test set looks nicer would be choosing the test set by hand.
- **No tolerance on the new metrics**: the test set and the fit set both change. Sanity check only: card-grouped CV MAE within ±0.02 of 0.4467 (CV is already card-grouped, so it should barely move); stop and report if it doesn't.
- The new ADR (`docs/adr/0002-card-grouped-test-split.md`) becomes the baseline [[face-side-orientation-guard]] measures against.

## Owner decisions (2026-10-27, round 3, after the first phase-2 attempt)

- **First attempt stopped on the sanity check and was rolled back.** Grouped-split CV MAE was 0.4694 (+0.0227 over 0.4467). The DB and `data/models` were restored from the backups above.
- **Fact:** the 0.4467 reference was one CV seed. Card-grouped 5-fold CV over seeds 0/1/2 gives 0.4467 / 0.4552 / 0.4534 (mean **0.4518**) on the old fit set, and 0.4694 / 0.4668 / 0.4610 (mean 0.4657) on the grouped fit set. The ~0.014 gap matches the changed card population; the bug signals (straddle 0, row counts, `split_legacy`) were clean.
- **Sanity check re-referenced** to the 3-seed mean: grouped 3-seed CV MAE mean within **±0.02 of 0.4518**; stop and report if it isn't. ADR 0002 records the 3-seed CV means for both fit sets beside the single-seed numbers.
- **Commit `ml/eval/baseline.json`** (written by `evaluate --save-baseline`) so later specs compare against a tracked file.
