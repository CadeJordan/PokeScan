# Data safety: automatic DB backups and test data isolation

## Intent

`data/pokescan.sqlite` and `data/images/` hold the only copy of the quota-limited harvest. Two guards protect them:

1. Every command that rewrites existing DB rows first takes a consistent, timestamped snapshot of the DB, so a bad run can be rolled back.
2. The test suite provably can't change anything under `data/`. Readers open the DB read-only, and a session check fails the run if a test changed something.

This replaces [[data-backup-before-destructive-runs]] and [[test-data-isolation]] (both 2026-10-18), which are out of date: they name `precompute_factors` (removed in [[retire-legacy-model]]) and a 248-test baseline. Both move to `intent/archive/` in this PR, marked superseded.

## Facts (2026-11-02)

- **The DB** is 7.1 MB. `data/backups/` exists and is gitignored. It holds hand-made snapshots (`pre-sticky-split.sqlite`, `pre-grouped-split.sqlite`, `pokescan.pre-tighter-crops-2026-10-31.sqlite`), model backups and `legacy-cnn-2026-11-01/`. Every snapshot so far was taken by hand before a destructive run.
- **Writers** (`ml.data.db.connect`, read-write):
  - `split.py` writes `split`; with `--reassign` it rewrites every PSA assignment. Its `--dry-run` already opens the DB `mode=ro`.
  - `precrop.py` writes crop paths and status; with `--force` it rewrites all of them.
  - The collectors (`collect.py`, `collect_all.py`, `ebay_collect.py`) upsert.
- **Readers that open the DB read-write anyway:**
  - `dataset.load_records` (`dataset.py:62`) and `evaluate.py:391`.
  - `connect()` creates the parent folder, runs `SCHEMA` + `_migrate` + the index script, and commits on exit. So every `ml.eval` run, and every `model`/`slow` test that loads records, can write the real DB: schema DDL, migrations.
- **Tests:**
  - Three tests open the real DB `mode=ro` by hand: `test_preprocess.py:521`, `test_review_adversarial.py:596`, `test_card_groups.py:46`.
  - The rest use `tmp_path` DBs. Several of them monkeypatch `get_settings` to point at one.
  - Nothing enforces any of this. There is no root `conftest.py`.

## Scope

### 1. Backups (`ml/data/db.py`)

- `backup_db(db_path, dest_dir=None, keep=5) -> Path`:
  - Uses `sqlite3.Connection.backup()`, which is consistent even while another process is writing.
  - Writes to `data/backups/auto/pokescan-<UTC YYYYmmddTHHMMSSZ>-<command>.sqlite`.
  - Prunes only files in `auto/` matching that pattern, keeping the newest `keep`. It never touches the hand-made backups beside it.
- Logs the path, and refuses (raises) if the copy fails, so the command doesn't run unprotected.
- Called at the start of the commands in Q1. Each gets `--no-backup`.

### 2. Read-only access (`ml/data/db.py`)

- `connect(db_path, read_only=False)`. With `read_only=True`:
  - opens `file:<quoted path>?mode=ro` (URI-quoted as in `phone_set.lookup_splits`, so `#`/`%` in the path work);
  - runs no DDL or migration and doesn't create folders;
  - raises `FileNotFoundError` if the DB is missing.
- `dataset.load_records` and `evaluate.py` read with `read_only=True` (Q4).
- The three hand-rolled `mode=ro` test helpers, and `split --dry-run`'s, switch to `connect(..., read_only=True)` if that's simpler. Behaviour stays the same.

### 3. Session guard (`conftest.py` at the repo root)

- At session start, snapshot `data/images/`, `data/crops/`, `data/models/` and `data/backups/`: relative path, size and mtime of every file. Also fingerprint the DB's schema: a sha256 of the `sqlite_master` SQL plus `PRAGMA user_version`, read through a read-only connection.
- At session end, compare. Any added, removed or changed file, or a changed DB schema, fails the session with the list (Q3).
- **Excluded:**
  - `*-journal` and `*-wal`;
  - `data/models/cache/`, whose feature caches can be rebuilt by `model` tests legitimately (to confirm during development; if no fast test writes it, don't exclude it);
  - the DB's row data. A live collector may be upserting while tests run, so the DB is protected by read-only access, not by the snapshot.
- Skipped cleanly when `data/` doesn't exist (CI).
- Cost: well under a few seconds for ~11k files (stat only). Measure it.

## Out of scope

- Backing up images, crops or models; off-machine backups.
- Changing what any command writes.
- Pruning or moving the existing hand-made backups.
- Collectors' resume logic.

## Acceptance criteria

- [x] `backup_db` on a `tmp_path` DB writes one snapshot that opens and has the same row count. It's consistent while a second connection holds an open write transaction.
- [x] Six backups with `keep=5` leave 5 in `auto/`. A hand-made file next to them, or in the parent folder, is untouched.
- [x] Each command in Q1 backs up before its first write (test on `tmp_path`, with the backup dir patched). `--no-backup` skips it. A failing backup stops the command before it writes.
- [x] `connect(path, read_only=True)` raises `sqlite3.OperationalError` on a write, runs no DDL (the schema of an old-schema DB is unchanged), and raises on a missing file.
- [x] `load_records` and `ml.eval` read-only. An `ml.eval` run leaves the DB's mtime unchanged (checked once on the real DB, read-only).
- [x] The guard fails a session in which a test writes a file under `data/images/` (meta-test: run pytest in a subprocess on a throwaway test with `data_dir` pointed at a `tmp_path` copy; never the real folder). It passes on the real suite.
- [x] Grep: no test opens the real DB except through `read_only=True`.
- [x] Fast suite passes (baseline 1584 passed, 1 skipped, 6 xfailed, plus the new tests); `ruff check .` clean; CI green (guard skipped there).
- [x] CLAUDE.md "Gotchas" and README mention automatic backups and `--no-backup`. The two old specs are archived as superseded.

## Assumptions (to check during development)

- No fast-suite test legitimately writes under `data/` today. If one does, list it and ask before changing it.
- `Connection.backup()` of a 7 MB DB takes well under a second.
- No current reader depends on `connect()` migrating the schema on read. All the DBs in use are already migrated.

## Questions for the owner

1. **Which commands back up automatically?** *Recommendation:* `split` (any mode except `--dry-run`) and `precrop` (any mode). Both rewrite or extend columns the pipeline depends on, and 7 MB is cheap. Not the collectors: they run for hours, only upsert, and are resumed often, so snapshots would pile up.
2. **Retention:** keep the newest 5 automatic backups in `data/backups/auto/`, never pruning anything outside it? *Recommendation:* yes.
3. **Guard strictness:** fail the session when a test changes `data/`, or only warn? *Recommendation:* fail. A warning in a 4-minute log gets missed.
4. **Read-only for real reads, not just tests:** switch `load_records` / `ml.eval` to read-only too, so evaluation can't migrate or write the DB? *Recommendation:* yes. They never need to write, and it's the main path a test reaches the real DB through.

## Owner decisions (2026-11-02)

1. **Backups:** `split` (any mode except `--dry-run`) and `precrop` (any mode) back up automatically; the collectors don't.
2. **Retention:** the newest 5 in `data/backups/auto/`; nothing outside `auto/` is ever pruned. Writing there is approved; it's the only new write under `data/`.
3. **Guard:** fail the session.
4. **Read-only:** `load_records` and `ml.eval` open the DB read-only, not just the tests.

## Results

Done 2026-11-02 (uncommitted; CI not run yet, the guard is off there by design).

- **Backups:** `db.backup_db(db_path, dest_dir=None, keep=5, *, command)` copies through a read-only `Connection.backup()` into a `.partial` file, runs `PRAGMA quick_check`, then renames it to `pokescan-<UTC>-<command>.sqlite`, so a failed copy never leaves a file that looks good or counts toward pruning. The default `dest_dir` is `<DB folder>/backups/auto/`, which is `data/backups/auto/` for the real DB and keeps `tmp_path` DBs' backups in `tmp_path`. Two backups in the same second get `-2`, `-3`. A missing DB raises `FileNotFoundError`. `split` (not `--dry-run`) and `precrop` call it first and have `--no-backup`.
- **Read-only:** `connect(path, read_only=True)` (keyword-only) uses the quoted `mode=ro` URI. `load_records`, `evaluate._crop_kinds_by_cert`, `split --dry-run` and the three real-DB tests use it.
- **Guard:** root `conftest.py`. It snapshots in `pytest_sessionstart`, so writes made during collection count too. The comparison runs in an autouse session fixture's teardown, so a change shows as `1 error` in the final count, with the list. It prints a header line with the file count and a summary line. Cost on the real data dir (10,622 files): 0.08 s to snapshot and 0.08–0.16 s to check.
- **`data/models/cache/` isn't excluded.** Neither the fast suite nor `-m model` wrote it, so per the spec there was no reason to.
- **Tests:** `ml/tests/test_data_safety.py` (22). Fast suite: 1606 passed, 1 skipped, 6 xfailed (baseline 1584 + 22). `-m model`: 10 passed in 16 min.
- **Real DB check:** mtime, size and sha256 of `data/pokescan.sqlite` were unchanged across the `-m model` run, whose `test_model_performance` is a full `ml.eval.run_evaluation` on the test split.
- **Assumptions held:** no fast test writes under `data/`; no reader relied on `connect()` migrating.
- **Surprising:** the first fast-suite run died silently at ~90% (in `test_resolution_pilot.py`), with no traceback and no failing test. That file passes alone, and an identical full rerun and the test agent's full run passed. Not reproduced; most likely the runner's ~4-minute tool timeout on a ~250 s suite, not this change.

### After the test agent (PASS, 16/17 mutations caught)

Fixed before the PR:
- **`backup_db` could prune the backup it had just made** when the clock was behind an existing backup's name (reproduced: it returned a path that no longer existed). The new file is now excluded from pruning. Test: `test_backup_never_prunes_the_one_it_just_made`.
- **The guard missed a same-schema DB swapped in for the real one.** It now also records the `certs` row count and fails if it drops (collectors only add rows). It also covers `data/images_full/` (the downloaded originals), and its failure message says a collector, `precrop` or `split` running at the same time is the likely cause. Test: `test_guard_fails_when_rows_drop_or_originals_change`.
- **The three hand-rolled readers** (`train_probe.load_rows`, `detection_report`, `resolution_pilot`) now use `connect(read_only=True)`. With a `#` in the path their unquoted URIs opened the DB read-write and created a stray file (reproduced on scratch). So no code or test opens the real DB read-only any other way.
- **Docs:** README says no-op runs still take a snapshot (copy one out of `auto/` to keep it), and that a missing DB is now an error; CLAUDE.md describes the guard's row count, `images_full` and the concurrent-writer caveat.

Not changed: a hard kill mid-copy can leave a `*.partial` file in `auto/` (harmless: it doesn't match the backup pattern). Fast suite after the fixes: 1608 passed, 1 skipped, 6 xfailed (248 s); the guard reported no change. `ruff check .` clean.
