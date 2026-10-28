# Test data isolation

## Intent

The test suite can never modify real data: `data/pokescan.sqlite`, `data/images/`, `data/crops/` or `data/models/`. Tests that need real data only read it, and that is enforced, not just a convention.

## Why

- `data/` holds the only copies of harvested PSA/eBay data: 923 MB of images, 726 MB of crops, a 6.5 MB SQLite DB and 3.7 GB of models. PSA quota is ~100–200 calls/day/token, so re-harvesting takes weeks.
- Today the tests behave: `test_precrop_filtering` uses `tmp_path`, `test_model_performance` and `test_preprocess` only read. But `get_settings()` hands any test the real paths, and `db.connect()` always opens read-write. One careless new test could corrupt the DB.
- Long-running jobs (e.g. `precompute_factors --force`, seen running 2026-10-17) write the DB at the same time, so an accidental test write could also collide with a live job.

## Scope

- `ml/data/db.py::connect` gains an opt-in `read_only=True` mode (`sqlite3.connect("file:...?mode=ro", uri=True)`). The default stays unchanged.
- Test code that opens the real DB goes through read-only access. `ml/eval/evaluate.py::run_evaluation` gets a read-only path, or the test passes one in.
- A session-level guard in the root conftest: record size and mtime of everything under `get_settings().data_dir` at session start (excluding `*-journal` and `*-wal`) and fail the session if a fast-suite test changed any of it.
  - It must not false-positive when an external job writes the DB concurrently. Option: guard only images/crops/models by mtime, and rely on read-only mode for the DB.
- `visual` outputs stay under `test-artifacts/` (gitignored); the guard checks nothing else is written in the repo.

## Out of scope

- Changing how collectors or training write the DB.
- Moving, copying or deleting any images.
- Backups (see [[data-backup-before-destructive-runs]]).

## Acceptance criteria

- [ ] `connect(path, read_only=True)` raises `sqlite3.OperationalError` on any write; there is a test for this using `tmp_path`.
- [ ] No test opens the real DB read-write: grep for `connect(` / `get_settings().db_path` in `*/tests/`, and every hit is read-only or a `tmp_path`.
- [ ] A test that deliberately writes into `data_dir` makes the guard fail (verified once, then removed or kept as an xfail meta-test).
- [ ] The fast suite is still 248 passed / 3 xfailed; `model` tests still pass against real weights.
- [ ] Running the suite while `precompute_factors` writes the DB doesn't trip the guard.

## Facts vs assumptions

- **Fact:** current test files read real data in exactly two places (`ml/tests/test_model_performance.py:27`, `backend/tests/test_preprocess.py:257`) plus the model-path fixtures.
- **Assumption:** `run_evaluation` only reads the DB. Verify before routing it through read-only mode.
