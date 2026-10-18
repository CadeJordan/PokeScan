# DB backup before destructive runs

## Intent

Any pipeline command that rewrites existing rows in `data/pokescan.sqlite` (`--force` reruns, `split`) first takes a timestamped, consistent snapshot of the DB, so a bad run can be rolled back.

## Why

- The DB is the hub for split assignments, crop paths, `*_subgrade` pseudo-labels and PSA harvest progress (see `HANDOFF.md`). PSA data is quota-limited (~100–200 calls/day/token), so it's expensive to rebuild.
- `precompute_factors --force`, `precrop --force` and `split` overwrite columns in place, with no undo. A stray `-journal` file was present during a live `--force` run on 2026-10-17, which is normal, but an interrupted run leaves only that.

## Scope

- A small helper in `ml/data/db.py`, `backup_db(db_path) -> Path`, using `sqlite3.Connection.backup()` (consistent even while another process holds the DB) to `data/backups/pokescan-<UTC timestamp>.sqlite`.
- Call it at the start of `split.py`, and of `precompute_factors.py` / `precrop.py` when `--force` is set. Add a `--no-backup` opt-out.
- Keep the last N backups (default 5) and prune older ones. The DB is 6.5 MB, so this is cheap.
- Add `data/backups/` to `.gitignore`. Only specific `data/` subpaths are ignored today, so it isn't covered yet.

## Out of scope

- Backing up images, crops or models. They're large; off-machine backup is a separate decision for the user.
- Changing what the commands write.

## Acceptance criteria

- [ ] Running `python -m ml.data.split` on a `tmp_path` DB creates exactly one backup, which opens and has the same row count.
- [ ] A backup taken while another connection holds a write transaction succeeds and is consistent (test with two connections on `tmp_path`).
- [ ] With 6 runs and N=5, 5 backups remain.
- [ ] `--no-backup` skips it.
- [ ] `data/backups/` is gitignored; `git status` stays clean after a run.
- [ ] All tests use `tmp_path`, never the real DB (see [[test-data-isolation]]).

## Facts vs assumptions

- **Fact:** `split.py` issues `UPDATE certs SET split=?`; precrop and precompute_factors have `--force`.
- **Assumption:** the collectors (`collect`, `collect_all`, `ebay_collect`) only upsert new rows and don't need backups. Verify.
