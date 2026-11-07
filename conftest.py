"""Session guard: the test suite must not change real data (intent/archive/data-safety).

At session start it records the relative path, size and mtime of every file
under `data/images`, `images_full`, `crops`, `models` and `backups`, plus a
fingerprint of the DB schema and its `certs` row count (read through a
read-only connection). After the last test it compares: any added, removed or
changed file, a changed schema, or fewer rows fails the session with the list.
The rows themselves aren't compared, since a live collector may be upserting
while the tests run (it only adds rows); readers open the DB read-only instead.
Skipped when the data folder doesn't exist (CI).
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

GUARDED_DIRS = ("images", "images_full", "crops", "models", "backups")
# SQLite's transient side files: a live writer creates and removes them.
EXCLUDED_SUFFIXES = ("-journal", "-wal")

Snapshot = dict[str, tuple[int, int]]


@dataclass
class _Guard:
    data_dir: Path
    db_path: Path
    files: Snapshot
    schema: str | None
    rows: int | None
    seconds: float
    report: str = ""


_GUARD = pytest.StashKey[_Guard | None]()


def _walk(root: Path, data_dir: Path, out: Snapshot) -> None:
    # os.scandir: on Windows the entry's stat comes with the directory listing.
    with os.scandir(root) as it:
        for entry in it:
            if entry.is_dir(follow_symlinks=False):
                _walk(Path(entry.path), data_dir, out)
            elif not entry.name.endswith(EXCLUDED_SUFFIXES):
                st = entry.stat(follow_symlinks=False)
                rel = Path(entry.path).relative_to(data_dir).as_posix()
                out[rel] = (st.st_size, st.st_mtime_ns)


def snapshot_files(data_dir: Path) -> Snapshot:
    out: Snapshot = {}
    for sub in GUARDED_DIRS:
        if (data_dir / sub).is_dir():
            _walk(data_dir / sub, data_dir, out)
    return out


def db_fingerprint(db_path: Path) -> tuple[str | None, int | None]:
    """sha256 of the `sqlite_master` SQL and `user_version`, and the `certs` row
    count (None without that table); (None, None) when there's no DB."""
    from ml.data.db import connect

    if not db_path.is_file():
        return None, None
    with connect(db_path, read_only=True) as conn:
        rows = conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        has_certs = any(r["type"] == "table" and r["name"] == "certs" for r in rows)
        count = conn.execute("SELECT COUNT(*) FROM certs").fetchone()[0] if has_certs else None
    h = hashlib.sha256(repr([tuple(r) for r in rows]).encode())
    h.update(f"user_version={version}".encode())
    return h.hexdigest(), count


def diff_snapshots(before: Snapshot, after: Snapshot) -> list[str]:
    out = [f"added    {p}" for p in sorted(after.keys() - before.keys())]
    out += [f"removed  {p}" for p in sorted(before.keys() - after.keys())]
    out += [
        f"changed  {p} (size {before[p][0]} -> {after[p][0]})"
        for p in sorted(before.keys() & after.keys())
        if before[p] != after[p]
    ]
    return out


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session: pytest.Session) -> None:
    # Settings() rather than get_settings(): the latter creates the data folders.
    from backend.app.core.config import Settings

    s = Settings()
    if not s.data_dir.is_dir():
        session.config.stash[_GUARD] = None
        return
    t0 = time.perf_counter()
    files = snapshot_files(s.data_dir)
    schema, rows = db_fingerprint(s.db_path)
    session.config.stash[_GUARD] = _Guard(
        s.data_dir, s.db_path, files, schema, rows, time.perf_counter() - t0
    )


def pytest_report_header(config: pytest.Config) -> str:
    g = config.stash.get(_GUARD, None)
    if g is None:
        return "data guard: off (no data folder)"
    return (
        f"data guard: {len(g.files)} files under {g.data_dir} "
        f"({', '.join(GUARDED_DIRS)}) + DB schema and row count, snapshot {g.seconds:.2f}s"
    )


@pytest.fixture(scope="session", autouse=True)
def _data_guard(request: pytest.FixtureRequest):
    """Fails the session (an error at the last test's teardown) if `data/` changed."""
    yield
    g = request.config.stash.get(_GUARD, None)
    if g is None:
        return
    t0 = time.perf_counter()
    problems = diff_snapshots(g.files, snapshot_files(g.data_dir))
    schema, rows = db_fingerprint(g.db_path)
    if schema != g.schema:
        problems.append(f"changed  DB schema of {g.db_path}")
    # Collectors only add rows, so fewer means something replaced or rewrote the DB.
    if g.rows is not None and (rows is None or rows < g.rows):
        problems.append(f"changed  DB certs rows {g.rows} -> {rows}")
    g.report = f"data guard: {len(problems) or 'no'} change(s), check {time.perf_counter() - t0:.2f}s"
    if problems:
        pytest.fail(
            f"tests changed real data under {g.data_dir}:\n  " + "\n  ".join(problems)
            + "\n(If a collector, precrop or split was running during the tests, that"
            " is the likely cause: rerun when it's done.)",
            pytrace=False,
        )


def pytest_terminal_summary(terminalreporter, config: pytest.Config) -> None:
    g = config.stash.get(_GUARD, None)
    if g is not None and g.report:
        terminalreporter.write_line(g.report)
