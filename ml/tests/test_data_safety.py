"""Data safety (intent/archive/data-safety): automatic DB backups, read-only readers and
the session guard in the root conftest. Everything here runs on `tmp_path`."""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from ml.data import dataset, db, precrop, split
from ml.data.db import backup_db, connect, upsert_cert

ROOT = Path(__file__).resolve().parents[2]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _count(path: Path, where: str = "1") -> int:
    with connect(path, read_only=True) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM certs WHERE {where}").fetchone()[0]


def _schema(path: Path) -> list[tuple]:
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY name").fetchall()
    finally:
        conn.close()


@pytest.fixture
def certs_db(tmp_path):
    """Temp DB of 30 Pokemon certs with images, no split, no precrop."""
    path = tmp_path / "pokescan.sqlite"
    with connect(path) as conn:
        for i in range(30):
            upsert_cert(conn, {
                "cert_number": f"{i:08d}", "source": "psa", "grade_int": 5 + i % 6,
                "is_pokemon": 1, "has_images": 1,
                "front_path": f"images/{i}_f.jpg", "back_path": f"images/{i}_b.jpg",
                "brand_title": "BRAND", "subject": f"CARD{i}", "card_number": str(i),
            })
    return path


def _autos(path: Path) -> list[Path]:
    d = path.parent / "backups" / "auto"
    return sorted(d.iterdir()) if d.is_dir() else []


# --------------------------------------------------------------------------- #
# backup_db
# --------------------------------------------------------------------------- #
def test_backup_is_a_full_copy(certs_db):
    out = backup_db(certs_db, command="split")
    assert out.parent == certs_db.parent / "backups" / "auto"
    assert out.name.startswith("pokescan-") and out.name.endswith("-split.sqlite")
    assert _autos(certs_db) == [out]
    assert _count(out) == _count(certs_db) == 30


def test_backup_is_consistent_during_a_write_transaction(certs_db):
    writer = sqlite3.connect(certs_db)
    try:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("DELETE FROM certs WHERE cert_number < '00000010'")
        out = backup_db(certs_db, command="split")  # the open transaction isn't copied
        writer.commit()
    finally:
        writer.close()
    assert _count(out) == 30
    assert _count(certs_db) == 20
    conn = sqlite3.connect(out)
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()


def test_backup_keeps_newest_and_never_touches_other_files(certs_db, monkeypatch):
    auto = certs_db.parent / "backups" / "auto"
    auto.mkdir(parents=True)
    hand_auto = auto / "pre-sticky-split.sqlite"
    hand_parent = auto.parent / "pokescan-20200101T000000Z-split.sqlite"  # pattern, wrong folder
    for p in (hand_auto, hand_parent):
        p.write_bytes(b"hand-made")
    t0 = datetime(2026, 11, 2, 12, 0, 0, tzinfo=timezone.utc)
    made = []
    for i in range(6):
        monkeypatch.setattr(db, "_utcnow", lambda i=i: t0 + timedelta(seconds=i))
        made.append(backup_db(certs_db, keep=5, command="precrop"))
    left = [p for p in _autos(certs_db) if p != hand_auto]
    assert left == made[1:]  # the oldest went
    assert hand_auto.read_bytes() == hand_parent.read_bytes() == b"hand-made"


def test_backup_never_prunes_the_one_it_just_made(certs_db, monkeypatch):
    # The clock is behind the names of existing backups (corrected clock, copied
    # folder): the new one sorts oldest but must survive the prune.
    for i in range(3):
        monkeypatch.setattr(db, "_utcnow", lambda i=i: datetime(2027, 1, 1, 0, 0, i, tzinfo=timezone.utc))
        backup_db(certs_db, keep=2, command="split")
    monkeypatch.setattr(db, "_utcnow", lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))
    out = backup_db(certs_db, keep=2, command="split")
    assert out.exists()
    assert len(_autos(certs_db)) == 2


def test_backups_within_one_second_get_distinct_names(certs_db, monkeypatch):
    monkeypatch.setattr(db, "_utcnow", lambda: datetime(2026, 11, 2, tzinfo=timezone.utc))
    a = backup_db(certs_db, command="split")
    b = backup_db(certs_db, command="split")
    assert a != b and a.exists() and b.exists()
    assert b.name == a.name.replace(".sqlite", "-2.sqlite")


def test_backup_explicit_dest_dir(certs_db, tmp_path):
    out = backup_db(certs_db, dest_dir=tmp_path / "elsewhere", command="split")
    assert out.parent == tmp_path / "elsewhere"
    assert not (certs_db.parent / "backups").exists()


def test_backup_raises_and_leaves_nothing_when_it_fails(certs_db, monkeypatch, tmp_path):
    with pytest.raises(FileNotFoundError):
        backup_db(tmp_path / "missing.sqlite", command="split")

    real_connect = sqlite3.connect

    class _FailingBackup:
        def __init__(self, conn):
            self._conn = conn

        def backup(self, *_a, **_k):
            raise sqlite3.OperationalError("disk I/O error")

        def __getattr__(self, name):
            return getattr(self._conn, name)

    def fake_connect(target, *a, **k):
        conn = real_connect(target, *a, **k)
        return _FailingBackup(conn) if str(target).startswith("file:") else conn

    monkeypatch.setattr(db.sqlite3, "connect", fake_connect)
    with pytest.raises(sqlite3.OperationalError):
        backup_db(certs_db, command="split")
    monkeypatch.undo()
    assert _autos(certs_db) == []  # not even a .partial


# --------------------------------------------------------------------------- #
# The commands back up before their first write
# --------------------------------------------------------------------------- #
@pytest.fixture
def spy_backup(monkeypatch):
    """Records the DB's digest at each backup call, then runs the real backup."""
    calls: list[tuple[Path, str]] = []

    def spy(db_path, *a, **k):
        calls.append((Path(db_path), _digest(Path(db_path))))
        return backup_db(db_path, *a, **k)

    monkeypatch.setattr(split, "backup_db", spy)
    monkeypatch.setattr(precrop, "backup_db", spy)
    return calls


def _fail_backup(*_a, **_k):
    raise OSError("backup disk full")


def _split(*args):
    return CliRunner().invoke(split.app, [*args])


@pytest.mark.parametrize("args", [(), ("--mode", "grouped", "--reassign")])
def test_split_backs_up_before_writing(certs_db, spy_backup, args):
    before = _digest(certs_db)
    assert _split(*args, "--db-path", str(certs_db)).exit_code == 0
    assert spy_backup == [(certs_db, before)]  # backed up the untouched DB
    (out,) = _autos(certs_db)
    assert out.name.endswith("-split.sqlite")
    assert _count(out, "split IS NULL") == 30  # the pre-run state
    assert _count(certs_db, "split IS NULL") == 0


def test_split_dry_run_and_no_backup_skip_it(certs_db, spy_backup):
    assert _split("--dry-run", "--db-path", str(certs_db)).exit_code == 0
    assert _split("--no-backup", "--db-path", str(certs_db)).exit_code == 0
    assert spy_backup == [] and _autos(certs_db) == []
    assert _count(certs_db, "split IS NULL") == 0  # the --no-backup run did write


def test_split_stops_before_writing_when_the_backup_fails(certs_db, monkeypatch):
    monkeypatch.setattr(split, "backup_db", _fail_backup)
    before = _digest(certs_db)
    result = _split("--mode", "grouped", "--reassign", "--db-path", str(certs_db))
    assert result.exit_code != 0 and isinstance(result.exception, OSError)
    assert _digest(certs_db) == before


@pytest.fixture
def precrop_env(certs_db, monkeypatch):
    """One cert's photos on disk (the rest are skipped); settings point at tmp."""
    data = certs_db.parent
    img = np.full((280, 200, 3), 200, np.uint8)
    for rel in ("images/0_f.jpg", "images/0_b.jpg"):
        (data / rel).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(data / rel), img)
    settings = SimpleNamespace(data_dir=data, db_path=certs_db)
    monkeypatch.setattr(precrop, "get_settings", lambda: settings)
    return settings


def _precrop(*args):
    return CliRunner().invoke(precrop.app, [*args])


@pytest.mark.parametrize("args", [(), ("--force",)])
def test_precrop_backs_up_before_writing(precrop_env, spy_backup, args):
    db_path = precrop_env.db_path
    before = _digest(db_path)
    assert _precrop(*args).exit_code == 0
    assert spy_backup == [(db_path, before)]
    (out,) = _autos(db_path)
    assert out.name.endswith("-precrop.sqlite")
    assert _count(out, "cropped_front_path IS NOT NULL") == 0
    assert _count(db_path, "cropped_front_path IS NOT NULL") == 1


def test_precrop_no_backup_skips_it(precrop_env, spy_backup):
    assert _precrop("--no-backup").exit_code == 0
    assert spy_backup == [] and _autos(precrop_env.db_path) == []
    assert _count(precrop_env.db_path, "cropped_front_path IS NOT NULL") == 1


def test_precrop_stops_before_writing_when_the_backup_fails(precrop_env, monkeypatch):
    monkeypatch.setattr(precrop, "backup_db", _fail_backup)
    before = _digest(precrop_env.db_path)
    result = _precrop("--force")
    assert result.exit_code != 0 and isinstance(result.exception, OSError)
    assert _digest(precrop_env.db_path) == before
    assert not (precrop_env.data_dir / "crops").exists()


# --------------------------------------------------------------------------- #
# connect(read_only=True)
# --------------------------------------------------------------------------- #
@pytest.fixture
def old_schema_db(tmp_path):
    """A DB from before the migrated columns and indexes existed."""
    path = tmp_path / "old #1 100%.sqlite"  # URI-special characters in the name
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE certs (cert_number TEXT PRIMARY KEY, grade_int INTEGER, "
        "is_pokemon INTEGER, has_images INTEGER, split TEXT)"
    )
    conn.execute("INSERT INTO certs VALUES ('1', 9, 1, 1, NULL)")
    conn.commit()
    conn.close()
    return path


def test_read_only_connect_runs_no_ddl_and_rejects_writes(old_schema_db):
    schema, digest = _schema(old_schema_db), _digest(old_schema_db)
    with connect(old_schema_db, read_only=True) as conn:
        assert conn.execute("SELECT grade_int FROM certs").fetchone()["grade_int"] == 9
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("UPDATE certs SET grade_int=10")
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE x (a)")
    assert _schema(old_schema_db) == schema
    assert _digest(old_schema_db) == digest
    with connect(old_schema_db) as conn:  # contrast: read-write migrates
        pass
    assert _schema(old_schema_db) != schema


def test_read_only_connect_missing_file_raises_and_creates_nothing(tmp_path):
    missing = tmp_path / "no" / "such.sqlite"
    with pytest.raises(FileNotFoundError), connect(missing, read_only=True):
        pass
    assert not missing.parent.exists()


# --------------------------------------------------------------------------- #
# Readers: load_records and ml.eval open the DB read-only
# --------------------------------------------------------------------------- #
@pytest.fixture
def reader_db(certs_db, monkeypatch):
    """A migrated DB missing one index: any read-write connect() would recreate it."""
    with connect(certs_db) as conn:
        conn.execute("UPDATE certs SET split='test', crop_kind_front='card', crop_kind_back='well'")
    conn = sqlite3.connect(certs_db)
    conn.execute("DROP INDEX idx_certs_source")
    conn.commit()
    conn.close()
    settings = SimpleNamespace(data_dir=certs_db.parent, db_path=certs_db)
    monkeypatch.setattr(dataset, "get_settings", lambda: settings)
    monkeypatch.setattr("backend.app.core.config.get_settings", lambda: settings)
    return certs_db


def test_load_records_is_read_only(reader_db):
    digest, mtime = _digest(reader_db), reader_db.stat().st_mtime_ns
    assert dataset.load_records("test", use_precrop=True) == []  # no crops on disk
    assert dataset.load_records("test", use_precrop=False) == []  # no photos on disk
    assert _digest(reader_db) == digest and reader_db.stat().st_mtime_ns == mtime


def test_load_records_missing_db_raises_instead_of_creating(tmp_path, monkeypatch):
    settings = SimpleNamespace(data_dir=tmp_path, db_path=tmp_path / "missing.sqlite")
    monkeypatch.setattr(dataset, "get_settings", lambda: settings)
    with pytest.raises(FileNotFoundError):
        dataset.load_records("test")
    assert not settings.db_path.exists()


def test_eval_crop_kinds_is_read_only(reader_db):
    from ml.eval.evaluate import _crop_kinds_by_cert

    digest = _digest(reader_db)
    kinds = _crop_kinds_by_cert(["00000001"])
    assert kinds["00000001"] == "well"  # worse side of card/well
    assert _digest(reader_db) == digest


# --------------------------------------------------------------------------- #
# Session guard (root conftest.py), run in a subprocess on a throwaway data dir
# --------------------------------------------------------------------------- #
@pytest.fixture
def guard_world(tmp_path):
    """A fake data dir and a test folder carrying a copy of the root conftest."""
    data = tmp_path / "data"
    for sub in ("images", "crops", "models/cache", "backups/auto"):
        (data / sub).mkdir(parents=True)
    (data / "images" / "a.jpg").write_bytes(b"a")
    (data / "crops" / "a.jpg").write_bytes(b"c")
    (data / "models" / "grade_head.npz").write_bytes(b"m")
    with connect(data / "pokescan.sqlite"):
        pass
    proj = tmp_path / "proj"
    proj.mkdir()
    shutil.copy(ROOT / "conftest.py", proj / "conftest.py")
    env = {
        **os.environ,
        "DATA_DIR": str(data),
        "DB_PATH": str(data / "pokescan.sqlite"),
        "IMAGES_DIR": str(data / "images"),
        "CROPS_DIR": str(data / "crops"),
        "MODELS_DIR": str(data / "models"),
    }

    def run(test_body: str) -> subprocess.CompletedProcess:
        (proj / "test_throwaway.py").write_text(
            "import os, sqlite3\nfrom pathlib import Path\n"
            "DATA = Path(os.environ['DATA_DIR'])\n\n"
            f"def test_throwaway():\n{test_body}\n"
        )
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", str(proj)],
            cwd=proj, env=env, capture_output=True, text=True, timeout=120,
        )

    return SimpleNamespace(data=data, run=run)


def test_guard_fails_a_session_that_writes_data(guard_world):
    r = guard_world.run(
        "    (DATA / 'images' / 'new.jpg').write_bytes(b'x')\n"
        "    (DATA / 'crops' / 'a.jpg').unlink()\n"
        "    (DATA / 'models' / 'grade_head.npz').write_bytes(b'mm')\n"
        "    conn = sqlite3.connect(DATA / 'pokescan.sqlite')\n"
        "    conn.execute('CREATE TABLE extra (a)')\n"
        "    conn.close()\n"
    )
    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    assert "1 passed, 1 error" in out
    for line in (
        "added    images/new.jpg",
        "removed  crops/a.jpg",
        "changed  models/grade_head.npz",
        "changed  DB schema",
    ):
        assert line in out, out


def test_guard_fails_when_rows_drop_or_originals_change(guard_world):
    # A same-schema DB swapped in for the real one keeps the schema; the row
    # count catches it. `images_full/` (the downloaded originals) is guarded too.
    (guard_world.data / "images_full").mkdir()
    with connect(guard_world.data / "pokescan.sqlite") as conn:
        conn.execute("INSERT INTO certs (cert_number) VALUES ('1')")
    r = guard_world.run(
        "    (DATA / 'images_full' / 'o.jpg').write_bytes(b'x')\n"
        "    conn = sqlite3.connect(DATA / 'pokescan.sqlite')\n"
        "    conn.execute('DELETE FROM certs')\n"
        "    conn.commit()\n"
        "    conn.close()\n"
    )
    out = r.stdout + r.stderr
    assert r.returncode != 0, out
    for line in ("added    images_full/o.jpg", "changed  DB certs rows 1 -> 0", "collector"):
        assert line in out, out
    assert "changed  DB schema" not in out


def test_guard_passes_a_read_only_session(guard_world):
    r = guard_world.run(
        "    assert (DATA / 'images' / 'a.jpg').read_bytes() == b'a'\n"
        "    (DATA / 'backups' / 'x.sqlite-journal').write_bytes(b'j')  # excluded\n"
        "    conn = sqlite3.connect(DATA / 'pokescan.sqlite')\n"
        "    conn.execute(\"INSERT INTO certs (cert_number) VALUES ('1')\")  # rows aren't guarded\n"
        "    conn.commit()\n"
        "    conn.close()\n"
    )
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "data guard: 3 files under" in out and "data guard: no change(s)" in out


def test_guard_is_off_without_a_data_dir(guard_world):
    shutil.rmtree(guard_world.data)
    r = guard_world.run("    pass\n")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "data guard: off" in out
