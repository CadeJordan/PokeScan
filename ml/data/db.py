"""SQLite metadata store for harvested PSA certs."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS certs (
    cert_number    TEXT PRIMARY KEY,   -- 'NNNNNNNN' for PSA, 'ebay:<itemId>' for eBay
    source         TEXT NOT NULL DEFAULT 'psa',  -- 'psa' | 'ebay'
    brand_title    TEXT,
    subject        TEXT,
    year           TEXT,
    card_number    TEXT,
    category       TEXT,
    grade_label    TEXT,
    grade_int      INTEGER,
    is_pokemon     INTEGER NOT NULL DEFAULT 0,
    front_url      TEXT,
    back_url       TEXT,
    front_path     TEXT,
    back_path      TEXT,
    has_images     INTEGER NOT NULL DEFAULT 0,
    split          TEXT,        -- 'train' | 'val' | 'test' | NULL
    fetched_at     TEXT NOT NULL DEFAULT (datetime('now')),
    raw_json       TEXT
);

CREATE INDEX IF NOT EXISTS idx_certs_grade ON certs(grade_int);
CREATE INDEX IF NOT EXISTS idx_certs_pokemon ON certs(is_pokemon);
CREATE INDEX IF NOT EXISTS idx_certs_split ON certs(split);
CREATE INDEX IF NOT EXISTS idx_certs_has_images ON certs(has_images);
"""

# Indexes that depend on columns added by `_migrate`. Created after migration
# so they don't blow up against a pre-existing table missing those columns.
POST_MIGRATION_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_certs_source ON certs(source);
"""


def _migrate(conn) -> None:
    """Best-effort additive migration for pre-existing DBs."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(certs)").fetchall()}
    if "source" not in cols:
        conn.execute("ALTER TABLE certs ADD COLUMN source TEXT NOT NULL DEFAULT 'psa'")


@contextmanager
def connect(db_path: Path, busy_timeout_ms: int = 5_000) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # `timeout` lets sqlite3 wait out transient locks (e.g. another process
    # finishing a write) instead of failing immediately; combined with the
    # PRAGMA below this also helps recover hot rollback journals left by
    # an earlier crash.
    conn = sqlite3.connect(str(db_path), timeout=busy_timeout_ms / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.executescript(POST_MIGRATION_INDEXES)
        yield conn
        conn.commit()
    finally:
        conn.close()


def upsert_cert(conn: sqlite3.Connection, row: dict) -> None:
    cols = ", ".join(row.keys())
    placeholders = ", ".join(f":{k}" for k in row)
    updates = ", ".join(f"{k}=excluded.{k}" for k in row if k != "cert_number")
    conn.execute(
        f"INSERT INTO certs ({cols}) VALUES ({placeholders}) "
        f"ON CONFLICT(cert_number) DO UPDATE SET {updates}",
        row,
    )


def cert_exists(conn: sqlite3.Connection, cert_number: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM certs WHERE cert_number=? LIMIT 1", (cert_number,)
        ).fetchone()
        is not None
    )


def pokemon_cert_count(conn: sqlite3.Connection, with_images_only: bool = True) -> int:
    sql = "SELECT COUNT(*) FROM certs WHERE is_pokemon=1"
    if with_images_only:
        sql += " AND has_images=1"
    return int(conn.execute(sql).fetchone()[0])


def grade_distribution(conn: sqlite3.Connection) -> dict[int, int]:
    rows = conn.execute(
        "SELECT grade_int, COUNT(*) FROM certs "
        "WHERE is_pokemon=1 AND has_images=1 AND grade_int IS NOT NULL "
        "GROUP BY grade_int ORDER BY grade_int"
    ).fetchall()
    return {int(r[0]): int(r[1]) for r in rows}
