"""SQLite metadata store for harvested PSA certs."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

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
    split          TEXT,        -- 'train' | 'val' | 'test' | 'excluded' | NULL
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


_FACTOR_COLUMNS = (
    "corners_subgrade",
    "edges_subgrade",
    "surface_subgrade",
    "factors_computed_at",
)

_PRECROP_COLUMNS = (
    "cropped_front_path",
    "cropped_back_path",
    "precrop_at",
)

# Precrop outcome per side, written by precrop.py. `crop_found_*` is 1 when a
# card outline was detected (0 = the crop is the padded raw slab photo, label
# included, and must never be trained on). `crop_kind_*` is 'card' (tight on
# the card), 'well' (the slab well/interior around the card: looser, but no
# label), 'small' (small-format card, or an outline inside the card),
# 'none' (not found) or 'unknown' (non-PSA framing, not classified).
_CROP_STATUS_COLUMNS = {
    "crop_found_front": "INTEGER",
    "crop_found_back": "INTEGER",
    "crop_kind_front": "TEXT",
    "crop_kind_back": "TEXT",
}


# The `split` a cert had before its first `split.py --reassign`, so the numbers
# measured on the old assignment stay reproducible. Written once, never updated.
_SPLIT_COLUMNS = ("split_legacy",)


def _migrate(conn) -> None:
    """Best-effort additive migration for pre-existing DBs."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(certs)").fetchall()}
    if "source" not in cols:
        conn.execute("ALTER TABLE certs ADD COLUMN source TEXT NOT NULL DEFAULT 'psa'")
    for col in _FACTOR_COLUMNS:
        if col not in cols:
            col_type = "REAL" if col.endswith("_subgrade") else "TEXT"
            conn.execute(f"ALTER TABLE certs ADD COLUMN {col} {col_type}")
    for col in _PRECROP_COLUMNS:
        if col not in cols:
            col_type = "TEXT"
            conn.execute(f"ALTER TABLE certs ADD COLUMN {col} {col_type}")
    for col, col_type in _CROP_STATUS_COLUMNS.items():
        if col not in cols:
            conn.execute(f"ALTER TABLE certs ADD COLUMN {col} {col_type}")
    for col in _SPLIT_COLUMNS:
        if col not in cols:
            conn.execute(f"ALTER TABLE certs ADD COLUMN {col} TEXT")


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


def update_factor_subgrades(
    conn: sqlite3.Connection,
    cert_number: str,
    *,
    corners: float,
    edges: float,
    surface: float,
) -> None:
    conn.execute(
        "UPDATE certs SET corners_subgrade=?, edges_subgrade=?, surface_subgrade=?, "
        "factors_computed_at=datetime('now') WHERE cert_number=?",
        (corners, edges, surface, cert_number),
    )


def factor_label_coverage(conn: sqlite3.Connection) -> dict[str, int]:
    """Return counts of pokemon certs with/without precomputed factor labels."""
    total = int(
        conn.execute(
            "SELECT COUNT(*) FROM certs WHERE is_pokemon=1 AND has_images=1 "
            "AND front_path IS NOT NULL AND back_path IS NOT NULL"
        ).fetchone()[0]
    )
    labeled = int(
        conn.execute(
            "SELECT COUNT(*) FROM certs WHERE is_pokemon=1 AND has_images=1 "
            "AND corners_subgrade IS NOT NULL"
        ).fetchone()[0]
    )
    return {"total": total, "labeled": labeled, "missing": total - labeled}


def update_precrop_paths(
    conn: sqlite3.Connection,
    cert_number: str,
    *,
    cropped_front_path: str,
    cropped_back_path: str,
    found_front: bool | None = None,
    found_back: bool | None = None,
    kind_front: str | None = None,
    kind_back: str | None = None,
) -> None:
    conn.execute(
        "UPDATE certs SET cropped_front_path=?, cropped_back_path=?, "
        "crop_found_front=?, crop_found_back=?, crop_kind_front=?, crop_kind_back=?, "
        "precrop_at=datetime('now') WHERE cert_number=?",
        (
            cropped_front_path,
            cropped_back_path,
            None if found_front is None else int(found_front),
            None if found_back is None else int(found_back),
            kind_front,
            kind_back,
            cert_number,
        ),
    )


def precrop_coverage(conn: sqlite3.Connection) -> dict[str, int]:
    total = int(
        conn.execute(
            "SELECT COUNT(*) FROM certs WHERE is_pokemon=1 AND has_images=1 "
            "AND front_path IS NOT NULL AND back_path IS NOT NULL"
        ).fetchone()[0]
    )
    done = int(
        conn.execute(
            "SELECT COUNT(*) FROM certs WHERE is_pokemon=1 AND has_images=1 "
            "AND cropped_front_path IS NOT NULL"
        ).fetchone()[0]
    )
    return {"total": total, "precropped": done, "missing": total - done}


def grade_distribution(conn: sqlite3.Connection) -> dict[int, int]:
    rows = conn.execute(
        "SELECT grade_int, COUNT(*) FROM certs "
        "WHERE is_pokemon=1 AND has_images=1 AND grade_int IS NOT NULL "
        "GROUP BY grade_int ORDER BY grade_int"
    ).fetchall()
    return {int(r[0]): int(r[1]) for r in rows}


def precrop_coverage_by_split(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    """Per split: usable certs, certs with crop files, and certs whose front
    and back outlines were both found (usable for training)."""
    rows = conn.execute(
        "SELECT COALESCE(split, 'none') AS split, COUNT(*) AS total, "
        "SUM(cropped_front_path IS NOT NULL AND cropped_back_path IS NOT NULL) AS cropped, "
        "SUM(COALESCE(crop_found_front, 0) = 1 AND COALESCE(crop_found_back, 0) = 1) AS found, "
        "SUM(crop_kind_front = 'card' AND crop_kind_back = 'card') AS card "
        "FROM certs WHERE is_pokemon=1 AND has_images=1 AND grade_int IS NOT NULL "
        "AND front_path IS NOT NULL AND back_path IS NOT NULL GROUP BY 1"
    ).fetchall()
    return {
        r["split"]: {k: int(r[k] or 0) for k in ("total", "cropped", "found", "card")}
        for r in rows
    }
