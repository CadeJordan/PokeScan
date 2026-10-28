"""Assign each Pokemon cert a `split` of {'train','val','test','excluded'}.

    python -m ml.data.split                              # sticky (default): only new certs
    python -m ml.data.split --dry-run                    # print what would change, write nothing
    python -m ml.data.split --mode grouped --reassign    # one card per split (destructive)

Run after `collect`. Only PSA-sourced certs (official catalogue photos,
verified grades) go to val/test. eBay listings (user photos, grade parsed from
the title) go to train, so val/test stay the high-fidelity PSA set.

Modes:
- `sticky` (default): a cert that already has a split keeps it, so re-running
  after a harvest never moves the test set. A new cert of a known printed card
  joins that card's split; a new card is placed by a hash of its card key
  (`ml.data.groups`), independent of every other row.
- `stratified --reassign`: the old per-grade shuffle over all PSA certs. Adding
  certs re-deals the whole grade.
- `grouped --reassign`: every PSA cert re-placed by the hash of its card, so no
  printed card spans train/val/test.

`--reassign` rewrites existing assignments; the first reassign keeps the old
ones in `split_legacy` and later ones leave it alone. In sticky and grouped mode
an eBay cert of a card with a PSA test cert gets 'excluded', which no loader reads.
"""

from __future__ import annotations

import logging
import random
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.logging import RichHandler

from backend.app.core.config import get_settings
from ml.data.db import connect, grade_distribution
from ml.data.groups import SPLIT_SALT, card_group_key, group_bucket

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True)])
log = logging.getLogger("split")

app = typer.Typer(add_completion=False)

MODES = ("sticky", "stratified", "grouped")
EVAL_SPLITS = ("train", "val", "test")
# Tie-break for a card whose assigned certs straddle splits: the new cert goes
# to test before val before train, so it never adds test-card leakage to train.
_TIE_ORDER = {"test": 0, "val": 1, "train": 2}


@dataclass(frozen=True)
class SplitRow:
    cert: str
    grade: int
    source: str | None
    group: str
    current_split: str | None

    @property
    def is_ebay(self) -> bool:
        return (self.source or "psa") == "ebay"


def _stratified_assign(
    cert_grades: list[tuple[str, int]],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, str]:
    rng = random.Random(seed)
    by_grade: dict[int, list[str]] = {}
    for cert, g in cert_grades:
        by_grade.setdefault(g, []).append(cert)

    out: dict[str, str] = {}
    for g, certs in by_grade.items():
        rng.shuffle(certs)
        n = len(certs)
        n_test = max(1, int(round(n * test_frac))) if n >= 5 else 0
        n_val = max(1, int(round(n * val_frac))) if n >= 5 else 0
        for i, c in enumerate(certs):
            if i < n_test:
                out[c] = "test"
            elif i < n_test + n_val:
                out[c] = "val"
            else:
                out[c] = "train"
        log.info(
            "grade %d: %d total -> train=%d val=%d test=%d",
            g, n, n - n_test - n_val, n_val, n_test,
        )
    return out


def assign_splits(
    rows: list[tuple[str, int, str | None]],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    ebay_split: str = "train",
) -> dict[str, str]:
    """rows: (cert_number, grade, source). eBay -> ``ebay_split``; PSA -> stratified."""
    if ebay_split not in ("train", "stratify"):
        raise ValueError("ebay_split must be 'train' or 'stratify'")
    psa = [(c, g) for c, g, src in rows if (src or "psa") != "ebay" or ebay_split == "stratify"]
    out = _stratified_assign(psa, val_frac=val_frac, test_frac=test_frac, seed=seed)
    for c, _g, _src in rows:
        if c not in out:
            out[c] = "train"
    n_ebay = sum(1 for *_, src in rows if src == "ebay")
    if n_ebay:
        log.info("ebay certs: %d (split=%s)", n_ebay, ebay_split)
    return out


def _bucket_split(group: str, val_frac: float, test_frac: float, salt: str) -> str:
    b = group_bucket(group, salt)
    if b < test_frac:
        return "test"
    if b < test_frac + val_frac:
        return "val"
    return "train"


def _assign_ebay(
    rows: list[SplitRow], psa_out: dict[str, str], *, keep_current: bool
) -> dict[str, str]:
    """eBay -> train, or 'excluded' when the card has a PSA test cert.

    keep_current: a cert that already has a split keeps it, except that train
    becomes 'excluded' once its card has a PSA test cert (that move only takes
    a test card out of train).
    """
    test_groups = {r.group for r in rows if psa_out.get(r.cert) == "test"}
    out: dict[str, str] = {}
    for r in rows:
        if not r.is_ebay:
            continue
        if keep_current and r.current_split not in (None, "train"):
            out[r.cert] = r.current_split
        else:
            out[r.cert] = "excluded" if r.group in test_groups else "train"
    return out


def assign_sticky(
    rows: list[SplitRow],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    salt: str = SPLIT_SALT,
) -> dict[str, str]:
    """Keep every existing split; place only unassigned certs (see module docstring)."""
    psa = [r for r in rows if not r.is_ebay]
    by_group: dict[str, Counter] = {}
    for r in psa:
        if r.current_split in EVAL_SPLITS:
            by_group.setdefault(r.group, Counter())[r.current_split] += 1

    out: dict[str, str] = {}
    for r in psa:
        if r.current_split is not None:
            out[r.cert] = r.current_split
        elif r.group in by_group:
            counts = by_group[r.group]
            out[r.cert] = min(counts, key=lambda s: (-counts[s], _TIE_ORDER[s]))
        else:
            out[r.cert] = _bucket_split(r.group, val_frac, test_frac, salt)
    out.update(_assign_ebay(rows, out, keep_current=True))
    return out


def assign_grouped(
    rows: list[SplitRow],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    salt: str = SPLIT_SALT,
) -> dict[str, str]:
    """Re-place every PSA cert by its card's hash: no card spans two splits."""
    out = {
        r.cert: _bucket_split(r.group, val_frac, test_frac, salt) for r in rows if not r.is_ebay
    }
    out.update(_assign_ebay(rows, out, keep_current=False))
    return out


def split_summary(rows: list[SplitRow], assignments: dict[str, str]) -> dict:
    """Per-split grade histogram, test straddle count and changed-row count."""
    hist: dict[str, Counter] = {}
    for r in rows:
        hist.setdefault(assignments[r.cert], Counter())[r.grade] += 1
    fit_groups = {r.group for r in rows if assignments[r.cert] in ("train", "val")}
    straddle = sum(
        1 for r in rows if assignments[r.cert] == "test" and r.group in fit_groups
    )
    return {
        "histogram": {s: dict(sorted(c.items())) for s, c in sorted(hist.items())},
        "test_straddle": straddle,
        "changed": sum(1 for r in rows if assignments[r.cert] != r.current_split),
    }


_ROWS_SQL = (
    "SELECT cert_number, grade_int, source, brand_title, subject, card_number, split "
    "FROM certs WHERE is_pokemon=1 AND has_images=1 AND grade_int IS NOT NULL"
)


def _read_rows(conn: sqlite3.Connection) -> list[SplitRow]:
    return [
        SplitRow(
            cert=r["cert_number"],
            grade=int(r["grade_int"]),
            source=r["source"],
            group=card_group_key(r["brand_title"], r["subject"], r["card_number"]),
            current_split=r["split"],
        )
        for r in conn.execute(_ROWS_SQL).fetchall()
    ]


def _has_legacy_split(conn: sqlite3.Connection) -> bool:
    """True once a reassign has recorded the original assignment in `split_legacy`."""
    sql = "SELECT 1 FROM certs WHERE split_legacy IS NOT NULL LIMIT 1"
    return conn.execute(sql).fetchone() is not None


def _assign_and_log(
    rows: list[SplitRow],
    mode: str,
    val_frac: float,
    test_frac: float,
    seed: int,
    salt: str,
    ebay_split: str,
) -> dict[str, str]:
    if mode == "sticky":
        out = assign_sticky(rows, val_frac=val_frac, test_frac=test_frac, salt=salt)
    elif mode == "grouped":
        out = assign_grouped(rows, val_frac=val_frac, test_frac=test_frac, salt=salt)
    else:
        out = assign_splits(
            [(r.cert, r.grade, r.source) for r in rows],
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
            ebay_split=ebay_split,
        )
    summary = split_summary(rows, out)
    for s, hist in summary["histogram"].items():
        log.info("%s: %d certs, grades %s", s, sum(hist.values()), hist)
    log.info(
        "test straddle: %d test certs share a card with train/val", summary["test_straddle"]
    )
    log.info("changed split: %d of %d rows (mode=%s)", summary["changed"], len(rows), mode)
    return out


@app.command()
def main(
    mode: str = typer.Option("sticky", help="sticky (default), stratified or grouped."),
    reassign: bool = typer.Option(
        False, help="Allow changing existing assignments (required by stratified/grouped)."
    ),
    dry_run: bool = typer.Option(False, help="Print the result; open the DB read-only."),
    val_frac: float = typer.Option(0.15, help="Validation fraction (per grade when stratified)."),
    test_frac: float = typer.Option(0.15, help="Test fraction (per grade when stratified)."),
    seed: int = typer.Option(42, help="Random seed (stratified only)."),
    salt: str = typer.Option(SPLIT_SALT, help="Card-hash salt (sticky/grouped)."),
    db_path: Path | None = typer.Option(None, help="Override SQLite DB path."),
    ebay_split: str = typer.Option(
        "train", help="'train' (default: eBay never in val/test) or 'stratify' (stratified only)."
    ),
) -> None:
    if mode not in MODES:
        raise typer.BadParameter(f"--mode must be one of {', '.join(MODES)}")
    if mode == "sticky" and reassign:
        raise typer.BadParameter("--reassign needs --mode stratified or grouped")
    if mode != "sticky" and not reassign:
        log.error("--mode %s rewrites existing splits; pass --reassign to confirm", mode)
        raise typer.Exit(code=1)
    if ebay_split != "train" and mode != "stratified":
        raise typer.BadParameter("--ebay-split stratify only works with --mode stratified")

    db = db_path or get_settings().db_path
    if dry_run:
        conn = sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = _read_rows(conn)
        finally:
            conn.close()
        if rows:
            _assign_and_log(rows, mode, val_frac, test_frac, seed, salt, ebay_split)
            log.info("dry run: nothing written")
        else:
            log.warning("no eligible certs - run `collect` first")
        return

    with connect(db) as conn:
        rows = _read_rows(conn)
        if not rows:
            log.warning("no eligible certs - run `collect` first")
            return
        assignments = _assign_and_log(rows, mode, val_frac, test_frac, seed, salt, ebay_split)
        if reassign and not _has_legacy_split(conn):
            # Keep the original assignment, on the first reassign only: a later one
            # would record certs that had no split then (NULL) as having its result.
            conn.execute("UPDATE certs SET split_legacy=split")
        conn.executemany(
            "UPDATE certs SET split=? WHERE cert_number=?",
            [(assignments[r.cert], r.cert) for r in rows if assignments[r.cert] != r.current_split],
        )
        log.info("grade distribution after split: %s", grade_distribution(conn))


if __name__ == "__main__":
    app()
