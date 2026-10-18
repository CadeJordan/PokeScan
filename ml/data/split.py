"""Stratified train/val/test split by grade.

Run after `collect` to assign each Pokemon cert a `split` of {'train','val','test'}.

Only PSA-sourced certs (official catalogue photos, verified grades) are
stratified into val/test. eBay listings (user photos, grade parsed from the
title) always go to train, so val/test stay the high-fidelity PSA set and
adding eBay data never reshuffles the PSA assignment.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import typer
from rich.logging import RichHandler

from backend.app.core.config import get_settings
from ml.data.db import connect, grade_distribution

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True)])
log = logging.getLogger("split")

app = typer.Typer(add_completion=False)


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


@app.command()
def main(
    val_frac: float = typer.Option(0.15, help="Validation fraction per grade."),
    test_frac: float = typer.Option(0.15, help="Test fraction per grade."),
    seed: int = typer.Option(42, help="Random seed."),
    db_path: Path | None = typer.Option(None, help="Override SQLite DB path."),
    ebay_split: str = typer.Option(
        "train", help="'train' (default: eBay never in val/test) or 'stratify' (old behaviour)."
    ),
) -> None:
    settings = get_settings()
    db = db_path or settings.db_path
    with connect(db) as conn:
        rows = conn.execute(
            "SELECT cert_number, grade_int, source FROM certs "
            "WHERE is_pokemon=1 AND has_images=1 AND grade_int IS NOT NULL"
        ).fetchall()
        if not rows:
            log.warning("no eligible certs - run `collect` first")
            return
        assignments = assign_splits(
            [(r["cert_number"], int(r["grade_int"]), r["source"]) for r in rows],
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
            ebay_split=ebay_split,
        )
        conn.executemany(
            "UPDATE certs SET split=? WHERE cert_number=?",
            [(split, cert) for cert, split in assignments.items()],
        )
        log.info("grade distribution after split: %s", grade_distribution(conn))


if __name__ == "__main__":
    app()
