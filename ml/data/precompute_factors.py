"""Precompute corner/edge/surface pseudo-labels from classical CV.

Runs `backend.app.ml.condition.compute_condition_subgrades` over every
Pokemon cert with front/back images and stores the 1-10 sub-grades in SQLite.
These labels supervise the multi-task factor heads during training.

Usage:
    python -m ml.data.precompute_factors
    python -m ml.data.precompute_factors --force   # recompute all
    python -m ml.data.precompute_factors --split train
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.logging import RichHandler
from tqdm import tqdm

from backend.app.core.config import get_settings
from backend.app.ml.condition import compute_condition_subgrades
from ml.data.db import connect, factor_label_coverage, update_factor_subgrades

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True)])
log = logging.getLogger("precompute_factors")

app = typer.Typer(add_completion=False)


def _resolve_image(images_root: Path, rel_path: str) -> Path | None:
    p = images_root / rel_path
    return p if p.exists() else None


@app.command()
def main(
    force: bool = typer.Option(False, help="Recompute even if labels already exist."),
    split: str | None = typer.Option(
        None, help="Only process certs in this split (train/val/test)."
    ),
    limit: int | None = typer.Option(None, help="Cap number of certs processed."),
    db_path: Path | None = typer.Option(None, help="Override SQLite DB path."),
) -> None:
    settings = get_settings()
    db = db_path or settings.db_path
    images_root = settings.images_dir.parent

    with connect(db) as conn:
        cov = factor_label_coverage(conn)
        log.info(
            "factor labels: %d / %d labeled (%d missing)",
            cov["labeled"],
            cov["total"],
            cov["missing"],
        )

        sql = (
            "SELECT cert_number, front_path, back_path FROM certs "
            "WHERE is_pokemon=1 AND has_images=1 "
            "AND front_path IS NOT NULL AND back_path IS NOT NULL"
        )
        params: list = []
        if split:
            sql += " AND split=?"
            params.append(split)
        if not force:
            sql += " AND corners_subgrade IS NULL"
        rows = conn.execute(sql, params).fetchall()

    if limit is not None:
        rows = rows[:limit]

    log.info("processing %d certs (force=%s, split=%s)", len(rows), force, split or "all")
    ok = skipped = failed = 0

    with connect(db) as conn:
        for row in tqdm(rows, desc="factor labels"):
            cert = row["cert_number"]
            front = _resolve_image(images_root, row["front_path"])
            back = _resolve_image(images_root, row["back_path"])
            if front is None or back is None:
                skipped += 1
                continue
            try:
                cond = compute_condition_subgrades(front.read_bytes(), back.read_bytes())
                update_factor_subgrades(
                    conn,
                    cert,
                    corners=cond.corners.grade,
                    edges=cond.edges.grade,
                    surface=cond.surface.grade,
                )
                ok += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                log.debug("failed %s: %s", cert, exc)

    with connect(db) as conn:
        cov = factor_label_coverage(conn)
    log.info(
        "done: ok=%d skipped=%d failed=%d | labeled now %d / %d",
        ok,
        skipped,
        failed,
        cov["labeled"],
        cov["total"],
    )


if __name__ == "__main__":
    app()
