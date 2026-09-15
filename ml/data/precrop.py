"""Batch card detection + dewarp to 600x840 JPEGs on disk.

Training previously ran `detect_and_crop` inside every DataLoader worker on
every epoch (~5 min/epoch). This one-time job writes dewarped crops under
`data/crops/` (mirroring the relative path of each original under `data/`)
and records `cropped_front_path` / `cropped_back_path` in SQLite.

Usage:
    python -m ml.data.precrop
    python -m ml.data.precrop --force
    python -m ml.data.precrop --split train
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.logging import RichHandler
from tqdm import tqdm

from backend.app.core.config import get_settings
from backend.app.ml.preprocess import detect_and_crop, encode_jpeg
from ml.data.db import connect, precrop_coverage, update_precrop_paths

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True)])
log = logging.getLogger("precrop")

app = typer.Typer(add_completion=False)


def cropped_rel_path(original_rel: str) -> str:
    """Map `images/foo.jpg` -> `crops/images/foo.jpg`."""
    return str(Path("crops") / original_rel)


def precrop_file(src: Path, dest: Path) -> bool:
    """Dewarp `src` to 600x840 and write JPEG to `dest`. Returns quad-found flag."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    crop = detect_and_crop(src.read_bytes())
    dest.write_bytes(encode_jpeg(crop.image))
    return crop.found_quad


@app.command()
def main(
    force: bool = typer.Option(False, help="Re-crop even if outputs already exist."),
    split: str | None = typer.Option(None, help="Only process certs in this split."),
    limit: int | None = typer.Option(None, help="Cap number of certs processed."),
    db_path: Path | None = typer.Option(None, help="Override SQLite DB path."),
) -> None:
    settings = get_settings()
    db = db_path or settings.db_path
    data_root = settings.data_dir

    with connect(db) as conn:
        cov = precrop_coverage(conn)
        log.info(
            "precrops: %d / %d done (%d missing)",
            cov["precropped"],
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
            sql += " AND cropped_front_path IS NULL"
        rows = conn.execute(sql, params).fetchall()

    if limit is not None:
        rows = rows[:limit]

    log.info("processing %d certs (force=%s, split=%s)", len(rows), force, split or "all")
    ok = skipped = failed = quad_miss = 0

    with connect(db) as conn:
        for row in tqdm(rows, desc="precrop"):
            cert = row["cert_number"]
            front_src = data_root / row["front_path"]
            back_src = data_root / row["back_path"]
            if not (front_src.exists() and back_src.exists()):
                skipped += 1
                continue

            front_rel = cropped_rel_path(row["front_path"])
            back_rel = cropped_rel_path(row["back_path"])
            front_dest = data_root / front_rel
            back_dest = data_root / back_rel

            try:
                f_quad = precrop_file(front_src, front_dest)
                b_quad = precrop_file(back_src, back_dest)
                if not (f_quad and b_quad):
                    quad_miss += 1
                update_precrop_paths(
                    conn,
                    cert,
                    cropped_front_path=front_rel,
                    cropped_back_path=back_rel,
                )
                ok += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                log.debug("failed %s: %s", cert, exc)

    with connect(db) as conn:
        cov = precrop_coverage(conn)
    log.info(
        "done: ok=%d skipped=%d failed=%d quad_fallback=%d | precropped now %d / %d",
        ok,
        skipped,
        failed,
        quad_miss,
        cov["precropped"],
        cov["total"],
    )


if __name__ == "__main__":
    app()
