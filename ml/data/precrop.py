"""Batch card detection + dewarp to 600x840 JPEGs on disk.

Training previously ran `detect_and_crop` inside every DataLoader worker on
every epoch (~5 min/epoch). This one-time job writes dewarped crops under
`data/crops/` (mirroring the relative path of each original under `data/`)
and records `cropped_front_path` / `cropped_back_path` in SQLite.

Usage:
    python -m ml.data.precrop
    python -m ml.data.precrop --force
    python -m ml.data.precrop --split train

Each cert also gets `crop_found_front/back` (outline detected; a miss means
the "crop" is the padded raw slab photo, label included) and
`crop_kind_front/back` ('card' | 'well' | 'small' | 'none' | 'unknown', see
`crop_kind`). Rows are committed in batches so an interrupted run keeps what
it has done (a single end-of-run commit once lost a 2-day run).
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.logging import RichHandler
from tqdm import tqdm

from backend.app.core.config import get_settings
from backend.app.ml.preprocess import detect_and_crop, encode_jpeg
from ml.data.db import (
    connect,
    precrop_coverage,
    precrop_coverage_by_split,
    update_precrop_paths,
)

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True)])
log = logging.getLogger("precrop")

app = typer.Typer(add_completion=False)


def cropped_rel_path(original_rel: str) -> str:
    """Map `images/foo.jpg` -> `crops/images/foo.jpg`."""
    return str(Path("crops") / original_rel)


COMMIT_EVERY = 50

# PSA cert photos are framed consistently: the card spans ~0.64-0.66 of the
# image height, the slab well/interior ~0.73-0.80. Anything taller than this
# is the holder around the card rather than the card itself. Shorter than
# SMALL_MAX is either a small-format card (promo inserts) or an inner frame /
# partial outline inside the card - worth a look before trusting it.
WELL_MIN_HEIGHT_FRAC = 0.70
SMALL_MAX_HEIGHT_FRAC = 0.60


def crop_kind(found: bool, quad, image_h: int, source: str = "psa") -> str:
    """Classify a detection: 'card', 'well', 'small', 'none' or 'unknown'.

    Only PSA photos have the fixed framing the height rule relies on; other
    sources are 'unknown' when found.
    """
    if not found or quad is None:
        return "none"
    if source != "psa":
        return "unknown"
    q = quad.reshape(4, 2)
    height = ((q[3][1] - q[0][1]) + (q[2][1] - q[1][1])) / 2.0
    frac = height / image_h
    if frac > WELL_MIN_HEIGHT_FRAC:
        return "well"
    return "small" if frac < SMALL_MAX_HEIGHT_FRAC else "card"


def precrop_file(src: Path, dest: Path, source: str = "psa") -> tuple[bool, str]:
    """Dewarp `src` to 600x840 and write JPEG to `dest`.

    Returns (quad found, crop kind).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    crop = detect_and_crop(src.read_bytes())
    dest.write_bytes(encode_jpeg(crop.image))
    h = crop.source_bgr.shape[0] if crop.source_bgr is not None else 1
    return crop.found_quad, crop_kind(crop.found_quad, crop.quad, h, source)


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
            "SELECT cert_number, source, front_path, back_path FROM certs "
            "WHERE is_pokemon=1 AND has_images=1 "
            "AND front_path IS NOT NULL AND back_path IS NOT NULL"
        )
        params: list = []
        if split:
            sql += " AND split=?"
            params.append(split)
        if not force:
            # Also redo rows cropped before crop status was recorded.
            sql += " AND (cropped_front_path IS NULL OR crop_found_front IS NULL)"
        rows = conn.execute(sql, params).fetchall()

    if limit is not None:
        rows = rows[:limit]

    log.info("processing %d certs (force=%s, split=%s)", len(rows), force, split or "all")
    ok = skipped = failed = quad_miss = 0

    with connect(db) as conn:
        for i, row in enumerate(tqdm(rows, desc="precrop")):
            if i and i % COMMIT_EVERY == 0:
                conn.commit()
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
                source = row["source"] or "psa"
                f_quad, f_kind = precrop_file(front_src, front_dest, source)
                b_quad, b_kind = precrop_file(back_src, back_dest, source)
                if not (f_quad and b_quad):
                    quad_miss += 1
                update_precrop_paths(
                    conn,
                    cert,
                    cropped_front_path=front_rel,
                    cropped_back_path=back_rel,
                    found_front=f_quad,
                    found_back=b_quad,
                    kind_front=f_kind,
                    kind_back=b_kind,
                )
                ok += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                log.warning("failed %s: %s", cert, exc)

    with connect(db) as conn:
        cov = precrop_coverage(conn)
        by_split = precrop_coverage_by_split(conn)
    log.info(
        "done: ok=%d skipped=%d failed=%d quad_fallback=%d | precropped now %d / %d",
        ok,
        skipped,
        failed,
        quad_miss,
        cov["precropped"],
        cov["total"],
    )
    for sp, c in sorted(by_split.items()):
        log.info(
            "%-5s total=%d cropped=%d found=%d (%.1f%%) card-tight=%d (%.1f%%)",
            sp, c["total"], c["cropped"], c["found"], 100 * c["found"] / max(1, c["total"]),
            c["card"], 100 * c["card"] / max(1, c["total"]),
        )


if __name__ == "__main__":
    app()
