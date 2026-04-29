"""Export a stratified subset of Pokemon cards for hand-annotation.

Produces a folder with paired front/back JPGs and a manifest CSV that can be
imported into CVAT, Roboflow, or Label Studio. We sample evenly across grade
strata so the annotated set has examples of clean (10s) and damaged (1-5)
cards alike.

Usage:
    python -m ml.data.export_for_annotation --total 800 --out data/annotation_set
"""

from __future__ import annotations

import csv
import logging
import random
import shutil
from pathlib import Path

import typer
from rich.logging import RichHandler

from backend.app.core.config import get_settings
from ml.data.db import connect

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True)])
log = logging.getLogger("export-annot")

app = typer.Typer(add_completion=False)


@app.command()
def main(
    total: int = typer.Option(800, help="Approximate total cards to export."),
    out: Path = typer.Option(Path("data/annotation_set"), help="Output directory."),
    seed: int = typer.Option(42),
) -> None:
    settings = get_settings()
    out.mkdir(parents=True, exist_ok=True)
    images_dst = out / "images"
    images_dst.mkdir(exist_ok=True)
    rng = random.Random(seed)

    with connect(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT cert_number, grade_int, front_path, back_path "
            "FROM certs "
            "WHERE is_pokemon=1 AND has_images=1 AND grade_int IS NOT NULL "
            "  AND front_path IS NOT NULL AND back_path IS NOT NULL"
        ).fetchall()
    if not rows:
        log.warning("no eligible certs - run `collect` first")
        return

    by_grade: dict[int, list] = {}
    for r in rows:
        by_grade.setdefault(int(r["grade_int"]), []).append(r)

    per_grade = max(1, total // len(by_grade))
    selected: list = []
    for g, items in by_grade.items():
        rng.shuffle(items)
        take = items[: min(per_grade, len(items))]
        log.info("grade %d: %d available -> %d selected", g, len(items), len(take))
        selected.extend(take)

    images_root = settings.images_dir.parent
    manifest = out / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["cert_number", "grade_int", "side", "filename", "source_path"]
        )
        for r in selected:
            for side, key in (("front", "front_path"), ("back", "back_path")):
                src = images_root / r[key]
                if not src.exists():
                    continue
                dst_name = f"{r['cert_number']}_{side}.jpg"
                dst = images_dst / dst_name
                if not dst.exists():
                    shutil.copyfile(src, dst)
                writer.writerow(
                    [r["cert_number"], int(r["grade_int"]), side, dst_name, str(src)]
                )

    log.info(
        "exported %d cards to %s (front+back), manifest at %s",
        len(selected), out, manifest,
    )
    log.info(
        "next step: annotate corner/edge/surface defects in CVAT/Roboflow/Label Studio "
        "using YOLO format. See ml/training/multitask.py for the head wiring."
    )


if __name__ == "__main__":
    app()
