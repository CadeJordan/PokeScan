"""Card-detection coverage on the real dataset photos.

    python -m ml.eval.detection_report                  # all certs with images
    python -m ml.eval.detection_report --split test --limit 200

Reports the quad-found rate per side and split, and writes overlays of the
failures to test-artifacts/detection_failures/ for eyeballing.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import typer
from tqdm import tqdm

from backend.app.core.config import PROJECT_ROOT, get_settings
from backend.app.ml.preprocess import detect_and_crop
from ml.data.db import connect

OUT_DIR = PROJECT_ROOT / "test-artifacts" / "detection_failures"

app = typer.Typer(add_completion=False)


def _overlay(crop, title: str):
    # Imported lazily: the helper lives with the tests.
    from backend.tests.helpers.overlay import draw_detection

    img = draw_detection(crop, title=title)
    scale = 700 / img.shape[0]
    return cv2.resize(img, (round(img.shape[1] * scale), 700))


@app.command()
def main(
    split: str | None = typer.Option(None, help="Only this split (default: all with images)."),
    limit: int | None = typer.Option(None, help="Cap number of certs."),
    max_failures: int = typer.Option(60, help="Max failure overlays to write."),
) -> None:
    settings = get_settings()
    sql = (
        "SELECT cert_number, split, front_path, back_path FROM certs "
        "WHERE is_pokemon=1 AND has_images=1 AND front_path IS NOT NULL AND back_path IS NOT NULL"
    )
    params: list = []
    if split:
        sql += " AND split=?"
        params.append(split)
    sql += " ORDER BY cert_number"
    if limit:
        sql += f" LIMIT {int(limit)}"
    with connect(settings.db_path) as conn:
        rows = conn.execute(sql, params).fetchall()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.jpg"):
        old.unlink()

    found: dict[tuple[str, str], list[bool]] = defaultdict(list)
    failures: list[str] = []
    for r in tqdm(rows, desc="detect"):
        for side in ("front", "back"):
            path = settings.data_dir / r[f"{side}_path"]
            if not path.exists():
                continue
            crop = detect_and_crop(path.read_bytes())
            found[(r["split"] or "none", side)].append(crop.found_quad)
            if not crop.found_quad:
                failures.append(path.name)
                if len(failures) <= max_failures:
                    cv2.imwrite(str(OUT_DIR / f"{Path(path).stem}.jpg"), _overlay(crop, path.stem))

    summary = {}
    typer.echo("\nsplit   side   found")
    for side in ("front", "back"):
        total = [v for (_s, sd), vs in found.items() if sd == side for v in vs]
        summary[side] = sum(total) / max(1, len(total))
    for (sp, side), vs in sorted(found.items()):
        typer.echo(f"{sp:7s} {side:6s} {sum(vs) / len(vs):6.1%}  (n={len(vs)})")
    typer.echo(f"ALL     front  {summary['front']:6.1%}\nALL     back   {summary['back']:6.1%}")
    (OUT_DIR / "summary.json").write_text(
        json.dumps({"rates": summary, "failures": failures}, indent=2)
    )
    typer.echo(f"\n{len(failures)} failures; overlays in {OUT_DIR}")


if __name__ == "__main__":
    app()
