"""Card-detection coverage on the real dataset photos.

    python -m ml.eval.detection_report                  # all certs with images
    python -m ml.eval.detection_report --split test --limit 200
    python -m ml.eval.detection_report --split test --variant exif_rot90

Reports the quad-found rate per side and split and the detect time, and writes
overlays of the failures to test-artifacts/detection_failures/ (a subfolder
per non-`orig` variant) for eyeballing. The DB is opened read-only.

`--variant` simulates phone uploads from the stored photos, re-encoded as JPEG
q=92 before detection: `exif_rot90` (pixels rotated 90 deg CCW plus EXIF
Orientation=6, like a portrait phone shot), `rot90` (rotated 90 deg CW, no
EXIF), `rot180`, and `up3x` (3x upscale, like a 12 MP photo). `orig` passes
the stored bytes unchanged.
"""

from __future__ import annotations

import io
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import typer
from PIL import Image
from tqdm import tqdm

from backend.app.core.config import PROJECT_ROOT, get_settings
from backend.app.ml.preprocess import detect_and_crop

OUT_DIR = PROJECT_ROOT / "test-artifacts" / "detection_failures"
VARIANTS = ("orig", "exif_rot90", "rot90", "rot180", "up3x")
_EXIF_ORIENTATION = 0x0112

app = typer.Typer(add_completion=False)


def _overlay(crop, title: str):
    # Imported lazily: the helper lives with the tests.
    from backend.tests.helpers.overlay import draw_detection

    img = draw_detection(crop, title=title)
    scale = 700 / img.shape[0]
    return cv2.resize(img, (round(img.shape[1] * scale), 700))


def make_variant(raw: bytes, variant: str) -> bytes:
    """The upload `variant` (see the module docstring) of a stored photo."""
    if variant == "orig":
        return raw
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    exif = Image.Exif()
    if variant == "exif_rot90":
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        exif[_EXIF_ORIENTATION] = 6  # a viewer rotates 90 CW to show it upright
    elif variant == "rot90":
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    elif variant == "rot180":
        bgr = cv2.rotate(bgr, cv2.ROTATE_180)
    elif variant == "up3x":
        bgr = cv2.resize(bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(
        buf, format="JPEG", quality=92, exif=exif
    )
    return buf.getvalue()


@app.command()
def main(
    split: str | None = typer.Option(None, help="Only this split (default: all with images)."),
    limit: int | None = typer.Option(None, help="Cap number of certs."),
    max_failures: int = typer.Option(60, help="Max failure overlays to write."),
    variant: str = typer.Option("orig", help=f"Upload simulation: {', '.join(VARIANTS)}."),
) -> None:
    if variant not in VARIANTS:
        raise typer.BadParameter(f"--variant must be one of {', '.join(VARIANTS)}")
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
    # Read-only: this report must never migrate or write the harvest DB.
    conn = sqlite3.connect(f"file:{settings.db_path.resolve().as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    out_dir = OUT_DIR if variant == "orig" else OUT_DIR / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    found: dict[tuple[str, str], list[bool]] = defaultdict(list)
    failures: list[str] = []
    detect_ms: list[float] = []
    for r in tqdm(rows, desc=f"detect ({variant})"):
        for side in ("front", "back"):
            path = settings.data_dir / r[f"{side}_path"]
            if not path.exists():
                continue
            data = make_variant(path.read_bytes(), variant)
            t0 = time.perf_counter()
            crop = detect_and_crop(data)
            detect_ms.append(1000 * (time.perf_counter() - t0))
            found[(r["split"] or "none", side)].append(crop.found_quad)
            if not crop.found_quad:
                failures.append(path.name)
                if len(failures) <= max_failures:
                    cv2.imwrite(str(out_dir / f"{Path(path).stem}.jpg"), _overlay(crop, path.stem))

    summary = {}
    typer.echo(f"\nvariant {variant}\nsplit   side   found")
    for side in ("front", "back"):
        total = [v for (_s, sd), vs in found.items() if sd == side for v in vs]
        summary[side] = sum(total) / max(1, len(total))
    for (sp, side), vs in sorted(found.items()):
        typer.echo(f"{sp:7s} {side:6s} {sum(vs) / len(vs):6.1%}  (n={len(vs)})")
    typer.echo(f"ALL     front  {summary['front']:6.1%}\nALL     back   {summary['back']:6.1%}")
    p50, p95 = (float(np.percentile(detect_ms, q)) if detect_ms else 0.0 for q in (50, 95))
    typer.echo(f"detect  p50 {p50:.0f} ms  p95 {p95:.0f} ms  (n={len(detect_ms)})")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "variant": variant,
                "rates": summary,
                "detect_ms_p50": round(p50, 1),
                "detect_ms_p95": round(p95, 1),
                "failures": failures,
            },
            indent=2,
        )
    )
    typer.echo(f"\n{len(failures)} failures; overlays in {out_dir}")


if __name__ == "__main__":
    app()
