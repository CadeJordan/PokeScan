"""Labelled cert records per split, read from the SQLite DB.

`load_records` gives each cert's front/back photo (or precrop) paths and PSA
grade; `ml.eval.evaluate` grades a split from them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from backend.app.core.config import get_settings
from ml.data.db import connect

log = logging.getLogger(__name__)


@dataclass(slots=True)
class CardRecord:
    cert_number: str
    front_path: Path
    back_path: Path
    grade: int


def _usable_crop(r, data_root: Path, crop_kinds: tuple[str, ...] | None) -> bool:
    """Precrop row is safe to train on: both sides cropped on a detected outline."""
    if not (r["cropped_front_path"] and r["cropped_back_path"]):
        return False
    if r["crop_found_front"] != 1 or r["crop_found_back"] != 1:
        return False
    if crop_kinds is not None and not (
        r["crop_kind_front"] in crop_kinds and r["crop_kind_back"] in crop_kinds
    ):
        return False
    return (data_root / r["cropped_front_path"]).exists() and (
        data_root / r["cropped_back_path"]
    ).exists()


def load_records(
    split: str,
    *,
    use_precrop: bool = True,
    crop_kinds: tuple[str, ...] | None = None,
) -> list[CardRecord]:
    """Records for `split`.

    use_precrop=True returns only certs whose front and back precrops exist
    and were cut on a detected card outline (`crop_found_*`=1), optionally
    restricted to `crop_kinds` (e.g. ("card",) for tight crops only). It never
    falls back to the raw photo: a raw PSA slab photo shows the grade on its
    label, so training on it leaks the answer. Dropped certs are logged.

    use_precrop=False returns the raw photos, for callers that run detection
    themselves (evaluate.py).
    """
    settings = get_settings()
    data_root = settings.data_dir
    out: list[CardRecord] = []
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT cert_number, grade_int, front_path, back_path, "
            "cropped_front_path, cropped_back_path, "
            "crop_found_front, crop_found_back, crop_kind_front, crop_kind_back FROM certs "
            "WHERE is_pokemon=1 AND has_images=1 AND grade_int IS NOT NULL "
            "  AND front_path IS NOT NULL AND back_path IS NOT NULL "
            "  AND split=?",
            (split,),
        ).fetchall()
    dropped = 0
    for r in rows:
        if use_precrop:
            if not _usable_crop(r, data_root, crop_kinds):
                dropped += 1
                continue
            front = data_root / r["cropped_front_path"]
            back = data_root / r["cropped_back_path"]
        else:
            front = data_root / r["front_path"]
            back = data_root / r["back_path"]
            if not (front.exists() and back.exists()):
                continue
        out.append(
            CardRecord(
                cert_number=r["cert_number"],
                front_path=front,
                back_path=back,
                grade=int(r["grade_int"]),
            )
        )
    if dropped:
        log.warning(
            "load_records(%s): dropped %d / %d certs without a usable precrop "
            "(missing, or card outline not found); run `python -m ml.data.precrop`",
            split,
            dropped,
            len(rows),
        )
    return out
