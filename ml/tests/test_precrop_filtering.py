"""Training never sees raw slab photos: load_records and train_probe.load_rows filtering.

Raw PSA slab photos print the grade on the label, so a cert whose precrop is
missing or was cut without a detected card outline must be dropped, never
replaced by the raw photo.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ml.data import dataset
from ml.data.db import connect, precrop_coverage_by_split, update_precrop_paths, upsert_cert
from ml.data.precrop import crop_kind


def _touch(root, rel):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Temp DB with one train cert per precrop situation."""
    settings = SimpleNamespace(data_dir=tmp_path, db_path=tmp_path / "t.sqlite")
    monkeypatch.setattr(dataset, "get_settings", lambda: settings)

    # cert -> (crop files written, found front/back, kind front/back)
    cases = {
        "ok_card": (True, (True, True), ("card", "card")),
        "ok_well": (True, (True, True), ("well", "card")),
        "not_found": (True, (False, True), ("none", "card")),
        "no_status": (True, (None, None), (None, None)),  # cropped before status existed
        "file_missing": (False, (True, True), ("card", "card")),
        "never_cropped": (None, None, None),
    }
    with connect(settings.db_path) as conn:
        for cert, (files, found, kind) in cases.items():
            front, back = f"images/{cert}_front.jpg", f"images/{cert}_back.jpg"
            _touch(tmp_path, front)
            _touch(tmp_path, back)
            upsert_cert(conn, {
                "cert_number": cert, "grade_int": 9, "is_pokemon": 1, "has_images": 1,
                "front_path": front, "back_path": back, "split": "train",
            })
            if files is None:
                continue
            cf, cb = f"crops/{front}", f"crops/{back}"
            if files:
                _touch(tmp_path, cf)
                _touch(tmp_path, cb)
            update_precrop_paths(
                conn, cert, cropped_front_path=cf, cropped_back_path=cb,
                found_front=found[0], found_back=found[1],
                kind_front=kind[0], kind_back=kind[1],
            )
    return settings


def test_precrop_keeps_only_detected_crops(db):
    recs = dataset.load_records("train", use_precrop=True)
    assert {r.cert_number for r in recs} == {"ok_card", "ok_well"}
    for r in recs:
        assert r.front_path.relative_to(db.data_dir).parts[0] == "crops"
        assert r.back_path.relative_to(db.data_dir).parts[0] == "crops"


def test_precrop_never_falls_back_to_raw_photo(db):
    for r in dataset.load_records("train", use_precrop=True):
        assert "crops" in r.front_path.parts and "crops" in r.back_path.parts


def test_drops_are_logged(db, caplog):
    with caplog.at_level("WARNING"):
        dataset.load_records("train", use_precrop=True)
    assert "dropped 4 / 6" in caplog.text


def test_crop_kinds_filter(db):
    recs = dataset.load_records("train", use_precrop=True, crop_kinds=("card",))
    assert [r.cert_number for r in recs] == ["ok_card"]


def test_raw_mode_returns_all_raw_photos(db):
    recs = dataset.load_records("train", use_precrop=False)
    assert len(recs) == 6
    assert all("crops" not in r.front_path.parts for r in recs)


def test_db_precrop_coverage_by_split(db):
    with connect(db.db_path) as conn:
        cov = precrop_coverage_by_split(conn)["train"]
    # DB-level counts (file existence is checked by load_records only)
    assert cov == {"total": 6, "cropped": 5, "found": 3, "card": 2}


# --------------------------------------------------------------------------- #
# train_probe.load_rows: the probe trains on usable precrops only
# --------------------------------------------------------------------------- #
def test_probe_training_rows_are_usable_precrops_only(db):
    from ml.training.train_probe import load_rows

    rows = load_rows(db.db_path)
    # Cut on a detected outline with both crop paths recorded; never a raw photo.
    assert {r["cert_number"] for r in rows} == {"ok_card", "ok_well", "file_missing"}
    assert all(r["cropped_front_path"] and r["cropped_back_path"] for r in rows)


# --------------------------------------------------------------------------- #
# precrop.crop_kind
# --------------------------------------------------------------------------- #
def _quad(top, bottom, left=60, right=540):
    return np.array([[left, top], [right, top], [right, bottom], [left, bottom]], np.float32)


def test_crop_kind():
    assert crop_kind(False, None, 1024) == "none"
    assert crop_kind(True, _quad(275, 940), 1024) == "card"   # 0.65 of height
    assert crop_kind(True, _quad(230, 985), 1024) == "well"   # 0.74 of height
    assert crop_kind(True, _quad(300, 880), 1024) == "small"  # 0.57 of height
    assert crop_kind(True, _quad(230, 985), 1024, source="ebay") == "unknown"
