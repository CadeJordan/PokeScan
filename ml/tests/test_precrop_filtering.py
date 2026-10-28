"""Training never sees raw slab photos: load_records filtering + train guard.

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


def test_precrop_coverage(db):
    assert dataset.precrop_coverage("train") == (2, 6)
    with connect(db.db_path) as conn:
        cov = precrop_coverage_by_split(conn)["train"]
    # DB-level counts (file existence is checked by load_records only)
    assert cov == {"total": 6, "cropped": 5, "found": 3, "card": 2}


# --------------------------------------------------------------------------- #
# train.py guard
# --------------------------------------------------------------------------- #
def test_train_guard_rejects_raw_photos():
    from ml.training.train import check_training_inputs

    with pytest.raises(SystemExit, match="raw slab photos"):
        check_training_inputs(use_precrop=False, crop_card=False)


@pytest.mark.parametrize("kept,total,ok", [(90, 100, True), (89, 100, False), (0, 0, False)])
def test_train_guard_coverage(monkeypatch, kept, total, ok):
    from ml.training import train

    monkeypatch.setattr(train, "precrop_coverage", lambda split, kinds=None: (kept, total))
    if ok:
        train.check_training_inputs(use_precrop=True, crop_card=False)
    else:
        with pytest.raises(SystemExit, match="usable precrop"):
            train.check_training_inputs(use_precrop=True, crop_card=False)


def test_train_guard_skips_coverage_for_on_the_fly_crops(monkeypatch):
    from ml.training import train

    def boom(*a, **k):
        raise AssertionError("coverage should not be checked")

    monkeypatch.setattr(train, "precrop_coverage", boom)
    train.check_training_inputs(use_precrop=False, crop_card=True)


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
