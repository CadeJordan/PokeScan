"""Independent re-test of commit a92911c (split_legacy first-only; sidecar-first export).

All DBs and model dirs live in tmp_path.
"""

from __future__ import annotations

import json
import sqlite3

import pytest
from typer.testing import CliRunner

from backend.app.ml import features
from backend.app.ml.features import feature_spec_hash
from ml.data.db import connect, upsert_cert
from ml.data.split import app
from ml.training import train_probe


def _run(*args):
    return CliRunner().invoke(app, [*args])


def _col(path, col):
    conn = sqlite3.connect(path)
    try:
        return dict(conn.execute(f"SELECT cert_number, {col} FROM certs").fetchall())
    finally:
        conn.close()


def _add_psa(conn, cert, card, grade=9, split=None):
    row = {
        "cert_number": cert, "source": "psa", "grade_int": grade, "is_pokemon": 1,
        "has_images": 1, "brand_title": "BRAND", "subject": f"CARD{card}",
        "card_number": str(card),
    }
    if split is not None:
        row["split"] = split
    upsert_cert(conn, row)


# --------------------------------------------------------------------------- #
# split_legacy
# --------------------------------------------------------------------------- #
def test_split_legacy_survives_stratified_then_grouped_then_sticky(tmp_path):
    db = tmp_path / "t.sqlite"
    with connect(db) as conn:
        for i in range(60):
            _add_psa(conn, f"{i:08d}", i % 25, grade=8 + i % 3,
                     split=("train", "val", "test")[i % 3])
    original = _col(db, "split")

    assert _run("--mode", "stratified", "--reassign", "--db-path", str(db)).exit_code == 0
    assert _col(db, "split_legacy") == original
    with connect(db) as conn:
        _add_psa(conn, "late0001", 3)  # arrives after the first reassign
    assert _run("--db-path", str(db)).exit_code == 0  # sticky places it
    assert _col(db, "split")["late0001"] in ("train", "val", "test")
    assert _run("--mode", "grouped", "--reassign", "--db-path", str(db)).exit_code == 0
    legacy = _col(db, "split_legacy")
    assert legacy == {**original, "late0001": None}


def test_first_reassign_with_every_cert_unassigned_documented_edge_case(tmp_path):
    # Spec (stable-grouped-split, Scope 2026-10-27): if every cert was unassigned
    # at the first reassign, the column stays all NULL and the next reassign
    # counts as the first. Pin that, so it's a known, deliberate behaviour.
    db = tmp_path / "t.sqlite"
    with connect(db) as conn:
        for i in range(40):
            _add_psa(conn, f"{i:08d}", i % 15)
    assert _run("--mode", "grouped", "--reassign", "--db-path", str(db)).exit_code == 0
    assert set(_col(db, "split_legacy").values()) == {None}
    first = _col(db, "split")
    assert None not in first.values()
    assert _run(
        "--mode", "grouped", "--reassign", "--salt", "x", "--db-path", str(db)
    ).exit_code == 0
    # The second reassign records the first reassign's result as "legacy".
    assert _col(db, "split_legacy") == first


def test_failed_reassign_leaves_split_legacy_unwritten(tmp_path, monkeypatch):
    # The legacy copy and the split update share one transaction: a crash in
    # between must not leave split_legacy filled (which would make the retry
    # skip the copy) while split is unchanged.
    from ml.data import split as split_mod

    db = tmp_path / "t.sqlite"
    with connect(db) as conn:
        for i in range(30):
            _add_psa(conn, f"{i:08d}", i % 10, split=("train", "val", "test")[i % 3])

    def boom(_conn):
        raise RuntimeError("crash after the legacy copy")

    monkeypatch.setattr(split_mod, "grade_distribution", boom)
    result = _run("--mode", "grouped", "--reassign", "--db-path", str(db))
    assert result.exit_code != 0
    assert set(_col(db, "split_legacy").values()) == {None}


# --------------------------------------------------------------------------- #
# export_onnx writes temp files first (smallfix 12); train_probe follows
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch")


def test_export_failing_in_torch_export_keeps_old_pair_and_train_probe_skips_reexport(
    tmp_path, monkeypatch
):
    # Until smallfix 12 the old sidecar was deleted before exporting, so this
    # left no sidecar. Now the old, consistent pair stays as it was.
    path = tmp_path / "card_features.onnx"
    path.write_bytes(b"old graph")
    path.with_suffix(".json").write_text(json.dumps({"feature_spec_hash": feature_spec_hash()}))
    assert train_probe._onnx_export_reason(path, force=False) is None

    monkeypatch.setattr(features, "build_region_model", lambda pretrained=True: object())

    def fail(*a, **k):
        raise RuntimeError("trace failed")

    monkeypatch.setattr(torch.onnx, "export", fail)
    with pytest.raises(RuntimeError, match="trace failed"):
        features.export_onnx(path)
    assert path.read_bytes() == b"old graph"
    assert train_probe._onnx_export_reason(path, force=False) is None


def test_train_probe_with_real_export_failing_writes_no_head(tmp_path, monkeypatch):
    # Like test_failed_export_writes_no_head, but through the real export_onnx
    # (only torch.onnx.export fails), with an old head + matching sidecar present.
    from types import SimpleNamespace

    import numpy as np

    from backend.app.ml.features import REGION_NAMES

    rng = np.random.default_rng(0)
    n = 120
    latent = rng.normal(size=n)
    grade = np.clip(np.round(8.5 + 1.2 * latent), 7, 10).astype(int)
    feats = latent[:, None, None, None] * rng.normal(size=(2, len(REGION_NAMES), 4))
    feats = feats + rng.normal(size=(n, 2, len(REGION_NAMES), 4))
    rows = [
        {"cert_number": f"c{i}", "grade_int": int(grade[i]),
         "split": "test" if i % 5 == 0 else "train", "brand_title": "set",
         "subject": f"card{i // 2}", "card_number": "1"}
        for i in range(n)
    ]
    models = tmp_path / "models"
    models.mkdir()
    settings = SimpleNamespace(db_path=tmp_path / "t.sqlite", data_dir=tmp_path, models_dir=models)
    monkeypatch.setattr(train_probe, "get_settings", lambda: settings)
    monkeypatch.setattr(train_probe, "load_rows", lambda _db: rows)
    monkeypatch.setattr(train_probe, "extract_features", lambda *a, **k: feats.astype(np.float32))
    monkeypatch.setattr(train_probe, "REPORT_PATH", tmp_path / "probe_report.json")

    (models / "card_features.onnx").write_bytes(b"old graph")
    (models / "card_features.json").write_text(json.dumps({"feature_spec_hash": "0ld0ld0ld0ld"}))
    (models / "grade_head.npz").write_bytes(b"old head")
    monkeypatch.setattr(features, "build_region_model", lambda pretrained=True: object())

    def fail(*a, **k):
        raise RuntimeError("trace failed")

    monkeypatch.setattr(torch.onnx, "export", fail)
    with pytest.raises(RuntimeError, match="trace failed"):
        train_probe.main(force_onnx=False, cv_folds=0)
    assert (models / "grade_head.npz").read_bytes() == b"old head"
    # The failed export leaves the old graph + sidecar as they were (smallfix 12).
    assert (models / "card_features.onnx").read_bytes() == b"old graph"
    assert json.loads((models / "card_features.json").read_text()) == {
        "feature_spec_hash": "0ld0ld0ld0ld"
    }
