"""Adversarial review tests for the sticky split and the probe-artifact changes.

Written by the independent tester (2026-10-27). Two tests here failed on the
branch until the review fixes; each docstring says why:

- test_split_legacy_not_filled_by_a_later_reassign
- test_failed_sidecar_write_leaves_no_stale_matching_sidecar

The rest pin edge cases that already behave correctly.
"""

from __future__ import annotations

import json
import sqlite3

import pytest
from typer.testing import CliRunner

from backend.app.ml import features
from ml.data.db import connect, upsert_cert
from ml.data.split import SplitRow, app, assign_grouped, assign_sticky, split_summary


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


def _psa(cert, group, split=None):
    return SplitRow(cert=cert, grade=9, source="psa", group=group, current_split=split)


def _ebay(cert, group, split=None):
    return SplitRow(cert=f"ebay:{cert}", grade=9, source="ebay", group=group, current_split=split)


# --------------------------------------------------------------------------- #
# split_legacy
# --------------------------------------------------------------------------- #
def test_split_legacy_not_filled_by_a_later_reassign(tmp_path):
    """FAILED BEFORE THE FIX. `split_legacy` is "the split a cert had before its first
    --reassign ... written once, never updated" (ml/data/db.py:72-74), and the
    README says later runs never overwrite it. But main() copies
    `split_legacy=split WHERE split_legacy IS NULL` on *every* reassign
    (ml/data/split.py:305), and NULL also means "had no split at the first
    reassign". So a cert that was unassigned (or harvested later and placed by
    sticky mode) gets its post-first-reassign split recorded as "legacy" by the
    second reassign, and `split_legacy='test'` no longer reproduces the old test set.
    """
    db = tmp_path / "t.sqlite"
    with connect(db) as conn:
        for i in range(60):
            _add_psa(conn, f"{i:08d}", i % 25, split=("train", "val", "test")[i % 3])
        _add_psa(conn, "new00001", 99)  # unassigned when the first reassign runs

    assert _run("--mode", "grouped", "--reassign", "--db-path", str(db)).exit_code == 0
    assert _col(db, "split_legacy")["new00001"] is None  # it had no split before
    first = _col(db, "split")["new00001"]
    assert first in ("train", "val", "test")

    assert _run(
        "--mode", "grouped", "--reassign", "--salt", "other", "--db-path", str(db)
    ).exit_code == 0
    # Still "no split before the first reassign", not the first reassign's choice.
    assert _col(db, "split_legacy")["new00001"] is None


# --------------------------------------------------------------------------- #
# sticky / grouped edge cases (pass)
# --------------------------------------------------------------------------- #
def test_new_psa_cert_of_an_ebay_only_card_is_hashed_and_ebay_follows():
    """A card with only eBay (train) certs has no PSA majority: the new PSA cert
    is placed by hash, and if that is test the eBay certs move to excluded."""
    for i in range(200):
        group = f"B|C{i}|1"
        out = assign_sticky([_ebay("a", group, split="train"), _psa("new", group)])
        alone = assign_sticky([_psa("new", group)])["new"]
        assert out["new"] == alone
        assert out["ebay:a"] == ("excluded" if alone == "test" else "train")


def test_sticky_is_idempotent_after_excluded_and_after_grouped():
    rows = [_psa(f"p{i}", f"B|C{i % 40}|1") for i in range(200)]
    rows += [_ebay(f"{i}", f"B|C{i % 50}|1") for i in range(60)]
    for first in (assign_sticky(rows), assign_grouped(rows)):
        placed = [SplitRow(r.cert, r.grade, r.source, r.group, first[r.cert]) for r in rows]
        assert "excluded" in first.values()
        again = assign_sticky(placed)
        assert again == first
        assert split_summary(placed, again)["changed"] == 0


def test_dry_run_on_missing_db_creates_nothing(tmp_path):
    db = tmp_path / "nope.sqlite"
    result = _run("--dry-run", "--db-path", str(db))
    assert result.exit_code != 0
    assert not db.exists()


def test_dry_run_reassign_leaves_split_legacy_alone(tmp_path):
    db = tmp_path / "t.sqlite"
    with connect(db) as conn:
        for i in range(30):
            _add_psa(conn, f"{i:08d}", i % 7, split="train")
    before = db.read_bytes()
    assert _run("--mode", "grouped", "--reassign", "--dry-run", "--db-path", str(db)).exit_code == 0
    assert db.read_bytes() == before
    assert set(_col(db, "split_legacy").values()) == {None}


# --------------------------------------------------------------------------- #
# Feature graph export: the sidecar must never vouch for a graph it didn't describe
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch")


class _Tiny(torch.nn.Module):
    def forward(self, x):
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return pooled[:, None, :].repeat(1, len(features.REGION_NAMES), 1)


def test_failed_sidecar_write_leaves_no_stale_matching_sidecar(tmp_path, monkeypatch):
    """FAILED BEFORE THE FIX. export_onnx overwrites `card_features.onnx` in place and
    only then rewrites `card_features.json` (backend/app/ml/features.py:142-155).
    If the process dies or the sidecar write fails in between, the new graph sits
    next to the *old* sidecar. After a FEATURE_SPEC change that old sidecar still
    carries the old hash, which equals the old head's hash, so
    `_check_feature_spec` passes and the API serves the old head on a graph built
    from another spec. Fix: remove the old sidecar before exporting (or write
    graph + sidecar to temp names and rename).
    """
    monkeypatch.setattr(features, "build_region_model", lambda pretrained=True: _Tiny().eval())
    path = tmp_path / "card_features.onnx"
    sidecar = path.with_suffix(".json")
    old_hash = "0ld0ld0ld0ld"
    path.write_bytes(b"old graph")
    sidecar.write_text(json.dumps({"feature_spec_hash": old_hash}))

    real_dumps = json.dumps

    def failing_dumps(obj, *a, **k):
        if isinstance(obj, dict) and "exported_at" in obj:
            raise OSError("disk full")
        return real_dumps(obj, *a, **k)

    monkeypatch.setattr(features.json, "dumps", failing_dumps)
    with pytest.raises(OSError):
        features.export_onnx(path)
    monkeypatch.setattr(features.json, "dumps", real_dumps)

    graph_replaced = path.read_bytes() != b"old graph"
    stale = sidecar.exists() and json.loads(sidecar.read_text()).get("feature_spec_hash") == old_hash
    assert not (graph_replaced and stale), "new graph left beside the old graph's sidecar"
