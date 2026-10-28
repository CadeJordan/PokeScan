"""ml/data/groups.py: the split and the card-grouped CV use the same card key,
and moving the key out of `probe.card_groups` didn't change any group."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from backend.app.core.config import get_settings
from ml.data.groups import card_group_key, group_bucket
from ml.training.probe import card_groups


def _card_groups_before(brand, subject, number):
    """`probe.card_groups` as it was before the refactor."""
    return np.array([f"{b}|{s}|{n}" for b, s, n in zip(brand, subject, number, strict=True)])


def test_card_groups_unchanged_by_refactor():
    brand = ["A", "A", "", "POKEMON JAPANESE", "B|C", "A"]
    subject = ["PIKACHU", "PIKACHU", "MEW", "CHARIZARD-HOLO", "X", ""]
    number = ["1", "2", "", "006", "7", "1"]
    assert (card_groups(brand, subject, number) == _card_groups_before(brand, subject, number)).all()


def test_group_key_treats_missing_as_empty():
    # train_probe passes `or ""`; split.py passes the raw (possibly NULL) columns.
    assert card_group_key(None, "MEW", None) == card_group_key("", "MEW", "") == "|MEW|"


def test_group_bucket_is_stable_and_salted():
    b = group_bucket("A|PIKACHU|1")
    assert 0 <= b < 1
    assert b == group_bucket("A|PIKACHU|1", salt="pokescan-split-v1")
    assert b != group_bucket("A|PIKACHU|1", salt="other")


@pytest.mark.slow
def test_card_groups_unchanged_on_real_db():
    """Reads only: the DB is opened read-only and nothing is written."""
    s = get_settings()
    if not s.db_path.exists():
        pytest.skip("no DB")
    conn = sqlite3.connect(f"file:{s.db_path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT brand_title, subject, card_number FROM certs").fetchall()
    finally:
        conn.close()
    assert rows
    # As train_probe calls it.
    cols = [[v or "" for v in col] for col in zip(*rows, strict=True)]
    groups = card_groups(*cols)
    assert (groups == _card_groups_before(*cols)).all()
    # split.py keys the raw columns; they must match the CV groups cert for cert.
    assert [card_group_key(*r) for r in rows] == list(groups)
