"""ml/data/split.py sticky and grouped modes: re-running never moves an assigned
cert, and a card-grouped re-split keeps every printed card in one split."""

from __future__ import annotations

import hashlib
import random
import sqlite3
from collections import Counter
from dataclasses import replace

import pytest
from typer.testing import CliRunner

from ml.data.db import connect, upsert_cert
from ml.data.split import SplitRow, app, assign_grouped, assign_sticky, split_summary


def _psa(cert, group, grade=9, split=None):
    return SplitRow(cert=cert, grade=grade, source="psa", group=group, current_split=split)


def _ebay(cert, group, split=None):
    return SplitRow(cert=f"ebay:{cert}", grade=9, source="ebay", group=group, current_split=split)


def _cards(n_certs, start=0, n_cards=200, seed=0):
    """`n_certs` PSA certs over `n_cards` printed cards (several certs per card)."""
    rng = random.Random(seed)
    return [
        _psa(f"{start + i:08d}", f"BRAND|CARD{rng.randrange(n_cards)}|1", grade=rng.randint(5, 10))
        for i in range(n_certs)
    ]


def _resplit(rows, assigned):
    """Rows as the DB holds them after `assigned` was written."""
    return [replace(r, current_split=assigned.get(r.cert, r.current_split)) for r in rows]


# --------------------------------------------------------------------------- #
# Pure functions
# --------------------------------------------------------------------------- #
def test_adding_certs_never_moves_existing_ones():
    rows = _cards(500)
    first = assign_sticky(rows)
    assert set(first.values()) == {"train", "val", "test"}
    rows = _resplit(rows, first)

    # 1 new cert of a known card, then 50 more (known and new cards).
    for added in (_cards(1, start=500, seed=1), _cards(50, start=501, n_cards=400, seed=2)):
        rows = rows + added
        out = assign_sticky(rows)
        moved = [c for c, s in first.items() if out[c] != s]
        assert moved == []
        assert all(out[r.cert] in ("train", "val", "test") for r in added)
        rows = _resplit(rows, out)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_new_cert_joins_its_cards_split(split):
    rows = [_psa("1", "A|X|1", split=split), _psa("2", "A|X|1")]
    assert assign_sticky(rows)["2"] == split


def test_straddling_card_takes_majority_then_test_first():
    majority = [_psa("1", "g", split="train"), _psa("2", "g", split="train"),
                _psa("3", "g", split="test"), _psa("new", "g")]
    assert assign_sticky(majority)["new"] == "train"
    tie = [_psa("1", "g", split="train"), _psa("2", "g", split="test"), _psa("new", "g")]
    assert assign_sticky(tie)["new"] == "test"
    tie = [_psa("1", "g", split="train"), _psa("2", "g", split="val"), _psa("new", "g")]
    assert assign_sticky(tie)["new"] == "val"


def test_new_card_shares_match_fractions():
    rows = [_psa(f"{i:08d}", f"BRAND|CARD{i}|{i}") for i in range(3000)]
    counts = Counter(assign_sticky(rows).values())
    assert counts["test"] / 3000 == pytest.approx(0.15, abs=0.02)
    assert counts["val"] / 3000 == pytest.approx(0.15, abs=0.02)


def _card_in(split):
    """A group key that hashes into `split` (default fractions, default salt)."""
    for i in range(1000):
        key = f"B|CARD{i}|1"
        if assign_sticky([_psa("x", key)])["x"] == split:
            return key
    raise AssertionError(split)


@pytest.mark.parametrize("assign", [assign_sticky, assign_grouped])
def test_ebay_of_a_test_card_is_excluded(assign):
    test_card, train_card = _card_in("test"), _card_in("train")
    rows = [
        _psa("1", test_card), _psa("2", train_card),
        _ebay("a", test_card), _ebay("b", train_card), _ebay("c", "EBAY|ONLY|1"),
        _ebay("d", test_card, split="train"),  # was train before its card entered test
    ]
    out = assign(rows)
    assert out["1"] == "test" and out["2"] == "train"
    assert out["ebay:a"] == out["ebay:d"] == "excluded"
    assert out["ebay:b"] == out["ebay:c"] == "train"


def test_assigned_ebay_keeps_its_split_in_sticky():
    rows = [_psa("1", "g", split="train"), _ebay("a", "g", split="val"),
            _ebay("b", "h", split="excluded")]
    out = assign_sticky(rows)
    assert out["ebay:a"] == "val" and out["ebay:b"] == "excluded"


@pytest.mark.parametrize("assign", [assign_sticky, assign_grouped])
def test_row_order_does_not_matter(assign):
    base = _cards(300)
    first = assign_sticky(base)
    rows = _resplit(base, first) + _cards(200, start=300, n_cards=400, seed=3)
    rows += [_ebay(i, r.group) for i, r in enumerate(rows[::7])]
    out = assign(rows)
    for seed in range(3):
        shuffled = rows[:]
        random.Random(seed).shuffle(shuffled)
        assert assign(shuffled) == out


def test_grouped_keeps_cards_in_one_split():
    rows = _resplit(_cards(500), assign_sticky(_cards(500)))
    out = assign_grouped(rows)
    per_group: dict[str, set] = {}
    for r in rows:
        per_group.setdefault(r.group, set()).add(out[r.cert])
    assert all(len(s) == 1 for s in per_group.values())
    assert split_summary(rows, out)["test_straddle"] == 0


def test_split_summary_counts():
    rows = [_psa("1", "g", grade=7, split="train"), _psa("2", "g", grade=9, split="train"),
            _psa("3", "h", grade=10)]
    out = {"1": "train", "2": "test", "3": "test"}
    s = split_summary(rows, out)
    assert s["histogram"] == {"test": {9: 1, 10: 1}, "train": {7: 1}}
    assert s["test_straddle"] == 1  # cert 2 shares card g with train
    assert s["changed"] == 2


# --------------------------------------------------------------------------- #
# CLI on a temp DB
# --------------------------------------------------------------------------- #
@pytest.fixture
def db(tmp_path):
    """Temp DB: 300 PSA certs over 120 cards, 40 eBay certs, all unassigned."""
    path = tmp_path / "t.sqlite"
    rng = random.Random(0)
    with connect(path) as conn:
        for i in range(300):
            card = rng.randrange(120)
            upsert_cert(conn, {
                "cert_number": f"{i:08d}", "source": "psa", "grade_int": rng.randint(5, 10),
                "is_pokemon": 1, "has_images": 1,
                "brand_title": "BRAND", "subject": f"CARD{card}", "card_number": str(card),
            })
        for i in range(40):
            card = rng.randrange(120)
            upsert_cert(conn, {
                "cert_number": f"ebay:{i}", "source": "ebay", "grade_int": rng.randint(5, 10),
                "is_pokemon": 1, "has_images": 1,
                "brand_title": "BRAND", "subject": f"CARD{card}", "card_number": str(card),
            })
    return path


def _run(*args):
    return CliRunner().invoke(app, [*args])


def _splits(path, col="split"):
    conn = sqlite3.connect(path)
    try:
        return dict(conn.execute(f"SELECT cert_number, {col} FROM certs").fetchall())
    finally:
        conn.close()


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cli_sticky_rerun_changes_nothing(db):
    assert _run("--db-path", str(db)).exit_code == 0
    first = _splits(db)
    assert None not in first.values()
    with connect(db) as conn:
        for i in range(300, 350):
            upsert_cert(conn, {
                "cert_number": f"{i:08d}", "source": "psa", "grade_int": 9,
                "is_pokemon": 1, "has_images": 1,
                "brand_title": "BRAND", "subject": f"CARD{i % 150}", "card_number": str(i % 150),
            })
    assert _run("--db-path", str(db)).exit_code == 0
    second = _splits(db)
    assert all(second[c] == s for c, s in first.items())
    assert _splits(db, "split_legacy") == dict.fromkeys(second)  # sticky never fills it


def test_cli_grouped_reassign_keeps_legacy(db):
    # Start from today's per-cert split (sticky on fresh rows is already grouped).
    conn = sqlite3.connect(db)
    rng = random.Random(1)
    conn.executemany(
        "UPDATE certs SET split=? WHERE cert_number=?",
        [(rng.choice(["train", "train", "val", "test"]), c)
         for (c,) in conn.execute("SELECT cert_number FROM certs WHERE source='psa'").fetchall()],
    )
    conn.execute("UPDATE certs SET split='train' WHERE source='ebay'")
    conn.commit()
    conn.close()
    before = _splits(db)

    assert _run("--mode", "grouped", "--reassign", "--db-path", str(db)).exit_code == 0
    after, legacy = _splits(db), _splits(db, "split_legacy")
    assert legacy == before
    assert any(after[c] != before[c] for c in before)  # the re-split moved something

    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT source, brand_title, subject, card_number, split FROM certs"
    ).fetchall()
    conn.close()
    psa_card: dict[tuple, set] = {}
    any_card: dict[tuple, set] = {}
    for src, b, s, n, sp in rows:
        if src == "psa":
            psa_card.setdefault((b, s, n), set()).add(sp)
        any_card.setdefault((b, s, n), set()).add(sp)
    assert all(len(s) == 1 for s in psa_card.values())
    # eBay rows only ever go to train, so a val card can have eBay train certs,
    # but a test card has nothing in train or val.
    assert all(s <= {"test", "excluded"} for s in any_card.values() if "test" in s)
    assert "excluded" in {sp for *_, sp in rows}

    # A second re-split (different salt, so it moves certs again) keeps the original.
    assert _run(
        "--mode", "grouped", "--reassign", "--salt", "other", "--db-path", str(db)
    ).exit_code == 0
    assert _splits(db) != after
    assert _splits(db, "split_legacy") == before


@pytest.mark.parametrize("mode", ["stratified", "grouped"])
def test_cli_reassign_modes_refuse_without_reassign(db, mode):
    digest = _digest(db)
    result = _run("--mode", mode, "--db-path", str(db))
    assert result.exit_code != 0
    assert _digest(db) == digest


def test_cli_stratified_reassign_runs(db):
    assert _run("--mode", "stratified", "--reassign", "--db-path", str(db)).exit_code == 0
    out = _splits(db)
    assert {out[f"ebay:{i}"] for i in range(40)} == {"train"}
    assert {"train", "val", "test"} <= set(out.values())


def test_cli_ebay_stratify_needs_stratified_mode(db):
    digest = _digest(db)
    assert _run("--ebay-split", "stratify", "--db-path", str(db)).exit_code != 0
    assert _digest(db) == digest


def test_cli_dry_run_writes_nothing(db, caplog):
    digest = _digest(db)
    with caplog.at_level("INFO", logger="split"):
        assert _run("--dry-run", "--db-path", str(db)).exit_code == 0
        assert _run("--mode", "grouped", "--reassign", "--dry-run", "--db-path", str(db)).exit_code == 0
    assert _digest(db) == digest
    assert set(_splits(db).values()) == {None}
    assert "changed split: 340 of 340 rows" in caplog.text
