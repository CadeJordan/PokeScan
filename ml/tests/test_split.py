"""ml/data/split.py: PSA certs are stratified, eBay certs stay in train."""

from __future__ import annotations

from collections import Counter

import pytest

from ml.data.split import assign_splits


def _psa_rows(n_per_grade=40):
    return [(f"{g}{i:04d}", g, "psa") for g in (7, 8, 9, 10) for i in range(n_per_grade)]


def _ebay_rows(n=200):
    return [(f"ebay:{i}", 1 + i % 10, "ebay") for i in range(n)]


def test_ebay_always_train():
    out = assign_splits(_psa_rows() + _ebay_rows())
    assert {out[c] for c, _, _ in _ebay_rows()} == {"train"}


def test_psa_still_gets_val_and_test():
    out = assign_splits(_psa_rows() + _ebay_rows())
    counts = Counter(out[c] for c, _, _ in _psa_rows())
    assert counts["val"] == counts["test"] == 4 * 6  # 15% of 40 per grade
    assert counts["train"] == 4 * 28


def test_adding_ebay_does_not_reshuffle_psa():
    psa = _psa_rows()
    alone = assign_splits(psa)
    mixed = assign_splits(psa[:80] + _ebay_rows() + psa[80:])
    assert all(alone[c] == mixed[c] for c, _, _ in psa)


def test_missing_source_treated_as_psa():
    rows = [(f"c{i}", 10, None) for i in range(20)]
    assert "test" in set(assign_splits(rows).values())


def test_stratify_mode_includes_ebay():
    out = assign_splits(_ebay_rows(), ebay_split="stratify")
    assert "test" in set(out.values())
    with pytest.raises(ValueError):
        assign_splits(_ebay_rows(), ebay_split="val")
