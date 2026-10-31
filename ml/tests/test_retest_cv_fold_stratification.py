"""Re-test of cv-fold-stratification: `probe._folds` stratifies on <=7 / 8 / 9 / 10 (synthetic data only)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from ml.tests import test_probe
from ml.training import probe as probe_mod


def _old_folds(y, groups, n, seed):
    skf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), np.minimum(y, 8), groups))


def _maximum8_folds(y, groups, n, seed):
    skf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), np.maximum(y, 8), groups))


@pytest.mark.parametrize("bad", [_old_folds, _maximum8_folds], ids=["minimum8", "maximum8"])
def test_balance_test_rejects_other_keys(monkeypatch, bad):
    """The new guard fails under the old key (9s and 10s pooled) and under <=8 pooled (<=7s swing)."""
    monkeypatch.setattr(probe_mod, "_folds", bad)
    failed = 0
    for data_seed in (0, 1, 2):
        try:
            test_probe.test_folds_balance_9_vs_10_and_spread_8s_and_low_grades(data_seed)
        except AssertionError:
            failed += 1
    assert failed == 3


def test_folds_balance_9_vs_10_on_a_9_10_only_set():
    """The smallfix case: on 9s and 10s only, the old key stratified nothing."""
    rng = np.random.default_rng(7)
    sizes = rng.choice([1, 1, 2, 3], 600)
    y = np.concatenate([np.full(s, rng.choice([10, 9], p=[0.7, 0.3])) for s in sizes])
    groups = np.concatenate([[f"c{i}"] * s for i, s in enumerate(sizes)])
    share = (y == 9).mean()
    for n, seed in ((5, 0), (4, 1)):
        folds = probe_mod._folds(y, groups, n, seed)
        assert all(abs((y[te] == 9).mean() - share) <= 0.02 for _, te in folds)
        assert all(not set(groups[tr]) & set(groups[te]) for tr, te in folds)


def test_rare_low_grades_are_pooled_without_sklearn_warning():
    """A single grade-1 cert no longer forms its own stratum (no 'least populated class' warning)."""
    y, groups = test_probe._graded_cards(0)
    y = y.copy()
    y[0] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        folds = probe_mod._folds(y, groups, 5, 0)
    assert sorted(np.concatenate([te for _, te in folds]).tolist()) == list(range(len(y)))
