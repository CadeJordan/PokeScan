"""Fitting the linear ordinal grade head (ml/training/probe.py) on synthetic features."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from backend.app.ml.features import REGION_NAMES
from backend.app.ml.grade_head import PROBE_NAMES
from ml.eval.evaluate import quadratic_weighted_kappa
from ml.training import probe as probe_mod
from ml.training.probe import card_groups, cross_validate, fit_grade_head, fit_ordinal_probe

D = 6


def _synthetic(n: int = 400, seed: int = 0):
    """Features where a latent condition score drives every region, plus noise."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    grade = np.clip(np.round(8.5 + 1.2 * latent), 1, 10).astype(int)
    direction = rng.normal(size=(2, len(REGION_NAMES), D))
    feats = latent[:, None, None, None] * direction + rng.normal(size=(n, 2, len(REGION_NAMES), D))
    groups = np.array([f"card{i // 2}" for i in range(n)])  # two certs per printed card
    return feats.astype(np.float32), grade, groups


def test_folded_probe_matches_explicit_pipeline(monkeypatch):
    """Scaler + whitened PCA + PO folded into (weight, bias) gives the same scores."""
    captured = {}
    real = probe_mod._fit_po

    def spy(z, y, l2, max_iter=300):
        w, th = real(z, y, l2, max_iter)
        captured["w"] = w
        return w, th

    monkeypatch.setattr(probe_mod, "_fit_po", spy)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 20)) * rng.uniform(0.5, 3, 20) + rng.normal(size=20)
    y = np.clip(np.round(8 + x[:, 0]), 1, 10).astype(int)
    probe = fit_ordinal_probe(x, y, pca_dim=5, l2=1e-3)

    z = StandardScaler().fit(x).transform(x)
    explicit = PCA(5, whiten=True, random_state=0).fit(z).transform(z) @ captured["w"]
    np.testing.assert_allclose(probe.score(x), explicit, atol=1e-8)


def test_probe_learns_ordering():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 10))
    y = np.clip(np.round(6 + 2 * x[:, 0]), 1, 10).astype(int)
    p = fit_ordinal_probe(x, y, pca_dim=None, l2=1e-3)
    assert np.corrcoef(p.soft_grade(x), y)[0, 1] > 0.9
    assert (np.diff(p.thresholds) > 0).all()


def test_grade_head_fit_and_cv():
    feats, y, groups = _synthetic()
    head = fit_grade_head(feats, y, groups)
    assert set(head.probes) == set(PROBE_NAMES)
    out = head(feats[0])
    assert out.cum_probs.shape == (9,)

    cum, regions = cross_validate(feats, y, groups, n_folds=3)
    hard = 1 + (cum > 0.5).sum(1)
    assert regions.shape == (len(y), len(PROBE_NAMES))
    assert quadratic_weighted_kappa(y, hard) > 0.5


def test_cv_keeps_card_groups_together(monkeypatch):
    feats, y, groups = _synthetic(n=120)
    seen = []
    real = probe_mod.fit_grade_head

    def spy(f, yy, g, metadata=None):
        seen.append(set(g))
        return real(f, yy, g, metadata)

    monkeypatch.setattr(probe_mod, "fit_grade_head", spy)
    cross_validate(feats, y, groups, n_folds=3)
    all_groups = set(groups)
    for train_groups in seen:
        held_out = all_groups - train_groups
        assert held_out and not (held_out & train_groups)


@pytest.mark.parametrize(
    ("brand", "subject", "number", "same"),
    [("A", "PIKACHU", "1", True), ("A", "PIKACHU", "2", False), ("B", "PIKACHU", "1", False)],
)
def test_card_groups(brand, subject, number, same):
    g = card_groups(["A", brand], ["PIKACHU", subject], ["1", number])
    assert (g[0] == g[1]) is same


def _graded_cards(seed: int, n_cards: int = 900):
    """Card-grouped grades like train+val: ~30% 9s among 9s+10s, some 8s and <=7s, 1-4 certs a card."""
    rng = np.random.default_rng(seed)
    y, groups = [], []
    for card, size in enumerate(rng.choice([1, 1, 1, 2, 2, 3, 4], n_cards)):
        base = rng.choice([10, 9, 8, 7, 6, 5, 3], p=[0.58, 0.27, 0.08, 0.03, 0.02, 0.01, 0.01])
        for _ in range(size):  # certs of one card mostly share its grade
            other = rng.choice([10, 9, 8, 7, 5], p=[0.58, 0.27, 0.08, 0.04, 0.03])
            y.append(base if rng.random() < 0.7 else other)
            groups.append(f"card{card}")
    return np.array(y), np.array(groups)


@pytest.mark.parametrize("data_seed", [0, 1, 2])
def test_folds_balance_9_vs_10_and_spread_8s_and_low_grades(data_seed):
    """Outer CV folds and the meta probe's inner OOF folds, at the seeds their callers use."""
    import inspect

    from ml.training.probe import INNER_FOLDS, oof_region_grades

    outer_seed = inspect.signature(cross_validate).parameters["seed"].default
    inner_seed = inspect.signature(oof_region_grades).parameters["seed"].default
    y, groups = _graded_cards(data_seed)

    def share9(yy):
        return (yy[yy >= 9] == 9).mean()

    def check(yy, gg, n, seed):
        folds = probe_mod._folds(yy, gg, n, seed)
        assert len(folds) == n
        for _, te in folds:
            assert abs(share9(yy[te]) - share9(yy)) <= 0.03
            for stratum in (yy == 8, yy <= 7):  # each fold gets its share, give or take 2
                assert abs(stratum[te].sum() - stratum.sum() / n) <= 2
        return folds

    for tr, _ in check(y, groups, 5, outer_seed):
        check(y[tr], groups[tr], INNER_FOLDS, inner_seed)  # inner folds inside each outer fold
    check(y, groups, INNER_FOLDS, inner_seed)  # inner folds of the served head (all train+val)
