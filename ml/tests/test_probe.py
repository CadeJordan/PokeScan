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
