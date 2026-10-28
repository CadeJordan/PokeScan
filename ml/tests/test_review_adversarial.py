"""Adversarial review tests for ml/training/probe.py (fitting the ordinal grade head).

Includes defects found in review and since fixed: rank-deficient PCA
whitening and out-of-range grades.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from backend.app.ml.features import REGION_NAMES
from backend.app.ml.grade_head import PROBE_NAMES
from ml.training import probe as probe_mod
from ml.training.probe import cross_validate, fit_ordinal_probe, oof_region_grades

D = 6


@pytest.fixture
def capture_w(monkeypatch):
    captured = {}
    real = probe_mod._fit_po

    def spy(z, y, l2, max_iter=300):
        w, th = real(z, y, l2, max_iter)
        captured["w"], captured["z"] = w, z
        return w, th

    monkeypatch.setattr(probe_mod, "_fit_po", spy)
    return captured


def _grades(x0: np.ndarray, centre: float = 8.0, scale: float = 1.5) -> np.ndarray:
    return np.clip(np.round(centre + scale * x0), 1, 10).astype(int)


def _explicit_scores(x, w, pca_dim):
    scaler = StandardScaler().fit(x)
    z = scaler.transform(np.asarray(x, np.float64))
    if pca_dim and pca_dim < min(z.shape):
        z = PCA(pca_dim, whiten=True, random_state=0).fit(z).transform(z)
    return z @ w


# --------------------------------------------------------------------------- #
# Folding scaler + PCA into (weight, bias)
# --------------------------------------------------------------------------- #
def _case(name: str):
    rng = np.random.default_rng(0)
    if name == "zero_var_cols_pca":
        x = rng.normal(size=(200, 30))
        x[:, 5:25] = 3.0
        return x, _grades(x[:, 0]), 8
    if name == "no_pca":
        x = rng.normal(size=(200, 30)) * rng.uniform(0.2, 5, 30) + 7
        return x, _grades(x[:, 0] - 7), None
    if name == "pca_eq_features":
        x = rng.normal(size=(200, 30))
        return x, _grades(x[:, 0]), 30
    if name == "pca_gt_features":
        x = rng.normal(size=(200, 30))
        return x, _grades(x[:, 0]), 128
    if name == "pca_gt_samples":
        x = rng.normal(size=(40, 200))
        return x, _grades(x[:, 0]), 128
    if name == "large_offset_float32":
        x = (rng.normal(size=(300, 20)) * 0.01 + 1000).astype(np.float32)
        return x, _grades((x[:, 0] - 1000) * 100), 5
    raise KeyError(name)


@pytest.mark.parametrize(
    "name",
    ["zero_var_cols_pca", "no_pca", "pca_eq_features", "pca_gt_features", "pca_gt_samples",
     "large_offset_float32"],
)
def test_folded_scores_match_explicit_pipeline(capture_w, name):
    x, y, pca_dim = _case(name)
    probe = fit_ordinal_probe(x, y, pca_dim=pca_dim, l2=1e-3)
    explicit = _explicit_scores(x, capture_w["w"], pca_dim)
    scale = max(1.0, np.abs(explicit).max())
    np.testing.assert_allclose(probe.score(x), explicit, atol=1e-6 * scale)
    assert np.isfinite(probe.weight).all() and np.isfinite(probe.bias)


def test_folding_generalises_to_new_points(capture_w):
    """Folding must hold for unseen x too, not just the training rows."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=(150, 12)) * rng.uniform(0.5, 4, 12) + rng.normal(size=12)
    y = _grades(x[:, 0])
    probe = fit_ordinal_probe(x, y, pca_dim=6, l2=1e-3)
    xn = rng.normal(size=(30, 12)) * 3
    z = StandardScaler().fit(x).transform(xn)
    pca = PCA(6, whiten=True, random_state=0).fit(StandardScaler().fit(x).transform(x))
    np.testing.assert_allclose(probe.score(xn), pca.transform(z) @ capture_w["w"], atol=1e-8)


def test_rank_deficient_features_give_bounded_weights():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3)) @ rng.normal(size=(3, 40))
    y = _grades(x[:, 0], scale=1.0)
    probe = fit_ordinal_probe(x, y, pca_dim=10)
    assert np.abs(probe.weight).max() < 1e6
    assert np.abs(probe.score(rng.normal(size=(5, 40)))).max() < 1e3


# --------------------------------------------------------------------------- #
# Proportional-odds fit on awkward label sets
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "labels",
    ["only_9_10", "only_10", "only_1", "gap_no_5_to_7", "two_extremes", "tiny_n3"],
)
def test_po_fit_degenerate_labels(labels):
    rng = np.random.default_rng(1)
    n = 3 if labels == "tiny_n3" else 200
    x = rng.normal(size=(n, 6))
    y = {
        "only_9_10": np.where(x[:, 0] > 0, 10, 9),
        "only_10": np.full(n, 10),
        "only_1": np.full(n, 1),
        "gap_no_5_to_7": np.where(x[:, 0] > 0.3, 9, np.where(x[:, 0] > -0.3, 8, 3)),
        "two_extremes": np.where(x[:, 0] > 0, 10, 1),
        "tiny_n3": np.array([1, 5, 10]),
    }[labels]
    p = fit_ordinal_probe(x, y, pca_dim=None)
    assert np.isfinite(p.weight).all() and np.isfinite(p.bias)
    assert np.isfinite(p.thresholds).all()
    assert (np.diff(p.thresholds) > 0).all()
    cum = p.cum_probs(x)
    assert np.isfinite(cum).all()
    assert (np.diff(cum, axis=1) <= 1e-12).all()
    soft = p.soft_grade(x)
    assert ((soft >= 1) & (soft <= 10)).all()
    if labels == "only_9_10":
        assert soft.min() >= 8.9
    if labels in ("only_10", "only_1"):
        assert np.allclose(soft, y[0], atol=0.05)


@pytest.mark.parametrize("bad", [0, 8.5])
def test_po_fit_rejects_out_of_range_grades(bad):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(100, 5))
    y = _grades(x[:, 0]).astype(float)
    y[:10] = bad
    with pytest.raises((ValueError, IndexError)):
        fit_ordinal_probe(x, y, pca_dim=None)


# --------------------------------------------------------------------------- #
# Card-grouped CV
# --------------------------------------------------------------------------- #
def _synthetic(n: int = 240, seed: int = 0, group_size: int = 3):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n)
    y = np.clip(np.round(8.5 + 1.2 * latent), 1, 10).astype(int)
    direction = rng.normal(size=(2, len(REGION_NAMES), D))
    feats = latent[:, None, None, None] * direction + rng.normal(size=(n, 2, len(REGION_NAMES), D))
    groups = np.array([f"card{i // group_size}" for i in range(n)])
    rng.shuffle(groups)
    return feats.astype(np.float32), y, groups


def test_every_fold_split_keeps_groups_disjoint(monkeypatch):
    feats, y, groups = _synthetic()
    splits = []
    real = probe_mod._folds

    def spy(yy, gg, n, seed):
        out = real(yy, gg, n, seed)
        splits.append((gg, out))
        return out

    monkeypatch.setattr(probe_mod, "_folds", spy)
    cross_validate(feats, y, groups, n_folds=3)
    assert len(splits) == 1 + 3  # outer CV + one inner OOF split per outer fold
    for gg, folds in splits:
        tested = np.concatenate([te for _, te in folds])
        assert np.sort(tested).tolist() == list(range(len(gg)))  # each row tested once
        for tr, te in folds:
            assert not set(gg[tr]) & set(gg[te])


@pytest.mark.slow
def test_cross_validate_deterministic():
    feats, y, groups = _synthetic(n=150)
    a = cross_validate(feats, y, groups, n_folds=3)
    b = cross_validate(feats, y, groups, n_folds=3)
    np.testing.assert_allclose(a[0], b[0], atol=1e-9)
    np.testing.assert_allclose(a[1], b[1], atol=1e-9)


def test_oof_region_grades_are_out_of_fold(monkeypatch):
    """Each row's OOF grade comes from probes not fitted on that row's card."""
    feats, y, groups = _synthetic(n=150)
    fitted_groups = []
    real = probe_mod.fit_region_probes

    def spy(f, yy):
        # identify rows by their exact feature vectors
        fitted_groups.append({tuple(r) for r in f[:, 0, 0, :2].round(5).tolist()})
        return real(f, yy)

    monkeypatch.setattr(probe_mod, "fit_region_probes", spy)
    out = oof_region_grades(feats, y, groups, n_folds=3)
    assert out.shape == (len(y), len(PROBE_NAMES))
    assert np.isfinite(out).all() and (out >= 1).all() and (out <= 10).all()
    keys = [tuple(r) for r in feats[:, 0, 0, :2].round(5).tolist()]
    for fitted in fitted_groups:
        inside = {groups[i] for i, k in enumerate(keys) if k in fitted}
        outside = {groups[i] for i, k in enumerate(keys) if k not in fitted}
        assert not inside & outside


@pytest.mark.slow
def test_cross_validate_handles_rare_grade_in_one_group():
    """A grade carried by a single card (absent from most training folds)."""
    feats, y, groups = _synthetic(n=150)
    y = y.copy()
    rare = groups == groups[0]
    y[rare] = 1
    cum, regions = cross_validate(feats, y, groups, n_folds=3)
    assert np.isfinite(cum).all() and np.isfinite(regions).all()
    assert (np.diff(cum, axis=1) <= 1e-12).all()
