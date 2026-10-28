"""train_face_guard: probe fitting, thresholds and gates, and the rotated-feature cache."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.ml.face_guard import CHECKS, FaceGuard
from backend.app.ml.features import REGION_NAMES
from ml.training import train_face_guard as tfg

D = 4


def test_fit_logistic_folds_the_scaler():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    x = rng.normal(5.0, [1.0, 10.0, 0.1], size=(200, 3))
    y = (x[:, 0] + 0.1 * x[:, 1] > 5.5).astype(int)
    probe = tfg.fit_logistic(x, y)
    scaler = StandardScaler().fit(x)
    ref = LogisticRegression(C=tfg.C, max_iter=5000).fit(scaler.transform(x), y)
    np.testing.assert_allclose(probe.prob(x), ref.predict_proba(scaler.transform(x))[:, 1], atol=1e-9)


def _synthetic(n, seed, back_flip_signal=True):
    """(up, rot, groups): side in dim 0 of CLS, flip in dim 1 (not for backs if disabled)."""
    rng = np.random.default_rng(seed)
    up = rng.normal(size=(n, 2, len(REGION_NAMES), D)).astype(np.float32)
    up[:, 0, 0, 0] -= 4.0  # fronts
    up[:, 1, 0, 0] += 4.0  # backs
    rot = up + rng.normal(scale=0.1, size=up.shape).astype(np.float32)
    rot[:, 0, 0, 1] += 6.0
    if back_flip_signal:
        rot[:, 1, 0, 1] += 6.0
    groups = np.array([f"card{i // 2}" for i in range(n)])  # two certs per card
    return up, rot, groups


def test_separable_faces_enable_every_check():
    up, rot, groups = _synthetic(400, 0)
    fit = np.arange(400) < 300
    guard, report = tfg.train_guard(up, rot, groups, fit, ~fit)
    assert guard.enabled == CHECKS
    assert guard.thresholds == {n: 0.9 for n in CHECKS}  # lowest grid value that passes
    assert report["disabled_reasons"] == {}
    for n in CHECKS:
        assert report["cv_grouped"][n]["auc"] > 0.999
        assert report["test"][n]["n"] in (2 * 100, 4 * 100)
    pairs = report["cv_grouped"]["side"]["pairs"]["0.9"]
    assert pairs["false_swap_rate"] == 0.0 and pairs["swapped_pairs_caught"] > 0.95


def test_check_that_misses_its_gate_is_disabled_with_a_reason():
    up, rot, groups = _synthetic(300, 1, back_flip_signal=False)
    fit = np.arange(300) < 240
    guard, report = tfg.train_guard(up, rot, groups, fit, ~fit)
    assert guard.enabled == ("side", "front_upright")
    reason = report["disabled_reasons"]["back_upright"]
    assert "CV AUC" in reason and "< 0.99" in reason
    assert guard.metadata["disabled_reasons"] == report["disabled_reasons"]


def test_test_split_never_moves_thresholds_or_cv():
    up, rot, groups = _synthetic(300, 2)
    fit = np.arange(300) < 240
    a, ra = tfg.train_guard(up, rot, groups, fit, ~fit)
    up2, rot2 = up.copy(), rot.copy()
    up2[~fit], rot2[~fit] = rot[~fit], up[~fit]  # scramble the test faces
    b, rb = tfg.train_guard(up2, rot2, groups, fit, ~fit)
    assert a.thresholds == b.thresholds and a.enabled == b.enabled
    assert ra["cv_grouped"] == rb["cv_grouped"]
    assert ra["test"] != rb["test"]


def test_cv_folds_keep_a_cards_faces_together():
    # Both orientations and both faces of a card must share a fold, or the
    # probe could score by recognising the card.
    from sklearn.model_selection import StratifiedGroupKFold

    up, rot, groups = _synthetic(60, 3)
    d = tfg.face_sets(up, rot, groups)["side"]
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=tfg.SEED)
    assert len(d["y"]) == 4 * 60
    for tr, te in skf.split(np.zeros(len(d["y"])), d["y"], d["groups"]):
        assert not set(d["groups"][tr]) & set(d["groups"][te])


def test_choose_threshold_and_gate():
    m = {"auc": 0.9995, "error_at_0.5": 0.002, "at_threshold": {
        "0.9": {"clean_fpr": 0.004}, "0.95": {"clean_fpr": 0.002}, "0.99": {"clean_fpr": 0.0}}}
    assert tfg.choose_threshold(m) == 0.95
    assert tfg.gate("side", m, 0.95) is None
    worse = {**m, "error_at_0.5": 0.004, "at_threshold": {t: {"clean_fpr": 0.01} for t in m["at_threshold"]}}
    t = tfg.choose_threshold(worse)
    assert t is None
    reason = tfg.gate("side", worse, t)
    assert "CV error" in reason and "no threshold" in reason


def test_false_positive_on_clean_faces():
    p = np.array([0.96, 0.04, 0.5])
    y = np.array([0, 1, 0])  # a front read as a back, a back read as a front, an unsure front
    np.testing.assert_array_equal(tfg.false_positive("side", p, y, 0.95), [True, True, False])
    np.testing.assert_array_equal(tfg.false_positive("front_upright", p, y, 0.95), [True, False, False])


def test_written_guard_loads(tmp_path):
    up, rot, groups = _synthetic(200, 4)
    fit = np.arange(200) < 160
    guard, _ = tfg.train_guard(up, rot, groups, fit, ~fit)
    path = tmp_path / "face_guard.npz"
    guard.save(path)
    loaded = FaceGuard.load(path)
    assert loaded.enabled == guard.enabled and loaded.thresholds == guard.thresholds
    # Fronts upright read as fronts and upright; a rotated front reads as flipped.
    assert loaded.decide(up[0]).notes == []
    assert loaded.decide(np.stack([rot[0, 0], up[0, 1]])).flip == (True, False)


# --------------------------------------------------------------------------- #
# Rotated-feature cache: train_probe.extract_features keying, with a variant
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch")


def test_rot180_cache_is_its_own_variant(tmp_path):
    from ml.tests.test_probe_cache import Factory, _write_crop
    from ml.training import train_probe

    rows = []
    for i in range(3):
        front, back = f"crops/images/c{i}_front.jpg", f"crops/images/c{i}_back.jpg"
        _write_crop(tmp_path / front, 2 * i)
        _write_crop(tmp_path / back, 2 * i + 1)
        rows.append({"cert_number": f"c{i}", "cropped_front_path": front, "cropped_back_path": back})
    cache = tmp_path / "cache" / "region_feats.npz"
    rot_cache = tmp_path / "cache" / "region_feats_rot180.npz"

    def extract(path, factory, **kw):
        return train_probe.extract_features(rows, tmp_path, path, model_factory=factory, **kw)

    up = extract(cache, Factory())
    before = cache.read_bytes()
    first = Factory()
    rot = extract(rot_cache, first, transform=tfg.rot180, variant="rot180")
    assert first.images == 6 and not np.allclose(up, rot)
    with np.load(rot_cache, allow_pickle=False) as z:
        assert str(z["variant"]) == "rot180"
    again = Factory()
    np.testing.assert_array_equal(extract(rot_cache, again, transform=tfg.rot180, variant="rot180"), rot)
    assert again.calls == 0
    # A cache of another variant is never reused (here: the upright cache read as rot180).
    other = Factory()
    extract(cache, other, transform=tfg.rot180, variant="rot180", save=False)
    assert other.images == 6
    # save=False never writes: the upright cache is byte-identical.
    assert cache.read_bytes() == before
    with np.load(cache, allow_pickle=False) as z:
        assert "variant" not in z.files  # the main cache format is unchanged
