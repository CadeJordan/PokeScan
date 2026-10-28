"""Linear ordinal grade head (backend/app/ml/grade_head.py) and region features."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.ml.features import FEAT_H, FEAT_W, REGION_NAMES, region_means, to_input
from backend.app.ml.grade_head import PROBE_NAMES, GradeHead, OrdinalProbe, decode, probe_input

D = 8


def _probe(n: int, seed: int = 0) -> OrdinalProbe:
    rng = np.random.default_rng(seed)
    return OrdinalProbe(
        weight=rng.normal(size=n), bias=0.3, thresholds=np.linspace(-4.0, 2.0, 9)
    )


def _head() -> GradeHead:
    feats = np.zeros((2, len(REGION_NAMES), D))
    probes = {n: _probe(probe_input(feats, n).size, i) for i, n in enumerate(PROBE_NAMES)}
    meta = OrdinalProbe(
        weight=np.full(len(PROBE_NAMES), 0.5), bias=-15.0, thresholds=np.linspace(-6, 2, 9)
    )
    return GradeHead(probes=probes, meta=meta, metadata={"kind": "test"})


def test_probe_input_sizes():
    feats = np.zeros((2, len(REGION_NAMES), D))
    sizes = {n: probe_input(feats, n).size for n in PROBE_NAMES}
    assert sizes == {"corners": 2 * 4 * D, "edges": 2 * 4 * D, "surface": 2 * D, "global": 2 * 2 * D}
    batched = np.zeros((5, 2, len(REGION_NAMES), D))
    assert probe_input(batched, "corners").shape == (5, 2 * 4 * D)


def test_cum_probs_monotone_and_soft_grade_in_range():
    p = _probe(16)
    x = np.random.default_rng(1).normal(size=(50, 16)) * 5
    cum = p.cum_probs(x)
    assert cum.shape == (50, 9)
    assert (np.diff(cum, axis=1) <= 1e-12).all()
    soft = p.soft_grade(x)
    assert ((soft >= 1.0) & (soft <= 10.0)).all()


def test_head_output_and_roundtrip(tmp_path):
    head = _head()
    feats = np.random.default_rng(2).normal(size=(2, len(REGION_NAMES), D)).astype(np.float32)
    out = head(feats)
    assert set(out.region_grades) == set(PROBE_NAMES)
    assert out.cum_probs.shape == (9,)

    path = tmp_path / "head.npz"
    head.save(path)
    loaded = GradeHead.load(path)
    assert loaded.metadata == {"kind": "test"}
    np.testing.assert_allclose(loaded(feats).cum_probs, out.cum_probs)
    # Weights are stored without pickle.
    with np.load(path, allow_pickle=False) as z:
        assert "meta.thresholds" in z.files


@pytest.mark.parametrize("grade", range(1, 11))
def test_decode_confident(grade):
    cum = np.where(np.arange(1, 10) < grade, 0.999, 0.001)
    expected, hard, conf, per_grade = decode(cum)
    assert hard == grade
    assert expected == pytest.approx(grade, abs=0.01)
    assert conf > 0.99
    assert per_grade.sum() == pytest.approx(1.0)


def test_decode_uncertain():
    _, hard, conf, _ = decode(np.full(9, 0.5))
    assert hard == 1 and conf < 0.6


def test_to_input_shape_and_normalisation():
    x = to_input(np.full((840, 600, 3), 255, np.uint8))
    assert x.shape == (3, FEAT_H, FEAT_W) and x.dtype == np.float32
    expected = (1.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    np.testing.assert_allclose(x[:, 0, 0], expected, rtol=1e-5)


def test_region_means_pick_the_right_patches():
    grid = np.zeros((1, 43, 31, 1))
    grid[0, 0, 0] = 25.0      # top-left corner patch
    grid[0, 42, 30] = 25.0    # bottom-right corner patch
    tl, tr, br, bl, top, right, bottom, left, centre = (r[0, 0] for r in region_means(grid))
    assert tl == pytest.approx(1.0) and br == pytest.approx(1.0)
    assert tr == bl == top == right == bottom == left == centre == 0.0


def test_extreme_scores_saturate_without_overflow():
    p = OrdinalProbe(weight=np.ones(1), bias=0.0, thresholds=np.linspace(-4.0, 4.0, 9))
    with np.errstate(over="raise"):
        cum = p.cum_probs(np.array([[-1e4], [1e4]]))
    np.testing.assert_allclose(cum, [np.zeros(9), np.ones(9)])


def test_load_survives_corrupt_metadata(tmp_path):
    path = tmp_path / "head.npz"
    _head().save(path)
    path.with_suffix(".json").write_text("{not json")
    assert GradeHead.load(path).metadata == {}


def test_check_feature_dim():
    head = _head()
    head.check_feature_dim(D)
    with pytest.raises(ValueError, match="don't match"):
        head.check_feature_dim(D + 1)
