"""Linear ordinal grade head (backend/app/ml/grade_head.py) and region features."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.ml.features import FEAT_H, FEAT_W, REGION_NAMES, region_means, to_input
from backend.app.ml.grade_head import (
    DEFAULT_TEN_CUTOFF,
    PROBE_NAMES,
    TEN_CUTOFF_KEY,
    GradeHead,
    OrdinalProbe,
    decode,
    probe_input,
    ten_cutoff,
)

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


# --------------------------------------------------------------------------- #
# P(10) cut-off (intent/grade-decision-rule)
# --------------------------------------------------------------------------- #
def _today(p):
    """decode before the cut-off: the hard grade is the median, 1 + #(P > 0.5)."""
    p = np.asarray(p, np.float64)
    per_grade = np.clip(np.concatenate([[1.0], p]) - np.concatenate([p, [0.0]]), 0.0, None)
    per_grade /= max(per_grade.sum(), 1e-12)
    hard = int(1 + (p > 0.5).sum())
    return float(1.0 + p.sum()), hard, float(per_grade[hard - 1]), per_grade


def _random_cums(n=2000, seed=0):
    """Monotone and non-monotone inputs, some with values at exactly 0.5."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        p = rng.uniform(size=9)
        if i % 2:
            p = np.sort(p)[::-1]
        if i % 3 == 0:
            p[rng.integers(9)] = 0.5
        if i % 7 == 0:
            p[8] = 0.5
        yield p


def test_decode_default_cutoff_is_todays_decode():
    for p in _random_cums():
        want = _today(p)
        for got in (decode(p), decode(p, 0.5), decode(p, DEFAULT_TEN_CUTOFF)):
            assert got[:3] == want[:3]
            np.testing.assert_array_equal(got[3], want[3])


@pytest.mark.parametrize("t", [0.5, 0.56, 0.6, 0.7])
@pytest.mark.parametrize(
    "p10, want", [("below", 9), ("at", 9), ("above", 10)]
)
def test_decode_cutoff_boundary(t, p10, want):
    # 10 only if P(10) > t: at exactly t it's a 9, as today at t = 0.5.
    value = {"below": np.nextafter(t, 0), "at": t, "above": np.nextafter(t, 1)}[p10]
    cum = np.array([0.999] * 8 + [value])
    _, hard, conf, per_grade = decode(cum, t)
    assert hard == want
    assert conf == per_grade[hard - 1]


def test_decode_cutoff_keeps_the_median_below_the_9_10_boundary():
    for p in _random_cums(seed=1):
        if p[8] <= 0.5:
            assert decode(p, 0.66)[:3] == decode(p)[:3]
        else:
            # Only a 10 can move, and only to the median of the rest.
            hard = decode(p, 0.66)[1]
            assert hard == (decode(p)[1] if p[8] > 0.66 else 1 + int((p[:8] > 0.5).sum()))


def test_decode_cutoff_confidence_is_p_of_the_hard_grade():
    cum = np.array([1, 1, 1, 1, 1, 1, 1, 0.9, 0.55])
    expected, hard, conf, per_grade = decode(cum, 0.6)
    assert hard == 9 and decode(cum)[1] == 10
    assert conf == pytest.approx(0.35)  # P(9) = 0.9 - 0.55, next to a taller P(10) = 0.55
    assert expected == decode(cum)[0]  # the soft grade doesn't depend on the rule
    for p in _random_cums(500, seed=2):
        _, hard, conf, per_grade = decode(p, 0.62)
        assert conf == per_grade[hard - 1]


@pytest.mark.parametrize("value", [0.5, 0.56, 0.7, 0.999, np.float64(0.6)])
def test_ten_cutoff_reads_valid_values(value):
    assert ten_cutoff({TEN_CUTOFF_KEY: value}) == pytest.approx(float(value))


@pytest.mark.parametrize(
    "value", [0.49, 1.0, 1, 0, -0.6, float("nan"), float("inf"), "0.6", None, True, [0.6], {}]
)
def test_ten_cutoff_rejects_bad_values(value):
    with pytest.raises(ValueError, match=TEN_CUTOFF_KEY):
        ten_cutoff({TEN_CUTOFF_KEY: value})


def test_old_head_without_cutoff_loads_and_decodes_as_today(tmp_path):
    path = tmp_path / "head.npz"
    _head().save(path)  # metadata {"kind": "test"}: no ten_cutoff
    head = GradeHead.load(path)
    assert TEN_CUTOFF_KEY not in head.metadata
    assert head.ten_cutoff == 0.5
    for p in _random_cums(200, seed=3):
        assert decode(p, head.ten_cutoff)[:3] == _today(p)[:3]


def test_head_cutoff_roundtrips(tmp_path):
    path = tmp_path / "head.npz"
    head = _head()
    head.metadata = {"kind": "test", TEN_CUTOFF_KEY: 0.58}
    head.save(path)
    assert GradeHead.load(path).ten_cutoff == 0.58
    path.with_suffix(".json").write_text('{"ten_cutoff": NaN}')
    with pytest.raises(ValueError, match=TEN_CUTOFF_KEY):
        _ = GradeHead.load(path).ten_cutoff


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


def test_feature_spec_hash_is_stable():
    from backend.app.ml.features import feature_spec_hash

    h = feature_spec_hash()
    assert h == feature_spec_hash()
    assert len(h) == 12 and int(h, 16) >= 0


@pytest.mark.parametrize(
    "field, value",
    [
        ("backbone", "vit_small_patch14_dinov2.lvd142m"),
        ("feat_h", 616),
        ("feat_w", 448),
        ("patch", 16),
        ("corner_patches", 4),
        ("edge_patches", 2),
        ("regions", ["cls", "mean"]),
        ("version", 2),
    ],
)
def test_feature_spec_hash_changes_with_every_field(monkeypatch, field, value):
    from backend.app.ml import features

    assert set(features.FEATURE_SPEC) == {
        "backbone", "feat_h", "feat_w", "patch", "corner_patches", "edge_patches", "regions", "version"
    }
    before = features.feature_spec_hash()
    monkeypatch.setitem(features.FEATURE_SPEC, field, value)
    assert features.feature_spec_hash() != before
