"""Adversarial review tests for the probe grade model, centering and the API.

Several tests pin defects found in review and since fixed: centering next to
a gasket close to the border colour, a corrupt grade_head.json, the region
hint vs. the shown grade, and the back-unmeasured centering note.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import backend.app.ml.centering as centering_mod
from backend.app.ml import condition
from backend.app.ml.centering import (
    CenteringResult,
    _grade_from_ratio,
    compute_centering_subgrade,
    measure_borders,
    measure_centering,
)
from backend.app.ml.features import REGION_NAMES
from backend.app.ml.grade_head import (
    NUM_GRADES,
    PROBE_NAMES,
    GradeHead,
    OrdinalProbe,
    decode,
    probe_input,
)
from backend.app.ml.inference import GradeInferenceService
from backend.app.ml.preprocess import CARD_H, CARD_W
from backend.tests.helpers.synth import make_card, make_scene, to_jpeg_bytes

D = 8


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _head(seed: int = 0, d: int = D, metadata: dict | None = None) -> GradeHead:
    rng = np.random.default_rng(seed)
    feats = np.zeros((2, len(REGION_NAMES), d))
    probes = {
        n: OrdinalProbe(
            weight=rng.normal(size=probe_input(feats, n).size) * 0.3,
            bias=float(rng.normal()),
            thresholds=np.sort(rng.normal(size=9) * 3),
        )
        for n in PROBE_NAMES
    }
    meta = OrdinalProbe(
        weight=np.full(len(PROBE_NAMES), 0.6), bias=-18.0, thresholds=np.linspace(-8, 4, 9)
    )
    return GradeHead(probes=probes, meta=meta, metadata=metadata or {"kind": "review"})


def _files(front: bytes, back: bytes) -> dict:
    return {"front": ("f.jpg", front, "image/jpeg"), "back": ("b.jpg", back, "image/jpeg")}


# --------------------------------------------------------------------------- #
# decode
# --------------------------------------------------------------------------- #
_DECODE_CASES = {
    "all_zero": np.zeros(9),
    "all_one": np.ones(9),
    "all_half": np.full(9, 0.5),
    "half_boundary": np.array([1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5]),
    "non_monotone": np.array([0.1, 0.9, 0.2, 0.95, 0.0, 1.0, 0.3, 0.7, 0.6]),
    "tiny": np.full(9, 1e-300),
    "near_one": np.full(9, 1 - 1e-16),
}


@pytest.mark.parametrize("name", list(_DECODE_CASES))
def test_decode_outputs_always_valid(name):
    expected, hard, conf, per_grade = decode(_DECODE_CASES[name])
    assert 1.0 <= expected <= 10.0
    assert 1 <= hard <= 10
    assert 0.0 <= conf <= 1.0
    assert per_grade.shape == (NUM_GRADES,)
    assert (per_grade >= 0).all()
    assert per_grade.sum() == pytest.approx(1.0)


def test_decode_random_monotone_inputs():
    rng = np.random.default_rng(0)
    for _ in range(500):
        p = np.sort(rng.uniform(size=9))[::-1]
        expected, hard, conf, per_grade = decode(p)
        assert per_grade.sum() == pytest.approx(1.0)
        assert 0.0 <= conf <= 1.0
        assert conf == pytest.approx(per_grade[hard - 1])
        # Hard grade is the median: P(grade <= hard) >= 0.5 and P(grade >= hard) >= 0.5.
        cdf = np.cumsum(per_grade)
        assert cdf[hard - 1] >= 0.5 - 1e-9
        assert (1 - (cdf[hard - 2] if hard >= 2 else 0.0)) >= 0.5 - 1e-9
        assert expected == pytest.approx(float((np.arange(1, 11) * per_grade).sum()))


def test_decode_exact_half_excluded():
    _, hard, _, _ = decode(np.array([1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5]))
    assert hard == 5


def test_cum_probs_finite_for_extreme_scores():
    p = OrdinalProbe(weight=np.array([1.0]), bias=0.0, thresholds=np.linspace(-4, 4, 9))
    with np.errstate(over="ignore"):
        cum = p.cum_probs(np.array([[1e6], [-1e6], [0.0]]))
    assert np.isfinite(cum).all()
    assert (cum[0] == 1.0).all() and (cum[1] == 0.0).all()
    soft = p.soft_grade(np.array([[1e6], [-1e6]]))
    assert soft.tolist() == [10.0, 1.0]


# --------------------------------------------------------------------------- #
# GradeHead io + batching
# --------------------------------------------------------------------------- #
def test_head_save_load_is_exact(tmp_path):
    head = _head(metadata={"kind": "review", "nested": {"a": [1, 2]}})
    path = tmp_path / "sub" / "head.npz"
    head.save(path)
    loaded = GradeHead.load(path)
    for n in (*PROBE_NAMES, "meta"):
        a = head.meta if n == "meta" else head.probes[n]
        b = loaded.meta if n == "meta" else loaded.probes[n]
        np.testing.assert_array_equal(a.weight, b.weight)
        np.testing.assert_array_equal(a.thresholds, b.thresholds)
        assert a.bias == b.bias
    assert loaded.metadata == head.metadata


def test_head_load_without_sidecar(tmp_path):
    path = tmp_path / "head.npz"
    _head().save(path)
    path.with_suffix(".json").unlink()
    assert GradeHead.load(path).metadata == {}


def test_head_load_refuses_pickled_arrays(tmp_path):
    head = _head()
    arrays = {}
    for name, p in {**head.probes, "meta": head.meta}.items():
        arrays[f"{name}.weight"] = p.weight
        arrays[f"{name}.bias"] = np.array(p.bias)
        arrays[f"{name}.thresholds"] = p.thresholds
    arrays["corners.weight"] = np.array([object(), 1.0], dtype=object)
    path = tmp_path / "evil.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        GradeHead.load(path)


def test_head_batched_matches_single():
    head = _head()
    feats = np.random.default_rng(3).normal(size=(6, 2, len(REGION_NAMES), D))
    batched = head.region_grades(feats)
    assert batched.shape == (6, len(PROBE_NAMES))
    for i in range(6):
        single = head(feats[i])
        np.testing.assert_allclose(batched[i], [single.region_grades[n] for n in PROBE_NAMES])
        np.testing.assert_allclose(head.meta.cum_probs(batched[i]), single.cum_probs)


# --------------------------------------------------------------------------- #
# GradeInferenceService model selection (tiny ONNX stand-in for DINOv2)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tiny_feature_onnx(tmp_path_factory):
    """A small ONNX graph with the same I/O contract as card_features.onnx."""
    torch = pytest.importorskip("torch")

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            g = torch.Generator().manual_seed(0)
            self.proj = torch.nn.Parameter(torch.randn(3 * 4, len(REGION_NAMES) * D, generator=g))

        def forward(self, x):
            pooled = torch.nn.functional.adaptive_avg_pool2d(x, (2, 2)).flatten(1)
            return (pooled @ self.proj).reshape(x.shape[0], len(REGION_NAMES), D)

    path = tmp_path_factory.mktemp("onnx") / "feat.onnx"
    torch.onnx.export(
        Tiny().eval(),
        torch.zeros(2, 3, 602, 434),
        str(path),
        input_names=["image"],
        output_names=["regions"],
        dynamic_axes={"image": {0: "n"}, "regions": {0: "n"}},
        opset_version=17,
    )
    return path


def test_selection_head_only_is_legacy(tmp_path):
    head = tmp_path / "h.npz"
    _head().save(head)
    svc = GradeInferenceService(
        head_path=head, features_path=tmp_path / "missing.onnx"
    )
    assert svc.kind == "legacy"


def test_selection_features_only_is_legacy(tmp_path, tiny_feature_onnx):
    svc = GradeInferenceService(head_path=tmp_path / "missing.npz", features_path=tiny_feature_onnx)
    assert svc.kind == "legacy"


def test_selection_both_is_probe_and_predicts(tmp_path, tiny_feature_onnx):
    head_path = tmp_path / "h.npz"
    _head(metadata={"kind": "dinov2_region_probe", "n_fit": 3}).save(head_path)
    svc = GradeInferenceService(head_path=head_path, features_path=tiny_feature_onnx)
    assert svc.kind == "probe" and svc.ready
    assert svc.metadata == {"kind": "dinov2_region_probe", "n_fit": 3}
    jpeg = to_jpeg_bytes(make_scene(seed=1).bgr)
    a = svc.predict(jpeg, jpeg)
    b = svc.predict(jpeg, jpeg)
    assert a == b
    exp, hard, conf, _ = decode(np.array(a.rank_probs))
    assert a.grade == pytest.approx(exp) and a.grade_int == hard
    assert a.confidence == pytest.approx(conf)
    assert all(x >= y for x, y in zip(a.rank_probs, a.rank_probs[1:], strict=False))
    assert set(a.region_grades) == set(PROBE_NAMES)
    assert all(1.0 <= g <= 10.0 for g in a.region_grades.values())


def test_selection_explicit_model_path_forces_legacy(tmp_path, tiny_feature_onnx):
    head_path = tmp_path / "h.npz"
    _head().save(head_path)
    svc = GradeInferenceService(
        model_path=tmp_path / "legacy.onnx", head_path=head_path, features_path=tiny_feature_onnx
    )
    assert svc.kind == "legacy"
    assert svc.model_path == tmp_path / "legacy.onnx"
    assert not svc.ready  # legacy file missing -> 503, not a silent probe


def test_corrupt_head_sidecar_does_not_crash_startup(tmp_path, tiny_feature_onnx):
    head_path = tmp_path / "h.npz"
    _head().save(head_path)
    head_path.with_suffix(".json").write_text("{not json")
    svc = GradeInferenceService(head_path=head_path, features_path=tiny_feature_onnx)
    assert svc.ready


# --------------------------------------------------------------------------- #
# Centering: synthetic accuracy - measured within 3 px, or abstain
# --------------------------------------------------------------------------- #
TOL_PX = 3.0
COLOURS = {
    "yellow": (238, 202, 52),
    "silver": (180, 180, 185),
    "white": (245, 245, 245),
    "blue_back": (30, 70, 160),
    "dark": (30, 30, 35),
}
BORDER_SETS = {
    "thin8": (8, 8, 10, 10),
    "centered": (36, 36, 40, 40),
    "wide60": (60, 60, 60, 60),
    "lr90_10": (54, 6, 40, 40),
    "tb90_10": (40, 40, 72, 8),
    "lr80_20": (20, 60, 40, 40),
}


def _jpeg(rgb: np.ndarray, q: int = 40) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    assert ok
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _with_gasket(rgb: np.ndarray, g: int, colour) -> tuple[np.ndarray, np.ndarray]:
    """Shrink the card into the canvas, leaving a g-px margin; -> (image, scale per side)."""
    inner = cv2.resize(rgb, (CARD_W - 2 * g, CARD_H - 2 * g), interpolation=cv2.INTER_AREA)
    out = np.empty_like(rgb)
    out[:] = colour
    out[g:-g, g:-g] = inner
    sx, sy = (CARD_W - 2 * g) / CARD_W, (CARD_H - 2 * g) / CARD_H
    return out, np.array([sx, sx, sy, sy])


def _apply(mod: str, card: np.ndarray, borders) -> tuple[np.ndarray, np.ndarray]:
    expected = np.array(borders, float)
    if mod == "jpeg":
        card = _jpeg(card)
    elif mod == "blur":
        card = cv2.GaussianBlur(card, (0, 0), 1.2)
    elif mod == "black_gasket":
        card, s = _with_gasket(card, 5, (5, 5, 5))
        expected = expected * s
    elif mod == "text":
        y = CARD_H - borders[3] // 2 + 3
        for x in range(80, 520, 14):
            cv2.putText(card, "W", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (15, 15, 15), 1)
    return card, expected


def _accurate_or_none(measured, expected) -> None:
    if measured is None:
        return
    err = np.abs(np.asarray(measured) - expected).max()
    assert err <= TOL_PX, f"confidently wrong: got {np.round(measured, 1)}, truth {np.round(expected, 1)}"


@pytest.mark.slow
@pytest.mark.parametrize("mod", ["none", "jpeg", "blur", "text"])
@pytest.mark.parametrize("borders", list(BORDER_SETS))
@pytest.mark.parametrize("colour", list(COLOURS))
def test_centering_accurate_or_abstains(colour, borders, mod):
    b = BORDER_SETS[borders]
    card, expected = _apply(mod, make_card(borders=b, border_rgb=COLOURS[colour], seed=3), b)
    _accurate_or_none(measure_borders(card), expected)


@pytest.mark.slow
@pytest.mark.parametrize("borders", list(BORDER_SETS))
@pytest.mark.parametrize("colour", [c for c in COLOURS if c != "dark"])
def test_centering_with_black_gasket(colour, borders):
    b = BORDER_SETS[borders]
    card, expected = _apply("black_gasket", make_card(borders=b, border_rgb=COLOURS[colour], seed=3), b)
    _accurate_or_none(measure_borders(card), expected)


# Gasket colour distinct from the border (Lab dE ~12-22, around the 12-unit
# border tolerance) but close enough that it used to join the border's run:
# the per-side mean shift drifted between the two, or locked onto a gasket
# wider than the narrow border. These must now be measured, and correctly.
_GASKET = {
    "dark_border_black_gasket": ((30, 30, 35), (15, 15, 15), 6, (54, 6, 40, 40)),
    "silver_border_lightgrey_gasket": ((180, 180, 185), (200, 200, 200), 6, (60, 8, 40, 40)),
    "silver_75_25_lightgrey_gasket12": ((180, 180, 185), (200, 200, 200), 12, (54, 18, 40, 40)),
    "gasket_joins_wide_side": ((218, 26, 127), (208, 4, 144), 13, (48, 16, 40, 40)),
    "gasket_wider_than_border": ((186, 77, 101), (215, 75, 99), 11, (42, 5, 40, 40)),
}


@pytest.mark.parametrize("case", list(_GASKET))
def test_centering_gasket_near_border_colour(case):
    border, gasket, g, b = _GASKET[case]
    card, s = _with_gasket(make_card(borders=b, border_rgb=border, seed=1), g, gasket)
    measured = measure_borders(card)
    assert measured is not None
    _accurate_or_none(measured, np.array(b, float) * s)


@pytest.mark.parametrize("borders", [(48, 24, 40, 40), (30, 50, 60, 25)])
def test_centering_flip_symmetry(borders):
    card = make_card(borders=borders, seed=5)
    a = measure_borders(card)
    h = measure_borders(np.ascontiguousarray(card[:, ::-1]))
    v = measure_borders(np.ascontiguousarray(card[::-1]))
    assert a is not None and h is not None and v is not None
    np.testing.assert_allclose(h, (a[1], a[0], a[2], a[3]), atol=0.5)
    np.testing.assert_allclose(v, (a[0], a[1], a[3], a[2]), atol=0.5)


@pytest.mark.parametrize(
    "borders", [(54, 18, 40, 40), (63, 9, 40, 40), (36, 36, 64, 16), (45, 27, 40, 40)]
)
def test_centering_never_overgrades_off_centre_card(borders):
    """Front grade must not be better than the true ratio allows."""
    res = measure_centering(make_card(borders=borders, seed=9), "front")
    left, right, top, bottom = borders
    true_worst = max(
        round(max(left, right) / (left + right) * 100), round(max(top, bottom) / (top + bottom) * 100)
    )
    if res is not None:
        assert res.grade <= _grade_from_ratio(true_worst, "front")


# --------------------------------------------------------------------------- #
# PSA tables
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("pct", "front", "back"),
    [
        (50, 10.0, 10.0), (60, 10.0, 10.0), (61, 9.0, 10.0), (65, 9.0, 10.0), (66, 8.0, 10.0),
        (70, 8.0, 10.0), (71, 7.0, 10.0), (75, 7.0, 10.0), (76, 6.0, 9.0), (80, 6.0, 9.0),
        (85, 5.0, 9.0), (86, 3.0, 9.0), (90, 3.0, 9.0), (91, 1.0, 1.0), (95, 1.0, 1.0),
        (100, 1.0, 1.0),
    ],
)
def test_psa_tables_boundaries(pct, front, back):
    assert _grade_from_ratio(pct, "front") == front
    assert _grade_from_ratio(pct, "back") == back


def test_back_table_monotone():
    grades = [_grade_from_ratio(p, "back") for p in range(50, 101)]
    assert all(a >= b for a, b in zip(grades, grades[1:], strict=False))


def test_worst_pct_uses_worse_axis():
    r = CenteringResult(left=30, right=30, top=20, bottom=60, grade=0.0)
    assert r.worst_pct == 75


# --------------------------------------------------------------------------- #
# compute_centering_subgrade: per-face abstention
# --------------------------------------------------------------------------- #
def _fake_measure(front_none: bool, back_none: bool):
    def fake(rgb, face="front"):
        if (face == "front" and front_none) or (face == "back" and back_none):
            return None
        return CenteringResult(left=40, right=30, top=40, bottom=40, grade=_grade_from_ratio(57, face))

    return fake


@pytest.fixture(scope="module")
def card_jpeg() -> bytes:
    return to_jpeg_bytes(make_scene(seed=3).bgr)


@pytest.mark.parametrize(
    ("front_none", "back_none"), [(True, False), (False, True), (True, True), (False, False)]
)
def test_subgrade_face_fields(monkeypatch, card_jpeg, front_none, back_none):
    monkeypatch.setattr(centering_mod, "measure_centering", _fake_measure(front_none, back_none))
    sub = compute_centering_subgrade(card_jpeg, card_jpeg)
    if front_none and back_none:
        assert sub is None
        return
    assert (sub.left_right is None) == front_none
    assert (sub.top_bottom is None) == front_none
    assert (sub.back_left_right is None) == back_none
    assert (sub.back_top_bottom is None) == back_none
    assert sub.grade == 10.0


def test_subgrade_worse_face_wins(monkeypatch, card_jpeg):
    def fake(rgb, face="front"):
        g = 9.0 if face == "front" else 10.0
        return CenteringResult(left=40, right=30, top=40, bottom=40, grade=g)

    monkeypatch.setattr(centering_mod, "measure_centering", fake)
    assert compute_centering_subgrade(card_jpeg, card_jpeg).grade == 9.0


# --------------------------------------------------------------------------- #
# API contract with the fake service
# --------------------------------------------------------------------------- #
def test_api_region_grades_none_uses_classical(client, fake_service, scene_jpeg, monkeypatch):
    called = []
    real = condition.compute_condition_subgrades

    def spy(*a, **k):
        called.append(1)
        return real(*a, **k)

    monkeypatch.setattr(condition, "compute_condition_subgrades", spy)
    assert fake_service.prediction.region_grades is None
    r = client.post("/api/grade", files=_files(scene_jpeg, scene_jpeg))
    assert r.status_code == 200
    assert called


@pytest.mark.parametrize(
    ("grade", "hint", "flags"),
    [
        # Hints follow the displayed (1-decimal) grade, so 9.26 counts as 9.3.
        (10.0, "clean", []), (9.26, "clean", []), (9.24, "slight wear", []),
        (8.66, "slight wear", []), (8.64, "wear", ["wear"]), (7.46, "wear", ["wear"]),
        (7.44, "heavy wear", ["wear"]), (1.0, "heavy wear", ["wear"]),
    ],
)
def test_api_region_hint_boundaries(client, fake_service, scene_jpeg, grade, hint, flags):
    fake_service.prediction.region_grades = dict.fromkeys(("corners", "edges", "surface", "global"), grade)
    r = client.post("/api/grade", files=_files(scene_jpeg, scene_jpeg))
    assert r.status_code == 200, r.text
    sub = r.json()["sub_grades"]
    for name in ("corners", "edges", "surface"):
        assert sub[name]["hint"] == hint
        assert sub[name]["flags"] == flags
    assert "global" not in sub


@pytest.mark.parametrize("grade", [9.27, 8.66, 7.46])
def test_api_region_hint_consistent_with_displayed_grade(client, fake_service, scene_jpeg, grade):
    from backend.app.api.grade import _region_hint

    fake_service.prediction.region_grades = dict.fromkeys(("corners", "edges", "surface", "global"), grade)
    r = client.post("/api/grade", files=_files(scene_jpeg, scene_jpeg))
    shown = r.json()["sub_grades"]["corners"]
    assert shown["hint"] == _region_hint(shown["grade"])[0]


def test_api_front_unmeasured_note(client, scene_jpeg, monkeypatch):
    monkeypatch.setattr(centering_mod, "measure_centering", _fake_measure(True, False))
    body = client.post("/api/grade", files=_files(scene_jpeg, scene_jpeg)).json()
    c = body["sub_grades"]["centering"]
    assert c["left_right"] is None and c["back_left_right"] is not None
    assert any("front" in n and n.startswith("centering:") for n in body["notes"])


def test_api_back_unmeasured_note(client, scene_jpeg, monkeypatch):
    monkeypatch.setattr(centering_mod, "measure_centering", _fake_measure(False, True))
    body = client.post("/api/grade", files=_files(scene_jpeg, scene_jpeg)).json()
    c = body["sub_grades"]["centering"]
    assert c["left_right"] is not None and c["back_left_right"] is None
    assert any("back" in n and n.startswith("centering:") for n in body["notes"])


def test_api_both_unmeasured_note(client, scene_jpeg, monkeypatch):
    monkeypatch.setattr(centering_mod, "measure_centering", _fake_measure(True, True))
    body = client.post("/api/grade", files=_files(scene_jpeg, scene_jpeg)).json()
    assert body["sub_grades"]["centering"] is None
    assert any(n.startswith("centering:") for n in body["notes"])


def test_api_survives_centering_exception(client, fake_service, scene_jpeg, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("cv exploded")

    monkeypatch.setattr(centering_mod, "compute_centering_subgrade", boom)
    fake_service.prediction.region_grades = {"corners": 9.5, "edges": 9.0, "surface": 8.0, "global": 9.0}
    r = client.post("/api/grade", files=_files(scene_jpeg, scene_jpeg))
    assert r.status_code == 200
    body = r.json()
    assert body["sub_grades"]["centering"] is None
    assert body["sub_grades"]["corners"]["grade"] == 9.5


def test_api_response_round_trips_back_fields(client, scene_jpeg, monkeypatch):
    monkeypatch.setattr(centering_mod, "measure_centering", _fake_measure(False, False))
    c = client.post("/api/grade", files=_files(scene_jpeg, scene_jpeg)).json()["sub_grades"]["centering"]
    assert set(c) == {"grade", "left_right", "top_bottom", "back_left_right", "back_top_bottom"}
    assert c["left_right"] == c["back_left_right"] == "57/43"
    assert c["top_bottom"] == c["back_top_bottom"] == "50/50"


# --------------------------------------------------------------------------- #
# Real DINOv2 feature model (model marker: needs data/models/card_features.onnx)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_extractor():
    from backend.app.core.config import get_settings
    from backend.app.ml.features import FeatureExtractor

    path = get_settings().feature_model_path
    if not path.exists():
        pytest.skip("card_features.onnx not exported")
    return FeatureExtractor(path)


@pytest.mark.model
def test_feature_extractor_shape_and_determinism(real_extractor):
    rng = np.random.default_rng(0)
    front = rng.integers(0, 256, (CARD_H, CARD_W, 3), dtype=np.uint8)
    back = make_card(seed=1)
    a = real_extractor(front, back)
    b = real_extractor(front, back)
    assert a.shape == (2, len(REGION_NAMES), 768) and a.dtype == np.float32
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)
    # The batch of two faces must not mix: back alone gives the same features.
    c = real_extractor(back, back)
    np.testing.assert_allclose(a[1], c[1], rtol=1e-4, atol=1e-4)


@pytest.mark.model
def test_onnx_matches_torch_region_model(real_extractor):
    torch = pytest.importorskip("torch")
    from backend.app.ml.features import build_region_model, to_input

    try:
        model = build_region_model(pretrained=True)
    except Exception as exc:  # noqa: BLE001 - weights not cached offline
        pytest.skip(f"timm weights unavailable: {exc}")
    rng = np.random.default_rng(1)
    imgs = [rng.integers(0, 256, (CARD_H, CARD_W, 3), dtype=np.uint8), make_card(seed=2)]
    x = np.stack([to_input(i) for i in imgs])
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    got = real_extractor(*imgs)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < 1e-3, rel
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3 * np.abs(ref).max())


@pytest.mark.model
def test_served_features_match_training_cache(real_extractor):
    """The ONNX graph reproduces the cached training features of real precrops."""
    from backend.app.core.config import get_settings

    s = get_settings()
    cache = s.models_dir / "cache" / "region_feats.npz"
    if not cache.exists():
        pytest.skip("no feature cache")
    with np.load(cache, allow_pickle=False) as z:
        keys, feats = z["keys"], z["feats"]
    rng = np.random.default_rng(0)
    checked = 0
    for i in rng.choice(len(keys), 6, replace=False):
        _, fp, bp = keys[i]
        paths = [s.data_dir / fp, s.data_dir / bp]
        if not all(p.exists() for p in paths):
            continue
        rgb = [cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for p in paths]
        got = real_extractor(*rgb)
        ref = feats[i].astype(np.float32)
        assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 2e-3
        checked += 1
    if not checked:
        pytest.skip("precrop images missing")


@pytest.mark.model
def test_real_service_on_raw_images_sane_and_deterministic():
    import sqlite3

    from backend.app.core.config import get_settings

    s = get_settings()
    if not (s.grade_head_path.exists() and s.feature_model_path.exists() and s.db_path.exists()):
        pytest.skip("probe model or DB missing")
    conn = sqlite3.connect(f"file:{s.db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT grade_int, front_path, back_path FROM certs WHERE split='test' "
            "AND has_images=1 AND front_path IS NOT NULL AND back_path IS NOT NULL "
            "AND crop_found_front=1 AND crop_found_back=1 ORDER BY cert_number LIMIT 200"
        ).fetchall()
    finally:
        conn.close()
    rows = [r for r in rows if (s.data_dir / r[1]).exists() and (s.data_dir / r[2]).exists()]
    if not rows:
        pytest.skip("raw images missing")
    pick = [rows[i] for i in np.random.default_rng(0).choice(len(rows), min(4, len(rows)), replace=False)]
    svc = GradeInferenceService()
    assert svc.kind == "probe"
    errors = []
    for grade, fp, bp in pick:
        fb, bb = (s.data_dir / fp).read_bytes(), (s.data_dir / bp).read_bytes()
        a, b = svc.predict(fb, bb), svc.predict(fb, bb)
        assert a.grade == pytest.approx(b.grade, abs=1e-6)
        assert a.grade_int == b.grade_int
        assert 1.0 <= a.grade <= 10.0 and 1 <= a.grade_int <= 10 and 0 <= a.confidence <= 1
        assert all(x >= y for x, y in zip(a.rank_probs, a.rank_probs[1:], strict=False))
        assert all(1.0 <= g <= 10.0 for g in a.region_grades.values())
        errors.append(abs(a.grade - grade))
    assert np.mean(errors) < 2.0, errors
