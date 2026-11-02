"""GradeInferenceService: probe loading and prediction (fake extractor), and real weights."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.ml.inference import GradeInferenceService
from backend.app.ml.preprocess import CARD_H, CARD_W, detect_and_crop
from backend.tests.helpers.synth import make_card, make_scene, to_jpeg_bytes


@pytest.fixture(scope="module")
def jpeg():
    return to_jpeg_bytes(make_scene(seed=0).bgr)


def test_predict_without_model_raises(tmp_path, jpeg):
    svc = GradeInferenceService(head_path=tmp_path / "h.npz", features_path=tmp_path / "f.onnx")
    assert not svc.ready
    with pytest.raises(RuntimeError, match="not loaded"):
        svc.predict(jpeg, jpeg)


# --------------------------------------------------------------------------- #
# Probe model (frozen DINOv2 features + linear ordinal head)
# --------------------------------------------------------------------------- #
class FakeExtractor:
    session = None

    def __init__(self, feats):
        self.feats = feats
        self.calls = 0

    def __call__(self, front_rgb, back_rgb):
        assert front_rgb.shape == back_rgb.shape == (CARD_H, CARD_W, 3)
        self.calls += 1
        return self.feats


def _probe_service(tmp_path, feats, head):
    svc = GradeInferenceService(head_path=tmp_path / "h.npz", features_path=tmp_path / "f.onnx")
    assert not svc.ready  # probe artifacts missing -> nothing loaded yet
    svc._extractor = FakeExtractor(feats)
    svc._session = object()
    svc._head = head
    return svc


def test_probe_prediction_uses_head_and_reports_regions(tmp_path, jpeg):
    from backend.app.ml.features import REGION_NAMES
    from backend.app.ml.grade_head import PROBE_NAMES, GradeHead, OrdinalProbe, probe_input

    feats = np.ones((2, len(REGION_NAMES), 4), np.float32)
    probes = {
        n: OrdinalProbe(np.zeros(probe_input(feats, n).size), 0.0, np.linspace(-9, -1, 9))
        for n in PROBE_NAMES
    }
    # Meta: confident grade 9 regardless of input.
    meta = OrdinalProbe(np.zeros(4), 0.0, np.array([-9.0] * 8 + [9.0]))
    svc = _probe_service(tmp_path, feats, GradeHead(probes, meta, {}))
    pred = svc.predict(jpeg, jpeg)
    assert pred.grade_int == 9
    assert pred.confidence > 0.99
    assert set(pred.region_grades) == set(PROBE_NAMES)
    assert all(9.0 < g <= 10.0 for g in pred.region_grades.values())
    assert svc._extractor.calls == 1


def test_probe_prediction_returns_crops(tmp_path, jpeg):
    from backend.app.ml.features import REGION_NAMES

    feats = np.ones((2, len(REGION_NAMES), 4), np.float32)
    svc = _probe_service(tmp_path, feats, _zero_head(feats))
    pred = svc.predict(jpeg, jpeg)
    assert len(pred.crops) == 2 and all(c.found_quad for c in pred.crops)


def test_probe_fallback_flag_propagates_and_crops_are_the_ones_used(tmp_path, jpeg):
    from backend.app.ml.features import REGION_NAMES

    feats = np.ones((2, len(REGION_NAMES), 4), np.float32)
    svc = _probe_service(tmp_path, feats, _zero_head(feats))
    blank = to_jpeg_bytes(np.full((900, 700, 3), 128, np.uint8))
    pred = svc.predict(jpeg, blank)
    assert pred.found_quad_front and not pred.found_quad_back
    front, back = pred.crops
    assert front.found_quad and not back.found_quad
    np.testing.assert_array_equal(front.image, detect_and_crop(jpeg).image)


def test_probe_features_see_crops_encoded_like_training(tmp_path, jpeg):
    # train_probe reads precrop's JPEGs, so served crops go through the same encode.
    from backend.app.ml.features import REGION_NAMES
    from backend.app.ml.inference import _jpeg_roundtrip
    from backend.app.ml.preprocess import detect_and_crop

    seen = []

    class Recorder(FakeExtractor):
        def __call__(self, front_rgb, back_rgb):
            seen.append(front_rgb)
            return super().__call__(front_rgb, back_rgb)

    feats = np.ones((2, len(REGION_NAMES), 4), np.float32)
    svc = _probe_service(tmp_path, feats, _zero_head(feats))
    svc._extractor = Recorder(feats)
    svc.predict(jpeg, jpeg)
    crop = detect_and_crop(jpeg).image
    np.testing.assert_array_equal(seen[0], _jpeg_roundtrip(crop))
    assert not np.array_equal(seen[0], crop)


def _zero_head(feats):
    from backend.app.ml.grade_head import PROBE_NAMES, GradeHead, OrdinalProbe, probe_input

    probes = {
        n: OrdinalProbe(np.zeros(probe_input(feats, n).size), 0.0, np.linspace(-4, 4, 9))
        for n in PROBE_NAMES
    }
    return GradeHead(probes, OrdinalProbe(np.zeros(4), 0.0, np.linspace(-4, 4, 9)), {})


@pytest.fixture
def probe_files(tmp_path, monkeypatch):
    """(head path, features path) with a stub extractor producing 4-dim features."""
    from backend.app.ml import features
    from backend.app.ml.features import REGION_NAMES

    class Session:
        def get_outputs(self):
            return [type("Output", (), {"shape": ["n", len(REGION_NAMES), 4]})()]

        def get_providers(self):
            return ["CPUExecutionProvider"]

    class Extractor:
        def __init__(self, _path):
            self.session = Session()

    monkeypatch.setattr(features, "FeatureExtractor", Extractor)
    feat_path = tmp_path / "f.onnx"
    feat_path.write_bytes(b"stub")
    return tmp_path / "h.npz", feat_path


def _save_head(path, dim, metadata=None):
    from backend.app.ml.features import REGION_NAMES

    head = _zero_head(np.zeros((2, len(REGION_NAMES), dim)))
    head.metadata = dict(metadata or {})
    head.save(path)


def _write_feature_sidecar(feat_path, spec_hash):
    import json

    feat_path.with_suffix(".json").write_text(json.dumps({"feature_spec_hash": spec_hash}))


def test_probe_files_load_as_probe(probe_files):
    head_path, feat_path = probe_files
    _save_head(head_path, 4)
    assert GradeInferenceService(head_path=head_path, features_path=feat_path).ready


def _break_probe(head_path, feat_path, problem):
    if problem == "corrupt_head":
        head_path.write_bytes(b"not an npz")
    elif problem == "dim_mismatch":
        _save_head(head_path, 5)
    elif problem == "bad_cutoff":
        _save_head(head_path, 4, {"ten_cutoff": 1.5})
    else:  # feature_spec_hash mismatch
        _save_head(head_path, 4, {"feature_spec_hash": "aaaaaaaaaaaa"})
        _write_feature_sidecar(feat_path, "bbbbbbbbbbbb")


PROBLEMS = ["corrupt_head", "dim_mismatch", "spec_mismatch", "bad_cutoff"]


@pytest.mark.parametrize("problem", PROBLEMS)
def test_unloadable_probe_fails_closed(probe_files, serve_env, problem):
    head_path, feat_path = probe_files
    _break_probe(head_path, feat_path, problem)
    svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert not svc.ready
    assert svc.probe_load_error
    assert svc.metadata == {"probe_load_error": svc.probe_load_error}


def test_feature_spec_hash_mismatch_names_both_hashes(probe_files, serve_env):
    head_path, feat_path = probe_files
    _break_probe(head_path, feat_path, "spec_mismatch")
    svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    err = svc.metadata["probe_load_error"]
    assert err.startswith("ValueError: feature_spec_hash")
    assert "aaaaaaaaaaaa" in err and "bbbbbbbbbbbb" in err


def test_matching_feature_spec_hashes_load_as_probe(probe_files, serve_env, caplog):
    from backend.app.ml.features import feature_spec_hash

    head_path, feat_path = probe_files
    _save_head(head_path, 4, {"feature_spec_hash": feature_spec_hash()})
    _write_feature_sidecar(feat_path, feature_spec_hash())
    with caplog.at_level("WARNING", logger="backend.app.ml.inference"):
        svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert svc.ready
    assert "probe_load_error" not in svc.metadata
    assert not [r for r in caplog.records if "feature_spec_hash" in r.getMessage()]


@pytest.mark.parametrize("sidecar", ["none", "code", "unreadable"])
def test_missing_feature_spec_hash_still_loads_with_warning(
    probe_files, serve_env, caplog, sidecar
):
    # Artifacts from before the check: a head without a hash, with no sidecar, one
    # matching the code, or one that doesn't parse.
    from backend.app.ml.features import feature_spec_hash

    head_path, feat_path = probe_files
    _save_head(head_path, 4)
    if sidecar == "code":
        _write_feature_sidecar(feat_path, feature_spec_hash())
    elif sidecar == "unreadable":
        feat_path.with_suffix(".json").write_text("{not json")
    with caplog.at_level("WARNING", logger="backend.app.ml.inference"):
        svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert svc.ready
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("feature_spec_hash missing" in r.getMessage() for r in warnings)


@pytest.mark.parametrize(
    "head_hash, sidecar, expected",
    [
        ("code", None, "f.json missing"),
        ("code", "[1, 2]", "f.json unreadable"),
        ("code", "{not json", "f.json unreadable"),
        ("code", "{}", "f.json None != head"),
        ("0123456789ab", "0123456789ab", "head 0123456789ab (h.npz) != code"),
        (None, "0123456789ab", "f.json 0123456789ab != code"),
    ],
)
def test_feature_spec_disagreement_fails_closed(
    probe_files, serve_env, head_hash, sidecar, expected
):
    # A head with a hash needs the code's hash and a matching sidecar; a head
    # without one still refuses a sidecar from another spec than the code's.
    from backend.app.ml.features import feature_spec_hash

    code = feature_spec_hash()
    head_path, feat_path = probe_files
    head_hash = code if head_hash == "code" else head_hash
    _save_head(head_path, 4, {"feature_spec_hash": head_hash} if head_hash else {})
    if sidecar is not None and sidecar[0] in "[{":
        feat_path.with_suffix(".json").write_text(sidecar)
    elif sidecar is not None:
        _write_feature_sidecar(feat_path, sidecar)
    svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert not svc.ready
    err = svc.metadata["probe_load_error"]
    assert err.startswith("ValueError: feature_spec_hash: ")
    assert expected in err


@pytest.mark.parametrize("value", [1.0, 0.45, -1, "0.6", None, True, float("nan"), float("inf")])
def test_bad_ten_cutoff_fails_closed(probe_files, serve_env, value):
    head_path, feat_path = probe_files
    _save_head(head_path, 4, {"ten_cutoff": value})
    svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert not svc.ready
    assert svc.metadata["probe_load_error"].startswith("ValueError: ten_cutoff ")


def _close_call_head(dim, metadata):
    """Every input: P(grade > k) ~ 1 for k <= 8 and P(10) = 0.55, so the median is 10."""
    from backend.app.ml.features import REGION_NAMES
    from backend.app.ml.grade_head import OrdinalProbe

    head = _zero_head(np.zeros((2, len(REGION_NAMES), dim)))
    head.meta = OrdinalProbe(np.zeros(4), 0.0, np.array([-9.0] * 8 + [-np.log(0.55 / 0.45)]))
    head.metadata = dict(metadata)
    return head


@pytest.mark.parametrize("cutoff, want", [(None, 10), (0.5, 10), (0.54, 10), (0.56, 9), (0.6, 9)])
def test_served_grade_uses_the_heads_cutoff_like_metrics_and_eval(
    probe_files, serve_env, tmp_path, jpeg, cutoff, want
):
    # One rule for the API, train_probe.metrics (CV and test numbers) and ml.eval.
    from backend.app.ml.features import REGION_NAMES
    from backend.app.ml.grade_head import decode
    from ml.eval.evaluate import Sample, predict_all
    from ml.training.train_probe import metrics

    head_path, feat_path = probe_files
    _close_call_head(4, {} if cutoff is None else {"ten_cutoff": cutoff}).save(head_path)
    svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert svc.ready
    if cutoff is not None:
        assert svc.metadata["ten_cutoff"] == cutoff  # shown by /api/health
    svc._extractor = FakeExtractor(np.ones((2, len(REGION_NAMES), 4), np.float32))
    pred = svc.predict(jpeg, jpeg)
    cum = np.array(pred.rank_probs)
    assert cum[-1] == pytest.approx(0.55) and decode(cum)[1] == 10
    assert pred.grade_int == want
    assert pred.grade_int == decode(cum, 0.5 if cutoff is None else cutoff)[1]
    # The highlighted bar is the confidence, here next to a taller P(10) when it's a 9.
    assert pred.confidence == pytest.approx(decode(cum)[3][want - 1])
    t = 0.5 if cutoff is None else cutoff
    assert metrics(np.array([want]), cum[None], t)["exact"] == 1.0
    (tmp_path / "f.jpg").write_bytes(jpeg)
    rows = predict_all(svc, [Sample("c", want, tmp_path / "f.jpg", tmp_path / "f.jpg")], progress=False)
    assert rows["pred_int"].tolist() == [want]
    assert rows["confidence"].tolist() == [pred.confidence]


@pytest.mark.parametrize("missing", ["head", "features", "both"])
def test_missing_probe_files_fail_closed(tmp_path, serve_env, missing):
    head_path, feat_path = tmp_path / "grade_head.npz", tmp_path / "card_features.onnx"
    if missing == "features":
        _save_head(head_path, 4)
    if missing == "head":
        feat_path.write_bytes(b"stub")
    svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert not svc.ready
    names = {"head": ["grade_head.npz"], "features": ["card_features.onnx"]}
    want = names.get(missing) or names["head"] + names["features"]
    assert svc.metadata["probe_load_error"] == "FileNotFoundError: " + ", ".join(want)


@pytest.mark.model
def test_real_probe_model_prediction_is_valid_and_deterministic():
    from backend.app.core.config import get_settings

    s = get_settings()
    if not (s.grade_head_path.exists() and s.feature_model_path.exists()):
        pytest.skip("probe model not trained (python -m ml.training.train_probe)")
    svc = GradeInferenceService()
    assert svc.ready
    assert svc.metadata.get("kind") == "dinov2_region_probe"
    front = to_jpeg_bytes(make_scene(seed=1).bgr)
    back = to_jpeg_bytes(make_scene(seed=2, card=make_card(border_rgb=(40, 80, 200))).bgr)
    a = svc.predict(front, back)
    b = svc.predict(front, back)
    assert 1.0 <= a.grade <= 10.0 and 1 <= a.grade_int <= 10
    assert 0.0 <= a.confidence <= 1.0
    assert all(x >= y for x, y in zip(a.rank_probs, a.rank_probs[1:], strict=False))
    assert set(a.region_grades) == {"corners", "edges", "surface", "global"}
    assert a.grade == pytest.approx(b.grade, abs=1e-5)
