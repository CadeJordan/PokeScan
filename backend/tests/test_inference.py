"""GradeInferenceService: tensor prep + logit decoding (fake session), and real weights."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.ml.inference import GradeInferenceService
from backend.app.ml.preprocess import CARD_H, CARD_W, detect_and_crop
from backend.tests.helpers.synth import make_card, make_scene, to_jpeg_bytes


class FakeSession:
    def __init__(self, logits):
        self.logits = np.asarray(logits, np.float32)
        self.feeds = None

    def run(self, outputs, feeds):
        assert outputs == ["corn_logits"]
        self.feeds = feeds
        return [self.logits[None, :]]


@pytest.fixture
def service(tmp_path):
    svc = GradeInferenceService(model_path=tmp_path / "missing.onnx")
    assert not svc.ready
    return svc


@pytest.fixture(scope="module")
def jpeg():
    return to_jpeg_bytes(make_scene(seed=0).bgr)


def test_predict_without_model_raises(service, jpeg):
    with pytest.raises(RuntimeError, match="not loaded"):
        service.predict(jpeg, jpeg)


def test_feeds_shape_and_normalisation(service, jpeg):
    service._session = FakeSession(np.zeros(9))
    service.predict(jpeg, jpeg)
    feeds = service._session.feeds
    assert set(feeds) == {"front", "back"}
    for x in feeds.values():
        assert x.shape == (1, 3, CARD_H, CARD_W)
        assert x.dtype == np.float32
        assert x.flags["C_CONTIGUOUS"]
        # ImageNet-normalised pixels live roughly in [-2.2, 2.7].
        assert x.min() >= -2.2 and x.max() <= 2.7


def test_to_tensor_values(service):
    white = np.full((CARD_H, CARD_W, 3), 255, np.uint8)
    x = service._to_tensor(white)
    expected = (1.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    np.testing.assert_allclose(x[0, :, 0, 0], expected, rtol=1e-5)


@pytest.mark.parametrize("expected_grade", range(1, 11))
def test_decodes_confident_logits(service, jpeg, expected_grade):
    # P(grade > k) high for k < expected_grade, low afterwards.
    logits = np.where(np.arange(1, 10) < expected_grade, 8.0, -8.0)
    service._session = FakeSession(logits)
    pred = service.predict(jpeg, jpeg)
    assert pred.grade_int == expected_grade
    assert pred.grade == pytest.approx(expected_grade, abs=0.01)
    assert pred.confidence > 0.99
    assert pred.found_quad_front and pred.found_quad_back


def test_non_monotone_logits_are_made_monotone(service, jpeg):
    service._session = FakeSession([5, -5, 5, 5, 5, -5, 5, 5, 5])
    pred = service.predict(jpeg, jpeg)
    probs = pred.rank_probs
    assert all(a >= b for a, b in zip(probs, probs[1:], strict=False))
    assert pred.grade_int == 2


def test_uncertain_logits_give_low_confidence(service, jpeg):
    service._session = FakeSession(np.zeros(9))
    pred = service.predict(jpeg, jpeg)
    assert 0.0 <= pred.confidence < 0.6
    assert 1 <= pred.grade_int <= 10


def test_fallback_flag_propagates(service, jpeg):
    service._session = FakeSession(np.zeros(9))
    blank = to_jpeg_bytes(np.full((900, 700, 3), 128, np.uint8))
    pred = service.predict(jpeg, blank)
    assert pred.found_quad_front and not pred.found_quad_back


def test_predict_returns_the_crops_it_used(service, jpeg):
    service._session = FakeSession(np.zeros(9))
    blank = to_jpeg_bytes(np.full((900, 700, 3), 128, np.uint8))
    pred = service.predict(jpeg, blank)
    front, back = pred.crops
    assert front.found_quad and not back.found_quad
    np.testing.assert_array_equal(front.image, detect_and_crop(jpeg).image)


# --------------------------------------------------------------------------- #
# Real model weights
# --------------------------------------------------------------------------- #
@pytest.mark.model
def test_real_model_prediction_is_valid_and_deterministic(model_path):
    svc = GradeInferenceService(model_path=model_path)
    assert svc.ready
    assert svc.metadata.get("num_thresholds") == 9
    front = to_jpeg_bytes(make_scene(seed=1).bgr)
    back = to_jpeg_bytes(make_scene(seed=2, card=make_card(border_rgb=(40, 80, 200))).bgr)
    a = svc.predict(front, back)
    b = svc.predict(front, back)
    assert 1.0 <= a.grade <= 10.0
    assert 1 <= a.grade_int <= 10
    assert 0.0 <= a.confidence <= 1.0
    assert len(a.rank_probs) == 9
    assert a.grade == pytest.approx(b.grade, abs=1e-5)


# --------------------------------------------------------------------------- #
# Test-time augmentation (opt-in)
# --------------------------------------------------------------------------- #
class BatchFakeSession:
    """Returns one logit row per batch item: row i = base + offsets[i]."""

    def __init__(self, base, offsets):
        self.base = np.asarray(base, np.float32)
        self.offsets = offsets
        self.feeds = None

    def run(self, outputs, feeds):
        self.feeds = feeds
        n = feeds["front"].shape[0]
        return [np.stack([self.base + self.offsets[i] for i in range(n)])]


def test_tta_off_by_default(service):
    assert service.tta == ()


def test_tta_unknown_variant_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown TTA"):
        GradeInferenceService(model_path=tmp_path / "missing.onnx", tta=("zoom",))


def test_tta_batches_variants_and_averages_probs(tmp_path, jpeg):
    from backend.app.ml.inference import DEFAULT_TTA

    svc = GradeInferenceService(model_path=tmp_path / "missing.onnx", tta=True)
    assert svc.tta == DEFAULT_TTA
    base = np.where(np.arange(1, 10) < 9, 8.0, 0.0)  # last threshold undecided
    offs = [0.0, 2.0, -2.0, 1.0]
    svc._session = BatchFakeSession(base, offs)
    pred = svc.predict(jpeg, jpeg)
    assert svc._session.feeds["front"].shape == (len(DEFAULT_TTA), 3, CARD_H, CARD_W)
    # hflip variant really is mirrored.
    f = svc._session.feeds["front"]
    np.testing.assert_allclose(f[1], f[0][:, :, ::-1], atol=1e-6)
    expected_last = np.mean([1 / (1 + np.exp(-(0.0 + o))) for o in offs])
    assert pred.rank_probs[-1] == pytest.approx(expected_last, rel=1e-4)
    assert all(a >= b for a, b in zip(pred.rank_probs, pred.rank_probs[1:], strict=False))


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
    assert svc.kind == "legacy"  # probe artifacts missing -> legacy fallback
    svc.kind = "probe"
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


def test_explicit_model_path_selects_legacy(tmp_path):
    svc = GradeInferenceService(model_path=tmp_path / "missing.onnx")
    assert svc.kind == "legacy"


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


def _save_head(path, dim):
    from backend.app.ml.features import REGION_NAMES

    _zero_head(np.zeros((2, len(REGION_NAMES), dim))).save(path)


def test_probe_files_load_as_probe(probe_files):
    head_path, feat_path = probe_files
    _save_head(head_path, 4)
    assert GradeInferenceService(head_path=head_path, features_path=feat_path).kind == "probe"


@pytest.mark.parametrize("problem", ["corrupt_head", "dim_mismatch"])
def test_unloadable_probe_falls_back_to_legacy(probe_files, problem):
    head_path, feat_path = probe_files
    if problem == "corrupt_head":
        head_path.write_bytes(b"not an npz")
    else:
        _save_head(head_path, 5)
    svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert svc.kind == "legacy"
    assert svc.model_path != head_path


def test_tta_rejected_for_probe_model(probe_files):
    head_path, feat_path = probe_files
    _save_head(head_path, 4)
    with pytest.raises(ValueError, match="legacy model only"):
        GradeInferenceService(head_path=head_path, features_path=feat_path, tta=True)


@pytest.mark.model
def test_real_probe_model_prediction_is_valid_and_deterministic():
    from backend.app.core.config import get_settings

    s = get_settings()
    if not (s.grade_head_path.exists() and s.feature_model_path.exists()):
        pytest.skip("probe model not trained (python -m ml.training.train_probe)")
    svc = GradeInferenceService()
    assert svc.kind == "probe" and svc.ready
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
