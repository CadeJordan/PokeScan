"""GradeInferenceService: tensor prep + logit decoding (fake session), and real weights."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.ml.inference import GradeInferenceService
from backend.app.ml.preprocess import CARD_H, CARD_W
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
