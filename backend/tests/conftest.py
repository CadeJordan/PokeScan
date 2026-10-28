from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.ml.inference import GradePrediction
from backend.tests.helpers.synth import make_card, make_scene, to_jpeg_bytes


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="session")
def card_rgb() -> np.ndarray:
    return make_card(seed=0)


@pytest.fixture(scope="session")
def scene_jpeg() -> bytes:
    """A clean synthetic photo of a card, as JPEG bytes (an API-style upload)."""
    return to_jpeg_bytes(make_scene(seed=0, condition="clean").bgr)


@pytest.fixture(scope="session")
def model_path():
    path = get_settings().active_model_path
    if not path.exists():
        pytest.skip(f"model weights not found at {path}")
    return path


@pytest.fixture(scope="session")
def tiny_legacy_onnx(tmp_path_factory):
    """A small ONNX graph with the legacy grade_model.onnx I/O contract."""
    torch = pytest.importorskip("torch")
    from backend.app.ml.preprocess import CARD_H, CARD_W

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("offsets", torch.linspace(4.0, -4.0, 9))

        def forward(self, front, back):
            shift = (front.mean((1, 2, 3)) - back.mean((1, 2, 3)))[:, None]
            return shift + self.offsets[None]

    path = tmp_path_factory.mktemp("legacy") / "grade_model.onnx"
    x = torch.zeros(1, 3, CARD_H, CARD_W)
    torch.onnx.export(
        Tiny().eval(),
        (x, x),
        str(path),
        input_names=["front", "back"],
        output_names=["corn_logits"],
        dynamic_axes={"front": {0: "n"}, "back": {0: "n"}, "corn_logits": {0: "n"}},
        opset_version=17,
    )
    return path


@pytest.fixture
def serve_env(monkeypatch: pytest.MonkeyPatch):
    """Inference reads fresh Settings from the environment; returns a setenv helper.

    The legacy opt-in starts off, so tests don't depend on the developer's .env.
    """
    import backend.app.ml.inference as inference
    from backend.app.core.config import Settings

    monkeypatch.setattr(inference, "get_settings", Settings)
    monkeypatch.setenv("ALLOW_LEGACY_MODEL", "0")

    def set_env(**env: object) -> None:
        for key, value in env.items():
            monkeypatch.setenv(key, str(value))

    return set_env


# --------------------------------------------------------------------------- #
# Fake inference service for API tests (no ONNX model needed)
# --------------------------------------------------------------------------- #
@dataclass
class FakeService:
    ready: bool = True
    metadata: dict = field(default_factory=lambda: {"temperature": 1.0})
    error: Exception | None = None
    calls: list[tuple[bytes, bytes]] = field(default_factory=list)
    prediction: GradePrediction = field(
        default_factory=lambda: GradePrediction(
            grade=8.62,
            grade_int=9,
            confidence=0.4123,
            rank_probs=[0.99, 0.99, 0.98, 0.97, 0.95, 0.9, 0.8, 0.55, 0.3],
            found_quad_front=True,
            found_quad_back=True,
        )
    )

    def predict(self, front: bytes, back: bytes) -> GradePrediction:
        self.calls.append((front, back))
        if self.error is not None:
            raise self.error
        return self.prediction


@pytest.fixture
def fake_service() -> FakeService:
    return FakeService()


@pytest.fixture
def client(fake_service: FakeService, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    import backend.app.api.grade as grade_mod
    import backend.app.main as main_mod

    monkeypatch.setattr(grade_mod, "get_inference_service", lambda: fake_service)
    monkeypatch.setattr(main_mod, "get_inference_service", lambda: fake_service)
    with TestClient(main_mod.create_app()) as c:
        yield c
