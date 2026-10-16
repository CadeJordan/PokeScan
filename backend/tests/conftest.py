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
