"""ONNX Runtime inference for the grade model.

Loads a single ONNX model on import; thread-safe for concurrent requests.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
import onnxruntime as ort

from backend.app.core.config import get_settings
from backend.app.ml.preprocess import CARD_H, CARD_W, detect_and_crop

log = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _scale_brightness(rgb: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)


# Test-time augmentation variants (applied to the cropped RGB card). Only
# condition-preserving changes: mirror + mild exposure. Off by default.
TTA_VARIANTS = {
    "identity": lambda im: im,
    "hflip": lambda im: np.ascontiguousarray(im[:, ::-1]),
    "dim": lambda im: _scale_brightness(im, 0.9),
    "bright": lambda im: _scale_brightness(im, 1.1),
}
DEFAULT_TTA = ("identity", "hflip", "dim", "bright")


@dataclass(slots=True)
class GradePrediction:
    grade: float           # soft expected grade in [1, 10]
    grade_int: int         # hard rank in {1..10}
    confidence: float      # in [0, 1]
    rank_probs: list[float]  # cumulative P(grade > k) for k in 1..9
    found_quad_front: bool
    found_quad_back: bool


class GradeInferenceService:
    """Wraps an ONNX session + the preprocessing pipeline."""

    def __init__(
        self, model_path: Path | None = None, tta: bool | tuple[str, ...] = False
    ) -> None:
        """``tta``: False (default, single pass), True (DEFAULT_TTA) or variant names.

        With TTA the ONNX model runs once on a batch of variants and the
        cumulative probabilities are averaged before decoding.
        """
        settings = get_settings()
        self.model_path = Path(model_path or settings.active_model_path)
        if tta is True:
            tta = DEFAULT_TTA
        unknown = set(tta or ()) - set(TTA_VARIANTS)
        if unknown:
            raise ValueError(f"unknown TTA variants: {sorted(unknown)}")
        self.tta: tuple[str, ...] = tuple(tta or ())
        self._session: ort.InferenceSession | None = None
        self._meta: dict = {}
        self._lock = Lock()
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            log.warning(
                "active model not found at %s - /grade endpoint will return 503",
                self.model_path,
            )
            return
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=sess_opts, providers=providers
        )
        meta_path = self.model_path.with_suffix(".json")
        if meta_path.exists():
            try:
                self._meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                log.exception("could not parse model metadata at %s", meta_path)
        log.info("loaded grade model %s (providers=%s)", self.model_path, providers)

    @property
    def ready(self) -> bool:
        return self._session is not None

    @property
    def metadata(self) -> dict:
        return dict(self._meta)

    def _to_tensor(self, rgb: np.ndarray) -> np.ndarray:
        x = rgb.astype(np.float32) / 255.0
        x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
        x = np.transpose(x, (2, 0, 1))[None, ...]  # (1, 3, H, W)
        return np.ascontiguousarray(x)

    def predict(self, front_bytes: bytes, back_bytes: bytes) -> GradePrediction:
        if self._session is None:
            raise RuntimeError("grade model not loaded")

        front_crop = detect_and_crop(front_bytes)
        back_crop = detect_and_crop(back_bytes)
        if front_crop.image.shape[:2] != (CARD_H, CARD_W):
            raise RuntimeError("preprocessing produced wrong shape")

        variants = self.tta or ("identity",)
        front_x = np.concatenate(
            [self._to_tensor(TTA_VARIANTS[v](front_crop.image)) for v in variants]
        )
        back_x = np.concatenate(
            [self._to_tensor(TTA_VARIANTS[v](back_crop.image)) for v in variants]
        )

        with self._lock:
            outputs = self._session.run(
                ["corn_logits"],
                {"front": front_x, "back": back_x},
            )
        logits = outputs[0]  # (V, K-1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = np.minimum.accumulate(probs, axis=1)  # enforce monotonicity
        probs = probs.mean(axis=0)  # average over TTA variants (V=1 without TTA)
        hard_rank = int(1 + (probs > 0.5).sum())
        expected = float(1.0 + probs.sum())

        # Per-class probabilities for confidence.
        cum_full = np.concatenate([[1.0], probs, [0.0]])
        per_class = np.clip(cum_full[:-1] - cum_full[1:], 0.0, None)
        s = per_class.sum()
        if s > 0:
            per_class /= s
        confidence = float(per_class[max(0, min(hard_rank - 1, len(per_class) - 1))])

        return GradePrediction(
            grade=expected,
            grade_int=hard_rank,
            confidence=confidence,
            rank_probs=[float(p) for p in probs],
            found_quad_front=front_crop.found_quad,
            found_quad_back=back_crop.found_quad,
        )


_SERVICE: GradeInferenceService | None = None


def get_inference_service() -> GradeInferenceService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = GradeInferenceService()
    return _SERVICE
