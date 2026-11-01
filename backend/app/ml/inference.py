"""ONNX Runtime inference for the grade model.

Two model kinds, picked at load time:

- probe (preferred): frozen DINOv2 region features (`card_features.onnx`) +
  the linear ordinal head (`grade_head.npz`). See `features.py` and
  `grade_head.py`; trained by `python -m ml.training.train_probe`.
- legacy: the end-to-end ConvNeXt CORN model (`grade_model.onnx`) from
  `python -m ml.training.export`. Used when an explicit `model_path` is given,
  or with ALLOW_LEGACY_MODEL=1 when the probe is missing or fails to load.

Otherwise the service fails closed: no model is loaded, `/api/grade` returns
503 and `metadata["probe_load_error"]` (shown by `/api/health`) says why.

With the probe model, the face guard (`face_guard.py`, `face_guard.npz` next to
the head) swaps swapped uploads back and rotates upside-down faces before the
head runs, with a note. When it reads both faces as the same side it changes
nothing and sets `GradePrediction.same_side`; `predict` still grades the pair,
and `/api/grade` refuses it with 422 (via `GradePrediction.rejection()`). It
fails open: if it can't be loaded or trusted,
`metadata["face_guard_load_error"]` says why and grading runs without it.

Thread-safe for concurrent requests.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

import cv2
import numpy as np
import onnxruntime as ort

from backend.app.core.config import get_settings
from backend.app.ml.preprocess import (
    CARD_H,
    CARD_W,
    CardCrop,
    ImageDecodeError,
    detect_and_crop,
    encode_jpeg,
)

if TYPE_CHECKING:
    from backend.app.ml.face_guard import GuardDecision

log = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _jpeg_roundtrip(rgb: np.ndarray) -> np.ndarray:
    buf = np.frombuffer(encode_jpeg(rgb), np.uint8)
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


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
    # Probe model only: soft grade suggested by each region group
    # (corners / edges / surface / global), used for the sub-grades.
    region_grades: dict[str, float] | None = None
    # The (front, back) crops the grade was computed from, so callers (the
    # centering and condition sub-grades) reuse them instead of detecting again.
    # Not part of the prediction's value: left out of == and repr (big arrays).
    crops: tuple[CardCrop, CardCrop] | None = field(default=None, compare=False, repr=False)
    # What the face guard changed about the input, for the response notes.
    notes: list[str] = field(default_factory=list)
    # The guard swapped the faces: `found_quad_*` and `crops` are in graded order,
    # so the uploaded front is the graded back.
    swapped: bool = False
    # "front" / "back" when the guard read both faces as that side (graded as is).
    same_side: str | None = None
    # The guard's decision with its probabilities (P(back) per uploaded row,
    # P(flipped) per face after any swap), for diagnostics such as `ml.eval`.
    # None without a guard. Not part of the prediction's value.
    guard_decision: GuardDecision | None = field(default=None, compare=False, repr=False)

    def rejection(self) -> tuple[str, str] | None:
        """`(kind, message)` when `/api/grade` refuses this prediction (422), else None.

        Probe model only: a face with no card outline (the padded whole photo,
        which the probe never trained on) is `"card_not_found"`, and a pair the
        face guard read as two of one side is `"same_side"`. The prediction is
        still made, so `ml.eval` grades these inputs and counts them. The legacy
        model grades them as before, with a note.
        """
        if self.region_grades is None:
            return None
        # Name faces as uploaded: after a guard swap the graded front is the uploaded back.
        found = (self.found_quad_front, self.found_quad_back)
        if self.swapped:
            found = found[::-1]
        missing = [face for face, ok in zip(("front", "back"), found, strict=True) if not ok]
        if missing:
            faces = " and ".join(missing)
            return "card_not_found", f"{faces}: card not found; retake the photo on a plain background"
        if self.same_side is not None:
            return "same_side", (
                f"both photos look like card {self.same_side}s; "
                "upload one photo of the front and one of the back"
            )
        return None


def detect_face(image: bytes, label: str) -> CardCrop:
    """`detect_and_crop` of one upload (decode, EXIF transpose, 1024 px cap,
    detection), with an undecodable upload's error naming its face."""
    try:
        return detect_and_crop(image)
    except ImageDecodeError as exc:
        raise ImageDecodeError(f"{label}: {exc}", face=label) from exc


def extractor_input(crop: CardCrop) -> np.ndarray:
    """The RGB array the probe's feature extractor sees for `crop`.

    Training read crops back from precrop's JPEGs, so the crop is JPEG
    round-tripped to match that encoding.
    """
    return _jpeg_roundtrip(crop.image)


def served_extractor_input(image: bytes, label: str = "face") -> np.ndarray:
    """What the probe path feeds the extractor for the upload `image`.

    `detect_face` then `extractor_input`: the steps `predict` runs on each
    upload before its first extraction. `ml.training.train_face_guard` trains
    the guard on features of this, so training and serving share one code path.
    """
    return extractor_input(detect_face(image, label))


def _rotate_crop(crop: CardCrop) -> CardCrop:
    """The crop turned 180 degrees. The quad is re-ordered to match, so it still
    lists (tl, tr, br, bl) of the rotated card in the frame of `source_bgr`."""
    quad = None if crop.quad is None else np.roll(crop.quad, 2, axis=0)
    image = np.ascontiguousarray(crop.image[::-1, ::-1])
    return dataclasses.replace(crop, image=image, quad=quad)


class GradeInferenceService:
    """Wraps the model session(s) + the preprocessing pipeline."""

    def __init__(
        self,
        model_path: Path | None = None,
        tta: bool | tuple[str, ...] = False,
        head_path: Path | None = None,
        features_path: Path | None = None,
    ) -> None:
        """``model_path`` selects the legacy CORN ONNX model explicitly. Otherwise
        the probe model (``head_path`` + ``features_path``, defaulting to the
        settings) is the only model served.

        If a probe file is missing or loading fails (corrupt file, head and
        feature model don't match, a `ten_cutoff` in the head's metadata that
        isn't a number in [0.5, 1)), the error is kept in ``metadata`` as
        ``probe_load_error`` and the service is not ready, unless
        ``settings.allow_legacy_model`` is set: then the legacy active model is
        loaded instead. ``kind`` stays "legacy" until a probe loads; it only
        matters when ``ready``.

        ``tta`` (legacy model only): False (default, single pass), True
        (DEFAULT_TTA) or variant names. With TTA the ONNX model runs once on a
        batch of variants and the cumulative probabilities are averaged.
        """
        settings = get_settings()
        if tta is True:
            tta = DEFAULT_TTA
        unknown = set(tta or ()) - set(TTA_VARIANTS)
        if unknown:
            raise ValueError(f"unknown TTA variants: {sorted(unknown)}")
        self.tta: tuple[str, ...] = tuple(tta or ())
        self.head_path = Path(head_path or settings.grade_head_path)
        self.features_path = Path(features_path or settings.feature_model_path)
        self.kind = "legacy"
        self.model_path = Path(model_path or settings.active_model_path)
        self.probe_load_error: str | None = None
        self.face_guard_load_error: str | None = None
        self._guard = None
        self._session: ort.InferenceSession | None = None
        self._meta: dict = {}
        self._ten_cutoff = 0.5  # the probe head's P(10) cut-off, set by _load_probe
        self._lock = Lock()
        if model_path is None:
            try:
                self._load_probe()
            except Exception as exc:  # noqa: BLE001
                self.probe_load_error = f"{type(exc).__name__}: {exc}"
                self._session, self._meta = None, {}
                trace = not isinstance(exc, FileNotFoundError)
                if not settings.allow_legacy_model:
                    log.error(
                        "probe grade model %s + %s not loaded (%s); /grade will return 503. "
                        "Set ALLOW_LEGACY_MODEL=1 to serve the legacy model instead",
                        self.head_path,
                        self.features_path,
                        self.probe_load_error,
                        exc_info=trace,
                    )
                    return
                log.warning(
                    "probe grade model %s + %s not loaded (%s); ALLOW_LEGACY_MODEL is set, "
                    "falling back to legacy %s",
                    self.head_path,
                    self.features_path,
                    self.probe_load_error,
                    self.model_path,
                    exc_info=trace,
                )
            else:
                self.kind, self.model_path = "probe", self.head_path
                if self.tta:
                    raise ValueError("TTA applies to the legacy model only")
                return
        self._load()

    def _load_probe(self) -> None:
        from backend.app.ml.features import FeatureExtractor
        from backend.app.ml.grade_head import GradeHead

        missing = [p.name for p in (self.head_path, self.features_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(", ".join(missing))
        head = GradeHead.load(self.head_path)
        self._check_feature_spec(head.metadata)
        self._ten_cutoff = head.ten_cutoff  # a bad cut-off fails closed, like a bad hash
        extractor = FeatureExtractor(self.features_path)
        dim = extractor.session.get_outputs()[0].shape[-1]
        if isinstance(dim, int):
            head.check_feature_dim(dim)
        self._extractor, self._head = extractor, head
        self._session = extractor.session
        self._meta = dict(head.metadata)
        log.info(
            "loaded probe grade model %s + %s (providers=%s)",
            self.head_path,
            self.features_path,
            self._session.get_providers(),
        )
        self._load_guard(dim if isinstance(dim, int) else None)

    @property
    def face_guard_path(self) -> Path:
        return self.head_path.with_name("face_guard.npz")

    def _load_guard(self, dim: int | None) -> None:
        """Load the face guard if `face_guard.npz` sits next to the head.

        Fails open: any problem (an error importing the guard or stat-ing its file,
        an unreadable file, a wrong weight shape or width, a bad bias or threshold,
        a `feature_spec_hash` in `face_guard.json` that is missing, unreadable or
        not the code's) leaves no guard, logs a WARNING and records
        `face_guard_load_error`; grading is then exactly as without a guard. Nothing
        here may make the probe unready.
        """
        path = self.face_guard_path
        try:
            from backend.app.ml.face_guard import FaceGuard
            from backend.app.ml.features import feature_spec_hash

            if not path.exists():
                return
            sidecar = path.with_suffix(".json")
            if not sidecar.exists():
                raise ValueError(f"{sidecar.name} missing")
            try:
                meta = json.loads(sidecar.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError(f"{sidecar.name} unreadable") from exc
            if not isinstance(meta, dict):
                raise ValueError(f"{sidecar.name} unreadable")
            have, code = meta.get("feature_spec_hash"), feature_spec_hash()
            if not isinstance(have, str) or not have:
                raise ValueError(f"feature_spec_hash {have!r} in {sidecar.name} is not a hash")
            if have != code:
                raise ValueError(f"feature_spec_hash {have} ({sidecar.name}) != code {code}")
            guard = FaceGuard.load(path)
            if dim is not None:
                guard.check_feature_dim(dim)
        except Exception as exc:  # noqa: BLE001
            self.face_guard_load_error = f"{type(exc).__name__}: {exc}"
            log.warning(
                "face guard %s not loaded (%s); grading without it - retrain with "
                "`python -m ml.training.train_face_guard`",
                path,
                self.face_guard_load_error,
            )
            return
        self._guard = guard
        log.info("loaded face guard %s (enabled: %s)", path, ", ".join(guard.enabled) or "none")

    def _check_feature_spec(self, head_meta: dict) -> None:
        """Raise ValueError unless head, feature graph and this code share one feature spec.

        A head with a `feature_spec_hash` key needs a non-empty string there, equal
        to the code's hash and to the sidecar's. A head without the key (artifacts
        from before the check) loads with a WARNING, unless the sidecar records
        another spec than the code's. An OS error reading the sidecar (e.g. a
        directory at that path) propagates, so it fails closed for every head.
        """
        from backend.app.ml.features import feature_spec_hash

        code_hash = feature_spec_hash()
        sidecar = self.features_path.with_suffix(".json")
        feat_meta = None
        if sidecar.exists():
            try:
                feat_meta = json.loads(sidecar.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                log.exception("could not parse feature model metadata at %s", sidecar)
        has_head_hash = "feature_spec_hash" in head_meta
        head_hash = head_meta.get("feature_spec_hash")
        feat_hash = feat_meta.get("feature_spec_hash") if isinstance(feat_meta, dict) else None
        problems = []
        if has_head_hash and (not isinstance(head_hash, str) or not head_hash):
            problems.append(f"head {head_hash!r} ({self.head_path.name}) is not a hash")
        elif has_head_hash:
            if head_hash != code_hash:
                problems.append(f"head {head_hash} ({self.head_path.name}) != code {code_hash}")
            if not sidecar.exists():
                problems.append(f"{sidecar.name} missing")
            elif not isinstance(feat_meta, dict):
                problems.append(f"{sidecar.name} unreadable")
            elif feat_hash != head_hash:
                problems.append(f"{sidecar.name} {feat_hash} != head {head_hash}")
        elif feat_hash and feat_hash != code_hash:
            problems.append(f"{sidecar.name} {feat_hash} != code {code_hash}")
        if problems:
            raise ValueError(
                f"feature_spec_hash: {'; '.join(problems)}: "
                "retrain with `python -m ml.training.train_probe`"
            )
        if not has_head_hash:
            # Artifacts from before the check: keep loading them.
            log.warning(
                "feature_spec_hash missing (head %s, %s %s); can't check that head and "
                "feature model match - retrain to record it",
                head_hash,
                sidecar.name,
                feat_hash,
            )

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
        meta = dict(self._meta)
        if self.probe_load_error is not None:
            meta["probe_load_error"] = self.probe_load_error
        # Only when a face_guard.npz exists: without one, metadata is as before.
        if self.kind == "probe" and (self._guard is not None or self.face_guard_load_error):
            guard = self._guard
            meta["face_guard"] = {
                "loaded": guard is not None,
                "enabled_checks": list(guard.enabled) if guard is not None else [],
                "thresholds": dict(guard.thresholds) if guard is not None else {},
            }
            if self.face_guard_load_error is not None:
                meta["face_guard_load_error"] = self.face_guard_load_error
        return meta

    def _to_tensor(self, rgb: np.ndarray) -> np.ndarray:
        x = rgb.astype(np.float32) / 255.0
        x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
        x = np.transpose(x, (2, 0, 1))[None, ...]  # (1, 3, H, W)
        return np.ascontiguousarray(x)

    def predict(self, front_bytes: bytes, back_bytes: bytes) -> GradePrediction:
        if self._session is None:
            raise RuntimeError("grade model not loaded")

        front_crop = detect_face(front_bytes, "front")
        back_crop = detect_face(back_bytes, "back")
        if front_crop.image.shape[:2] != (CARD_H, CARD_W):
            raise RuntimeError("preprocessing produced wrong shape")
        if self.kind == "probe":
            return self._predict_probe(front_crop, back_crop)

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
            crops=(front_crop, back_crop),
        )

    def _predict_probe(self, front_crop: CardCrop, back_crop: CardCrop) -> GradePrediction:
        from backend.app.ml.grade_head import decode

        front, back = (extractor_input(c) for c in (front_crop, back_crop))
        with self._lock:
            feats = self._extractor(front, back)
        notes: list[str] = []
        decision = None
        if self._guard is not None:
            try:
                decision = self._guard.decide(
                    feats, (front_crop.found_quad, back_crop.found_quad)
                )
            except Exception:  # noqa: BLE001 - a bad guard never fails the request
                log.warning(
                    "face guard %s failed on this request; graded it unguarded",
                    self.face_guard_path,
                    exc_info=True,
                )
        if decision is not None:
            notes = list(decision.notes)
            if decision.swap:
                front_crop, back_crop = back_crop, front_crop
                feats = np.ascontiguousarray(feats[::-1])
            if any(decision.flip):
                flip_front, flip_back = decision.flip
                front_crop = _rotate_crop(front_crop) if flip_front else front_crop
                back_crop = _rotate_crop(back_crop) if flip_back else back_crop
                front, back = (extractor_input(c) for c in (front_crop, back_crop))
                with self._lock:
                    feats = self._extractor(front, back)
        out = self._head(feats)
        expected, hard, confidence, _ = decode(out.cum_probs, self._ten_cutoff)
        return GradePrediction(
            grade=expected,
            grade_int=hard,
            confidence=confidence,
            rank_probs=[float(p) for p in out.cum_probs],
            found_quad_front=front_crop.found_quad,
            found_quad_back=back_crop.found_quad,
            region_grades=out.region_grades,
            crops=(front_crop, back_crop),
            notes=notes,
            swapped=decision is not None and decision.swap,
            same_side=None if decision is None else decision.same_side,
            guard_decision=decision,
        )


_SERVICE: GradeInferenceService | None = None


def get_inference_service() -> GradeInferenceService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = GradeInferenceService()
    return _SERVICE
