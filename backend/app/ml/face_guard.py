"""Face side and orientation guard for the probe grade model.

The grade head is position dependent: it concatenates [front, back], and the
corner probe concatenates TL/TR/BR/BL. So a swapped upload or an upside-down
photo shifts the grade silently. Before the head runs, `FaceGuard` checks each
face's global DINOv2 features (CLS + mean, `GLOBAL_SLICE`) with three linear
logistic probes:

- `side`: P(back), trained on fronts and backs in both orientations;
- `front_upright`: P(flipped 180 degrees) for a front;
- `back_upright`: P(flipped 180 degrees) for a back.

Each probe is a `(weight, bias)` pair with the standardisation folded in, so
the guard is plain numpy. Each check has its own threshold, chosen from
card-grouped out-of-fold CV on train+val (`ml/training/train_face_guard.py`).
A check that missed its training gate is stored but not enabled.

Saved as `face_guard.npz` (weights, biases, thresholds; no pickle) plus a
`face_guard.json` sidecar (enabled checks, CV metrics, `feature_spec_hash`).
Unlike the grade head, the guard fails open: a guard that can't be trusted is
dropped and grading runs as if it didn't exist (see `inference.py`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend.app.ml.features import GLOBAL_SLICE

log = logging.getLogger(__name__)

CHECKS = ("side", "front_upright", "back_upright")
FACES = ("front", "back")

NOTE_SWAPPED = "front and back photos looked swapped; graded them the other way round"
NOTE_BOTH = "both photos look like card {}s"
NOTE_FLIPPED = "{} photo was upside down; rotated it"


def guard_input(feats: np.ndarray) -> np.ndarray:
    """(..., n_regions, D) face features -> (..., 2*D) CLS + mean vector."""
    sub = np.asarray(feats)[..., GLOBAL_SLICE, :]
    return sub.reshape(*sub.shape[:-2], -1)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -z))


@dataclass(slots=True)
class LogisticProbe:
    """P(positive) = sigmoid(x . weight + bias)."""

    weight: np.ndarray  # (2*D,)
    bias: float

    def prob(self, x: np.ndarray) -> np.ndarray:
        return _sigmoid(np.asarray(x, np.float64) @ self.weight + self.bias)


@dataclass(slots=True)
class GuardDecision:
    swap: bool                     # rows were front/back swapped
    flip: tuple[bool, bool]        # (front, back) after the swap: rotate 180 degrees
    notes: list[str] = field(default_factory=list)
    p_back: tuple[float, float] | None = None   # P(back) per uploaded row, if checked
    p_flip: tuple[float, float] | None = None   # P(flipped) per face after the swap


@dataclass(slots=True)
class FaceGuard:
    probes: dict[str, LogisticProbe]   # one per name in CHECKS
    thresholds: dict[str, float]       # one per name in CHECKS
    enabled: tuple[str, ...]           # checks that passed their training gate
    metadata: dict = field(default_factory=dict)

    @property
    def side_threshold(self) -> float:
        return self.thresholds["side"]

    def check_feature_dim(self, dim: int | None) -> None:
        """Raise ValueError unless every probe is valid for `dim`-wide region features.

        Each weight must be 1-D with shape (2*dim,) (any length when `dim` is None),
        each bias a finite scalar, each threshold finite and in (0.5, 1): a lower
        side threshold lets "looks back" and "looks front" overlap, and a flip
        threshold at or below 0.5 rotates faces the probe is unsure about.
        """
        for name in CHECKS:
            w = np.asarray(self.probes[name].weight)
            if w.ndim != 1:
                raise ValueError(f"face guard probe {name!r} has weight shape {w.shape}; need 1-D")
            if dim is not None and w.shape != (2 * dim,):
                raise ValueError(
                    f"face guard probe {name!r} has {w.size} weights; "
                    f"{dim}-dim features need {2 * dim}"
                )
            if not np.isfinite(w).all():
                raise ValueError(f"face guard probe {name!r} has non-finite weights")
            if not _finite_scalar(self.probes[name].bias):
                raise ValueError(f"face guard probe {name!r} bias {self.probes[name].bias!r} is not finite")
            t = self.thresholds[name]
            if not _finite_scalar(t) or not 0.5 < t < 1.0:
                raise ValueError(f"face guard threshold {name!r} {t!r} is not in (0.5, 1)")

    def p_back(self, feats: np.ndarray) -> np.ndarray:
        """(2, n_regions, D) [front, back] features -> (2,) P(back) per row."""
        return self.probes["side"].prob(guard_input(feats))

    def p_flip(self, feats: np.ndarray) -> np.ndarray:
        """(2, n_regions, D) [front, back] features -> (2,) P(flipped), each face with its probe."""
        x = guard_input(feats)
        return np.array(
            [self.probes["front_upright"].prob(x[0]), self.probes["back_upright"].prob(x[1])]
        )

    def check_sides(self, feats: np.ndarray) -> tuple[bool, list[str], tuple[float, float] | None]:
        """(swap?, notes, P(back) per row). Swap only when both rows read as the other side."""
        if "side" not in self.enabled:
            return False, [], None
        p = self.p_back(feats)
        t = self.side_threshold
        looks_back = p > t
        looks_front = p < 1.0 - t
        p_back = (float(p[0]), float(p[1]))
        if looks_back[0] and looks_front[1]:
            return True, [NOTE_SWAPPED], p_back
        if looks_back.all():
            return False, [NOTE_BOTH.format("back")], p_back
        if looks_front.all():
            return False, [NOTE_BOTH.format("front")], p_back
        return False, [], p_back

    def check_orientation(
        self, feats: np.ndarray
    ) -> tuple[tuple[bool, bool], list[str], tuple[float, float] | None]:
        """((flip front?, flip back?), notes, P(flipped) per face) for [front, back] rows."""
        checks = ("front_upright", "back_upright")
        if not any(c in self.enabled for c in checks):
            return (False, False), [], None
        p = self.p_flip(feats)
        flip = tuple(
            bool(c in self.enabled and p[i] > self.thresholds[c]) for i, c in enumerate(checks)
        )
        notes = [NOTE_FLIPPED.format(face) for face, f in zip(FACES, flip, strict=True) if f]
        return flip, notes, (float(p[0]), float(p[1]))

    def decide(self, feats: np.ndarray) -> GuardDecision:
        """Side check, then the orientation check on the (possibly swapped) rows.

        When both rows read as the same side (a "both photos look like" note), the
        orientation check is skipped: it isn't known which upright probe applies to
        which face, and the front probe fires on upright backs (5.5% at its
        threshold), so a rotation there would be a guess.
        """
        swap, notes, p_back = self.check_sides(feats)
        if notes and not swap:  # same-side pair: note only
            return GuardDecision(False, (False, False), notes, p_back, None)
        rows = feats[::-1] if swap else feats
        flip, flip_notes, p_flip = self.check_orientation(rows)
        return GuardDecision(swap, flip, notes + flip_notes, p_back, p_flip)

    # ------------------------------------------------------------------ io
    def save(self, path: Path) -> None:
        """Weights and thresholds to `path` (.npz, no pickle); metadata to the .json sidecar."""
        arrays: dict[str, np.ndarray] = {}
        for name in CHECKS:
            arrays[f"{name}.weight"] = np.asarray(self.probes[name].weight, np.float64)
            arrays[f"{name}.bias"] = np.array(float(self.probes[name].bias))
            arrays[f"{name}.threshold"] = np.array(float(self.thresholds[name]))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        meta = {
            **self.metadata,
            "enabled_checks": list(self.enabled),
            "thresholds": {n: float(self.thresholds[n]) for n in CHECKS},
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path) -> FaceGuard:
        """Read `path` and its sidecar. Enabled checks come from the sidecar's
        `enabled_checks`; without a readable sidecar no check is enabled."""
        with np.load(path, allow_pickle=False) as z:
            probes = {
                n: LogisticProbe(
                    weight=z[f"{n}.weight"].astype(np.float64), bias=_scalar(z, f"{n}.bias")
                )
                for n in CHECKS
            }
            thresholds = {n: _scalar(z, f"{n}.threshold") for n in CHECKS}
        metadata: dict = {}
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            try:
                loaded = json.loads(meta_path.read_text())
                metadata = loaded if isinstance(loaded, dict) else {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                log.exception("could not parse face guard metadata at %s", meta_path)
        enabled = metadata.get("enabled_checks", [])
        if not isinstance(enabled, list) or not set(enabled) <= set(CHECKS):
            raise ValueError(f"{meta_path.name}: enabled_checks {enabled!r} is not a list of {CHECKS}")
        guard = cls(
            probes=probes,
            thresholds=thresholds,
            enabled=tuple(n for n in CHECKS if n in enabled),
            metadata={k: v for k, v in metadata.items() if k not in ("enabled_checks", "thresholds")},
        )
        guard.check_feature_dim(None)  # shapes, finiteness, threshold range; width checked later
        return guard


def _finite_scalar(v) -> bool:
    return isinstance(v, (int, float, np.floating, np.integer)) and bool(np.isfinite(v))


def _scalar(z, key: str) -> float:
    """A 0-d numeric array from the npz as a float; ValueError for any other shape."""
    a = z[key]
    if a.shape != () or a.dtype.kind not in "iuf":
        raise ValueError(f"{key} has shape {a.shape} / dtype {a.dtype}; need a numeric scalar")
    return float(a)
