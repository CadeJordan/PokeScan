"""Linear ordinal grade head over frozen DINOv2 region features.

Structure (fitted by `ml/training/train_probe.py`):

- Four region probes, each a proportional-odds model over one feature group of
  both faces: corners (4 corner regions x 2 faces), edges (4 edge strips x 2),
  surface (centre x 2) and global (CLS + mean x 2). Each gives a soft grade:
  the grade that region alone suggests.
- A meta probe, also proportional-odds, over the four soft grades. Its
  cumulative probabilities are the model output.

Proportional odds: P(grade > k) = sigmoid(w . x + c - b_k), k = 1..9, with
increasing thresholds b. One shared score makes the probabilities monotone by
construction (no cummin needed) and uses the few low-grade examples for every
threshold at once. Standardisation and PCA are folded into `w` and `c` at fit
time, so inference is one dot product per probe and plain numpy.

The region soft grades double as the corners / edges / surface sub-grades.
They are trained against the overall PSA grade (there are no sub-grade
labels), so they mean "how this region compares with cards of each grade",
not PSA's own sub-grade.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.app.ml.features import (
    CENTER_SLICE,
    CORNER_SLICE,
    EDGE_SLICE,
    GLOBAL_SLICE,
    REGION_NAMES,
)

log = logging.getLogger(__name__)

NUM_GRADES = 10
PROBE_NAMES = ("corners", "edges", "surface", "global")
_SLICES = {
    "corners": CORNER_SLICE,
    "edges": EDGE_SLICE,
    "surface": CENTER_SLICE,
    "global": GLOBAL_SLICE,
}


def probe_input(feats: np.ndarray, name: str) -> np.ndarray:
    """(..., 2, n_regions, D) face features -> (..., n) input vector for probe `name`."""
    sub = feats[..., _SLICES[name], :]
    return sub.reshape(*feats.shape[:-3], -1)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -z))  # 1 / (1 + e^-z) without overflow


@dataclass(slots=True)
class OrdinalProbe:
    """P(grade > k) = sigmoid(x . weight + bias - thresholds[k-1])."""

    weight: np.ndarray       # (n,)
    bias: float
    thresholds: np.ndarray   # (NUM_GRADES - 1,), increasing

    def score(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, np.float64) @ self.weight + self.bias

    def cum_probs(self, x: np.ndarray) -> np.ndarray:
        """(..., n) -> (..., 9) cumulative P(grade > k), non-increasing in k."""
        return _sigmoid(self.score(x)[..., None] - self.thresholds)

    def soft_grade(self, x: np.ndarray) -> np.ndarray:
        return 1.0 + self.cum_probs(x).sum(-1)


@dataclass(slots=True)
class HeadOutput:
    cum_probs: np.ndarray          # (9,) P(grade > k), k = 1..9
    region_grades: dict[str, float]  # soft grade per region probe


@dataclass(slots=True)
class GradeHead:
    probes: dict[str, OrdinalProbe]
    meta: OrdinalProbe
    metadata: dict

    def check_feature_dim(self, dim: int) -> None:
        """Raise ValueError unless the probes fit `dim`-wide region features."""
        feats = np.zeros((2, len(REGION_NAMES), dim))
        for n in PROBE_NAMES:
            want = probe_input(feats, n).size
            if self.probes[n].weight.size != want:
                raise ValueError(
                    f"grade head probe {n!r} has {self.probes[n].weight.size} weights; "
                    f"{dim}-dim features need {want} (head and feature model don't match)"
                )
        if self.meta.weight.size != len(PROBE_NAMES):
            raise ValueError(f"meta probe has {self.meta.weight.size} weights, not {len(PROBE_NAMES)}")

    def region_grades(self, feats: np.ndarray) -> np.ndarray:
        """(..., 2, n_regions, D) -> (..., len(PROBE_NAMES)) soft grades."""
        return np.stack(
            [self.probes[n].soft_grade(probe_input(feats, n)) for n in PROBE_NAMES], -1
        )

    def __call__(self, feats: np.ndarray) -> HeadOutput:
        regions = self.region_grades(feats)
        return HeadOutput(
            cum_probs=self.meta.cum_probs(regions),
            region_grades={n: float(v) for n, v in zip(PROBE_NAMES, regions, strict=True)},
        )

    # ------------------------------------------------------------------ io
    def save(self, path: Path) -> None:
        """Weights to `path` (.npz, no pickle); metadata to the .json sidecar."""
        arrays: dict[str, np.ndarray] = {}
        for name, p in {**self.probes, "meta": self.meta}.items():
            arrays[f"{name}.weight"] = p.weight
            arrays[f"{name}.bias"] = np.array(p.bias)
            arrays[f"{name}.thresholds"] = p.thresholds
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        path.with_suffix(".json").write_text(json.dumps(self.metadata, indent=2))

    @classmethod
    def load(cls, path: Path) -> GradeHead:
        with np.load(path, allow_pickle=False) as z:

            def probe(name: str) -> OrdinalProbe:
                return OrdinalProbe(
                    weight=z[f"{name}.weight"].astype(np.float64),
                    bias=float(z[f"{name}.bias"]),
                    thresholds=z[f"{name}.thresholds"].astype(np.float64),
                )

            probes = {n: probe(n) for n in PROBE_NAMES}
            meta = probe("meta")
        metadata = {}
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                log.exception("could not parse grade head metadata at %s", meta_path)
        return cls(probes=probes, meta=meta, metadata=metadata)


def decode(cum_probs: np.ndarray) -> tuple[float, int, float, np.ndarray]:
    """Cumulative probs -> (expected grade, hard grade, confidence, per-grade probs).

    The hard grade is the median of the predicted distribution (1 + number of
    thresholds with P > 0.5); confidence is the probability of that grade.
    """
    p = np.asarray(cum_probs, np.float64)
    per_grade = np.clip(np.concatenate([[1.0], p]) - np.concatenate([p, [0.0]]), 0.0, None)
    per_grade /= max(per_grade.sum(), 1e-12)
    hard = int(1 + (p > 0.5).sum())
    return float(1.0 + p.sum()), hard, float(per_grade[hard - 1]), per_grade
