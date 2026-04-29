"""Heuristic front/back classifier for Pokemon TCG cards.

The English Pokemon TCG card back has been visually identical since 1999:
a deep-blue background dominating ~70% of the card surface, with a yellow
Pokeball-and-text accent in the centre. This is a *very* tight color
signature in HSV space and is essentially never produced by the front of a
Pokemon card.

So we score each candidate image with a single scalar `back_score`:

    back_score = w1 * fraction_of_deep_blue_pixels
               + w2 * fraction_of_pokeball_yellow_pixels

Higher score = more likely to be a back. To pair up front and back we just
pick `argmax(back_score)` as back, and another image as front.

This is intentionally classical/heuristic so it has zero training cost and
no extra runtime dependencies. The `pick_front_back` interface below lets
us swap in a learned classifier later without changing callers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


# Hue is in OpenCV's 0-180 range. Pokemon back blue is roughly H~100-130,
# heavily saturated. The Pokeball yellow is H~20-32.
_BLUE_H_LOW, _BLUE_H_HIGH = 100, 130
_BLUE_S_MIN = 80
_BLUE_V_MIN = 40
_YELLOW_H_LOW, _YELLOW_H_HIGH = 18, 35
_YELLOW_S_MIN = 110
_YELLOW_V_MIN = 120

# Empirically calibrated so a clean PSA back image scores ~0.7+ and any
# typical card front scores < 0.25.
_BLUE_WEIGHT = 1.4
_YELLOW_WEIGHT = 4.0

# If neither image clears this score, we conclude the listing has no clear
# back shot (e.g. two front photos at different angles) and skip it.
DEFAULT_MIN_BACK_SCORE = 0.30


@dataclass(slots=True)
class SideScore:
    index: int
    back_score: float
    blue_fraction: float
    yellow_fraction: float


def back_score_image(image_bgr: np.ndarray) -> SideScore:
    """Compute a continuous "is this a Pokemon card back?" score in [0, 1]."""
    if image_bgr is None or image_bgr.size == 0:
        return SideScore(index=-1, back_score=0.0, blue_fraction=0.0, yellow_fraction=0.0)

    # Downscale for speed - color statistics don't need full resolution.
    h, w = image_bgr.shape[:2]
    if max(h, w) > 512:
        scale = 512.0 / max(h, w)
        image_bgr = cv2.resize(
            image_bgr,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)

    blue_mask = (
        (h_chan >= _BLUE_H_LOW)
        & (h_chan <= _BLUE_H_HIGH)
        & (s_chan >= _BLUE_S_MIN)
        & (v_chan >= _BLUE_V_MIN)
    )
    yellow_mask = (
        (h_chan >= _YELLOW_H_LOW)
        & (h_chan <= _YELLOW_H_HIGH)
        & (s_chan >= _YELLOW_S_MIN)
        & (v_chan >= _YELLOW_V_MIN)
    )

    blue_fraction = float(blue_mask.mean())
    yellow_fraction = float(yellow_mask.mean())
    score = min(1.0, _BLUE_WEIGHT * blue_fraction + _YELLOW_WEIGHT * yellow_fraction)
    return SideScore(
        index=-1,
        back_score=score,
        blue_fraction=blue_fraction,
        yellow_fraction=yellow_fraction,
    )


def _load(path_or_bytes: Path | bytes) -> np.ndarray | None:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        arr = np.frombuffer(path_or_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.imread(str(path_or_bytes), cv2.IMREAD_COLOR)
    return img


def pick_front_back(
    candidates: list[Path] | list[bytes],
    *,
    min_back_score: float = DEFAULT_MIN_BACK_SCORE,
) -> tuple[int, int] | None:
    """Choose which candidate is the front and which is the back.

    Returns ``(front_index, back_index)`` into ``candidates`` or ``None``
    if no image looks confidently like a back.
    """
    if len(candidates) < 2:
        return None

    scores: list[SideScore] = []
    for i, c in enumerate(candidates):
        img = _load(c)
        if img is None:
            scores.append(SideScore(index=i, back_score=0.0, blue_fraction=0.0, yellow_fraction=0.0))
            continue
        s = back_score_image(img)
        scores.append(SideScore(index=i, back_score=s.back_score, blue_fraction=s.blue_fraction, yellow_fraction=s.yellow_fraction))

    best = max(scores, key=lambda s: s.back_score)
    if best.back_score < min_back_score:
        return None

    # Front = the lowest-scoring candidate (most "front-like"). If all
    # remaining candidates are tied at zero, fall back to the first non-back.
    others = [s for s in scores if s.index != best.index]
    front_pick = min(others, key=lambda s: s.back_score)
    return front_pick.index, best.index
