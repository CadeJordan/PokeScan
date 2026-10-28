"""Classical-CV centering measurement.

Centering is the ratio of opposite border widths. On the dewarped
CARD_W x CARD_H crop, for each side:

1. Take colour profiles running inward from the side (Lab, lines spanning the
   middle 76% of the side so rounded corners don't interfere).
2. The border colour is the dominant colour of the outer bands of all four
   sides pooled (mode-seeking mean shift), refined per side for lighting; each
   pixel's distance to it gives a profile per line.
3. Per chunk of lines (three per side), take a low percentile of that distance
   over lines at each depth. Text printed on the border (the copyright line on
   modern cards) or art reaching into it only affects a minority of lines, so
   the percentile profile stays border-like across the whole border.
4. The border is the run of border-like depths starting near the crop edge
   (slab photos often include a few px of gasket outside the card). Its inner
   and outer edges are the half-contrast crossings into what is next to it.
   A gasket close to the border colour can join the run, or capture the
   per-side colour when it is wider than a narrow border; both are detected
   and split off (see `_border_edges`, `_side_width`).
5. A side is measured when at least two chunks give widths that agree; a face
   when all four sides are.

Faces that can't be measured confidently (full-art or silver-bordered cards
whose border blends into the art, heavy glare) return None instead of a
guess. Checked on 2,434 PSA slab photos (a real PSA 10 is within 60/40 front
and 75/25 back): the previous gradient-projection method put 44% of PSA 10
fronts outside 60/40 and 17% outside 70/30. This one measures 81% of backs
(0 of 1,234 PSA 10s outside 75/25) but only ~17% of fronts - it abstains on
most full-art, silver-bordered and Japanese fronts - and of the PSA 10 fronts
it does measure, 6% fall outside 60/40 and 4% outside 70/30.

PSA tolerances: front 60/40 for a 10 down to 90/10 for a 3; back 75/25 for a
10 and 90/10 for 9 and below; past 90/10 on either face, a 1. The worse face
decides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from backend.app.ml.preprocess import CARD_H, CARD_W, CardCrop, detect_and_crop

if TYPE_CHECKING:
    from backend.app.api.grade import CenteringSubGrade

SIDES = ("left", "right", "top", "bottom")
_DEPTH = 0.16             # search depth from each side, x card width
_SPAN = (0.12, 0.88)      # part of each side used (skips the rounded corners)
_BAND = 0.07              # outer band that defines the border colour, x card width
_MAX_START = 18           # px: the border must start this close to the crop edge
_TOL = 12.0               # Lab distance for a pixel to count as border colour
_PERCENTILE = 35
_CHUNKS = 3
_AGREE_PX, _AGREE_FRAC = 3.0, 0.15  # chunk widths agree within max(px, frac x width)
_MIN_CONTRAST = 8.0       # border vs. inside, in Lab distance
_STEP = 6.0               # Lab jump that splits a gasket off a border-like run
_MIN_GASKET = 3           # px


@dataclass(slots=True)
class CenteringResult:
    left: float
    right: float
    top: float
    bottom: float
    grade: float

    @property
    def lr_ratio(self) -> tuple[int, int]:
        total = max(1e-6, self.left + self.right)
        left = round(self.left / total * 100)
        return left, 100 - left

    @property
    def tb_ratio(self) -> tuple[int, int]:
        total = max(1e-6, self.top + self.bottom)
        t = round(self.top / total * 100)
        return t, 100 - t

    @property
    def worst_pct(self) -> int:
        return max(max(self.lr_ratio), max(self.tb_ratio))


# PSA tolerances per the public grading guide: the highest grade each
# off-centeredness allows, e.g. 60 means up to 60/40. Grades sharing a
# tolerance with the grade above are left out (4 is 85/15 like 5, 2 is 90/10
# like 3); past 90/10 only a 1 is possible.
_PSA_TOLERANCES = [
    (10, 60),
    (9, 65),
    (8, 70),
    (7, 75),
    (6, 80),
    (5, 85),
    (3, 90),
]
_PSA_BACK_TOLERANCES = [(10, 75), (9, 90)]


def _grade_from_ratio(worst_side_pct: int, face: str = "front") -> float:
    """Map the worse of L/R or T/B (the higher of the two sides) to a grade."""
    table = _PSA_TOLERANCES if face == "front" else _PSA_BACK_TOLERANCES
    for grade, threshold in table:
        if worst_side_pct <= threshold:
            return float(grade)
    return 1.0


def _profiles(lab: np.ndarray, side: str) -> np.ndarray:
    """(lines, depth, 3) Lab profiles running inward from `side`."""
    h, w = lab.shape[:2]
    d = int(_DEPTH * w)
    if side == "left":
        p = lab[int(_SPAN[0] * h) : int(_SPAN[1] * h), :d]
    elif side == "right":
        p = lab[int(_SPAN[0] * h) : int(_SPAN[1] * h), w - d :][:, ::-1]
    elif side == "top":
        p = lab[:d, int(_SPAN[0] * w) : int(_SPAN[1] * w)].transpose(1, 0, 2)
    else:
        p = lab[h - d :, int(_SPAN[0] * w) : int(_SPAN[1] * w)][::-1].transpose(1, 0, 2)
    return np.ascontiguousarray(p)


def _dominant(
    pixels: np.ndarray, start: np.ndarray | None = None, radii=(40.0, 25.0, 15.0, 10.0)
) -> np.ndarray:
    """Mode of a colour cloud: mean shift from `start` (default: the median)."""
    c = np.median(pixels, axis=0) if start is None else start
    for radius in radii:
        for _ in range(8):
            near = np.linalg.norm(pixels - c, axis=1) < radius
            if near.sum() < 20:
                break
            c = pixels[near].mean(0)
    return c


def _crossing(dist: np.ndarray, i: int, target: float, step: int) -> float:
    """Sub-pixel position where `dist` rises above `target` walking from i by `step`."""
    j = i
    while 0 <= j + step < len(dist) and dist[j + step] < target:
        j += step
    k = j + step
    if not 0 <= k < len(dist) or dist[k] == dist[j]:
        return j + 0.5 * step
    return j + step * float(np.clip((target - dist[j]) / (dist[k] - dist[j]), 0, 1))


def _step(colours: np.ndarray, start: int, end: int) -> int | None:
    """Gasket step: the sharpest colour step in the outer _MAX_START px of a run
    [start, end] that starts at the crop edge, as the index of the first pixel
    past it, or None. Each side must be steady colour (the outer part at least
    _MIN_GASKET px, so a card's anti-aliased edge pixel doesn't count), with up
    to two px of blend between them (anti-aliasing, JPEG, slight blur).
    """
    if start > 2:
        return None
    best, best_k = 0.0, None
    for k in range(start + _MIN_GASKET + 1, min(end - 1, start + _MAX_START)):
        jump = float(np.linalg.norm(colours[k + 1] - colours[k - 2]))
        if jump > best:
            best, best_k = jump, k
    if best_k is None or best < _STEP:
        return None
    # The run's last px blend into what is inside the border; leave them out.
    outer, inner = colours[start : best_k - 1], colours[best_k + 1 : max(best_k + 3, end - 1)]
    contrast = float(np.linalg.norm(np.median(outer, 0) - np.median(inner, 0)))
    spread = max(
        float(np.linalg.norm(seg - np.median(seg, 0), axis=1).max()) for seg in (outer, inner)
    )
    return best_k if contrast >= _STEP and spread < 0.5 * contrast else None


def _step_crossing(colours: np.ndarray, k: int, start: int, end: int) -> float:
    """Sub-pixel position of the step at k: walking outward from its inner side,
    where the colour is halfway to the outer side's."""
    co = np.median(colours[start : k - 1], 0)
    ci = np.median(colours[k + 1 : max(k + 3, end - 1)], 0)
    t = (colours - co) @ (ci - co) / float((ci - co) @ (ci - co))
    return _crossing(1.0 - t, k + 1, 0.5, -1) + 0.5


def _border_edges(
    dist: np.ndarray, colours: np.ndarray, ref: np.ndarray | None
) -> tuple[float, float, bool] | None:
    """(outer, inner, gasket split off) from a distance-to-border-colour profile.

    `colours` is the matching Lab profile and `ref` the border colour pooled
    over all four sides (None: don't look for a gasket). A gasket or slab edge
    outside the card can be close enough to the border colour to join its run;
    if the run's outer part is a separate steady colour further from `ref`
    than the rest, the border starts at the step between them.
    """
    near = np.flatnonzero(dist[:_MAX_START] < _TOL)
    if not len(near):
        return None
    start = end = int(near[0])
    # Extend through the border, bridging single-pixel dropouts.
    while end + 1 < len(dist) and (
        dist[end + 1] < _TOL or (end + 2 < len(dist) and dist[end + 2] < _TOL)
    ):
        end += 1
    if end - start < 3 or end + 12 >= len(dist):
        return None
    outer = None
    k = None if ref is None else _step(colours, start, end)
    if k is not None:
        outer_c = np.median(colours[start : k - 1], 0)
        inner_c = np.median(colours[k + 1 : max(k + 3, end - 1)], 0)
        if np.linalg.norm(inner_c - ref) < np.linalg.norm(outer_c - ref):
            # The outer part lies outside the card: the border starts at the step.
            outer = _step_crossing(colours, k, start, end)
            start = k + 1
    split = outer is not None
    if end - start < 2:
        return None
    plateau = float(np.median(dist[start : end + 1]))
    inside = float(np.median(dist[end + 4 : end + 12]))
    if inside - plateau < _MIN_CONTRAST:
        return None
    inner = _crossing(dist, end, 0.5 * (plateau + inside), +1) + 0.5
    if outer is None:
        outer = 0.0
        if start > 0:
            outside = float(np.median(dist[max(0, start - 4) : start]))
            if outside > plateau:
                outer = _crossing(dist, start, 0.5 * (plateau + outside), -1) + 0.5
    return outer, inner, split


def _chunk_edges(profile: np.ndarray, colour: np.ndarray, ref: np.ndarray) -> list:
    """(outer, inner) border edges for each chunk of lines where they're clear."""
    dist = np.linalg.norm(profile - colour, axis=2)
    chunks = [
        (np.percentile(dist[c], _PERCENTILE, axis=0), np.median(profile[c], axis=0))
        for c in np.array_split(np.arange(len(dist)), _CHUNKS)
    ]
    edges = [_border_edges(d, c, ref) for d, c in chunks]
    if not all(e is not None and e[2] for e in edges):
        # A gasket runs the whole side: a step in only some chunks is lighting
        # or wear on the card's own edge, so measure those without splitting.
        edges = [_border_edges(d, c, None) for d, c in chunks]
    return [e[:2] for e in edges if e is not None]


def _side_width(profile: np.ndarray, band: np.ndarray, pooled: np.ndarray) -> float | None:
    # Shift the pooled border colour to this side's lighting.
    local = _dominant(band, start=pooled, radii=(20.0, 12.0))
    edges = _chunk_edges(profile, local, pooled)
    if len(edges) >= 2 and np.linalg.norm(local - pooled) > _TOL:
        # A large shift may instead have locked onto a gasket wider than a
        # narrow border. Then a pooled-colour border sits at the inner end of
        # the local run, starting well after it.
        alt = _chunk_edges(profile, pooled, pooled)
        if len(alt) >= 2:
            (lo, li), (ao, ai) = np.median(edges, 0), np.median(alt, 0)
            if ao >= lo + 2 and ai <= li + 2:
                edges = alt
    if len(edges) < 2:
        return None
    # At least two chunks must agree: one that disagrees has had art or text
    # join (or break) its border run. With three chunks the median is in the
    # closest pair.
    widths = np.sort([e[1] - e[0] for e in edges])
    gaps = np.diff(widths)
    i = int(np.argmin(gaps))
    if gaps[i] > max(_AGREE_PX, _AGREE_FRAC * widths[i + 1]):
        return None
    return float(np.median(widths))


def measure_borders(rgb: np.ndarray) -> tuple[float, float, float, float] | None:
    """(left, right, top, bottom) border widths in px, or None if any side is unclear."""
    if rgb.shape[:2] != (CARD_H, CARD_W):
        raise ValueError(f"expected {CARD_H}x{CARD_W} canvas, got {rgb.shape[:2]}")
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    profiles = [_profiles(lab, side) for side in SIDES]
    bands = [p[:, : int(_BAND * CARD_W)].reshape(-1, 3) for p in profiles]
    # The border is one ink on all four sides, so pool them: on an off-centre
    # card the narrow side's band is mostly art, but the wide sides carry it.
    # A short local mean shift per side then absorbs per-side lighting.
    pooled = _dominant(np.concatenate(bands))
    widths = [_side_width(p, band, pooled) for p, band in zip(profiles, bands, strict=True)]
    if any(w is None for w in widths):
        return None
    left, right, top, bottom = widths
    return left, right, top, bottom


def measure_centering(rgb: np.ndarray, face: str = "front") -> CenteringResult | None:
    borders = measure_borders(rgb)
    if borders is None:
        return None
    left, right, top, bottom = (round(b, 1) for b in borders)
    res = CenteringResult(left=left, right=right, top=top, bottom=bottom, grade=0.0)
    res.grade = _grade_from_ratio(res.worst_pct, face)
    return res


def _ratio(pair: tuple[int, int]) -> str:
    return f"{pair[0]}/{pair[1]}"


def _measure_face(image: bytes | CardCrop, face: str) -> CenteringResult | None:
    """Measure one face, detecting only when given bytes.

    A fallback crop (no card outline found) is not measured: its "borders"
    are the padded whole photo, not the card's.
    """
    crop = image if isinstance(image, CardCrop) else detect_and_crop(image)
    if not crop.found_quad:
        return None
    return measure_centering(crop.image, face)


def compute_centering_subgrade(
    front: bytes | CardCrop, back: bytes | CardCrop | None = None
) -> CenteringSubGrade | None:
    """Public entrypoint used by the API: crop, measure each face, worse face wins.

    Each face is upload bytes or the `CardCrop` the grade model already used.
    Returns None when neither face could be measured.
    """
    # Imported here so this ML module doesn't load the API layer at import time.
    from backend.app.api.grade import CenteringSubGrade

    front_res = _measure_face(front, "front")
    back_res = _measure_face(back, "back") if back is not None else None
    measured = [r.grade for r in (front_res, back_res) if r is not None]
    if not measured:
        return None
    return CenteringSubGrade(
        grade=min(measured),
        left_right=_ratio(front_res.lr_ratio) if front_res else None,
        top_bottom=_ratio(front_res.tb_ratio) if front_res else None,
        back_left_right=_ratio(back_res.lr_ratio) if back_res else None,
        back_top_bottom=_ratio(back_res.tb_ratio) if back_res else None,
    )
