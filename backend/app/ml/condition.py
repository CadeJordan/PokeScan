"""Classical-CV condition analysis: corners, edges, surface sub-grades.

This mirrors the `centering` module's philosophy: interpretable geometry +
appearance measurements, no labels or model required. It produces a 1-10
sub-grade per factor (corners / edges / surface) plus human-readable flags
(`whitening`, `wear`, `rounded`, `jagged`, `scratches`, `creases`) for the UI.

What each factor measures:
- corners: whitening (a bright, desaturated fringe at the very rim), the
  cleanliness of the factory cut (sharp vs rounded), and fraying/jaggedness.
- edges: rim whitening plus edge wear (roughness of the physical boundary).
- surface: strong, long, straight linear defects isolated from the print/art -
  shorter ones read as scratches, very long ones as creases.

Whitening is measured as *rim contrast*: the white-pixel fraction in the
outermost few pixels minus the fraction just inside it. A clean colored border
has ~0 contrast; chipping/whitening makes the extreme rim noticeably whiter
than the ink just inside. This separates damage from cards that simply have a
light border (a plain white-fraction measure does not).

Whitening + surface run on the dewarped 600x840 crop. Corner-cut and edge-wear
*geometry* run on the original card contour, because the perspective warp forces
corners to perfect right angles and erases the physical corner shape.

The thresholds below were calibrated against PSA cert images (grade-10 vs
grades 4-7) so a clean card lands near 10 and real wear degrades it.

This is the classical-CV half of the "hybrid" plan: `compute_condition_subgrades`
is the single seam. Swapping in the trained `MultiTaskGradeModel` factor heads
later means only re-implementing that function.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import sqrt

import cv2
import numpy as np

from backend.app.ml.preprocess import CardCrop, detect_and_crop

log = logging.getLogger(__name__)

FACTOR_NAMES = ("corners", "edges", "surface")

# Severity -> grade buckets. Severity is in [0, 1] (0 pristine, 1 destroyed).
_SEVERITY_BUCKETS = (
    (0.05, 10.0),
    (0.12, 9.0),
    (0.22, 8.0),
    (0.35, 7.0),
    (0.50, 6.0),
    (0.65, 5.0),
    (0.78, 4.0),
    (0.90, 3.0),
    (1.00, 2.0),
)

# Whitening: a pixel is "white-ish" if bright and desaturated. We measure it as
# *rim contrast* (outer rim white fraction minus the band just inside) along the
# edges, skipping the literal corner tip (which is structurally white on every
# card - the case/background). Floors are set above the clean-card population so
# only genuinely anomalous whitening fires; whitening is the noisiest signal so
# it acts mainly as a flag rather than the dominant grade driver.
_WHITE_SAT_MAX = 60.0
_WHITE_VAL_FLOOR = 160.0
_RIM_PX = 3  # thickness of the extreme outer rim

# Normalisation: (floor = clean baseline, full = severity saturates to 1.0).
# Calibrated on PSA grade-10 vs grade 4-7 cert images (worst of front+back).
_WHITE_FLOOR = 0.55
_WHITE_FULL = 0.75
_ROUND_FLOOR = 0.017
_ROUND_FULL = 0.050
_JAG_FLOOR = 0.023
_JAG_FULL = 0.048
_WEAR_FLOOR = 0.024
_WEAR_FULL = 0.042
_SURF_FLOOR = 15.0
_SURF_FULL = 30.0


@dataclass(slots=True)
class FactorResult:
    grade: float
    severity: float
    flags: list[str] = field(default_factory=list)

    @property
    def hint(self) -> str:
        return ", ".join(self.flags) if self.flags else "clean"


@dataclass(slots=True)
class ConditionResult:
    corners: FactorResult
    edges: FactorResult
    surface: FactorResult


def _grade_from_severity(severity: float) -> float:
    severity = float(np.clip(severity, 0.0, 1.0))
    for threshold, grade in _SEVERITY_BUCKETS:
        if severity <= threshold:
            return grade
    return 1.0


def _norm(value: float, floor: float, full: float) -> float:
    if full <= floor:
        return 0.0
    return float(np.clip((value - floor) / (full - floor), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Whitening (rim contrast, on the warped crop)
# --------------------------------------------------------------------------- #
def _white_frac(hsv_region: np.ndarray) -> float:
    if hsv_region.size == 0:
        return 0.0
    s = hsv_region[..., 1].astype(np.float32)
    v = hsv_region[..., 2].astype(np.float32)
    return float(((s < _WHITE_SAT_MAX) & (v > _WHITE_VAL_FLOOR)).mean())


def _rim_contrast(outer: np.ndarray, inner: np.ndarray) -> float:
    return max(0.0, _white_frac(outer) - _white_frac(inner))


def _edge_whitening(hsv: np.ndarray, strip_t: int) -> list[float]:
    """Rim-contrast whitening per edge mid-segment: [top, right, bottom, left]."""
    h, w = hsv.shape[:2]
    cx, cy = int(round(0.12 * w)), int(round(0.12 * h))
    o = _RIM_PX
    top = _rim_contrast(hsv[0:o, cx : w - cx], hsv[o : o + strip_t, cx : w - cx])
    bottom = _rim_contrast(
        hsv[h - o : h, cx : w - cx], hsv[h - o - strip_t : h - o, cx : w - cx]
    )
    left = _rim_contrast(hsv[cy : h - cy, 0:o], hsv[cy : h - cy, o : o + strip_t])
    right = _rim_contrast(
        hsv[cy : h - cy, w - o : w], hsv[cy : h - cy, w - o - strip_t : w - o]
    )
    return [top, right, bottom, left]


# Corner whitening is intentionally not used to drive the corner grade: at
# slab-photo resolution the corner region is structurally white at the rim on
# every card (clean and worn measure the same), so it cannot be separated from
# damage. Corner condition is driven by geometry (rounding / jaggedness), which
# does separate and correlates with corner wear. The ML factor heads can revisit
# whitening later (see the swap seam below).


# --------------------------------------------------------------------------- #
# Geometry (corner cut + edge wear, on the original contour)
# --------------------------------------------------------------------------- #
def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if points.shape[0] < 5:
        return None
    vx, vy, x0, y0 = cv2.fitLine(
        np.asarray(points, dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01
    ).ravel()
    return np.array([x0, y0], dtype=np.float64), np.array([vx, vy], dtype=np.float64)


def _side_points(
    pts: np.ndarray, a: np.ndarray, b: np.ndarray, t_lo: float, t_hi: float, perp_frac: float
) -> np.ndarray:
    ab = b - a
    seg_len = float(np.linalg.norm(ab))
    if seg_len <= 1e-6:
        return np.empty((0, 2))
    direction = ab / seg_len
    rel = pts - a
    t = (rel @ direction) / seg_len
    proj = np.outer(t * seg_len, direction)
    perp = np.linalg.norm(rel - proj, axis=1)
    sel = (t >= t_lo) & (t <= t_hi) & (perp < perp_frac * seg_len)
    return pts[sel]


def _line_intersection(
    l1: tuple[np.ndarray, np.ndarray], l2: tuple[np.ndarray, np.ndarray]
) -> np.ndarray | None:
    (p1, d1), (p2, d2) = l1, l2
    a = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], dtype=np.float64)
    det = np.linalg.det(a)
    if abs(det) < 1e-6:
        return None
    t = np.linalg.solve(a, p2 - p1)
    return p1 + t[0] * d1


def _point_line_distance(points: np.ndarray, line: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    p0, d = line
    d = d / (np.linalg.norm(d) + 1e-9)
    normal = np.array([-d[1], d[0]])
    return np.abs((points - p0) @ normal)


def _corner_geometry(pts: np.ndarray, quad: np.ndarray, mean_len: float) -> list[tuple[float, float]]:
    """Per corner (tl, tr, br, bl): (rounding_norm, jaggedness_norm)."""
    results: list[tuple[float, float]] = []
    for i in range(4):
        v = quad[i]
        prev_v = quad[(i - 1) % 4]
        next_v = quad[(i + 1) % 4]
        line1 = _fit_line(_side_points(pts, v, prev_v, 0.15, 0.55, 0.05))
        line2 = _fit_line(_side_points(pts, v, next_v, 0.15, 0.55, 0.05))
        if line1 is None or line2 is None:
            results.append((0.0, 0.0))
            continue
        apex = _line_intersection(line1, line2)
        if apex is None:
            apex = v
        short_side = min(
            float(np.linalg.norm(v - prev_v)), float(np.linalg.norm(v - next_v))
        )
        radius = max(8.0, 0.14 * short_side)
        near = pts[np.linalg.norm(pts - v, axis=1) < radius]
        if near.shape[0] < 5:
            results.append((0.0, 0.0))
            continue
        rounding = float(np.min(np.linalg.norm(near - apex, axis=1))) / mean_len
        nearest_line = np.minimum(
            _point_line_distance(near, line1), _point_line_distance(near, line2)
        )
        jaggedness = float(sqrt(np.mean(nearest_line ** 2))) / mean_len
        results.append((rounding, jaggedness))
    return results


def _edge_roughness(pts: np.ndarray, quad: np.ndarray) -> list[float | None]:
    """Per edge (top, right, bottom, left): RMS boundary roughness vs edge length."""
    rough: list[float | None] = []
    for i in range(4):
        a = quad[i]
        b = quad[(i + 1) % 4]
        seg_len = float(np.linalg.norm(b - a))
        if seg_len <= 1e-6:
            rough.append(None)
            continue
        direction = (b - a) / seg_len
        rel = pts - a
        t = (rel @ direction) / seg_len
        proj = np.outer(t * seg_len, direction)
        perp = np.linalg.norm(rel - proj, axis=1)
        sel = (t >= 0.12) & (t <= 0.88) & (perp < 0.06 * seg_len)
        chosen = perp[sel]
        if chosen.size < 10:
            rough.append(None)
            continue
        rough.append(float(sqrt(np.mean(chosen ** 2))) / seg_len)
    return rough


# --------------------------------------------------------------------------- #
# Surface (scratches + creases, on the warped crop interior)
# --------------------------------------------------------------------------- #
def _surface_defects(rgb: np.ndarray) -> float:
    """Return a strong-line length density for the card interior.

    Isolates strong, long, straight lines from the print/art by high-pass
    filtering (image minus a small Gaussian blur), keeping only strong
    residuals, then requiring long Hough segments. Ordinary artwork is
    lower-contrast at this scale and is suppressed; scratches and creases - both
    long, high-contrast linear features - raise the density. (We don't separate
    scratch vs crease: at slab-photo resolution a long straight defect is
    reported generically; a card-design frame line of the same length is the
    main confounder, so the threshold is set above the clean-card population.)
    """
    h, w = rgb.shape[:2]
    iy, ix = int(round(0.08 * h)), int(round(0.08 * w))
    region = rgb[iy : h - iy, ix : w - ix]
    if region.size == 0:
        return 0.0
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), 3)
    resid = cv2.absdiff(gray, background)
    strong = ((resid > 28).astype(np.uint8)) * 255
    rh, rw = gray.shape
    min_len = max(12, int(round(0.25 * min(rh, rw))))
    lines = cv2.HoughLinesP(
        strong, 1, np.pi / 180, threshold=60, minLineLength=min_len, maxLineGap=4
    )
    total_len = 0.0
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            total_len += float(np.hypot(float(x2 - x1), float(y2 - y1)))
    return total_len / (rh + rw)


# --------------------------------------------------------------------------- #
# Per-side aggregation
# --------------------------------------------------------------------------- #
def _analyze_side(crop: CardCrop) -> dict[str, FactorResult]:
    rgb = crop.image
    h, w = rgb.shape[:2]
    strip_t = max(6, int(round(0.02 * min(h, w))))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    edge_white = _edge_whitening(hsv, strip_t)       # [top, right, bottom, left]

    corner_geom: list[tuple[float, float]] | None = None
    edge_rough: list[float | None] | None = None
    if crop.contour is not None and crop.quad is not None:
        try:
            pts = crop.contour.reshape(-1, 2).astype(np.float64)
            quad = crop.quad.astype(np.float64)
            mean_len = float(
                np.mean([np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)])
            )
            if mean_len > 1e-6:
                corner_geom = _corner_geometry(pts, quad, mean_len)
                edge_rough = _edge_roughness(pts, quad)
        except Exception as exc:  # noqa: BLE001
            log.debug("corner/edge geometry failed: %s", exc)

    return {
        "corners": _corner_factor(corner_geom),
        "edges": _edge_factor(edge_white, edge_rough),
        "surface": _surface_factor(rgb),
    }


def _corner_factor(geom: list[tuple[float, float]] | None) -> FactorResult:
    if geom is None:
        # No card contour (e.g. fallback crop) - cannot assess corner geometry.
        return FactorResult(grade=_grade_from_severity(0.0), severity=0.0, flags=[])
    worst_sev = 0.0
    worst_flags: list[str] = []
    for rounding, jaggedness in geom:
        sev_round = _norm(rounding, _ROUND_FLOOR, _ROUND_FULL)
        sev_jag = _norm(jaggedness, _JAG_FLOOR, _JAG_FULL)
        sev = max(sev_round, sev_jag)
        if sev > worst_sev:
            worst_sev = sev
            flags: list[str] = []
            if sev_round > 0.30:
                flags.append("rounded")
            if sev_jag > 0.30:
                flags.append("jagged")
            worst_flags = flags
    return FactorResult(
        grade=_grade_from_severity(worst_sev), severity=worst_sev, flags=worst_flags
    )


def _edge_factor(edge_white: list[float], rough: list[float | None] | None) -> FactorResult:
    worst_sev = 0.0
    worst_flags: list[str] = []
    for i in range(4):
        sev_white = _norm(edge_white[i], _WHITE_FLOOR, _WHITE_FULL)
        sev_wear = 0.0
        if rough is not None and rough[i] is not None:
            sev_wear = _norm(rough[i], _WEAR_FLOOR, _WEAR_FULL)
        sev = max(sev_white, sev_wear)
        if sev > worst_sev:
            worst_sev = sev
            flags: list[str] = []
            if sev_white > 0.30:
                flags.append("whitening")
            if sev_wear > 0.30:
                flags.append("wear")
            worst_flags = flags
    return FactorResult(
        grade=_grade_from_severity(worst_sev), severity=worst_sev, flags=worst_flags
    )


def _surface_factor(rgb: np.ndarray) -> FactorResult:
    density = _surface_defects(rgb)
    sev = _norm(density, _SURF_FLOOR, _SURF_FULL)
    flags: list[str] = []
    if sev > 0.30:
        flags.append("scratches/creases")
    return FactorResult(grade=_grade_from_severity(sev), severity=sev, flags=flags)


# --------------------------------------------------------------------------- #
# Public entrypoint (the swap seam)
# --------------------------------------------------------------------------- #
def compute_condition_subgrades(front_bytes: bytes, back_bytes: bytes) -> ConditionResult:
    """Analyze both sides and keep the worse grade per factor (PSA-style)."""
    front = _analyze_side(detect_and_crop(front_bytes))
    back = _analyze_side(detect_and_crop(back_bytes))
    merged: dict[str, FactorResult] = {}
    for name in FACTOR_NAMES:
        f = front[name]
        b = back[name]
        merged[name] = f if f.grade <= b.grade else b
    return ConditionResult(
        corners=merged["corners"], edges=merged["edges"], surface=merged["surface"]
    )
