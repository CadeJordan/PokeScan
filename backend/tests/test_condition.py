"""Corner / edge / surface analysis (backend/app/ml/condition.py).

The thresholds are calibrated on real PSA photos, so these tests check
direction rather than exact grades: a clean synthetic card must score
pristine, and adding a specific kind of damage must make that factor worse.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import backend.app.ml.condition as condition
from backend.app.ml.condition import (
    _SEVERITY_BUCKETS,
    FactorResult,
    _analyze_side,
    _grade_from_severity,
    _norm,
    compute_condition_subgrades,
)
from backend.app.ml.preprocess import detect_and_crop
from backend.tests.conftest import bgr_to_rgb
from backend.tests.helpers.synth import make_card, make_scene, to_jpeg_bytes

BG_RGB = (45, 40, 35)


# --------------------------------------------------------------------------- #
# Synthetic damage
# --------------------------------------------------------------------------- #
def whitened_rim(card: np.ndarray, t: int = 4) -> np.ndarray:
    c = card.copy()
    c[:t], c[-t:], c[:, :t], c[:, -t:] = 250, 250, 250, 250
    return c


def scratched(card: np.ndarray, n: int = 20) -> np.ndarray:
    c = card.copy()
    rng = np.random.default_rng(3)
    for _ in range(n):
        x0, y0 = int(rng.integers(60, 540)), int(rng.integers(60, 300))
        x1 = x0 + int(rng.integers(-150, 150))
        y1 = y0 + int(rng.integers(300, 500))
        cv2.line(c, (x0, y0), (x1, y1), (250, 250, 250), 2)
    return c


def rounded_corners(card: np.ndarray, r: int = 60) -> np.ndarray:
    c = card.copy()
    h, w = c.shape[:2]
    keep = np.zeros((h, w), np.uint8)
    cv2.rectangle(keep, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(keep, (0, r), (w, h - r), 255, -1)
    for x, y in [(r, r), (w - r, r), (w - r, h - r), (r, h - r)]:
        cv2.circle(keep, (x, y), r, 255, -1)
    c[keep == 0] = BG_RGB
    return c


def nicked_edges(card: np.ndarray) -> np.ndarray:
    c = card.copy()
    rng = np.random.default_rng(1)
    h, w = c.shape[:2]
    for _ in range(30):
        d = int(rng.integers(10, 22))
        x, y = int(rng.integers(80, w - 80)), int(rng.integers(80, h - 80))
        cv2.circle(c, (x, 0), d, BG_RGB, -1)
        cv2.circle(c, (w, y), d, BG_RGB, -1)
        cv2.circle(c, (0, y), d, BG_RGB, -1)
    return c


def _analyze(card: np.ndarray, seed: int = 0) -> dict[str, FactorResult]:
    scene = make_scene(seed=seed, condition="clean", card=card)
    crop = detect_and_crop(bgr_to_rgb(scene.bgr))
    assert crop.found_quad
    return _analyze_side(crop)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("severity", "grade"),
    [(-1.0, 10.0), (0.0, 10.0), (0.05, 10.0), (0.06, 9.0), (0.12, 9.0), (0.5, 6.0),
     (0.9, 3.0), (0.95, 2.0), (1.0, 2.0), (5.0, 2.0)],
)
def test_grade_from_severity(severity, grade):
    assert _grade_from_severity(severity) == grade


def test_grade_from_severity_monotone():
    grades = [_grade_from_severity(s) for s in np.linspace(0, 1, 201)]
    assert all(a >= b for a, b in zip(grades, grades[1:], strict=False))
    assert {g for _, g in _SEVERITY_BUCKETS} >= set(grades)


@pytest.mark.parametrize(
    ("value", "expected"), [(0.0, 0.0), (1.0, 0.0), (1.5, 0.5), (2.0, 1.0), (9.0, 1.0)]
)
def test_norm(value, expected):
    assert _norm(value, 1.0, 2.0) == pytest.approx(expected)


def test_norm_degenerate_range():
    assert _norm(5.0, 2.0, 2.0) == 0.0


def test_factor_hint():
    assert FactorResult(10.0, 0.0).hint == "clean"
    assert FactorResult(5.0, 0.6, ["rounded", "jagged"]).hint == "rounded, jagged"


# --------------------------------------------------------------------------- #
# Behaviour on synthetic cards
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(3))
def test_clean_card_is_pristine(seed):
    res = _analyze(make_card(seed=seed), seed)
    for name, f in res.items():
        assert f.grade == 10.0, f"{name}: {f}"
        assert f.flags == []


def _corner_rounding(card: np.ndarray) -> float:
    scene = make_scene(seed=0, condition="clean", card=card)
    crop = detect_and_crop(bgr_to_rgb(scene.bgr))
    pts = crop.contour.reshape(-1, 2).astype(np.float64)
    quad = crop.quad.astype(np.float64)
    mean_len = float(np.mean([np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]))
    return max(r for r, _ in condition._corner_geometry(pts, quad, mean_len))


def test_corner_rounding_grows_with_radius():
    # Floors are calibrated on real slab photos, where factory corner radius
    # plus noise already reaches ~0.045 for gem-mint cards, so synthetic
    # rounding is checked on the raw measurement rather than the grade.
    raw = [_corner_rounding(rounded_corners(make_card(), r=r)) for r in (0, 30, 60)]
    assert raw[0] < raw[1] < raw[2]


def test_rounding_past_floor_degrades_corners():
    worn = condition._ROUND_FLOOR + 0.6 * (condition._ROUND_FULL - condition._ROUND_FLOOR)
    res = condition._corner_factor([(0.0, 0.0), (worn, 0.0), (0.0, 0.0), (0.0, 0.0)])
    assert res.grade <= 6.0
    assert res.flags == ["rounded"]
    clean = condition._corner_factor([(condition._ROUND_FLOOR, condition._JAG_FLOOR)] * 4)
    assert clean.grade == 10.0


def test_scratches_degrade_surface_only():
    res = _analyze(scratched(make_card()))
    assert res["surface"].grade <= 7.0
    assert res["surface"].flags == ["scratches/creases"]
    assert res["corners"].grade == 10.0


def test_heavier_scratching_is_worse():
    light = _analyze(scratched(make_card(), n=20))["surface"].severity
    heavy = _analyze(scratched(make_card(), n=60))["surface"].severity
    assert heavy > light


@pytest.mark.parametrize("seed", range(2))
def test_rim_whitening_degrades_edges(seed):
    # 3 px: whitening is rim *contrast*, so the white must not reach the
    # band just inside the rim.
    res = _analyze(whitened_rim(make_card(seed=seed), t=3), seed)
    assert res["edges"].grade < 10.0


@pytest.mark.xfail(
    reason="small edge nicks are smoothed away by Canny dilation + CHAIN_APPROX_SIMPLE",
    strict=False,
)
def test_edge_nicks_degrade_edges():
    res = _analyze(nicked_edges(make_card()))
    assert res["edges"].grade < 10.0


def test_fallback_crop_skips_geometry():
    crop = detect_and_crop(np.full((900, 700, 3), 128, np.uint8))
    res = _analyze_side(crop)
    assert res["corners"].grade == 10.0  # no contour -> cannot assess, not "damaged"


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def test_worse_side_wins():
    clean = to_jpeg_bytes(make_scene(seed=0, card=make_card(seed=0)).bgr)
    damaged = to_jpeg_bytes(make_scene(seed=0, card=scratched(make_card(seed=0))).bgr)
    a = compute_condition_subgrades(clean, damaged)
    b = compute_condition_subgrades(damaged, clean)
    assert a.surface.grade == b.surface.grade <= 7.0
    assert a.corners.grade == 10.0


def test_condition_picks_min_per_factor(monkeypatch):
    sides = iter(
        [
            {"corners": FactorResult(9.0, 0.1), "edges": FactorResult(5.0, 0.6),
             "surface": FactorResult(10.0, 0.0)},
            {"corners": FactorResult(7.0, 0.3), "edges": FactorResult(8.0, 0.2),
             "surface": FactorResult(10.0, 0.0)},
        ]
    )
    monkeypatch.setattr(condition, "detect_and_crop", lambda _b: None)
    monkeypatch.setattr(condition, "_analyze_side", lambda _c: next(sides))
    res = compute_condition_subgrades(b"front", b"back")
    assert (res.corners.grade, res.edges.grade, res.surface.grade) == (7.0, 5.0, 10.0)


def test_accepts_detected_crops_without_detecting_again(monkeypatch):
    clean = to_jpeg_bytes(make_scene(seed=0, card=make_card(seed=0)).bgr)
    damaged = to_jpeg_bytes(make_scene(seed=0, card=scratched(make_card(seed=0))).bgr)
    expected = compute_condition_subgrades(clean, damaged)
    crops = detect_and_crop(clean), detect_and_crop(damaged)

    def boom(*_a, **_k):
        raise AssertionError("detect_and_crop must not run when given a CardCrop")

    monkeypatch.setattr(condition, "detect_and_crop", boom)
    assert compute_condition_subgrades(*crops) == expected
