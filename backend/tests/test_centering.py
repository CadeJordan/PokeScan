"""Centering measurement (backend/app/ml/centering.py) on cards with known borders."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.ml.centering import (
    CenteringResult,
    _grade_from_ratio,
    compute_centering_subgrade,
    measure_centering,
)
from backend.app.ml.preprocess import detect_and_crop
from backend.tests.conftest import bgr_to_rgb
from backend.tests.helpers.overlay import draw_centering, save_overlay
from backend.tests.helpers.synth import make_card, make_scene, to_jpeg_bytes

# (left, right, top, bottom) border widths in px on the 600x840 card.
BORDERS = {
    "centered": (36, 36, 40, 40),
    "left_heavy": (48, 24, 40, 40),
    "right_heavy": (22, 50, 40, 40),
    "top_heavy": (36, 36, 60, 25),
    "corner_off": (50, 22, 30, 55),
}
TOL_PX = 3


@pytest.mark.parametrize("name", BORDERS)
@pytest.mark.parametrize("seed", range(2))
def test_measures_known_borders_on_crop(name, seed):
    borders = BORDERS[name]
    res = measure_centering(make_card(borders=borders, seed=seed))
    measured = (res.left, res.right, res.top, res.bottom)
    assert np.abs(np.subtract(measured, borders)).max() <= TOL_PX, measured


@pytest.mark.parametrize("name", ["centered", "left_heavy", "top_heavy"])
def test_measures_known_borders_through_detection(name):
    """End-to-end: synthetic photo -> detect/warp -> centering."""
    borders = BORDERS[name]
    scene = make_scene(seed=4, condition="perspective", card=make_card(borders=borders, seed=4))
    crop = detect_and_crop(bgr_to_rgb(scene.bgr))
    assert crop.found_quad
    res = measure_centering(crop.image)
    measured = (res.left, res.right, res.top, res.bottom)
    assert np.abs(np.subtract(measured, borders)).max() <= TOL_PX + 2, measured


@pytest.mark.xfail(
    reason="borders thinner than the 3% outer skip band (18px x / 25px y) cannot be measured",
    strict=False,
)
def test_very_thin_border():
    borders = (12, 60, 40, 40)
    res = measure_centering(make_card(borders=borders))
    assert abs(res.left - 12) <= TOL_PX


def test_ratios_and_grade_for_known_card():
    res = measure_centering(make_card(borders=BORDERS["left_heavy"]))
    lr = res.lr_ratio
    assert sum(lr) == 100
    assert 64 <= lr[0] <= 69  # 48/(48+24) = 66.7
    assert res.grade == 8.0   # 67/33 is within the 70/30 tolerance for an 8


@pytest.mark.parametrize(
    ("worst_pct", "grade"),
    [(50, 10.0), (60, 10.0), (61, 9.0), (65, 9.0), (70, 8.0), (75, 7.0),
     (80, 6.0), (85, 5.0), (90, 4.0), (95, 3.0), (96, 2.0), (100, 2.0)],
)
def test_grade_from_ratio_psa_boundaries(worst_pct, grade):
    assert _grade_from_ratio(worst_pct) == grade


def test_grade_from_ratio_monotone():
    grades = [_grade_from_ratio(p) for p in range(50, 101)]
    assert all(a >= b for a, b in zip(grades, grades[1:], strict=False))


def test_ratio_properties_handle_zero():
    r = CenteringResult(left=0, right=0, top=0, bottom=0, grade=10.0)
    assert sum(r.lr_ratio) == 100 and sum(r.tb_ratio) == 100


def test_rejects_wrong_canvas_size():
    with pytest.raises(ValueError):
        measure_centering(np.zeros((100, 100, 3), np.uint8))


def test_compute_centering_subgrade_formats(scene_jpeg):
    sub = compute_centering_subgrade(scene_jpeg)
    assert 1.0 <= sub.grade <= 10.0
    for s in (sub.left_right, sub.top_bottom):
        a, b = map(int, s.split("/"))
        assert a + b == 100


def test_centering_on_jpeg_upload_matches_known_borders():
    borders = BORDERS["right_heavy"]
    scene = make_scene(seed=5, card=make_card(borders=borders, seed=5))
    sub = compute_centering_subgrade(to_jpeg_bytes(scene.bgr))
    left_pct = int(sub.left_right.split("/")[0])
    assert abs(left_pct - round(22 / 72 * 100)) <= 3


@pytest.mark.visual
@pytest.mark.parametrize("name", BORDERS)
def test_overlay_centering(name):
    borders = BORDERS[name]
    scene = make_scene(seed=6, condition="perspective", card=make_card(borders=borders, seed=6))
    crop = detect_and_crop(bgr_to_rgb(scene.bgr))
    res = measure_centering(crop.image)
    img = draw_centering(
        crop.image,
        (res.left, res.right, res.top, res.bottom),
        expected=borders,
        title=f"{name}  LR {res.lr_ratio[0]}/{res.lr_ratio[1]}  grade {res.grade:g}",
    )
    assert save_overlay(img, f"centering_{name}").exists()
