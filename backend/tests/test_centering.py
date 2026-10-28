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


def test_very_thin_border():
    """An 80/20 card: the narrow side's outer band is mostly art, so the border
    colour has to come from the other sides."""
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
     (80, 6.0), (85, 5.0), (86, 3.0), (90, 3.0), (91, 1.0), (100, 1.0)],
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


@pytest.mark.parametrize(
    ("worst_pct", "grade"), [(55, 10.0), (75, 10.0), (76, 9.0), (90, 9.0), (91, 1.0)]
)
def test_back_tolerances(worst_pct, grade):
    assert _grade_from_ratio(worst_pct, "back") == grade


def test_abstains_on_featureless_card():
    assert measure_centering(np.full((840, 600, 3), 128, np.uint8)) is None


def test_abstains_when_border_matches_art():
    """A full-art style card: no border colour distinct from the art."""
    card = make_card(borders=(36, 36, 40, 40), border_rgb=(60, 90, 140))
    card[:] = card[420:421, 300:301]  # one flat colour everywhere
    assert measure_centering(card) is None


def test_back_face_and_worse_face_wins():
    front = make_scene(seed=7, card=make_card(borders=(36, 36, 40, 40), seed=7))
    # 52/20 back is 72/28: fine for a 10 on the back (<=75/25) ...
    back = make_scene(seed=8, card=make_card(borders=(52, 20, 40, 40), border_rgb=(40, 80, 200), seed=8))
    sub = compute_centering_subgrade(to_jpeg_bytes(front.bgr), to_jpeg_bytes(back.bgr))
    assert sub.grade == 10.0
    assert sub.back_left_right is not None
    assert abs(int(sub.back_left_right.split("/")[0]) - 72) <= 3
    # ... but the same offset on the front caps the grade at 8 (<=70/30 fails).
    sub2 = compute_centering_subgrade(to_jpeg_bytes(back.bgr), to_jpeg_bytes(front.bgr))
    assert sub2.grade == 7.0



# --------------------------------------------------------------------------- #
# Crops from the grade model: no second detection, no fallback canvases
# --------------------------------------------------------------------------- #
@pytest.fixture
def card_crop(scene_jpeg):
    crop = detect_and_crop(scene_jpeg)
    assert crop.found_quad
    return crop


@pytest.fixture
def no_detection(monkeypatch):
    import backend.app.ml.centering as centering_mod

    def boom(*_a, **_k):
        raise AssertionError("detect_and_crop must not run when given a CardCrop")

    monkeypatch.setattr(centering_mod, "detect_and_crop", boom)


def test_fallback_crop_is_not_measured(card_crop, no_detection):
    from backend.app.ml.preprocess import CardCrop

    # Even a fallback canvas that looks like a bordered card is skipped:
    # found_quad=False means it is the padded photo, not the card.
    fallback = CardCrop(image=make_card(seed=0), found_quad=False)
    assert measure_centering(fallback.image, "front") is not None
    sub = compute_centering_subgrade(fallback, card_crop)
    assert sub is not None
    assert sub.left_right is None and sub.top_bottom is None
    assert sub.back_left_right is not None and sub.back_top_bottom is not None
    assert compute_centering_subgrade(fallback, fallback) is None


def test_fallback_detection_from_bytes_is_not_measured(scene_jpeg):
    blank = to_jpeg_bytes(np.full((900, 700, 3), 128, np.uint8))
    sub = compute_centering_subgrade(blank, scene_jpeg)
    assert sub.left_right is None and sub.back_left_right is not None


def test_card_crop_gives_same_result_as_bytes(scene_jpeg, card_crop, monkeypatch):
    from_bytes = compute_centering_subgrade(scene_jpeg, scene_jpeg)
    import backend.app.ml.centering as centering_mod

    monkeypatch.setattr(centering_mod, "detect_and_crop", None)  # any call would fail
    assert compute_centering_subgrade(card_crop, card_crop) == from_bytes
    assert from_bytes.left_right is not None
