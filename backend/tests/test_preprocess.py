"""Card detection + dewarp (backend/app/ml/preprocess.py) on synthetic scenes.

Ground truth comes from `helpers.synth`: every scene is rendered with a known
homography, so the true card corners are exact. Run `pytest -m visual` to also
write outline overlays to test-artifacts/overlays/.
"""

from __future__ import annotations

import io
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from backend.app.core.config import get_settings
from backend.app.ml.preprocess import (
    CARD_H,
    CARD_W,
    _order_quad,
    detect_and_crop,
    encode_jpeg,
)
from backend.tests.conftest import bgr_to_rgb
from backend.tests.helpers.overlay import draw_detection, save_overlay
from backend.tests.helpers.synth import (
    CONDITIONS,
    corner_error,
    make_scene,
    polygon_iou,
    quad_diagonal,
    to_jpeg_bytes,
)

# Conditions the classical detector is expected to handle.
SUPPORTED = (
    "clean", "rotated", "perspective", "gradient_bg", "noisy_bg",
    "blur", "noise", "bright", "jpeg",
    # Formerly known failures, fixed by the multi-threshold/chroma edge maps,
    # side snapping and slab-holder handling in _find_card_quad.
    "dark", "occluded", "slab", "cluttered_bg", "low_contrast",
)
# Known weaknesses, kept as non-strict xfails so improvements show up as XPASS.
KNOWN_HARD = {
    "cut_off": "card partly outside the frame has no closed outline",
}
assert set(SUPPORTED) | set(KNOWN_HARD) == set(CONDITIONS)

MAX_CORNER_ERR = 0.02  # fraction of card diagonal
MIN_IOU = 0.95


def _detect(scene):
    return detect_and_crop(bgr_to_rgb(scene.bgr))


def _assert_matches(crop, scene):
    assert crop.found_quad, f"{scene.condition}/seed={scene.seed}: no quad found"
    err = corner_error(crop.quad, scene.quad) / quad_diagonal(scene.quad)
    iou = polygon_iou(crop.quad, scene.quad, scene.bgr.shape[:2])
    assert err <= MAX_CORNER_ERR, f"corner error {err:.3%} of diagonal"
    assert iou >= MIN_IOU, f"IoU {iou:.3f}"


# --------------------------------------------------------------------------- #
# Unit: corner ordering
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(10))
def test_order_quad_returns_tl_tr_br_bl(seed):
    rng = np.random.default_rng(seed)
    expected = np.array([[100, 50], [400, 60], [410, 500], [90, 490]], np.float32)
    expected += rng.uniform(-15, 15, expected.shape).astype(np.float32)
    shuffled = expected[rng.permutation(4)]
    np.testing.assert_allclose(_order_quad(shuffled), expected)


def test_order_quad_accepts_contour_shape():
    pts = np.array([[[10, 10]], [[10, 90]], [[60, 90]], [[60, 10]]], np.float32)
    out = _order_quad(pts)
    assert out.shape == (4, 2)
    np.testing.assert_allclose(out[0], [10, 10])
    np.testing.assert_allclose(out[2], [60, 90])


# --------------------------------------------------------------------------- #
# Input handling / output contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["ndarray", "bytes", "pil"])
def test_detect_and_crop_accepts_input_types(kind):
    scene = make_scene(seed=1)
    rgb = bgr_to_rgb(scene.bgr)
    image = {
        "ndarray": rgb,
        "bytes": to_jpeg_bytes(scene.bgr, quality=95),
        "pil": Image.fromarray(rgb),
    }[kind]
    crop = detect_and_crop(image)
    assert crop.image.shape == (CARD_H, CARD_W, 3)
    assert crop.image.dtype == np.uint8
    assert crop.found_quad
    assert crop.quad.shape == (4, 2)
    assert crop.source_bgr.shape == scene.bgr.shape


def test_png_bytes_with_alpha_are_accepted():
    rgba = np.dstack([bgr_to_rgb(make_scene(seed=2).bgr), np.full((1200, 1000), 255, np.uint8)])
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    assert detect_and_crop(buf.getvalue()).found_quad


def test_rejects_non_rgb_array():
    with pytest.raises(ValueError):
        detect_and_crop(np.zeros((100, 100), np.uint8))


def test_encode_jpeg_roundtrip(card_rgb):
    data = encode_jpeg(card_rgb)
    assert data[:2] == b"\xff\xd8"
    decoded = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    assert decoded.shape == card_rgb.shape
    assert np.abs(decoded.astype(int) - card_rgb.astype(int)).mean() < 4


# --------------------------------------------------------------------------- #
# Accuracy against ground truth
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("condition", SUPPORTED)
def test_detects_card_outline(condition, seed):
    scene = make_scene(seed=seed, condition=condition)
    _assert_matches(_detect(scene), scene)


@pytest.mark.parametrize(
    "condition",
    [pytest.param(c, marks=pytest.mark.xfail(reason=r, strict=False)) for c, r in KNOWN_HARD.items()],
)
def test_known_hard_conditions(condition):
    scene = make_scene(seed=0, condition=condition)
    _assert_matches(_detect(scene), scene)


@pytest.mark.slow
def test_detection_rate_sweep(capsys):
    """Random sweep across all conditions; prints a per-condition report."""
    stats: dict[str, list[float | None]] = defaultdict(list)
    for seed in range(200):
        cond = CONDITIONS[seed % len(CONDITIONS)]
        scene = make_scene(seed=1000 + seed, condition=cond)
        crop = _detect(scene)
        ok = (
            crop.found_quad
            and corner_error(crop.quad, scene.quad) / quad_diagonal(scene.quad) <= MAX_CORNER_ERR
        )
        stats[cond].append(1.0 if ok else 0.0)

    with capsys.disabled():
        print("\n  condition        accurate-detection rate")
        for cond in CONDITIONS:
            rate = float(np.mean(stats[cond]))
            print(f"  {cond:15s}  {rate:6.1%}  (n={len(stats[cond])})")

    supported_rate = np.mean([v for c in SUPPORTED for v in stats[c]])
    assert supported_rate >= 0.97


# Detection runs on every API request (front + back). ~0.13 s/image on a
# 1024x600 slab photo on a dev laptop; the budget is loose so only a real
# regression (like the 2x slowdown numpy channel max/min once caused) fails.
DETECT_BUDGET_S = 0.30


@pytest.mark.slow
def test_detection_speed():
    import time

    scenes = [make_scene(seed=s, condition="slab", size=(1024, 640)) for s in range(8)]
    rgbs = [bgr_to_rgb(s.bgr) for s in scenes]
    detect_and_crop(rgbs[0])  # warm-up
    t0 = time.perf_counter()
    for rgb in rgbs:
        detect_and_crop(rgb)
    per_image = (time.perf_counter() - t0) / len(rgbs)
    assert per_image <= DETECT_BUDGET_S, f"{per_image:.3f}s per image"


# --------------------------------------------------------------------------- #
# Negatives / fallback
# --------------------------------------------------------------------------- #
def _square_scene() -> np.ndarray:
    img = np.full((1000, 1000, 3), 40, np.uint8)
    cv2.rectangle(img, (200, 200), (800, 800), (230, 200, 60), -1)
    return img


@pytest.mark.parametrize(
    "image",
    [
        pytest.param(np.full((900, 700, 3), 128, np.uint8), id="blank"),
        pytest.param(
            np.random.default_rng(0).integers(0, 255, (900, 700, 3), dtype=np.uint8),
            id="noise",
        ),
        pytest.param(_square_scene(), id="square"),
    ],
)
def test_no_card_falls_back_to_padded_resize(image):
    crop = detect_and_crop(image)
    assert not crop.found_quad
    assert crop.quad is None and crop.contour is None
    assert crop.image.shape == (CARD_H, CARD_W, 3)


def test_fallback_preserves_aspect_with_black_padding():
    # 1000x1000 grey square -> scaled to 600x600, padded top/bottom with black.
    crop = detect_and_crop(np.full((1000, 1000, 3), 128, np.uint8))
    assert crop.image[:100].max() == 0 and crop.image[-100:].max() == 0
    assert crop.image[CARD_H // 2, CARD_W // 2].tolist() == [128, 128, 128]


# --------------------------------------------------------------------------- #
# Warp quality: crop should look like the original card, upright, not mirrored
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("condition", ["clean", "perspective", "rotated"])
def test_warp_reproduces_card(condition):
    scene = make_scene(seed=3, condition=condition)
    crop = _detect(scene)
    assert crop.found_quad
    diff = np.abs(crop.image.astype(np.float32) - scene.card_rgb.astype(np.float32)).mean()
    flipped = np.abs(
        crop.image.astype(np.float32) - scene.card_rgb[:, ::-1].astype(np.float32)
    ).mean()
    upside_down = np.abs(
        crop.image.astype(np.float32) - scene.card_rgb[::-1, ::-1].astype(np.float32)
    ).mean()
    assert diff < 12, f"mean abs diff {diff:.1f}"
    assert diff < flipped and diff < upside_down


# --------------------------------------------------------------------------- #
# Visual: draw outlines for inspection
# --------------------------------------------------------------------------- #
@pytest.mark.visual
@pytest.mark.parametrize("condition", CONDITIONS)
def test_overlay_synthetic(condition):
    scene = make_scene(seed=7, condition=condition)
    crop = _detect(scene)
    path = save_overlay(draw_detection(crop, scene.quad, title=condition), f"detect_{condition}")
    assert path.exists()


def _real_samples(n: int = 12) -> list[Path]:
    images_dir = get_settings().images_dir
    if not images_dir.exists():
        return []
    files = sorted(images_dir.glob("*_front.jpg"))
    if not files:
        return []
    step = max(1, len(files) // n)
    return files[::step][:n]


@pytest.mark.visual
def test_overlay_real_photos():
    samples = _real_samples()
    if not samples:
        pytest.skip("no real photos under data/images/")
    found = 0
    for path in samples:
        crop = detect_and_crop(path.read_bytes())
        found += crop.found_quad
        save_overlay(draw_detection(crop, title=path.stem), f"real_{path.stem}")
    print(f"\nreal photos: quad found on {found}/{len(samples)}")
