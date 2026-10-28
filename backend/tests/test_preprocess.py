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
    MAX_LONG_EDGE,
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
    if kind == "ndarray":
        assert crop.source_bgr.shape == scene.bgr.shape
    else:  # uploads are capped like ingest: 1200x1000 -> 1024 on the long edge
        assert max(crop.source_bgr.shape[:2]) == 1024


def test_png_bytes_with_alpha_are_accepted():
    rgba = np.dstack([bgr_to_rgb(make_scene(seed=2).bgr), np.full((1200, 1000), 255, np.uint8)])
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    assert detect_and_crop(buf.getvalue()).found_quad


# --------------------------------------------------------------------------- #
# Upload normalisation (EXIF orientation, long-edge cap) and sideways retry
# --------------------------------------------------------------------------- #
_EXIF_ORIENTATION = 0x0112
# Stored pixels for an upright photo under each EXIF orientation: the viewer
# applies the inverse to display it upright (3: 180, 6: 90 CW, 8: 90 CCW).
_STORED_FOR_ORIENTATION = {
    3: lambda rgb: rgb[::-1, ::-1],
    6: lambda rgb: np.rot90(rgb, 1),   # stored 90 CCW
    8: lambda rgb: np.rot90(rgb, -1),  # stored 90 CW
}


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation of two equally sized images."""
    x = a.astype(np.float64).ravel()
    y = b.astype(np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


def _jpeg(rgb: np.ndarray, orientation: int | None = None) -> bytes:
    exif = Image.Exif()
    if orientation is not None:
        exif[_EXIF_ORIENTATION] = orientation
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="JPEG", quality=92, exif=exif)
    return buf.getvalue()


@pytest.fixture(scope="module")
def upright_crop():
    crop = detect_and_crop(_jpeg(bgr_to_rgb(make_scene(seed=0).bgr)))
    assert crop.found_quad and crop.rotation == 0
    return crop


def test_max_long_edge_matches_ingest():
    from ml.data.image_io import IMAGE_MAX_LONG_EDGE

    assert MAX_LONG_EDGE == IMAGE_MAX_LONG_EDGE


@pytest.mark.parametrize("orientation", [3, 6, 8])
def test_exif_orientation_is_applied(orientation, upright_crop):
    rgb = bgr_to_rgb(make_scene(seed=0).bgr)
    crop = detect_and_crop(_jpeg(_STORED_FOR_ORIENTATION[orientation](rgb), orientation))
    assert crop.found_quad
    assert crop.rotation == 0  # upright after the transpose: no retry needed
    assert _ncc(crop.image, upright_crop.image) >= 0.9


def test_high_res_upload_is_capped():
    scene = make_scene(seed=0, size=(3600, 3000))
    crop = detect_and_crop(to_jpeg_bytes(scene.bgr))
    assert crop.found_quad
    assert max(crop.source_bgr.shape[:2]) == 1024
    assert crop.scale == pytest.approx(1024 / 3600, rel=1e-3)


def test_small_upload_is_not_resized():
    crop = detect_and_crop(to_jpeg_bytes(make_scene(seed=0, size=(840, 700)).bgr))
    assert crop.source_bgr.shape[:2] == (840, 700)
    assert crop.scale == 1.0


def test_numpy_input_is_not_normalised():
    scene = make_scene(seed=0, size=(1500, 1250))
    crop = detect_and_crop(bgr_to_rgb(scene.bgr))
    assert crop.found_quad
    assert crop.source_bgr.shape == scene.bgr.shape
    assert crop.scale == 1.0


def test_sideways_upload_is_found_by_retry(upright_crop):
    sideways = cv2.rotate(make_scene(seed=0).bgr, cv2.ROTATE_90_CLOCKWISE)
    crop = detect_and_crop(to_jpeg_bytes(sideways))
    assert crop.found_quad
    assert crop.rotation == 90
    # The retry turns the photo portrait again, and the quad, contour and
    # source are all in that rotated frame.
    assert crop.source_bgr.shape[0] > crop.source_bgr.shape[1]
    assert crop.quad[:, 0].max() < crop.source_bgr.shape[1]
    assert crop.quad[:, 1].max() < crop.source_bgr.shape[0]
    # Which way up the card ends is not this retry's job (face-side guard).
    ncc = max(_ncc(crop.image, upright_crop.image), _ncc(crop.image, upright_crop.image[::-1, ::-1]))
    assert ncc >= 0.9



def test_sideways_card_in_portrait_photo_is_not_retried():
    # Only landscape photos are retried: on upright portrait slab photos the
    # rotated search finds false quads (inner panels of full-art cards). The
    # accepted cost is that a sideways card inside a portrait photo falls back.
    sideways = cv2.rotate(make_scene(seed=0).bgr, cv2.ROTATE_90_CLOCKWISE)  # 1000x1200
    h, w = sideways.shape[:2]
    pad = (w + 200 - h) // 2
    portrait = cv2.copyMakeBorder(sideways, pad, pad, 0, 0, cv2.BORDER_REPLICATE)
    assert portrait.shape[0] > portrait.shape[1]
    crop = detect_and_crop(to_jpeg_bytes(portrait))
    assert not crop.found_quad
    assert crop.rotation == 0

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
# Real data: upload normalisation is a no-op on the training photos
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_stored_precrops_are_reproduced_byte_identical():
    """Stored photos are upright and 1024 px, so serving reproduces precrop exactly.

    Checks a random sample of test-split faces plus every stored fallback face
    (`crop_found_* = 0`, any split): those are where a retry could find a new
    (false) quad and change the crop. Reads only: the DB is opened read-only
    and nothing is written.
    """
    import sqlite3

    s = get_settings()
    if not s.db_path.exists():
        pytest.skip("no DB")
    conn = sqlite3.connect(f"file:{s.db_path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT front_path, cropped_front_path, back_path, cropped_back_path FROM certs "
            "WHERE split='test' AND has_images=1 AND cropped_front_path IS NOT NULL "
            "AND cropped_back_path IS NOT NULL ORDER BY cert_number"
        ).fetchall()
        fallback_rows = conn.execute(
            "SELECT front_path, cropped_front_path, back_path, cropped_back_path, "
            "crop_found_front, crop_found_back FROM certs "
            "WHERE has_images=1 AND (crop_found_front=0 OR crop_found_back=0) "
            "ORDER BY cert_number"
        ).fetchall()
    finally:
        conn.close()
    faces = [
        (s.data_dir / raw, s.data_dir / crop)
        for fp, fc, bp, bc in rows
        for raw, crop in ((fp, fc), (bp, bc))
        if raw and (s.data_dir / raw).exists() and (s.data_dir / crop).exists()
    ]
    if len(faces) < 50:
        pytest.skip(f"only {len(faces)} test faces with a stored precrop")
    picks = [faces[i] for i in np.random.default_rng(0).choice(len(faces), 50, replace=False)]
    fallbacks = [
        (s.data_dir / raw, s.data_dir / crop)
        for fp, fc, bp, bc, ff, fb in fallback_rows
        for raw, crop, found in ((fp, fc, ff), (bp, bc, fb))
        if found == 0 and raw and crop
        and (s.data_dir / raw).exists() and (s.data_dir / crop).exists()
    ]
    picks += [f for f in fallbacks if f not in picks]
    mismatched = []
    for raw, crop in picks:
        result = detect_and_crop(raw.read_bytes())
        if encode_jpeg(result.image) != crop.read_bytes():
            mismatched.append(raw.name)
        elif (raw, crop) in fallbacks:
            assert not result.found_quad and result.rotation == 0, raw.name
    assert not mismatched, f"{len(mismatched)}/{len(picks)} differ: {mismatched}"


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
