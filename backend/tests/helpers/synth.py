"""Synthetic card images with exactly-known geometry.

There are no hand-labelled card outlines in the dataset, so CV tests render a
fake card, warp it into a scene with a known homography and push the 4 card
corners through the same transform. That gives pixel-exact ground truth for
the detected quad, and (for centering) exact inner-frame border widths.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from backend.app.ml.preprocess import CARD_H, CARD_W

# Conditions understood by `make_scene`. Each is a short tag so it can be used
# directly as a pytest id.
CONDITIONS = (
    "clean",
    "rotated",
    "perspective",
    "gradient_bg",
    "noisy_bg",
    "cluttered_bg",
    "blur",
    "noise",
    "dark",
    "bright",
    "jpeg",
    "low_contrast",
    "slab",
    "occluded",
    "cut_off",
)


@dataclass(slots=True)
class Scene:
    bgr: np.ndarray            # scene image, BGR uint8
    quad: np.ndarray           # (4, 2) float32 true card corners, card TL, TR, BR, BL
    card_rgb: np.ndarray       # the un-warped CARD_H x CARD_W card (RGB)
    condition: str
    seed: int
    meta: dict = field(default_factory=dict)


def make_card(
    borders: tuple[int, int, int, int] = (36, 36, 40, 40),
    border_rgb: tuple[int, int, int] = (238, 202, 52),
    seed: int = 0,
    width: int = CARD_W,
    height: int = CARD_H,
) -> np.ndarray:
    """Render a Pokemon-ish card: coloured border around an inner art frame.

    `borders` is (left, right, top, bottom) in pixels from the card edge to
    the inner frame, so centering tests know the true answer.
    """
    rng = np.random.default_rng(seed)
    left, right, top, bottom = borders
    card = np.empty((height, width, 3), np.uint8)
    card[:] = border_rgb

    # Inner frame: dark blue-grey panel with a vertical gradient "artwork".
    x0, x1, y0, y1 = left, width - right, top, height - bottom
    ramp = np.linspace(0, 1, y1 - y0, dtype=np.float32)[:, None, None]
    c_top = np.array([60, 90, 140], np.float32)
    c_bot = np.array([30, 45, 80], np.float32)
    card[y0:y1, x0:x1] = (c_top * (1 - ramp) + c_bot * ramp).astype(np.uint8)

    # A few soft blobs inside the art so the crop isn't a flat colour, kept
    # away from the frame edges so they don't compete with the frame gradient.
    inner_w, inner_h = x1 - x0, y1 - y0
    for _ in range(5):
        cx = int(x0 + rng.uniform(0.25, 0.75) * inner_w)
        cy = int(y0 + rng.uniform(0.2, 0.55) * inner_h)
        r = int(rng.uniform(0.04, 0.09) * inner_w)
        colour = tuple(int(v) for v in rng.integers(70, 160, 3))
        cv2.circle(card, (cx, cy), r, colour, -1, lineType=cv2.LINE_AA)
    card[y0:y1, x0:x1] = cv2.GaussianBlur(card[y0:y1, x0:x1], (0, 0), 2.0)

    # Text box near the bottom of the frame, like an attack description.
    tb_y0 = int(y0 + 0.68 * inner_h)
    tb_y1 = int(y0 + 0.88 * inner_h)
    cv2.rectangle(
        card, (x0 + inner_w // 10, tb_y0), (x1 - inner_w // 10, tb_y1), (200, 200, 190), -1
    )
    return card


def _background(kind: str, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "gradient":
        gx = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
        gy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        a = np.array([40, 60, 90], np.float32)
        b = np.array([120, 110, 100], np.float32)
        return (a + (b - a) * (0.6 * gx + 0.4 * gy)).astype(np.uint8)
    if kind == "noisy":
        base = np.full((h, w, 3), 90, np.float32)
        base += rng.normal(0, 18, (h, w, 3))
        return np.clip(base, 0, 255).astype(np.uint8)
    if kind == "cluttered":
        bg = _background("gradient", h, w, rng)
        # Wood-grain-ish stripes plus random small shapes (other objects on a table).
        for y in range(0, h, 9):
            shade = int(rng.integers(-12, 12))
            bg[y : y + 4] = np.clip(bg[y : y + 4].astype(int) + shade, 0, 255).astype(np.uint8)
        for _ in range(25):
            colour = tuple(int(v) for v in rng.integers(20, 235, 3))
            p = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            if rng.random() < 0.5:
                cv2.circle(bg, p, int(rng.integers(8, 40)), colour, -1)
            else:
                q = (p[0] + int(rng.integers(-60, 60)), p[1] + int(rng.integers(-60, 60)))
                cv2.line(bg, p, q, colour, int(rng.integers(2, 6)))
        return bg
    if kind == "light":
        return np.full((h, w, 3), (205, 195, 70), np.uint8)  # BGR; close-ish to border
    return np.full((h, w, 3), (45, 40, 35), np.uint8)  # dark solid table


def _target_quad(
    h: int,
    w: int,
    rng: np.random.Generator,
    scale: float,
    angle_deg: float,
    jitter: float,
    shift: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    card_h = scale * h
    card_w = card_h * CARD_W / CARD_H
    corners = np.array(
        [[-card_w / 2, -card_h / 2], [card_w / 2, -card_h / 2],
         [card_w / 2, card_h / 2], [-card_w / 2, card_h / 2]],
        np.float32,
    )
    t = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], np.float32)
    corners = corners @ rot.T
    corners += rng.uniform(-jitter, jitter, corners.shape).astype(np.float32) * card_w
    centre = np.array([w / 2 + shift[0] * w, h / 2 + shift[1] * h], np.float32)
    return (corners + centre).astype(np.float32)


def make_scene(
    seed: int = 0,
    condition: str = "clean",
    size: tuple[int, int] = (1200, 1000),
    card: np.ndarray | None = None,
) -> Scene:
    """Render `card` into a `size=(h, w)` scene under the named condition."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    rng = np.random.default_rng(seed)
    h, w = size
    if card is None:
        card = make_card(seed=seed)

    bg_kind = {
        "gradient_bg": "gradient",
        "noisy_bg": "noisy",
        "cluttered_bg": "cluttered",
        "low_contrast": "light",
    }.get(condition, "solid")
    scene = _background(bg_kind, h, w, rng)

    scale = float(rng.uniform(0.55, 0.8))
    angle = float(rng.uniform(-4, 4))
    jitter = 0.0
    shift = (0.0, 0.0)
    if condition == "rotated":
        angle = float(rng.choice([-1, 1]) * rng.uniform(12, 25))
        scale = min(scale, 0.65)
    elif condition == "perspective":
        jitter = 0.06
    elif condition == "cut_off":
        shift = (0.0, 0.25)

    quad = _target_quad(h, w, rng, scale, angle, jitter, shift)
    src = np.array([[0, 0], [CARD_W, 0], [CARD_W, CARD_H], [0, CARD_H]], np.float32)
    homography = cv2.getPerspectiveTransform(src, quad)
    card_bgr = cv2.cvtColor(card, cv2.COLOR_RGB2BGR)

    if condition == "slab":
        # Clear plastic slab: a bigger, lighter, translucent rectangle + label.
        slab_quad = _target_quad(h, w, np.random.default_rng(seed), scale * 1.22, angle, 0.0)
        slab_quad[:, 1] -= 0.04 * h
        overlay = scene.copy()
        cv2.fillConvexPoly(overlay, slab_quad.astype(np.int32), (185, 185, 180))
        scene = cv2.addWeighted(overlay, 0.35, scene, 0.65, 0)
        cv2.polylines(scene, [slab_quad.astype(np.int32)], True, (225, 225, 225), 3)
        lbl = slab_quad.copy()
        lbl[2:, 1] = lbl[:2, 1] + 0.12 * scale * h
        cv2.fillConvexPoly(scene, lbl.astype(np.int32), (235, 235, 235))

    warped = cv2.warpPerspective(card_bgr, homography, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((CARD_H, CARD_W), 255, np.uint8), homography, (w, h))
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    scene = (warped * alpha + scene * (1 - alpha)).astype(np.uint8)

    if condition == "slab":
        # Glare streak across the plastic.
        glare = np.zeros_like(scene)
        cv2.line(glare, (int(0.2 * w), 0), (int(0.6 * w), h), (255, 255, 255), 40)
        glare = cv2.GaussianBlur(glare, (0, 0), 25)
        scene = cv2.add(scene, (glare * 0.35).astype(np.uint8))
    elif condition == "occluded":
        # Fingertip over the middle of the right edge.
        mid = ((quad[1] + quad[2]) / 2).astype(int)
        cv2.ellipse(scene, tuple(int(v) for v in mid), (60, 35), 0, 0, 360, (120, 150, 200), -1)
    elif condition == "blur":
        scene = cv2.GaussianBlur(scene, (0, 0), 3.0)
    elif condition == "noise":
        scene = np.clip(scene + rng.normal(0, 22, scene.shape), 0, 255).astype(np.uint8)
    elif condition == "dark":
        scene = cv2.convertScaleAbs(scene, alpha=0.35, beta=0)
    elif condition == "bright":
        scene = cv2.convertScaleAbs(scene, alpha=1.25, beta=45)
    elif condition == "jpeg":
        ok, buf = cv2.imencode(".jpg", scene, [int(cv2.IMWRITE_JPEG_QUALITY), 25])
        assert ok
        scene = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return Scene(
        bgr=scene,
        quad=quad,
        card_rgb=card,
        condition=condition,
        seed=seed,
        meta={"scale": scale, "angle": angle, "jitter": jitter},
    )


def to_jpeg_bytes(bgr: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


# --------------------------------------------------------------------------- #
# Geometry metrics
# --------------------------------------------------------------------------- #
def polygon_iou(a: np.ndarray, b: np.ndarray, shape: tuple[int, int]) -> float:
    """IoU of two polygons, by rasterising them onto a `shape=(h, w)` canvas."""
    ma = np.zeros(shape, np.uint8)
    mb = np.zeros(shape, np.uint8)
    cv2.fillPoly(ma, [np.round(a).astype(np.int32)], 1)
    cv2.fillPoly(mb, [np.round(b).astype(np.int32)], 1)
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum()
    return float(inter / union) if union else 0.0


def corner_error(detected: np.ndarray, truth: np.ndarray) -> float:
    """Max corner distance (px) after best cyclic alignment of the two quads.

    The detector orders corners by image position; the ground truth is ordered
    by card position. They agree for small rotations but alignment makes the
    metric independent of that convention.
    """
    d = detected.reshape(4, 2).astype(np.float64)
    t = truth.reshape(4, 2).astype(np.float64)
    return float(
        min(np.linalg.norm(np.roll(d, k, axis=0) - t, axis=1).max() for k in range(4))
    )


def quad_diagonal(quad: np.ndarray) -> float:
    q = quad.reshape(4, 2)
    return float(np.linalg.norm(q[2] - q[0]))
