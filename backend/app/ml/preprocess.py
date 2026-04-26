"""Card detection + perspective dewarp.

The PSA cert images we train on are slab photos: a clear plastic case with the
card inside. We want just the card. Approach:

1. Convert to grayscale, blur, Canny edge detection.
2. Find external contours, sort by area.
3. For the largest few, approximate to 4 points; pick the first that fits.
4. Apply a perspective warp to a fixed canvas.

Falls back to a simple bounding-box crop if no quad is found, which keeps the
pipeline robust on cropped/scanned images that already exclude the slab.

This is intentionally classical (no model dependency at preprocess time).
A YOLO upgrade is straightforward to swap in later.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

CARD_W = 600
CARD_H = 840


@dataclass(slots=True)
class CardCrop:
    image: np.ndarray  # (CARD_H, CARD_W, 3) uint8 RGB
    found_quad: bool


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array(
        [
            pts[np.argmin(s)],     # tl
            pts[np.argmin(diff)],  # tr
            pts[np.argmax(s)],     # br
            pts[np.argmax(diff)],  # bl
        ],
        dtype=np.float32,
    )


def _find_card_quad(bgr: np.ndarray) -> np.ndarray | None:
    h, w = bgr.shape[:2]
    img_area = h * w
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(gray, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]

    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.15 * img_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = _order_quad(approx.astype(np.float32))
            # Sanity check: aspect ratio in the [0.55, 0.80] range (cards are 5:7 ~= 0.71).
            (tl, tr, br, bl) = quad
            wA = np.linalg.norm(br - bl)
            wB = np.linalg.norm(tr - tl)
            hA = np.linalg.norm(tr - br)
            hB = np.linalg.norm(tl - bl)
            quad_w = max(wA, wB)
            quad_h = max(hA, hB)
            if quad_h <= 0:
                continue
            ar = quad_w / quad_h
            if 0.50 <= ar <= 0.85:
                return quad
    return None


def _warp_to_canvas(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    dst = np.array(
        [[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(bgr, M, (CARD_W, CARD_H))


def _resize_pad(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = min(CARD_W / w, CARD_H / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((CARD_H, CARD_W, 3), dtype=np.uint8)
    y0 = (CARD_H - new_h) // 2
    x0 = (CARD_W - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def detect_and_crop(image: np.ndarray | bytes | Image.Image) -> CardCrop:
    """Detect the card in `image` and return a 600x840 RGB crop.

    Accepts a numpy BGR/RGB array, raw bytes, or a PIL image.
    """
    if isinstance(image, (bytes, bytearray)):
        pil = Image.open(io.BytesIO(image)).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    elif isinstance(image, Image.Image):
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("expected HxWx3 image")
        # We assume incoming numpy arrays are RGB unless they're flagged otherwise.
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    quad = _find_card_quad(bgr)
    if quad is not None:
        warped = _warp_to_canvas(bgr, quad)
        rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        return CardCrop(image=rgb, found_quad=True)

    fallback = _resize_pad(bgr)
    rgb = cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB)
    return CardCrop(image=rgb, found_quad=False)


def encode_jpeg(rgb: np.ndarray, quality: int = 92) -> bytes:
    """Helper for returning preprocessed images via the API in dev mode."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()
