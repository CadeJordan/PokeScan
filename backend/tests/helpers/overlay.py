"""Debug overlays: draw detected vs. true card outlines and centering lines.

Images are written under `test-artifacts/overlays/` so they can be opened after
a `pytest -m visual` run.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from backend.app.core.config import PROJECT_ROOT
from backend.app.ml.preprocess import CardCrop

ARTIFACTS_DIR = PROJECT_ROOT / "test-artifacts"
OVERLAY_DIR = ARTIFACTS_DIR / "overlays"

# BGR
GREEN = (60, 200, 60)     # ground truth
RED = (40, 40, 230)       # detected quad
YELLOW = (0, 215, 255)    # raw contour
CYAN = (230, 200, 0)      # measured centering lines
WHITE = (255, 255, 255)

CORNER_NAMES = ("TL", "TR", "BR", "BL")


def _thickness(img: np.ndarray) -> int:
    return max(2, round(max(img.shape[:2]) / 400))


def _label(img: np.ndarray, text: str, org: tuple[int, int], colour=WHITE, scale=0.8) -> None:
    t = _thickness(img)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), t + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, t, cv2.LINE_AA)


def draw_quad(img: np.ndarray, quad: np.ndarray, colour, numbered: bool = False) -> None:
    pts = np.round(quad.reshape(4, 2)).astype(np.int32)
    t = _thickness(img)
    cv2.polylines(img, [pts], True, colour, t, cv2.LINE_AA)
    if numbered:
        for name, (x, y) in zip(CORNER_NAMES, pts, strict=True):
            cv2.circle(img, (int(x), int(y)), 3 * t, colour, -1, cv2.LINE_AA)
            _label(img, name, (int(x) + 8, int(y) - 8), colour, 0.7)


def draw_detection(
    crop: CardCrop,
    true_quad: np.ndarray | None = None,
    title: str = "",
) -> np.ndarray:
    """Source image with outlines, side-by-side with the warped crop (BGR).

    green = ground truth, yellow = raw contour, red = detected quad (numbered).
    """
    src = crop.source_bgr.copy() if crop.source_bgr is not None else None
    if src is None:
        src = cv2.cvtColor(crop.image, cv2.COLOR_RGB2BGR)
    if true_quad is not None:
        draw_quad(src, true_quad, GREEN)
    if crop.contour is not None:
        cv2.drawContours(src, [crop.contour], -1, YELLOW, _thickness(src), cv2.LINE_AA)
    if crop.quad is not None:
        draw_quad(src, crop.quad, RED, numbered=True)

    status = "FOUND" if crop.found_quad else "NOT FOUND (fallback)"
    _label(src, f"{title}  {status}".strip(), (12, 36), WHITE if crop.found_quad else RED, 1.0)

    warped = cv2.cvtColor(crop.image, cv2.COLOR_RGB2BGR)
    scale = src.shape[0] / warped.shape[0]
    warped = cv2.resize(warped, (round(warped.shape[1] * scale), src.shape[0]))
    gap = np.full((src.shape[0], 12, 3), 255, np.uint8)
    return np.hstack([src, gap, warped])


def draw_centering(
    rgb: np.ndarray,
    measured: tuple[int, int, int, int],
    expected: tuple[int, int, int, int] | None = None,
    title: str = "",
) -> np.ndarray:
    """Draw inner-frame borders on a warped crop. Tuples are (left, right, top, bottom)."""
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    h, w = img.shape[:2]

    def lines(b, colour, dashed=False):
        left, right, top, bottom = b
        segs = [
            ((left, 0), (left, h - 1)),
            ((w - right, 0), (w - right, h - 1)),
            ((0, top), (w - 1, top)),
            ((0, h - bottom), (w - 1, h - bottom)),
        ]
        for p, q in segs:
            if not dashed:
                cv2.line(img, p, q, colour, 2, cv2.LINE_AA)
                continue
            n = 40
            for i in range(0, n, 2):
                a = (p[0] + (q[0] - p[0]) * i // n, p[1] + (q[1] - p[1]) * i // n)
                c = (p[0] + (q[0] - p[0]) * (i + 1) // n, p[1] + (q[1] - p[1]) * (i + 1) // n)
                cv2.line(img, a, c, colour, 2, cv2.LINE_AA)

    if expected is not None:
        lines(expected, GREEN)
    lines(measured, CYAN, dashed=expected is not None)
    left, right, top, bottom = measured
    _label(img, title, (10, 30), WHITE, 0.7)
    _label(img, f"L{left} R{right} T{top} B{bottom}", (10, h - 14), CYAN, 0.7)
    return img


def save_overlay(img: np.ndarray, name: str) -> Path:
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    path = OVERLAY_DIR / f"{name}.png"
    cv2.imwrite(str(path), img)
    return path
