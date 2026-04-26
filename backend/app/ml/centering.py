"""Classical-CV centering measurement.

After the card has been dewarped to a fixed CARD_W x CARD_H canvas, the
"inner art frame" is detectable as the dominant rectangular gradient inside
the card border. We:

1. Convert to grayscale, take the gradient magnitude.
2. Project gradient energy onto the X and Y axes.
3. Find the strongest left/right peaks within a plausible band (skipping the
   outer ~3% which is the white border itself), and likewise top/bottom.
4. Compute L/R and T/B ratios out of 100 and map to a centering grade using
   PSA's published tolerances.

This is geometry, not learning - no labels needed. It's the right tool for
this sub-task, and it produces interpretable ratios you can show in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from backend.app.api.grade import CenteringSubGrade
from backend.app.ml.preprocess import CARD_H, CARD_W, detect_and_crop


@dataclass(slots=True)
class CenteringResult:
    left: int
    right: int
    top: int
    bottom: int
    grade: float

    @property
    def lr_ratio(self) -> tuple[int, int]:
        total = max(1, self.left + self.right)
        l = round(self.left / total * 100)
        return l, 100 - l

    @property
    def tb_ratio(self) -> tuple[int, int]:
        total = max(1, self.top + self.bottom)
        t = round(self.top / total * 100)
        return t, 100 - t


# PSA tolerances per the public grading guide. Worst side dictates the bucket.
# Values are the maximum off-centeredness percentage allowed for that grade.
# e.g. 60 means up to 60/40, 70 means up to 70/30. Front-of-card values.
_PSA_TOLERANCES = [
    (10, 60),
    (9, 65),
    (8, 70),
    (7, 75),
    (6, 80),
    (5, 85),
    (4, 90),
    (3, 95),
]


def _grade_from_ratio(worst_side_pct: int) -> float:
    """Map the worse of L/R or T/B (the higher of the two sides) to a grade."""
    for grade, threshold in _PSA_TOLERANCES:
        if worst_side_pct <= threshold:
            return float(grade)
    return 2.0


def _find_inner_edges(rgb: np.ndarray) -> tuple[int, int, int, int]:
    """Find the inner art-frame edges as pixel coordinates.

    Returns (left_x, right_x, top_y, bottom_y) measured from the respective
    card edges (i.e. left_x is distance from the LEFT card edge).
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    h, w = mag.shape
    # Project gradient energy onto each axis.
    col_energy = mag.sum(axis=0)
    row_energy = mag.sum(axis=1)

    # The outermost 3% on each side is the white card border itself; skip it
    # so we lock onto the inner art frame, not the outer card edge.
    pad_x = max(2, int(round(w * 0.03)))
    pad_y = max(2, int(round(h * 0.03)))

    # Search inward from each side for the first column/row whose energy
    # crosses 60% of the local maximum within a search window of ~25% of the
    # card dimension. This is a robust proxy for "where does the inner frame
    # start", and avoids being fooled by glare or background gradients.
    def _scan(profile: np.ndarray, start: int, end: int, forward: bool) -> int:
        seg = profile[start:end]
        if seg.size == 0:
            return start
        threshold = float(seg.max()) * 0.6
        idxs = np.where(seg >= threshold)[0]
        if idxs.size == 0:
            return start + (seg.argmax() if forward else seg.size - 1 - seg[::-1].argmax())
        return start + (idxs[0] if forward else idxs[-1])

    band_x = max(8, int(round(w * 0.25)))
    band_y = max(8, int(round(h * 0.25)))

    left_x = _scan(col_energy, pad_x, pad_x + band_x, forward=True)
    right_x = _scan(col_energy, w - pad_x - band_x, w - pad_x, forward=False)
    top_y = _scan(row_energy, pad_y, pad_y + band_y, forward=True)
    bottom_y = _scan(row_energy, h - pad_y - band_y, h - pad_y, forward=False)

    left = max(0, left_x)
    right = max(0, w - right_x)
    top = max(0, top_y)
    bottom = max(0, h - bottom_y)
    return left, right, top, bottom


def measure_centering(rgb: np.ndarray) -> CenteringResult:
    if rgb.shape[:2] != (CARD_H, CARD_W):
        raise ValueError(f"expected {CARD_H}x{CARD_W} canvas, got {rgb.shape[:2]}")
    left, right, top, bottom = _find_inner_edges(rgb)
    lr_total = max(1, left + right)
    tb_total = max(1, top + bottom)
    worst_side_pct = round(
        max(
            max(left, right) / lr_total,
            max(top, bottom) / tb_total,
        )
        * 100
    )
    grade = _grade_from_ratio(worst_side_pct)
    return CenteringResult(left=left, right=right, top=top, bottom=bottom, grade=grade)


def compute_centering_subgrade(image_bytes: bytes) -> CenteringSubGrade:
    """Public entrypoint used by the API. Crops + measures + returns."""
    crop = detect_and_crop(image_bytes)
    res = measure_centering(crop.image)
    lr = res.lr_ratio
    tb = res.tb_ratio
    return CenteringSubGrade(
        grade=round(res.grade, 1),
        left_right=f"{lr[0]}/{lr[1]}",
        top_bottom=f"{tb[0]}/{tb[1]}",
    )
