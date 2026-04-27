"""Pre-ingest image quality filter.

Cheap, classical-CV checks that reject training samples whose information
content is too low to be useful for the grader:

- min resolution on the long edge (eBay's full-size variant is typically
  >=1200px; <600px is almost certainly a thumbnail or icon)
- Laplacian-variance blur score
- min file size sanity guard

Intentionally simple. We err toward keeping borderline images: training
augmentations will tolerate noise, and excluding too aggressively starves
already-rare grade classes (PSA 1-4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ImageSource = Path | bytes

log = logging.getLogger(__name__)


# Tunables - empirically chosen so PSA catalogue images all pass and only
# obviously bad eBay listings (icons, heavy motion blur) are dropped.
MIN_LONG_EDGE_PX = 600
MIN_FILESIZE_BYTES = 8_000
MIN_LAPLACIAN_VAR = 60.0


@dataclass(slots=True)
class QualityResult:
    ok: bool
    reason: str
    width: int = 0
    height: int = 0
    blur_score: float = 0.0


def _laplacian_variance(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_image(
    source: ImageSource,
    *,
    min_long_edge_px: int = MIN_LONG_EDGE_PX,
    min_filesize_bytes: int = MIN_FILESIZE_BYTES,
    min_laplacian_var: float = MIN_LAPLACIAN_VAR,
) -> QualityResult:
    """Inspect an image. Accepts either an on-disk Path or raw bytes.

    Prefer passing raw bytes when the image is going to be recompressed
    on save: Laplacian-variance scales with resolution, so measuring
    blur on the original capture gives a stable threshold across
    different downsampling targets.
    """
    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
        size = len(raw)
        if size < min_filesize_bytes:
            return QualityResult(ok=False, reason=f"file too small ({size}B)")
        arr = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        path = source
        if not path.exists():
            return QualityResult(ok=False, reason="missing file")
        size = path.stat().st_size
        if size < min_filesize_bytes:
            return QualityResult(ok=False, reason=f"file too small ({size}B)")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        return QualityResult(ok=False, reason="undecodable image")

    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge < min_long_edge_px:
        return QualityResult(
            ok=False, reason=f"resolution {w}x{h} below min {min_long_edge_px}px",
            width=w, height=h,
        )

    blur = _laplacian_variance(image)
    if blur < min_laplacian_var:
        return QualityResult(
            ok=False,
            reason=f"too blurry (lap_var={blur:.1f} < {min_laplacian_var})",
            width=w, height=h, blur_score=blur,
        )

    return QualityResult(ok=True, reason="ok", width=w, height=h, blur_score=blur)
