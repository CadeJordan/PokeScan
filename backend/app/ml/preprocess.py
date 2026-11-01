"""Card detection + perspective dewarp.

The PSA cert images we train on are slab photos: a clear plastic case with the
card inside. We want just the card. Approach:

1. Canny edge maps at several settings (fixed, median-relative, CLAHE, and a
   chroma channel for coloured borders on grey/black backgrounds).
2. Every contour (RETR_LIST) whose hull fits a card-shaped, roughly
   rectangular quad becomes a candidate; each side is snapped onto edge
   pixels running along it and scored by how much of it is edge-supported.
3. Take the largest well-supported, card-proportioned candidate; while it is
   a slab holder (grey gasket strip around a nested outline whose border is
   brighter/more colourful), step inward to the card. Past the first step a
   strong outline steps into a weak one only across a flat gasket (otherwise
   the weak one is an inner frame), and as a last step a flat gasket around
   a textured border of the same brightness and colour (silver full-art
   cards) counts as a holder.
4. Apply a perspective warp to a fixed canvas.

On PSA slab photos (measured with `precrop.crop_kind`; see
intent/tighter-slab-crops) this lands on the card itself for ~96% of fronts
and ~98% of backs; most of the rest are the slab well/gasket window around
the card, which excludes the grade label.

Uploads (bytes or PIL images) are first normalised like ingest
(`ml.data.image_io`): rotated upright per their EXIF orientation and resized
to at most `MAX_LONG_EDGE` px on the long edge, so a phone photo reaches the
detector in the same form as the training photos. The stored training photos
are already upright and 1024 px, so this is a no-op for them. Numpy arrays are
not normalised: callers passing arrays own their preprocessing.

The aspect gate only accepts portrait outlines, so when no quad is found in a
landscape photo (wider than tall, after EXIF transpose and the cap; this
applies to numpy inputs too) the photo is rotated 90 degrees clockwise and
searched once more: a card shot sideways is then found, though it may come out
upside down. Portrait photos are not retried, because there the rotated search
finds false quads (inner text panels of full-art cards); the cost is that a
sideways card inside a portrait photo is not found.

If there is still no quad, the whole photo is resized and padded onto the
canvas (no crop at all), which keeps the pipeline running on images that
already exclude the slab; callers see `found_quad=False`.

This is intentionally classical (no model dependency at preprocess time).
A YOLO upgrade is straightforward to swap in later.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

CARD_W = 600
CARD_H = 840
# Long-edge cap for uploads. Must equal `ml.data.image_io.IMAGE_MAX_LONG_EDGE`,
# the cap applied to every training photo at ingest.
MAX_LONG_EDGE = 1024
_EXIF_ORIENTATION = 0x0112
# EXIF orientation -> the transpose that turns the image upright, as in
# `ImageOps.exif_transpose`.
_EXIF_TRANSPOSE = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}
# The only formats `decode_image` opens from bytes: the ones `/api/grade`
# accepts (`backend.app.api.grade.ALLOWED_TYPES`: image/jpeg, image/png,
# image/webp; keep the two in sync). The content type is client-declared, so
# without this any format PIL knows (TIFF, GIF, BMP, ...) would reach its
# decoder; bytes of another format are an `ImageDecodeError` (400). Stored
# training photos are JPEG (`ml.data.image_io`).
UPLOAD_FORMATS = ("JPEG", "PNG", "WEBP")
# What PIL raises for a file it can't decode. Besides `OSError` (unidentified,
# truncated, or not one of `UPLOAD_FORMATS`) its plugins raise `SyntaxError`
# ("broken PNG file", "not a TIFF file"), `ValueError`, `IndexError`,
# `TypeError` and `struct.error` on malformed headers and chunks. `KeyError`
# and `OverflowError` are a safety net: PIL's TIFF loader raised them for a
# malformed IFD (a sub-IFD pointer in the wrong IFD, a strip offset past
# `ssize_t`), and the allowed formats' decoders share that code.
_DECODE_ERRORS = (
    OSError,
    SyntaxError,
    ValueError,
    IndexError,
    TypeError,
    KeyError,
    OverflowError,
    struct.error,
    Image.DecompressionBombError,
)


class ImageDecodeError(ValueError):
    """The upload isn't a readable image (corrupt, truncated, or not an image).
    `face` ("front" / "back") is set once the caller knows which upload it was."""

    def __init__(self, message: str, face: str | None = None) -> None:
        super().__init__(message)
        self.face = face


@dataclass(slots=True)
class CardCrop:
    image: np.ndarray  # (CARD_H, CARD_W, 3) uint8 RGB
    found_quad: bool
    # Original-image artifacts, populated only when a card quad was detected.
    # `quad` is the 4-point card outline (tl, tr, br, bl) in source pixels;
    # `contour` is the full external contour (pre-approxPolyDP) for fidelity in
    # corner/edge geometry analysis; `source_bgr` is the decoded source image.
    # All three are in the frame of `source_bgr`, i.e. after `rotation`.
    quad: np.ndarray | None = None
    contour: np.ndarray | None = None
    source_bgr: np.ndarray | None = None
    # Degrees clockwise the photo was rotated before the quad was found (0 or
    # 90, see the sideways retry), and the normalised / decoded size ratio
    # from `decode_image` (1.0 when not resized, and for numpy inputs).
    rotation: int = 0
    scale: float = 1.0


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


CARD_AR = CARD_W / CARD_H  # 5:7 ~= 0.714
# Accepted width/height range for a detected card. Tight enough to reject the
# whole PSA slab (~0.60) and the label, loose enough for ~20 deg of tilt.
_AR_MIN, _AR_MAX = 0.65, 0.82
_MIN_AREA_FRAC = 0.15
# Outlines within this fraction of the image size from the border are the
# frame or the slab (which fills slab photos), not a card.
_BORDER_FRAC = 0.012
_MIN_FILL = 0.90        # hull area / quad area: how well 4 points explain the shape
_MAX_CORNER_COS = 0.21  # corner angles within ~78-102 degrees
# Fraction of points along the quad outline that sit on edge pixels (mean,
# weakest side) for an outline to count as "strong". Weak outlines are only
# used when there is no strong one: a dark card border can hide one side, but
# a hull closed by a chord across the card (broken slab outline) also has one
# unsupported side.
_STRONG_SUPPORT_MEAN, _STRONG_SUPPORT_SIDE = 0.85, 0.75
_BETTER_SUPPORT = 0.3
_MIN_INNER_SUPPORT = 0.3  # worst-side support for a weak outline to step into
_CONTAIN_FRAC = 0.97
_WEAK_NEST_MAX = 0.9
# Narrowest "card-proportioned" aspect. Slab interiors and hulls closed across
# the card come out narrow (~0.60-0.67); a tilted card can come out wider than
# 5:7 (up to ~0.79), so only the low side is used to demote outlines.
_AR_GOOD_MIN = 0.675
_MIN_ORIENT_COS = 0.9  # edge gradient within ~25 deg of a side's normal
_SNAP_PASSES = 2
_SNAP_FRAC = 0.012  # max per-endpoint side shift when snapping, x image size
# Slab handling: a nested card-shaped outline at this area ratio inside the
# chosen one replaces it when the ring between them looks like a dark gasket.
_NEST_MIN, _NEST_MAX = 0.5, 0.97
_HOLDER_MAX_LUMA = 130
_HOLDER_CONTRAST = 18
_HOLDER_MAX_CHROMA = 30
_HOLDER_STRIP = 0.03  # width of the strip checked outside the card, x its size
# A gasket the same brightness and colour as the card's border (silver or grey
# borders in a grey well) is told apart by texture instead: moulded plastic is
# flat, the printed card is not. Luma median absolute deviations, in grey levels.
_GASKET_MAX_MAD = 20
_GASKET_MIN_TEXTURE = 15  # the band's MAD, so a flat ring around a flat band isn't one
_GASKET_TEXTURE_RATIO = 2.0


def _quad_aspect(quad: np.ndarray) -> float:
    tl, tr, br, bl = quad
    quad_w = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    quad_h = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    return float(quad_w / quad_h) if quad_h > 0 else 0.0


def _fit_quad(hull: np.ndarray) -> np.ndarray | None:
    """Fit 4 corners to a convex hull.

    Real cards have rounded corners, so approxPolyDP at a single epsilon often
    yields 6-8 points. Try a few epsilons, then fall back to the min-area
    rectangle when the hull is rectangular enough.
    """
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
    if cv2.contourArea(hull) / max(cv2.contourArea(box), 1.0) >= _MIN_FILL:
        return box
    return None


def _chroma(bgr: np.ndarray) -> np.ndarray:
    """max(B,G,R) - min(B,G,R): colourfulness that stays ~0 on dark greys
    (HSV saturation blows up there)."""
    b, g, r = cv2.split(bgr)  # cv2 max/min: same values, ~20x faster than axis=2
    return cv2.subtract(cv2.max(cv2.max(b, g), r), cv2.min(cv2.min(b, g), r))


def _edge_maps(bgr: np.ndarray, chroma: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """(edge map, source channel) pairs from a few settings, most conservative first.

    The chroma channel separates a coloured card border from a grey or
    black background of similar brightness (e.g. a blue card back on the
    dark gasket of a PSA slab).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 7, 50, 50)
    med = float(np.median(smooth))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(smooth)
    chroma = cv2.bilateralFilter(chroma, 7, 50, 50)
    variants = [
        (smooth, 60, 180),
        (smooth, max(5, int(0.5 * med)), max(15, int(1.2 * med))),
        (clahe, 40, 120),
        (chroma, 30, 90),
    ]
    kernel = np.ones((3, 3), np.uint8)
    return [
        (cv2.dilate(cv2.Canny(img, lo, hi), kernel, iterations=1), img)
        for img, lo, hi in variants
    ]


def _well_formed(quad: np.ndarray, hull: np.ndarray, hull_area: float) -> bool:
    """Ordered quad is convex, roughly rectangular, and explains the hull."""
    if not cv2.isContourConvex(quad.reshape(4, 1, 2)):
        return False
    q_area = cv2.contourArea(quad)
    inter, _ = cv2.intersectConvexConvex(
        quad.reshape(4, 1, 2).astype(np.float32), hull.astype(np.float32)
    )
    if inter / max(q_area + hull_area - inter, 1.0) < _MIN_FILL:
        return False
    return _rectangular(quad)


def _rectangular(quad: np.ndarray) -> bool:
    if not cv2.isContourConvex(quad.reshape(4, 1, 2).astype(np.float32)):
        return False
    for i in range(4):
        a, b, c = quad[i - 1], quad[i], quad[(i + 1) % 4]
        u, v = a - b, c - b
        cos = float(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-6))
        if abs(cos) > _MAX_CORNER_COS:
            return False
    return True


class _Grad:
    """Gradient field used to check that an edge pixel runs along a quad side."""

    def __init__(self, gray: np.ndarray):
        g = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy) + 1e-6
        # Unit gradient, so `oriented` needs two lookups and no division.
        self.ux = gx / mag
        self.uy = gy / mag

    def oriented(self, edges: np.ndarray, xs: np.ndarray, ys: np.ndarray, nrm: np.ndarray) -> np.ndarray:
        """Edge pixels at (xs, ys) whose gradient is along the side normal.

        Card sides are straight edges, so their gradient points across the
        side; text and artwork edges crossed by a chord point every which way.
        Orientation is only computed where there is an edge pixel (sparse).
        """
        hit = edges[ys, xs] > 0
        hx, hy = xs[hit], ys[hit]
        hit[hit] = np.abs(self.ux[hy, hx] * nrm[0] + self.uy[hy, hx] * nrm[1]) >= _MIN_ORIENT_COS
        return hit


def _side_frame(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    v = b - a
    length = float(np.linalg.norm(v))
    if length < 1:
        return None
    return v, np.array([-v[1], v[0]], np.float32) / length


def _edge_support(
    edges: np.ndarray, grad: _Grad, quad: np.ndarray, n: int = 40, tol: int = 2
) -> tuple[float, float]:
    """(mean, worst-side) fraction of points along the quad on an aligned edge.

    A contour broken by glare or a label gets closed by the convex hull with a
    chord that crosses the card; such a side has little aligned edge support.
    """
    h, w = edges.shape
    t = np.linspace(0.1, 0.9, n, dtype=np.float32)[:, None]
    offs = np.arange(-tol, tol + 1, dtype=np.float32)[:, None, None]
    sides = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        frame = _side_frame(a, b)
        if frame is None:
            return 0.0, 0.0
        v, nrm = frame
        pts = (a + v * t)[None] + offs * nrm  # (2*tol+1, n, 2)
        xs = np.clip(np.round(pts[..., 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(pts[..., 1]).astype(int), 0, h - 1)
        sides.append(float(grad.oriented(edges, xs, ys, nrm).any(axis=0).mean()))
    return float(np.mean(sides)), min(sides)


def _line_intersection(p1, p2, p3, p4) -> np.ndarray | None:
    d1, d2 = p2 - p1, p4 - p3
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-6:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / den
    return p1 + t * d1


def _snap_quad(edges: np.ndarray, grad: _Grad, quad: np.ndarray, n: int = 40) -> np.ndarray | None:
    """Slide/tilt each side onto the nearest run of aligned edge pixels.

    Hull-derived corners are biased outward when the card contour merges with
    a nearby outline (slab well, glare); snapping each side independently to
    the edge map and re-intersecting gives a quad that sits on the card edge.
    """
    h, w = edges.shape
    r = max(4, round(_SNAP_FRAC * max(h, w)))
    offs = np.arange(-r, r + 1, dtype=np.float32)
    t = np.linspace(0.1, 0.9, n, dtype=np.float32)
    # Endpoint offsets d1 (at a) and d2 (at b), interpolated along the side.
    shift = offs[:, None, None] * (1 - t) + offs[None, :, None] * t  # (R, R, n)
    # Prefer the smallest move among equally supported positions.
    penalty = (np.abs(offs)[:, None] + np.abs(offs)[None, :]) / (2 * r) * 1e-3
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        frame = _side_frame(a, b)
        if frame is None:
            return None
        v, nrm = frame
        base = a + v * t[:, None]  # (n, 2)
        pts = base[None, None] + shift[..., None] * nrm  # (R, R, n, 2)
        xs = np.clip(np.round(pts[..., 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(pts[..., 1]).astype(int), 0, h - 1)
        # Integer count / n equals the bool mean exactly, without float64 temporaries.
        hits = grad.oriented(edges, xs, ys, nrm).view(np.uint8).sum(axis=-1, dtype=np.uint16)
        score = hits / n - penalty
        bi, bj = np.unravel_index(int(np.argmax(score)), score.shape)
        lines.append((a + offs[bi] * nrm, b + offs[bj] * nrm))
    corners = []
    for i in range(4):
        p = _line_intersection(*lines[i - 1], *lines[i])
        if p is None:
            return None
        corners.append(p)
    return np.array(corners, np.float32)


# (hull_area, quad, contour, worst-side support, strong)
_Cand = tuple[float, np.ndarray, np.ndarray, float, bool]


def _candidates(gray: np.ndarray, edges: np.ndarray, grad: _Grad) -> list[_Cand]:
    """Card-shaped outlines in one edge map (see `_Cand`)."""
    h, w = gray.shape
    img_area = h * w
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mx, my = max(3, round(_BORDER_FRAC * w)), max(3, round(_BORDER_FRAC * h))
    out = []
    min_area = _MIN_AREA_FRAC * img_area
    for c in contours:
        # The hull fits in the bounding box, so this cheap check skips the
        # many small edge fragments before computing any hull.
        _, _, bw, bh = cv2.boundingRect(c)
        if bw * bh < min_area:
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(hull)
        if x <= mx or y <= my or x + bw >= w - mx or y + bh >= h - my:
            continue
        pts = _fit_quad(hull)
        if pts is None:
            continue
        quad = _order_quad(pts)
        if not _AR_MIN <= _quad_aspect(quad) <= _AR_MAX:
            continue
        if not _well_formed(quad, hull, hull_area):
            continue
        # Two passes: heavily rounded corners pull hull vertices well inside
        # the true corners, further than one snap radius.
        for _ in range(_SNAP_PASSES):
            quad = _snap_quad(edges, grad, quad)
            if quad is None:
                break
        if quad is None or not _AR_MIN <= _quad_aspect(quad) <= _AR_MAX:
            continue
        if not _rectangular(quad):
            continue
        mean_sup, side_sup = _edge_support(edges, grad, quad)
        strong = mean_sup >= _STRONG_SUPPORT_MEAN and side_sup >= _STRONG_SUPPORT_SIDE
        out.append((hull_area, quad, c, side_sup, strong))
    return out


def _poly_mask(shape: tuple[int, int], quad: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
    return mask


def _shrink(quad: np.ndarray, frac: float) -> np.ndarray:
    centre = quad.mean(axis=0)
    return centre + (quad - centre) * (1.0 - frac)


def _is_holder(
    gray: np.ndarray,
    chroma: np.ndarray,
    outer: np.ndarray,
    inner: np.ndarray,
    allow_darker: bool = False,
    allow_flat: bool = False,
    require_flat: bool = False,
) -> bool:
    """True when `outer` is a slab well / holder around the card `inner`.

    In a PSA slab the strip just outside the card is the gasket: grey and
    not bright. The card's own border just inside `inner` is then clearly
    brighter (silver, white) or more colourful (yellow, blue, tan). Between a
    card's outer edge and an inner frame the ring is the card border itself,
    which is coloured or bright and so fails the first test.

    `allow_darker` also accepts a clearly darker band: the gasket window seen
    from the slab interior, or a dark-bordered card in a grey well. It is only
    used for the first step from the outermost outline, since inside a card's
    silver rim a darker band is just dark artwork.

    `allow_flat` also accepts a band no brighter or more colourful than the
    ring when the ring is flat and the band clearly textured: a silver- or
    grey-bordered full-art card in a grey well, whose border matches the
    gasket in brightness and colour. `_find_card_quad` only tries it when no
    other step applies, on strong nested outlines.

    `require_flat` asks for that flat-ring, textured-band pattern on top of
    the brightness or colour contrast: a gasket, not a card's own border.
    """
    kernel = np.ones((5, 5), np.uint8)
    inner_mask = _poly_mask(gray.shape, inner)
    # Only the strip just outside `inner`: further out a holder can be
    # anything (bright slab plastic, the label), but right around a slabbed
    # card is the gasket.
    around = cv2.bitwise_and(
        _poly_mask(gray.shape, outer), _poly_mask(gray.shape, _shrink(inner, -_HOLDER_STRIP))
    )
    ring = cv2.subtract(around, cv2.dilate(inner_mask, kernel, iterations=1))
    ring = cv2.erode(ring, kernel, iterations=1)
    band = cv2.subtract(inner_mask, _poly_mask(gray.shape, _shrink(inner, 0.04)))
    band = cv2.erode(band, kernel, iterations=1)
    if cv2.countNonZero(ring) < 50 or cv2.countNonZero(band) < 50:
        return False
    ring_luma = float(np.median(gray[ring > 0]))
    band_luma = float(np.median(gray[band > 0]))
    ring_chroma = float(np.median(chroma[ring > 0]))
    # Upper quartile: a thin silver rim inside the band should not hide a
    # coloured border or artwork just inside it.
    band_chroma = float(np.percentile(chroma[band > 0], 75))
    if ring_luma >= _HOLDER_MAX_LUMA or ring_chroma >= _HOLDER_MAX_CHROMA:
        return False
    contrast = (
        band_luma > ring_luma + _HOLDER_CONTRAST
        or (allow_darker and band_luma < ring_luma - _HOLDER_CONTRAST)
        or band_chroma > ring_chroma + _HOLDER_CONTRAST
    )
    if contrast and not require_flat:
        return True
    if not (allow_flat or require_flat):
        return False
    ring_mad = float(np.median(np.abs(gray[ring > 0].astype(np.float32) - ring_luma)))
    band_mad = float(np.median(np.abs(gray[band > 0].astype(np.float32) - band_luma)))
    flat = ring_mad <= _GASKET_MAX_MAD and band_mad >= max(
        _GASKET_MIN_TEXTURE, _GASKET_TEXTURE_RATIO * ring_mad
    )
    return flat and (contrast or allow_flat)


def _contains(outer: np.ndarray, inner: np.ndarray) -> bool:
    """`inner` lies (almost) entirely inside `outer`.

    Measured by area so that a slightly skewed outer quad (one side snapped
    onto a nearby slanted edge) still counts as containing the card.
    """
    inter, _ = cv2.intersectConvexConvex(
        outer.reshape(4, 1, 2).astype(np.float32), inner.reshape(4, 1, 2).astype(np.float32)
    )
    return inter >= _CONTAIN_FRAC * cv2.contourArea(inner.astype(np.float32))


def _find_card_quad(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Find the card outline.

    Returns `(quad, contour)` where `quad` is the ordered 4-point outline and
    `contour` is the full contour (Nx1x2) that produced it, or None.

    Contours are searched with RETR_LIST rather than RETR_EXTERNAL: in slab
    photos the slab's own outline often touches the image border and would
    otherwise hide the card nested inside it. Candidates from every edge map
    are pooled; the largest card-shaped outline wins (preferring outlines
    well supported by edge pixels on all sides), except that when it is
    a slab well/holder (see `_is_holder`) we step inward to the card.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    chroma = _chroma(bgr)
    cands: list[_Cand] = []
    grads: dict[int, _Grad] = {}  # two edge maps share the smoothed gray
    for edges, src in _edge_maps(bgr, chroma):
        if id(src) not in grads:
            grads[id(src)] = _Grad(src)
        cands.extend(_candidates(gray, edges, grads[id(src)]))
    if not cands:
        return None
    # Prefer, in order: well-supported outlines with the card's proportions,
    # then well-supported ones, then card-proportioned ones, then anything.
    # Slab interiors and hulls closed across the card are usually off-shape.
    good = [_quad_aspect(c[1]) >= _AR_GOOD_MIN for c in cands]
    for tier in (
        [c for c, g in zip(cands, good, strict=True) if c[4] and g],
        [c for c in cands if c[4]],
        [c for c, g in zip(cands, good, strict=True) if g],
        cands,
    ):
        if tier:
            best = max(tier, key=lambda s: s[0])
            break
    # Stepping inward may land on any reasonably supported outline, strong or
    # not: the card edge against a dark gasket is often faint on one side.
    inward = sorted(
        (c for c, g in zip(cands, good, strict=True) if c[4] or (g and c[3] >= _MIN_INNER_SUPPORT)),
        key=lambda s: -s[0],
    )

    for step in range(3):  # slab -> well -> card at most
        nested = [
            c for c in inward
            if _NEST_MIN <= c[0] / best[0] <= _NEST_MAX
            # a weak outline hugging `best` is usually a ghost of the same edge
            and (c[4] or c[0] / best[0] <= _WEAK_NEST_MAX)
            and _contains(best[1], c[1])
        ]
        # Step inward to the largest nested outline that sits in a holder ring
        # or, when `best` itself is weak (e.g. a hull closed across the card),
        # is much better supported. Near-duplicates of `best` (other edge
        # maps) match neither: same support, and too thin a ring.
        # Past the first step, a strong `best` is a well or the card itself,
        # so a weak nested outline there is usually an inner frame or a
        # partial outline in the artwork: it needs a flat gasket around it.
        inner = next(
            (
                c for c in nested
                if (not best[4] and c[3] >= best[3] + _BETTER_SUPPORT)
                or _is_holder(
                    gray, chroma, best[1], c[1], allow_darker=step == 0,
                    require_flat=step > 0 and best[4] and not c[4],
                )
            ),
            None,
        )
        if inner is None:
            # Last resort: a flat gasket around a card border of the same
            # brightness and colour. Only strong nested outlines are tried,
            # largest first, and the search stops after this step: inside a
            # card a flat ring is its plain border, not a holder.
            best = next(
                (c for c in nested if c[4] and _is_holder(gray, chroma, best[1], c[1], allow_flat=True)),
                best,
            )
            break
        best = inner
    return best[1], best[2]


def _warp_to_canvas(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    dst = np.array(
        [[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(quad, dst)  # noqa: N806
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


def decode_image(image: bytes | bytearray | Image.Image) -> tuple[np.ndarray, float]:
    """Decode an upload the way ingest stores photos; returns `(bgr, scale)`.

    EXIF orientation is applied, then the image is converted to RGB (alpha is
    dropped) and, if its long edge exceeds `MAX_LONG_EDGE`, resized with
    LANCZOS to exactly that, rounding like `ml.data.image_io.compress_jpeg`.
    `scale` is the normalised size over the decoded size, 1.0 when not resized.

    Bytes are opened only as one of `UPLOAD_FORMATS` (JPEG, PNG, WebP); any
    other format is an `ImageDecodeError`. A PIL `Image.Image` is passed in
    already opened, so its format isn't checked: the caller chose it.

    Once the image opens, a malformed EXIF block never rejects it: an
    orientation that can't be read is taken as upright, and one that reads is
    applied even when the rest of the block is broken (`_apply_exif_orientation`).
    The exception is EXIF PIL reads while opening: a JPEG whose EXIF makes
    PIL's JPEG `open` fail (e.g. a one-element XResolution, with
    ResolutionUnit set, raises `IndexError` in its dpi read) is reported by PIL
    as unidentified, so it is an `ImageDecodeError` (owner decision 2026-10-28:
    rare, accepted as a 400).

    Raises `ImageDecodeError` when PIL can't decode the image (any of
    `_DECODE_ERRORS`, chained). Only the decode below is wrapped: its body is
    PIL calls, `np.array` and `cv2.cvtColor`, so no other error becomes one.
    """
    try:
        return _decode_image(image)
    except ImageDecodeError:  # a ValueError itself: never wrap it twice
        raise
    except _DECODE_ERRORS as exc:
        raise ImageDecodeError(str(exc) or type(exc).__name__) from exc


def _decode_image(image: bytes | bytearray | Image.Image) -> tuple[np.ndarray, float]:
    if isinstance(image, (bytes, bytearray)):
        # Another format raises UnidentifiedImageError (an OSError).
        pil = Image.open(io.BytesIO(image), formats=UPLOAD_FORMATS)
    else:
        pil = image
    # `Image.open` is lazy: decode the pixels first, so that a failure in the
    # EXIF steps below can only come from the EXIF block, never the pixels
    # (after a failed load PIL would hand back the partial image).
    pil.load()
    # A palette image without its palette (a PNG of colour type 3 with no PLTE
    # chunk) can't be coloured, and PIL's convert asserts on it when it has a
    # transparency entry, so refuse it here as unreadable.
    if pil.mode in ("P", "PA") and pil.palette is None:
        raise ImageDecodeError("palette image has no palette")
    pil = _apply_exif_orientation(pil)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    w, h = pil.size
    longest = max(w, h)
    scale = 1.0
    if longest > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / longest
        pil = pil.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR), scale


def _apply_exif_orientation(pil: Image.Image) -> Image.Image:
    """`pil` turned upright per its EXIF orientation, as `ImageOps.exif_transpose`
    does. `pil` must already be loaded.

    A malformed EXIF block never fails the decode. When the orientation can't be
    read, or isn't 2-8, the image is taken as upright and returned as is. When it
    reads but `exif_transpose` fails (it re-serialises the EXIF after rotating,
    which raises e.g. `AttributeError` for an ASCII tag holding a rational), the
    known orientation is applied directly, giving the same pixels.

    Both steps catch any `Exception`, not just `_DECODE_ERRORS`: the pixels are
    already decoded, so a failure here can only come from the EXIF block and
    can't hide a truncated image. Nothing else in the decode is broadened.
    """
    try:
        method = _EXIF_TRANSPOSE.get(pil.getexif().get(_EXIF_ORIENTATION, 1))
    except Exception:  # unreadable EXIF (or an unhashable tag value): upright
        return pil
    if method is None:  # upright (1), absent or out of range. `exif_transpose`
        return pil  # would only copy, ~10 ms on a 12 MP photo, so skip it.
    try:
        return ImageOps.exif_transpose(pil)
    except Exception:  # the orientation is known: apply it without the EXIF
        return pil.transpose(method)


def detect_and_crop(image: np.ndarray | bytes | Image.Image) -> CardCrop:
    """Detect the card in `image` and return a 600x840 RGB crop.

    Accepts raw bytes or a PIL image, which are normalised by `decode_image`,
    or a numpy RGB array, which is used as is (the caller owns its
    orientation and resolution).
    """
    scale = 1.0
    if isinstance(image, (bytes, bytearray, Image.Image)):
        bgr, scale = decode_image(image)
    else:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("expected HxWx3 image")
        # We assume incoming numpy arrays are RGB unless they're flagged otherwise.
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    rotation = 0
    found = _find_card_quad(bgr)
    if found is None and bgr.shape[1] > bgr.shape[0]:
        # Sideways card: a landscape outline fails the aspect gate, so look
        # again with the photo turned portrait. Only landscape photos are
        # retried: on upright portrait photos the rotated search locks onto
        # inner panels of full-art cards (2 of the 21 stored fallback faces),
        # and every stored photo is portrait. Everything downstream (quad,
        # contour, source_bgr) is in the rotated frame.
        rotated = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        found = _find_card_quad(rotated)
        if found is not None:
            bgr, rotation = rotated, 90
    if found is not None:
        quad, contour = found
        warped = _warp_to_canvas(bgr, quad)
        rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        return CardCrop(
            image=rgb,
            found_quad=True,
            quad=quad,
            contour=contour,
            source_bgr=bgr,
            rotation=rotation,
            scale=scale,
        )

    fallback = _resize_pad(bgr)
    rgb = cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB)
    return CardCrop(image=rgb, found_quad=False, source_bgr=bgr, scale=scale)


def encode_jpeg(rgb: np.ndarray, quality: int = 92) -> bytes:
    """Helper for returning preprocessed images via the API in dev mode."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()
