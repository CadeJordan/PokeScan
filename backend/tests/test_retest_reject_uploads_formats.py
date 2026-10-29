"""Fourth re-test of intent/reject-unusable-uploads, after the D4 fix: bytes are
opened only as `preprocess.UPLOAD_FORMATS` (JPEG, PNG, WebP), and `KeyError` /
`OverflowError` joined `_DECODE_ERRORS`.

Checks that the format restriction rejects every other format PIL can write,
accepts the real-world variants of the three (progressive/CMYK/grey JPEG, MPO,
Ultra HDR-style MPO, 16-bit/interlaced/palette PNG, APNG, lossless/alpha/
animated WebP) exactly as main d496022 decoded them, a hand-built PNG matrix and
a header-only fuzz of the allowed formats, and the 400s end to end, including a
declared content type that doesn't match the bytes. One confirmed defect (D5, a
palette PNG with no palette: 500) was a strict xfail; fixed by refusing such an
image in `_decode_image`, so its tests now pin the 400.

Synthetic images, stub services/guards and tmp_path only; nothing under data/
is read or written.
"""

from __future__ import annotations

import io
import itertools
import random
import struct
import warnings
import zlib

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageOps

from backend.app.ml.preprocess import (
    MAX_LONG_EDGE,
    UPLOAD_FORMATS,
    ImageDecodeError,
    decode_image,
    detect_and_crop,
)
from backend.tests.test_face_guard import _feats, _guard, _service
from backend.tests.test_face_guard import pair as _pair_fixture

pair = _pair_fixture

ORIENTATION = 0x0112
UNREADABLE = "not a readable image (corrupt or truncated file)"


def _post(svc, monkeypatch, front: bytes, back: bytes, ctype: str = "image/jpeg", back_ctype=None):
    """POST through the real app; server errors come back as 500 instead of raising."""
    import backend.app.api.grade as grade_mod
    import backend.app.main as main_mod

    monkeypatch.setattr(grade_mod, "get_inference_service", lambda: svc)
    monkeypatch.setattr(main_mod, "get_inference_service", lambda: svc)
    files = {"front": ("f", front, ctype), "back": ("b", back, back_ctype or ctype)}
    with TestClient(main_mod.create_app(), raise_server_exceptions=False) as c:
        return c.post("/api/grade", files=files)


def _save(im: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Pillow deprecates saving some modes
        im.save(buf, fmt, **kw)
    return buf.getvalue()


def _main_decode(data: bytes) -> tuple[np.ndarray, float]:
    """`decode_image` as on main d496022 (no format restriction), for valid images."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pil = Image.open(io.BytesIO(data))
        if pil.getexif().get(ORIENTATION, 1) != 1:
            pil = ImageOps.exif_transpose(pil)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
    w, h = pil.size
    scale = 1.0
    if max(w, h) > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / max(w, h)
        pil = pil.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR), scale


def _chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def _png(ihdr: bytes, raw: bytes, *extra: bytes) -> bytes:
    """A PNG from an IHDR body, unfiltered scanlines and chunks placed before IDAT."""
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + b"".join(extra)
            + _chunk(b"IDAT", zlib.compress(raw)) + _chunk(b"IEND", b""))


def _interlaced_png(rgb: np.ndarray) -> bytes:
    """An Adam7-interlaced 8-bit RGB PNG of `rgb` (PIL can read but not write these)."""
    h, w, _ = rgb.shape
    passes = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
    raw = b""
    for x0, y0, dx, dy in passes:
        sub = rgb[y0::dy, x0::dx]
        if sub.size:
            raw += b"".join(b"\x00" + row.tobytes() for row in sub)
    return _png(struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 1), raw)


RGB = np.random.default_rng(0).integers(0, 256, (90, 60, 3), dtype=np.uint8)
FRAME2 = np.ascontiguousarray(RGB[::-1])


def _exif(orientation: int) -> Image.Exif:
    exif = Image.new("RGB", (1, 1)).getexif()
    exif[ORIENTATION] = orientation
    exif[0x010F] = "Apple"
    exif[0x0128] = 2
    exif[0x011A] = 72.0
    exif[0x011B] = 72.0
    return exif


def _ultra_hdr_like() -> bytes:
    """An MPO whose APP1 XMP names hdrgm (Ultra HDR): PIL keeps it a plain JPEG."""
    mpo = _save(Image.fromarray(RGB), "MPO", save_all=True, append_images=[Image.fromarray(FRAME2)])
    xmp = b'http://ns.adobe.com/xap/1.0/\x00<x:xmpmeta hdrgm:Version="1.0"/>'
    return mpo[:2] + b"\xff\xe1" + struct.pack(">H", len(xmp) + 2) + xmp + mpo[2:]


def _real_world_variants() -> dict[str, bytes]:
    im = Image.fromarray(RGB)
    f2 = Image.fromarray(FRAME2)
    jpeg = _save(im, "JPEG")
    png = _save(im, "PNG")
    return {
        "jpeg_baseline": jpeg,
        "jpeg_progressive": _save(im, "JPEG", progressive=True),
        "jpeg_cmyk": _save(im.convert("CMYK"), "JPEG"),
        "jpeg_grey": _save(im.convert("L"), "JPEG"),
        "jpeg_444": _save(im, "JPEG", subsampling=0, quality=95),
        "jpeg_optimized": _save(im, "JPEG", optimize=True),
        "jpeg_restart_markers": _save(im, "JPEG", restart_marker_blocks=1),
        "jpeg_icc_comment_dpi": _save(im, "JPEG", icc_profile=b"\x00" * 128, comment=b"hi", dpi=(300, 300)),
        "jpeg_exif6": _save(im, "JPEG", exif=_exif(6)),
        "jpeg_trailing_bytes": jpeg + b"\x00" * 512,
        "jpeg_cv2": cv2.imencode(".jpg", RGB[..., ::-1])[1].tobytes(),
        "mpo": _save(im, "MPO", save_all=True, append_images=[f2]),
        "mpo_exif6": _save(im, "MPO", save_all=True, append_images=[f2], exif=_exif(6)),
        "ultra_hdr_like": _ultra_hdr_like(),
        "png_rgb": png,
        "png_rgba": _save(im.convert("RGBA"), "PNG"),
        "png_la": _save(im.convert("LA"), "PNG"),
        "png_palette": _save(im.quantize(64), "PNG"),
        "png_palette_trns": _save(im.quantize(64), "PNG", transparency=3),
        "png_grey": _save(im.convert("L"), "PNG"),
        "png_1bit": _save(im.convert("1"), "PNG"),
        "png_grey16": _save(Image.fromarray(RGB[..., 0].astype(np.uint16) * 257), "PNG"),
        "png_rgb48": cv2.imencode(".png", RGB[..., ::-1].astype(np.uint16) * 257)[1].tobytes(),
        "png_rgba64": cv2.imencode(".png", np.dstack([RGB, RGB[..., :1]]).astype(np.uint16) * 257)[1].tobytes(),
        "png_interlaced": _interlaced_png(RGB),
        "png_exif6": _save(im, "PNG", exif=_exif(6)),
        "png_trailing_bytes": png + b"junk" * 16,
        "apng": _save(im, "PNG", save_all=True, append_images=[f2]),
        "apng_hidden_default": _save(im, "PNG", save_all=True, default_image=True, append_images=[f2]),
        "webp_lossy": _save(im, "WEBP"),
        "webp_lossless": _save(im, "WEBP", lossless=True),
        "webp_alpha": _save(im.convert("RGBA"), "WEBP"),
        "webp_alpha_lossless": _save(im.convert("RGBA"), "WEBP", lossless=True),
        "webp_animated": _save(im, "WEBP", save_all=True, append_images=[f2]),
        "webp_exif6_icc": _save(im, "WEBP", exif=_exif(6), icc_profile=b"\x00" * 128),
        "webp_cv2": cv2.imencode(".webp", RGB[..., ::-1])[1].tobytes(),
    }


VARIANTS = _real_world_variants()


# --------------------------------------------------------------------------- #
# The format restriction keeps every real-world JPEG / PNG / WebP variant
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_real_world_variants_open_as_an_upload_format(name):
    # MPO is what PIL's JPEG opener returns for a multi-picture JPEG.
    assert Image.open(io.BytesIO(VARIANTS[name])).format in (*UPLOAD_FORMATS, "MPO")


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_real_world_variants_decode_exactly_as_on_main(name):
    bgr, scale = decode_image(VARIANTS[name])
    ref, ref_scale = _main_decode(VARIANTS[name])
    assert scale == ref_scale
    np.testing.assert_array_equal(bgr, ref)


@pytest.mark.parametrize("name", ["jpeg_exif6", "mpo_exif6", "png_exif6", "webp_exif6_icc"])
def test_orientation_is_applied_in_every_upload_format(name):
    bgr, _ = decode_image(VARIANTS[name])
    assert bgr.shape[:2] == RGB.shape[1::-1]


def test_an_interlaced_png_decodes_to_the_same_pixels_as_a_plain_one():
    bgr, _ = decode_image(VARIANTS["png_interlaced"])
    np.testing.assert_array_equal(bgr, cv2.cvtColor(RGB, cv2.COLOR_RGB2BGR))


@pytest.mark.parametrize("name", ["mpo", "apng", "apng_hidden_default", "webp_animated"])
def test_multi_frame_uploads_are_graded_on_their_first_frame(name):
    bgr, _ = decode_image(VARIANTS[name])
    first = Image.open(io.BytesIO(VARIANTS[name])).convert("RGB")
    np.testing.assert_array_equal(bgr, cv2.cvtColor(np.array(first), cv2.COLOR_RGB2BGR))


def test_ultra_hdr_like_mpo_is_a_plain_jpeg_and_decodes():
    data = VARIANTS["ultra_hdr_like"]
    assert Image.open(io.BytesIO(data)).format == "JPEG"
    assert decode_image(data)[0].shape[:2] == RGB.shape[:2]


def test_upload_formats_are_pil_format_ids():
    # A typo ("JPG", "webp") would make Image.open silently skip that format.
    Image.init()
    assert set(UPLOAD_FORMATS) <= set(Image.OPEN)
    assert len(set(UPLOAD_FORMATS)) == len(UPLOAD_FORMATS) == 3


# --------------------------------------------------------------------------- #
# ... and rejects every other format PIL can write, as bytes
# --------------------------------------------------------------------------- #
def _other_format_bytes() -> dict[str, bytes]:
    """One valid image per PIL-writable format outside the upload formats that
    PIL itself can open again (AVIF, BLP, BMP, DDS, DIB, EPS, GIF, ICNS, ICO, IM,
    JPEG2000, MSP, PCX, PPM, QOI, SGI, SPIDER, TGA, TIFF, XBM with Pillow 12)."""
    Image.init()
    out = {}
    rgb = RGB[:32, :24]
    for fmt in sorted(Image.SAVE):
        if fmt in UPLOAD_FORMATS or fmt == "MPO":
            continue
        for mode in ("RGB", "RGBA", "L", "1", "P"):
            im = Image.fromarray(rgb).quantize(16) if mode == "P" else Image.fromarray(rgb).convert(mode)
            try:
                data = _save(im, fmt)
                if Image.open(io.BytesIO(data)).format != fmt:
                    continue
            except Exception:  # noqa: BLE001  - no save handler / can't read back
                continue
            out[fmt] = data
            break
    return out


OTHER = _other_format_bytes()


def test_the_other_formats_cover_what_this_pillow_writes():
    assert {"BMP", "GIF", "TIFF", "PPM", "ICO", "QOI", "SGI", "TGA", "PCX", "DDS", "JPEG2000"} <= set(OTHER)


@pytest.mark.parametrize("fmt", sorted(OTHER))
def test_every_other_pil_format_is_a_decode_error(fmt):
    with pytest.raises(ImageDecodeError) as info:
        decode_image(OTHER[fmt])
    assert isinstance(info.value.__cause__, Image.UnidentifiedImageError)


@pytest.mark.parametrize("fmt", sorted(OTHER))
def test_every_other_pil_format_still_decodes_as_a_pil_image(fmt):
    # A PIL input is the caller's choice: not format-checked.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im = Image.open(io.BytesIO(OTHER[fmt]))
            im.load()
    except OSError:  # EPS needs Ghostscript to rasterise
        pytest.skip(f"PIL can't load {fmt} here")
    bgr, _ = decode_image(Image.open(io.BytesIO(OTHER[fmt])))
    assert bgr.ndim == 3 and bgr.shape[2] == 3


@pytest.mark.parametrize("fmt", ["GIF", "TIFF", "BMP", "QOI"])
def test_fuzzed_bytes_of_other_formats_are_always_decode_errors(fmt):
    # Stronger than test_retest_reject_uploads_fix.py's fuzz, which only checks
    # nothing but ImageDecodeError escapes: here every input must be refused.
    rnd = random.Random(fmt)
    base = _save(Image.fromarray(RGB[:32, :24]), fmt)
    for _ in range(100):
        d = bytearray(base)
        for _ in range(rnd.randint(1, 8)):
            d[rnd.randrange(4, len(d))] = rnd.randrange(256)  # keep the signature
        with pytest.raises(ImageDecodeError):
            decode_image(bytes(d))


# --------------------------------------------------------------------------- #
# End to end: the declared content type is not the format
# --------------------------------------------------------------------------- #
def _as(fmt: str, jpeg: bytes, **kw) -> bytes:
    rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(jpeg, np.uint8), 1), cv2.COLOR_BGR2RGB)
    return _save(Image.fromarray(rgb), fmt, **kw)


@pytest.mark.parametrize("ctype", ["image/jpeg", "image/png", "image/webp"])
@pytest.mark.parametrize("fmt", ["PNG", "WEBP", "JPEG"])
def test_any_upload_format_under_any_allowed_content_type_is_graded(tmp_path, pair, monkeypatch, fmt, ctype):
    front, back = (_as(fmt, x, **({"lossless": True} if fmt == "WEBP" else {})) for x in pair)
    r = _post(_service(tmp_path, _guard(), _feats()), monkeypatch, front, back, ctype=ctype)
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("ctype", ["image/jpeg", "image/png", "image/webp"])
@pytest.mark.parametrize("fmt", ["GIF", "TIFF", "BMP", "JPEG2000"])
@pytest.mark.parametrize("face", ["front", "back"])
def test_another_format_under_any_allowed_content_type_is_a_400(tmp_path, pair, monkeypatch, fmt, ctype, face):
    bad = _as(fmt, pair[0])
    front, back = (bad, pair[1]) if face == "front" else (pair[0], bad)
    r = _post(_service(tmp_path, _guard(), _feats()), monkeypatch, front, back, ctype=ctype)
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == f"{face}: {UNREADABLE}"


def test_a_tiff_front_beats_a_gif_back_and_both_beat_card_not_found(tmp_path, pair, monkeypatch):
    blank = _save(Image.new("RGB", (300, 400), (128, 128, 128)), "PNG")
    svc = _service(tmp_path, _guard(), _feats())
    r = _post(svc, monkeypatch, _as("TIFF", pair[0]), _as("GIF", pair[1]))
    assert r.status_code == 400 and r.json()["detail"] == f"front: {UNREADABLE}"
    r = _post(svc, monkeypatch, blank, _as("GIF", pair[1]), ctype="image/png", back_ctype="image/jpeg")
    assert r.status_code == 400 and r.json()["detail"] == f"back: {UNREADABLE}"


def test_an_mpo_phone_photo_is_graded_end_to_end(tmp_path, pair, monkeypatch):
    rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(pair[0], np.uint8), 1), cv2.COLOR_BGR2RGB)
    mpo = _save(Image.fromarray(rgb), "MPO", save_all=True, append_images=[Image.fromarray(rgb[::-1].copy())])
    assert Image.open(io.BytesIO(mpo)).format == "MPO"
    assert detect_and_crop(mpo).found_quad
    r = _post(_service(tmp_path, _guard(), _feats()), monkeypatch, mpo, pair[1])
    assert r.status_code == 200, r.text


# --------------------------------------------------------------------------- #
# D5 (fixed): a palette PNG with tRNS but no PLTE was a 500, now a 400
# --------------------------------------------------------------------------- #
# PNG colour type 3 requires PLTE. Without it PIL opens the file as mode "P"
# with `palette` None; with a one-entry tRNS, `info["transparency"]` is an int,
# and `Image.convert("RGB")` takes the single-transparent-colour branch, which
# does `assert self.palette is not None` (AssertionError; under `python -O`,
# AttributeError on `None.mode`). Neither is in `_DECODE_ERRORS`, so it left
# `decode_image` unwrapped and `/api/grade` returned 500. Found by a header-only
# fuzz (PLTE chunk type garbled, CRC fixed); same on main. Fixed: `_decode_image`
# refuses a palette image with no palette as an `ImageDecodeError`.
def _palette_png(plte: bool, trns: bytes | None) -> bytes:
    w = h = 8
    raw = b"".join(b"\x00" + bytes(x % 4 for x in range(w)) for _ in range(h))
    extra = []
    if plte:
        extra.append(_chunk(b"PLTE", bytes([0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255])))
    if trns is not None:
        extra.append(_chunk(b"tRNS", trns))
    return _png(struct.pack(">IIBBBBB", w, h, 8, 3, 0, 0, 0), raw, *extra)


D5 = _palette_png(plte=False, trns=b"\x00")


def test_d5_input_is_a_palette_png_without_plte_and_one_transparent_index():
    im = Image.open(io.BytesIO(D5))
    im.load()  # the pixels decode
    assert (im.format, im.mode, im.palette, im.info["transparency"]) == ("PNG", "P", None, 0)
    with pytest.raises((AssertionError, AttributeError)):
        im.convert("RGB")


@pytest.mark.parametrize(
    "plte, trns",
    [(True, b"\x00"), (True, None), (True, b"\x00\x80\xff"), (False, None), (False, b"\x00\x80\xff")],
    ids=["plte_trns1", "plte", "plte_trns3", "no_plte", "no_plte_trns3"],
)
def test_d5_neighbours_decode_or_are_decode_errors(plte, trns):
    try:
        bgr, _ = decode_image(_palette_png(plte, trns))
    except ImageDecodeError:
        return
    assert bgr.shape == (8, 8, 3)


def test_d5_palette_png_without_plte_is_a_decode_error():
    with pytest.raises(ImageDecodeError):
        decode_image(D5)


@pytest.mark.parametrize("face", ["front", "back"])
def test_d5_palette_png_without_plte_is_a_400_naming_the_face(tmp_path, pair, monkeypatch, face):
    front, back = (D5, pair[1]) if face == "front" else (pair[0], D5)
    r = _post(_service(tmp_path, _guard(), _feats()), monkeypatch, front, back, ctype="image/png")
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == f"{face}: {UNREADABLE}"


# --------------------------------------------------------------------------- #
# PNG colour type x bit depth x PLTE x tRNS x interlace matrix
# --------------------------------------------------------------------------- #
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_DEPTHS = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
_PLTES = {"none": None, "ok": bytes(range(12)), "one": b"\x01\x02\x03", "odd": b"\x01\x02",
          "empty": b"", "big": bytes(800)}
_TRNS = {"none": None, "1": b"\x00", "2": b"\x00\x01", "3": b"\x00\x01\x02", "6": bytes(6),
         "8": bytes(8), "300": bytes(300), "empty": b""}
_ADAM7 = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]


def _matrix_png(ct, bd, plte, trns, interlace, rnd, w=9, h=7) -> bytes:
    ch = _CHANNELS[ct]
    windows = _ADAM7 if interlace else [(0, 0, 1, 1)]
    raw = b""
    for x0, y0, dx, dy in windows:
        pw, ph = len(range(x0, w, dx)), len(range(y0, h, dy))
        if pw and ph:
            rb = (pw * ch * bd + 7) // 8
            raw += b"".join(b"\x00" + bytes(rnd.randrange(2) for _ in range(rb)) for _ in range(ph))
    extra = [_chunk(b"PLTE", plte)] if plte is not None else []
    extra += [_chunk(b"tRNS", trns)] if trns is not None else []
    return _png(struct.pack(">IIBBBBB", w, h, bd, ct, 0, 0, interlace), raw, *extra)


def test_png_header_matrix_decodes_or_is_a_decode_error():
    # 1,440 PNGs; before the D5 fix the only other outcome was D5 (colour type 3,
    # no PLTE, one-byte tRNS).
    rnd = random.Random(0)
    other = []
    for ct, (pk, plte), (tk, trns), il in itertools.product(_CHANNELS, _PLTES.items(), _TRNS.items(), (0, 1)):
        for bd in _DEPTHS[ct]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    decode_image(_matrix_png(ct, bd, plte, trns, il, rnd))
            except ImageDecodeError:
                pass
            except Exception as exc:  # noqa: BLE001
                other.append((ct, bd, pk, tk, il, repr(exc)))
    assert other == []


# --------------------------------------------------------------------------- #
# Header-only fuzz of the upload formats (pixels mostly intact, PNG CRCs fixed)
# --------------------------------------------------------------------------- #
def _fix_crcs(d: bytearray) -> None:
    i = 8
    while i + 12 <= len(d):
        n = struct.unpack(">I", d[i : i + 4])[0]
        if i + 12 + n > len(d):
            return
        d[i + 8 + n : i + 12 + n] = struct.pack(">I", zlib.crc32(bytes(d[i + 4 : i + 8 + n])) & 0xFFFFFFFF)
        i += 12 + n


def _header_regions(d: bytes) -> list[tuple[int, int]]:
    if d[:4] == b"\x89PNG":  # every chunk but IDAT bodies
        out, i = [], 8
        while i + 12 <= len(d):
            n = struct.unpack(">I", d[i : i + 4])[0]
            out.append((i, i + 8 if d[i + 4 : i + 8] == b"IDAT" else i + 12 + n))
            i += 12 + n
        return out
    if d[:2] == b"\xff\xd8":  # everything before the first SOS
        return [(2, d.find(b"\xff\xda"))]
    out, i = [(0, 30)], 12  # WebP: RIFF header, chunk headers, metadata chunks
    while i + 8 <= len(d):
        n = struct.unpack("<I", d[i + 4 : i + 8])[0]
        meta = d[i : i + 4] in (b"VP8X", b"EXIF", b"ICCP", b"ANIM")
        out.append((i, i + 8 + (n if meta else 0)))
        i += 8 + n + (n & 1)
    return out


FUZZ_NAMES = ["jpeg_progressive", "jpeg_cmyk", "jpeg_exif6", "mpo_exif6", "png_rgb", "png_palette_trns",
              "png_interlaced", "png_exif6", "apng", "png_rgb48", "webp_lossless", "webp_alpha",
              "webp_animated", "webp_exif6_icc"]


@pytest.mark.parametrize("name", FUZZ_NAMES)
def test_header_fuzz_of_upload_formats_decodes_or_is_a_decode_error(monkeypatch, name):
    # Scratch runs: 192,000 such inputs (16 bases x 12,000) plus 96,000 whole-file
    # fuzzes; the only other outcome was D5 (8 inputs, all a garbled PLTE type in
    # a palette PNG with a one-entry tRNS), now fixed. This keeps a seeded slice.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 20_000_000)  # a fuzzed size can't eat RAM
    base = VARIANTS[name]
    regions = [(a, b) for a, b in _header_regions(base) if b > a]
    rnd = random.Random(name)
    other = []
    for _ in range(150):
        d = bytearray(base)
        for _ in range(rnd.randint(1, 6)):
            a, b = rnd.choice(regions)
            i = rnd.randrange(a, b)
            if rnd.random() < 0.7:
                d[i] = rnd.randrange(256)
            else:
                d[i : i + 4] = rnd.choice([b"\xff\xff\xff\xff", b"\x00\x00\x00\x00", b"\x7f\xff\xff\xff"])
        if base[:4] == b"\x89PNG":
            _fix_crcs(d)
        data = bytes(d)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                decode_image(data)
        except ImageDecodeError:
            pass
        except Exception as exc:  # noqa: BLE001
            other.append(repr(exc))
    assert other == []
