"""Third re-test of intent/reject-unusable-uploads, after the D2/N1 fix
(`preprocess._apply_exif_orientation`): a readable orientation is applied even
when `ImageOps.exif_transpose` fails, any other EXIF failure means upright, and
only the pixel decode maps PIL errors to `ImageDecodeError`.

Checks the fallback transpose is pixel-identical to `exif_transpose` across
formats and modes, that the broad EXIF catch hides neither a pixel failure nor
an error that should surface, EXIF-only fuzzing (pixels intact), and the 400/422
rules end to end. The confirmed defects were strict xfails; after the owner's
2026-10-28 decision D4 is fixed (bytes decode only as JPEG/PNG/WebP) and D3 is
an accepted 400, and the tests pin both.

Synthetic images, stub services/guards and tmp_path only; nothing under data/
is read or written.
"""

from __future__ import annotations

import io
import struct
import warnings
import zlib

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageOps, TiffImagePlugin
from PIL import __version__ as pillow_version

from backend.app.ml.inference import GradeInferenceService
from backend.app.ml.preprocess import ImageDecodeError, decode_image, detect_and_crop
from backend.tests.helpers.synth import make_scene, to_jpeg_bytes
from backend.tests.test_face_guard import _feats, _guard, _service
from backend.tests.test_face_guard import pair as _pair_fixture

pair = _pair_fixture

ORIENTATION = 0x0112
UNREADABLE = "not a readable image (corrupt or truncated file)"
CARD_NOT_FOUND = "card not found; retake the photo on a plain background"


def _post(svc, monkeypatch, front: bytes, back: bytes, ctype: str = "image/jpeg"):
    """POST through the real app; server errors come back as 500 instead of raising."""
    import backend.app.api.grade as grade_mod
    import backend.app.main as main_mod

    monkeypatch.setattr(grade_mod, "get_inference_service", lambda: svc)
    monkeypatch.setattr(main_mod, "get_inference_service", lambda: svc)
    files = {"front": ("f", front, ctype), "back": ("b", back, ctype)}
    with TestClient(main_mod.create_app(), raise_server_exceptions=False) as c:
        return c.post("/api/grade", files=files)


def _save(im: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Pillow deprecates saving some modes
        im.save(buf, fmt, **kw)
    return buf.getvalue()


def _bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


@pytest.fixture(scope="module")
def portrait_rgb() -> np.ndarray:
    """Portrait, under MAX_LONG_EDGE, with a white top band: rotation shows."""
    rgb = np.ascontiguousarray(make_scene(seed=0).bgr[..., ::-1][:300, :200])
    rgb[:12] = 255
    return rgb


@pytest.fixture(scope="module")
def blank() -> bytes:
    """No card in the photo: detection falls back to the padded canvas."""
    return to_jpeg_bytes(np.full((1000, 750, 3), 128, np.uint8))


def _exif_entry(raw: bytearray, tag: int, start: int) -> int:
    """Offset of `tag`'s 12-byte entry in IFD0 of big-endian TIFF bytes at `start`."""
    assert raw[start : start + 4] == b"MM\x00*"
    ifd = start + struct.unpack(">I", raw[start + 4 : start + 8])[0]
    for k in range(struct.unpack(">H", raw[ifd : ifd + 2])[0]):
        e = ifd + 2 + 12 * k
        if struct.unpack(">H", raw[e : e + 2])[0] == tag:
            return e
    raise KeyError(tag)


# --------------------------------------------------------------------------- #
# D3: a JPEG whose EXIF XResolution PIL can't index is a 400, not graded
# --------------------------------------------------------------------------- #
def _bad_xresolution_exif(kind: str, orientation: int = 6) -> bytes:
    """EXIF with a valid orientation, ResolutionUnit, and an XResolution PIL reads
    as a one-element sequence: "ascii" is a one-digit string ("7"), "undefined" a
    single byte. PIL's JPEG `_open` reads dpi from it (`_read_dpi_from_exif`):
    `float(x[0]) / x[1]` raises IndexError, which it doesn't catch, so
    `Image.open` takes the file for "not a JPEG" (UnidentifiedImageError)."""
    exif = Image.Exif()
    exif[0x0110] = "7"  # Model: ASCII, stored inline
    exif[ORIENTATION] = orientation
    exif[0x0128] = 2  # ResolutionUnit: inch
    raw = bytearray(exif.tobytes())
    assert raw[:6] == b"Exif\x00\x00"
    e = _exif_entry(raw, 0x0110, 6)
    raw[e : e + 2] = struct.pack(">H", 0x011A)  # Model -> XResolution
    if kind == "undefined":
        raw[e + 2 : e + 4] = struct.pack(">H", 7)  # UNDEFINED
        raw[e + 4 : e + 8] = struct.pack(">I", 1)  # one byte: "7"
    return bytes(raw)


def _with_raw_exif(rgb: np.ndarray, fmt: str, exif: bytes) -> bytes:
    opts = {"lossless": True} if fmt == "WEBP" else {"quality": 95} if fmt == "JPEG" else {}
    return _save(Image.fromarray(rgb), fmt, exif=exif, **opts)


def _clean_upright(rgb: np.ndarray, orientation: int) -> np.ndarray:
    """`exif_transpose` of `rgb` under a clean orientation tag, as BGR."""
    exif = Image.Exif()
    exif[ORIENTATION] = orientation
    png = _save(Image.fromarray(rgb), "PNG", exif=exif.tobytes())
    return _bgr(np.array(ImageOps.exif_transpose(Image.open(io.BytesIO(png))).convert("RGB")))


def _strip_app1(jpeg: bytes) -> bytes:
    i = jpeg.find(b"\xff\xe1")
    n = struct.unpack(">H", jpeg[i + 2 : i + 4])[0]
    return jpeg[:i] + jpeg[i + 2 + n :]


@pytest.mark.parametrize("kind", ["ascii", "undefined"])
def test_d3_input_has_intact_pixels_and_only_its_exif_is_malformed(portrait_rgb, kind):
    # Pins the input of the D3 tests below: without the APP1 segment it decodes,
    # and PIL's own EXIF reader parses the block (orientation 6).
    data = _with_raw_exif(portrait_rgb, "JPEG", _bad_xresolution_exif(kind))
    bgr, _ = decode_image(_strip_app1(data))
    assert bgr.shape[:2] == portrait_rgb.shape[:2]
    exif = Image.Exif()
    exif.load(_bad_xresolution_exif(kind))
    assert exif[ORIENTATION] == 6
    assert len(exif[0x011A]) == 1
    with pytest.raises(Image.UnidentifiedImageError):
        Image.open(io.BytesIO(data))


@pytest.mark.parametrize("fmt", ["PNG", "WEBP"])
@pytest.mark.parametrize("kind", ["ascii", "undefined"])
def test_the_same_exif_in_png_and_webp_is_graded_with_its_orientation(portrait_rgb, fmt, kind):
    data = _with_raw_exif(portrait_rgb, fmt, _bad_xresolution_exif(kind))
    bgr, _ = decode_image(data)
    np.testing.assert_array_equal(bgr, _clean_upright(portrait_rgb, 6))


# Changed 2026-10-28 (D3, owner decision): these were strict xfails expecting the
# photo to be graded. The owner accepted the 400 instead (rare: 0.34% of an
# EXIF-only JPEG fuzz; no EXIF strip-and-retry), so they now pin it. PIL's JPEG
# open reads dpi from EXIF and lets IndexError escape for a one-element
# XResolution; Image.open then reports UnidentifiedImageError.
@pytest.mark.parametrize("kind", ["ascii", "undefined"])
def test_d3_jpeg_with_an_unindexable_xresolution_is_a_decode_error(portrait_rgb, kind):
    data = _with_raw_exif(portrait_rgb, "JPEG", _bad_xresolution_exif(kind))
    with pytest.raises(ImageDecodeError) as info:
        decode_image(data)
    assert isinstance(info.value.__cause__, Image.UnidentifiedImageError)


@pytest.mark.parametrize("face", ["front", "back"])
def test_d3_jpeg_with_an_unindexable_xresolution_is_a_400_naming_the_face(
    tmp_path, pair, monkeypatch, face
):
    rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(pair[0 if face == "front" else 1], np.uint8), 1),
                       cv2.COLOR_BGR2RGB)
    bad = _with_raw_exif(rgb, "JPEG", _bad_xresolution_exif("ascii", orientation=1))
    front, back = (bad, pair[1]) if face == "front" else (pair[0], bad)
    r = _post(_service(tmp_path, _guard(), _feats()), monkeypatch, front, back)
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == f"{face}: {UNREADABLE}"


def test_d3_is_never_a_500(tmp_path, pair, monkeypatch):
    # Whatever the fix, it stays a 4xx-or-200, never a server error.
    rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(pair[0], np.uint8), 1), cv2.COLOR_BGR2RGB)
    bad = _with_raw_exif(rgb, "JPEG", _bad_xresolution_exif("ascii", orientation=1))
    r = _post(_service(tmp_path, _guard(), _feats()), monkeypatch, bad, pair[1])
    assert r.status_code in (200, 400), r.text


def _malformed_mp_index_jpeg(rgb: np.ndarray) -> bytes:
    """A JPEG whose APP2 MP index claims 2 images but holds one 16-byte entry:
    PIL's `_getmp` raises struct.error, which `jpeg_factory` doesn't catch."""
    ifd = TiffImagePlugin.ImageFileDirectory_v2(prefix=b"MM")
    ifd[0xB000] = b"0100"
    ifd.tagtype[0xB000] = 7
    ifd[0xB001] = 2
    ifd.tagtype[0xB001] = 4
    ifd[0xB002] = struct.pack(">LLLHH", 0x20030000, 1000, 0, 0, 0)
    ifd.tagtype[0xB002] = 7
    payload = b"MPF\x00" + b"MM\x00*\x00\x00\x00\x08" + ifd.tobytes(8)
    app2 = b"\xff\xe2" + struct.pack(">H", len(payload) + 2) + payload
    return _save(Image.fromarray(rgb), "JPEG", extra=app2)


def test_a_malformed_mp_index_is_a_400_or_a_grade_never_a_500(tmp_path, pair, monkeypatch):
    # Nit (not EXIF, so outside the spec's EXIF rule): today it's a 400.
    rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(pair[1], np.uint8), 1), cv2.COLOR_BGR2RGB)
    bad = _malformed_mp_index_jpeg(rgb)
    r = _post(_service(tmp_path, _guard(), _feats()), monkeypatch, pair[0], bad)
    assert r.status_code in (200, 400), r.text
    if r.status_code == 400:
        assert r.json()["detail"] == f"back: {UNREADABLE}"


# --------------------------------------------------------------------------- #
# D4: TIFF bytes (sent as image/jpeg) whose IFD PIL can't use gave a 500; now 400
# --------------------------------------------------------------------------- #
# PIL's TIFF `load()` reads the EXIF sub-IFDs and applies the orientation itself
# (`TiffImageFile.load_end` -> `get_ifd`, `exif_transpose(in_place=True)`), so
# for TIFF the EXIF step runs inside the pixel decode, which wraps only
# `_DECODE_ERRORS`. Scratch fuzz of IFD0 entries (10,000 TIFFs): 134 KeyError
# and 2 OverflowError, each a 500 through /api/grade.
def _tiff_entry(d: bytearray, tag: int) -> tuple[str, int]:
    end = "<" if d[:2] == b"II" else ">"
    off = struct.unpack(end + "I", d[4:8])[0]
    for k in range(struct.unpack(end + "H", d[off : off + 2])[0]):
        e = off + 2 + 12 * k
        if struct.unpack(end + "H", d[e : e + 2])[0] == tag:
            return end, e
    raise KeyError(tag)


def _tiff(rgb: np.ndarray) -> bytearray:
    im = Image.fromarray(rgb)
    exif = im.getexif()
    exif[ORIENTATION] = 6
    exif[0x0131] = "PokeScan"  # Software
    return bytearray(_save(im, "TIFF", exif=exif))


def _tiff_interop_pointer_in_ifd0(rgb: np.ndarray) -> bytes:
    """IFD0's Software tag renamed to the Interoperability IFD pointer (0xA005),
    which belongs in the Exif IFD: `Exif.get_ifd(0xA005)` raises KeyError."""
    d = _tiff(rgb)
    end, e = _tiff_entry(d, 0x0131)
    d[e : e + 2] = struct.pack(end + "H", 0xA005)
    return bytes(d)


def _tiff_strip_offset_past_ssize_t(rgb: np.ndarray) -> bytes:
    """StripOffsets retyped LONG8 pointing at 0xFFFF_FFFF_FFFF_FFFF:
    `ImageFile.load` -> `seek(offset)` raises OverflowError."""
    d = _tiff(rgb)
    end, e = _tiff_entry(d, 0x0111)
    n = len(d)
    d += b"\xff" * 8
    d[e + 2 : e + 4] = struct.pack(end + "H", 16)
    d[e + 4 : e + 8] = struct.pack(end + "I", 1)
    d[e + 8 : e + 12] = struct.pack(end + "I", n)
    return bytes(d)


TIFF_CASES = {
    "interop_pointer_in_ifd0": (_tiff_interop_pointer_in_ifd0, KeyError),
    "strip_offset_past_ssize_t": (_tiff_strip_offset_past_ssize_t, OverflowError),
}


def _pil_load_error(data: bytes) -> BaseException | None:
    """What this Pillow's own TIFF `load()` raises on `data`, or None if it loads."""
    try:
        Image.open(io.BytesIO(data)).load()
    except Exception as exc:  # noqa: BLE001
        return exc
    return None


@pytest.mark.parametrize("case", sorted(TIFF_CASES))
def test_d4_inputs_raise_what_pil_raises(portrait_rgb, case):
    # Pins the inputs of the D4 tests below (and the PIL errors the safety net
    # catches). Found on Pillow 12.1; Pillow 13 loads the interop-pointer TIFF
    # without error, so a Pillow that no longer raises skips rather than fails.
    make, exc = TIFF_CASES[case]
    data = make(portrait_rgb)
    assert Image.open(io.BytesIO(data)).format == "TIFF"
    err = _pil_load_error(data)
    if err is None:
        pytest.skip(f"Pillow {pillow_version} loads the {case} TIFF without error")
    assert isinstance(err, exc)


# Fixed 2026-10-28 (D4, owner-approved): bytes decode only as JPEG/PNG/WebP, so
# TIFF bytes never reach PIL's TIFF loader (a decode error, 400), and KeyError /
# OverflowError joined `_DECODE_ERRORS` as a safety net. These were strict xfails;
# within their own "decode error or decodes" / "never a 500" contract they now
# pin the outcome: ImageDecodeError and 400.
@pytest.mark.parametrize("case", sorted(TIFF_CASES))
def test_d4_tiff_with_an_unusable_ifd_is_a_decode_error_or_decodes(portrait_rgb, case):
    data = TIFF_CASES[case][0](portrait_rgb)
    with pytest.raises(ImageDecodeError) as info:
        decode_image(data)
    assert isinstance(info.value.__cause__, Image.UnidentifiedImageError)


@pytest.mark.parametrize("case", sorted(TIFF_CASES))
def test_d4_tiff_with_an_unusable_ifd_opened_by_the_caller_is_a_decode_error(portrait_rgb, case):
    # A PIL input skips the format check, so PIL's own KeyError / OverflowError
    # from load() is what the widened `_DECODE_ERRORS` must catch. On a Pillow
    # that loads the file cleanly (Pillow 13, interop pointer), it must decode.
    make, exc = TIFF_CASES[case]
    data = make(portrait_rgb)
    err = _pil_load_error(data)
    if err is None:
        bgr, _ = decode_image(Image.open(io.BytesIO(data)))
        assert bgr.ndim == 3
        return
    assert isinstance(err, exc)
    with pytest.raises(ImageDecodeError) as info:
        decode_image(Image.open(io.BytesIO(data)))
    assert isinstance(info.value.__cause__, exc)


@pytest.mark.parametrize("face", ["front", "back"])
@pytest.mark.parametrize("case", sorted(TIFF_CASES))
def test_d4_tiff_with_an_unusable_ifd_is_never_a_500(tmp_path, pair, monkeypatch, case, face):
    rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(pair[0], np.uint8), 1), cv2.COLOR_BGR2RGB)
    bad = TIFF_CASES[case][0](rgb)
    front, back = (bad, pair[1]) if face == "front" else (pair[0], bad)
    r = _post(_service(tmp_path, _guard(), _feats()), monkeypatch, front, back)
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == f"{face}: {UNREADABLE}"


# --------------------------------------------------------------------------- #
# The fallback transpose is pixel-identical to exif_transpose, any format/mode
# --------------------------------------------------------------------------- #
def _mode_image(mode: str, rgb: np.ndarray) -> Image.Image:
    im = Image.fromarray(rgb)
    if mode == "I;16":
        g = (rgb[..., 0].astype(np.uint16) * 257) ^ np.arange(rgb.shape[1], dtype=np.uint16)
        return Image.fromarray(g)  # mode I;16
    if mode == "F":
        return Image.fromarray(rgb[..., 1].astype(np.float32) / 7.0)
    if mode == "RGBA":
        a = np.linspace(0, 255, rgb.shape[0], dtype=np.uint8)[:, None].repeat(rgb.shape[1], 1)
        return Image.fromarray(np.dstack([rgb, a]))
    if mode == "P":
        return im.quantize(16)
    return im.convert(mode)


FORMAT_MODES = [
    ("PNG", "RGBA"), ("PNG", "P"), ("PNG", "LA"), ("PNG", "L"), ("PNG", "I;16"), ("PNG", "1"),
    ("WEBP", "RGBA"), ("WEBP", "RGB"),
    ("TIFF", "RGBA"), ("TIFF", "CMYK"), ("TIFF", "I;16"), ("TIFF", "F"), ("TIFF", "P"),
    ("JPEG", "L"), ("JPEG", "CMYK"), ("JPEG", "RGB"),
]


def _oriented(rgb, fmt, mode, orientation) -> bytes:
    im = _mode_image(mode, rgb)
    assert im.mode == mode
    exif = im.getexif()
    exif[ORIENTATION] = orientation
    opts = {"lossless": True} if fmt == "WEBP" else {"quality": 95} if fmt == "JPEG" else {}
    if fmt == "PNG" and mode == "P":
        opts["transparency"] = 3
    data = _save(im, fmt, exif=exif.tobytes(), **opts)
    reread = Image.open(io.BytesIO(data))
    assert reread.mode == mode and reread.getexif().get(ORIENTATION) == orientation
    return data


@pytest.mark.parametrize("fmt,mode", FORMAT_MODES, ids=[f"{f}-{m}" for f, m in FORMAT_MODES])
@pytest.mark.parametrize("orientation", range(2, 9))
def test_forced_fallback_is_pixel_identical_to_exif_transpose(
    monkeypatch, portrait_rgb, fmt, mode, orientation
):
    data = _oriented(portrait_rgb, fmt, mode, orientation)

    # Changed 2026-10-28 (D4, owner-approved): TIFF bytes are a decode error now,
    # so TIFF goes in as a freshly opened PIL image (not format-checked).
    def source():
        return Image.open(io.BytesIO(data)) if fmt == "TIFF" else data

    expected, _ = decode_image(source())  # real exif_transpose
    calls = []
    real = ImageOps.exif_transpose

    def failing(image, *, in_place=False):
        if in_place:  # PIL's TIFF load() applies the orientation itself this way
            return real(image, in_place=True)
        calls.append(1)
        raise AttributeError("re-serialising EXIF failed")

    monkeypatch.setattr(ImageOps, "exif_transpose", failing)
    bgr, scale = decode_image(source())
    # TIFF: load() already turned it upright and dropped the tag, so the EXIF
    # step has nothing to do (and must not rotate twice).
    assert calls == ([] if fmt == "TIFF" else [1]) and scale == 1.0
    np.testing.assert_array_equal(bgr, expected)
    if orientation >= 5:
        assert bgr.shape[:2] == portrait_rgb.shape[1::-1]


PNG_MODES = ["RGBA", "P", "LA", "I;16", "L"]


@pytest.mark.parametrize("mode", PNG_MODES)
@pytest.mark.parametrize("orientation", range(2, 9))
def test_natural_fallback_matches_clean_exif_in_every_png_mode(portrait_rgb, mode, orientation):
    # XResolution renamed to InkNames (ASCII): exif_transpose raises for real.
    im = _mode_image(mode, portrait_rgb)
    exif = im.getexif()
    exif[ORIENTATION] = orientation
    exif[0x011A] = TiffImagePlugin.IFDRational(72, 1)
    raw = bytearray(exif.tobytes())
    e = _exif_entry(raw, 0x011A, 6)
    raw[e : e + 2] = struct.pack(">H", 0x014D)
    opts = {"transparency": 3} if mode == "P" else {}
    broken = _save(im, "PNG", exif=bytes(raw), **opts)
    with pytest.raises(AttributeError):
        ImageOps.exif_transpose(Image.open(io.BytesIO(broken)))
    clean_exif = Image.Exif()
    clean_exif[ORIENTATION] = orientation
    clean = _save(im, "PNG", exif=clean_exif.tobytes(), **opts)
    np.testing.assert_array_equal(decode_image(broken)[0], decode_image(clean)[0])


def test_fallback_on_a_photo_over_the_size_cap_resizes_after_rotating(monkeypatch):
    # Orientation 6 on a 1400x900 landscape: upright it's 900x1400, then capped
    # to 658x1024, the same with and without the fallback.
    rgb = np.ascontiguousarray(np.tile(make_scene(seed=3).bgr[..., ::-1], (2, 2, 1))[:900, :1400])
    assert rgb.shape[:2] == (900, 1400)
    im = Image.fromarray(rgb)
    exif = im.getexif()
    exif[ORIENTATION] = 6
    data = _save(im, "JPEG", exif=exif.tobytes(), quality=95)
    expected, s0 = decode_image(data)
    monkeypatch.setattr(ImageOps, "exif_transpose", lambda *a, **k: (_ for _ in ()).throw(TypeError()))
    bgr, s1 = decode_image(data)
    assert bgr.shape[:2] == (1024, 658) and s0 == s1
    np.testing.assert_array_equal(bgr, expected)


# --------------------------------------------------------------------------- #
# What the broad EXIF catch must not hide
# --------------------------------------------------------------------------- #
def _jpeg6(rgb) -> bytes:
    im = Image.fromarray(rgb)
    exif = im.getexif()
    exif[ORIENTATION] = 6
    return _save(im, "JPEG", exif=exif.tobytes(), quality=95)


def test_an_out_of_memory_fallback_transpose_surfaces(monkeypatch, portrait_rgb):
    # exif_transpose fails, then the direct transpose runs out of memory: that's
    # a server error (MemoryError), not a 400 and not an unrotated grade.
    data = _jpeg6(portrait_rgb)
    monkeypatch.setattr(ImageOps, "exif_transpose", lambda *a, **k: (_ for _ in ()).throw(MemoryError()))

    def transpose(self, method):
        raise MemoryError

    monkeypatch.setattr(Image.Image, "transpose", transpose)
    with pytest.raises(MemoryError):
        decode_image(data)


def test_a_pixel_error_in_the_fallback_transpose_is_a_decode_error(monkeypatch, portrait_rgb):
    # The direct transpose sits inside the wrapped decode, outside the broad catch.
    data = _jpeg6(portrait_rgb)
    monkeypatch.setattr(ImageOps, "exif_transpose", lambda *a, **k: (_ for _ in ()).throw(KeyError(1)))

    def transpose(self, method):
        raise OSError("pixel failure")

    monkeypatch.setattr(Image.Image, "transpose", transpose)
    with pytest.raises(ImageDecodeError) as info:
        decode_image(data)
    assert isinstance(info.value.__cause__, OSError)


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit])
def test_base_exceptions_in_the_exif_step_are_not_swallowed(portrait_rgb, exc):
    im = Image.open(io.BytesIO(_jpeg6(portrait_rgb)))

    def getexif():
        raise exc

    im.getexif = getexif
    with pytest.raises(exc):
        decode_image(im)


@pytest.mark.parametrize("value", [6.0, TiffImagePlugin.IFDRational(6, 1)], ids=["float", "rational"])
def test_a_numeric_orientation_equal_to_6_is_applied_like_exif_transpose(portrait_rgb, value):
    data = _jpeg6(portrait_rgb)
    expected, _ = decode_image(data)
    im = Image.open(io.BytesIO(data))
    real = im.getexif()
    real[ORIENTATION] = value
    im.getexif = lambda: real
    bgr, _ = decode_image(im)
    np.testing.assert_array_equal(bgr, expected)


@pytest.mark.parametrize("fail", [False, True], ids=["exif_transpose", "fallback"])
def test_a_pil_input_is_left_as_given_and_decodes_the_same_twice(monkeypatch, portrait_rgb, fail):
    # exif_transpose strips the orientation from its *copy*; neither path may
    # change the caller's image, so a second decode rotates again.
    if fail:
        monkeypatch.setattr(ImageOps, "exif_transpose", lambda *a, **k: (_ for _ in ()).throw(ValueError()))
    im = Image.open(io.BytesIO(_jpeg6(portrait_rgb)))
    info_exif = im.info["exif"]
    first, _ = decode_image(im)
    second, _ = decode_image(im)
    np.testing.assert_array_equal(first, second)
    assert first.shape[:2] == portrait_rgb.shape[1::-1]
    assert im.size == portrait_rgb.shape[1::-1] and im.info["exif"] == info_exif
    assert im.getexif().get(ORIENTATION) == 6


# --------------------------------------------------------------------------- #
# EXIF-only fuzz (pixels intact): never a 500-class error; PNG/WebP always graded
# --------------------------------------------------------------------------- #
def _phone_exif(orientation: int) -> bytes:
    exif = Image.Exif()
    exif[0x010F] = "Apple"
    exif[0x0110] = "iPhone 15"
    exif[ORIENTATION] = orientation
    exif[0x011A] = TiffImagePlugin.IFDRational(72, 1)
    exif[0x011B] = TiffImagePlugin.IFDRational(72, 1)
    exif[0x0128] = 2
    exif[0x0132] = "2026:10:28 12:00:00"
    ifd = exif.get_ifd(0x8769)
    ifd[0x829A] = TiffImagePlugin.IFDRational(1, 60)
    ifd[0x8827] = 100
    ifd[0x9003] = "2026:10:28 12:00:00"
    return exif.tobytes()


def _exif_span(fmt: str, data: bytes) -> tuple[int, int]:
    """(start, end) of the TIFF bytes of the EXIF block in `data`."""
    if fmt == "JPEG":
        i = data.find(b"Exif\x00\x00")
        return i + 6, i - 2 + struct.unpack(">H", data[i - 2 : i])[0]
    if fmt == "PNG":
        i = data.find(b"eXIf")
        return i + 4, i + 4 + struct.unpack(">I", data[i - 4 : i])[0]
    i = data.find(b"EXIF")
    s = i + 8 + (6 if data[i + 8 : i + 14] == b"Exif\x00\x00" else 0)
    return s, i + 8 + struct.unpack("<I", data[i + 4 : i + 8])[0]


def _mutate(raw: bytearray, rng) -> None:
    """Rewrite entry tag ids / types / counts, or random bytes; never the length."""
    n = struct.unpack(">H", raw[8:10])[0]
    for _ in range(int(rng.integers(1, 5))):
        e = 10 + 12 * int(rng.integers(max(n, 1)))
        if rng.random() < 0.6 and e + 12 <= len(raw):
            field = int(rng.integers(3))
            if field == 0:
                tag = int(rng.choice([0x011A, 0x011B, 0x0128, 0x0112, 0x014D, 0x011F, 0x8769,
                                      0x8825, 0xA005, 0x927C, int(rng.integers(65536))]))
                raw[e : e + 2] = struct.pack(">H", tag)
            elif field == 1:
                raw[e + 2 : e + 4] = struct.pack(">H", int(rng.integers(0, 20)))
            else:
                raw[e + 4 : e + 8] = struct.pack(">I", int(rng.choice([0, 1, 2, 3, 0xFFFF, rng.integers(1 << 32)])))
        else:
            raw[int(rng.integers(len(raw)))] = int(rng.integers(256))


def _fuzzed(fmt: str, rgb: np.ndarray, n: int, seed: int):
    rng = np.random.default_rng(seed)
    for orientation in (1, 3, 6):
        base = _with_raw_exif(rgb, fmt, _phone_exif(orientation))
        s, e = _exif_span(fmt, base)
        assert base[s : s + 4] == b"MM\x00*"
        for _ in range(n):
            data = bytearray(base)
            raw = bytearray(data[s:e])
            _mutate(raw, rng)
            assert len(raw) == e - s
            data[s:e] = raw
            if fmt == "PNG":  # keep the chunk CRC valid: only the EXIF is broken
                data[e : e + 4] = struct.pack(">I", zlib.crc32(bytes(data[s - 4 : e])) & 0xFFFFFFFF)
            yield bytes(data)


FUZZ_RGB = np.random.default_rng(1).integers(0, 256, (64, 48, 3), dtype=np.uint8)


@pytest.mark.parametrize("fmt", ["PNG", "WEBP"])
def test_exif_only_fuzz_of_png_and_webp_always_decodes(fmt):
    # Scratch runs: 45,000 per format, all graded.
    for data in _fuzzed(fmt, FUZZ_RGB, 150, seed=5):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bgr, _ = decode_image(data)
        assert bgr.size == FUZZ_RGB.size


def test_exif_only_fuzz_of_jpeg_never_raises_anything_but_a_decode_error():
    # Scratch runs: 45,000 JPEGs, 0.34% ImageDecodeError (all D3), nothing else.
    for data in _fuzzed("JPEG", FUZZ_RGB, 150, seed=5):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                decode_image(data)
        except ImageDecodeError:
            pass


# --------------------------------------------------------------------------- #
# End to end: precedence, face naming, legacy path
# --------------------------------------------------------------------------- #
def test_an_undecodable_back_beats_a_card_not_found_front(tmp_path, pair, blank, monkeypatch):
    r = _post(_service(tmp_path, _guard(), _feats()), monkeypatch, blank, pair[1][:300])
    assert r.status_code == 400
    assert r.json()["detail"] == f"back: {UNREADABLE}"


def test_an_undecodable_front_beats_a_card_not_found_back(tmp_path, pair, blank, monkeypatch):
    r = _post(_service(tmp_path, _guard(), _feats()), monkeypatch, b"\x00" * 64, blank)
    assert r.status_code == 400
    assert r.json()["detail"] == f"front: {UNREADABLE}"


def test_both_faces_missing_after_a_swap_names_both(tmp_path, blank, monkeypatch):
    svc = _service(tmp_path, _guard(), _feats(side=(10.0, -10.0)))
    pred = svc.predict(blank, blank)
    assert pred.swapped
    assert pred.rejection() == ("card_not_found", f"front and back: {CARD_NOT_FOUND}")


def test_a_rotated_fallback_face_is_still_card_not_found(tmp_path, pair, blank, monkeypatch):
    # The guard turns the (blank) uploaded front 180 degrees; it still has no quad.
    flipped = _feats(flip=(10.0, -10.0))
    svc = _service(tmp_path, _guard(), flipped, _feats())
    pred = svc.predict(blank, pair[1])
    assert any("upside down" in n for n in pred.notes)
    assert pred.rejection() == ("card_not_found", f"front: {CARD_NOT_FOUND}")
    r = _post(_service(tmp_path, _guard(), flipped, _feats()), monkeypatch, blank, pair[1])
    assert r.status_code == 422 and r.json()["detail"] == f"front: {CARD_NOT_FOUND}"


def test_a_broken_exif_photo_is_graded_by_the_legacy_model(
    tiny_legacy_onnx, serve_env, pair, monkeypatch
):
    rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(pair[0], np.uint8), 1), cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    exif = im.getexif()
    exif[ORIENTATION] = 1
    exif[0x011A] = TiffImagePlugin.IFDRational(72, 1)
    raw = bytearray(exif.tobytes())
    e = _exif_entry(raw, 0x011A, 6)
    raw[e : e + 2] = struct.pack(">H", 0x014D)
    bad = _save(im, "PNG", exif=bytes(raw))
    svc = GradeInferenceService(model_path=tiny_legacy_onnx)
    assert svc.kind == "legacy" and svc.ready
    r = _post(svc, monkeypatch, bad, pair[1], ctype="image/png")
    assert r.status_code == 200, r.text


def test_fallback_rotation_feeds_detection_the_upright_photo(portrait_rgb, monkeypatch):
    # detect_and_crop sees the same pixels whether exif_transpose or the
    # fallback turned the photo upright.
    data = _jpeg6(np.ascontiguousarray(make_scene(seed=4).bgr[..., ::-1]))
    ref = detect_and_crop(data)
    monkeypatch.setattr(ImageOps, "exif_transpose", lambda *a, **k: (_ for _ in ()).throw(AttributeError()))
    got = detect_and_crop(data)
    assert got.found_quad == ref.found_quad
    np.testing.assert_array_equal(got.image, ref.image)
