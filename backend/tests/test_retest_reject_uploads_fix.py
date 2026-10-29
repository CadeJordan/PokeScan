"""Re-test of the D1 fix for intent/reject-unusable-uploads: `decode_image` wraps
PIL's `_DECODE_ERRORS` in `ImageDecodeError`, decodes the pixels eagerly with
`pil.load()`, and ignores a malformed EXIF block (orientation 1 when it can't be
read; when it reads but `exif_transpose` fails, that orientation is applied
directly — D2/N1 fix, 2026-10-28).

Checks what the fix could have broken or left open: valid images (modes,
animation, progressive JPEG, the decompression-bomb warning band, every EXIF
orientation) decode exactly as before the fix; a pixel failure is never
swallowed by the EXIF fallback; errors outside PIL decoding keep their type
(no 400); and EXIF that PIL can read but `exif_transpose` can't re-serialise.

Synthetic images, stub services/guards and tmp_path only; nothing under data/
is read or written.
"""

from __future__ import annotations

import io
import struct
import warnings

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageFile, ImageOps, TiffImagePlugin

from backend.app.ml.preprocess import MAX_LONG_EDGE, UPLOAD_FORMATS, ImageDecodeError, decode_image
from backend.tests.helpers.synth import make_scene
from backend.tests.test_face_guard import _feats, _guard, _service
from backend.tests.test_face_guard import pair as _pair_fixture

pair = _pair_fixture

UNREADABLE = "not a readable image (corrupt or truncated file)"
ORIENTATION = 0x0112


def _post(svc, monkeypatch, front: bytes, back: bytes, ctype: str = "image/jpeg"):
    """POST through the real app; server errors come back as 500 instead of raising."""
    import backend.app.api.grade as grade_mod
    import backend.app.main as main_mod

    monkeypatch.setattr(grade_mod, "get_inference_service", lambda: svc)
    monkeypatch.setattr(main_mod, "get_inference_service", lambda: svc)
    files = {"front": ("f", front, ctype), "back": ("b", back, ctype)}
    with TestClient(main_mod.create_app(), raise_server_exceptions=False) as c:
        return c.post("/api/grade", files=files)


def _reference_decode(data: bytes) -> tuple[np.ndarray, float]:
    """`decode_image` as it was before the fix (main d496022), for valid images."""
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


def _save(im: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    im.save(buf, fmt, **kw)
    return buf.getvalue()


@pytest.fixture(scope="module")
def scene_rgb() -> np.ndarray:
    return np.ascontiguousarray(make_scene(seed=0).bgr[..., ::-1])


@pytest.fixture(scope="module")
def portrait_rgb(scene_rgb) -> np.ndarray:
    """Portrait, under MAX_LONG_EDGE, top rows white: orientation shows in the pixels."""
    rgb = scene_rgb[:900, :600].copy()
    rgb[:20] = 255
    return rgb


# --------------------------------------------------------------------------- #
# Valid images decode exactly as before the fix (eager load changes nothing)
# --------------------------------------------------------------------------- #
def _valid_images() -> dict[str, bytes]:
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (300, 200, 3), dtype=np.uint8)
    f0, f1 = Image.new("RGB", (200, 300), (255, 0, 0)), Image.new("RGB", (200, 300), (0, 255, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Pillow deprecates saving mode I as PNG
        i16 = _save(Image.fromarray(rng.integers(0, 65535, (300, 200), dtype=np.uint16)), "PNG")
    return {
        "I16_png": i16,
        "F_tiff": _save(Image.fromarray(rng.random((300, 200)).astype(np.float32)), "TIFF"),
        "I32_tiff": _save(Image.fromarray(rng.integers(0, 1 << 20, (300, 200), dtype=np.int32)), "TIFF"),
        "CMYK_jpeg": _save(Image.new("CMYK", (200, 300), (10, 20, 30, 40)), "JPEG"),
        "LA_png": _save(Image.new("LA", (200, 300), (10, 128)), "PNG"),
        "P_transparency_png": _save(Image.new("P", (200, 300)), "PNG", transparency=0),
        "1bit_png": _save(Image.new("1", (200, 300)), "PNG"),
        "RGBA_png": _save(Image.new("RGBA", (200, 300), (1, 2, 3, 0)), "PNG"),
        "RGBA_webp": _save(Image.new("RGBA", (200, 300), (1, 2, 3, 128)), "WEBP"),
        "LAB_tiff": _save(Image.new("LAB", (200, 300), (50, 0, 0)), "TIFF"),
        "jpeg_tiff": _save(Image.fromarray(rgb), "TIFF", compression="jpeg"),
        "progressive_jpeg": _save(Image.fromarray(rgb), "JPEG", progressive=True),
        "animated_gif": _save(f0, "GIF", save_all=True, append_images=[f1]),
        "apng": _save(f0, "PNG", save_all=True, append_images=[f1]),
        "animated_webp": _save(f0, "WEBP", save_all=True, append_images=[f1], lossless=True),
        "bmp": _save(Image.fromarray(rgb), "BMP"),
        "big_jpeg": _save(Image.fromarray(np.tile(rgb, (5, 7, 1))), "JPEG"),  # > MAX_LONG_EDGE
    }


VALID = _valid_images()


# Changed 2026-10-28 (D4, owner-approved): bytes are decoded only as JPEG, PNG or
# WebP (`preprocess.UPLOAD_FORMATS`). TIFF, GIF and BMP bytes are now a decode
# error, so the pixel checks below feed those formats as an opened PIL image,
# which isn't format-checked, and their bytes are pinned as a decode error.
def _decoder_input(data: bytes) -> bytes | Image.Image:
    """`data` itself in an upload format, else `data` opened as a PIL image."""
    im = Image.open(io.BytesIO(data))
    return data if im.format in UPLOAD_FORMATS else im


NON_UPLOAD = sorted(n for n, d in VALID.items() if Image.open(io.BytesIO(d)).format not in UPLOAD_FORMATS)


def test_the_non_upload_valid_images_are_the_tiff_gif_and_bmp_ones():
    assert NON_UPLOAD == ["F_tiff", "I32_tiff", "LAB_tiff", "animated_gif", "bmp", "jpeg_tiff"]


@pytest.mark.parametrize("name", sorted(VALID))
def test_valid_images_decode_exactly_as_before_the_fix(name):
    bgr, scale = decode_image(_decoder_input(VALID[name]))
    ref, ref_scale = _reference_decode(VALID[name])
    assert scale == ref_scale
    np.testing.assert_array_equal(bgr, ref)


@pytest.mark.parametrize("name", NON_UPLOAD)
def test_valid_bytes_outside_the_upload_formats_are_a_decode_error(name):
    with pytest.raises(ImageDecodeError) as info:
        decode_image(VALID[name])
    assert isinstance(info.value.__cause__, Image.UnidentifiedImageError)


@pytest.mark.parametrize("name", ["animated_gif", "apng", "animated_webp"])
def test_animated_uploads_are_graded_on_their_first_frame(name):
    bgr, _ = decode_image(_decoder_input(VALID[name]))
    assert bgr[0, 0].tolist() == [0, 0, 255]  # frame 0 is red (BGR)


@pytest.mark.parametrize("fmt", ["JPEG", "WEBP", "PNG", "TIFF"])
@pytest.mark.parametrize("orientation", range(1, 9))
def test_every_valid_orientation_is_still_applied(portrait_rgb, fmt, orientation):
    # The EXIF fallback must only kick in when EXIF is broken.
    im = Image.fromarray(portrait_rgb)
    exif = im.getexif()
    exif[ORIENTATION] = orientation
    opts = {"lossless": True} if fmt == "WEBP" else {}
    data = _save(im, fmt, exif=exif.tobytes(), **opts)
    expected = np.array(ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB"))
    bgr, scale = decode_image(_decoder_input(data))  # TIFF: as a PIL image (D4)
    assert scale == 1.0
    np.testing.assert_array_equal(bgr, cv2.cvtColor(expected, cv2.COLOR_RGB2BGR))
    if orientation in (5, 6, 7, 8):
        assert bgr.shape[:2] == portrait_rgb.shape[1::-1]


def test_an_image_in_the_decompression_bomb_warning_band_still_decodes(monkeypatch, scene_rgb):
    # Between MAX_IMAGE_PIXELS and 2x it PIL only warns; eager loading mustn't reject it.
    data = _save(Image.fromarray(scene_rgb), "JPEG")
    n = scene_rgb.shape[0] * scene_rgb.shape[1]
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", int(n / 1.5))
    with pytest.warns(Image.DecompressionBombWarning):
        bgr, _ = decode_image(data)
    assert bgr.ndim == 3


def test_an_already_loaded_pil_image_is_decoded_as_given(scene_rgb):
    bgr, scale = decode_image(Image.fromarray(scene_rgb[:900, :600]))
    assert scale == 1.0
    np.testing.assert_array_equal(bgr, cv2.cvtColor(scene_rgb[:900, :600], cv2.COLOR_RGB2BGR))


# --------------------------------------------------------------------------- #
# The EXIF fallback never swallows a pixel failure
# --------------------------------------------------------------------------- #
def _garble_tiff_header(data: bytes) -> bytes:
    out = bytearray(data)
    i = out.find(b"MM\x00*")
    assert i > 0
    out[i + 1] = 0xAF
    return bytes(out)


@pytest.mark.parametrize("fmt", ["WEBP", "PNG"])
def test_truncated_pixels_with_broken_exif_are_still_a_decode_error(portrait_rgb, fmt):
    im = Image.fromarray(portrait_rgb)
    exif = im.getexif()
    exif[ORIENTATION] = 6
    opts = {"lossless": True} if fmt == "WEBP" else {"compress_level": 0}
    data = _garble_tiff_header(_save(im, fmt, exif=exif.tobytes(), **opts))
    # PNG puts eXIf before IDAT, so cutting in half truncates the pixels after it.
    with pytest.raises(ImageDecodeError):
        decode_image(data[: len(data) // 2])


def test_load_truncated_images_stays_off():
    # If something set it, a cut-off upload would decode as a partial image and be graded.
    assert ImageFile.LOAD_TRUNCATED_IMAGES is False


def test_exif_transpose_only_ever_sees_decoded_pixels(monkeypatch, portrait_rgb):
    # exif_transpose only runs on loaded pixels, so its suppress can't hide a
    # decode error: a failure in load() surfaces before it is reached.
    im = Image.fromarray(portrait_rgb)
    exif = im.getexif()
    exif[ORIENTATION] = 6
    data = _save(im, "JPEG", exif=exif.tobytes())
    calls = []
    real = ImageOps.exif_transpose

    def spy(image, **kw):
        calls.append(image.im is not None)  # pixels already decoded
        return real(image, **kw)

    monkeypatch.setattr(ImageOps, "exif_transpose", spy)
    decode_image(data)
    assert calls == [True]


# --------------------------------------------------------------------------- #
# Errors outside PIL decoding keep their type (never a 400)
# --------------------------------------------------------------------------- #
def test_an_opencv_error_is_not_a_decode_error(monkeypatch, scene_rgb):
    import backend.app.ml.preprocess as preprocess

    def cvt(*a, **kw):
        raise cv2.error("opencv bug")

    monkeypatch.setattr(preprocess.cv2, "cvtColor", cvt)
    with pytest.raises(cv2.error):
        decode_image(_save(Image.fromarray(scene_rgb), "JPEG"))


def test_out_of_memory_is_not_a_decode_error(monkeypatch, scene_rgb):
    def convert(self, *a, **kw):
        raise MemoryError

    data = _save(Image.new("L", (200, 300)), "PNG")
    monkeypatch.setattr(Image.Image, "convert", convert)
    with pytest.raises(MemoryError):
        decode_image(data)


def test_a_non_pil_input_is_a_type_bug_not_a_400():
    # memoryview isn't bytes/bytearray: a caller bug, not an unreadable upload.
    with pytest.raises(AttributeError):
        decode_image(memoryview(b"\xff\xd8\xff"))


# --------------------------------------------------------------------------- #
# EXIF PIL can read but `exif_transpose` can't re-serialise
# --------------------------------------------------------------------------- #
def _retagged_exif(rgb: np.ndarray, fmt: str, src_tag: int, dst_tag: int, orientation: int = 3) -> bytes:
    """A readable image whose EXIF has a valid orientation and one entry renamed
    to a tag PIL types differently (e.g. XResolution, a RATIONAL, renamed to
    InkNames, an ASCII tag). `getexif()` parses it; `exif_transpose`, which
    re-serialises the EXIF after rotating, raises."""
    im = Image.fromarray(rgb)
    exif = im.getexif()
    exif[ORIENTATION] = orientation
    exif[0x0110] = "Model"
    exif[0x011A] = TiffImagePlugin.IFDRational(72, 1)
    raw = bytearray(exif.tobytes())
    assert raw[6:8] == b"MM"
    i = raw.find(struct.pack(">H", src_tag), 8)
    raw[i : i + 2] = struct.pack(">H", dst_tag)
    opts = {"lossless": True} if fmt == "WEBP" else {}
    return _save(im, fmt, exif=bytes(raw), **opts)


def _as_stored(data: bytes) -> np.ndarray:
    """The pixels as stored, ignoring EXIF (BGR)."""
    return cv2.cvtColor(np.array(Image.open(io.BytesIO(data)).convert("RGB")), cv2.COLOR_RGB2BGR)


INK_NAMES_AS_RATIONAL = (0x011A, 0x014D)  # exif_transpose: AttributeError in write_string
X_POSITION_AS_ASCII = (0x0110, 0x011F)    # exif_transpose: TypeError in write_rational


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_retagged_exif_reaches_pil_with_a_readable_orientation(portrait_rgb, fmt):
    # Pins the inputs below: the orientation reads fine, the transpose raises.
    for tags, exc in ((INK_NAMES_AS_RATIONAL, AttributeError), (X_POSITION_AS_ASCII, TypeError)):
        data = _retagged_exif(portrait_rgb, fmt, *tags)
        im = Image.open(io.BytesIO(data))
        assert im.getexif().get(ORIENTATION) == 3
        with pytest.raises(exc):
            ImageOps.exif_transpose(im)


def _clean_exif_upright(data: bytes, orientation: int) -> np.ndarray:
    """`ImageOps.exif_transpose` of a copy of the stored pixels whose EXIF holds
    only `orientation` (lossless PNG, so the pixels are unchanged), as BGR."""
    exif = Image.Exif()
    exif[ORIENTATION] = orientation
    clean = _save(Image.open(io.BytesIO(data)).convert("RGB"), "PNG", exif=exif.tobytes())
    upright = ImageOps.exif_transpose(Image.open(io.BytesIO(clean)))
    return cv2.cvtColor(np.array(upright.convert("RGB")), cv2.COLOR_RGB2BGR)


# Changed 2026-10-28 (D2/N1, owner-approved): these two pinned the old rule, where
# a failing exif_transpose meant upright although the orientation was readable.
# Now the readable orientation is applied (D2 fixed the AttributeError's 500).
@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_exif_transpose_type_error_still_applies_the_orientation(portrait_rgb, fmt):
    data = _retagged_exif(portrait_rgb, fmt, *X_POSITION_AS_ASCII)
    bgr, _ = decode_image(data)
    np.testing.assert_array_equal(bgr, _clean_exif_upright(data, 3))
    np.testing.assert_array_equal(bgr, _as_stored(data)[::-1, ::-1])  # 3: rotate 180


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_exif_transpose_attribute_error_still_applies_the_orientation(portrait_rgb, fmt):
    data = _retagged_exif(portrait_rgb, fmt, *INK_NAMES_AS_RATIONAL)
    bgr, _ = decode_image(data)
    np.testing.assert_array_equal(bgr, _clean_exif_upright(data, 3))
    np.testing.assert_array_equal(bgr, _as_stored(data)[::-1, ::-1])  # 3: rotate 180


@pytest.mark.parametrize("tags", [INK_NAMES_AS_RATIONAL, X_POSITION_AS_ASCII], ids=["attr", "type"])
@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
@pytest.mark.parametrize("orientation", range(2, 9))
def test_fallback_rotation_matches_exif_transpose_of_clean_exif(portrait_rgb, fmt, tags, orientation):
    data = _retagged_exif(portrait_rgb, fmt, *tags, orientation=orientation)
    with pytest.raises((AttributeError, TypeError)):  # the fallback is what runs
        ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    bgr, scale = decode_image(data)
    assert scale == 1.0
    np.testing.assert_array_equal(bgr, _clean_exif_upright(data, orientation))


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
@pytest.mark.parametrize("orientation", [0, 9, 65535])
def test_broken_exif_with_an_out_of_range_orientation_is_upright(portrait_rgb, fmt, orientation):
    data = _retagged_exif(portrait_rgb, fmt, *INK_NAMES_AS_RATIONAL, orientation=orientation)
    bgr, _ = decode_image(data)
    np.testing.assert_array_equal(bgr, _as_stored(data))


@pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
def test_truncated_pixels_with_unserialisable_exif_are_still_a_decode_error(portrait_rgb, fmt):
    # The EXIF step's broad catch must not swallow a pixel error. JPEG's APP1 and
    # PNG's eXIf precede the pixels, so the cut keeps the readable broken EXIF.
    data = _retagged_exif(portrait_rgb, fmt, *INK_NAMES_AS_RATIONAL, orientation=6)
    cut = data[: len(data) // 2]
    assert Image.open(io.BytesIO(cut)).getexif().get(ORIENTATION) == 6
    with pytest.raises(ImageDecodeError) as info:
        decode_image(cut)
    assert isinstance(info.value.__cause__, OSError)


@pytest.mark.parametrize("face", ["front", "back"])
def test_photo_with_unserialisable_exif_is_graded_not_500(tmp_path, pair, monkeypatch, face):
    rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(pair[0 if face == "front" else 1], np.uint8), 1),
                       cv2.COLOR_BGR2RGB)
    bad = _retagged_exif(rgb, "JPEG", *INK_NAMES_AS_RATIONAL, orientation=3)
    front, back = (bad, pair[1]) if face == "front" else (pair[0], bad)
    svc = _service(tmp_path, _guard(), _feats())
    r = _post(svc, monkeypatch, front, back)
    assert r.status_code == 200, r.text


def test_photo_with_type_error_exif_is_graded(tmp_path, pair, monkeypatch):
    rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(pair[0], np.uint8), 1), cv2.COLOR_BGR2RGB)
    bad = _retagged_exif(rgb, "JPEG", *X_POSITION_AS_ASCII)
    svc = _service(tmp_path, _guard(), _feats())
    r = _post(svc, monkeypatch, bad, pair[1])
    assert r.status_code == 200, r.text


# --------------------------------------------------------------------------- #
# Fuzzed corrupt uploads: always a 400-class error or a decode, never anything else
# --------------------------------------------------------------------------- #
def _fuzz_bases() -> dict[str, bytes]:
    rgb = np.random.default_rng(1).integers(0, 256, (48, 32, 3), dtype=np.uint8)
    im = Image.fromarray(rgb)
    return {
        "JPEG": _save(im, "JPEG"),
        "PJPEG": _save(im, "JPEG", progressive=True),
        "PNG": _save(im, "PNG"),
        "WEBP": _save(im, "WEBP"),
        "WEBPL": _save(im, "WEBP", lossless=True),
        "GIF": _save(im, "GIF"),
        "TIFF": _save(im, "TIFF"),
        "BMP": _save(im, "BMP"),
        "QOI": _save(im, "QOI"),
        "SGI": _save(im, "SGI"),
    }


FUZZ_BASES = _fuzz_bases()


@pytest.mark.parametrize("fmt", sorted(FUZZ_BASES))
def test_fuzzed_uploads_decode_or_raise_image_decode_error(monkeypatch, fmt):
    # Scratch runs of 21,600 such inputs found nothing else; this keeps a seeded slice.
    # Since 2026-10-28 (D4) GIF/TIFF/BMP/QOI/SGI bytes never reach their decoder:
    # every input of those formats is an ImageDecodeError.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 20_000_000)  # a fuzzed size can't eat RAM
    rng = np.random.default_rng(sum(fmt.encode()))
    base = FUZZ_BASES[fmt]
    for _ in range(60):
        data = bytearray(base)
        if rng.random() < 0.3:
            data = data[: int(rng.integers(1, len(data)))]
        else:
            for _ in range(int(rng.integers(1, 9))):
                data[int(rng.integers(len(data)))] = int(rng.integers(256))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                decode_image(bytes(data))
        except ImageDecodeError:
            pass


def test_decode_error_detail_never_leaks_pil_text(tmp_path, pair, monkeypatch):
    svc = _service(tmp_path, _guard(), _feats())
    r = _post(svc, monkeypatch, pair[0], b"P6\n200 30_\n255\n" + bytes(range(256)) * 20)
    assert r.status_code == 400
    assert r.json()["detail"] == f"back: {UNREADABLE}"
