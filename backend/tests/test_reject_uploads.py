"""Rejecting unusable uploads (intent/reject-unusable-uploads).

An undecodable image gives 400 (any model). With the probe model, a face with no
card outline or a pair the face guard reads as two of one side gives 422; the
legacy model grades those as before. Synthetic images and stub services only.
"""

from __future__ import annotations

import io
import struct

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageFile, ImageOps

from backend.app.ml.face_guard import NOTE_BOTH, NOTE_SWAPPED
from backend.app.ml.inference import GradeInferenceService, GradePrediction
from backend.app.ml.preprocess import ImageDecodeError, decode_image, detect_and_crop
from backend.tests.helpers.synth import make_scene, to_jpeg_bytes
from backend.tests.test_face_guard import _feats, _guard, _service
from backend.tests.test_face_guard import pair as _pair_fixture

pair = _pair_fixture

CARD_NOT_FOUND = "card not found; retake the photo on a plain background"
UNREADABLE = "not a readable image (corrupt or truncated file)"
CORRUPT_CASES = ("garbage", "jpeg_half", "jpeg_200_bytes", "png_half")


@pytest.fixture(scope="module")
def corrupt() -> dict[str, bytes]:
    jpeg = to_jpeg_bytes(make_scene(seed=0).bgr)
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(make_scene(seed=0).bgr, cv2.COLOR_BGR2RGB)).save(buf, "PNG")
    png = buf.getvalue()
    return {
        "garbage": b"not an image at all" * 10,
        "jpeg_half": jpeg[: len(jpeg) // 2],
        "jpeg_200_bytes": jpeg[:200],
        "png_half": png[: len(png) // 2],
    }


@pytest.fixture(scope="module")
def blank() -> bytes:
    """A photo with no card in it: detection falls back to the padded canvas."""
    return to_jpeg_bytes(np.full((1000, 750, 3), 128, np.uint8))


def _post(svc, monkeypatch, front: bytes, back: bytes, raise_server_exceptions: bool = True):
    import backend.app.api.grade as grade_mod
    import backend.app.main as main_mod

    monkeypatch.setattr(grade_mod, "get_inference_service", lambda: svc)
    monkeypatch.setattr(main_mod, "get_inference_service", lambda: svc)
    files = {"front": ("f.jpg", front, "image/jpeg"), "back": ("b.jpg", back, "image/jpeg")}
    with TestClient(main_mod.create_app(), raise_server_exceptions=raise_server_exceptions) as c:
        return c.post("/api/grade", files=files)


def _probe_prediction(**fields) -> GradePrediction:
    base = dict(
        grade=9.5,
        grade_int=10,
        confidence=0.6,
        rank_probs=[0.99] * 8 + [0.6],
        found_quad_front=True,
        found_quad_back=True,
        region_grades={"corners": 9.5, "edges": 9.5, "surface": 9.5, "global": 9.5},
    )
    return GradePrediction(**{**base, **fields})


# --------------------------------------------------------------------------- #
# Undecodable images -> 400
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", CORRUPT_CASES)
def test_decode_image_raises_image_decode_error(corrupt, case):
    with pytest.raises(ImageDecodeError) as info:
        decode_image(corrupt[case])
    assert info.value.face is None
    assert isinstance(info.value.__cause__, OSError)
    with pytest.raises(ImageDecodeError):
        detect_and_crop(corrupt[case])


def test_decompression_bomb_is_a_decode_error(monkeypatch):
    jpeg = to_jpeg_bytes(make_scene(seed=0).bgr)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)  # the scene is 1.2 MP: > 2x the limit
    with pytest.raises(ImageDecodeError) as info:
        decode_image(jpeg)
    assert isinstance(info.value.__cause__, Image.DecompressionBombError)


@pytest.mark.parametrize("case", CORRUPT_CASES)
@pytest.mark.parametrize("face", ["front", "back"])
def test_unreadable_upload_gives_400_naming_the_face(tmp_path, pair, corrupt, monkeypatch, case, face):
    svc = _service(tmp_path, _guard(), _feats())
    front, back = (corrupt[case], pair[1]) if face == "front" else (pair[0], corrupt[case])
    r = _post(svc, monkeypatch, front, back)
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == f"{face}: {UNREADABLE}"


def test_both_unreadable_names_the_front(tmp_path, corrupt, monkeypatch):
    svc = _service(tmp_path, _guard(), _feats())
    r = _post(svc, monkeypatch, corrupt["garbage"], corrupt["jpeg_half"])
    assert r.status_code == 400
    assert r.json()["detail"] == f"front: {UNREADABLE}"


def test_unreadable_upload_gives_400_with_the_legacy_model(
    tiny_legacy_onnx, serve_env, pair, corrupt, monkeypatch
):
    svc = GradeInferenceService(model_path=tiny_legacy_onnx)
    assert svc.kind == "legacy" and svc.ready
    r = _post(svc, monkeypatch, pair[0], corrupt["png_half"])
    assert r.status_code == 400
    assert r.json()["detail"] == f"back: {UNREADABLE}"


def test_predict_labels_the_decode_error(tmp_path, pair, corrupt):
    svc = _service(tmp_path, _guard(), _feats())
    with pytest.raises(ImageDecodeError) as info:
        svc.predict(pair[0], corrupt["garbage"])
    assert info.value.face == "back"
    assert str(info.value).startswith("back: ")


# PIL's plugins raise these, not OSError, for malformed headers and chunks
# (re-test 2026-10-28: a broken PNG chunk, a bad TIFF header in EXIF, a bad PPM header).
PIL_DECODE_ERRORS = [
    SyntaxError("broken PNG file"),
    ValueError("bad header"),
    IndexError("index out of range"),
    TypeError("bad tile"),
    struct.error("unpack requires a buffer"),
]
# Safety net added 2026-10-28 (D4): PIL's TIFF loader raised these for a
# malformed IFD; the allowed formats' decoders share that code.
SAFETY_NET_ERRORS = [KeyError(0xA005), OverflowError("Python int too large to convert to C ssize_t")]


@pytest.mark.parametrize("exc", PIL_DECODE_ERRORS + SAFETY_NET_ERRORS, ids=lambda e: type(e).__name__)
def test_other_pil_decode_errors_are_decode_errors(monkeypatch, exc):
    def load(self):
        raise exc

    monkeypatch.setattr(ImageFile.ImageFile, "load", load)
    with pytest.raises(ImageDecodeError) as info:
        decode_image(to_jpeg_bytes(make_scene(seed=0).bgr))
    assert info.value.__cause__ is exc and info.value.face is None


@pytest.mark.parametrize("exc", PIL_DECODE_ERRORS + SAFETY_NET_ERRORS, ids=lambda e: type(e).__name__)
def test_other_pil_decode_errors_give_400(tmp_path, pair, monkeypatch, exc):
    real_open = Image.open

    def open_(fp, *args, **kwargs):
        if fp.getvalue() == b"bad upload":
            raise exc
        return real_open(fp, *args, **kwargs)

    monkeypatch.setattr(Image, "open", open_)
    svc = _service(tmp_path, _guard(), _feats())
    r = _post(svc, monkeypatch, pair[0], b"bad upload", raise_server_exceptions=False)
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == f"back: {UNREADABLE}"


def test_a_decode_error_is_not_wrapped_twice(monkeypatch):
    # ImageDecodeError is a ValueError, one of the wrapped types.
    err = ImageDecodeError("already one")

    def load(self):
        raise err

    monkeypatch.setattr(ImageFile.ImageFile, "load", load)
    with pytest.raises(ImageDecodeError) as info:
        decode_image(to_jpeg_bytes(make_scene(seed=0).bgr))
    assert info.value is err


# --------------------------------------------------------------------------- #
# Bytes decode only as JPEG, PNG or WebP (D4, owner decision 2026-10-28)
# --------------------------------------------------------------------------- #
def _encoded(fmt: str, **kw) -> bytes:
    rgb = cv2.cvtColor(make_scene(seed=0).bgr, cv2.COLOR_BGR2RGB)[:300, :200]
    im = Image.fromarray(np.ascontiguousarray(rgb))
    buf = io.BytesIO()
    im.save(buf, fmt, **kw)
    return buf.getvalue()


OTHER_FORMATS = ("TIFF", "GIF", "BMP", "PPM", "ICO", "QOI", "SGI", "TGA", "PCX")


def test_upload_formats_match_the_api_content_types():
    from backend.app.api.grade import ALLOWED_TYPES
    from backend.app.ml.preprocess import UPLOAD_FORMATS

    assert {f"image/{f.lower()}" for f in UPLOAD_FORMATS} == ALLOWED_TYPES


@pytest.mark.parametrize("fmt", OTHER_FORMATS)
def test_valid_bytes_of_another_format_are_a_decode_error(fmt):
    data = _encoded(fmt)
    assert Image.open(io.BytesIO(data)).format == fmt  # PIL itself reads it
    with pytest.raises(ImageDecodeError) as info:
        decode_image(data)
    assert isinstance(info.value.__cause__, Image.UnidentifiedImageError)
    with pytest.raises(ImageDecodeError):
        detect_and_crop(data)


@pytest.mark.parametrize("fmt", OTHER_FORMATS)
@pytest.mark.parametrize("face", ["front", "back"])
def test_valid_bytes_of_another_format_sent_as_jpeg_give_400(tmp_path, pair, monkeypatch, fmt, face):
    data = _encoded(fmt)
    front, back = (data, pair[1]) if face == "front" else (pair[0], data)
    r = _post(_service(tmp_path, _guard(), _feats()), monkeypatch, front, back)
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == f"{face}: {UNREADABLE}"


@pytest.mark.parametrize(
    "fmt, kw",
    [
        ("JPEG", {}),
        ("JPEG", {"progressive": True}),
        ("PNG", {}),
        ("WEBP", {}),
        ("WEBP", {"lossless": True}),
        ("MPO", {}),  # a phone's multi-picture JPEG: PIL's JPEG opener returns it
    ],
    ids=["jpeg", "progressive_jpeg", "png", "webp", "webp_lossless", "mpo"],
)
def test_jpeg_png_and_webp_bytes_still_decode(fmt, kw):
    data = _encoded(fmt, **kw)
    bgr, scale = decode_image(data)
    ref = Image.open(io.BytesIO(data)).convert("RGB")
    assert scale == 1.0
    np.testing.assert_array_equal(bgr, cv2.cvtColor(np.array(ref), cv2.COLOR_RGB2BGR))


@pytest.mark.parametrize("fmt", ["TIFF", "GIF", "BMP"])
def test_a_pil_image_of_another_format_still_decodes(fmt):
    # A PIL input is already opened by the caller: its format isn't checked.
    data = _encoded(fmt)
    bgr, scale = decode_image(Image.open(io.BytesIO(data)))
    ref = Image.open(io.BytesIO(data)).convert("RGB")
    assert scale == 1.0
    np.testing.assert_array_equal(bgr, cv2.cvtColor(np.array(ref), cv2.COLOR_RGB2BGR))
    assert detect_and_crop(Image.open(io.BytesIO(data))).image.shape == (840, 600, 3)


def _with_orientation(rgb: np.ndarray, orientation: int, fmt: str = "JPEG") -> bytes:
    im = Image.fromarray(rgb)
    exif = im.getexif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    opts = {"lossless": True} if fmt == "WEBP" else {"quality": 95}
    im.save(buf, fmt, exif=exif.tobytes(), **opts)
    return buf.getvalue()


@pytest.fixture(scope="module")
def portrait_rgb() -> np.ndarray:
    """Portrait and under MAX_LONG_EDGE, so a rotation shows in the shape."""
    rgb = cv2.cvtColor(make_scene(seed=0).bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb[:900, :600])


def test_malformed_exif_is_ignored_and_the_photo_decoded_upright(portrait_rgb):
    # A WebP with orientation 6 (rotate 90) in an EXIF block whose TIFF header is
    # garbled: getexif() raises SyntaxError, so the tag is ignored. (PIL's JPEG
    # reader swallows the same corruption itself.)
    webp = bytearray(_with_orientation(portrait_rgb, 6, "WEBP"))
    i = webp.find(b"MM\x00*")
    assert i > 0
    webp[i + 1] = 0xAF
    with pytest.raises(SyntaxError):
        Image.open(io.BytesIO(bytes(webp))).getexif()
    bgr, scale = decode_image(bytes(webp))
    assert scale == 1.0
    np.testing.assert_array_equal(bgr, cv2.cvtColor(portrait_rgb, cv2.COLOR_RGB2BGR))


# Anything the EXIF step raises: the pixels are already decoded by then, so it
# catches any Exception (D2, 2026-10-28: exif_transpose raised AttributeError).
EXIF_STEP_ERRORS = [*PIL_DECODE_ERRORS, AttributeError("x"), KeyError(0x0112), Exception("any")]


@pytest.mark.parametrize("exc", EXIF_STEP_ERRORS, ids=lambda e: type(e).__name__)
def test_a_failing_exif_read_means_upright(portrait_rgb, exc):
    jpeg = _with_orientation(portrait_rgb, 6)
    rotated, _ = decode_image(jpeg)
    assert rotated.shape[:2] == portrait_rgb.shape[1::-1]

    def getexif():
        raise exc

    # Patched on the opened image only: PIL's own `open` probes formats with
    # getexif and takes some of these errors to mean "not this format".
    im = Image.open(io.BytesIO(jpeg))
    im.getexif = getexif
    bgr, _ = decode_image(im)
    assert bgr.shape[:2] == portrait_rgb.shape[:2]


# Changed 2026-10-28 (N1, owner-approved): was `..._means_upright`. The orientation
# read fine, so it is now applied directly instead of being dropped.
@pytest.mark.parametrize("exc", EXIF_STEP_ERRORS, ids=lambda e: type(e).__name__)
@pytest.mark.parametrize("orientation", range(2, 9))
def test_a_failing_exif_transpose_still_applies_the_orientation(monkeypatch, portrait_rgb, exc, orientation):
    jpeg = _with_orientation(portrait_rgb, orientation)
    expected, _ = decode_image(jpeg)  # through the real exif_transpose

    def exif_transpose(image, **kwargs):
        raise exc

    monkeypatch.setattr(ImageOps, "exif_transpose", exif_transpose)
    bgr, _ = decode_image(jpeg)
    np.testing.assert_array_equal(bgr, expected)
    if orientation >= 5:
        assert bgr.shape[:2] == portrait_rgb.shape[1::-1]


@pytest.mark.parametrize("value", [[6], {"o": 6}], ids=["list", "dict"])
def test_an_unhashable_orientation_means_upright(portrait_rgb, value):
    # A value `exif_transpose`'s own lookup can't hash (TypeError): taken as upright.
    im = Image.open(io.BytesIO(_with_orientation(portrait_rgb, 1)))
    im.getexif = lambda: {0x0112: value}
    bgr, _ = decode_image(im)
    plain, _ = decode_image(_with_orientation(portrait_rgb, 1))
    np.testing.assert_array_equal(bgr, plain)


def test_truncated_pixels_with_a_failing_exif_step_are_still_a_decode_error(monkeypatch, portrait_rgb):
    # The EXIF step's broad catch can't hide a pixel error: load() fails before it.
    def exif_transpose(image, **kwargs):
        raise AttributeError("x")

    monkeypatch.setattr(ImageOps, "exif_transpose", exif_transpose)
    jpeg = _with_orientation(portrait_rgb, 6)
    with pytest.raises(ImageDecodeError) as info:
        decode_image(jpeg[: len(jpeg) // 2])
    assert isinstance(info.value.__cause__, OSError)


def test_truncated_pixels_with_an_orientation_tag_are_still_a_decode_error(portrait_rgb):
    # The EXIF rule must not swallow a pixel error: the pixels are decoded first.
    jpeg = _with_orientation(portrait_rgb, 6)
    with pytest.raises(ImageDecodeError) as info:
        decode_image(jpeg[: len(jpeg) // 2])
    assert isinstance(info.value.__cause__, OSError)


@pytest.mark.parametrize(
    "exc",
    [ValueError("bad shape"), TypeError("bad arg"), SyntaxError("x"), IndexError("i"),
     KeyError("k"), OverflowError("o")],
    ids=lambda e: type(e).__name__,
)
def test_errors_outside_decoding_are_still_500(tmp_path, pair, monkeypatch, exc):
    svc = _service(tmp_path, _guard(), _feats())

    def boom(front_rgb, back_rgb):
        raise exc

    svc._extractor = boom
    r = _post(svc, monkeypatch, *pair, raise_server_exceptions=False)
    assert r.status_code == 500


def test_a_detection_error_after_decoding_is_still_500(tmp_path, pair, monkeypatch):
    import backend.app.ml.preprocess as preprocess

    def find(bgr):
        raise ValueError("detector bug")

    monkeypatch.setattr(preprocess, "_find_card_quad", find)
    svc = _service(tmp_path, _guard(), _feats())
    with pytest.raises(ValueError, match="detector bug") as info:
        svc.predict(*pair)
    assert not isinstance(info.value, ImageDecodeError)
    r = _post(svc, monkeypatch, *pair, raise_server_exceptions=False)
    assert r.status_code == 500


# --------------------------------------------------------------------------- #
# Card not found -> 422 (probe model only)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "front_ok, back_ok, faces",
    [(False, True, "front"), (True, False, "back"), (False, False, "front and back")],
)
def test_card_not_found_gives_422(client, fake_service, scene_jpeg, front_ok, back_ok, faces):
    fake_service.prediction = _probe_prediction(found_quad_front=front_ok, found_quad_back=back_ok)
    files = {"front": ("f.jpg", scene_jpeg, "image/jpeg"), "back": ("b.jpg", scene_jpeg, "image/jpeg")}
    r = client.post("/api/grade", files=files)
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == f"{faces}: {CARD_NOT_FOUND}"


def test_legacy_model_still_grades_a_fallback_face(client, fake_service, scene_jpeg):
    fake_service.prediction.found_quad_front = False  # region_grades None: legacy path
    files = {"front": ("f.jpg", scene_jpeg, "image/jpeg"), "back": ("b.jpg", scene_jpeg, "image/jpeg")}
    r = client.post("/api/grade", files=files)
    assert r.status_code == 200, r.text
    assert "front: card outline not auto-detected; used center-padded fallback" in r.json()["notes"]


def test_card_not_found_through_the_probe_service(tmp_path, pair, blank, monkeypatch):
    assert not detect_and_crop(blank).found_quad
    svc = _service(tmp_path, _guard(), _feats())
    pred = svc.predict(pair[0], blank)
    assert pred.rejection() == ("card_not_found", f"back: {CARD_NOT_FOUND}")
    r = _post(svc, monkeypatch, pair[0], blank)
    assert r.status_code == 422
    assert r.json()["detail"] == f"back: {CARD_NOT_FOUND}"


def test_card_not_found_names_the_uploaded_face_after_a_swap(tmp_path, pair, blank, monkeypatch):
    # Uploaded front has no card; the guard reads the pair as swapped. The graded
    # back is then the uploaded front, and the message must still say "front".
    svc = _service(tmp_path, _guard(), _feats(side=(10.0, -10.0)))
    pred = svc.predict(blank, pair[1])
    assert pred.notes[0] == NOTE_SWAPPED and pred.swapped
    assert (pred.found_quad_front, pred.found_quad_back) == (True, False)
    assert pred.rejection() == ("card_not_found", f"front: {CARD_NOT_FOUND}")
    svc = _service(tmp_path, _guard(), _feats(side=(10.0, -10.0)))
    r = _post(svc, monkeypatch, blank, pair[1])
    assert r.status_code == 422
    assert r.json()["detail"] == f"front: {CARD_NOT_FOUND}"


# --------------------------------------------------------------------------- #
# Same-side pair -> 422 (probe model only)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("side", ["back", "front"])
def test_same_side_gives_422(client, fake_service, scene_jpeg, side):
    fake_service.prediction = _probe_prediction(same_side=side, notes=[NOTE_BOTH.format(side)])
    files = {"front": ("f.jpg", scene_jpeg, "image/jpeg"), "back": ("b.jpg", scene_jpeg, "image/jpeg")}
    r = client.post("/api/grade", files=files)
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == (
        f"both photos look like card {side}s; upload one photo of the front and one of the back"
    )


@pytest.mark.parametrize("side, kind", [((10.0, 10.0), "back"), ((-10.0, -10.0), "front")])
def test_same_side_is_set_by_the_probe_service(tmp_path, pair, monkeypatch, side, kind):
    svc = _service(tmp_path, _guard(), _feats(side=side))
    pred = svc.predict(*pair)
    assert pred.same_side == kind and not pred.swapped
    assert pred.notes == [NOTE_BOTH.format(kind)]  # unchanged for ml.eval
    assert svc._guard.decide(_feats(side=side)).same_side == kind
    svc = _service(tmp_path, _guard(), _feats(side=side))
    r = _post(svc, monkeypatch, *pair)
    assert r.status_code == 422
    assert r.json()["detail"].startswith(f"both photos look like card {kind}s; ")


@pytest.mark.parametrize(
    "feats",
    [
        _feats(side=(10.0, -10.0)),                                  # swap
        _feats(side=(-10.0, 10.0), flip=(10.0, -10.0)),              # front flipped
        _feats(side=(-10.0, 0.0)),                                   # unsure side
        _feats(),                                                    # clean
    ],
    ids=["swap", "flip", "unsure", "clean"],
)
def test_other_guard_decisions_leave_same_side_unset(tmp_path, pair, feats):
    svc = _service(tmp_path, _guard(), feats, _feats(tag=2.0))
    pred = svc.predict(*pair)
    assert pred.same_side is None
    assert pred.rejection() is None


def test_no_side_check_means_no_same_side_rejection(tmp_path, pair):
    svc = _service(tmp_path, _guard(enabled=("front_upright",)), _feats(side=(10.0, 10.0)))
    pred = svc.predict(*pair)
    assert pred.same_side is None and pred.rejection() is None


def test_guard_error_grades_unguarded(tmp_path, pair, monkeypatch):
    from backend.app.ml.face_guard import FaceGuard

    def decide(self, feats):
        raise FloatingPointError("boom")

    monkeypatch.setattr(FaceGuard, "decide", decide)
    svc = _service(tmp_path, _guard(), _feats(side=(10.0, 10.0)))
    r = _post(svc, monkeypatch, *pair)
    assert r.status_code == 200, r.text


# --------------------------------------------------------------------------- #
# Order
# --------------------------------------------------------------------------- #
def test_card_not_found_takes_precedence_over_same_side():
    pred = _probe_prediction(found_quad_back=False, same_side="back")
    assert pred.rejection() == ("card_not_found", f"back: {CARD_NOT_FOUND}")


def test_legacy_prediction_is_never_rejected():
    pred = _probe_prediction(region_grades=None, found_quad_front=False, same_side="back")
    assert pred.rejection() is None


# --------------------------------------------------------------------------- #
# Evaluation counts what the API would refuse
# --------------------------------------------------------------------------- #
def test_eval_counts_would_reject(tmp_path):
    from ml.eval.evaluate import Sample, build_report, predict_all

    preds = [
        _probe_prediction(),
        _probe_prediction(found_quad_front=False),
        _probe_prediction(found_quad_front=False, found_quad_back=False),
        _probe_prediction(same_side="front"),
    ]

    class Service:
        def __init__(self):
            self.it = iter(preds)

        def predict(self, front, back):
            return next(self.it)

    img = tmp_path / "x.jpg"
    img.write_bytes(b"")
    samples = [Sample(cert=str(i), grade=10, front=img, back=img) for i in range(len(preds))]
    rows = predict_all(Service(), samples, progress=False)
    report = build_report(rows, n_boot=10)
    assert report["pipeline"]["n_would_reject"] == {"card_not_found": 2, "same_side": 1}
    # Every input is still graded: the metrics cover all four.
    assert report["overall"]["n"] == 4
