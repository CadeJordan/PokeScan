"""Re-test of intent/reject-unusable-uploads: 400 for undecodable uploads, 422 for
a face with no card outline or a same-side pair (probe model only).

Adversarial edges the developer's `test_reject_uploads.py` doesn't cover:
decode errors PIL raises as something other than `OSError`, face naming after
a guard swap combined with both-fallback / flip, precedence across the three
cases, the side threshold boundary, non-decode errors that must stay 500, the
legacy service end to end, and `ml.eval` with old-style rows, predictions
without `rejection` and `--exclude-fallback`.

Synthetic images, stub services/guards and tmp_path only; nothing under data/
is read or written.
"""

from __future__ import annotations

import dataclasses
import io
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.ml.face_guard import NOTE_BOTH, NOTE_FLIPPED, NOTE_SWAPPED
from backend.app.ml.inference import GradeInferenceService, GradePrediction
from backend.app.ml.preprocess import ImageDecodeError, decode_image, detect_and_crop
from backend.tests.test_face_guard import THRESHOLDS, _feats, _guard, _logit, _service
from backend.tests.test_face_guard import pair as _pair_fixture

pair = _pair_fixture

CARD_NOT_FOUND = "card not found; retake the photo on a plain background"
UNREADABLE = "not a readable image (corrupt or truncated file)"
SAME_SIDE = "both photos look like card {}s; upload one photo of the front and one of the back"


@pytest.fixture(scope="module")
def blank() -> bytes:
    """A photo with no card in it: detection falls back to the padded canvas."""
    from backend.tests.helpers.synth import to_jpeg_bytes

    return to_jpeg_bytes(np.full((1000, 750, 3), 128, np.uint8))


def _post(svc, monkeypatch, front: bytes, back: bytes, ctype: str = "image/jpeg"):
    """POST through the real app; server errors come back as 500 instead of raising."""
    import backend.app.api.grade as grade_mod
    import backend.app.main as main_mod

    monkeypatch.setattr(grade_mod, "get_inference_service", lambda: svc)
    monkeypatch.setattr(main_mod, "get_inference_service", lambda: svc)
    files = {"front": ("f", front, ctype), "back": ("b", back, ctype)}
    with TestClient(main_mod.create_app(), raise_server_exceptions=False) as c:
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


def _png_chunks(png: bytes) -> list[tuple[int, int, bytes]]:
    """(offset, length, type) of each chunk after the 8-byte signature."""
    out, i = [], 8
    while i < len(png):
        (n,) = struct.unpack(">I", png[i : i + 4])
        out.append((i, n, png[i + 4 : i + 8]))
        i += 12 + n
    return out


@pytest.fixture(scope="module")
def png_bad_chunk_type() -> bytes:
    """A PNG whose second IDAT chunk header is garbled (type isn't 4 ASCII letters).

    Uncompressed noise, so PIL writes several IDAT chunks; `Image.open` succeeds
    and the corruption surfaces during `load()`, as PIL's `SyntaxError("broken
    PNG file (chunk ...)")`, which isn't an `OSError`.
    """
    rgb = np.random.default_rng(0).integers(0, 256, (300, 200, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG", compress_level=0)
    png = bytearray(buf.getvalue())
    idats = [c for c in _png_chunks(bytes(png)) if c[2] == b"IDAT"]
    assert len(idats) >= 2
    off = idats[1][0]
    png[off + 4 : off + 8] = b"ID\x00T"
    return bytes(png)


# A PPM header PIL recognises by its magic but can't parse: PIL raises ValueError.
# Uploads' content type is client-declared, so these bytes can arrive as image/jpeg.
PPM_BAD_HEADER = b"P6\n200 30_\n255\n" + bytes(range(256)) * 20


# --------------------------------------------------------------------------- #
# Decode errors PIL doesn't raise as OSError -> still 400, never 500
# --------------------------------------------------------------------------- #
def test_png_bad_chunk_type_reaches_pil_as_syntax_error(png_bad_chunk_type):
    # Pins the input: PIL itself raises SyntaxError, not OSError, for it.
    im = Image.open(io.BytesIO(png_bad_chunk_type))
    with pytest.raises(SyntaxError):
        im.load()


def test_png_with_a_garbled_chunk_header_is_a_decode_error(png_bad_chunk_type):
    with pytest.raises(ImageDecodeError):
        decode_image(png_bad_chunk_type)


@pytest.mark.parametrize("face", ["front", "back"])
def test_png_with_a_garbled_chunk_header_gives_400(tmp_path, pair, png_bad_chunk_type, monkeypatch, face):
    svc = _service(tmp_path, _guard(), _feats())
    front, back = (png_bad_chunk_type, pair[1]) if face == "front" else (pair[0], png_bad_chunk_type)
    r = _post(svc, monkeypatch, front, back, ctype="image/png")
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == f"{face}: {UNREADABLE}"


def test_unparseable_header_sent_as_jpeg_gives_400(tmp_path, pair, monkeypatch):
    with pytest.raises(ValueError) as info:  # noqa: PT011 - pins PIL's own error type
        Image.open(io.BytesIO(PPM_BAD_HEADER))
    assert not isinstance(info.value, ImageDecodeError)
    svc = _service(tmp_path, _guard(), _feats())
    r = _post(svc, monkeypatch, PPM_BAD_HEADER, pair[1])
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == f"front: {UNREADABLE}"


@pytest.fixture(scope="module")
def webp_bad_exif(pair) -> bytes:
    """A readable WebP of the synthetic front whose EXIF chunk has a garbled TIFF
    header (`MM\\x00*` -> `M\\xaf\\x00*`). `Image.open` succeeds; `getexif()`
    raises PIL's `SyntaxError("not a TIFF file ...")`."""
    im = Image.open(io.BytesIO(pair[0])).convert("RGB")
    exif = im.getexif()
    exif[0x0112] = 1
    buf = io.BytesIO()
    im.save(buf, "WEBP", exif=exif.tobytes(), quality=95)
    data = bytearray(buf.getvalue())
    i = data.find(b"MM\x00*")
    assert i > 0
    data[i + 1] = 0xAF
    return bytes(data)


def test_webp_bad_exif_reaches_pil_as_syntax_error(webp_bad_exif):
    im = Image.open(io.BytesIO(webp_bad_exif))
    with pytest.raises(SyntaxError):
        im.getexif()


def test_webp_with_a_malformed_exif_header_does_not_500(tmp_path, pair, webp_bad_exif, monkeypatch):
    # The pixels are fine, so the bad EXIF is ignored and the pair is graded.
    svc = _service(tmp_path, _guard(), _feats())
    r = _post(svc, monkeypatch, webp_bad_exif, pair[1], ctype="image/webp")
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("value", [0, 9, 65535])
def test_out_of_range_exif_orientation_still_decodes(value):
    from backend.tests.helpers.synth import make_scene

    im = Image.fromarray(make_scene(seed=0).bgr[..., ::-1].copy())
    exif = im.getexif()
    exif[0x0112] = value
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif.tobytes())
    bgr, _ = decode_image(buf.getvalue())
    assert bgr.ndim == 3


def test_decode_error_from_a_pil_image_input_is_wrapped():
    # A PIL image whose data is truncated: the lazy load fails inside decode_image.
    from backend.tests.helpers.synth import make_scene, to_jpeg_bytes

    jpeg = to_jpeg_bytes(make_scene(seed=0).bgr)
    im = Image.open(io.BytesIO(jpeg[: len(jpeg) // 2]))
    with pytest.raises(ImageDecodeError) as info:
        decode_image(im)
    assert isinstance(info.value.__cause__, OSError)


def test_decode_error_is_a_value_error_with_no_face_until_predict_labels_it(pair):
    with pytest.raises(ValueError) as info:  # noqa: PT011
        detect_and_crop(b"\x00" * 64)
    assert isinstance(info.value, ImageDecodeError) and info.value.face is None


# --------------------------------------------------------------------------- #
# Non-decode errors keep their status (no OSError is turned into a 400)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("exc", [OSError("onnx model file vanished"), MemoryError()])
def test_extractor_errors_are_not_reported_as_unreadable_images(tmp_path, pair, monkeypatch, exc):
    svc = _service(tmp_path, _guard(), _feats())

    def boom(front_rgb, back_rgb):
        raise exc

    svc._extractor = boom
    r = _post(svc, monkeypatch, *pair)
    assert r.status_code == 500


def test_runtime_error_is_still_500(client, fake_service, scene_jpeg):
    fake_service.error = RuntimeError("grade model not loaded")
    files = {"front": ("f.jpg", scene_jpeg, "image/jpeg"), "back": ("b.jpg", scene_jpeg, "image/jpeg")}
    r = client.post("/api/grade", files=files)
    assert r.status_code == 500 and r.json()["detail"] == "grade model not loaded"


def test_unlabelled_decode_error_from_a_service_names_no_face(client, fake_service, scene_jpeg):
    fake_service.error = ImageDecodeError("bad")
    files = {"front": ("f.jpg", scene_jpeg, "image/jpeg"), "back": ("b.jpg", scene_jpeg, "image/jpeg")}
    r = client.post("/api/grade", files=files)
    assert r.status_code == 400 and r.json()["detail"] == f"image: {UNREADABLE}"


# --------------------------------------------------------------------------- #
# Precedence: decode > card not found > same side
# --------------------------------------------------------------------------- #
def test_decode_error_on_the_back_beats_card_not_found_on_the_front(tmp_path, pair, blank, monkeypatch):
    svc = _service(tmp_path, _guard(), _feats())
    r = _post(svc, monkeypatch, blank, b"garbage" * 30)
    assert r.status_code == 400
    assert r.json()["detail"] == f"back: {UNREADABLE}"


def test_decode_error_on_the_front_is_raised_before_the_back_is_decoded(tmp_path, pair, monkeypatch):
    import backend.app.ml.inference as inference

    seen = []
    real = inference.detect_and_crop

    def spy(image):
        seen.append(image)
        return real(image)

    monkeypatch.setattr(inference, "detect_and_crop", spy)
    svc = _service(tmp_path, _guard(), _feats())
    with pytest.raises(ImageDecodeError) as info:
        svc.predict(b"garbage" * 30, b"garbage too" * 30)
    assert info.value.face == "front" and len(seen) == 1


def test_fallback_face_and_same_side_through_the_real_service_is_card_not_found(
    tmp_path, pair, blank, monkeypatch
):
    svc = _service(tmp_path, _guard(), _feats(side=(10.0, 10.0)))
    pred = svc.predict(pair[0], blank)
    assert pred.same_side == "back" and not pred.found_quad_back
    assert pred.rejection() == ("card_not_found", f"back: {CARD_NOT_FOUND}")
    r = _post(_service(tmp_path, _guard(), _feats(side=(10.0, 10.0))), monkeypatch, pair[0], blank)
    assert r.status_code == 422 and r.json()["detail"] == f"back: {CARD_NOT_FOUND}"


# --------------------------------------------------------------------------- #
# Face naming after a swap
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "front_ok, back_ok, faces",
    [(False, True, "back"), (True, False, "front"), (False, False, "front and back")],
)
def test_rejection_reverses_the_flags_after_a_swap(front_ok, back_ok, faces):
    # Flags are in graded order: after a swap the graded front is the uploaded back.
    pred = _probe_prediction(found_quad_front=front_ok, found_quad_back=back_ok, swapped=True)
    assert pred.rejection() == ("card_not_found", f"{faces}: {CARD_NOT_FOUND}")


def test_swap_with_both_faces_missing_names_both(tmp_path, blank, monkeypatch):
    svc = _service(tmp_path, _guard(), _feats(side=(10.0, -10.0)))
    pred = svc.predict(blank, blank)
    assert pred.swapped
    assert pred.rejection() == ("card_not_found", f"front and back: {CARD_NOT_FOUND}")


def test_swap_then_flip_still_names_the_uploaded_face(tmp_path, pair, blank):
    # Uploaded back has no card; the guard swaps, then rotates the graded front
    # (the uploaded back). The rotated crop keeps found_quad=False.
    feats = _feats(side=(10.0, -10.0), flip=(-10.0, 10.0))  # after swap: front row flipped
    svc = _service(tmp_path, _guard(), feats, _feats(tag=2.0))
    pred = svc.predict(pair[0], blank)
    assert pred.swapped and pred.notes == [NOTE_SWAPPED, NOTE_FLIPPED.format("front")]
    assert (pred.found_quad_front, pred.found_quad_back) == (False, True)
    assert pred.rejection() == ("card_not_found", f"back: {CARD_NOT_FOUND}")


def test_guard_error_with_a_fallback_face_is_still_card_not_found(tmp_path, pair, blank, monkeypatch):
    from backend.app.ml.face_guard import FaceGuard

    def decide(self, feats):
        raise FloatingPointError("boom")

    monkeypatch.setattr(FaceGuard, "decide", decide)
    svc = _service(tmp_path, _guard(), _feats(side=(10.0, -10.0)))
    pred = svc.predict(blank, pair[1])
    assert not pred.swapped and pred.same_side is None and pred.notes == []
    assert pred.rejection() == ("card_not_found", f"front: {CARD_NOT_FOUND}")


def test_no_guard_with_a_fallback_face_is_still_card_not_found(tmp_path, pair, blank):
    svc = _service(tmp_path, None, _feats(side=(10.0, 10.0)))
    pred = svc.predict(pair[0], blank)
    assert pred.same_side is None and not pred.swapped
    assert pred.rejection() == ("card_not_found", f"back: {CARD_NOT_FOUND}")


# --------------------------------------------------------------------------- #
# same_side: only from the side check's same-side short-circuit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "p, expected",
    [
        ((0.951, 0.951), "back"),
        ((0.949, 0.999), None),   # row 0 just under the threshold
        ((0.049, 0.049), "front"),
        ((0.051, 0.001), None),   # row 0 just over 1 - threshold
        ((0.999, 0.5), None),     # one back, one unsure
        ((0.001, 0.5), None),     # one front, one unsure
    ],
)
def test_same_side_at_the_side_threshold_boundary(p, expected):
    assert THRESHOLDS["side"] == 0.95
    d = _guard().decide(_feats(side=(_logit(p[0]), _logit(p[1]))))
    assert d.same_side == expected
    assert (d.notes == [NOTE_BOTH.format(expected)]) if expected else (d.notes == [])


@pytest.mark.parametrize("t", [0.9, 0.99])
def test_same_side_follows_a_non_default_side_threshold(t):
    g = _guard(thresholds={**THRESHOLDS, "side": t})
    p_in, p_out = (t + 1) / 2, t - (t - 0.5) / 2
    assert g.decide(_feats(side=(_logit(p_in), _logit(p_in)))).same_side == "back"
    assert g.decide(_feats(side=(_logit(p_out), _logit(p_in)))).same_side is None
    assert g.decide(_feats(side=(_logit(1 - p_in), _logit(1 - p_in)))).same_side == "front"


def test_swap_decision_never_sets_same_side():
    d = _guard().decide(_feats(side=(10.0, -10.0), flip=(10.0, 10.0)))
    assert d.swap and d.same_side is None


def test_same_side_with_only_orientation_checks_enabled_is_none():
    d = _guard(enabled=("front_upright", "back_upright")).decide(_feats(side=(10.0, 10.0)))
    assert d.same_side is None and d.p_back is None


# --------------------------------------------------------------------------- #
# Exact owner-approved wording (shown to users verbatim)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fields, expected",
    [
        (dict(found_quad_front=False), ("card_not_found", "front: " + CARD_NOT_FOUND)),
        (dict(found_quad_back=False), ("card_not_found", "back: " + CARD_NOT_FOUND)),
        (
            dict(found_quad_front=False, found_quad_back=False),
            ("card_not_found", "front and back: " + CARD_NOT_FOUND),
        ),
        (dict(same_side="back"), ("same_side", SAME_SIDE.format("back"))),
        (dict(same_side="front"), ("same_side", SAME_SIDE.format("front"))),
        ({}, None),
    ],
)
def test_rejection_wording_is_exactly_the_spec(fields, expected):
    assert _probe_prediction(**fields).rejection() == expected


def test_rejection_leaves_a_legacy_swapped_or_same_side_prediction_alone():
    for fields in (dict(swapped=True, found_quad_back=False), dict(same_side="front")):
        assert _probe_prediction(region_grades=None, **fields).rejection() is None


def test_new_fields_take_part_in_equality_but_crops_still_dont(pair):
    a = _probe_prediction()
    assert a == _probe_prediction(crops=(detect_and_crop(pair[0]), detect_and_crop(pair[1])))
    assert a != dataclasses.replace(a, same_side="back")
    assert a != dataclasses.replace(a, swapped=True)
    assert "same_side=None" in repr(a) and "crops" not in repr(a)


# --------------------------------------------------------------------------- #
# Legacy service end to end: never rejected
# --------------------------------------------------------------------------- #
def test_legacy_service_grades_a_fallback_face_with_the_note(
    tiny_legacy_onnx, serve_env, pair, blank, monkeypatch
):
    svc = GradeInferenceService(model_path=tiny_legacy_onnx)
    assert svc.kind == "legacy" and svc.ready
    pred = svc.predict(blank, pair[1])
    assert pred.region_grades is None and not pred.found_quad_front
    assert pred.rejection() is None and not pred.swapped and pred.same_side is None
    r = _post(svc, monkeypatch, blank, pair[1])
    assert r.status_code == 200, r.text
    assert "front: card outline not auto-detected; used center-padded fallback" in r.json()["notes"]


# --------------------------------------------------------------------------- #
# ml.eval
# --------------------------------------------------------------------------- #
def test_build_report_on_rows_without_reject_has_no_count():
    from ml.eval.evaluate import build_report

    rows = {
        "grade": np.array([10, 9]),
        "pred_int": np.array([10, 9]),
        "pred_soft": np.array([9.8, 9.1]),
        "confidence": np.array([0.6, 0.5]),
        "quad_front": np.array([True, False]),
        "quad_back": np.array([True, True]),
        "latency": np.array([0.1, 0.1]),
    }
    report = build_report(rows, n_boot=0)
    assert "n_would_reject" not in report["pipeline"]
    assert report["pipeline"]["n_fallback"] == 1


def test_predictions_without_rejection_count_as_graded(tmp_path):
    from ml.eval.evaluate import Sample, build_report, predict_all

    class Pred:  # no rejection(): an older or third-party service
        grade, grade_int, confidence = 9.0, 9, 0.5
        found_quad_front, found_quad_back = False, True

    class Service:
        def predict(self, front, back):
            return Pred()

    img = tmp_path / "x.jpg"
    img.write_bytes(b"")
    rows = predict_all(Service(), [Sample("0", 9, img, img), Sample("1", 9, img, img)], progress=False)
    assert rows["reject"].tolist() == ["", ""]
    assert build_report(rows, n_boot=0)["pipeline"]["n_would_reject"] == {
        "card_not_found": 0,
        "same_side": 0,
    }


def _eval_with(tmp_path, monkeypatch, preds):
    import backend.app.ml.inference as inference
    import ml.eval.evaluate as ev
    from ml.tests.test_eval_metrics import _FakeService, _jpeg_samples

    samples = _jpeg_samples(tmp_path, n=len(preds))
    by_front = {s.front.read_bytes(): i for i, s in enumerate(samples)}

    class Svc(_FakeService):
        def predict(self, front, back):
            return preds[by_front.get(front, by_front.get(back, 0))]

    monkeypatch.setattr(inference, "GradeInferenceService", Svc)
    monkeypatch.setattr(ev, "load_samples", lambda split, limit=None: samples)
    return ev


PREDS = [
    _probe_prediction(),
    _probe_prediction(found_quad_front=False),
    _probe_prediction(found_quad_back=False, swapped=True),
    _probe_prediction(same_side="back"),
]


def test_exclude_fallback_still_counts_every_cert_the_api_would_refuse(tmp_path, monkeypatch):
    ev = _eval_with(tmp_path, monkeypatch, PREDS)
    r = ev.run_evaluation(progress=False, n_boot=0, robustness_limit=0, exclude_fallback=True)
    assert r["overall"]["n"] == 2  # the two fallback certs are out of the metrics
    assert r["pipeline"]["excluded_from_metrics"] == 2
    assert r["pipeline"]["n_would_reject"] == {"card_not_found": 2, "same_side": 1}


def test_eval_report_metrics_still_cover_rejected_certs(tmp_path, monkeypatch):
    ev = _eval_with(tmp_path, monkeypatch, PREDS)
    r = ev.run_evaluation(progress=False, n_boot=0, perturbations=["swap_faces"])
    assert r["overall"]["n"] == 4 and len(r["predictions"]) == 4
    assert r["pipeline"]["n_would_reject"] == {"card_not_found": 2, "same_side": 1}
    assert r["robustness"]["swap_faces"]["n"] == 4
    assert "n_would_reject" not in r["robustness"]["swap_faces"]


def test_cli_prints_the_would_refuse_line(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    ev = _eval_with(tmp_path, monkeypatch, PREDS)
    monkeypatch.setattr(ev, "BASELINE_PATH", tmp_path / "no_baseline.json")
    out = tmp_path / "out"
    result = CliRunner().invoke(ev.app, ["--no-robustness", "--n-boot", "0", "--out-dir", str(out)])
    assert result.exit_code == 0, result.output
    assert "/api/grade would refuse: card_not_found 2  same_side 1" in result.output
