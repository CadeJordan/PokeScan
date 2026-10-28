"""HTTP contract for /api/grade and /api/health, with a fake inference service."""

from __future__ import annotations

import pytest

import backend.app.ml.condition as condition
from backend.app.api.grade import MAX_BYTES


def _files(front: bytes, back: bytes, front_type="image/jpeg", back_type="image/jpeg"):
    return {
        "front": ("front.jpg", front, front_type),
        "back": ("back.jpg", back, back_type),
    }


def test_health_model_loaded(client, fake_service):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "metadata": fake_service.metadata}


def test_health_model_missing(client, fake_service):
    fake_service.ready = False
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is False


def test_grade_happy_path(client, fake_service, scene_jpeg):
    r = client.post("/api/grade", files=_files(scene_jpeg, scene_jpeg))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["grade"] == 8.62
    assert body["grade_int"] == 9
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["rank_probs"]) == 9
    assert body["notes"] == []
    assert fake_service.calls == [(scene_jpeg, scene_jpeg)]

    sub = body["sub_grades"]
    assert sub["centering"]["left_right"].count("/") == 1
    for name in ("corners", "edges", "surface"):
        assert 1.0 <= sub[name]["grade"] <= 10.0
        assert isinstance(sub[name]["flags"], list)


@pytest.mark.parametrize("content_type", ["image/png", "image/webp"])
def test_grade_accepts_other_image_types(client, scene_jpeg, content_type):
    r = client.post("/api/grade", files=_files(scene_jpeg, scene_jpeg, content_type, content_type))
    assert r.status_code == 200


def test_grade_reports_missing_quads_in_notes(client, fake_service, scene_jpeg):
    fake_service.prediction.found_quad_front = False
    fake_service.prediction.found_quad_back = False
    r = client.post("/api/grade", files=_files(scene_jpeg, scene_jpeg))
    notes = r.json()["notes"]
    assert any(n.startswith("front:") for n in notes)
    assert any(n.startswith("back:") for n in notes)


def test_grade_rejects_unsupported_type(client, scene_jpeg):
    r = client.post("/api/grade", files=_files(scene_jpeg, b"GIF89a", back_type="image/gif"))
    assert r.status_code == 415
    assert "back" in r.json()["detail"]


def test_grade_rejects_empty_upload(client, scene_jpeg):
    r = client.post("/api/grade", files=_files(b"", scene_jpeg))
    assert r.status_code == 400
    assert "front" in r.json()["detail"]


def test_grade_rejects_oversized_upload(client, scene_jpeg):
    r = client.post("/api/grade", files=_files(scene_jpeg, b"\0" * (MAX_BYTES + 1)))
    assert r.status_code == 413


def test_grade_missing_field(client, scene_jpeg):
    r = client.post("/api/grade", files={"front": ("f.jpg", scene_jpeg, "image/jpeg")})
    assert r.status_code == 422


def test_grade_503_when_model_not_loaded(client, fake_service, scene_jpeg):
    fake_service.ready = False
    r = client.post("/api/grade", files=_files(scene_jpeg, scene_jpeg))
    assert r.status_code == 503
    assert fake_service.calls == []


def test_unloadable_probe_gives_503_and_health_says_why(
    tmp_path, serve_env, tiny_legacy_onnx, scene_jpeg, monkeypatch
):
    # A real service with no probe files (a legacy model exists, opt-in off).
    from fastapi.testclient import TestClient

    import backend.app.api.grade as grade_mod
    import backend.app.main as main_mod
    from backend.app.ml.inference import GradeInferenceService

    serve_env(ACTIVE_MODEL_PATH=tiny_legacy_onnx)
    svc = GradeInferenceService(head_path=tmp_path / "grade_head.npz", features_path=tmp_path / "f.onnx")
    monkeypatch.setattr(grade_mod, "get_inference_service", lambda: svc)
    monkeypatch.setattr(main_mod, "get_inference_service", lambda: svc)
    with TestClient(main_mod.create_app()) as c:
        r = c.post("/api/grade", files=_files(scene_jpeg, scene_jpeg))
        assert r.status_code == 503
        assert "/api/health" in r.json()["detail"]
        health = c.get("/api/health").json()
    assert health["model_loaded"] is False
    assert health["metadata"]["probe_load_error"] == "FileNotFoundError: grade_head.npz, f.onnx"


def test_grade_500_on_inference_error(client, fake_service, scene_jpeg):
    fake_service.error = RuntimeError("onnx exploded")
    r = client.post("/api/grade", files=_files(scene_jpeg, scene_jpeg))
    assert r.status_code == 500
    assert r.json()["detail"] == "onnx exploded"


def test_subgrade_failure_does_not_fail_request(client, scene_jpeg, monkeypatch):
    def boom(*_a, **_k):
        raise ValueError("cv failure")

    monkeypatch.setattr(condition, "compute_condition_subgrades", boom)
    r = client.post("/api/grade", files=_files(scene_jpeg, scene_jpeg))
    assert r.status_code == 200
    body = r.json()
    assert body["sub_grades"]["corners"] is None
    assert any("unavailable" in n for n in body["notes"])


def test_cors_allows_configured_origin(client):
    r = client.options(
        "/api/grade",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] in ("http://localhost:3000", "*")


def test_grade_uses_region_grades_from_probe_model(client, fake_service, scene_jpeg, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("classical condition analysis should not run")

    monkeypatch.setattr(condition, "compute_condition_subgrades", boom)
    fake_service.prediction.region_grades = {
        "corners": 9.61, "edges": 8.9, "surface": 7.2, "global": 9.0,
    }
    r = client.post("/api/grade", files=_files(scene_jpeg, scene_jpeg))
    assert r.status_code == 200, r.text
    sub = r.json()["sub_grades"]
    assert sub["corners"] == {"grade": 9.6, "hint": "clean", "flags": []}
    assert sub["edges"]["hint"] == "slight wear"
    assert sub["surface"] == {"grade": 7.2, "hint": "heavy wear", "flags": ["wear"]}


def test_centering_reports_both_faces(client, scene_jpeg):
    r = client.post("/api/grade", files=_files(scene_jpeg, scene_jpeg))
    c = r.json()["sub_grades"]["centering"]
    assert c["left_right"] and c["back_left_right"]
    assert 1.0 <= c["grade"] <= 10.0


def test_unmeasurable_centering_is_a_note_not_a_grade(client):
    import numpy as np

    from backend.tests.helpers.synth import to_jpeg_bytes

    blank = to_jpeg_bytes(np.full((900, 700, 3), 128, np.uint8))
    r = client.post("/api/grade", files=_files(blank, blank))
    body = r.json()
    assert body["sub_grades"]["centering"] is None
    assert any(n.startswith("centering:") for n in body["notes"])


@pytest.mark.parametrize("probe", [True, False], ids=["probe", "legacy"])
def test_grade_reuses_the_service_crops(client, fake_service, scene_jpeg, monkeypatch, probe):
    """Each face is detected once per request: sub-grades reuse the model's crops."""
    import backend.app.ml.centering as centering
    import backend.app.ml.preprocess as preprocess
    from backend.app.ml.preprocess import detect_and_crop

    front, back = detect_and_crop(scene_jpeg), detect_and_crop(scene_jpeg)
    front.rotation = 90
    fake_service.prediction.crops = (front, back)
    if probe:
        fake_service.prediction.region_grades = dict.fromkeys(
            ("corners", "edges", "surface", "global"), 9.5
        )

    def boom(*_a, **_k):
        raise AssertionError("detect_and_crop called again")

    for mod in (preprocess, centering, condition):
        monkeypatch.setattr(mod, "detect_and_crop", boom)
    r = client.post("/api/grade", files=_files(scene_jpeg, scene_jpeg))
    assert r.status_code == 200, r.text
    body = r.json()
    # Sub-grade failures are swallowed into notes, so check they really ran.
    assert body["sub_grades"]["centering"]["left_right"] is not None
    assert body["sub_grades"]["corners"] is not None
    assert "front: photo was sideways; rotated it" in body["notes"]
    assert not any(n.startswith("back: photo") for n in body["notes"])
    assert not any("unavailable" in n for n in body["notes"])
