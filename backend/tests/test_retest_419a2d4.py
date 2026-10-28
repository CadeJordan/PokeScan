"""Re-test of 419a2d4 (face guard) and 7cb97c5 (smallfix 12 export): adversarial cases.

Synthetic features, hand-built guards and tmp_path artifacts only; nothing
under data/ is read or written. Known defects are `xfail(strict=True)`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.app.ml.face_guard import CHECKS, NOTE_SWAPPED, FaceGuard, LogisticProbe
from backend.app.ml.features import feature_spec_hash
from backend.app.ml.inference import GradeInferenceService, _rotate_crop
from backend.app.ml.preprocess import _warp_to_canvas, detect_and_crop
from backend.tests import test_inference as _inference_tests
from backend.tests.test_face_guard import (
    D,
    SeqExtractor,
    _feats,
    _guard,
    _probe_ready,
    _save_guard,
    _service,
)
from backend.tests.test_face_guard import pair as _pair_fixture

probe_files = _inference_tests.probe_files
pair = _pair_fixture


def _ready_service(head_path, feat_path) -> GradeInferenceService:
    svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert svc.kind == "probe" and svc.ready, svc.metadata.get("probe_load_error")
    return svc


# --------------------------------------------------------------------------- #
# Guard path in _predict_probe
# --------------------------------------------------------------------------- #
def test_swap_also_swaps_quads_and_found_flags(tmp_path, pair):
    feats = _feats(side=(10.0, -10.0))
    svc = _service(tmp_path, _guard(), feats)
    pred = svc.predict(*pair)
    front, back = (detect_and_crop(b) for b in pair)
    assert pred.notes == [NOTE_SWAPPED]
    np.testing.assert_array_equal(pred.crops[0].quad, back.quad)
    np.testing.assert_array_equal(pred.crops[1].quad, front.quad)
    np.testing.assert_array_equal(pred.crops[0].source_bgr, back.source_bgr)
    assert (pred.found_quad_front, pred.found_quad_back) == (back.found_quad, front.found_quad)


def test_rotated_quad_rewarps_to_the_rotated_crop(pair):
    """Rolling the quad by 2 is a 180 degree turn: warping the source with the new
    quad gives the rotated crop (so anything re-warping from the quad, and the
    (tl, tr, br, bl) labels, follow the card as graded)."""
    crop = detect_and_crop(pair[0])
    assert crop.found_quad and crop.quad is not None
    rot = _rotate_crop(crop)
    rewarped = cv2.cvtColor(_warp_to_canvas(crop.source_bgr, rot.quad), cv2.COLOR_BGR2RGB)
    straight = cv2.cvtColor(_warp_to_canvas(crop.source_bgr, crop.quad), cv2.COLOR_BGR2RGB)
    err_rot = np.abs(rewarped.astype(int) - rot.image.astype(int)).mean()
    err_unrolled = np.abs(straight.astype(int) - rot.image.astype(int)).mean()
    assert err_rot < 2.0 < err_unrolled
    # The original crop is untouched (no aliasing through the view).
    np.testing.assert_array_equal(crop.image, detect_and_crop(pair[0]).image)
    assert rot.image.flags["C_CONTIGUOUS"]


def test_rotate_crop_fallback_without_quad(pair):
    crop = detect_and_crop(pair[0])
    import dataclasses

    fallback = dataclasses.replace(crop, found_quad=False, quad=None, contour=None, source_bgr=None)
    rot = _rotate_crop(fallback)
    assert rot.quad is None and rot.found_quad is False
    np.testing.assert_array_equal(rot.image, crop.image[::-1, ::-1])


def test_flip_only_check_enabled_without_side_check(tmp_path, pair):
    # side disabled: no swap and no side note, but the orientation check still acts.
    svc = _service(tmp_path, _guard(enabled=("front_upright",)), _feats(side=(10.0, -10.0), flip=(10.0, 10.0)), _feats(tag=2.0))
    pred = svc.predict(*pair)
    assert pred.notes == ["front photo was upside down; rotated it"]  # back_upright disabled
    assert svc._extractor.calls == 2


def test_guard_notes_come_first_and_are_unique_in_the_response(probe_files, serve_env, pair, monkeypatch):
    from fastapi.testclient import TestClient

    import backend.app.api.grade as grade_mod
    import backend.app.main as main_mod

    head_path, feat_path = _probe_ready(probe_files)
    _save_guard(head_path)
    svc = _ready_service(head_path, feat_path)
    svc._extractor = SeqExtractor(_feats(side=(10.0, -10.0), flip=(-10.0, 10.0)), _feats(tag=1.0))
    monkeypatch.setattr(grade_mod, "get_inference_service", lambda: svc)
    monkeypatch.setattr(main_mod, "get_inference_service", lambda: svc)
    files = {"front": ("f.jpg", pair[0], "image/jpeg"), "back": ("b.jpg", pair[1], "image/jpeg")}
    with TestClient(main_mod.create_app()) as c:
        r = c.post("/api/grade", files=files)
    assert r.status_code == 200
    notes = r.json()["notes"]
    assert notes[:2] == [NOTE_SWAPPED, "front photo was upside down; rotated it"]
    assert len(notes) == len(set(notes))


def test_guard_is_safe_under_concurrent_requests(tmp_path, pair):
    """The guard is read-only after load; concurrent predicts give the same answer."""
    from concurrent.futures import ThreadPoolExecutor

    feats = _feats(side=(10.0, -10.0))
    svc = _service(tmp_path, _guard(), feats)
    svc._extractor = lambda f, b: feats  # stateless, unlike SeqExtractor
    first = svc.predict(*pair)
    with ThreadPoolExecutor(8) as ex:
        preds = list(ex.map(lambda _: svc.predict(*pair), range(16)))
    assert all(p == first for p in preds)
    assert all(p.notes == [NOTE_SWAPPED] for p in preds)


# --------------------------------------------------------------------------- #
# Fail open: more load-error paths
# --------------------------------------------------------------------------- #
def _assert_fails_open(svc, expected: str):
    assert svc.ready and svc.kind == "probe" and svc.probe_load_error is None
    assert svc._guard is None
    assert expected in svc.metadata["face_guard_load_error"]
    assert svc.metadata["face_guard"]["loaded"] is False


@pytest.mark.parametrize(
    "problem, expected",
    [
        ("npz_missing_key", "back_upright.weight"),
        ("enabled_not_a_list", "enabled_checks"),
        ("enabled_unknown_check", "enabled_checks"),
        ("enabled_unhashable", "TypeError"),
        ("sidecar_is_a_directory", "Error"),
        ("npz_is_a_directory", "Error"),
        ("hash_not_a_string", "is not a hash"),
        ("npz_with_pickled_object", "pickle"),
    ],
)
def test_more_untrusted_guards_fail_open(probe_files, serve_env, problem, expected):
    head_path, feat_path = _probe_ready(probe_files)
    path = _save_guard(head_path)
    sidecar = path.with_suffix(".json")
    meta = json.loads(sidecar.read_text())
    if problem == "npz_missing_key":
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files if not k.startswith("back_upright")}
        np.savez(path, **arrays)
    elif problem == "enabled_not_a_list":
        sidecar.write_text(json.dumps({**meta, "enabled_checks": "side"}))
    elif problem == "enabled_unknown_check":
        sidecar.write_text(json.dumps({**meta, "enabled_checks": ["side", "sideways"]}))
    elif problem == "enabled_unhashable":
        sidecar.write_text(json.dumps({**meta, "enabled_checks": [["side"]]}))
    elif problem == "sidecar_is_a_directory":
        sidecar.unlink()
        sidecar.mkdir()
    elif problem == "npz_is_a_directory":
        path.unlink()
        path.mkdir()
    elif problem == "hash_not_a_string":
        sidecar.write_text(json.dumps({**meta, "feature_spec_hash": 428}))
    elif problem == "npz_with_pickled_object":
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["side.weight"] = np.array([{"x": 1}], dtype=object)
        np.savez(path, **arrays)
    _assert_fails_open(_ready_service(head_path, feat_path), expected)


def test_guard_loads_with_no_enabled_checks_and_does_nothing(probe_files, serve_env, pair):
    head_path, feat_path = _probe_ready(probe_files)
    _save_guard(head_path, enabled=())
    svc = _ready_service(head_path, feat_path)
    assert svc.metadata["face_guard"]["loaded"] is True
    assert svc.metadata["face_guard"]["enabled_checks"] == []
    svc._extractor = SeqExtractor(_feats(side=(10.0, -10.0), flip=(10.0, 10.0)))
    pred = svc.predict(*pair)
    assert pred.notes == [] and svc._extractor.calls == 1


def test_explicit_legacy_model_ignores_a_guard_file(tmp_path, serve_env, tiny_legacy_onnx):
    _save_guard(tmp_path / "grade_head.npz")
    svc = GradeInferenceService(
        model_path=tiny_legacy_onnx, head_path=tmp_path / "grade_head.npz"
    )
    assert svc.kind == "legacy"
    assert "face_guard" not in svc.metadata and "face_guard_load_error" not in svc.metadata


@pytest.mark.xfail(
    strict=True,
    reason="inference.py:_load_guard runs `path.exists()` (and the FaceGuard import) outside its "
    "try: an OSError other than not-found from that stat propagates to _load_probe, so the "
    "probe fails closed (503) because of the guard",
)
def test_guard_stat_error_never_makes_the_probe_unready(probe_files, serve_env, monkeypatch):
    head_path, feat_path = _probe_ready(probe_files)
    _save_guard(head_path)
    real_exists = Path.exists

    def exists(self, *a, **k):
        if self.name == "face_guard.npz":
            raise PermissionError(13, "Access is denied", str(self))
        return real_exists(self, *a, **k)

    monkeypatch.setattr(Path, "exists", exists)
    svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert svc.ready, svc.metadata.get("probe_load_error")
    assert "face_guard_load_error" in svc.metadata


@pytest.mark.xfail(
    strict=True,
    reason="FaceGuard.check_feature_dim compares weight.size, not shape: a (1, 2D) weight passes "
    "the load check, then every /api/grade fails with a matmul ValueError (500) instead of the "
    "guard failing open at load",
)
def test_guard_weight_of_wrong_shape_fails_open_at_load(probe_files, serve_env, pair):
    head_path, feat_path = _probe_ready(probe_files)
    path = _save_guard(head_path)
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    for name in CHECKS:
        arrays[f"{name}.weight"] = arrays[f"{name}.weight"].reshape(1, 2 * D)
    np.savez(path, **arrays)
    svc = _ready_service(head_path, feat_path)
    svc._extractor = SeqExtractor(_feats())
    if svc._guard is not None:
        svc.predict(*pair)  # raises ValueError today
    assert svc._guard is None and "face_guard_load_error" in svc.metadata


@pytest.mark.xfail(
    strict=True,
    reason="FaceGuard.load doesn't validate thresholds: a side threshold below 0.5 makes "
    "'looks back' and 'looks front' overlap, so an unsure pair (P(back) 0.5 / 0.5) is swapped",
)
def test_side_threshold_below_half_is_refused(tmp_path):
    guard = _guard(thresholds={"side": 0.4, "front_upright": 0.9, "back_upright": 0.9})
    path = tmp_path / "face_guard.npz"
    guard.save(path)
    loaded = FaceGuard.load(path)
    swap, _, _ = loaded.check_sides(_feats(side=(0.0, 0.0)))
    assert not swap


def test_save_load_keeps_float64_weights(tmp_path):
    w = np.linspace(-1, 1, 2 * D) / 3.0
    g = FaceGuard({n: LogisticProbe(w.copy(), 0.1) for n in CHECKS}, dict.fromkeys(CHECKS, 0.95), CHECKS)
    g.save(tmp_path / "g.npz")
    loaded = FaceGuard.load(tmp_path / "g.npz")
    assert loaded.probes["side"].weight.dtype == np.float64
    np.testing.assert_array_equal(loaded.probes["side"].weight, w)


# --------------------------------------------------------------------------- #
# smallfix 12: export interrupted at each remaining step
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch")


@pytest.fixture
def exported(tmp_path, monkeypatch):
    from backend.app.ml import features
    from backend.tests.test_feature_export import _save_head as save_export_head
    from backend.tests.test_feature_export import _Tiny

    monkeypatch.setattr(features, "build_region_model", lambda pretrained=True: _Tiny().eval())
    graph = tmp_path / "card_features.onnx"
    features.export_onnx(graph)
    head = tmp_path / "grade_head.npz"
    save_export_head(head, 3)
    return features, graph, head, graph.read_bytes(), graph.with_suffix(".json").read_bytes()


def _names(d):
    return sorted(p.name for p in d.iterdir())


def test_export_failing_to_drop_old_sidecar_keeps_old_pair(exported, serve_env, monkeypatch):
    features, graph, head, old_graph, old_sidecar = exported
    real_unlink = Path.unlink

    def unlink(self, *a, **k):
        if self.name == "card_features.json":
            raise PermissionError(13, "in use", str(self))
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(PermissionError):
        features.export_onnx(graph)
    monkeypatch.setattr(Path, "unlink", real_unlink)
    assert graph.read_bytes() == old_graph
    assert graph.with_suffix(".json").read_bytes() == old_sidecar
    assert _names(graph.parent) == ["card_features.json", "card_features.onnx", "grade_head.json", "grade_head.npz"]
    _ready_service(head, graph)


@pytest.mark.parametrize("failing_call", [1, 2])
def test_export_failing_at_a_rename_fails_closed_and_cleans_up(exported, serve_env, monkeypatch, failing_call):
    """os.replace of the graph (1) or of the sidecar (2) fails: the old sidecar is
    already gone, so the pair never loads as consistent; no temp file is left."""
    features, graph, head, old_graph, _ = exported
    real_replace = os.replace
    calls = []

    def replace(src, dst):
        calls.append(dst)
        if len(calls) == failing_call:
            raise PermissionError(13, "in use", str(dst))
        return real_replace(src, dst)

    monkeypatch.setattr(features.os, "replace", replace)
    with pytest.raises(PermissionError):
        features.export_onnx(graph)
    monkeypatch.setattr(features.os, "replace", real_replace)
    assert _names(graph.parent) == ["card_features.onnx", "grade_head.json", "grade_head.npz"]
    svc = GradeInferenceService(head_path=head, features_path=graph)
    assert not svc.ready
    assert "card_features.json missing" in svc.metadata["probe_load_error"]


def test_stale_temp_files_from_a_killed_export_are_overwritten(exported, serve_env):
    features, graph, head, _, _ = exported
    (graph.parent / "card_features.onnx.tmp").write_bytes(b"half a graph")
    (graph.parent / "card_features.json.tmp").write_text("{")
    features.export_onnx(graph)
    assert _names(graph.parent) == ["card_features.json", "card_features.onnx", "grade_head.json", "grade_head.npz"]
    meta = json.loads(graph.with_suffix(".json").read_text())
    assert meta["feature_spec_hash"] == feature_spec_hash()
    _ready_service(head, graph)
