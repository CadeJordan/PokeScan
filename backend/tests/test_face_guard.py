"""Face side / orientation guard: decisions, inference integration, artifacts.

Synthetic features and hand-built `FaceGuard`s only; artifacts go to tmp_path.
Features are 4-dim per region, so a guard input (CLS + mean of one face) is
8-dim: dimension 0 drives the side probe (P(back) = sigmoid(x0)), dimension 1
the upright probes (P(flipped) = sigmoid(x1)).
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from backend.app.ml.face_guard import (
    CHECKS,
    NOTE_SWAPPED,
    FaceGuard,
    LogisticProbe,
)
from backend.app.ml.features import REGION_NAMES, feature_spec_hash
from backend.app.ml.grade_head import decode
from backend.app.ml.inference import GradeInferenceService, _jpeg_roundtrip
from backend.app.ml.preprocess import detect_and_crop
from backend.tests import test_inference as _inference_tests
from backend.tests.helpers.synth import make_card, make_scene, to_jpeg_bytes
from backend.tests.test_inference import (
    _probe_service,
    _save_head,
    _write_feature_sidecar,
    _zero_head,
)

probe_files = _inference_tests.probe_files  # fixture: stub extractor with 4-dim features

D = 4
THRESHOLDS = {"side": 0.95, "front_upright": 0.9, "back_upright": 0.99}


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def _guard(enabled=CHECKS, thresholds=None, metadata=None) -> FaceGuard:
    def probe(dim):
        w = np.zeros(2 * D)
        w[dim] = 1.0
        return LogisticProbe(w, 0.0)

    return FaceGuard(
        probes={"side": probe(0), "front_upright": probe(1), "back_upright": probe(1)},
        thresholds=dict(thresholds or THRESHOLDS),
        enabled=tuple(enabled),
        metadata=dict(metadata or {}),
    )


def _feats(side=(-10.0, 10.0), flip=(-10.0, -10.0), tag=0.0) -> np.ndarray:
    """(2, n_regions, D): CLS dim 0 = side logit, dim 1 = flip logit per row.

    `tag` fills the other regions, so tests can tell which call's features the
    head saw; row 1 gets tag + 1 so a swap is visible too.
    """
    f = np.zeros((2, len(REGION_NAMES), D), np.float32)
    f[:, 2:, :] = np.array([tag, tag + 1.0], np.float32)[:, None, None]
    for row in range(2):
        f[row, 0, 0] = side[row]
        f[row, 0, 1] = flip[row]
    return f


class SeqExtractor:
    """Returns the given features call by call and records the images it saw."""

    session = None

    def __init__(self, *feats):
        self.feats = list(feats)
        self.inputs: list[tuple[np.ndarray, np.ndarray]] = []

    @property
    def calls(self) -> int:
        return len(self.inputs)

    def __call__(self, front_rgb, back_rgb):
        self.inputs.append((front_rgb, back_rgb))
        return self.feats[min(len(self.inputs), len(self.feats)) - 1]


class RecordingHead:
    def __init__(self, head):
        self.head, self.seen = head, []

    def __call__(self, feats):
        self.seen.append(np.array(feats))
        return self.head(feats)


@pytest.fixture(scope="module")
def pair():
    """Distinct (front, back) JPEG uploads of synthetic cards."""
    front = to_jpeg_bytes(make_scene(seed=1).bgr)
    back = to_jpeg_bytes(make_scene(seed=2, card=make_card(border_rgb=(40, 80, 200))).bgr)
    return front, back


def _service(tmp_path, guard, *feats):
    svc = _probe_service(tmp_path, feats[0], _zero_head(feats[0]))
    svc._extractor = SeqExtractor(*feats)
    svc._head = RecordingHead(svc._head)
    svc._guard = guard
    return svc


# --------------------------------------------------------------------------- #
# Side check
# --------------------------------------------------------------------------- #
def test_swapped_rows_are_swapped_back_with_crops(tmp_path, pair):
    feats = _feats(side=(10.0, -10.0))  # front slot reads as a back, back slot as a front
    svc = _service(tmp_path, _guard(), feats)
    pred = svc.predict(*pair)
    assert pred.notes == [NOTE_SWAPPED]
    np.testing.assert_array_equal(svc._head.seen[0], feats[::-1])
    assert svc._extractor.calls == 1
    front_crop, back_crop = (detect_and_crop(b) for b in pair)
    np.testing.assert_array_equal(pred.crops[0].image, back_crop.image)
    np.testing.assert_array_equal(pred.crops[1].image, front_crop.image)


def test_swap_then_flip_uses_the_swapped_rows(tmp_path, pair):
    # Row 1 (the real front) is flipped; after the swap the front probe sees it.
    first = _feats(side=(10.0, -10.0), flip=(-10.0, 10.0))
    second = _feats(tag=5.0)
    svc = _service(tmp_path, _guard(), first, second)
    pred = svc.predict(*pair)
    assert pred.notes == [NOTE_SWAPPED, "front photo was upside down; rotated it"]
    assert svc._extractor.calls == 2
    front_crop, back_crop = (detect_and_crop(b) for b in pair)
    rotated_real_front = back_crop.image[::-1, ::-1]
    np.testing.assert_array_equal(pred.crops[0].image, rotated_real_front)
    np.testing.assert_array_equal(svc._extractor.inputs[1][0], _jpeg_roundtrip(rotated_real_front))
    np.testing.assert_array_equal(svc._extractor.inputs[1][1], _jpeg_roundtrip(front_crop.image))


@pytest.mark.parametrize(
    "side, note",
    [((-10.0, -10.0), "both photos look like card fronts"), ((10.0, 10.0), "both photos look like card backs")],
)
def test_same_side_pair_gets_a_note_and_no_swap(tmp_path, pair, side, note):
    feats = _feats(side=side)
    svc = _service(tmp_path, _guard(), feats)
    pred = svc.predict(*pair)
    assert pred.notes == [note]
    np.testing.assert_array_equal(svc._head.seen[0], feats)
    np.testing.assert_array_equal(pred.crops[0].image, detect_and_crop(pair[0]).image)


def test_disabled_check_does_nothing(tmp_path, pair):
    feats = _feats(side=(10.0, -10.0), flip=(10.0, 10.0))
    svc = _service(tmp_path, _guard(enabled=()), feats)
    pred = svc.predict(*pair)
    assert pred.notes == [] and svc._extractor.calls == 1
    np.testing.assert_array_equal(svc._head.seen[0], feats)


# --------------------------------------------------------------------------- #
# Orientation check
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("face", [0, 1])
def test_flipped_face_is_rotated_and_extracted_again(tmp_path, pair, face):
    flip = [-10.0, -10.0]
    flip[face] = 10.0
    first, second = _feats(flip=tuple(flip)), _feats(tag=7.0)
    svc = _service(tmp_path, _guard(), first, second)
    pred = svc.predict(*pair)
    label = ("front", "back")[face]
    assert pred.notes == [f"{label} photo was upside down; rotated it"]
    assert svc._extractor.calls == 2
    np.testing.assert_array_equal(svc._head.seen[0], second)  # head sees the re-extraction
    crops = [detect_and_crop(b) for b in pair]
    want = [c.image for c in crops]
    want[face] = crops[face].image[::-1, ::-1]
    for i in range(2):
        np.testing.assert_array_equal(pred.crops[i].image, want[i])
        np.testing.assert_array_equal(svc._extractor.inputs[1][i], _jpeg_roundtrip(want[i]))
    # The quad still lists (tl, tr, br, bl) of the card as it is now oriented.
    np.testing.assert_array_equal(pred.crops[face].quad, np.roll(crops[face].quad, 2, axis=0))
    other = 1 - face
    np.testing.assert_array_equal(pred.crops[other].quad, crops[other].quad)


def test_both_faces_flipped_extract_once_more(tmp_path, pair):
    svc = _service(tmp_path, _guard(), _feats(flip=(10.0, 10.0)), _feats(tag=3.0))
    pred = svc.predict(*pair)
    assert pred.notes == [
        "front photo was upside down; rotated it",
        "back photo was upside down; rotated it",
    ]
    assert svc._extractor.calls == 2


def test_prediction_carries_the_guard_decision(tmp_path, pair):
    # Rows swapped, and the real front (row 1) flipped: p_back is per uploaded
    # row, p_flip per face after the swap.
    svc = _service(tmp_path, _guard(), _feats(side=(10.0, -10.0), flip=(-10.0, 10.0)), _feats())
    pred = svc.predict(*pair)
    d = pred.guard_decision
    assert d.swap and d.flip == (True, False) and d.notes == pred.notes
    assert d.p_back[0] > 0.99 and d.p_back[1] < 0.01
    assert d.p_flip[0] > 0.99 and d.p_flip[1] < 0.01
    plain = _service(tmp_path, None, _feats()).predict(*pair)
    assert plain.guard_decision is None
    # Not part of the prediction's value.
    unguarded = _service(tmp_path, _guard(enabled=()), _feats()).predict(*pair)
    assert unguarded.guard_decision is not None and unguarded == plain


def test_upright_pair_extracts_once(tmp_path, pair):
    svc = _service(tmp_path, _guard(), _feats())
    pred = svc.predict(*pair)
    assert pred.notes == [] and svc._extractor.calls == 1


# --------------------------------------------------------------------------- #
# No change without a guard, or just below the thresholds
# --------------------------------------------------------------------------- #
def _fields(pred):
    return (
        pred.grade,
        pred.grade_int,
        pred.confidence,
        pred.rank_probs,
        pred.found_quad_front,
        pred.found_quad_back,
        pred.region_grades,
        pred.notes,
    )


def test_no_guard_file_is_bit_identical_to_the_head(probe_files, serve_env, pair):
    head_path, feat_path = probe_files
    _save_head(head_path, D)
    svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert svc.kind == "probe" and svc._guard is None
    assert "face_guard" not in svc.metadata and "face_guard_load_error" not in svc.metadata
    feats = _feats(side=(10.0, -10.0), flip=(10.0, 10.0))  # would trigger every check
    svc._extractor = SeqExtractor(feats)
    pred = svc.predict(*pair)
    out = svc._head(feats)
    expected, hard, conf, _ = decode(out.cum_probs)
    assert pred.grade == expected and pred.grade_int == hard and pred.confidence == conf
    assert pred.rank_probs == [float(p) for p in out.cum_probs]
    assert pred.region_grades == out.region_grades
    assert pred.notes == [] and svc._extractor.calls == 1
    np.testing.assert_array_equal(pred.crops[0].image, detect_and_crop(pair[0]).image)


def test_just_below_thresholds_changes_nothing(tmp_path, pair):
    eps = 1e-4  # in logit space; float32 features keep ~3e-7 here
    t_side, t_front, t_back = THRESHOLDS["side"], THRESHOLDS["front_upright"], THRESHOLDS["back_upright"]
    cases = [
        # front slot almost reads as a back (no swap, no note), back slot is a sure back
        _feats(side=(_logit(t_side) - eps, 10.0), flip=(_logit(t_front) - eps, _logit(t_back) - eps)),
        # front slot almost reads as a back, back slot a sure front: still no swap
        _feats(side=(_logit(t_side) - eps, -10.0)),
        # back slot almost reads as a front, front slot a sure back: no swap
        _feats(side=(10.0, _logit(1 - t_side) + eps)),
        # both almost fronts / almost backs: no note
        _feats(side=(_logit(1 - t_side) + eps, -10.0)),
    ]
    for feats in cases:
        plain = _service(tmp_path, None, feats).predict(*pair)
        svc = _service(tmp_path, _guard(), feats)
        guarded = svc.predict(*pair)
        assert _fields(guarded) == _fields(plain)
        assert guarded.notes == [] and svc._extractor.calls == 1
        np.testing.assert_array_equal(svc._head.seen[0], feats)
    # And just above, each check acts.
    above = _service(tmp_path, _guard(), _feats(flip=(_logit(t_front) + eps, -10.0)), _feats())
    assert above.predict(*pair).notes == ["front photo was upside down; rotated it"]


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
def test_save_load_round_trips_exactly_without_pickle(tmp_path):
    rng = np.random.default_rng(0)
    guard = FaceGuard(
        probes={n: LogisticProbe(rng.normal(size=2 * D), float(rng.normal())) for n in CHECKS},
        thresholds={"side": 0.99, "front_upright": 0.95, "back_upright": 0.9},
        enabled=("side", "front_upright"),
        metadata={"feature_spec_hash": "abc123def456", "disabled_reasons": {"back_upright": "AUC"}},
    )
    path = tmp_path / "face_guard.npz"
    guard.save(path)
    with np.load(path, allow_pickle=False) as z:  # raises on any object array
        assert all(z[k].dtype != object for k in z.files)
    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert sidecar["enabled_checks"] == ["side", "front_upright"]
    loaded = FaceGuard.load(path)
    for n in CHECKS:
        np.testing.assert_array_equal(loaded.probes[n].weight, guard.probes[n].weight)
        assert loaded.probes[n].bias == guard.probes[n].bias
    assert loaded.thresholds == guard.thresholds
    assert loaded.enabled == guard.enabled
    assert loaded.metadata == guard.metadata


def _save_guard(head_path, metadata=None, dim=D, enabled=CHECKS):
    meta = {"feature_spec_hash": feature_spec_hash()} if metadata is None else metadata
    guard = _guard(enabled=enabled, metadata=meta)
    if dim != D:
        guard.probes = {n: LogisticProbe(np.zeros(2 * dim), 0.0) for n in CHECKS}
    path = head_path.with_name("face_guard.npz")
    guard.save(path)
    return path


def _probe_ready(probe_files):
    head_path, feat_path = probe_files
    _save_head(head_path, D, {"feature_spec_hash": feature_spec_hash()})
    _write_feature_sidecar(feat_path, feature_spec_hash())
    return head_path, feat_path


def test_guard_next_to_head_loads(probe_files, serve_env):
    head_path, feat_path = _probe_ready(probe_files)
    _save_guard(head_path, enabled=("side", "front_upright"))
    svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert svc.ready and svc._guard is not None
    assert svc.metadata["face_guard"] == {
        "loaded": True,
        "enabled_checks": ["side", "front_upright"],
        "thresholds": THRESHOLDS,
    }
    assert "face_guard_load_error" not in svc.metadata


def _break_guard(path, problem):
    sidecar = path.with_suffix(".json")
    if problem == "sidecar_missing":
        sidecar.unlink()
    elif problem == "hash_missing":
        sidecar.write_text(json.dumps({"enabled_checks": list(CHECKS)}))
    elif problem == "hash_empty":
        sidecar.write_text(json.dumps({"feature_spec_hash": "", "enabled_checks": list(CHECKS)}))
    elif problem == "hash_mismatch":
        sidecar.write_text(json.dumps({"feature_spec_hash": "0123456789ab", "enabled_checks": []}))
    elif problem == "sidecar_unreadable":
        sidecar.write_text("{not json")
    elif problem == "sidecar_not_a_dict":
        sidecar.write_text("[1, 2]")
    elif problem == "corrupt_npz":
        path.write_bytes(b"not an npz")


@pytest.mark.parametrize(
    "problem, expected",
    [
        ("sidecar_missing", "face_guard.json missing"),
        ("hash_missing", "feature_spec_hash None in face_guard.json is not a hash"),
        ("hash_empty", "feature_spec_hash '' in face_guard.json is not a hash"),
        ("hash_mismatch", "feature_spec_hash 0123456789ab (face_guard.json) != code"),
        ("sidecar_unreadable", "face_guard.json unreadable"),
        ("sidecar_not_a_dict", "face_guard.json unreadable"),
        ("corrupt_npz", ""),
        ("dim_mismatch", "has 10 weights; 4-dim features need 8"),
    ],
)
def test_untrusted_guard_is_disabled_and_api_stays_ready(
    probe_files, serve_env, caplog, pair, problem, expected
):
    head_path, feat_path = _probe_ready(probe_files)
    if problem == "dim_mismatch":
        _save_guard(head_path, dim=5)
    else:
        _break_guard(_save_guard(head_path), problem)
    with caplog.at_level(logging.WARNING, logger="backend.app.ml.inference"):
        svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert svc.kind == "probe" and svc.ready
    assert svc.probe_load_error is None and "probe_load_error" not in svc.metadata
    assert svc._guard is None
    err = svc.metadata["face_guard_load_error"]
    assert expected in err
    assert svc.metadata["face_guard"]["loaded"] is False
    assert any(
        r.levelname == "WARNING" and "face guard" in r.getMessage() for r in caplog.records
    )
    # Grades exactly as without a guard.
    feats = _feats(side=(10.0, -10.0), flip=(10.0, 10.0))
    svc._extractor = SeqExtractor(feats)
    pred = svc.predict(*pair)
    assert pred.notes == [] and svc._extractor.calls == 1


def test_health_reports_untrusted_guard_while_grading(probe_files, serve_env, pair, monkeypatch):
    from fastapi.testclient import TestClient

    import backend.app.api.grade as grade_mod
    import backend.app.main as main_mod

    head_path, feat_path = _probe_ready(probe_files)
    _break_guard(_save_guard(head_path), "hash_mismatch")
    svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    svc._extractor = SeqExtractor(_feats())
    monkeypatch.setattr(grade_mod, "get_inference_service", lambda: svc)
    monkeypatch.setattr(main_mod, "get_inference_service", lambda: svc)
    files = {
        "front": ("f.jpg", pair[0], "image/jpeg"),
        "back": ("b.jpg", pair[1], "image/jpeg"),
    }
    with TestClient(main_mod.create_app()) as c:
        health = c.get("/api/health").json()
        r = c.post("/api/grade", files=files)
    assert health["model_loaded"] is True
    assert health["metadata"]["face_guard"]["loaded"] is False
    assert "0123456789ab" in health["metadata"]["face_guard_load_error"]
    assert r.status_code == 200


def test_grade_response_includes_guard_notes(client, fake_service, scene_jpeg):
    fake_service.prediction.notes = [NOTE_SWAPPED]
    files = {
        "front": ("f.jpg", scene_jpeg, "image/jpeg"),
        "back": ("b.jpg", scene_jpeg, "image/jpeg"),
    }
    r = client.post("/api/grade", files=files)
    assert r.status_code == 200
    assert r.json()["notes"][0] == NOTE_SWAPPED


def test_guard_probabilities_stay_out_of_the_response(client, fake_service, scene_jpeg):
    """`GradePrediction.guard_decision` is for diagnostics: the response schema and
    the confidence are what they are without it."""
    from backend.app.api.grade import GradeResponse
    from backend.app.ml.face_guard import GuardDecision

    files = {
        "front": ("f.jpg", scene_jpeg, "image/jpeg"),
        "back": ("b.jpg", scene_jpeg, "image/jpeg"),
    }
    plain = client.post("/api/grade", files=files).json()
    fake_service.prediction.guard_decision = GuardDecision(
        True, (True, False), [NOTE_SWAPPED], (0.98, 0.01), (0.995, 0.2)
    )
    fake_service.prediction.notes = [NOTE_SWAPPED]
    r = client.post("/api/grade", files=files)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == set(GradeResponse.model_fields)
    assert "p_back" not in r.text and "p_flip" not in r.text
    assert body["confidence"] == plain["confidence"] and body["grade"] == plain["grade"]
