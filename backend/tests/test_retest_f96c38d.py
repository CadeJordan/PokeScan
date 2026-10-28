"""Re-test of f96c38d (fixes for the 419a2d4 re-test): face guard fail-open edges.

Synthetic features, hand-built guards and tmp_path artifacts only; nothing
under data/ is read or written.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pytest

from backend.app.ml.face_guard import CHECKS, FaceGuard, LogisticProbe
from backend.app.ml.inference import GradeInferenceService
from backend.tests import test_inference as _inference_tests
from backend.tests.test_face_guard import (
    D,
    _feats,
    _guard,
    _probe_ready,
    _save_guard,
    _service,
)
from backend.tests.test_face_guard import pair as _pair_fixture

probe_files = _inference_tests.probe_files
pair = _pair_fixture


def _assert_fails_open(svc, expected: str):
    assert svc.ready and svc.kind == "probe" and svc.probe_load_error is None
    assert svc._guard is None
    assert expected in svc.metadata["face_guard_load_error"]
    assert svc.metadata["face_guard"]["loaded"] is False


def _rewrite(path, **arrays):
    with np.load(path) as z:
        out = {k: z[k] for k in z.files}
    out.update(arrays)
    np.savez(path, **out)


# --------------------------------------------------------------------------- #
# Load: every error fails open
# --------------------------------------------------------------------------- #
def test_guard_import_error_fails_open(probe_files, serve_env, monkeypatch):
    head_path, feat_path = _probe_ready(probe_files)
    _save_guard(head_path)
    # A None entry makes `from backend.app.ml.face_guard import ...` raise ImportError.
    monkeypatch.setitem(sys.modules, "backend.app.ml.face_guard", None)
    svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    _assert_fails_open(svc, "ModuleNotFoundError")


def test_guard_sidecar_that_is_a_directory_fails_open(probe_files, serve_env):
    head_path, feat_path = _probe_ready(probe_files)
    path = _save_guard(head_path)
    sidecar = path.with_suffix(".json")
    sidecar.unlink()
    sidecar.mkdir()
    svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    _assert_fails_open(svc, "Error")


def test_zero_width_weight_loads_alone_but_fails_open_against_the_extractor(
    tmp_path, probe_files, serve_env
):
    path = tmp_path / "g.npz"
    _guard().save(path)
    _rewrite(path, **{"side.weight": np.zeros(0)})
    FaceGuard.load(path)  # width unknown here: 1-D and finite is all load can check
    head_path, feat_path = _probe_ready(probe_files)
    _rewrite(_save_guard(head_path), **{"side.weight": np.zeros(0)})
    svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    _assert_fails_open(svc, f"has 0 weights; {D}-dim features need {2 * D}")


# --------------------------------------------------------------------------- #
# Load: value checks, edges
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "key, value",
    [
        ("side.threshold", np.array(0.5 + 1e-9)),
        ("side.threshold", np.array(1.0 - 1e-9)),
        ("front_upright.threshold", np.array(0.9, np.float32)),
        ("back_upright.threshold", np.array(0.95, np.float16)),
        ("side.bias", np.array(-3, np.int64)),
        ("side.weight", np.zeros(2 * D, np.float32)),
    ],
)
def test_valid_edge_values_still_load(tmp_path, key, value):
    path = tmp_path / "g.npz"
    _guard().save(path)
    _rewrite(path, **{key: value})
    guard = FaceGuard.load(path)
    guard.check_feature_dim(D)
    assert all(isinstance(guard.thresholds[n], float) for n in CHECKS)


@pytest.mark.parametrize(
    "key, value, expected",
    [
        ("side.threshold", np.array(1), "threshold 'side' 1.0 is not in (0.5, 1)"),
        ("side.threshold", np.array(True), "side.threshold has shape () / dtype bool"),
        ("side.threshold", np.array(0.9 + 0j), "side.threshold has shape () / dtype complex"),
        ("side.threshold", np.array(np.inf), "threshold 'side' inf is not in (0.5, 1)"),
        ("front_upright.threshold", np.array(-np.inf), "is not in (0.5, 1)"),
        ("back_upright.threshold", np.array([0.9]), "back_upright.threshold has shape (1,)"),
        ("side.threshold", np.array(0.0), "is not in (0.5, 1)"),
        ("side.bias", np.array(np.inf), "bias inf is not finite"),
        ("side.bias", np.array(-np.inf), "bias -inf is not finite"),
        ("side.bias", np.zeros((1, 1)), "side.bias has shape (1, 1)"),
        ("side.weight", np.array(1.0), "has weight shape (); need 1-D"),
        ("side.weight", np.full(2 * D, np.nan), "non-finite weights"),
        ("back_upright.weight", np.zeros((2 * D, 1)), "has weight shape (8, 1); need 1-D"),
    ],
)
def test_bad_values_are_refused_at_load(tmp_path, key, value, expected):
    path = tmp_path / "g.npz"
    _guard().save(path)
    _rewrite(path, **{key: value})
    with pytest.raises(ValueError) as exc:
        FaceGuard.load(path)
    assert expected in str(exc.value)


def test_threshold_checks_also_apply_to_disabled_checks(probe_files, serve_env):
    """A disabled check's threshold is validated too (the trainer stores 0.99 for one)."""
    head_path, feat_path = _probe_ready(probe_files)
    path = _save_guard(head_path, enabled=("side", "front_upright"))
    _rewrite(path, **{"back_upright.threshold": np.array(0.4)})
    svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    _assert_fails_open(svc, "threshold 'back_upright' 0.4")


def test_check_feature_dim_on_an_in_memory_guard_rejects_bad_thresholds_and_biases():
    guard = _guard(thresholds={"side": 0.95, "front_upright": float("nan"), "back_upright": 0.99})
    with pytest.raises(ValueError, match="front_upright"):
        guard.check_feature_dim(D)
    guard = _guard()
    guard.probes["back_upright"] = LogisticProbe(np.zeros(2 * D), float("nan"))
    with pytest.raises(ValueError, match="bias nan is not finite"):
        guard.check_feature_dim(None)
    _guard().check_feature_dim(None)
    _guard().check_feature_dim(D)


# --------------------------------------------------------------------------- #
# Request: a guard error grades unguarded
# --------------------------------------------------------------------------- #
def test_guard_of_wrong_width_with_unknown_extractor_width_grades_unguarded(tmp_path, pair, caplog):
    """When the graph's width is dynamic (no load-time width check), a guard of the
    wrong width raises in decide on each request; the request is graded unguarded."""
    feats = _feats(side=(10.0, -10.0), flip=(10.0, 10.0))
    bad = _guard()
    bad.probes = {n: LogisticProbe(np.zeros(2 * (D + 1)), 0.0) for n in CHECKS}
    svc = _service(tmp_path / "a", bad, feats)
    plain = _service(tmp_path / "b", None, feats)
    with caplog.at_level(logging.WARNING, logger="backend.app.ml.inference"):
        pred = svc.predict(*pair)
    assert pred == plain.predict(*pair) and pred.notes == []
    np.testing.assert_array_equal(svc._head.seen[0], feats)
    assert svc._extractor.calls == 1
    assert any("graded it unguarded" in r.getMessage() for r in caplog.records)
    assert svc.metadata.get("face_guard", {}).get("loaded", True)  # guard itself is kept


@pytest.mark.parametrize("exc", [ZeroDivisionError("x"), MemoryError(), KeyError("side")])
def test_any_exception_from_decide_grades_unguarded(tmp_path, pair, monkeypatch, exc):
    feats = _feats(side=(10.0, -10.0))
    svc = _service(tmp_path, _guard(), feats)

    def decide(self, f):
        raise exc

    monkeypatch.setattr(FaceGuard, "decide", decide)
    pred = svc.predict(*pair)
    assert pred.notes == []
    np.testing.assert_array_equal(svc._head.seen[0], feats)


# --------------------------------------------------------------------------- #
# Same-side short-circuit: only same-side pairs skip the orientation check
# --------------------------------------------------------------------------- #
def test_side_check_disabled_still_runs_the_orientation_check():
    guard = _guard(enabled=("front_upright", "back_upright"))
    d = guard.decide(_feats(side=(10.0, 10.0), flip=(10.0, -10.0)))
    assert d.flip == (True, False) and d.notes == ["front photo was upside down; rotated it"]
    assert d.p_back is None and d.p_flip is not None


def test_swapped_pair_still_runs_the_orientation_check_after_the_short_circuit():
    d = _guard().decide(_feats(side=(10.0, -10.0), flip=(10.0, -10.0)))
    # Row 0 (the real back) is flipped; after the swap it is the back face.
    assert d.swap and d.flip == (False, True)
    assert d.notes[1:] == ["back photo was upside down; rotated it"]


def test_same_side_pair_reports_p_back_but_no_p_flip():
    d = _guard().decide(_feats(side=(10.0, 10.0), flip=(10.0, 10.0)))
    assert d.p_back is not None and d.p_back[0] > 0.99 and d.p_flip is None
    assert d.notes == ["both photos look like card backs"]
