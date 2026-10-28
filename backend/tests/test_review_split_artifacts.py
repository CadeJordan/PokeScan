"""Adversarial review tests for fail-closed probe loading (independent tester, 2026-10-27).

`test_artifacts_from_another_feature_spec_than_the_code_are_refused` pinned a gap
(a strict xfail until the load check compared the artifacts with the code's spec).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from backend.app.ml.features import REGION_NAMES, feature_spec_hash
from backend.app.ml.inference import GradeInferenceService


def _zero_head(dim):
    from backend.app.ml.grade_head import PROBE_NAMES, GradeHead, OrdinalProbe, probe_input

    feats = np.zeros((2, len(REGION_NAMES), dim))
    probes = {
        n: OrdinalProbe(np.zeros(probe_input(feats, n).size), 0.0, np.linspace(-4, 4, 9))
        for n in PROBE_NAMES
    }
    return GradeHead(probes, OrdinalProbe(np.zeros(4), 0.0, np.linspace(-4, 4, 9)), {})


@pytest.fixture
def probe_files(tmp_path, monkeypatch):
    """(head path, features path) with a stub extractor producing 4-dim features."""
    from backend.app.ml import features

    class Session:
        def get_outputs(self):
            return [type("Output", (), {"shape": ["n", len(REGION_NAMES), 4]})()]

        def get_providers(self):
            return ["CPUExecutionProvider"]

    class Extractor:
        def __init__(self, _path):
            self.session = Session()

    monkeypatch.setattr(features, "FeatureExtractor", Extractor)
    feat_path = tmp_path / "f.onnx"
    feat_path.write_bytes(b"stub")
    return tmp_path / "h.npz", feat_path


def _save_head(path, metadata=None):
    head = _zero_head(4)
    head.metadata = dict(metadata or {})
    head.save(path)


def test_artifacts_from_another_feature_spec_than_the_code_are_refused(probe_files, serve_env):
    """Head and sidecar agree with each other but not with the code serving them.

    FEATURE_SPEC's docs say to bump `version` whenever `to_input` changes, and
    serving runs `to_input` from the code (features.py FeatureExtractor.__call__).
    After such a bump, restarting the API without retraining serves an old
    head+graph on new preprocessing: `_check_feature_spec`
    (backend/app/ml/inference.py:172-196) only compared the two artifacts, never
    `feature_spec_hash()`, so this loaded as a ready probe with no error.
    """
    head_path, feat_path = probe_files
    stale = "0123456789ab"
    assert stale != feature_spec_hash()
    _save_head(head_path, {"feature_spec_hash": stale})
    feat_path.with_suffix(".json").write_text(json.dumps({"feature_spec_hash": stale}))
    svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert not svc.ready
    assert "feature_spec_hash" in svc.metadata.get("probe_load_error", "")


def test_non_dict_sidecar_fails_closed(probe_files, serve_env):
    head_path, feat_path = probe_files
    _save_head(head_path, {"feature_spec_hash": feature_spec_hash()})
    feat_path.with_suffix(".json").write_text("[1, 2]")
    svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert not svc.ready
    assert svc.probe_load_error.startswith("ValueError: feature_spec_hash")
    assert "f.json unreadable" in svc.probe_load_error


def test_explicit_model_path_reports_no_probe_error(probe_files, serve_env, tmp_path):
    head_path, feat_path = probe_files  # the probe would load, but isn't asked for
    _save_head(head_path, {"feature_spec_hash": feature_spec_hash()})
    svc = GradeInferenceService(
        model_path=tmp_path / "missing.onnx", head_path=head_path, features_path=feat_path
    )
    assert not svc.ready and svc.kind == "legacy"
    assert "probe_load_error" not in svc.metadata


def test_opt_in_without_legacy_file_stays_not_ready_with_error(serve_env, tmp_path):
    serve_env(ALLOW_LEGACY_MODEL=1, ACTIVE_MODEL_PATH=tmp_path / "gone.onnx")
    svc = GradeInferenceService(head_path=tmp_path / "h.npz", features_path=tmp_path / "f.onnx")
    assert not svc.ready
    assert svc.metadata == {"probe_load_error": "FileNotFoundError: h.npz, f.onnx"}


def test_tta_with_missing_probe_fails_closed_without_raising(serve_env, tmp_path):
    svc = GradeInferenceService(
        tta=True, head_path=tmp_path / "h.npz", features_path=tmp_path / "f.onnx"
    )
    assert not svc.ready
    assert svc.probe_load_error == "FileNotFoundError: h.npz, f.onnx"


def test_tta_with_opt_in_serves_legacy_with_tta(serve_env, tmp_path, tiny_legacy_onnx):
    serve_env(ALLOW_LEGACY_MODEL=1, ACTIVE_MODEL_PATH=tiny_legacy_onnx)
    svc = GradeInferenceService(
        tta=True, head_path=tmp_path / "h.npz", features_path=tmp_path / "f.onnx"
    )
    assert svc.ready and svc.kind == "legacy" and svc.tta
    assert svc.probe_load_error
