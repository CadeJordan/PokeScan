"""Independent re-test of commit a92911c (load check vs the code's feature spec).

Pins edge cases of `GradeInferenceService._check_feature_spec` that the fixer's
tests don't name explicitly. Synthetic artifacts in tmp_path only.
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


def _sidecar(feat_path, payload):
    feat_path.with_suffix(".json").write_text(
        payload if isinstance(payload, str) else json.dumps(payload)
    )


def _warned(caplog):
    return any(
        r.levelname == "WARNING" and "feature_spec_hash missing" in r.getMessage()
        for r in caplog.records
    )


# --- head with a hash: strict ------------------------------------------------ #
@pytest.mark.parametrize(
    "mangle", [str.upper, lambda h: f" {h}", lambda h: f"{h}\n", lambda h: h[:-1]]
)
def test_head_hash_must_match_code_exactly(probe_files, serve_env, mangle):
    head_path, feat_path = probe_files
    bad = mangle(feature_spec_hash())
    _save_head(head_path, {"feature_spec_hash": bad})
    _sidecar(feat_path, {"feature_spec_hash": bad})
    svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert not svc.ready
    assert "!= code" in svc.probe_load_error


def test_sidecar_hash_case_differs_from_head_fails_closed(probe_files, serve_env):
    head_path, feat_path = probe_files
    _save_head(head_path, {"feature_spec_hash": feature_spec_hash()})
    _sidecar(feat_path, {"feature_spec_hash": feature_spec_hash().upper()})
    svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert not svc.ready
    assert "f.json" in svc.probe_load_error and "!= head" in svc.probe_load_error


def test_all_three_agree_loads_without_warning(probe_files, serve_env, caplog):
    head_path, feat_path = probe_files
    _save_head(head_path, {"feature_spec_hash": feature_spec_hash()})
    _sidecar(feat_path, {"feature_spec_hash": feature_spec_hash(), "exported_at": "x"})
    with caplog.at_level("WARNING", logger="backend.app.ml.inference"):
        svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert svc.ready and svc.kind == "probe"
    assert "probe_load_error" not in svc.metadata
    assert not _warned(caplog)


# --- head without a hash: lenient, except a sidecar from another spec -------- #
@pytest.mark.parametrize(
    "payload",
    [
        {"feature_spec_hash": None},  # dict without a usable hash
        {"exported_at": "x"},  # dict without the key
        "[1, 2]",  # parses, not a dict
        "null",
        '"428e7a366e2e"',  # a bare string, not a dict
    ],
)
def test_no_hash_head_with_hashless_sidecar_loads_with_warning(
    probe_files, serve_env, caplog, payload
):
    head_path, feat_path = probe_files
    _save_head(head_path)
    _sidecar(feat_path, payload)
    with caplog.at_level("WARNING", logger="backend.app.ml.inference"):
        svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert svc.ready and svc.kind == "probe"
    assert _warned(caplog)


def test_no_hash_head_with_sidecar_matching_code_loads_with_warning(
    probe_files, serve_env, caplog
):
    # The tester's "sidecar present, head hash missing, sidecar matches code" case.
    head_path, feat_path = probe_files
    _save_head(head_path)
    _sidecar(feat_path, {"feature_spec_hash": feature_spec_hash()})
    with caplog.at_level("WARNING", logger="backend.app.ml.inference"):
        svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert svc.ready and svc.kind == "probe"
    assert _warned(caplog)


@pytest.mark.parametrize("empty", ["", None, 0])
def test_falsy_head_hash_counts_as_no_hash(probe_files, serve_env, caplog, empty):
    # Observed behaviour (nit): "" / null / 0 in grade_head.json is treated as a
    # pre-spec head, so it loads with a WARNING rather than failing closed.
    head_path, feat_path = probe_files
    _save_head(head_path, {"feature_spec_hash": empty})
    with caplog.at_level("WARNING", logger="backend.app.ml.inference"):
        svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert svc.ready and svc.kind == "probe"
    assert _warned(caplog)


def test_unreadable_sidecar_os_error_fails_closed_even_for_no_hash_head(probe_files, serve_env):
    # Observed behaviour (nit): an OS-level read error (here: a directory at the
    # sidecar path) isn't caught by _check_feature_spec, so even a pre-spec head
    # fails closed (with the OSError in probe_load_error) instead of warning.
    head_path, feat_path = probe_files
    _save_head(head_path)
    feat_path.with_suffix(".json").mkdir()
    svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert not svc.ready
    assert svc.probe_load_error.split(":")[0] in ("PermissionError", "IsADirectoryError")


# --- no stale error ---------------------------------------------------------- #
def test_failed_then_successful_load_has_no_stale_error(probe_files, serve_env):
    head_path, feat_path = probe_files
    _save_head(head_path, {"feature_spec_hash": feature_spec_hash()})
    bad = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert not bad.ready and "f.json missing" in bad.probe_load_error
    _sidecar(feat_path, {"feature_spec_hash": feature_spec_hash()})
    good = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert good.ready and "probe_load_error" not in good.metadata
    assert good.metadata == {"feature_spec_hash": feature_spec_hash()}


def test_with_opt_in_and_spec_mismatch_legacy_serves_and_error_kept(
    probe_files, serve_env, tiny_legacy_onnx
):
    head_path, feat_path = probe_files
    serve_env(ALLOW_LEGACY_MODEL=1, ACTIVE_MODEL_PATH=tiny_legacy_onnx)
    _save_head(head_path, {"feature_spec_hash": feature_spec_hash()})  # no sidecar
    svc = GradeInferenceService(head_path=head_path, features_path=feat_path)
    assert svc.ready and svc.kind == "legacy"
    assert "f.json missing" in svc.metadata["probe_load_error"]
