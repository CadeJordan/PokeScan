"""`features.export_onnx` writes graph and sidecar to temp names, then renames
(smallfix 12): a failed export leaves the old, consistent pair loadable.

A tiny torch module stands in for DINOv2; everything lives in tmp_path.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from backend.app.ml import features
from backend.app.ml.features import REGION_NAMES, feature_spec_hash
from backend.app.ml.inference import GradeInferenceService

torch = pytest.importorskip("torch")


class _Tiny(torch.nn.Module):
    """(n, 3, H, W) -> (n, len(REGION_NAMES), 3): D = 3."""

    def forward(self, x):
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return pooled[:, None, :].repeat(1, len(REGION_NAMES), 1)


def _save_head(path, dim):
    from backend.app.ml.grade_head import PROBE_NAMES, GradeHead, OrdinalProbe, probe_input

    feats = np.zeros((2, len(REGION_NAMES), dim))
    probes = {
        n: OrdinalProbe(np.zeros(probe_input(feats, n).size), 0.0, np.linspace(-4, 4, 9))
        for n in PROBE_NAMES
    }
    head = GradeHead(probes, OrdinalProbe(np.zeros(4), 0.0, np.linspace(-4, 4, 9)), {})
    head.metadata = {"feature_spec_hash": feature_spec_hash()}
    head.save(path)


@pytest.fixture
def old_pair(tmp_path, monkeypatch):
    """A real exported graph + sidecar and a head that loads with them.

    Returns (graph path, head path, {path: bytes} of the old pair).
    """
    monkeypatch.setattr(features, "build_region_model", lambda pretrained=True: _Tiny().eval())
    graph = tmp_path / "card_features.onnx"
    features.export_onnx(graph)
    head = tmp_path / "grade_head.npz"
    _save_head(head, 3)
    snapshot = {p: p.read_bytes() for p in (graph, graph.with_suffix(".json"))}
    return graph, head, snapshot


def _assert_old_pair_intact(graph, head, snapshot):
    assert {p: p.read_bytes() for p in snapshot} == snapshot
    assert sorted(p.name for p in graph.parent.iterdir()) == [
        "card_features.json", "card_features.onnx", "grade_head.json", "grade_head.npz",
    ]  # no temp files left behind
    svc = GradeInferenceService(head_path=head, features_path=graph)
    assert svc.ready and svc.kind == "probe", svc.metadata.get("probe_load_error")


def test_export_failing_before_writing_leaves_old_pair_loadable(old_pair, serve_env, monkeypatch):
    graph, head, snapshot = old_pair

    def oom(*a, **k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(torch.onnx, "export", oom)
    with pytest.raises(RuntimeError, match="out of memory"):
        features.export_onnx(graph)
    _assert_old_pair_intact(graph, head, snapshot)


def test_export_failing_after_graph_temp_leaves_old_pair_and_no_temp_files(
    old_pair, serve_env, monkeypatch
):
    graph, head, snapshot = old_pair
    real_export = torch.onnx.export
    written = []

    def export_then_record(model, args, f, *a, **k):
        real_export(model, args, f, *a, **k)
        written.append(f)

    real_write_text = features.Path.write_text

    def failing_write_text(self, *a, **k):
        if self.name.endswith(".tmp"):
            raise OSError("disk full")
        return real_write_text(self, *a, **k)

    monkeypatch.setattr(torch.onnx, "export", export_then_record)
    monkeypatch.setattr(features.Path, "write_text", failing_write_text)
    with pytest.raises(OSError, match="disk full"):
        features.export_onnx(graph)
    monkeypatch.setattr(features.Path, "write_text", real_write_text)
    assert written and written[0].endswith(".tmp")  # the graph temp was written
    _assert_old_pair_intact(graph, head, snapshot)


def test_export_replaces_both_files_and_leaves_no_temp_files(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "build_region_model", lambda pretrained=True: _Tiny().eval())
    graph = tmp_path / "card_features.onnx"
    graph.write_bytes(b"old graph")
    graph.with_suffix(".json").write_text(json.dumps({"feature_spec_hash": "0ld0ld0ld0ld"}))
    features.export_onnx(graph)
    assert graph.read_bytes() != b"old graph"
    meta = json.loads(graph.with_suffix(".json").read_text())
    assert meta["feature_spec_hash"] == feature_spec_hash()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "card_features.json", "card_features.onnx",
    ]
