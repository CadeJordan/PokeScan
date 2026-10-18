"""Exported artifacts agree: TorchScript vs ONNX, and the metadata sidecar is sane.

Both graphs come from `CalibratedExportModule`, so both already include the
temperature from grade_model.json - outputs should match directly.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.model


def test_metadata_sidecar(models_dir):
    meta = json.loads((models_dir / "grade_model.json").read_text())
    assert meta["num_thresholds"] == 9
    assert (meta["image_h"], meta["image_w"]) == (840, 600)
    assert 0.1 < meta["temperature"] < 10
    if meta.get("calibration_nll_after") is not None:
        assert meta["calibration_nll_after"] <= meta["calibration_nll_before"] + 1e-6


def test_onnx_io_signature(models_dir):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(models_dir / "grade_model.onnx"), providers=["CPUExecutionProvider"])
    inputs = {i.name: i.shape for i in sess.get_inputs()}
    assert set(inputs) == {"front", "back"}
    assert inputs["front"][1:] == [3, 840, 600]
    assert [o.name for o in sess.get_outputs()] == ["corn_logits"]


def test_torchscript_matches_onnx(models_dir):
    import onnxruntime as ort

    pt = models_dir / "grade_model.pt"
    if not pt.exists():
        pytest.skip("no TorchScript export")
    torch.manual_seed(0)
    front = torch.randn(2, 3, 840, 600)
    back = torch.randn(2, 3, 840, 600)

    ts = torch.jit.load(str(pt), map_location="cpu").eval()
    with torch.no_grad():
        expected = ts(front, back).numpy()

    sess = ort.InferenceSession(str(models_dir / "grade_model.onnx"), providers=["CPUExecutionProvider"])
    got = sess.run(["corn_logits"], {"front": front.numpy(), "back": back.numpy()})[0]
    assert got.shape == (2, 9)
    np.testing.assert_allclose(got, expected, atol=1e-3, rtol=1e-3)
