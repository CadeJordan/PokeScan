"""Frozen DINOv2 region features for a dewarped card face.

The grade model is a frozen self-supervised ViT (DINOv2-B with registers)
plus small linear ordinal heads (`grade_head.py`). With ~2.6k labelled
certs, fine-tuning a CNN end to end collapsed to predicting the majority
grade; frozen features + a linear probe do not have that failure mode and
are cheap to refit as data grows.

Each face is resized to FEAT_H x FEAT_W (a 43 x 31 grid of 14 px patches) and
summarised as REGION_NAMES vectors: the CLS token, the mean patch token, and
mean tokens over the four corners, four edge strips and the centre. Defects are
local, so these region means carry more grade signal than one global vector.

`build_region_model` is used at training/export time (needs torch + timm);
serving runs the exported ONNX graph through `FeatureExtractor`.

`FEATURE_SPEC` fingerprints the feature definition. The training cache, the
exported graph (its `card_features.json` sidecar) and the head metadata all
record `feature_spec_hash()`, so artifacts built from different definitions
are re-extracted, re-exported or refused at load instead of silently mixed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

BACKBONE = "vit_base_patch14_reg4_dinov2.lvd142m"
FEAT_H, FEAT_W = 602, 434
PATCH = 14
GRID_H, GRID_W = FEAT_H // PATCH, FEAT_W // PATCH  # 43 x 31
CORNER_PATCHES = 5  # corner blocks are 5x5 patches (~70 px on the 600x840 crop)
EDGE_PATCHES = 3    # edge strips are 3 patches deep

REGION_NAMES = (
    "cls",
    "mean",
    "corner_tl",
    "corner_tr",
    "corner_br",
    "corner_bl",
    "edge_top",
    "edge_right",
    "edge_bottom",
    "edge_left",
    "center",
)
CORNER_SLICE = slice(2, 6)
EDGE_SLICE = slice(6, 10)
CENTER_SLICE = slice(10, 11)
GLOBAL_SLICE = slice(0, 2)

# Bump "version" whenever to_input or region_means changes behaviour; the other
# fields are the constants above.
FEATURE_SPEC: dict = {
    "backbone": BACKBONE,
    "feat_h": FEAT_H,
    "feat_w": FEAT_W,
    "patch": PATCH,
    "corner_patches": CORNER_PATCHES,
    "edge_patches": EDGE_PATCHES,
    "regions": list(REGION_NAMES),
    "version": 1,
}


def feature_spec_hash() -> str:
    """First 12 hex chars of the sha256 of FEATURE_SPEC (sorted-key JSON)."""
    blob = json.dumps(FEATURE_SPEC, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_input(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 RGB card crop -> (3, FEAT_H, FEAT_W) float32, ImageNet-normalised."""
    small = cv2.resize(rgb, (FEAT_W, FEAT_H), interpolation=cv2.INTER_AREA)
    x = (small.astype(np.float32) / 255.0 - _MEAN) / _STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def region_means(patches):
    """(B, GRID_H, GRID_W, D) patch tokens -> list of (B, D) region means.

    Works on numpy arrays and torch tensors alike (used inside the exported graph).
    """
    k, e = CORNER_PATCHES, EDGE_PATCHES
    regions = [
        patches[:, :k, :k],
        patches[:, :k, -k:],
        patches[:, -k:, -k:],
        patches[:, -k:, :k],
        patches[:, :e, k:-k],
        patches[:, k:-k, -e:],
        patches[:, -e:, k:-k],
        patches[:, k:-k, :e],
        patches[:, k:-k, k:-k],
    ]
    return [r.mean((1, 2)) for r in regions]


def build_region_model(pretrained: bool = True):
    """torch module: (B, 3, FEAT_H, FEAT_W) -> (B, len(REGION_NAMES), D)."""
    import timm
    import torch

    class RegionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Fixed img_size resamples the position embedding once at load
            # time; dynamic sizing would do it per call, which ONNX can't export.
            self.backbone = timm.create_model(
                BACKBONE, pretrained=pretrained, num_classes=0, img_size=(FEAT_H, FEAT_W)
            )

        def forward(self, x):
            tokens = self.backbone.forward_features(x)
            cls = tokens[:, 0]
            patches = tokens[:, self.backbone.num_prefix_tokens :].reshape(
                x.shape[0], GRID_H, GRID_W, -1
            )
            return torch.stack([cls, patches.mean((1, 2)), *region_means(patches)], 1)

    return RegionModel().eval()


def export_onnx(path: Path, pretrained: bool = True) -> None:
    """Export the region model to `path`, plus a `.json` sidecar with the feature spec."""
    import torch

    model = build_region_model(pretrained)
    dummy = torch.zeros(2, 3, FEAT_H, FEAT_W)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["image"],
        output_names=["regions"],
        dynamic_axes={"image": {0: "n"}, "regions": {0: "n"}},
        opset_version=17,
    )
    sidecar = {
        "feature_spec": FEATURE_SPEC,
        "feature_spec_hash": feature_spec_hash(),
        "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))


class FeatureExtractor:
    """ONNX Runtime wrapper around the exported region model."""

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)

    def __call__(self, front_rgb: np.ndarray, back_rgb: np.ndarray) -> np.ndarray:
        """-> (2, len(REGION_NAMES), D) float32 for [front, back]."""
        batch = np.stack([to_input(front_rgb), to_input(back_rgb)])
        return self.session.run(["regions"], {"image": batch})[0]
