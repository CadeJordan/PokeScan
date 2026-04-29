"""Export a trained Lightning checkpoint to TorchScript + ONNX.

Bakes temperature scaling into the exported graph so inference clients don't
need to know about it.

Usage:
    python -m ml.training.export --checkpoint data/models/checkpoints/best-XX.ckpt
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import typer
from rich.logging import RichHandler

from backend.app.core.config import get_settings
from backend.app.ml.model import GradeModel, NUM_THRESHOLDS
from ml.data.dataset import PSACardDataModule
from ml.training.calibrate import fit_temperature
from ml.training.lightning_module import GradeLightningModule

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True)])
log = logging.getLogger("export")

app = typer.Typer(add_completion=False)


class CalibratedExportModule(nn.Module):
    """Inference graph: forward(front, back) -> (B, K-1) temperature-scaled logits.

    Confidence + ranks are computed on the consumer side via decode_corn_logits.
    """

    def __init__(self, model: GradeModel, temperature: float) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, front: torch.Tensor, back: torch.Tensor) -> torch.Tensor:
        return self.model(front, back) / self.temperature


@app.command()
def main(
    checkpoint: Path = typer.Option(..., exists=True, readable=True),
    out_torchscript: Path = typer.Option(
        Path("data/models/grade_model.pt"), help="TorchScript output path."
    ),
    out_onnx: Path = typer.Option(
        Path("data/models/grade_model.onnx"), help="ONNX output path."
    ),
    image_h: int = typer.Option(840),
    image_w: int = typer.Option(600),
    batch_size: int = typer.Option(8),
    skip_calibration: bool = typer.Option(False, help="Skip temperature scaling."),
) -> None:
    settings = get_settings()
    log.info("loading checkpoint: %s", checkpoint)
    lit = GradeLightningModule.load_from_checkpoint(str(checkpoint), map_location="cpu")
    model: GradeModel = lit.model.eval()

    if skip_calibration:
        T = 1.0
        cal = None
    else:
        log.info("fitting temperature on validation split")
        dm = PSACardDataModule(
            batch_size=batch_size, num_workers=0, image_h=image_h, image_w=image_w, balanced=False
        )
        cal = fit_temperature(model, dm.val_loader(), device="cpu")
        T = cal.temperature
        log.info(
            "temperature: %.4f  (NLL before=%.4f -> after=%.4f)",
            cal.temperature, cal.nll_before, cal.nll_after,
        )

    export_module = CalibratedExportModule(model, T).eval()

    out_torchscript.parent.mkdir(parents=True, exist_ok=True)
    dummy_front = torch.randn(1, 3, image_h, image_w)
    dummy_back = torch.randn(1, 3, image_h, image_w)

    log.info("exporting TorchScript -> %s", out_torchscript)
    traced = torch.jit.trace(export_module, (dummy_front, dummy_back), strict=False)
    traced.save(str(out_torchscript))

    log.info("exporting ONNX -> %s", out_onnx)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_module,
        (dummy_front, dummy_back),
        str(out_onnx),
        input_names=["front", "back"],
        output_names=["corn_logits"],
        dynamic_axes={
            "front": {0: "batch"},
            "back": {0: "batch"},
            "corn_logits": {0: "batch"},
        },
        opset_version=17,
    )

    metadata = {
        "torchscript": str(out_torchscript),
        "onnx": str(out_onnx),
        "image_h": image_h,
        "image_w": image_w,
        "num_thresholds": NUM_THRESHOLDS,
        "front_weight": float(model.front_weight),
        "temperature": float(T),
        "calibration_nll_before": cal.nll_before if cal else None,
        "calibration_nll_after": cal.nll_after if cal else None,
    }
    meta_path = out_onnx.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2))
    log.info("wrote metadata -> %s", meta_path)

    # Promote this artifact to the active model path the API will load.
    settings.active_model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.active_model_path.write_bytes(out_onnx.read_bytes())
    log.info("active model set: %s", settings.active_model_path)


if __name__ == "__main__":
    app()
