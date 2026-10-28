"""Export a trained Lightning checkpoint to TorchScript + ONNX.

Bakes temperature scaling and per-threshold bias offsets into the exported
graph (``corn_logits = logits / T + b``), so inference clients don't need to
know about either.

Usage:
    python -m ml.training.export --checkpoint data/models/checkpoints/best-XX.ckpt
    # preview the val calibration without writing any model files:
    python -m ml.training.export --checkpoint ... --calibration-only
    # export side-by-side without replacing the served model:
    python -m ml.training.export --checkpoint ... --out-onnx data/models/candidate.onnx \
        --out-torchscript data/models/candidate.pt --no-activate
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
from backend.app.ml.model import NUM_THRESHOLDS, GradeModel
from ml.data.dataset import PSACardDataModule
from ml.training.calibrate import (
    OBJECTIVES,
    collect_logits,
    fit_temperature_on_logits,
    fit_threshold_offsets,
    offset_metrics,
)
from ml.training.lightning_module import GradeLightningModule, MultiTaskGradeLightningModule

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True)])
log = logging.getLogger("export")

app = typer.Typer(add_completion=False)


class CalibratedExportModule(nn.Module):
    """Inference graph: forward(front, back) -> (B, K-1) calibrated CORN logits.

    Output is ``model(front, back) / temperature + offsets``. Confidence + ranks
    are computed on the consumer side via decode_corn_logits (whose cummin keeps
    thresholds monotone even if offsets would not).
    """

    def __init__(
        self, model: GradeModel, temperature: float, offsets: list[float] | None = None
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        off = torch.zeros(NUM_THRESHOLDS) if offsets is None else torch.as_tensor(
            offsets, dtype=torch.float32
        )
        if off.shape != (NUM_THRESHOLDS,):
            raise ValueError(f"offsets must have {NUM_THRESHOLDS} values, got {tuple(off.shape)}")
        self.register_buffer("offsets", off.float())

    def forward(self, front: torch.Tensor, back: torch.Tensor) -> torch.Tensor:
        return self.model(front, back) / self.temperature + self.offsets


def load_grade_model(checkpoint: Path) -> GradeModel:
    """Grade-only GradeModel from a single-task or multitask Lightning checkpoint."""
    ckpt_meta = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    hparams = ckpt_meta.get("hyper_parameters") or {}
    if "grade_weight" in hparams:
        lit = MultiTaskGradeLightningModule.load_from_checkpoint(str(checkpoint), map_location="cpu")
        log.info("loaded multitask checkpoint (exporting grade backbone only)")
        return lit.model.base.eval()
    lit = GradeLightningModule.load_from_checkpoint(str(checkpoint), map_location="cpu")
    return lit.model.eval()


def _load_offsets_file(path: Path) -> tuple[float | None, list[float]]:
    """Read ``threshold_offsets`` (+ optional temperature) from a calibration/metadata JSON."""
    data = json.loads(path.read_text())
    offsets = data.get("threshold_offsets") or data.get("offsets")
    if offsets is None or len(offsets) != NUM_THRESHOLDS:
        raise typer.BadParameter(f"{path} has no {NUM_THRESHOLDS}-value 'threshold_offsets'")
    return data.get("temperature"), [float(v) for v in offsets]


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
    device: str = typer.Option(
        "auto", help="Device for collecting val logits: auto | cpu | cuda."
    ),
    skip_calibration: bool = typer.Option(False, help="Skip temperature scaling and offsets."),
    fit_offsets: bool = typer.Option(
        True, help="Fit per-threshold bias offsets on val (removes systematic grade bias)."
    ),
    offset_objective: str = typer.Option(
        "mae_qwk", help=f"Offset objective on val: {' | '.join(OBJECTIVES)}."
    ),
    offsets_json: Path | None = typer.Option(
        None, exists=True, readable=True,
        help="Apply offsets (+temperature) from a calibration/metadata JSON instead of fitting.",
    ),
    calibration_only: bool = typer.Option(
        False, help="Fit + report val calibration, write it to --calibration-out, export nothing."
    ),
    calibration_out: Path = typer.Option(
        Path("test-artifacts/calibration.json"), help="Where --calibration-only writes its JSON."
    ),
    activate: bool = typer.Option(
        True, help="Copy the ONNX to ACTIVE_MODEL_PATH (the model the API serves)."
    ),
) -> None:
    settings = get_settings()
    log.info("loading checkpoint: %s", checkpoint)
    model = load_grade_model(checkpoint)
    if offset_objective not in OBJECTIVES:
        raise typer.BadParameter(f"--offset-objective must be one of {OBJECTIVES}")
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

    temp = 1.0
    offsets: list[float] | None = None
    calib: dict = {"temperature": 1.0, "threshold_offsets": None}
    if offsets_json is not None:
        t_file, offsets = _load_offsets_file(offsets_json)
        temp = float(t_file) if t_file is not None else 1.0
        calib.update(temperature=temp, threshold_offsets=offsets, offsets_source=str(offsets_json))
        log.info("using offsets from %s (T=%.4f): %s", offsets_json, temp, offsets)
    elif not skip_calibration:
        log.info("collecting validation logits (device=%s)", dev)
        dm = PSACardDataModule(
            batch_size=batch_size, num_workers=0, image_h=image_h, image_w=image_w,
            balanced=False, crop_card=False, use_precrop=True,
        )
        logits, grades = collect_logits(model, dm.val_loader(), device=dev)
        model.cpu()
        cal = fit_temperature_on_logits(logits, grades)
        temp = cal.temperature
        log.info(
            "temperature: %.4f  (NLL before=%.4f -> after=%.4f)",
            cal.temperature, cal.nll_before, cal.nll_after,
        )
        calib.update(
            temperature=temp,
            n_val=int(len(grades)),
            calibration_nll_before=cal.nll_before,
            calibration_nll_after=cal.nll_after,
            val_metrics_raw=offset_metrics(logits.numpy(), grades.numpy()),
        )
        if fit_offsets:
            res = fit_threshold_offsets(logits, grades, temperature=temp, objective=offset_objective)
            temp, offsets = res.temperature, res.offsets
            calib.update(
                temperature=temp,
                threshold_offsets=offsets,
                offset_objective=offset_objective,
                val_metrics_temperature_only=res.before,
                val_metrics_calibrated=res.after,
                # calibration_nll_* stay the temperature-only fit (raw vs T);
                # the NLL with offsets is val_metrics_calibrated["nll"].
            )
            b, a = res.before, res.after
            log.info("offsets (objective=%s, final T=%.4f): %s", offset_objective, temp,
                     ", ".join(f"{v:+.2f}" for v in offsets))
            log.info("val MAE %.3f -> %.3f | exact %.1f%% -> %.1f%% | QWK %.3f -> %.3f | bias %+.2f -> %+.2f",
                     b["mae"], a["mae"], 100 * b["exact"], 100 * a["exact"], b["qwk"], a["qwk"],
                     b["bias"], a["bias"])

    if calibration_only:
        calibration_out.parent.mkdir(parents=True, exist_ok=True)
        calibration_out.write_text(json.dumps({"checkpoint": str(checkpoint), **calib}, indent=2))
        log.info("wrote calibration -> %s (no model files written)", calibration_out)
        return

    export_module = CalibratedExportModule(model, temp, offsets).eval()

    out_torchscript.parent.mkdir(parents=True, exist_ok=True)
    dummy_front = torch.randn(1, 3, image_h, image_w)
    dummy_back = torch.randn(1, 3, image_h, image_w)

    log.info("exporting TorchScript -> %s", out_torchscript)
    traced = torch.jit.trace(export_module, (dummy_front, dummy_back), strict=False)
    traced.save(str(out_torchscript))

    log.info("exporting ONNX -> %s", out_onnx)
    export_onnx(export_module, out_onnx, image_h, image_w)

    metadata = {
        "torchscript": str(out_torchscript),
        "onnx": str(out_onnx),
        "checkpoint": str(checkpoint),
        "image_h": image_h,
        "image_w": image_w,
        "num_thresholds": NUM_THRESHOLDS,
        "front_weight": float(model.front_weight),
        "calibration_nll_before": None,
        "calibration_nll_after": None,
        **calib,
        "temperature": float(temp),
        "threshold_offsets": offsets,
    }
    meta_path = out_onnx.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2))
    log.info("wrote metadata -> %s", meta_path)

    if activate:
        # Promote this artifact to the active model path the API will load.
        settings.active_model_path.parent.mkdir(parents=True, exist_ok=True)
        if settings.active_model_path.resolve() != out_onnx.resolve():
            settings.active_model_path.write_bytes(out_onnx.read_bytes())
            settings.active_model_path.with_suffix(".json").write_text(meta_path.read_text())
        log.info("active model set: %s", settings.active_model_path)
    else:
        log.info("--no-activate: served model left unchanged (%s)", settings.active_model_path)


def export_onnx(module: nn.Module, out_onnx: Path, image_h: int, image_w: int) -> None:
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    dummy_front = torch.randn(1, 3, image_h, image_w)
    dummy_back = torch.randn(1, 3, image_h, image_w)
    torch.onnx.export(
        module,
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


if __name__ == "__main__":
    app()
