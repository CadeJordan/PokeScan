"""LightningModule wrapping the GradeModel + CORN loss."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from backend.app.ml.model import GradeModel, corn_loss, decode_corn_logits
from ml.training.metrics import exact_match, mean_absolute_error, off_by_one


class GradeLightningModule(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        dropout: float = 0.2,
        front_weight: float = 0.7,
        lr: float = 3e-4,
        weight_decay: float = 0.05,
        max_epochs: int = 30,
        aux_mse_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = GradeModel(
            backbone=backbone,
            pretrained=pretrained,
            dropout=dropout,
            front_weight=front_weight,
        )

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        front = batch["front"]
        back = batch["back"]
        grade = batch["grade"]
        logits = self.model(front, back)
        loss_corn = corn_loss(logits, grade)

        out = decode_corn_logits(logits)
        loss_aux = torch.nn.functional.mse_loss(
            out.expected_grade, grade.float()
        )
        loss = loss_corn + self.hparams.aux_mse_weight * loss_aux

        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}/loss_corn", loss_corn, on_step=False, on_epoch=True)
        self.log(f"{stage}/loss_mse", loss_aux, on_step=False, on_epoch=True)
        self.log(f"{stage}/mae", mean_absolute_error(out.predicted_grade, grade),
                 prog_bar=(stage == "val"), on_step=False, on_epoch=True)
        self.log(f"{stage}/exact", exact_match(out.predicted_grade, grade),
                 on_step=False, on_epoch=True)
        self.log(f"{stage}/off_by_one", off_by_one(out.predicted_grade, grade),
                 on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: dict, _idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, _idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: dict, _idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self) -> dict[str, Any]:
        opt = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = CosineAnnealingLR(opt, T_max=self.hparams.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}
