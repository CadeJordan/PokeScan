"""LightningModule wrapping the GradeModel + CORN loss."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from backend.app.ml.model import GradeModel, corn_loss, decode_corn_logits
from backend.app.ml.multitask import FACTOR_NAMES, MultiTaskGradeModel, multi_task_loss
from ml.training.metrics import exact_match, mean_absolute_error, off_by_one


class _EpochQWKMixin:
    """Accumulates hard predictions over an epoch and logs ``{stage}/qwk``.

    QWK is not decomposable over batches, so it is computed once at epoch end
    from all val/test predictions (single-process; with DDP each rank would
    log its own shard).
    """

    _QWK_STAGES = ("val", "test")

    def _qwk_reset(self, stage: str) -> None:
        if not hasattr(self, "_qwk_buf"):
            self._qwk_buf: dict[str, tuple[list, list]] = {}
        self._qwk_buf[stage] = ([], [])

    def _qwk_accumulate(self, stage: str, pred: torch.Tensor, grade: torch.Tensor) -> None:
        if stage not in self._QWK_STAGES:
            return
        if stage not in getattr(self, "_qwk_buf", {}):
            self._qwk_reset(stage)
        preds, grades = self._qwk_buf[stage]
        preds.append(pred.detach().long().cpu())
        grades.append(grade.detach().long().cpu())

    def _qwk_log(self, stage: str) -> None:
        from ml.eval.evaluate import quadratic_weighted_kappa

        preds, grades = getattr(self, "_qwk_buf", {}).get(stage, ([], []))
        if not preds:
            return
        qwk = quadratic_weighted_kappa(torch.cat(grades).numpy(), torch.cat(preds).numpy())
        if np.isnan(qwk):
            qwk = 0.0
        self.log(f"{stage}/qwk", float(qwk), prog_bar=(stage == "val"))
        self._qwk_reset(stage)

    def on_validation_epoch_start(self) -> None:
        self._qwk_reset("val")

    def on_validation_epoch_end(self) -> None:
        self._qwk_log("val")

    def on_test_epoch_start(self) -> None:
        self._qwk_reset("test")

    def on_test_epoch_end(self) -> None:
        self._qwk_log("test")


class GradeLightningModule(_EpochQWKMixin, pl.LightningModule):
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
        self._qwk_accumulate(stage, out.predicted_grade, grade)
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


class MultiTaskGradeLightningModule(_EpochQWKMixin, pl.LightningModule):
    """Grade + corner/edge/surface heads trained with classical-CV pseudo-labels."""

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        dropout: float = 0.2,
        front_weight: float = 0.7,
        lr: float = 3e-4,
        weight_decay: float = 0.05,
        max_epochs: int = 30,
        grade_weight: float = 0.6,
        factor_weight: float = 0.4 / 3.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = MultiTaskGradeModel(
            backbone=backbone,
            pretrained=pretrained,
            dropout=dropout,
            front_weight=front_weight,
        )

    def _factor_batch(self, batch: dict) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        labels: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}
        for name in FACTOR_NAMES:
            t = batch[name]
            labels[name] = t
            masks[name] = ~torch.isnan(t)
        return labels, masks

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        out = self.model(batch["front"], batch["back"])
        factor_labels, factor_masks = self._factor_batch(batch)
        loss, parts = multi_task_loss(
            out,
            batch["grade"],
            factor_labels,
            factor_masks,
            grade_weight=self.hparams.grade_weight,
            factor_weight=self.hparams.factor_weight,
        )

        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        for key, val in parts.items():
            self.log(f"{stage}/{key}", val, on_step=False, on_epoch=True)

        decoded = decode_corn_logits(out.grade_logits)
        grade = batch["grade"]
        self._qwk_accumulate(stage, decoded.predicted_grade, grade)
        self.log(
            f"{stage}/mae",
            mean_absolute_error(decoded.predicted_grade, grade),
            prog_bar=(stage == "val"),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{stage}/exact",
            exact_match(decoded.predicted_grade, grade),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{stage}/off_by_one",
            off_by_one(decoded.predicted_grade, grade),
            on_step=False,
            on_epoch=True,
        )

        for name in FACTOR_NAMES:
            mask = factor_masks[name]
            if mask.sum() == 0:
                continue
            pred = out.factors[name][mask]
            target = factor_labels[name][mask]
            self.log(
                f"{stage}/mae_{name}",
                torch.nn.functional.l1_loss(pred, target),
                on_step=False,
                on_epoch=True,
            )
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
