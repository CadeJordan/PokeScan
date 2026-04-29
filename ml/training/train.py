"""Train the MVP grade model.

Usage:
    python -m ml.training.train --config ml/training/configs/mvp.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
import typer
import yaml
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from rich.logging import RichHandler

from backend.app.core.config import get_settings
from ml.data.dataset import PSACardDataModule
from ml.training.lightning_module import GradeLightningModule

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True)])
log = logging.getLogger("train")

app = typer.Typer(add_completion=False)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@app.command()
def main(
    config: Path = typer.Option(
        Path("ml/training/configs/mvp.yaml"),
        exists=True,
        readable=True,
        help="YAML config file.",
    ),
    use_wandb: bool = typer.Option(False, help="Log to Weights & Biases."),
    devices: int = typer.Option(1, help="GPUs (or CPU cores if no GPU)."),
) -> None:
    cfg = _load_yaml(config)
    pl.seed_everything(cfg.get("seed", 42), workers=True)
    settings = get_settings()

    dm = PSACardDataModule(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        image_h=cfg["image_h"],
        image_w=cfg["image_w"],
        balanced=cfg.get("balanced_sampler", True),
    )

    module = GradeLightningModule(
        backbone=cfg["backbone"],
        pretrained=cfg["pretrained"],
        dropout=cfg["dropout"],
        front_weight=cfg["front_weight"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        max_epochs=cfg["max_epochs"],
        aux_mse_weight=cfg["aux_mse_weight"],
    )

    ckpt_dir = settings.models_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-{epoch:02d}-{val/mae:.3f}",
            monitor="val/mae",
            mode="min",
            save_top_k=3,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(monitor="val/mae", patience=cfg["early_stop_patience"], mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    loggers: list = [CSVLogger(save_dir=settings.models_dir / "logs", name="pokescan")]
    if use_wandb:
        loggers.append(WandbLogger(project=settings.wandb_project))

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=cfg["max_epochs"],
        accelerator=accelerator,
        devices=devices if accelerator == "gpu" else 1,
        precision=cfg["precision"] if accelerator == "gpu" else 32,
        gradient_clip_val=cfg["gradient_clip_val"],
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=10,
        deterministic=False,
    )

    log.info("starting training (accelerator=%s, devices=%d)", accelerator, devices)
    trainer.fit(module, train_dataloaders=dm.train_loader(), val_dataloaders=dm.val_loader())
    log.info("running test split evaluation")
    trainer.test(module, dataloaders=dm.test_loader(), ckpt_path="best")


if __name__ == "__main__":
    app()
