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
from ml.data.dataset import PSACardDataModule, precrop_coverage
from ml.training.lightning_module import GradeLightningModule, MultiTaskGradeLightningModule

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True)])
log = logging.getLogger("train")

app = typer.Typer(add_completion=False)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def monitor_settings(cfg: dict) -> tuple[str, str]:
    """(metric, mode) for checkpointing + early stopping.

    Default is val/qwk (maximise): MAE rewards always predicting the majority
    grade, QWK does not. ``monitor_mode`` defaults from the metric name.
    """
    monitor = cfg.get("monitor", "val/qwk")
    default_mode = "max" if monitor.rsplit("/", 1)[-1] in ("qwk", "exact", "off_by_one") else "min"
    mode = cfg.get("monitor_mode", default_mode)
    if mode not in ("min", "max"):
        raise ValueError(f"monitor_mode must be 'min' or 'max', got {mode!r}")
    return monitor, mode


def checkpoint_filename(monitor: str) -> str:
    """e.g. 'best-{epoch:02d}-qwk{val/qwk:.3f}' -> best-07-qwk0.512.ckpt."""
    tag = monitor.rsplit("/", 1)[-1]
    return f"best-{{epoch:02d}}-{tag}{{{monitor}:.3f}}"


MIN_PRECROP_COVERAGE = 0.90


def check_training_inputs(
    use_precrop: bool,
    crop_card: bool,
    crop_kinds: tuple[str, ...] | None = None,
    min_coverage: float = MIN_PRECROP_COVERAGE,
) -> None:
    """Fail fast instead of training on raw slab photos or a gutted train split.

    Raw PSA slab photos show the grade on the label, so a model trained on
    them learns to read the label. `load_records` drops certs without a
    usable precrop; if that leaves less than `min_coverage` of the train
    split, precrop is stale or detection broke - stop rather than train on a
    skewed subset.
    """
    if not use_precrop and not crop_card:
        raise SystemExit(
            "use_precrop=false and crop_card=false would train on raw slab photos "
            "(the PSA label shows the grade). Enable one of them."
        )
    if not use_precrop:
        return
    kept, total = precrop_coverage("train", crop_kinds)
    cov = kept / total if total else 0.0
    log.info("train precrop coverage: %d / %d (%.1f%%)", kept, total, 100 * cov)
    if cov < min_coverage:
        raise SystemExit(
            f"only {kept}/{total} ({cov:.1%}) train certs have a usable precrop "
            f"(need >= {min_coverage:.0%}). Run `python -m ml.data.precrop --force`."
        )


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
    sampler_power: float | None = typer.Option(
        None, help="Override config sampler_power (0=uniform, 0.5=1/sqrt, 1=inverse freq)."
    ),
    aug_strength: float | None = typer.Option(None, help="Override config aug_strength."),
    run_name: str | None = typer.Option(
        None, help="Sub-folder for checkpoints/logs (keeps experiment runs apart)."
    ),
) -> None:
    cfg = _load_yaml(config)
    if sampler_power is not None:
        cfg["sampler_power"] = sampler_power
    if aug_strength is not None:
        cfg["aug_strength"] = aug_strength
    if run_name is not None:
        cfg["run_name"] = run_name
    pl.seed_everything(cfg.get("seed", 42), workers=True)
    settings = get_settings()

    dm = PSACardDataModule(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        image_h=cfg["image_h"],
        image_w=cfg["image_w"],
        balanced=cfg.get("balanced_sampler", True),
        crop_card=cfg.get("crop_card", False),
        use_precrop=cfg.get("use_precrop", True),
        sampler_power=cfg.get("sampler_power", 0.5),
        aug_strength=cfg.get("aug_strength", 1.0),
        crop_kinds=cfg.get("crop_kinds"),
    )
    check_training_inputs(dm.use_precrop, dm.crop_card, dm.crop_kinds)
    log.info(
        "sampler_power=%.2f aug_strength=%.2f", dm.sampler_power, dm.aug_strength
    )

    module: GradeLightningModule | MultiTaskGradeLightningModule
    if cfg.get("multitask", False):
        module = MultiTaskGradeLightningModule(
            backbone=cfg["backbone"],
            pretrained=cfg["pretrained"],
            dropout=cfg["dropout"],
            front_weight=cfg["front_weight"],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            max_epochs=cfg["max_epochs"],
            grade_weight=cfg.get("grade_weight", 0.6),
            factor_weight=cfg.get("factor_weight", 0.4 / 3.0),
        )
        log.info("multitask mode: grade + corners/edges/surface pseudo-label heads")
    else:
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

    monitor, mode = monitor_settings(cfg)
    run_name = cfg.get("run_name")
    ckpt_dir = settings.models_dir / "checkpoints"
    if run_name:
        ckpt_dir = ckpt_dir / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log.info("checkpoints -> %s (monitor=%s, mode=%s)", ckpt_dir, monitor, mode)
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=checkpoint_filename(monitor),
            monitor=monitor,
            mode=mode,
            save_top_k=3,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(monitor=monitor, patience=cfg["early_stop_patience"], mode=mode),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    loggers: list = [
        CSVLogger(save_dir=settings.models_dir / "logs", name=run_name or "pokescan")
    ]
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
