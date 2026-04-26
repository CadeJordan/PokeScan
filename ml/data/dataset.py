"""PyTorch Dataset + Lightning DataModule for PSA Pokemon cards.

Each example yields:
    front:    (3, H, W) float32 tensor in [0, 1]
    back:     (3, H, W) float32 tensor in [0, 1]
    grade:    long scalar in {1..10}
    cert:     str
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from backend.app.core.config import get_settings
from ml.data.db import connect

# Standard Pokemon card aspect ratio is 2.5 x 3.5 in (= 5:7). 600 x 840 keeps
# defects visible while remaining trainable on a single GPU.
DEFAULT_H = 840
DEFAULT_W = 600


@dataclass(slots=True)
class CardRecord:
    cert_number: str
    front_path: Path
    back_path: Path
    grade: int


def _build_transforms(image_h: int, image_w: int, train: bool) -> A.Compose:
    """Augmentations are deliberately conservative.

    Anything that could change the apparent grade (added scratches, blur,
    aggressive cropping at edges) is forbidden. We DO allow tiny rotations,
    color jitter, and small shifts because we expect the dataset to contain
    those naturally from the slab photos.
    """
    if train:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=max(image_h, image_w)),
                A.PadIfNeeded(image_h, image_w, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.Affine(
                    rotate=(-3, 3),
                    translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                    scale=(0.98, 1.02),
                    p=0.7,
                ),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.0, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max(image_h, image_w)),
            A.PadIfNeeded(image_h, image_w, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class PSACardDataset(Dataset):
    def __init__(
        self,
        records: list[CardRecord],
        image_h: int = DEFAULT_H,
        image_w: int = DEFAULT_W,
        train: bool = False,
    ) -> None:
        self.records = records
        self.transform = _build_transforms(image_h, image_w, train)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        front = _load_image(rec.front_path)
        back = _load_image(rec.back_path)
        front_t = self.transform(image=front)["image"]
        back_t = self.transform(image=back)["image"]
        return {
            "front": front_t,
            "back": back_t,
            "grade": torch.tensor(rec.grade, dtype=torch.long),
            "cert": rec.cert_number,
        }


def load_records(split: str) -> list[CardRecord]:
    settings = get_settings()
    images_root = settings.images_dir.parent  # data/
    out: list[CardRecord] = []
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT cert_number, grade_int, front_path, back_path FROM certs "
            "WHERE is_pokemon=1 AND has_images=1 AND grade_int IS NOT NULL "
            "  AND front_path IS NOT NULL AND back_path IS NOT NULL "
            "  AND split=?",
            (split,),
        ).fetchall()
    for r in rows:
        front = images_root / r["front_path"]
        back = images_root / r["back_path"]
        if not (front.exists() and back.exists()):
            continue
        out.append(
            CardRecord(
                cert_number=r["cert_number"],
                front_path=front,
                back_path=back,
                grade=int(r["grade_int"]),
            )
        )
    return out


def _balanced_sampler(records: list[CardRecord]) -> WeightedRandomSampler:
    """Inverse-frequency weighting so rare grades (1-4) appear more often."""
    counts: dict[int, int] = {}
    for r in records:
        counts[r.grade] = counts.get(r.grade, 0) + 1
    weights = torch.tensor(
        [1.0 / counts[r.grade] for r in records], dtype=torch.double
    )
    return WeightedRandomSampler(weights, num_samples=len(records), replacement=True)


class PSACardDataModule:
    """Plain DataModule (no Lightning subclass to keep import light)."""

    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 4,
        image_h: int = DEFAULT_H,
        image_w: int = DEFAULT_W,
        balanced: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_h = image_h
        self.image_w = image_w
        self.balanced = balanced

    def train_loader(self) -> DataLoader:
        records = load_records("train")
        ds = PSACardDataset(records, self.image_h, self.image_w, train=True)
        sampler = _balanced_sampler(records) if self.balanced else None
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_loader(self) -> DataLoader:
        ds = PSACardDataset(load_records("val"), self.image_h, self.image_w, train=False)
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_loader(self) -> DataLoader:
        ds = PSACardDataset(load_records("test"), self.image_h, self.image_w, train=False)
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
