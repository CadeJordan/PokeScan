"""PyTorch Dataset + Lightning DataModule for PSA Pokemon cards.

Each example yields:
    front:    (3, H, W) float32 tensor in [0, 1]
    back:     (3, H, W) float32 tensor in [0, 1]
    grade:    long scalar in {1..10}
    cert:     str
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import albumentations as A  # noqa: N812
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from backend.app.core.config import get_settings
from ml.data.db import connect

log = logging.getLogger(__name__)

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
    corners: float | None = None
    edges: float | None = None
    surface: float | None = None


def _photometric_transforms(strength: float) -> list[A.BasicTransform]:
    """Lighting / colour augmentations that do not change apparent condition.

    Brightness, contrast, gamma, white balance, a small hue shift and a rare
    mild CLAHE all mimic different phone cameras and room lighting. Blur, noise,
    scratches and crops are deliberately absent (see CLAUDE.md): they would
    change what a grader sees. ``strength`` scales every range linearly;
    0 disables the photometric block entirely.
    """
    s = float(strength)
    if s <= 0:
        return []
    return [
        A.RandomBrightnessContrast(
            brightness_limit=0.25 * s, contrast_limit=0.2 * s, p=0.7
        ),
        A.RandomGamma(gamma_limit=(100 - 20 * s, 100 + 20 * s), p=0.3),
        # White balance: colour temperature from warm indoor to cool daylight.
        A.PlanckianJitter(
            mode="blackbody",
            temperature_limit=(int(max(3000, 6500 - 2000 * s)), int(min(15000, 6500 + 3000 * s))),
            p=0.3,
        ),
        A.HueSaturationValue(
            hue_shift_limit=3 * s, sat_shift_limit=10 * s, val_shift_limit=0, p=0.3
        ),
        A.CLAHE(clip_limit=(1.0, 1.0 + 1.0 * s), tile_grid_size=(8, 8), p=0.05 * s),
    ]


def _build_transforms(
    image_h: int, image_w: int, train: bool, aug_strength: float = 1.0
) -> A.Compose:
    """Augmentations are deliberately conservative.

    Anything that could change the apparent grade (added scratches, blur,
    noise, aggressive cropping at edges) is forbidden. We DO allow tiny
    rotations, small shifts and condition-preserving lighting/colour changes
    (``_photometric_transforms``) because real user photos vary in exactly
    those ways. Validation/test transforms are fully deterministic.
    """
    if train:
        return A.Compose(
            [
                A.LongestMaxSize(max_size=max(image_h, image_w)),
                A.PadIfNeeded(image_h, image_w, border_mode=cv2.BORDER_CONSTANT, fill=0),
                A.Affine(
                    rotate=(-3, 3),
                    translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                    scale=(0.98, 1.02),
                    p=0.7,
                ),
                *_photometric_transforms(aug_strength),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max(image_h, image_w)),
            A.PadIfNeeded(image_h, image_w, border_mode=cv2.BORDER_CONSTANT, fill=0),
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
        crop_card: bool = False,
        aug_strength: float = 1.0,
    ) -> None:
        self.records = records
        self.crop_card = crop_card
        self.transform = _build_transforms(image_h, image_w, train, aug_strength)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        front = _load_image(rec.front_path)
        back = _load_image(rec.back_path)
        if self.crop_card:
            from backend.app.ml.preprocess import detect_and_crop

            front = detect_and_crop(front).image
            back = detect_and_crop(back).image
        front_t = self.transform(image=front)["image"]
        back_t = self.transform(image=back)["image"]
        return {
            "front": front_t,
            "back": back_t,
            "grade": torch.tensor(rec.grade, dtype=torch.long),
            "cert": rec.cert_number,
            "corners": torch.tensor(
                rec.corners if rec.corners is not None else float("nan"), dtype=torch.float32
            ),
            "edges": torch.tensor(
                rec.edges if rec.edges is not None else float("nan"), dtype=torch.float32
            ),
            "surface": torch.tensor(
                rec.surface if rec.surface is not None else float("nan"), dtype=torch.float32
            ),
        }


def _usable_crop(r, data_root: Path, crop_kinds: tuple[str, ...] | None) -> bool:
    """Precrop row is safe to train on: both sides cropped on a detected outline."""
    if not (r["cropped_front_path"] and r["cropped_back_path"]):
        return False
    if r["crop_found_front"] != 1 or r["crop_found_back"] != 1:
        return False
    if crop_kinds is not None and not (
        r["crop_kind_front"] in crop_kinds and r["crop_kind_back"] in crop_kinds
    ):
        return False
    return (data_root / r["cropped_front_path"]).exists() and (
        data_root / r["cropped_back_path"]
    ).exists()


def load_records(
    split: str,
    *,
    use_precrop: bool = True,
    crop_kinds: tuple[str, ...] | None = None,
) -> list[CardRecord]:
    """Records for `split`.

    use_precrop=True returns only certs whose front and back precrops exist
    and were cut on a detected card outline (`crop_found_*`=1), optionally
    restricted to `crop_kinds` (e.g. ("card",) for tight crops only). It never
    falls back to the raw photo: a raw PSA slab photo shows the grade on its
    label, so training on it leaks the answer. Dropped certs are logged.

    use_precrop=False returns the raw photos, for callers that run detection
    themselves (evaluate.py, or the dataset with crop_card=True).
    """
    settings = get_settings()
    data_root = settings.data_dir
    out: list[CardRecord] = []
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT cert_number, grade_int, front_path, back_path, "
            "cropped_front_path, cropped_back_path, "
            "crop_found_front, crop_found_back, crop_kind_front, crop_kind_back, "
            "corners_subgrade, edges_subgrade, surface_subgrade FROM certs "
            "WHERE is_pokemon=1 AND has_images=1 AND grade_int IS NOT NULL "
            "  AND front_path IS NOT NULL AND back_path IS NOT NULL "
            "  AND split=?",
            (split,),
        ).fetchall()
    dropped = 0
    for r in rows:
        if use_precrop:
            if not _usable_crop(r, data_root, crop_kinds):
                dropped += 1
                continue
            front = data_root / r["cropped_front_path"]
            back = data_root / r["cropped_back_path"]
        else:
            front = data_root / r["front_path"]
            back = data_root / r["back_path"]
            if not (front.exists() and back.exists()):
                continue
        out.append(
            CardRecord(
                cert_number=r["cert_number"],
                front_path=front,
                back_path=back,
                grade=int(r["grade_int"]),
                corners=float(r["corners_subgrade"]) if r["corners_subgrade"] is not None else None,
                edges=float(r["edges_subgrade"]) if r["edges_subgrade"] is not None else None,
                surface=float(r["surface_subgrade"]) if r["surface_subgrade"] is not None else None,
            )
        )
    if dropped:
        log.warning(
            "load_records(%s): dropped %d / %d certs without a usable precrop "
            "(missing, or card outline not found); run `python -m ml.data.precrop`",
            split,
            dropped,
            len(rows),
        )
    return out


def precrop_coverage(split: str, crop_kinds: tuple[str, ...] | None = None) -> tuple[int, int]:
    """(certs with a usable precrop, all certs) for `split`."""
    total = len(load_records(split, use_precrop=False))
    return len(load_records(split, use_precrop=True, crop_kinds=crop_kinds)), total


def sampler_weights(grades: list[int], power: float = 0.5) -> torch.Tensor:
    """Per-sample weights ``count(grade) ** -power``.

    power=0 -> uniform (natural distribution), power=1 -> full inverse
    frequency (every grade drawn equally often; with 1 grade-1 card vs ~1000
    grade-10s that card is drawn ~1000x more, which biased the model low),
    power=0.5 -> 1/sqrt(count), the default compromise.
    """
    if power < 0:
        raise ValueError("sampler_power must be >= 0")
    counts: dict[int, int] = {}
    for g in grades:
        counts[g] = counts.get(g, 0) + 1
    return torch.tensor([counts[g] ** -float(power) for g in grades], dtype=torch.double)


def _balanced_sampler(records: list[CardRecord], power: float = 0.5) -> WeightedRandomSampler:
    """Frequency-tempered sampler so rare grades (1-6) appear more often."""
    weights = sampler_weights([r.grade for r in records], power)
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
        crop_card: bool = False,
        use_precrop: bool = True,
        sampler_power: float = 0.5,
        aug_strength: float = 1.0,
        crop_kinds: tuple[str, ...] | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_h = image_h
        self.image_w = image_w
        self.balanced = balanced
        self.crop_card = crop_card
        self.use_precrop = use_precrop
        self.sampler_power = sampler_power
        self.aug_strength = aug_strength
        self.crop_kinds = tuple(crop_kinds) if crop_kinds else None

    def train_loader(self) -> DataLoader:
        records = load_records("train", use_precrop=self.use_precrop, crop_kinds=self.crop_kinds)
        ds = PSACardDataset(
            records,
            self.image_h,
            self.image_w,
            train=True,
            crop_card=self.crop_card,
            aug_strength=self.aug_strength,
        )
        sampler = _balanced_sampler(records, self.sampler_power) if self.balanced else None
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
        ds = PSACardDataset(
            load_records("val", use_precrop=self.use_precrop, crop_kinds=self.crop_kinds),
            self.image_h,
            self.image_w,
            train=False,
            crop_card=self.crop_card,
        )
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_loader(self) -> DataLoader:
        ds = PSACardDataset(
            load_records("test", use_precrop=self.use_precrop, crop_kinds=self.crop_kinds),
            self.image_h,
            self.image_w,
            train=False,
            crop_card=self.crop_card,
        )
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
