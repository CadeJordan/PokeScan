"""YOLO/RT-DETR defect detector loader (Phase C).

Wraps a YOLO weights file trained on hand-annotated corner/edge/surface
defects and produces overlay-friendly detection results to surface in the UI.

Until weights exist on disk, `DefectDetector.ready` is False and the API
returns an empty defect list - the rest of the pipeline is unaffected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DEFECT_CLASSES = (
    "corner_whitening",
    "corner_ding",
    "edge_chip",
    "edge_whitening",
    "surface_scratch",
    "surface_print_defect",
    "surface_dent",
)


@dataclass(slots=True)
class Detection:
    cls: str
    confidence: float
    bbox: tuple[float, float, float, float]  # xyxy normalized to [0, 1]


class DefectDetector:
    def __init__(self, weights_path: Path) -> None:
        self.weights_path = Path(weights_path)
        self._model = None
        self._load()

    def _load(self) -> None:
        if not self.weights_path.exists():
            log.info("defect detector weights not found at %s - skipping", self.weights_path)
            return
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(str(self.weights_path))
            log.info("loaded YOLO defect detector from %s", self.weights_path)
        except ImportError:
            log.warning(
                "ultralytics not installed; pip install ultralytics to enable defect detection",
            )

    @property
    def ready(self) -> bool:
        return self._model is not None

    def detect(self, rgb: np.ndarray, conf_threshold: float = 0.35) -> list[Detection]:
        if self._model is None:
            return []
        results = self._model.predict(rgb, conf=conf_threshold, verbose=False)
        out: list[Detection] = []
        h, w = rgb.shape[:2]
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_idx = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls[0])
                if cls_idx >= len(DEFECT_CLASSES):
                    continue
                conf = float(box.conf.item()) if hasattr(box.conf, "item") else float(box.conf[0])
                xyxy = box.xyxy.tolist()[0]
                out.append(
                    Detection(
                        cls=DEFECT_CLASSES[cls_idx],
                        confidence=conf,
                        bbox=(
                            xyxy[0] / w,
                            xyxy[1] / h,
                            xyxy[2] / w,
                            xyxy[3] / h,
                        ),
                    )
                )
        return out
