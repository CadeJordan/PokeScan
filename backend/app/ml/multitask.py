"""Phase C: multi-task model that adds factor heads on top of the Phase A backbone.

Hooks:
- `MultiTaskGradeModel` reuses the dual-stream ConvNeXt backbone from Phase A.
- Adds three regression heads: corners, edges, surface (each predicts a 1-10 sub-grade).
- Centering is NOT a head - it's classical CV (see backend.app.ml.centering).
- A YOLO/RT-DETR defect detector runs in parallel as an interpretability layer.

Training combines:
    L = w_grade * corn_loss(grade_logits)
      + w_aux  * mse(expected_grade, true_grade)
      + sum_k w_k * smoothl1(factor_k_pred, factor_k_label)

Factor heads are *only* trained on the hand-annotated subset. On unlabeled
batches the per-factor terms are masked out so we don't pollute the gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from backend.app.ml.model import GradeModel, NUM_THRESHOLDS, decode_corn_logits

FACTOR_NAMES = ("corners", "edges", "surface")


@dataclass(slots=True)
class MultiTaskOutput:
    grade_logits: torch.Tensor   # (B, K-1) CORN logits for overall grade
    factors: dict[str, torch.Tensor]  # {name: (B,) predicted sub-grade in [1, 10]}


class MultiTaskGradeModel(nn.Module):
    """Wraps a GradeModel and adds per-factor regression heads."""

    def __init__(
        self,
        base: GradeModel | None = None,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        dropout: float = 0.2,
        front_weight: float = 0.7,
    ) -> None:
        super().__init__()
        self.base = base or GradeModel(
            backbone=backbone,
            pretrained=pretrained,
            dropout=dropout,
            front_weight=front_weight,
        )
        feat_dim = self.base.backbone.num_features
        self.factor_heads = nn.ModuleDict(
            {name: nn.Linear(feat_dim, 1) for name in FACTOR_NAMES}
        )

    def _shared_features(self, front: torch.Tensor, back: torch.Tensor) -> torch.Tensor:
        f_front = self.base._embed(front)
        f_back = self.base._embed(back)
        return self.base.front_weight * f_front + (1.0 - self.base.front_weight) * f_back

    def forward(self, front: torch.Tensor, back: torch.Tensor) -> MultiTaskOutput:
        feats = self._shared_features(front, back)
        feats = self.base.dropout(feats)
        grade_logits = self.base.fc(feats)
        # Constrain each factor to (1, 10) via 1 + 9 * sigmoid.
        factors = {
            name: 1.0 + 9.0 * torch.sigmoid(head(feats).squeeze(1))
            for name, head in self.factor_heads.items()
        }
        return MultiTaskOutput(grade_logits=grade_logits, factors=factors)


def multi_task_loss(
    out: MultiTaskOutput,
    grade: torch.Tensor,
    factor_labels: dict[str, torch.Tensor | None],
    factor_masks: dict[str, torch.Tensor | None],
    *,
    grade_weight: float = 0.6,
    factor_weight: float = 0.4 / 3.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combined CORN + per-factor SmoothL1 loss with masking.

    `factor_labels[name]` is a (B,) tensor of true sub-grades or None.
    `factor_masks[name]` is a (B,) bool tensor selecting samples that have a
    label for this factor.
    """
    from backend.app.ml.model import corn_loss  # local to avoid cycles

    grade_loss = corn_loss(out.grade_logits, grade)
    decoded = decode_corn_logits(out.grade_logits)
    aux = nn.functional.mse_loss(decoded.expected_grade, grade.float())

    losses: dict[str, torch.Tensor] = {"grade": grade_loss, "grade_aux": aux}
    factor_terms: list[torch.Tensor] = []
    for name in FACTOR_NAMES:
        labels = factor_labels.get(name)
        mask = factor_masks.get(name)
        if labels is None or mask is None or mask.sum() == 0:
            losses[f"factor_{name}"] = torch.zeros((), device=grade.device)
            continue
        pred = out.factors[name][mask]
        target = labels[mask].float()
        term = nn.functional.smooth_l1_loss(pred, target)
        losses[f"factor_{name}"] = term
        factor_terms.append(term)

    total = grade_weight * grade_loss + 0.1 * aux + factor_weight * sum(
        factor_terms
    ) if factor_terms else grade_weight * grade_loss + 0.1 * aux
    return total, losses
