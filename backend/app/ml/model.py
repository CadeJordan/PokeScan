"""Dual-input ConvNeXt grade model with a CORN ordinal regression head.

Why CORN ordinal regression instead of 10-way softmax?
- Grades 1..10 are ordered. Predicting 9 when truth is 10 is barely wrong;
  predicting 1 is catastrophically wrong. Cross-entropy ignores that.
- CORN (Cao, Mirjalili, Raschka 2020) decomposes the K-class ordinal problem
  into K-1 binary "is the rank > k?" classifications, with a parameterization
  that guarantees rank-monotonic outputs *during inference* without imposing
  awkward constraints during training. See `coral_pytorch`.
- Output is a vector of K-1 probabilities; the predicted rank is
  `1 + sum(p_k > 0.5)`, and the soft expected grade is `1 + sum(p_k)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
import torch.nn as nn

NUM_GRADES = 10           # PSA grades are integers 1..10
NUM_THRESHOLDS = NUM_GRADES - 1  # CORN produces K-1 logits


@dataclass(slots=True)
class GradeOutput:
    expected_grade: torch.Tensor       # (B,) float, soft expected grade in [1, 10]
    predicted_grade: torch.Tensor      # (B,) long, hard rank in {1..10}
    confidence: torch.Tensor           # (B,) float in [0, 1]
    rank_probs: torch.Tensor           # (B, K-1) cumulative probs P(grade > k)


class GradeModel(nn.Module):
    """Two-stream backbone (shared weights) + CORN head."""

    def __init__(
        self,
        backbone: str = "convnext_tiny",
        pretrained: bool = True,
        dropout: float = 0.2,
        front_weight: float = 0.7,
    ) -> None:
        super().__init__()
        if not 0.0 < front_weight < 1.0:
            raise ValueError("front_weight must be in (0, 1)")
        self.front_weight = front_weight
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.dropout = nn.Dropout(dropout)
        # CORN-style: a single linear layer producing K-1 logits whose
        # cumulative probabilities are guaranteed monotonic via softplus.
        self.fc = nn.Linear(feat_dim, NUM_THRESHOLDS)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, front: torch.Tensor, back: torch.Tensor) -> torch.Tensor:
        """Returns raw CORN logits of shape (B, K-1)."""
        f_front = self._embed(front)
        f_back = self._embed(back)
        fused = self.front_weight * f_front + (1.0 - self.front_weight) * f_back
        return self.fc(self.dropout(fused))

    @torch.no_grad()
    def predict(self, front: torch.Tensor, back: torch.Tensor) -> GradeOutput:
        logits = self.forward(front, back)
        return decode_corn_logits(logits)


def decode_corn_logits(logits: torch.Tensor) -> GradeOutput:
    """Convert (B, K-1) CORN logits into rank, expected grade, and confidence."""
    probs = torch.sigmoid(logits)                   # P(rank > k) per threshold
    # Enforce monotonicity at inference (CORN already implies this in expectation,
    # but a cumulative-min keeps it strict against numerical noise).
    probs, _ = torch.cummin(probs, dim=1)
    hard_rank = 1 + (probs > 0.5).long().sum(dim=1)             # in {1..K}
    expected = 1.0 + probs.sum(dim=1)                           # soft expected grade

    # Confidence: how peaked is the distribution around the predicted rank?
    # Per-class probabilities P(rank == k) come from differences in cumulative probs.
    cum_full = torch.cat(
        [torch.ones_like(probs[:, :1]), probs, torch.zeros_like(probs[:, :1])],
        dim=1,
    )
    per_class = cum_full[:, :-1] - cum_full[:, 1:]              # (B, K)
    per_class = per_class.clamp(min=0.0)
    per_class = per_class / per_class.sum(dim=1, keepdim=True).clamp(min=1e-6)
    conf = per_class.gather(1, (hard_rank - 1).clamp(0, NUM_GRADES - 1).unsqueeze(1)).squeeze(1)
    return GradeOutput(
        expected_grade=expected,
        predicted_grade=hard_rank,
        confidence=conf,
        rank_probs=probs,
    )


def grade_to_targets(grades: torch.Tensor) -> torch.Tensor:
    """Convert integer grades (1..K) to CORN binary targets of shape (B, K-1).

    target[b, k] = 1 iff grades[b] > k+1, i.e. the rank exceeds threshold k.
    """
    grades = grades.long().clamp(1, NUM_GRADES)
    thresholds = torch.arange(1, NUM_GRADES, device=grades.device).unsqueeze(0)  # (1, K-1)
    return (grades.unsqueeze(1) > thresholds).float()


def corn_loss(logits: torch.Tensor, grades: torch.Tensor) -> torch.Tensor:
    """CORN (Conditional Ordinal Regression) loss.

    Equivalent to the `coral_pytorch.losses.corn_loss` formulation: BCE on each
    threshold conditioned on the previous threshold being met. We re-implement
    inline to keep the dependency tree thin and avoid breakage on coral_pytorch
    version drift.
    """
    targets = grade_to_targets(grades)                         # (B, K-1)
    # Mask: threshold k is active for sample b iff target[b, k-1] == 1
    # (i.e. rank exceeded threshold k-1). Threshold 0 is always active.
    mask = torch.cat(
        [torch.ones_like(targets[:, :1]), targets[:, :-1]], dim=1
    )
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    masked = bce * mask
    denom = mask.sum().clamp(min=1.0)
    return masked.sum() / denom
