"""Grading metrics: MAE, exact-match, off-by-one accuracy, per-grade confusion."""

from __future__ import annotations

import torch


def mean_absolute_error(pred_grade: torch.Tensor, true_grade: torch.Tensor) -> torch.Tensor:
    return (pred_grade.float() - true_grade.float()).abs().mean()


def exact_match(pred_grade: torch.Tensor, true_grade: torch.Tensor) -> torch.Tensor:
    return (pred_grade.long() == true_grade.long()).float().mean()


def off_by_one(pred_grade: torch.Tensor, true_grade: torch.Tensor) -> torch.Tensor:
    diff = (pred_grade.long() - true_grade.long()).abs()
    return (diff <= 1).float().mean()


def confusion_matrix(
    pred_grade: torch.Tensor, true_grade: torch.Tensor, num_classes: int = 10
) -> torch.Tensor:
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    p = pred_grade.long().clamp(1, num_classes) - 1
    t = true_grade.long().clamp(1, num_classes) - 1
    for ti, pi in zip(t.tolist(), p.tolist(), strict=True):
        cm[ti, pi] += 1
    return cm
