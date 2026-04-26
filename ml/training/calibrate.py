"""Temperature scaling on the validation set.

Confidence from raw CORN sigmoids tends to be over-confident. We learn a single
scalar T that minimises the validation NLL on the per-class CORN distribution.
The same T is applied at inference, baked into the exported model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backend.app.ml.model import GradeModel, NUM_GRADES, decode_corn_logits


@dataclass(slots=True)
class CalibrationResult:
    temperature: float
    nll_before: float
    nll_after: float


def _per_class_log_probs(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """CORN -> per-class probabilities, temperature-scaled, log-space."""
    scaled = logits / temperature
    probs = torch.sigmoid(scaled)
    probs, _ = torch.cummin(probs, dim=1)
    cum_full = torch.cat(
        [torch.ones_like(probs[:, :1]), probs, torch.zeros_like(probs[:, :1])], dim=1
    )
    per_class = (cum_full[:, :-1] - cum_full[:, 1:]).clamp(min=1e-8)
    per_class = per_class / per_class.sum(dim=1, keepdim=True)
    return per_class.log()


def fit_temperature(
    model: GradeModel,
    val_loader: DataLoader,
    device: str = "cpu",
    max_iter: int = 100,
) -> CalibrationResult:
    model.eval().to(device)
    all_logits: list[torch.Tensor] = []
    all_grades: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in val_loader:
            front = batch["front"].to(device)
            back = batch["back"].to(device)
            logits = model(front, back)
            all_logits.append(logits.cpu())
            all_grades.append(batch["grade"].cpu())
    logits = torch.cat(all_logits)
    grades = torch.cat(all_grades).long().clamp(1, NUM_GRADES) - 1  # 0..K-1

    def _nll(t: torch.Tensor) -> torch.Tensor:
        log_probs = _per_class_log_probs(logits, t)
        return F.nll_loss(log_probs, grades)

    one = torch.tensor(1.0)
    nll_before = _nll(one).item()

    temperature = torch.nn.Parameter(torch.ones(1) * 1.0)
    opt = torch.optim.LBFGS([temperature], lr=0.1, max_iter=max_iter)

    def _closure() -> torch.Tensor:
        opt.zero_grad()
        loss = _nll(temperature.clamp(min=0.05))
        loss.backward()
        return loss

    opt.step(_closure)
    t_final = float(temperature.detach().clamp(min=0.05).item())
    nll_after = _nll(torch.tensor(t_final)).item()
    return CalibrationResult(temperature=t_final, nll_before=nll_before, nll_after=nll_after)
