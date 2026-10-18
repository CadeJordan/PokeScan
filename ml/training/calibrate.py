"""Post-hoc calibration on the validation set: temperature + threshold offsets.

1. Temperature scaling. Confidence from raw CORN sigmoids tends to be
   over-confident. We learn a single scalar T that minimises the validation
   NLL on the per-class CORN distribution.
2. Per-threshold bias offsets. A model trained with a skewed sampler can be
   systematically biased (e.g. grading true 10s ~1 grade low). We fit 9 additive
   offsets ``b_k`` on the temperature-scaled logits (``logits / T + b``) that
   minimise val MAE - QWK (or either alone) by coordinate search over a grid.
   Monotone thresholds are preserved at decode time by the cumulative-min in
   ``decode_corn_logits`` / the ONNX consumer, so no constraint is needed here.
   Afterwards T is re-fitted on the offset logits; a positive rescale never
   changes the hard grade, so this restores calibration for free.

Both are baked into the exported graph (``export.CalibratedExportModule``) as
``logits / T + b``, so the API keeps reading ``corn_logits`` unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from backend.app.ml.model import NUM_GRADES, NUM_THRESHOLDS, GradeModel

OBJECTIVES = ("mae_qwk", "mae", "qwk")


@dataclass(slots=True)
class CalibrationResult:
    temperature: float
    nll_before: float
    nll_after: float


@dataclass(slots=True)
class OffsetResult:
    offsets: list[float]          # additive, applied after dividing by temperature
    temperature: float            # final temperature to pair with ``offsets``
    objective: str
    before: dict = field(default_factory=dict)   # val metrics with T only
    after: dict = field(default_factory=dict)    # val metrics with T + offsets

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_logits(
    model: GradeModel, loader: DataLoader, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw CORN logits (N, K-1) and integer grades (N,) over a loader."""
    model.eval().to(device)
    all_logits: list[torch.Tensor] = []
    all_grades: list[torch.Tensor] = []
    for batch in loader:
        logits = model(batch["front"].to(device), batch["back"].to(device))
        all_logits.append(logits.float().cpu())
        all_grades.append(batch["grade"].cpu())
    return torch.cat(all_logits), torch.cat(all_grades).long()


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


def corn_nll(logits: torch.Tensor | np.ndarray, grades: torch.Tensor | np.ndarray) -> float:
    """Per-class CORN negative log-likelihood of (already scaled) logits."""
    lg = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    target = torch.as_tensor(np.asarray(grades)).long().clamp(1, NUM_GRADES) - 1
    return float(F.nll_loss(_per_class_log_probs(lg, torch.tensor(1.0)), target).item())


def fit_temperature_on_logits(
    logits: torch.Tensor,
    grades: torch.Tensor,
    max_iter: int = 100,
    bounds: tuple[float, float] = (0.05, 20.0),
) -> CalibrationResult:
    target = grades.long().clamp(1, NUM_GRADES) - 1  # 0..K-1

    def _nll(t: torch.Tensor) -> torch.Tensor:
        return F.nll_loss(_per_class_log_probs(logits, t), target)

    nll_before = _nll(torch.tensor(1.0)).item()
    temperature = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([temperature], lr=0.1, max_iter=max_iter)

    def _closure() -> torch.Tensor:
        opt.zero_grad()
        loss = _nll(temperature.clamp(*bounds))
        loss.backward()
        return loss

    opt.step(_closure)
    t_final = float(temperature.detach().clamp(*bounds).item())
    nll_after = _nll(torch.tensor(t_final)).item()
    return CalibrationResult(temperature=t_final, nll_before=nll_before, nll_after=nll_after)


def fit_temperature(
    model: GradeModel,
    val_loader: DataLoader,
    device: str = "cpu",
    max_iter: int = 100,
) -> CalibrationResult:
    logits, grades = collect_logits(model, val_loader, device)
    return fit_temperature_on_logits(logits, grades, max_iter=max_iter)


# --------------------------------------------------------------------------- #
# Threshold offsets
# --------------------------------------------------------------------------- #
def _hard_grades(scaled_logits: np.ndarray) -> np.ndarray:
    """Hard CORN rank: 1 + number of *leading* thresholds with p > 0.5.

    Equivalent to ``1 + sum(cummin(sigmoid(z)) > 0.5)`` in decode_corn_logits.
    """
    positive = scaled_logits > 0
    return 1 + np.cumprod(positive, axis=1).sum(axis=1)


def _soft_grades(scaled_logits: np.ndarray) -> np.ndarray:
    p = 1.0 / (1.0 + np.exp(-scaled_logits))
    return 1.0 + np.minimum.accumulate(p, axis=1).sum(axis=1)


def offset_metrics(scaled_logits: np.ndarray, grades: np.ndarray) -> dict:
    from ml.eval.evaluate import quadratic_weighted_kappa

    pred = _hard_grades(scaled_logits)
    soft = _soft_grades(scaled_logits)
    grades = np.asarray(grades, int)
    return {
        "mae": float(np.abs(pred - grades).mean()),
        "exact": float((pred == grades).mean()),
        "qwk": float(quadratic_weighted_kappa(grades, pred)),
        "bias": float((soft - grades).mean()),
    }


def _objective_fn(objective: str):
    from ml.eval.evaluate import quadratic_weighted_kappa

    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")

    def score(pred: np.ndarray, grades: np.ndarray) -> float:  # lower is better
        mae = float(np.abs(pred - grades).mean())
        if objective == "mae":
            return mae
        qwk = quadratic_weighted_kappa(grades, pred)
        qwk = 0.0 if np.isnan(qwk) else float(qwk)
        return -qwk if objective == "qwk" else mae - qwk

    return score


def fit_threshold_offsets(
    logits: torch.Tensor | np.ndarray,
    grades: torch.Tensor | np.ndarray,
    temperature: float = 1.0,
    objective: str = "mae_qwk",
    search_range: float = 4.0,
    steps: int = 81,
    sweeps: int = 4,
    l2: float = 1e-3,
    refit_temperature: bool = True,
    refit_bounds: tuple[float, float] = (0.5, 2.0),
) -> OffsetResult:
    """Fit 9 additive offsets on ``logits / temperature`` by coordinate search.

    The hard grade is piecewise-constant in the offsets, so we search each
    threshold on a grid (``[-search_range, search_range]``, ``steps`` points)
    while holding the others fixed, for ``sweeps`` passes. A small L2 penalty
    keeps unsupported thresholds (e.g. grade 1-3 with no val cards) at 0 and
    breaks ties toward the smallest shift. The temperature re-fit is bounded
    (``refit_bounds``, relative to ``temperature``) because the offsets were
    fitted on the same val set - an unbounded re-fit can chase it to T -> 0.
    """
    z = (np.asarray(logits, np.float64) / float(temperature))
    y = np.asarray(grades, int)
    if z.ndim != 2 or z.shape[1] != NUM_THRESHOLDS:
        raise ValueError(f"expected (N, {NUM_THRESHOLDS}) logits, got {z.shape}")
    score = _objective_fn(objective)
    grid = np.linspace(-search_range, search_range, steps)
    b = np.zeros(NUM_THRESHOLDS)

    def total(bv: np.ndarray) -> float:
        return score(_hard_grades(z + bv), y) + l2 * float((bv**2).sum())

    best = total(b)
    for _ in range(sweeps):
        improved = False
        for k in range(NUM_THRESHOLDS):
            for v in grid:
                cand = b.copy()
                cand[k] = v
                s = total(cand)
                if s < best - 1e-12:
                    best, b, improved = s, cand, True
        if not improved:
            break

    before = offset_metrics(z, y)
    after = offset_metrics(z + b, y)

    # Re-fit temperature on the offset logits. Scaling by 1/T2 > 0 keeps every
    # sign (hence every hard grade) unchanged, so only confidence moves.
    t_final, offsets = float(temperature), b
    if refit_temperature:
        cal = fit_temperature_on_logits(
            torch.as_tensor(z + b, dtype=torch.float32), torch.as_tensor(y), bounds=refit_bounds
        )
        t2 = cal.temperature
        t_final = float(temperature) * t2
        offsets = b / t2
        after = offset_metrics(z / t2 + offsets, y)
        after["nll"] = cal.nll_after
        before["nll"] = corn_nll(z, y)
    return OffsetResult(
        offsets=[float(v) for v in offsets],
        temperature=t_final,
        objective=objective,
        before=before,
        after=after,
    )


def apply_calibration(
    logits: torch.Tensor | np.ndarray, temperature: float, offsets: list[float] | None
) -> np.ndarray:
    """``logits / T + b`` - exactly what the exported graph outputs."""
    z = np.asarray(logits, np.float64) / float(temperature)
    if offsets is not None:
        z = z + np.asarray(offsets, np.float64)
    return z
