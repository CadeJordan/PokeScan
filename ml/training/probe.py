"""Fitting for the linear ordinal grade head (`backend/app/ml/grade_head.py`).

Everything here works on cached region features (N, 2, n_regions, D) and
integer grades; `train_probe.py` is the CLI around it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from backend.app.ml.grade_head import (
    NUM_GRADES,
    PROBE_NAMES,
    GradeHead,
    OrdinalProbe,
    probe_input,
)
from ml.data.groups import card_group_key

# Grouped-CV defaults, chosen on card-grouped 5-fold CV over train+val
# (PCA 64/128/256 x L2 1e-3..3e-2 all land within ~0.01 MAE of each other).
PCA_DIM = 128
L2 = 3e-3
META_L2 = 1e-3
INNER_FOLDS = 4


def _grades(y: np.ndarray) -> np.ndarray:
    """Integer grades 1..NUM_GRADES; anything else (0, 8.5, NaN) is a data bug."""
    y = np.asarray(y)
    ok = y.dtype.kind in "iu" or (y.dtype.kind == "f" and np.isfinite(y).all())
    yi = y.astype(int) if ok else None
    if yi is None or (y != yi).any() or yi.min() < 1 or yi.max() > NUM_GRADES:
        raise ValueError(f"grades must be integers 1..{NUM_GRADES}, got {np.unique(y)[:20]}")
    return yi


def _fit_po(z: np.ndarray, y: np.ndarray, l2: float, max_iter: int = 300):
    """Proportional-odds fit on (already whitened) z -> (w, thresholds)."""
    zt = torch.tensor(z, dtype=torch.float64)
    yt = torch.tensor(_grades(y))
    w = torch.zeros(zt.shape[1], dtype=torch.float64, requires_grad=True)
    b0 = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
    gaps = torch.zeros(NUM_GRADES - 2, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b0, gaps], max_iter=max_iter, line_search_fn="strong_wolfe")
    rows = torch.arange(len(yt))

    def thresholds() -> torch.Tensor:
        return torch.cat([b0[None], b0 + torch.cumsum(F.softplus(gaps) + 1e-3, 0)])

    def closure() -> torch.Tensor:
        opt.zero_grad()
        p = torch.sigmoid((zt @ w)[:, None] - thresholds()[None])
        ones = torch.ones(len(p), 1, dtype=p.dtype)
        cum = torch.cat([ones, p, torch.zeros_like(ones)], 1)
        per_grade = (cum[:, :-1] - cum[:, 1:]).clamp_min(1e-9)
        loss = -torch.log(per_grade[rows, yt - 1]).mean() + l2 * (w**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), thresholds().detach().numpy()


def fit_ordinal_probe(
    x: np.ndarray, y: np.ndarray, pca_dim: int | None = PCA_DIM, l2: float = L2
) -> OrdinalProbe:
    """Standardise (+ whitened PCA) + proportional odds, folded into one weight vector."""
    x = np.asarray(x, np.float64)
    y = _grades(y)
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    if pca_dim and pca_dim < min(z.shape):
        pca = PCA(pca_dim, random_state=0).fit(z)
        # Whiten only components with real variance: past the data's rank they
        # are noise, and dividing by ~0 would blow the folded weights up.
        ev = pca.explained_variance_
        keep = ev > ev[0] * 1e-8
        comps = pca.components_[keep] / np.sqrt(ev[keep])[:, None]
        w, th = _fit_po((z - pca.mean_) @ comps.T, y, l2)
        v = comps.T @ w
        offset = -pca.mean_ @ v
    else:
        w, th = _fit_po(z, y, l2)
        v, offset = w, 0.0
    weight = v / scaler.scale_
    bias = float(offset - (scaler.mean_ / scaler.scale_) @ v)
    return OrdinalProbe(weight=weight, bias=bias, thresholds=th)


def card_groups(brand: list[str], subject: list[str], number: list[str]) -> np.ndarray:
    """Group key per cert: the same printed card shares a key across certs."""
    return np.array(
        [card_group_key(b, s, n) for b, s, n in zip(brand, subject, number, strict=True)]
    )


def _folds(y: np.ndarray, groups: np.ndarray, n: int, seed: int):
    # Stratify on grade with <=8 pooled: the low grades are too rare to spread.
    skf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), np.minimum(y, 8), groups))


def fit_region_probes(feats: np.ndarray, y: np.ndarray) -> dict[str, OrdinalProbe]:
    return {n: fit_ordinal_probe(probe_input(feats, n), y) for n in PROBE_NAMES}


def region_grades(probes: dict[str, OrdinalProbe], feats: np.ndarray) -> np.ndarray:
    return np.stack([probes[n].soft_grade(probe_input(feats, n)) for n in PROBE_NAMES], -1)


def oof_region_grades(
    feats: np.ndarray, y: np.ndarray, groups: np.ndarray, n_folds: int = INNER_FOLDS, seed: int = 1
) -> np.ndarray:
    """Out-of-fold region soft grades, so the meta probe never sees fitted-on scores."""
    out = np.zeros((len(y), len(PROBE_NAMES)))
    for tr, te in _folds(y, groups, n_folds, seed):
        out[te] = region_grades(fit_region_probes(feats[tr], y[tr]), feats[te])
    return out


def fit_grade_head(
    feats: np.ndarray, y: np.ndarray, groups: np.ndarray, metadata: dict | None = None
) -> GradeHead:
    meta = fit_ordinal_probe(oof_region_grades(feats, y, groups), y, pca_dim=None, l2=META_L2)
    return GradeHead(probes=fit_region_probes(feats, y), meta=meta, metadata=metadata or {})


def cross_validate(
    feats: np.ndarray, y: np.ndarray, groups: np.ndarray, n_folds: int = 5, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Card-grouped CV of the whole head -> (OOF cum probs (N, 9), OOF region grades (N, 4))."""
    cum = np.zeros((len(y), NUM_GRADES - 1))
    regions = np.zeros((len(y), len(PROBE_NAMES)))
    for tr, te in _folds(y, groups, n_folds, seed):
        head = fit_grade_head(feats[tr], y[tr], groups[tr])
        regions[te] = head.region_grades(feats[te])
        cum[te] = head.meta.cum_probs(regions[te])
    return cum, regions
