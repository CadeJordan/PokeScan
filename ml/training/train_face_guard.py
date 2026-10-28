"""Train the face side / orientation guard (`backend/app/ml/face_guard.py`).

    python -m ml.training.train_face_guard

Steps:
1. The rows `train_probe` uses (`load_rows`) and their upright region features,
   read from the `train_probe` cache (data/models/cache/region_feats.npz, never
   rewritten here).
2. Features of every crop rotated 180 degrees (`np.rot90(rgb, 2)` of the stored
   precrop), cached in data/models/cache/region_feats_rot180.npz with the same
   keying and invalidation as the main cache (`train_probe.extract_features`).
3. Three logistic probes over one face's CLS + mean features: `side` (P(back),
   both orientations), `front_upright` and `back_upright` (P(flipped)). Each is
   sklearn `LogisticRegression` on standardised features, the scaler folded into
   (weight, bias).
4. Card-grouped 5-fold CV (`StratifiedGroupKFold` on `card_groups`) over
   train+val: AUC, error at 0.5, and the false-positive rate on clean (correctly
   placed, upright) faces. Each threshold is the lowest of THRESHOLD_GRID whose
   out-of-fold clean-face FPR is <= FPR_LIMIT. A check that misses its gate is
   disabled, with the reason in the sidecar.
5. Fit on train+val, report once on test, and write data/models/face_guard.npz
   (+ .json) and test-artifacts/eval/face_guard_report.json.

The test split is never used to choose anything. Restart the API afterwards.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import typer
from rich.logging import RichHandler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from backend.app.core.config import PROJECT_ROOT, get_settings
from backend.app.ml.face_guard import CHECKS, FaceGuard, LogisticProbe, guard_input
from backend.app.ml.features import FEATURE_SPEC, feature_spec_hash
from ml.training.probe import card_groups
from ml.training.train_probe import _git_commit, extract_features, load_rows

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True)])
log = logging.getLogger("train_face_guard")

app = typer.Typer(add_completion=False)

REPORT_PATH = PROJECT_ROOT / "test-artifacts" / "eval" / "face_guard_report.json"
THRESHOLD_GRID = (0.9, 0.95, 0.99)
FPR_LIMIT = 0.002
GATES = {
    "side": {"max_error": 0.003, "min_auc": 0.999},
    "front_upright": {"min_auc": 0.99},
    "back_upright": {"min_auc": 0.99},
}
C = 1.0
SEED = 0


def rot180(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(rgb, 2))


# --------------------------------------------------------------------------- #
# Datasets: one row per face
# --------------------------------------------------------------------------- #
def face_sets(up: np.ndarray, rot: np.ndarray, cert_groups: np.ndarray) -> dict[str, dict]:
    """Per check: X (n, 2D), y, groups, and `clean` (upright faces in their right slot).

    `up`/`rot` are (N, 2, n_regions, D) features of [front, back], upright and
    rotated. side: y = is back, over all four face kinds; *_upright: y = flipped.
    `clean` marks the faces a false positive is counted on.
    """
    xu, xr = guard_input(up), guard_input(rot)  # (N, 2, 2D)
    n = len(up)
    zeros, ones = np.zeros(n, int), np.ones(n, int)
    side_x = np.concatenate([xu[:, 0], xr[:, 0], xu[:, 1], xr[:, 1]])
    side = {
        "X": side_x,
        "y": np.concatenate([zeros, zeros, ones, ones]),
        "groups": np.tile(cert_groups, 4),
        "clean": np.concatenate([ones, zeros, ones, zeros]).astype(bool),
        "upright": np.concatenate([ones, zeros, ones, zeros]).astype(bool),
    }
    out = {"side": side}
    for face, name in ((0, "front_upright"), (1, "back_upright")):
        out[name] = {
            "X": np.concatenate([xu[:, face], xr[:, face]]),
            "y": np.concatenate([zeros, ones]),
            "groups": np.tile(cert_groups, 2),
            "clean": np.concatenate([ones, zeros]).astype(bool),
        }
    return out


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def fit_logistic(x: np.ndarray, y: np.ndarray, c: float = C) -> LogisticProbe:
    """Logistic regression on standardised x, the scaler folded into (weight, bias)."""
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(C=c, max_iter=5000).fit(scaler.transform(x), y)
    coef = clf.coef_[0] / scaler.scale_
    bias = float(clf.intercept_[0] - coef @ scaler.mean_)
    return LogisticProbe(weight=coef.astype(np.float64), bias=bias)


def oof_probs(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_folds: int, seed: int = SEED):
    """Out-of-fold P(positive) from card-grouped stratified K-fold."""
    p = np.full(len(y), np.nan)
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(np.zeros(len(y)), y, groups):
        p[te] = fit_logistic(x[tr], y[tr]).prob(x[te])
    return p


# --------------------------------------------------------------------------- #
# Metrics and thresholds
# --------------------------------------------------------------------------- #
def false_positive(name: str, p: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    """Per face: would the check act on it at threshold t, wrongly if the face is clean?

    side acts on a front read as a back (P(back) > t) or a back read as a front
    (P(back) < 1 - t); the upright checks act on P(flipped) > t.
    """
    if name == "side":
        return np.where(y == 1, p < 1.0 - t, p > t)
    return p > t


def detected(name: str, p: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    """Per face: confidently right. side: the true side beyond t; upright: flipped beyond t."""
    if name == "side":
        return np.where(y == 1, p > t, p < 1.0 - t)
    return p > t


def probe_metrics(name: str, p: np.ndarray, y: np.ndarray, clean: np.ndarray) -> dict:
    """AUC, error at 0.5, and per threshold the clean-face FPR and the detection rate."""
    err = (p > 0.5) != (y == 1)
    out = {
        "n": int(len(y)),
        "n_clean": int(clean.sum()),
        "auc": float(roc_auc_score(y, p)),
        "error_at_0.5": float(err.mean()),
        "n_errors_at_0.5": int(err.sum()),
        "at_threshold": {},
    }
    target = ~clean if name != "side" else np.ones(len(y), bool)
    for t in THRESHOLD_GRID:
        fp = false_positive(name, p, y, t)[clean]
        out["at_threshold"][str(t)] = {
            "clean_fpr": float(fp.mean()),
            "n_clean_fp": int(fp.sum()),
            # upright: share of flipped faces caught; side: share of faces whose side is sure
            "detection_rate": float(detected(name, p, y, t)[target].mean()),
        }
    return out


def pair_side_metrics(p_front: np.ndarray, p_back: np.ndarray, t: float) -> dict:
    """Side check on whole pairs (upright faces): false swaps, swaps caught, same-side notes."""
    def swap(pf, pb):  # the front slot reads as a back and the back slot as a front
        return (pf > t) & (pb < 1.0 - t)

    clean_both = ((p_front > t) & (p_back > t)) | ((p_front < 1 - t) & (p_back < 1 - t))
    return {
        "n_pairs": int(len(p_front)),
        "false_swap_rate": float(swap(p_front, p_back).mean()),
        "swapped_pairs_caught": float(swap(p_back, p_front).mean()),
        "clean_same_side_note_rate": float(clean_both.mean()),
    }


def choose_threshold(metrics: dict) -> float | None:
    """Lowest grid threshold whose out-of-fold clean-face FPR is <= FPR_LIMIT."""
    for t in THRESHOLD_GRID:
        if metrics["at_threshold"][str(t)]["clean_fpr"] <= FPR_LIMIT:
            return t
    return None


def gate(name: str, metrics: dict, threshold: float | None) -> str | None:
    """Why check `name` misses its gate (None: it passes)."""
    g = GATES[name]
    problems = []
    if metrics["auc"] < g["min_auc"]:
        problems.append(f"CV AUC {metrics['auc']:.5f} < {g['min_auc']}")
    if "max_error" in g and metrics["error_at_0.5"] > g["max_error"]:
        problems.append(f"CV error {metrics['error_at_0.5']:.4%} > {g['max_error']:.1%}")
    if threshold is None:
        fprs = ", ".join(
            f"{t}: {metrics['at_threshold'][str(t)]['clean_fpr']:.3%}" for t in THRESHOLD_GRID
        )
        problems.append(f"no threshold in {THRESHOLD_GRID} has clean-face FPR <= {FPR_LIMIT:.1%} ({fprs})")
    return "; ".join(problems) or None


def _rounded(obj):
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _rounded(v) for k, v in obj.items()}
    return obj


def train_guard(
    up: np.ndarray,
    rot: np.ndarray,
    groups: np.ndarray,
    fit_mask: np.ndarray,
    test_mask: np.ndarray,
    cv_folds: int = 5,
) -> tuple[FaceGuard, dict]:
    """Fit the three probes; thresholds and gates from CV on `fit_mask` only.

    Returns the guard (metadata filled in) and the report. Test rows only get
    scored by the final probes.
    """
    fit_sets = face_sets(up[fit_mask], rot[fit_mask], groups[fit_mask])
    test_sets = face_sets(up[test_mask], rot[test_mask], groups[test_mask]) if test_mask.any() else None
    probes, thresholds, enabled, reasons = {}, {}, [], {}
    cv, test = {}, {}
    for name in CHECKS:
        d = fit_sets[name]
        p = oof_probs(d["X"], d["y"], d["groups"], cv_folds)
        m = probe_metrics(name, p, d["y"], d["clean"])
        if name == "side":
            up_only = d["upright"]
            m["upright_only"] = {
                "auc": float(roc_auc_score(d["y"][up_only], p[up_only])),
                "error_at_0.5": float(((p[up_only] > 0.5) != (d["y"][up_only] == 1)).mean()),
            }
            n = int(fit_mask.sum())
            m["pairs"] = {str(t): pair_side_metrics(p[:n], p[2 * n : 3 * n], t) for t in THRESHOLD_GRID}
        t = choose_threshold(m)
        reason = gate(name, m, t)
        cv[name] = m
        probes[name] = fit_logistic(d["X"], d["y"])
        # A disabled check keeps the strictest grid value, so nothing reads as a choice.
        thresholds[name] = float(t if t is not None else THRESHOLD_GRID[-1])
        if reason:
            reasons[name] = reason
            log.warning("%s disabled: %s", name, reason)
        else:
            enabled.append(name)
        log.info(
            "%s: CV AUC %.5f, error %.4f%%, threshold %s, clean FPR %s",
            name,
            m["auc"],
            100 * m["error_at_0.5"],
            t,
            m["at_threshold"][str(thresholds[name])]["clean_fpr"],
        )
        if test_sets is not None:
            td = test_sets[name]
            tp = probes[name].prob(td["X"])
            tm = probe_metrics(name, tp, td["y"], td["clean"])
            if name == "side":
                n = int(test_mask.sum())
                tm["pairs"] = {
                    str(thresholds[name]): pair_side_metrics(tp[:n], tp[2 * n : 3 * n], thresholds[name])
                }
            test[name] = tm

    metadata = {
        "kind": "face_guard",
        "inputs": "cls + mean region features of one face (GLOBAL_SLICE)",
        "feature_spec_hash": feature_spec_hash(),
        "feature_spec": FEATURE_SPEC,
        "disabled_reasons": reasons,
        "threshold_grid": list(THRESHOLD_GRID),
        "fpr_limit": FPR_LIMIT,
        "gates": GATES,
        "C": C,
        "cv_folds": cv_folds,
        "cv_seed": SEED,
        "n_fit_certs": int(fit_mask.sum()),
        "n_test_certs": int(test_mask.sum()),
        "cv_grouped": _rounded(cv),
        "test": _rounded(test),
    }
    guard = FaceGuard(probes=probes, thresholds=thresholds, enabled=tuple(enabled), metadata=metadata)
    report = {
        **metadata,
        "enabled_checks": list(enabled),
        "thresholds": thresholds,
    }
    return guard, report


@app.command()
def main(
    cv_folds: int = typer.Option(5, help="Card-grouped CV folds over train+val."),
    out: Path | None = typer.Option(None, help="Guard path (default data/models/face_guard.npz)."),
) -> None:
    settings = get_settings()
    rows = load_rows(settings.db_path)
    if not rows:
        raise SystemExit("no certs with usable precrops - run `python -m ml.data.precrop`")
    cache_dir = settings.models_dir / "cache"
    # Upright features: read the train_probe cache, never write it.
    up = extract_features(rows, settings.data_dir, cache_dir / "region_feats.npz", save=False)
    rot = extract_features(
        rows,
        settings.data_dir,
        cache_dir / "region_feats_rot180.npz",
        transform=rot180,
        variant="rot180",
    )
    split = np.array([r["split"] for r in rows])
    groups = card_groups(
        [r["brand_title"] or "" for r in rows],
        [r["subject"] or "" for r in rows],
        [r["card_number"] or "" for r in rows],
    )
    fit_mask = np.isin(split, ("train", "val"))
    test_mask = split == "test"
    log.info("certs: fit (train+val)=%d test=%d", fit_mask.sum(), test_mask.sum())

    guard, report = train_guard(up, rot, groups, fit_mask, test_mask, cv_folds)
    guard.metadata["trained_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    guard.metadata["git_commit"] = _git_commit()
    path = out or settings.models_dir / "face_guard.npz"
    guard.save(path)
    log.info("wrote %s (enabled: %s)", path, ", ".join(guard.enabled) or "none")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps({**report, **guard.metadata, "enabled_checks": list(guard.enabled)}, indent=2))
    log.info("report -> %s", REPORT_PATH)


if __name__ == "__main__":
    app()
