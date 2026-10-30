"""Train the face side / orientation guard (`backend/app/ml/face_guard.py`).

    python -m ml.training.train_face_guard                     # served-path features
    python -m ml.training.train_face_guard --features precrop  # the 2026-10-28 source

Steps:
1. The rows `train_probe` uses (`load_rows`) and one face's features per row,
   upright and rotated 180 degrees, from one of two sources (`--features`):
   - `served` (default): the original DB photos (`front_path` / `back_path`, the
     ones `ml.eval.evaluate` feeds the service) through the probe path's own
     preprocessing, `inference.served_extractor_input`: decode, EXIF transpose,
     1024 px cap, detection, `_jpeg_roundtrip`. The rotated face is the photo
     rotated before all that, with `evaluate.perturb_image("rotate_180", ...)`
     (cv2 decode, rotate, JPEG q95), exactly as the harness's `rotate_180`.
     Cached in data/models/cache/region_feats_served.npz and
     region_feats_served_rot180.npz. The two precrop caches are only read, for
     `transfer_from_precrop` (step 2), and never written.
   - `precrop`: the precrops as stored (the `train_probe` cache
     data/models/cache/region_feats.npz, read only) and the precrops rotated with
     `np.rot90(rgb, 2)` (data/models/cache/region_feats_rot180.npz).
   Every cache is keyed and invalidated like the main one
   (`train_probe.extract_features`: source file size + mtime, feature spec,
   variant). A served cache isn't invalidated by a change to the preprocessing
   code: delete it after changing `preprocess.py` or `inference.extractor_input`.
2. With `served`, `transfer_from_precrop`: probes fit on precrop features of the
   training folds, scored on the held-out certs' served features (same folds),
   at the thresholds precrop CV chooses, i.e. how the precrop-trained guard does
   on served faces out of fold.
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
   (+ .json, recording `feature_source`) and
   test-artifacts/eval/face_guard_report.json. `--no-save` writes only the
   report (and the feature caches).

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
from backend.app.ml.features import FEATURE_SPEC, build_region_model, feature_spec_hash
from backend.app.ml.inference import served_extractor_input
from ml.eval.evaluate import perturb_image
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

FEATURE_SOURCES = ("served", "precrop")
# Per source: the (upright, rotated) cache files in data/models/cache, with their variants.
CACHES = {
    "precrop": (("region_feats.npz", ""), ("region_feats_rot180.npz", "rot180")),
    "served": (("region_feats_served.npz", "served"), ("region_feats_served_rot180.npz", "served_rot180")),
}
FEATURE_SOURCE_NOTES = {
    "precrop": "stored precrops (upright) and np.rot90(precrop, 2) (rotated)",
    "served": (
        "original DB photos through inference.served_extractor_input (decode, EXIF transpose, "
        "1024 px cap, detection, _jpeg_roundtrip); rotated: the photo rotated first with "
        "ml.eval.evaluate.perturb_image('rotate_180') (cv2 decode, rotate 180, JPEG q95)"
    ),
}
SERVED_PATHS = ("front_path", "back_path")


def rot180(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(rgb, 2))


def served_upright(path: Path) -> np.ndarray:
    """The extractor input `/api/grade` makes of the photo at `path`."""
    return served_extractor_input(path.read_bytes())


def served_rot180(path: Path) -> np.ndarray:
    """The same for the photo rotated 180 degrees as `ml.eval.evaluate`'s `rotate_180` does."""
    return served_extractor_input(perturb_image("rotate_180", path.read_bytes()))


def load_face_features(
    source: str,
    rows,
    data_dir: Path,
    cache_dir: Path,
    *,
    model_factory=None,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """(upright, rotated) (N, 2, n_regions, D) features of `rows` from `source`.

    `served` writes only its own two caches. `precrop` reads the `train_probe`
    cache without writing it and keeps region_feats_rot180.npz up to date, as
    the 2026-10-28 training did. `model_factory` defaults to `build_region_model`.
    """
    if source not in FEATURE_SOURCES:
        raise ValueError(f"unknown feature source {source!r}; known: {', '.join(FEATURE_SOURCES)}")
    if source == "precrop":
        return precrop_features(rows, data_dir, cache_dir, save_rot=True, model_factory=model_factory)
    (up_name, up_variant), (rot_name, rot_variant) = CACHES["served"]
    model_factory = model_factory or build_region_model
    missing = [r["cert_number"] for r in rows if not all(r[k] for k in SERVED_PATHS)]
    if missing:
        raise ValueError(f"{len(missing)} certs have no original photo path, e.g. {missing[:3]}")
    kw = {"model_factory": model_factory, "path_keys": SERVED_PATHS, "workers": workers}
    up = extract_features(rows, data_dir, cache_dir / up_name, load=served_upright, variant=up_variant, **kw)
    rot = extract_features(rows, data_dir, cache_dir / rot_name, load=served_rot180, variant=rot_variant, **kw)
    return up, rot


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


def oof_probs(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
    seed: int = SEED,
    x_eval: np.ndarray | None = None,
):
    """Out-of-fold P(positive) from card-grouped stratified K-fold.

    Each fold's probe is fit on `x` of the other folds and scores the held-out
    rows of `x_eval` (default `x`). The folds depend only on y, groups and seed,
    so two feature sources of the same faces share them.
    """
    x_eval = x if x_eval is None else x_eval
    p = np.full(len(y), np.nan)
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(np.zeros(len(y)), y, groups):
        p[te] = fit_logistic(x[tr], y[tr]).prob(x_eval[te])
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


def transfer_metrics(
    src_up: np.ndarray,
    src_rot: np.ndarray,
    dst_up: np.ndarray,
    dst_rot: np.ndarray,
    groups: np.ndarray,
    cv_folds: int = 5,
) -> dict:
    """Probes trained on the source features, scored out of fold on the destination's.

    Per check: the source's own OOF metrics and the threshold training on it
    chooses, and the destination's OOF metrics with its clean-face FPR at that
    threshold. Both sources must hold the same faces in the same order; pass
    train+val rows only.
    """
    if not (src_up.shape == src_rot.shape == dst_up.shape == dst_rot.shape and len(groups) == len(src_up)):
        raise ValueError("transfer_metrics: the two sources are not the same faces")
    src, dst = face_sets(src_up, src_rot, groups), face_sets(dst_up, dst_rot, groups)
    out = {}
    for name in CHECKS:
        s, d = src[name], dst[name]
        ps = oof_probs(s["X"], s["y"], s["groups"], cv_folds)
        pd = oof_probs(s["X"], s["y"], s["groups"], cv_folds, x_eval=d["X"])
        ms, md = probe_metrics(name, ps, s["y"], s["clean"]), probe_metrics(name, pd, d["y"], d["clean"])
        t = choose_threshold(ms)
        at = md["at_threshold"][str(t if t is not None else THRESHOLD_GRID[-1])]
        entry = {
            "source_threshold": t,
            "source_cv": ms,
            "dest_cv": md,
            "dest_clean_fpr_at_source_threshold": at["clean_fpr"],
            "dest_n_clean_fp_at_source_threshold": at["n_clean_fp"],
        }
        if name == "side":
            n = len(src_up)
            entry["dest_pairs"] = pair_side_metrics(pd[:n], pd[2 * n : 3 * n], t or THRESHOLD_GRID[-1])
        out[name] = entry
        log.info(
            "%s transfer: dest clean FPR %.3f%% (%d of %d) at source threshold %s",
            name,
            100 * at["clean_fpr"],
            at["n_clean_fp"],
            md["n_clean"],
            t,
        )
    return out


def train_guard(
    up: np.ndarray,
    rot: np.ndarray,
    groups: np.ndarray,
    fit_mask: np.ndarray,
    test_mask: np.ndarray,
    cv_folds: int = 5,
    feature_source: str = "precrop",
) -> tuple[FaceGuard, dict]:
    """Fit the three probes; thresholds and gates from CV on `fit_mask` only.

    Returns the guard (metadata filled in, incl. `feature_source`: which
    features `up` / `rot` are) and the report. Test rows only get scored by the
    final probes.
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
        "feature_source": feature_source,
        "feature_source_note": FEATURE_SOURCE_NOTES.get(feature_source, ""),
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
    features: str = typer.Option(
        "served", help="Feature source: served (the /api/grade preprocessing of the DB photos) or precrop."
    ),
    save: bool = typer.Option(True, help="Write the guard; --no-save writes only the report and caches."),
    report: Path = typer.Option(REPORT_PATH, help="Where to write the JSON report."),
    workers: int = typer.Option(8, help="Threads preprocessing photos for served-path extraction."),
) -> None:
    if features not in FEATURE_SOURCES:
        raise typer.BadParameter(f"{features!r}; known: {', '.join(FEATURE_SOURCES)}", param_hint="--features")
    settings = get_settings()
    rows = load_rows(settings.db_path)
    if not rows:
        raise SystemExit("no certs with usable precrops - run `python -m ml.data.precrop`")
    cache_dir = settings.models_dir / "cache"
    up, rot = load_face_features(features, rows, settings.data_dir, cache_dir, workers=workers)
    split = np.array([r["split"] for r in rows])
    groups = card_groups(
        [r["brand_title"] or "" for r in rows],
        [r["subject"] or "" for r in rows],
        [r["card_number"] or "" for r in rows],
    )
    fit_mask = np.isin(split, ("train", "val"))
    test_mask = split == "test"
    log.info("certs: fit (train+val)=%d test=%d", fit_mask.sum(), test_mask.sum())

    transfer = None
    if features == "served":
        # How the precrop-trained guard does on served faces, out of fold. The
        # precrop caches are read only (save=False).
        pre_up, pre_rot = precrop_features(rows, settings.data_dir, cache_dir, save_rot=False)
        transfer = transfer_metrics(
            pre_up[fit_mask], pre_rot[fit_mask], up[fit_mask], rot[fit_mask], groups[fit_mask], cv_folds
        )

    guard, result = train_guard(up, rot, groups, fit_mask, test_mask, cv_folds, feature_source=features)
    guard.metadata["trained_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    guard.metadata["git_commit"] = _git_commit()
    guard.metadata["feature_caches"] = [name for name, _ in CACHES[features]]
    if save:
        path = out or settings.models_dir / "face_guard.npz"
        guard.save(path)
        log.info("wrote %s (enabled: %s)", path, ", ".join(guard.enabled) or "none")
    else:
        log.info("--no-save: no guard written (it would enable: %s)", ", ".join(guard.enabled) or "none")

    full = {**result, **guard.metadata, "enabled_checks": list(guard.enabled), "saved": save}
    if transfer is not None:
        full["transfer_from_precrop"] = _rounded(transfer)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(full, indent=2))
    log.info("report -> %s", report)


def precrop_features(rows, data_dir: Path, cache_dir: Path, *, save_rot: bool, model_factory=None):
    """(upright, rotated) precrop features. region_feats.npz is only read; the
    rotated cache is written only with `save_rot`."""
    (up_name, _), (rot_name, rot_variant) = CACHES["precrop"]
    model_factory = model_factory or build_region_model
    up = extract_features(rows, data_dir, cache_dir / up_name, model_factory=model_factory, save=False)
    rot = extract_features(
        rows,
        data_dir,
        cache_dir / rot_name,
        model_factory=model_factory,
        transform=rot180,
        variant=rot_variant,
        save=save_rot,
    )
    return up, rot


if __name__ == "__main__":
    app()
