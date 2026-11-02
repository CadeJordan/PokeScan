"""Evaluate the exported ONNX grade model end-to-end on a dataset split.

Runs the real serving path (`GradeInferenceService`: detect/crop -> ONNX ->
CORN decode) on raw front/back photos and writes a JSON report plus charts.

    python -m ml.eval.evaluate --split test
    python -m ml.eval.evaluate --split test --limit 50 --no-robustness
    python -m ml.eval.evaluate --split test --perturbations rotate_180,swap_faces
    python -m ml.eval.evaluate --split test --save-baseline   # pin current numbers
    python -m ml.eval.evaluate --split test --tta              # + test-time augmentation
    python -m ml.eval.evaluate --manifest data/raw_eval/manifest.csv   # raw-card set
    python -m ml.eval.evaluate --model-path data/models/candidate.onnx # side-by-side

Metrics
- accuracy: MAE (soft + hard), exact, off-by-one, quadratic weighted kappa
- uncertainty: 95% bootstrap CIs for MAE / QWK / exact / bias; per-grade
  Wilson CI for recall and bootstrap CI for MAE. A manifest with a `cert`
  column (several photo pairs per card) resamples cards, not rows
- per grade: support, recall, MAE, bias; confusion matrix; 9-vs-10 band
- calibration: ECE, reliability bins, accuracy vs. coverage
- pipeline: card-quad found rate, MAE split by quad found vs. fallback; for a
  manifest with extra columns (e.g. the phone set's `shot`, `in_db_split`),
  count / MAE / exact / would-reject per value of each (`by_tag`)
- robustness: MAE delta / grade-flip rate under blur, JPEG, rotation, lighting,
  and swapped front/back uploads (`--perturbations` picks a subset)
- face guard: per prediction, and per cert in each robustness run, the guard's
  P(back) / P(flipped) when a guard scored it (`p_back`, `p_flip`)
- speed: p50 / p95 latency per front+back pair
- baselines: majority-grade and train-mean predictors
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import typer

from backend.app.core.config import PROJECT_ROOT

log = logging.getLogger(__name__)

NUM_GRADES = 10
DEFAULT_OUT = PROJECT_ROOT / "test-artifacts" / "eval"
BASELINE_PATH = Path(__file__).with_name("baseline.json")


# --------------------------------------------------------------------------- #
# Metrics (pure numpy, unit-tested in ml/tests/test_eval_metrics.py)
# --------------------------------------------------------------------------- #
def confusion(true: np.ndarray, pred: np.ndarray, k: int = NUM_GRADES) -> np.ndarray:
    cm = np.zeros((k, k), np.int64)
    t = np.clip(np.asarray(true, int), 1, k) - 1
    p = np.clip(np.asarray(pred, int), 1, k) - 1
    np.add.at(cm, (t, p), 1)
    return cm


def quadratic_weighted_kappa(true: np.ndarray, pred: np.ndarray, k: int = NUM_GRADES) -> float:
    """Cohen's kappa with quadratic weights - the standard ordinal agreement score.

    1.0 = perfect, 0.0 = chance-level, negative = worse than chance.
    """
    observed = confusion(true, pred, k).astype(np.float64)
    n = observed.sum()
    if n == 0:
        return float("nan")
    idx = np.arange(k)
    weights = (idx[:, None] - idx[None, :]) ** 2 / (k - 1) ** 2
    expected = np.outer(observed.sum(1), observed.sum(0)) / n
    denom = (weights * expected).sum()
    if denom == 0:
        return 1.0 if (weights * observed).sum() == 0 else 0.0
    return float(1.0 - (weights * observed).sum() / denom)


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> tuple[float, list[dict]]:
    """ECE over equal-width confidence bins, plus the per-bin table."""
    confidence = np.asarray(confidence, np.float64)
    correct = np.asarray(correct, np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        m = (confidence > lo) & (confidence <= hi) if i else (confidence >= lo) & (confidence <= hi)
        if not m.any():
            bins.append({"lo": lo, "hi": hi, "n": 0, "confidence": None, "accuracy": None})
            continue
        conf, acc = confidence[m].mean(), correct[m].mean()
        ece += m.mean() * abs(conf - acc)
        bins.append(
            {"lo": lo, "hi": hi, "n": int(m.sum()), "confidence": float(conf), "accuracy": float(acc)}
        )
    return float(ece), bins


def coverage_curve(confidence: np.ndarray, correct: np.ndarray) -> list[dict]:
    """Accuracy of the most-confident X% of predictions (selective prediction)."""
    order = np.argsort(-np.asarray(confidence))
    correct = np.asarray(correct, np.float64)[order]
    conf = np.asarray(confidence)[order]
    out = []
    for cov in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        n = max(1, int(round(cov * len(correct))))
        out.append(
            {"coverage": cov, "accuracy": float(correct[:n].mean()), "min_confidence": float(conf[n - 1])}
        )
    return out


def summarize(true: np.ndarray, pred_int: np.ndarray, pred_soft: np.ndarray) -> dict:
    true = np.asarray(true, int)
    pred_int = np.asarray(pred_int, int)
    pred_soft = np.asarray(pred_soft, np.float64)
    diff = np.abs(pred_int - true)
    return {
        "n": int(len(true)),
        "mae_soft": float(np.abs(pred_soft - true).mean()),
        "mae": float(diff.mean()),
        "exact": float((diff == 0).mean()),
        "off_by_one": float((diff <= 1).mean()),
        "qwk": quadratic_weighted_kappa(true, pred_int),
        "bias": float((pred_soft - true).mean()),
    }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (well-behaved at small n / p=0,1)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def _group_members(groups) -> list[np.ndarray]:
    """Row indices of each group, groups in sorted label order."""
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    inverse = inverse.ravel()
    return [np.flatnonzero(inverse == g) for g in range(int(inverse.max()) + 1)]


def bootstrap_ci(
    stat: Callable[[np.ndarray], float],
    n: int,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI of ``stat(indices)`` resampling ``n`` items with replacement.

    With ``groups`` (one label per item) whole groups are resampled instead, so
    the rows of one card (several photo pairs) stay together.
    """
    if n <= 0 or n_boot <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    if groups is None:
        vals = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)], np.float64)
    else:
        members = _group_members(groups)
        k = len(members)
        vals = np.array(
            [stat(np.concatenate([members[g] for g in rng.integers(0, k, k)])) for _ in range(n_boot)],
            np.float64,
        )
    vals = vals[~np.isnan(vals)]
    if not len(vals):
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def overall_ci(
    true: np.ndarray,
    pred_int: np.ndarray,
    pred_soft: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> dict:
    """95% bootstrap CIs for the headline metrics (same resamples for each metric).

    With ``groups`` (one label per row, e.g. the cert) whole groups are
    resampled, so a card with two photo pairs is one draw; the result then also
    has ``n_groups``.
    """
    true = np.asarray(true, int)
    pred_int = np.asarray(pred_int, int)
    pred_soft = np.asarray(pred_soft, np.float64)
    n = len(true)
    if n == 0 or n_boot <= 0:
        return {}

    def pct(v: np.ndarray) -> list[float]:
        v = v[~np.isnan(v)]
        return [float(x) for x in np.percentile(v, [2.5, 97.5])] if len(v) else [float("nan")] * 2

    if groups is None:
        idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
        t, p, s = true[idx], pred_int[idx], pred_soft[idx]
        diff = np.abs(p - t)
        qwk = np.array([quadratic_weighted_kappa(t[i], p[i]) for i in range(n_boot)])
        return {
            "n_boot": int(n_boot),
            "mae": pct(diff.mean(1)),
            "exact": pct((diff == 0).mean(1).astype(np.float64)),
            "qwk": pct(qwk),
            "bias": pct((s - t).mean(1)),
        }
    # Resamples differ in length (cards can have different numbers of rows): loop.
    members = _group_members(groups)
    k = len(members)
    draws = np.random.default_rng(seed).integers(0, k, (n_boot, k))
    stats = np.empty((n_boot, 4), np.float64)
    for b in range(n_boot):
        i = np.concatenate([members[g] for g in draws[b]])
        diff = np.abs(pred_int[i] - true[i])
        stats[b] = (diff.mean(), (diff == 0).mean(), quadratic_weighted_kappa(true[i], pred_int[i]),
                    (pred_soft[i] - true[i]).mean())
    return {
        "n_boot": int(n_boot),
        "n_groups": k,
        "mae": pct(stats[:, 0]),
        "exact": pct(stats[:, 1]),
        "qwk": pct(stats[:, 2]),
        "bias": pct(stats[:, 3]),
    }


def per_grade(
    true: np.ndarray,
    pred_int: np.ndarray,
    pred_soft: np.ndarray,
    n_boot: int = 1000,
    groups: np.ndarray | None = None,
) -> dict:
    """Per true grade: support, recall (+Wilson 95% CI), MAE (+bootstrap CI), bias.

    ``groups`` makes the MAE CI resample groups (cards); the Wilson recall CI
    still counts rows.
    """
    true = np.asarray(true, int)
    pred_int = np.asarray(pred_int, int)
    pred_soft = np.asarray(pred_soft, np.float64)
    out = {}
    for g in range(1, NUM_GRADES + 1):
        m = true == g
        if not m.any():
            continue
        hits = int((pred_int[m] == g).sum())
        err = np.abs(pred_int[m] - g).astype(np.float64)
        out[str(g)] = {
            "support": int(m.sum()),
            "recall": float(hits / m.sum()),
            "recall_ci": list(wilson_interval(hits, int(m.sum()))),
            "mae": float(err.mean()),
            "mae_ci": list(bootstrap_ci(
                lambda i, e=err: e[i].mean(), len(err), n_boot, seed=g,
                groups=None if groups is None else np.asarray(groups)[m],
            )),
            "bias": float((pred_soft[m] - g).mean()),
        }
    return out


def baselines(true: np.ndarray, train_grades: np.ndarray | None) -> dict:
    ref = train_grades if train_grades is not None and len(train_grades) else true
    majority = int(np.bincount(ref, minlength=NUM_GRADES + 1).argmax())
    mean = float(np.mean(ref))
    return {
        "majority_grade": {"grade": majority, **summarize(true, np.full_like(true, majority), np.full(len(true), float(majority)))},
        "train_mean": {"grade": mean, **summarize(true, np.full_like(true, round(mean)), np.full(len(true), mean))},
    }


# --------------------------------------------------------------------------- #
# Perturbations for robustness
# --------------------------------------------------------------------------- #
def _rotate(bgr: np.ndarray, deg: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(bgr, rot, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _jpeg(bgr: np.ndarray, q: int) -> np.ndarray:
    _, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


PERTURBATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "blur": lambda im: cv2.GaussianBlur(im, (0, 0), 2.0),
    "jpeg_q40": lambda im: _jpeg(im, 40),
    "rotate_5deg": lambda im: _rotate(im, 5.0),
    "darker": lambda im: cv2.convertScaleAbs(im, alpha=0.7, beta=0),
    "brighter": lambda im: cv2.convertScaleAbs(im, alpha=1.15, beta=25),
    "rotate_180": lambda im: cv2.rotate(im, cv2.ROTATE_180),
}

# Perturbations of the (front, back) upload pair, on the raw bytes.
PAIR_PERTURBATIONS: dict[str, Callable[[bytes, bytes], tuple[bytes, bytes]]] = {
    "swap_faces": lambda front, back: (back, front),
}
ALL_PERTURBATIONS: tuple[str, ...] = (*PERTURBATIONS, *PAIR_PERTURBATIONS)


def parse_perturbations(text: str | None) -> list[str] | None:
    """Comma list of perturbation names -> list (None or "" -> None, i.e. all).

    Raises ValueError naming any unknown perturbation and the known ones. Text
    with no name in it (only commas and whitespace) raises the same error rather
    than silently meaning "all".
    """
    if text is None or text == "":
        return None
    names = [n.strip() for n in text.split(",") if n.strip()]
    unknown = [n for n in names if n not in ALL_PERTURBATIONS] if names else [repr(text)]
    if unknown:
        raise ValueError(
            f"unknown perturbation(s) {', '.join(unknown)}; known: {', '.join(ALL_PERTURBATIONS)}"
        )
    return list(dict.fromkeys(names))  # drop repeats, keep order


def apply_perturbation(name: str, front: bytes, back: bytes) -> tuple[bytes, bytes]:
    """Perturbed (front, back) upload bytes: a pair perturbation, or an image one per face."""
    if name in PAIR_PERTURBATIONS:
        return PAIR_PERTURBATIONS[name](front, back)
    return perturb_image(name, front), perturb_image(name, back)


def perturb_image(name: str, image: bytes) -> bytes:
    """One upload's bytes under the image perturbation `name`: decoded with
    `cv2.imdecode`, perturbed, and re-encoded as JPEG quality 95 (`_encode`).

    `ml.training.train_face_guard` builds its served-path rotated faces with
    `perturb_image("rotate_180", ...)`, so they match this harness's `rotate_180`.
    """
    fn = PERTURBATIONS[name]
    return _encode(fn(cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)))


def _encode(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


# --------------------------------------------------------------------------- #
# Running the model
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Sample:
    cert: str
    grade: int
    front: Path
    back: Path
    crop_kind: str = "unknown"  # combined front/back kind recorded by precrop
    # A manifest's extra columns (e.g. shot, cert, in_db_split); empty for DB splits.
    tags: dict[str, str] = field(default_factory=dict)


# Worst-first: a pair is as good as its worse side.
_KIND_ORDER = ("none", "small", "well", "unknown", "card")


def combine_crop_kinds(front: str | None, back: str | None) -> str:
    kinds = [k if k in _KIND_ORDER else "unknown" for k in (front, back)]
    return min(kinds, key=_KIND_ORDER.index)


def _crop_kinds_by_cert(certs: list[str]) -> dict[str, str]:
    """Crop kinds as recorded by precrop (same detector, same raw photos)."""
    from backend.app.core.config import get_settings
    from ml.data.db import connect

    if not certs:
        return {}
    out: dict[str, str] = {}
    with connect(get_settings().db_path) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(certs)")}
        if not {"crop_kind_front", "crop_kind_back"} <= cols:
            return {}
        for r in conn.execute("SELECT cert_number, crop_kind_front, crop_kind_back FROM certs"):
            out[r["cert_number"]] = combine_crop_kinds(r["crop_kind_front"], r["crop_kind_back"])
    return out


def load_samples(split: str, limit: int | None = None) -> list[Sample]:
    from ml.data.dataset import load_records

    # Raw photos, not precrops: evaluate the full serving path incl. detection.
    recs = load_records(split, use_precrop=False)
    recs.sort(key=lambda r: r.cert_number)
    if limit:
        recs = recs[:limit]
    kinds = _crop_kinds_by_cert([r.cert_number for r in recs])
    return [
        Sample(r.cert_number, r.grade, r.front_path, r.back_path, kinds.get(r.cert_number, "unknown"))
        for r in recs
    ]


_MANIFEST_COLUMNS = ("front", "back", "grade", "id")


def load_manifest(path: Path, limit: int | None = None) -> list[Sample]:
    """Labelled photos outside the DB, e.g. raw (unslabbed) cards.

    CSV with header columns ``front,back,grade`` and optional ``id``. Any other
    column is kept as a per-sample tag (`Sample.tags`, e.g. the phone set's
    ``shot``, ``cert``, ``in_db_split``): the report breaks results down by
    each (`by_tag`), and a ``cert`` column makes the CIs resample cards.
    Relative image paths resolve against the CSV's folder. Rows whose images
    are missing are skipped with a warning; a bad grade raises.
    """
    import csv

    path = Path(path)
    base = path.parent
    out: list[Sample] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = {c.strip().lower() for c in (reader.fieldnames or [])}
        missing = {"front", "back", "grade"} - cols
        if missing:
            raise ValueError(f"{path}: manifest missing columns {sorted(missing)}")
        tag_cols = [
            c for c in dict.fromkeys(c.strip().lower() for c in reader.fieldnames if c)
            if c not in _MANIFEST_COLUMNS
        ]
        for i, raw in enumerate(reader, start=2):
            row = {k.strip().lower(): (v or "").strip() for k, v in raw.items() if k}
            grade = int(float(row["grade"]))
            if not 1 <= grade <= NUM_GRADES:
                raise ValueError(f"{path}:{i}: grade {grade} outside 1..{NUM_GRADES}")
            front, back = Path(row["front"]), Path(row["back"])
            front = front if front.is_absolute() else base / front
            back = back if back.is_absolute() else base / back
            if not (front.exists() and back.exists()):
                log.warning("%s:%d: image missing, skipped (%s / %s)", path, i, front, back)
                continue
            tags = {c: row.get(c, "") for c in tag_cols}
            out.append(Sample(row.get("id") or f"{path.stem}:{i}", grade, front, back, tags=tags))
    out.sort(key=lambda s: s.cert)
    return out[:limit] if limit else out


def predict_all(
    service, samples: list[Sample], perturb: str | None = None, progress: bool = True
) -> dict:
    """Predict every sample; `perturb` names an entry of ALL_PERTURBATIONS to apply first."""
    from tqdm import tqdm

    keys = ("cert", "grade", "pred_int", "pred_soft", "confidence", "quad_front", "quad_back",
            "latency", "crop_kind", "notes", "reject")
    # Manifest tags become `tag:<column>` rows (none for DB splits).
    tag_keys = [f"{TAG_PREFIX}{c}" for c in dict.fromkeys(c for s in samples for c in s.tags)]
    rows = {k: [] for k in (*keys, *GUARD_PROBS, *tag_keys)}
    it = tqdm(samples, desc=perturb or "eval", disable=not progress)
    for s in it:
        front, back = s.front.read_bytes(), s.back.read_bytes()
        if perturb is not None:
            front, back = apply_perturbation(perturb, front, back)
        t0 = time.perf_counter()
        p = service.predict(front, back)
        rows["latency"].append(time.perf_counter() - t0)
        rows["cert"].append(s.cert)
        rows["notes"].append("; ".join(getattr(p, "notes", None) or []))
        # Would /api/grade refuse it? The kind, or "" when it would grade it.
        rejection = p.rejection() if hasattr(p, "rejection") else None
        rows["reject"].append(rejection[0] if rejection else "")
        rows["grade"].append(s.grade)
        rows["pred_int"].append(p.grade_int)
        rows["pred_soft"].append(p.grade)
        rows["confidence"].append(p.confidence)
        rows["quad_front"].append(p.found_quad_front)
        rows["quad_back"].append(p.found_quad_back)
        rows["crop_kind"].append(s.crop_kind)
        for key in tag_keys:
            rows[key].append(s.tags.get(key[len(TAG_PREFIX):], ""))
        decision = getattr(p, "guard_decision", None)
        for key in GUARD_PROBS:
            rows[key].append(_pair(getattr(decision, key, None)))
    return {k: np.asarray(v) for k, v in rows.items()}


TAG_PREFIX = "tag:"
# The tag column that names the card: CIs resample by it, and it gets no
# `by_tag` breakdown (one entry per card says nothing).
GROUP_TAG = "cert"

# The face guard's probabilities per prediction, (n, 2) with NaN where the guard
# didn't compute one: P(back) per uploaded row, P(flipped) per face after a swap.
GUARD_PROBS = ("p_back", "p_flip")


def _pair(p) -> tuple[float, float]:
    return (float("nan"), float("nan")) if p is None else (float(p[0]), float(p[1]))


def _guard_probs(rows: dict, i: int) -> dict:
    """The guard probabilities of row `i` that were computed, rounded for the report."""
    out = {}
    for key in GUARD_PROBS:
        if key in rows and np.isfinite(rows[key][i]).all():
            out[key] = [round(float(v), 6) for v in rows[key][i]]
    return out


def filter_rows(rows: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in rows.items()}


def crop_kind_breakdown(rows: dict) -> dict:
    """Count and MAE per crop kind, so loose (well) or bad crops are visible."""
    kinds = rows.get("crop_kind")
    if kinds is None or len(kinds) == 0:
        return {}
    err = np.abs(rows["pred_int"] - rows["grade"])
    return {
        str(k): {"n": int((kinds == k).sum()), "mae": float(err[kinds == k].mean())}
        for k in sorted(set(kinds.tolist()), key=_KIND_ORDER.index)
    }


def tag_breakdown(rows: dict) -> dict:
    """Per manifest tag column (but `GROUP_TAG`), per value: count, MAE, exact, would-reject.

    Empty when the rows have no tags (DB splits).
    """
    err = np.abs(rows["pred_int"] - rows["grade"])
    out = {}
    for key, values in rows.items():
        if not key.startswith(TAG_PREFIX) or key == TAG_PREFIX + GROUP_TAG:
            continue
        out[key[len(TAG_PREFIX):]] = {
            str(v): {
                "n": int(m.sum()),
                "mae": float(err[m].mean()),
                "exact": float((err[m] == 0).mean()),
                **({"n_would_reject": would_reject_counts(rows["reject"][m])} if "reject" in rows else {}),
            }
            for v in sorted(set(values.tolist()))
            for m in [values == v]
        }
    return out


def would_reject_counts(reject: np.ndarray) -> dict[str, int]:
    """How many inputs `/api/grade` would refuse (422), per rejection kind."""
    return {kind: int((reject == kind).sum()) for kind in ("card_not_found", "same_side")}


def build_report(
    rows: dict, train_grades: np.ndarray | None = None, n_boot: int = 1000
) -> dict:
    true, pi, ps, conf = rows["grade"], rows["pred_int"], rows["pred_soft"], rows["confidence"]
    correct = pi == true
    ece, rel_bins = expected_calibration_error(conf, correct)
    both_found = rows["quad_front"] & rows["quad_back"]
    band = np.isin(true, [9, 10])
    lat = rows["latency"] * 1000
    groups = rows.get(TAG_PREFIX + GROUP_TAG)  # a manifest's cert column: resample cards
    by_tag = tag_breakdown(rows)
    report = {
        "overall": summarize(true, pi, ps),
        "overall_ci": overall_ci(true, pi, ps, n_boot, groups=groups),
        "per_grade": per_grade(true, pi, ps, n_boot, groups=groups),
        "confusion_matrix": confusion(true, pi).tolist(),
        "grade_9_vs_10": summarize(true[band], pi[band], ps[band]) if band.any() else None,
        "calibration": {
            "ece": ece,
            "mean_confidence": float(conf.mean()),
            "accuracy": float(correct.mean()),
            "reliability": rel_bins,
            "coverage": coverage_curve(conf, correct),
        },
        "pipeline": {
            "quad_found_front": float(rows["quad_front"].mean()),
            "quad_found_back": float(rows["quad_back"].mean()),
            "mae_both_found": float(np.abs(pi - true)[both_found].mean()) if both_found.any() else None,
            "mae_fallback": float(np.abs(pi - true)[~both_found].mean()) if (~both_found).any() else None,
            "n_fallback": int((~both_found).sum()),
            "crop_kinds": crop_kind_breakdown(rows),
            **({"n_would_reject": would_reject_counts(rows["reject"])} if "reject" in rows else {}),
            **({"by_tag": by_tag} if by_tag else {}),
        },
        "latency_ms": {
            "p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95)),
            "mean": float(lat.mean()),
        },
        "baselines": baselines(true, train_grades),
    }
    if "cert" in rows:
        # Per cert, so two runs can be compared cert by cert.
        notes = rows.get("notes", [""] * len(true))
        report["predictions"] = [
            {
                "cert": str(c),
                "grade": int(g),
                "pred_int": int(i),
                "pred_soft": float(f),
                **({"notes": str(n)} if n else {}),
                **_guard_probs(rows, k),
            }
            for k, (c, g, i, f, n) in enumerate(zip(rows["cert"], true, pi, ps, notes, strict=True))
        ]
    return report


def robustness(service, samples: list[Sample], clean_rows: dict, names: list[str]) -> dict:
    """Run each perturbation on `samples` and compare with `clean_rows[:n]`.

    `clean_rows` must hold the same certs as `samples`, in the same order: the
    clean run of exactly these certs (run_evaluation passes the certs the report's
    metrics cover, so under exclude_fallback the fallback certs are left out of
    both).
    """
    n = len(samples)
    certs = [str(s.cert) for s in samples]
    if "cert" in clean_rows and [str(c) for c in clean_rows["cert"][:n]] != certs:
        raise ValueError("robustness: clean rows and samples are not the same certs in the same order")
    clean_mae = float(np.abs(clean_rows["pred_int"][:n] - clean_rows["grade"][:n]).mean())
    out = {}
    for name in names:
        rows = predict_all(service, samples, perturb=name)
        notes = rows.get("notes")
        out[name] = {
            "n": n,
            "mae": float(np.abs(rows["pred_int"] - rows["grade"]).mean()),
            "mae_delta": float(np.abs(rows["pred_int"] - rows["grade"]).mean() - clean_mae),
            "grade_flip_rate": float((rows["pred_int"] != clean_rows["pred_int"][:n]).mean()),
            "mean_abs_soft_shift": float(np.abs(rows["pred_soft"] - clean_rows["pred_soft"][:n]).mean()),
            "n_with_notes": int(sum(bool(x) for x in notes)) if notes is not None else 0,
            # Per cert, in the order of `cert` (= report["predictions"][:n], since
            # run_evaluation runs robustness on the certs the predictions list).
            "cert": certs,
            "pred_int": [int(v) for v in rows["pred_int"]],
            "pred_soft": [round(float(v), 4) for v in rows["pred_soft"]],
            "notes": [str(x) for x in notes] if notes is not None else [""] * n,
            # The face guard's probabilities per cert ({} without a guard).
            "guard": [_guard_probs(rows, i) for i in range(n)],
        }
    return out


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
# Reference palette (dataviz skill): sequential blue ramp, light chart surface.
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_GRID = "#e4e3df"
_SERIES = "#2a78d6"
_SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style(ax, title: str) -> None:
    ax.set_facecolor(_SURFACE)
    ax.set_title(title, loc="left", color=_INK, fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_INK_2, labelsize=9)
    ax.xaxis.label.set_color(_INK_2)
    ax.yaxis.label.set_color(_INK_2)


def write_charts(report: dict, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    paths = []

    # Confusion matrix: row-normalised so rare grades are readable; counts annotated.
    cm = np.asarray(report["confusion_matrix"], float)
    rows_present = cm.sum(1) > 0
    norm = np.divide(cm, cm.sum(1, keepdims=True), out=np.zeros_like(cm), where=cm.sum(1, keepdims=True) > 0)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=_SURFACE)
    cmap = LinearSegmentedColormap.from_list("seq", _SEQ)
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    for i in range(NUM_GRADES):
        for j in range(NUM_GRADES):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="#ffffff" if norm[i, j] > 0.55 else _INK)
    ticks = np.arange(NUM_GRADES)
    ax.set_xticks(ticks, [str(t + 1) for t in ticks])
    ax.set_yticks(ticks, [f"{t + 1}" + ("" if rows_present[t] else " –") for t in ticks])
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("True PSA grade")
    _style(ax, "Confusion matrix (colour = share of true grade, number = count)")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=_INK_2, labelsize=8)
    fig.tight_layout()
    p = out_dir / "confusion_matrix.png"
    fig.savefig(p, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    paths.append(p)

    # Reliability diagram.
    bins = [b for b in report["calibration"]["reliability"] if b["n"]]
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=_SURFACE)
    ax.plot([0, 1], [0, 1], color=_INK_2, lw=1, ls="--")
    ax.text(0.62, 0.55, "perfect calibration", color=_INK_2, fontsize=8, rotation=38)
    ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins],
            color=_SERIES, lw=2, marker="o", ms=8, markeredgecolor=_SURFACE, markeredgewidth=2)
    for b in bins:
        ax.annotate(f"n={b['n']}", (b["confidence"], b["accuracy"]), textcoords="offset points",
                    xytext=(6, -12), fontsize=7, color=_INK_2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean confidence in bin")
    ax.set_ylabel("Exact-grade accuracy in bin")
    ax.grid(color=_GRID, lw=0.6)
    _style(ax, f"Reliability  (ECE {report['calibration']['ece']:.3f})")
    fig.tight_layout()
    p = out_dir / "reliability.png"
    fig.savefig(p, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    paths.append(p)

    # Accuracy vs coverage.
    cov = report["calibration"]["coverage"]
    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=_SURFACE)
    ax.plot([c["coverage"] * 100 for c in cov], [c["accuracy"] for c in cov],
            color=_SERIES, lw=2, marker="o", ms=8, markeredgecolor=_SURFACE, markeredgewidth=2)
    ax.set_xlabel("Most-confident predictions kept (%)")
    ax.set_ylabel("Exact-grade accuracy")
    ax.set_ylim(0, 1)
    ax.grid(color=_GRID, lw=0.6)
    _style(ax, "Accuracy when rejecting low-confidence predictions")
    fig.tight_layout()
    p = out_dir / "coverage.png"
    fig.savefig(p, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    paths.append(p)
    return paths


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_evaluation(
    split: str = "test",
    limit: int | None = None,
    robustness_limit: int | None = 100,
    perturbations: list[str] | None = None,
    model_path: Path | None = None,
    progress: bool = True,
    manifest: Path | None = None,
    tta: bool = False,
    n_boot: int = 1000,
    exclude_fallback: bool = False,
) -> dict:
    """Evaluate a model. `exclude_fallback` drops cards where live detection
    fell back to the raw photo (for PSA slabs the label shows the grade) from
    the metrics; the pipeline section still describes every card."""
    from backend.app.ml.inference import GradeInferenceService

    service = GradeInferenceService(model_path=model_path, tta=tta)
    if not service.ready:
        reason = service.metadata.get("probe_load_error")
        if reason:
            raise RuntimeError(f"probe model not loaded: {reason}")
        raise RuntimeError(f"model not loaded from {service.model_path}")
    if manifest is not None:
        samples = load_manifest(manifest, limit)
        split = f"manifest:{Path(manifest).name}"
    else:
        samples = load_samples(split, limit)
    if not samples:
        raise RuntimeError(f"no samples with images on disk for split={split!r}")

    try:
        train_grades = np.asarray([s.grade for s in load_samples("train")], int)
    except Exception:  # noqa: BLE001
        train_grades = None

    rows = predict_all(service, samples, progress=progress)
    report = build_report(rows, train_grades, n_boot=n_boot)
    # The certs the metrics (and report["predictions"]) cover; robustness runs on
    # the first robustness_limit of these.
    metric_samples, metric_rows = samples, rows
    if exclude_fallback:
        both_found = rows["quad_front"] & rows["quad_back"]
        if not both_found.any():
            raise RuntimeError("every card fell back; nothing left to evaluate")
        pipeline = report["pipeline"]
        metric_rows = filter_rows(rows, both_found)
        metric_samples = [s for s, keep in zip(samples, both_found, strict=True) if keep]
        report = build_report(metric_rows, train_grades, n_boot=n_boot)
        report["pipeline"] = {**pipeline, "excluded_from_metrics": int((~both_found).sum())}
    report["meta"] = {
        "split": split,
        "manifest": str(manifest) if manifest is not None else None,
        "tta": list(service.tta),
        "exclude_fallback": exclude_fallback,
        "model_path": str(service.model_path),
        "model_metadata": service.metadata,
        "providers": service._session.get_providers(),
    }
    if robustness_limit and perturbations is not False:
        names = perturbations or list(ALL_PERTURBATIONS)
        unknown = [n for n in names if n not in ALL_PERTURBATIONS]
        if unknown:
            raise ValueError(f"unknown perturbation(s): {', '.join(unknown)}")
        # Robustness runs on the certs the metrics cover, not on every loaded
        # sample: under exclude_fallback the fallback certs are dropped here too.
        # That keeps every existing robustness field's meaning - `n` certs, MAE and
        # flips against the clean run of those same certs, per-cert lists in the
        # order of report["predictions"][:n] - rather than redefining the lists as
        # "samples[:n], fallback certs included". `cert` names each entry.
        report["robustness"] = robustness(
            service, metric_samples[:robustness_limit], metric_rows, names
        )
    return report


def compare_to_baseline(report: dict, baseline: dict, margins: dict | None = None) -> list[str]:
    """Return human-readable regressions (empty list = OK)."""
    margins = margins or {"mae": 0.10, "off_by_one": 0.03, "qwk": 0.05, "ece": 0.03}
    cur = {**report["overall"], "ece": report["calibration"]["ece"]}
    problems = []
    for key, margin in margins.items():
        if key not in baseline:
            continue
        lower_is_better = key in ("mae", "ece")
        limit = baseline[key] + margin if lower_is_better else baseline[key] - margin
        bad = cur[key] > limit if lower_is_better else cur[key] < limit
        if bad:
            problems.append(f"{key}: {cur[key]:.4f} vs baseline {baseline[key]:.4f} (limit {limit:.4f})")
    return problems


app = typer.Typer(add_completion=False)


@app.command()
def main(
    split: str = typer.Option("test", help="Dataset split to evaluate."),
    limit: int | None = typer.Option(None, help="Cap number of cards."),
    robustness_limit: int = typer.Option(100, help="Cards used per perturbation (0 disables)."),
    no_robustness: bool = typer.Option(False, help="Skip the perturbation runs."),
    perturbations: str | None = typer.Option(
        None,
        help="Comma list of perturbations to run (default: all of "
        + ", ".join(ALL_PERTURBATIONS) + ").",
    ),
    model_path: Path | None = typer.Option(None, help="Override ONNX model path."),
    out_dir: Path = typer.Option(DEFAULT_OUT, help="Where to write report.json and charts."),
    save_baseline: bool = typer.Option(False, help="Pin these numbers as the regression baseline."),
    manifest: Path | None = typer.Option(
        None, exists=True, readable=True,
        help="Evaluate a labelled CSV (front,back,grade[,id], extra columns kept as tags) "
        "instead of a DB split, e.g. raw cards or the phone set.",
    ),
    tta: bool = typer.Option(
        False, help="Test-time augmentation: average over hflip + dim/bright variants."
    ),
    n_boot: int = typer.Option(1000, help="Bootstrap resamples for 95% CIs (0 disables)."),
    exclude_fallback: bool = typer.Option(
        False, help="Leave cards whose detection fell back (label visible) out of the metrics."
    ),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if save_baseline and (manifest is not None or tta):
        raise typer.BadParameter("--save-baseline only pins a plain DB split (no --manifest / --tta)")
    try:
        names = parse_perturbations(perturbations)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--perturbations") from exc
    report = run_evaluation(
        split=split,
        limit=limit,
        robustness_limit=0 if no_robustness else robustness_limit,
        perturbations=names,
        model_path=model_path,
        manifest=manifest,
        tta=tta,
        n_boot=n_boot,
        exclude_fallback=exclude_fallback,
    )
    split = report["meta"]["split"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    charts = write_charts(report, out_dir)

    o, c, b = report["overall"], report["calibration"], report["baselines"]
    typer.echo(f"\n{split}: n={o['n']}")
    typer.echo(f"  MAE {o['mae']:.3f} (soft {o['mae_soft']:.3f})  exact {o['exact']:.1%}  "
               f"off-by-one {o['off_by_one']:.1%}  QWK {o['qwk']:.3f}  bias {o['bias']:+.2f}")
    ci = report.get("overall_ci") or {}
    if ci:
        if "n_groups" in ci:
            typer.echo(f"  (CIs resample {ci['n_groups']} cards, not rows)")
        typer.echo(f"  95% CI: MAE [{ci['mae'][0]:.3f}, {ci['mae'][1]:.3f}]  "
                   f"QWK [{ci['qwk'][0]:.3f}, {ci['qwk'][1]:.3f}]  "
                   f"exact [{ci['exact'][0]:.1%}, {ci['exact'][1]:.1%}]  "
                   f"bias [{ci['bias'][0]:+.2f}, {ci['bias'][1]:+.2f}]")
    typer.echo(f"  ECE {c['ece']:.3f}  mean conf {c['mean_confidence']:.2f}")
    typer.echo("  per grade:  n   recall [95% CI]        MAE [95% CI]          bias")
    for g, r in sorted(report["per_grade"].items(), key=lambda kv: -int(kv[0])):
        rl, rh = r["recall_ci"]
        ml_, mh = r["mae_ci"]
        typer.echo(f"    {g:>2} {r['support']:>5}  {r['recall']:6.1%} [{rl:5.1%}, {rh:5.1%}]  "
                   f"{r['mae']:5.2f} [{ml_:4.2f}, {mh:4.2f}]  {r['bias']:+.2f}")
    typer.echo(f"  baseline majority={b['majority_grade']['grade']}: MAE {b['majority_grade']['mae']:.3f} "
               f"exact {b['majority_grade']['exact']:.1%}")
    pl = report["pipeline"]
    typer.echo(f"  quad found: front {pl['quad_found_front']:.1%}  back {pl['quad_found_back']:.1%}"
               + (f"  ({pl['excluded_from_metrics']} fallback cards excluded from metrics)"
                  if "excluded_from_metrics" in pl else ""))
    if "n_would_reject" in pl:
        typer.echo("  /api/grade would refuse: " + "  ".join(
            f"{k} {n}" for k, n in pl["n_would_reject"].items()))
    if pl.get("crop_kinds"):
        typer.echo("  crop kinds: " + "  ".join(
            f"{k} n={v['n']} MAE {v['mae']:.2f}" for k, v in pl["crop_kinds"].items()))
    for tag, values in pl.get("by_tag", {}).items():
        typer.echo(f"  by {tag}: " + "  ".join(
            f"{k} n={v['n']} MAE {v['mae']:.2f} exact {v['exact']:.0%}"
            + (f" refused {sum(v['n_would_reject'].values())}" if "n_would_reject" in v else "")
            for k, v in values.items()))
    typer.echo(f"  latency p50 {report['latency_ms']['p50']:.0f} ms  p95 {report['latency_ms']['p95']:.0f} ms")
    for name, r in report.get("robustness", {}).items():
        typer.echo(f"  {name:12s} MAE {r['mae']:.3f} ({r['mae_delta']:+.3f})  flips {r['grade_flip_rate']:.1%}")
    typer.echo(f"\nwrote {out_dir / 'report.json'} and {len(charts)} charts")

    if BASELINE_PATH.exists():
        problems = compare_to_baseline(report, json.loads(BASELINE_PATH.read_text()))
        typer.echo("regressions vs baseline: " + ("; ".join(problems) if problems else "none"))
    if save_baseline:
        BASELINE_PATH.write_text(json.dumps({**o, "ece": c["ece"], "split": split}, indent=2))
        typer.echo(f"saved baseline -> {BASELINE_PATH}")


if __name__ == "__main__":
    app()
