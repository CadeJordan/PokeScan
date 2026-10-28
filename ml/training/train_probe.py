"""Train the frozen-DINOv2 + linear ordinal grade head.

    python -m ml.training.train_probe                  # features -> CV -> fit -> test -> export
    python -m ml.training.train_probe --force-onnx     # also re-export the feature ONNX

Steps:
1. Region features for every cert with a usable precrop (front and back
   outline found), extracted with the same graph that is exported for serving.
   Cached in data/models/cache/ so refitting the head takes seconds.
2. Card-grouped 5-fold CV over train+val: certs of the same printed card never
   straddle folds, so a probe can't score by recognising the card.
3. Final head fitted on train+val, evaluated once on the test split.
4. Writes data/models/grade_head.npz (+ .json metadata, returned by
   /api/health) and data/models/card_features.onnx.

Run `python -m ml.data.precrop` first. Restart the API afterwards.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import typer
from rich.logging import RichHandler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from backend.app.core.config import PROJECT_ROOT, get_settings
from backend.app.ml.features import (
    BACKBONE,
    REGION_NAMES,
    build_region_model,
    export_onnx,
    to_input,
)
from backend.app.ml.grade_head import decode
from ml.eval.evaluate import expected_calibration_error, summarize
from ml.training.probe import L2, PCA_DIM, card_groups, cross_validate, fit_grade_head

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True)])
log = logging.getLogger("train_probe")

app = typer.Typer(add_completion=False)

REPORT_PATH = PROJECT_ROOT / "test-artifacts" / "eval" / "probe_report.json"


def load_rows(db_path: Path) -> list[sqlite3.Row]:
    """Certs with a usable precrop (read-only: never migrates or writes the DB)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT cert_number, grade_int, split, brand_title, subject, card_number, "
            "cropped_front_path, cropped_back_path FROM certs "
            "WHERE is_pokemon=1 AND has_images=1 AND grade_int IS NOT NULL AND split IS NOT NULL "
            "AND crop_found_front=1 AND crop_found_back=1 "
            "AND cropped_front_path IS NOT NULL AND cropped_back_path IS NOT NULL "
            "ORDER BY cert_number"
        ).fetchall()
    finally:
        conn.close()


def _read_rgb(path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def extract_features(rows: list[sqlite3.Row], data_dir: Path, cache: Path, batch: int = 8) -> np.ndarray:
    """(N, 2, n_regions, D) float32 features, reusing cached certs whose crop paths match."""
    import torch

    cached: dict[tuple[str, str, str], np.ndarray] = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["backbone"]) == BACKBONE:
            for key, f in zip(map(tuple, z["keys"]), z["feats"], strict=True):
                cached[key] = f
    keys = [(r["cert_number"], r["cropped_front_path"], r["cropped_back_path"]) for r in rows]
    todo = [i for i, k in enumerate(keys) if k not in cached]
    log.info("features: %d cached, %d to extract", len(keys) - len(todo), len(todo))
    if todo:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_region_model().to(device)
        with torch.no_grad():
            for s in tqdm(range(0, len(todo), batch), desc="features"):
                idx = todo[s : s + batch]
                imgs = [
                    to_input(_read_rgb(data_dir / keys[i][side]))
                    for i in idx
                    for side in (1, 2)
                ]
                out = model(torch.from_numpy(np.stack(imgs)).to(device)).float().cpu().numpy()
                for j, i in enumerate(idx):
                    cached[keys[i]] = out[2 * j : 2 * j + 2].astype(np.float16)
        cache.parent.mkdir(parents=True, exist_ok=True)
        all_keys = list(cached)
        np.savez(
            cache,
            backbone=np.array(BACKBONE),
            keys=np.array(all_keys),
            feats=np.stack([cached[k] for k in all_keys]),
        )
    return np.stack([cached[k] for k in keys]).astype(np.float32)


def metrics(y: np.ndarray, cum: np.ndarray) -> dict:
    decoded = [decode(c) for c in cum]
    soft = np.array([d[0] for d in decoded])
    hard = np.array([d[1] for d in decoded])
    conf = np.array([d[2] for d in decoded])
    out = summarize(y, hard, soft)
    out["ece"], _ = expected_calibration_error(conf, hard == y)
    if (y <= 8).any() and (y >= 9).any():
        out["auc_le8_vs_ge9"] = float(roc_auc_score(y <= 8, -soft))
    top = y >= 9
    if (y[top] == 10).any() and (y[top] == 9).any():
        out["auc_10_vs_9"] = float(roc_auc_score(y[top] == 10, soft[top]))
    return out


def majority_baseline(y: np.ndarray, majority: int) -> dict:
    pred = np.full(len(y), majority)
    return summarize(y, pred, pred.astype(float))


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _rounded(d: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


@app.command()
def main(
    force_onnx: bool = typer.Option(False, help="Re-export the feature ONNX even if it exists."),
    cv_folds: int = typer.Option(5, help="Card-grouped CV folds over train+val (0 = skip CV)."),
) -> None:
    settings = get_settings()
    rows = load_rows(settings.db_path)
    if not rows:
        raise SystemExit("no certs with usable precrops - run `python -m ml.data.precrop`")
    feats = extract_features(rows, settings.data_dir, settings.models_dir / "cache" / "region_feats.npz")
    y = np.array([r["grade_int"] for r in rows], int)
    split = np.array([r["split"] for r in rows])
    groups = card_groups(
        [r["brand_title"] or "" for r in rows],
        [r["subject"] or "" for r in rows],
        [r["card_number"] or "" for r in rows],
    )
    fit_mask = np.isin(split, ("train", "val"))
    test_mask = split == "test"
    majority = int(np.bincount(y[fit_mask]).argmax())
    log.info("certs: fit (train+val)=%d test=%d", fit_mask.sum(), test_mask.sum())

    report: dict = {"backbone": BACKBONE, "pca_dim": PCA_DIM, "l2": L2}
    if cv_folds:
        cum, _ = cross_validate(feats[fit_mask], y[fit_mask], groups[fit_mask], n_folds=cv_folds)
        report["cv_grouped"] = _rounded(metrics(y[fit_mask], cum))
        report["cv_majority_baseline"] = _rounded(majority_baseline(y[fit_mask], majority))
        log.info("grouped CV: %s", report["cv_grouped"])

    head = fit_grade_head(feats[fit_mask], y[fit_mask], groups[fit_mask])
    if test_mask.any():
        test_cum = head.meta.cum_probs(head.region_grades(feats[test_mask]))
        report["test"] = _rounded(metrics(y[test_mask], test_cum))
        report["test_majority_baseline"] = _rounded(majority_baseline(y[test_mask], majority))
        # The split was made per cert, so some test cards also appear in train+val.
        unseen = ~np.isin(groups[test_mask], groups[fit_mask])
        if unseen.sum() >= 20:
            report["test_unseen_cards"] = _rounded(metrics(y[test_mask][unseen], test_cum[unseen]))
        log.info("test: %s", report["test"])

    head.metadata = {
        "kind": "dinov2_region_probe",
        "backbone": BACKBONE,
        "regions": list(REGION_NAMES),
        "n_fit": int(fit_mask.sum()),
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        **{k: v for k, v in report.items() if k in ("cv_grouped", "test")},
    }
    head_path = settings.models_dir / "grade_head.npz"
    head.save(head_path)
    log.info("wrote %s", head_path)

    onnx_path = settings.models_dir / "card_features.onnx"
    if force_onnx or not onnx_path.exists():
        log.info("exporting feature model -> %s", onnx_path)
        export_onnx(onnx_path)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    log.info("report -> %s", REPORT_PATH)


if __name__ == "__main__":
    app()
