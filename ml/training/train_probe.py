"""Train the frozen-DINOv2 + linear ordinal grade head.

    python -m ml.training.train_probe                  # features -> CV -> fit -> test -> export
    python -m ml.training.train_probe --force-onnx     # also re-export the feature ONNX

Steps:
1. Region features for every cert with a usable precrop (front and back
   outline found), extracted with the same graph that is exported for serving.
   Cached in data/models/cache/ so refitting the head takes seconds; an entry is
   reused only while its crop files (size + mtime) and the feature spec
   (`features.feature_spec_hash`) are unchanged.
2. Card-grouped 5-fold CV over train+val: certs of the same printed card never
   straddle folds, so a probe can't score by recognising the card.
3. Final head fitted on train+val, evaluated once on the test split.
4. Writes data/models/card_features.onnx (+ .json sidecar) when it is missing
   or was exported from another feature spec, then data/models/grade_head.npz
   (+ .json metadata, returned by /api/health). The graph comes first so a
   failed export never leaves a new head beside an old graph.

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
    FEATURE_SPEC,
    REGION_NAMES,
    build_region_model,
    export_onnx,
    feature_spec_hash,
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
            "cropped_front_path, cropped_back_path, front_path, back_path FROM certs "
            "WHERE is_pokemon=1 AND has_images=1 AND grade_int IS NOT NULL "
            "AND split IN ('train', 'val', 'test') "
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


def _file_sig(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def extract_features(
    rows: list[sqlite3.Row],
    data_dir: Path,
    cache: Path,
    batch: int = 8,
    model_factory=build_region_model,
    transform=None,
    variant: str = "",
    save: bool = True,
    path_keys: tuple[str, str] = ("cropped_front_path", "cropped_back_path"),
    load=None,
    workers: int = 1,
) -> np.ndarray:
    """(N, 2, n_regions, D) float32 features, reusing cached certs whose source files are unchanged.

    Cache keys are (cert, front path, back path, front sig, back sig), sig being
    the source file's "size:mtime_ns", so a rewritten file is re-extracted. The
    whole cache is discarded when it was built from another feature spec, and
    only the current rows' entries are saved back.

    The source files are the row's `path_keys` columns (default the precrops),
    relative to `data_dir`. `load` (Path -> RGB) reads one into the image fed to
    `to_input` (default: the file decoded as is); `transform` (RGB -> RGB) is
    applied after it. A cache of other sources, loaders or transforms must name
    them in `variant` (e.g. "rot180"), which the cache records and must match
    like the feature spec. `save=False` reads the cache but never writes it
    (missing certs are still extracted). `workers` > 1 loads a batch's images in
    that many threads (the result doesn't depend on it).
    """
    import torch

    spec_hash = feature_spec_hash()
    cached: dict[tuple[str, ...], np.ndarray] = {}
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            have = str(z["feature_spec_hash"]) if "feature_spec_hash" in z.files else None
            have_variant = str(z["variant"]) if "variant" in z.files else ""
            if have == spec_hash and have_variant == variant:
                cached = dict(zip(map(tuple, z["keys"]), z["feats"], strict=True))
            else:
                log.info(
                    "feature cache %s has feature_spec_hash %s variant %r, not %s %r: "
                    "re-extracting every cert",
                    cache,
                    have,
                    have_variant,
                    spec_hash,
                    variant,
                )
    front_key, back_key = path_keys
    keys = [
        (
            r["cert_number"],
            r[front_key],
            r[back_key],
            _file_sig(data_dir / r[front_key]),
            _file_sig(data_dir / r[back_key]),
        )
        for r in rows
    ]
    todo = [i for i, k in enumerate(keys) if k not in cached]
    log.info("features %s: %d cached, %d to extract", cache.name, len(keys) - len(todo), len(todo))
    read = load or _read_rgb

    def image(path: str) -> np.ndarray:
        rgb = read(data_dir / path)
        return to_input(transform(rgb) if transform else rgb)

    if todo:
        from concurrent.futures import ThreadPoolExecutor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model_factory().to(device)
        pool = ThreadPoolExecutor(workers) if workers > 1 else None
        try:
            with torch.no_grad():
                for s in tqdm(range(0, len(todo), batch), desc="features"):
                    idx = todo[s : s + batch]
                    paths = [keys[i][side] for i in idx for side in (1, 2)]
                    imgs = list(pool.map(image, paths)) if pool else [image(p) for p in paths]
                    out = model(torch.from_numpy(np.stack(imgs)).to(device)).float().cpu().numpy()
                    for j, i in enumerate(idx):
                        cached[keys[i]] = out[2 * j : 2 * j + 2].astype(np.float16)
        finally:
            if pool is not None:
                pool.shutdown()
    if save and (todo or set(cached) != set(keys)):
        # Save the current rows only: entries for dropped certs or old crops go.
        cache.parent.mkdir(parents=True, exist_ok=True)
        extra = {"variant": np.array(variant)} if variant else {}
        np.savez(
            cache,
            feature_spec_hash=np.array(spec_hash),
            keys=np.array(keys),
            feats=np.stack([cached[k] for k in keys]),
            **extra,
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


def _onnx_export_reason(onnx_path: Path, force: bool) -> str | None:
    """Why card_features.onnx must be (re-)exported, or None if it matches the feature spec."""
    if force:
        return "--force-onnx"
    sidecar = onnx_path.with_suffix(".json")
    if not onnx_path.exists():
        return "graph missing"
    if not sidecar.exists():
        return f"{sidecar.name} missing"
    try:
        have = json.loads(sidecar.read_text()).get("feature_spec_hash")
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return f"{sidecar.name} unreadable"
    if have != feature_spec_hash():
        return f"feature_spec_hash {have} != {feature_spec_hash()}"
    return None


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
        # Unless the split is card-grouped (`split.py --mode grouped`), some test
        # cards also appear in train+val; report both halves.
        unseen = ~np.isin(groups[test_mask], groups[fit_mask])
        report["n_test_seen"] = int((~unseen).sum())
        if unseen.sum() >= 20:
            report["test_unseen_cards"] = _rounded(metrics(y[test_mask][unseen], test_cum[unseen]))
        if (~unseen).sum() >= 20:
            report["test_seen_cards"] = _rounded(metrics(y[test_mask][~unseen], test_cum[~unseen]))
        log.info("test: %s", report["test"])

    head.metadata = {
        "kind": "dinov2_region_probe",
        "backbone": BACKBONE,
        "regions": list(REGION_NAMES),
        "n_fit": int(fit_mask.sum()),
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "feature_spec_hash": feature_spec_hash(),
        "feature_spec": FEATURE_SPEC,
        **{
            k: v
            for k, v in report.items()
            if k in ("cv_grouped", "test", "test_seen_cards", "n_test_seen")
        },
    }
    # Graph first: if the export fails, no new head is written beside an old graph.
    onnx_path = settings.models_dir / "card_features.onnx"
    reason = _onnx_export_reason(onnx_path, force_onnx)
    if reason:
        log.info("exporting feature model -> %s (%s)", onnx_path, reason)
        export_onnx(onnx_path)

    head_path = settings.models_dir / "grade_head.npz"
    head.save(head_path)
    log.info("wrote %s", head_path)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    log.info("report -> %s", REPORT_PATH)


if __name__ == "__main__":
    app()
