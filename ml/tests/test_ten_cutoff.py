"""train_probe's P(10) cut-off: the tuning criterion and what main() stores (intent/grade-decision-rule)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from backend.app.ml.features import REGION_NAMES
from backend.app.ml.grade_head import GradeHead, decode
from ml.training import train_probe

D = 8


def _cum(grade: int, p10: float | None = None) -> np.ndarray:
    """Confident cum probs for `grade`, or near-certain >= 9 with P(10) = p10."""
    cum = np.where(np.arange(1, 10) < grade, 0.999, 0.001)
    if p10 is not None:
        cum[:8], cum[8] = 0.999, p10
    return cum


def _design(groups):
    """(grades, cum) from [(count, grade, p10 or None), ...]."""
    y = np.array([g for n, g, _ in groups for _ in range(n)])
    cum = np.array([_cum(g, p) for n, g, p in groups for _ in range(n)])
    return y, cum


# 9s that only a cut-off above 0.55 calls 9; 10s that a cut-off of 0.66 or more loses.
FIT = [(5, 9, 0.55), (20, 10, 0.65), (75, 10, 0.95), (10, 8, None), (10, 7, None)]


def test_highest_cutoff_within_the_exact_floor():
    y, cum = _design(FIT)
    t, tuning = train_probe.tune_ten_cutoff(y, [cum] * 5)
    # 0.56-0.64 gain the five 9s; from 0.66 the twenty 10s are lost (> 1 pp).
    assert t == 0.64
    assert tuning["median"]["exact"] == [round(115 / 120, 4)] * 5
    assert tuning["chosen"]["exact"] == [1.0] * 5
    assert tuning["grid_mean"]["0.66"]["exact"] == round(100 / 120, 4)
    assert tuning["n"] == 120 and tuning["n_seeds"] == 5 and tuning["criterion"]


def test_mae_ceiling_binds_without_seed_spread():
    # Losing two 10s of 200 costs exactly 1 pp (allowed) but 0.01 MAE over an SD of 0.
    y, cum = _design([(2, 10, 0.53), (198, 10, 0.95)])
    t, tuning = train_probe.tune_ten_cutoff(y, [cum] * 5)
    assert t == 0.52
    assert tuning["median"]["mae_sd"] == 0.0


def test_seed_spread_loosens_the_mae_ceiling():
    # The same two 10s, now with 8s whose calls differ by seed: the median's MAE SD
    # (~0.029) is bigger than the 0.009 the cut-off costs, so the top of the grid passes.
    y, base = _design([(2, 10, 0.53), (198, 10, 0.95), (20, 8, None)])
    cums = []
    for s in range(5):
        cum = base.copy()
        cum[200 : 200 + 4 * s, 7] = 0.6  # called 9
        cums.append(cum)
    t, tuning = train_probe.tune_ten_cutoff(y, cums)
    assert t == 0.70
    assert tuning["median"]["mae_sd"] > 0.02


def test_cutoff_never_below_the_median():
    y, cum = _design([(50, 10, 0.51), (10, 9, 0.2)])  # any cut-off above 0.5 loses 50 10s
    assert train_probe.tune_ten_cutoff(y, [cum] * 5)[0] == 0.5


# --------------------------------------------------------------------------- #
# main(): seeds 0-4, cv_grouped = seed 0, t stored in the head
# --------------------------------------------------------------------------- #
@pytest.fixture
def training(tmp_path, monkeypatch):
    """main() over synthetic rows; cross_validate returns the FIT design for every seed."""
    rng = np.random.default_rng(0)
    y_fit, cum_fit = _design(FIT)
    y_test = np.array([7, 8, 9, 9, 10, 10, 10, 10] * 4)
    y = np.concatenate([y_fit, y_test])
    latent = y - 9 + rng.normal(scale=0.5, size=len(y))
    direction = rng.normal(size=(2, len(REGION_NAMES), D))
    feats = latent[:, None, None, None] * direction + rng.normal(size=(len(y), 2, len(REGION_NAMES), D))
    rows = [
        {
            "cert_number": f"c{i}",
            "grade_int": int(g),
            "split": "train" if i < len(y_fit) else "test",
            "brand_title": "set",
            "subject": f"card{i // 2}",
            "card_number": "1",
        }
        for i, g in enumerate(y)
    ]
    models = tmp_path / "models"
    models.mkdir()
    (models / "card_features.onnx").write_bytes(b"onnx")
    from backend.app.ml.features import feature_spec_hash

    (models / "card_features.json").write_text(json.dumps({"feature_spec_hash": feature_spec_hash()}))
    seeds = []

    def cross_validate(f, yy, groups, n_folds=5, seed=0):
        assert len(yy) == len(y_fit) and n_folds == 3
        seeds.append(seed)
        return cum_fit, np.zeros((len(yy), 4))

    settings = SimpleNamespace(db_path=tmp_path / "t.sqlite", data_dir=tmp_path, models_dir=models)
    monkeypatch.setattr(train_probe, "get_settings", lambda: settings)
    monkeypatch.setattr(train_probe, "load_rows", lambda _db: rows)
    monkeypatch.setattr(train_probe, "extract_features", lambda *a, **k: feats.astype(np.float32))
    monkeypatch.setattr(train_probe, "cross_validate", cross_validate)
    monkeypatch.setattr(train_probe, "REPORT_PATH", tmp_path / "probe_report.json")
    return SimpleNamespace(
        models=models, seeds=seeds, y_fit=y_fit, cum_fit=cum_fit, y_test=y_test,
        feats_test=feats[len(y_fit):], report=tmp_path / "probe_report.json",
    )


def test_main_tunes_on_seeds_0_to_4_and_stores_the_cutoff(training):
    train_probe.main(force_onnx=False, cv_folds=3)
    assert training.seeds == [0, 1, 2, 3, 4]
    meta = json.loads((training.models / "grade_head.json").read_text())
    assert meta["ten_cutoff"] == 0.64
    assert meta["ten_cutoff_tuning"]["chosen"]["exact"] == [1.0] * 5
    # cv_grouped (seed 0) and the test metrics use the tuned cut-off.
    assert meta["cv_grouped"] == train_probe._rounded(train_probe.metrics(training.y_fit, training.cum_fit, 0.64))
    assert meta["cv_grouped"]["exact"] == 1.0
    head = GradeHead.load(training.models / "grade_head.npz")
    assert head.ten_cutoff == 0.64
    test_cum = head.meta.cum_probs(head.region_grades(training.feats_test))
    assert meta["test"] == train_probe._rounded(train_probe.metrics(training.y_test, test_cum, 0.64))
    report = json.loads(training.report.read_text())
    assert report["ten_cutoff"] == 0.64 and report["cv_grouped"] == meta["cv_grouped"]


def test_main_without_cv_keeps_the_median(training):
    train_probe.main(force_onnx=False, cv_folds=0)
    assert training.seeds == []
    meta = json.loads((training.models / "grade_head.json").read_text())
    assert meta["ten_cutoff"] == 0.5 and "ten_cutoff_tuning" not in meta
    head = GradeHead.load(training.models / "grade_head.npz")
    test_cum = head.meta.cum_probs(head.region_grades(training.feats_test))
    assert meta["test"] == train_probe._rounded(train_probe.metrics(training.y_test, test_cum))
    assert [decode(c, head.ten_cutoff)[1] for c in test_cum] == [decode(c)[1] for c in test_cum]
