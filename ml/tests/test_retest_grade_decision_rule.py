"""Independent retest of intent/grade-decision-rule (ADR 0003).

Gaps not pinned by test_grade_head.py / test_inference.py / test_ten_cutoff.py:
- the shipped head's `ten_cutoff` is what its own stored `ten_cutoff_tuning` gives under
  decision 1 (re-derived from the stored per-seed and grid numbers, not from code);
- on the saved OOF predictions (if present) `tune_ten_cutoff` picks the shipped cut-off;
- decode at a non-monotone input with P(>8) <= 0.5 < P(>9): the cut-off never adds a
  grade the median wouldn't give, and confidence is still P(grade_int);
- the cut-off grid and the criterion constants are the ones the owner decided.
All real data is read only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.app.ml.grade_head import DEFAULT_TEN_CUTOFF, TEN_CUTOFF_KEY, decode, ten_cutoff
from ml.training import train_probe

ROOT = Path(__file__).resolve().parents[2]
HEAD_JSON = ROOT / "data" / "models" / "grade_head.json"
OOF = ROOT / "test-artifacts" / "eval" / "grade_decision_rule" / "oof_seeds.npz"
# The decision-rule head itself, backed up before tighter-slab-crops retrained it.
DECISION_RULE_HEAD_JSON = ROOT / "data" / "backups" / "grade_head.json.pre-tighter-crops-2026-10-31"


def test_grid_and_seeds_are_the_owners():
    assert train_probe.CUTOFF_GRID == (0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7)
    assert train_probe.CUTOFF_SEEDS == (0, 1, 2, 3, 4)
    assert DEFAULT_TEN_CUTOFF == 0.5 and TEN_CUTOFF_KEY == "ten_cutoff"


@pytest.mark.parametrize("t", [0.5, 0.58, 0.7])
def test_non_monotone_p9_above_cutoff_with_p8_below_half(t):
    # P(>8) = 0.4 <= 0.5 < P(>9) = 0.75: the count rule gives 9 (as the median did),
    # never 10, and the cut-off can't change it since P(>9) > t.
    p = np.array([1, 1, 1, 1, 1, 1, 1, 0.4, 0.75])
    _, hard, conf, per_grade = decode(p, t)
    assert hard == decode(p)[1] == 9
    assert conf == per_grade[hard - 1]


def test_cutoff_only_ever_lowers_the_grade_by_one():
    rng = np.random.default_rng(11)
    for _ in range(3000):
        p = rng.uniform(size=9)
        if rng.random() < 0.5:
            p = np.sort(p)[::-1]
        p = np.round(p, 2)  # lands exactly on grid values sometimes
        med = decode(p)[1]
        for t in train_probe.CUTOFF_GRID:
            hard = decode(p, t)[1]
            assert hard in (med, med - 1)
            assert (hard == med - 1) == (0.5 < p[8] <= t)


def _criterion_from_tuning(tuning: dict) -> float:
    med = tuning["median"]
    floor = np.mean(med["exact"]) - 0.01
    ceiling = np.mean(med["mae"]) + np.std(med["mae"], ddof=1)
    ok = [
        float(t)
        for t, g in tuning["grid_mean"].items()
        if g["exact"] >= floor - 1e-4 and g["mae"] <= ceiling + 1e-4  # stored values are 4 dp
    ]
    return max(ok)


@pytest.mark.skipif(not HEAD_JSON.exists(), reason="no shipped head")
def test_shipped_cutoff_matches_its_stored_tuning():
    meta = json.loads(HEAD_JSON.read_text())
    if TEN_CUTOFF_KEY not in meta:
        pytest.skip("head predates the cut-off")
    t = ten_cutoff(meta)
    tuning = meta["ten_cutoff_tuning"]
    assert tuning["n_seeds"] == 5 and tuning["n"] == meta["n_fit"]
    assert _criterion_from_tuning(tuning) == pytest.approx(t)
    assert np.mean(tuning["chosen"]["exact"]) == pytest.approx(tuning["grid_mean"][f"{t:.2f}"]["exact"], abs=1e-4)
    # cv_grouped is seed 0 under the served cut-off
    assert meta["cv_grouped"]["exact"] == pytest.approx(tuning["chosen"]["exact"][0], abs=1e-4)
    assert meta["cv_grouped"]["mae"] == pytest.approx(tuning["chosen"]["mae"][0], abs=1e-4)


@pytest.mark.skipif(not OOF.exists(), reason="no saved OOF predictions (gitignored)")
def test_saved_oof_predictions_reproduce_the_shipped_cutoff():
    z = np.load(OOF, allow_pickle=False)
    y = z["y"].astype(int)
    cums = [z[f"cum{s}"] for s in range(5)]
    t, tuning = train_probe.tune_ten_cutoff(y, cums)
    assert t == 0.58  # the 2026-10-31 decision-rule head's OOF predictions
    if HEAD_JSON.exists():
        meta = json.loads(HEAD_JSON.read_text())
        # Compare only with a head tuned on these same predictions: a later retrain
        # (e.g. tighter-slab-crops, t 0.56) has other OOF predictions and its own t.
        same_oof = meta.get("ten_cutoff_tuning", {}).get("median") == tuning["median"]
        if same_oof and meta["n_fit"] == len(y):
            assert meta[TEN_CUTOFF_KEY] == t
            assert meta["ten_cutoff_tuning"]["grid_mean"] == tuning["grid_mean"]
    if DECISION_RULE_HEAD_JSON.exists():
        # The head these predictions came from must still match them exactly.
        meta = json.loads(DECISION_RULE_HEAD_JSON.read_text())
        assert meta["ten_cutoff_tuning"]["median"] == tuning["median"]
        assert meta["n_fit"] == len(y)
        assert meta[TEN_CUTOFF_KEY] == t
        assert meta["ten_cutoff_tuning"]["grid_mean"] == tuning["grid_mean"]


# --------------------------------------------------------------------------- #
# Criterion details the targeted tests leave open (mutants that survived them)
# --------------------------------------------------------------------------- #
def _confident(grade: int, p10: float | None = None) -> np.ndarray:
    cum = np.where(np.arange(1, 10) < grade, 0.999, 0.001)
    if p10 is not None:
        cum[:8], cum[8] = 0.999, p10
    return cum


def test_exact_floor_is_one_pp_not_more():
    # 1,000 certs: 15 10s with P(10) = 0.53 (lost by any t >= 0.54: -1.5 pp exact),
    # 985 confident 10s. The MAE ceiling is wide (seed spread), so only the floor decides.
    y = np.array([10] * 1000 + [8] * 200)
    cums = []
    for s in range(5):
        cum = np.array([_confident(10, 0.53)] * 15 + [_confident(10, 0.95)] * 985 + [_confident(8)] * 200)
        cum[1000 : 1000 + 30 * s, 7] = 0.6  # some 8s called 9, varying by seed -> big MAE SD
        cums.append(cum)
    t, tuning = train_probe.tune_ten_cutoff(y, cums)
    assert tuning["median"]["mae_sd"] > 0.02
    assert t == 0.52  # 0.54 costs 1.25 pp (15 / 1200) > 1 pp


def test_mae_sd_uses_ddof_1():
    # Seed 0's median calls 22 of 50 8s a 9 (MAE + 22/1050), the other seeds none, so the
    # median's MAE SD is 0.0094 with ddof 1 but 0.0084 with ddof 0. Losing nine 10s with
    # P(10) = 0.53 costs 9/1050 = 0.0086 MAE (and 0.86 pp exact, inside the floor): only
    # the ddof-1 ceiling lets t go above 0.52.
    n, lost, bumped = 1000, 9, 22
    y = np.array([10] * n + [8] * 50)
    cums = []
    for s in range(5):
        cum = np.array([_confident(10, 0.53)] * lost + [_confident(10, 0.95)] * (n - lost) + [_confident(8)] * 50)
        if s == 0:
            cum[n : n + bumped, 7] = 0.6
        cums.append(cum)
    mae0 = np.array([bumped / len(y)] + [0.0] * 4)
    cost = lost / len(y)
    assert mae0.std(ddof=0) < cost < mae0.std(ddof=1) and cost < 0.01
    t, tuning = train_probe.tune_ten_cutoff(y, cums)
    assert tuning["median"]["mae_sd"] == round(float(mae0.std(ddof=1)), 4)
    assert t == 0.70


def test_main_reports_seed_0_as_cv_grouped(tmp_path, monkeypatch):
    # The existing fixture returns the same OOF for every seed; here seeds differ.
    from types import SimpleNamespace

    from backend.app.ml.features import REGION_NAMES, feature_spec_hash

    rng = np.random.default_rng(0)
    y_fit = np.array([9] * 20 + [10] * 80)
    rows = [
        {"cert_number": f"c{i}", "grade_int": int(g), "split": "train", "brand_title": "s",
         "subject": f"card{i // 2}", "card_number": "1"}
        for i, g in enumerate(y_fit)
    ]
    feats = rng.normal(size=(len(y_fit), 2, len(REGION_NAMES), 8)).astype(np.float32)
    feats += (y_fit - 9.5)[:, None, None, None]
    models = tmp_path / "models"
    models.mkdir()
    (models / "card_features.onnx").write_bytes(b"onnx")
    (models / "card_features.json").write_text(json.dumps({"feature_spec_hash": feature_spec_hash()}))

    def cross_validate(f, yy, groups, n_folds=5, seed=0):
        # seed s calls the first 4*s 10s a 9 at every cut-off (P(>8) below 0.5)
        cum = np.array([_confident(int(g)) for g in yy])
        cum[20 : 20 + 4 * seed, 7:] = 0.001
        return cum, np.zeros((len(yy), 4))

    settings = SimpleNamespace(db_path=tmp_path / "t.sqlite", data_dir=tmp_path, models_dir=models)
    monkeypatch.setattr(train_probe, "get_settings", lambda: settings)
    monkeypatch.setattr(train_probe, "load_rows", lambda _db: rows)
    monkeypatch.setattr(train_probe, "extract_features", lambda *a, **k: feats)
    monkeypatch.setattr(train_probe, "cross_validate", cross_validate)
    monkeypatch.setattr(train_probe, "REPORT_PATH", tmp_path / "probe_report.json")
    train_probe.main(force_onnx=False, cv_folds=5)
    meta = json.loads((models / "grade_head.json").read_text())
    assert meta["cv_grouped"]["exact"] == 1.0  # seed 0 is perfect; seeds 1-4 are not
    assert meta["ten_cutoff_tuning"]["median"]["exact"][0] == 1.0
