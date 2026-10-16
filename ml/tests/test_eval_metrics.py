"""Training metrics (ml/training/metrics.py) and eval metrics (ml/eval/evaluate.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.eval.evaluate import (
    baselines,
    build_report,
    compare_to_baseline,
    confusion,
    coverage_curve,
    expected_calibration_error,
    quadratic_weighted_kappa,
    summarize,
)
from ml.training.metrics import confusion_matrix, exact_match, mean_absolute_error, off_by_one

TRUE = [10, 9, 8, 10, 7]
PRED = [10, 10, 6, 9, 7]


# --------------------------------------------------------------------------- #
# ml/training/metrics.py
# --------------------------------------------------------------------------- #
def test_training_metrics_hand_computed():
    t, p = torch.tensor(TRUE), torch.tensor(PRED)
    assert mean_absolute_error(p, t).item() == pytest.approx(4 / 5)
    assert exact_match(p, t).item() == pytest.approx(2 / 5)
    assert off_by_one(p, t).item() == pytest.approx(4 / 5)


def test_training_confusion_matrix():
    cm = confusion_matrix(torch.tensor(PRED), torch.tensor(TRUE))
    assert cm.shape == (10, 10)
    assert cm.sum().item() == 5
    assert cm[9, 9] == 1 and cm[9, 8] == 1 and cm[8, 9] == 1 and cm[7, 5] == 1 and cm[6, 6] == 1


def test_confusion_matrix_clamps_out_of_range():
    cm = confusion_matrix(torch.tensor([0, 11]), torch.tensor([1, 10]))
    assert cm[0, 0] == 1 and cm[9, 9] == 1


def test_numpy_confusion_matches_torch():
    a = confusion(np.array(TRUE), np.array(PRED))
    b = confusion_matrix(torch.tensor(PRED), torch.tensor(TRUE)).numpy()
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- #
# Quadratic weighted kappa
# --------------------------------------------------------------------------- #
def test_qwk_perfect():
    assert quadratic_weighted_kappa(np.array(TRUE), np.array(TRUE)) == pytest.approx(1.0)


def test_qwk_matches_sklearn():
    from sklearn.metrics import cohen_kappa_score

    rng = np.random.default_rng(0)
    t = rng.integers(1, 11, 300)
    p = np.clip(t + rng.integers(-2, 3, 300), 1, 10)
    expected = cohen_kappa_score(t, p, weights="quadratic", labels=list(range(1, 11)))
    assert quadratic_weighted_kappa(t, p) == pytest.approx(expected, abs=1e-9)


def test_qwk_penalises_far_errors_more():
    t = np.array([1, 3, 5, 7, 9, 10] * 10)
    near = np.clip(t + 1, 1, 10)
    far = np.clip(11 - t, 1, 10)
    assert quadratic_weighted_kappa(t, near) > quadratic_weighted_kappa(t, far)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def test_ece_zero_when_calibrated():
    conf = np.array([0.8] * 10)
    correct = np.array([1] * 8 + [0] * 2)
    ece, bins = expected_calibration_error(conf, correct)
    assert ece == pytest.approx(0.0)
    assert sum(b["n"] for b in bins) == 10


def test_ece_overconfident():
    ece, _ = expected_calibration_error(np.array([0.95] * 4), np.array([1, 0, 0, 0]))
    assert ece == pytest.approx(0.70)


def test_ece_includes_zero_confidence():
    _, bins = expected_calibration_error(np.array([0.0, 1.0]), np.array([0, 1]))
    assert sum(b["n"] for b in bins) == 2


def test_coverage_curve_prefers_confident():
    conf = np.array([0.9, 0.8, 0.3, 0.2])
    correct = np.array([1, 1, 0, 0])
    curve = {c["coverage"]: c["accuracy"] for c in coverage_curve(conf, correct)}
    assert curve[0.5] == 1.0 and curve[1.0] == 0.5


# --------------------------------------------------------------------------- #
# Report assembly + regression gate
# --------------------------------------------------------------------------- #
def _rows(n=40, seed=0):
    rng = np.random.default_rng(seed)
    grade = rng.integers(6, 11, n)
    pred = np.clip(grade + rng.integers(-1, 2, n), 1, 10)
    return {
        "grade": grade,
        "pred_int": pred,
        "pred_soft": pred + rng.uniform(-0.4, 0.4, n),
        "confidence": rng.uniform(0.2, 0.9, n),
        "quad_front": rng.random(n) > 0.1,
        "quad_back": rng.random(n) > 0.1,
        "latency": rng.uniform(0.05, 0.2, n),
    }


def test_build_report_shape():
    r = build_report(_rows(), train_grades=np.array([10, 10, 9, 8]))
    assert r["overall"]["n"] == 40
    assert r["overall"]["off_by_one"] == 1.0
    assert set(r["per_grade"]) <= {str(g) for g in range(1, 11)}
    assert np.asarray(r["confusion_matrix"]).sum() == 40
    assert 0.0 <= r["calibration"]["ece"] <= 1.0
    assert r["baselines"]["majority_grade"]["grade"] == 10
    assert r["latency_ms"]["p95"] >= r["latency_ms"]["p50"]


def test_write_charts(tmp_path):
    from ml.eval.evaluate import write_charts

    paths = write_charts(build_report(_rows()), tmp_path)
    assert all(p.exists() and p.stat().st_size > 1000 for p in paths)


def test_summarize_bias_sign():
    s = summarize(np.array([8, 8]), np.array([9, 9]), np.array([9.0, 9.0]))
    assert s["bias"] == pytest.approx(1.0)


def test_baselines_majority():
    b = baselines(np.array([9, 10]), np.array([10, 10, 10, 9]))
    assert b["majority_grade"]["grade"] == 10
    assert b["majority_grade"]["exact"] == 0.5


def test_compare_to_baseline():
    report = {"overall": {"mae": 0.80, "off_by_one": 0.85, "qwk": 0.60}, "calibration": {"ece": 0.05}}
    base = {"mae": 0.65, "off_by_one": 0.86, "qwk": 0.61, "ece": 0.04}
    problems = compare_to_baseline(report, base)
    assert len(problems) == 1 and problems[0].startswith("mae")


# --------------------------------------------------------------------------- #
# Confidence intervals
# --------------------------------------------------------------------------- #
def test_wilson_interval_known_values():
    from ml.eval.evaluate import wilson_interval

    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    lo, hi = wilson_interval(0, 4)
    assert lo == 0.0 and 0.4 < hi < 0.55          # not degenerate at p=0
    lo, hi = wilson_interval(3, 3)
    assert hi == 1.0 and lo < 0.5
    assert all(np.isnan(v) for v in wilson_interval(0, 0))


def test_bootstrap_ci_contains_point_and_narrows_with_n():
    from ml.eval.evaluate import bootstrap_ci

    rng = np.random.default_rng(0)
    small = rng.integers(0, 3, 30).astype(float)
    big = rng.integers(0, 3, 3000).astype(float)
    lo_s, hi_s = bootstrap_ci(lambda i: small[i].mean(), len(small), 500)
    lo_b, hi_b = bootstrap_ci(lambda i: big[i].mean(), len(big), 500)
    assert lo_s <= small.mean() <= hi_s
    assert lo_b <= big.mean() <= hi_b
    assert (hi_b - lo_b) < (hi_s - lo_s)
    # Deterministic for a fixed seed.
    assert bootstrap_ci(lambda i: small[i].mean(), 30, 200) == bootstrap_ci(lambda i: small[i].mean(), 30, 200)


def test_overall_ci_brackets_point_estimates():
    from ml.eval.evaluate import overall_ci

    r = _rows(200)
    ci = overall_ci(r["grade"], r["pred_int"], r["pred_soft"], n_boot=300)
    s = summarize(r["grade"], r["pred_int"], r["pred_soft"])
    for key in ("mae", "qwk", "exact", "bias"):
        lo, hi = ci[key]
        assert lo <= s[key] <= hi, key
    assert overall_ci(np.array([]), np.array([]), np.array([])) == {}


def test_per_grade_has_cis():
    from ml.eval.evaluate import per_grade

    r = _rows(100)
    pg = per_grade(r["grade"], r["pred_int"], r["pred_soft"], n_boot=200)
    for row in pg.values():
        lo, hi = row["recall_ci"]
        assert 0.0 <= lo <= row["recall"] <= hi <= 1.0
        mlo, mhi = row["mae_ci"]
        assert mlo <= row["mae"] + 1e-9 and row["mae"] <= mhi + 1e-9


def test_build_report_has_overall_ci():
    r = build_report(_rows(), n_boot=100)
    assert set(r["overall_ci"]) >= {"mae", "qwk", "exact", "bias"}
    assert build_report(_rows(), n_boot=0)["overall_ci"] == {}


# --------------------------------------------------------------------------- #
# Manifest (raw-card eval set)
# --------------------------------------------------------------------------- #
def test_load_manifest(tmp_path):
    from ml.eval.evaluate import load_manifest

    (tmp_path / "imgs").mkdir()
    for name in ("a_f.jpg", "a_b.jpg", "b_f.jpg", "b_b.jpg"):
        (tmp_path / "imgs" / name).write_bytes(b"x")
    abs_front = (tmp_path / "imgs" / "b_f.jpg").resolve()
    csv_path = tmp_path / "raw.csv"
    csv_path.write_text(
        "Front,Back,Grade,id\n"
        "imgs/a_f.jpg,imgs/a_b.jpg,8,card-a\n"
        f"{abs_front},imgs/b_b.jpg,10.0,\n"
        "imgs/missing.jpg,imgs/a_b.jpg,5,gone\n",
        encoding="utf-8",
    )
    samples = load_manifest(csv_path)
    assert [s.grade for s in samples] == [8, 10] or [s.grade for s in samples] == [10, 8]
    by_id = {s.cert: s for s in samples}
    assert by_id["card-a"].front == tmp_path / "imgs" / "a_f.jpg"
    assert any(c.startswith("raw:") for c in by_id)       # auto id when blank
    assert len(load_manifest(csv_path, limit=1)) == 1


def test_load_manifest_validates(tmp_path):
    from ml.eval.evaluate import load_manifest

    p = tmp_path / "bad.csv"
    p.write_text("front,grade\nx.jpg,9\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_manifest(p)
    (tmp_path / "f.jpg").write_bytes(b"x")
    p.write_text("front,back,grade\nf.jpg,f.jpg,11\n")
    with pytest.raises(ValueError, match="outside"):
        load_manifest(p)
