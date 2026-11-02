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


def test_load_manifest_keeps_extra_columns_as_tags(tmp_path):
    from ml.eval.evaluate import load_manifest

    (tmp_path / "f.jpg").write_bytes(b"x")
    p = tmp_path / "phone.csv"
    p.write_text(
        "front,back,grade,id,Shot,cert,in_db_split\n"
        "f.jpg,f.jpg,9,1_careful,careful,1,test\n"
        "f.jpg,f.jpg,9,1_casual, casual ,1,\n",
        encoding="utf-8",
    )
    samples = {s.cert: s for s in load_manifest(p)}
    assert samples["1_careful"].tags == {"shot": "careful", "cert": "1", "in_db_split": "test"}
    assert samples["1_casual"].tags == {"shot": "casual", "cert": "1", "in_db_split": ""}
    # No extra columns: no tags.
    p.write_text("front,back,grade,id\nf.jpg,f.jpg,9,a\n", encoding="utf-8")
    assert load_manifest(p)[0].tags == {}


# --------------------------------------------------------------------------- #
# Manifest tags: by_tag breakdown and card-level (grouped) bootstrap
# --------------------------------------------------------------------------- #
def _tagged_rows():
    rows = _rows(n=6)
    rows["grade"] = np.array([10, 10, 9, 9, 8, 8])
    rows["pred_int"] = np.array([10, 9, 9, 9, 6, 8])
    rows["reject"] = np.array(["", "card_not_found", "", "same_side", "card_not_found", ""])
    rows["tag:shot"] = np.array(["careful", "casual"] * 3)
    rows["tag:cert"] = np.array(["a", "a", "b", "b", "c", "c"])
    rows["tag:in_db_split"] = np.array(["absent", "absent", "train", "train", "test", "test"])
    return rows


def test_by_tag_breakdown():
    pl = build_report(_tagged_rows(), n_boot=0)["pipeline"]
    by_tag = pl["by_tag"]
    assert list(by_tag) == ["shot", "in_db_split"]  # the cert column only groups
    assert by_tag["shot"]["careful"] == {
        "n": 3, "mae": pytest.approx(2 / 3), "exact": pytest.approx(2 / 3),
        "n_would_reject": {"card_not_found": 1, "same_side": 0},
    }
    assert by_tag["shot"]["casual"] == {
        "n": 3, "mae": pytest.approx(1 / 3), "exact": pytest.approx(2 / 3),
        "n_would_reject": {"card_not_found": 1, "same_side": 1},
    }
    assert list(by_tag["in_db_split"]) == ["absent", "test", "train"]
    assert by_tag["in_db_split"]["test"]["mae"] == 1.0
    assert by_tag["in_db_split"]["train"]["exact"] == 1.0


def test_db_split_report_has_no_by_tag_and_row_cis():
    """A report without tags (every DB split) is what it was before tags existed."""
    from ml.eval.evaluate import overall_ci, per_grade

    rows = _rows()
    r = build_report(rows, n_boot=50)
    assert "by_tag" not in r["pipeline"]
    assert set(r["pipeline"]) == {
        "quad_found_front", "quad_found_back", "mae_both_found", "mae_fallback", "n_fallback",
        "crop_kinds",
    }
    assert "n_groups" not in r["overall_ci"]
    assert r["overall_ci"] == overall_ci(rows["grade"], rows["pred_int"], rows["pred_soft"], 50)
    assert r["per_grade"] == per_grade(rows["grade"], rows["pred_int"], rows["pred_soft"], 50)


def test_grouped_bootstrap_resamples_cards():
    """Every card shot twice, identically: card-level CIs equal those of one row per
    card (same draws), while row-level CIs on the doubled rows come out narrower."""
    from ml.eval.evaluate import bootstrap_ci, overall_ci

    one = _rows(n=30, seed=3)
    t, p, s = one["grade"], one["pred_int"], one["pred_soft"]
    cards = np.arange(30)
    t2, p2, s2, g2 = (np.repeat(x, 2) for x in (t, p, s, cards))
    grouped = overall_ci(t2, p2, s2, n_boot=300, groups=g2)
    plain = overall_ci(t, p, s, n_boot=300)
    assert grouped["n_groups"] == 30
    for key in ("mae", "exact", "qwk", "bias"):
        assert grouped[key] == pytest.approx(plain[key]), key
    rows_ci = overall_ci(t2, p2, s2, n_boot=300)
    assert rows_ci["mae"][1] - rows_ci["mae"][0] < plain["mae"][1] - plain["mae"][0]

    err = np.abs(p - t).astype(float)
    err2 = np.repeat(err, 2)
    assert bootstrap_ci(lambda i: err2[i].mean(), 60, 200, groups=g2) == pytest.approx(
        bootstrap_ci(lambda i: err[i].mean(), 30, 200)
    )
    # Group labels needn't be sorted or contiguous: rows of a card stay together.
    seen = []
    bootstrap_ci(lambda i: seen.append(np.asarray(i)) or 0.0, 4, 20,
                 groups=np.array(["y", "x", "y", "x"]))
    for i in seen:
        assert np.isin([0, 2], i).all() or not np.isin([0, 2], i).any()
        assert (i == 0).sum() == (i == 2).sum() and (i == 1).sum() == (i == 3).sum()


def test_report_with_cert_tag_uses_card_level_cis():
    rows = _tagged_rows()
    r = build_report(rows, n_boot=100)
    assert r["overall_ci"]["n_groups"] == 3
    rows.pop("tag:cert")
    assert "n_groups" not in build_report(rows, n_boot=100)["overall_ci"]


# --------------------------------------------------------------------------- #
# Crop kinds + excluding fallback (label-visible) cards
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("front", "back", "expected"),
    [("card", "card", "card"), ("card", "well", "well"), ("well", "small", "small"),
     ("none", "card", "none"), ("card", None, "unknown"), ("unknown", "card", "unknown")],
)
def test_combine_crop_kinds(front, back, expected):
    from ml.eval.evaluate import combine_crop_kinds

    assert combine_crop_kinds(front, back) == expected


def test_crop_kind_breakdown_in_report():
    rows = _rows(n=6)
    rows["grade"] = np.array([10, 10, 9, 9, 8, 8])
    rows["pred_int"] = np.array([10, 9, 9, 9, 6, 8])
    rows["crop_kind"] = np.array(["card", "card", "well", "well", "small", "card"])
    kinds = build_report(rows, n_boot=0)["pipeline"]["crop_kinds"]
    assert list(kinds) == ["small", "well", "card"]  # worst first
    assert kinds["card"] == {"n": 3, "mae": pytest.approx(1 / 3)}
    assert kinds["small"]["mae"] == 2.0


class _FakePrediction:
    def __init__(self, grade, found):
        self.grade = float(grade)
        self.grade_int = grade
        self.confidence = 0.5
        self.found_quad_front = found
        self.found_quad_back = True


class _FakeService:
    """Predicts the true grade only when the card is NOT found (a 'label leak')."""

    ready = True
    tta = ()
    model_path = "fake.onnx"
    metadata: dict = {}

    class _session:  # noqa: N801
        @staticmethod
        def get_providers():
            return ["CPUExecutionProvider"]

    def __init__(self, *_a, **_k):
        pass

    def predict(self, front: bytes, back: bytes):
        grade, found = int(front[:2]), front[2:] == b"found"
        return _FakePrediction(grade if not found else 1, found)


def test_exclude_fallback_drops_leaky_cards(tmp_path, monkeypatch):
    import backend.app.ml.inference as inference
    import ml.eval.evaluate as ev

    samples = []
    for i, (grade, found) in enumerate([(10, False), (9, False), (10, True), (8, True)]):
        f = tmp_path / f"{i}.jpg"
        f.write_bytes(f"{grade:02d}".encode() + (b"found" if found else b"fallback"))
        samples.append(ev.Sample(str(i), grade, f, f, "card" if found else "none"))
    monkeypatch.setattr(inference, "GradeInferenceService", _FakeService)
    monkeypatch.setattr(ev, "load_samples", lambda split, limit=None: samples)

    leaky = ev.run_evaluation(robustness_limit=0, progress=False, n_boot=0)
    clean = ev.run_evaluation(robustness_limit=0, progress=False, n_boot=0, exclude_fallback=True)

    assert leaky["overall"]["n"] == 4 and leaky["overall"]["exact"] == 0.5
    assert clean["overall"]["n"] == 2 and clean["overall"]["exact"] == 0.0
    assert clean["pipeline"]["excluded_from_metrics"] == 2
    assert clean["pipeline"]["n_fallback"] == 2  # pipeline still describes every card
    assert clean["meta"]["exclude_fallback"] is True
    assert "by_tag" not in leaky["pipeline"] and "by_tag" not in clean["pipeline"]


def test_manifest_run_carries_tags_into_report(tmp_path, monkeypatch):
    """A phone-set manifest run end to end: tags reach rows, by_tag and card-level CIs."""
    import backend.app.ml.inference as inference
    import ml.eval.evaluate as ev

    lines = ["front,back,grade,id,shot,cert,in_db_split"]
    for cert, grade, split in (("1", 10, "absent"), ("2", 9, "train")):
        for shot, found in (("careful", False), ("casual", True)):
            f = tmp_path / f"{cert}_{shot}.jpg"
            f.write_bytes(f"{grade:02d}".encode() + (b"found" if found else b"fallback"))
            lines.append(f"{f.name},{f.name},{grade},{cert}_{shot},{shot},{cert},{split}")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(inference, "GradeInferenceService", _FakeService)
    monkeypatch.setattr(ev, "load_samples", lambda split, limit=None: [])

    service = _FakeService()
    rows = ev.predict_all(service, ev.load_manifest(manifest), progress=False)
    assert list(rows["tag:shot"]) == ["careful", "casual", "careful", "casual"]
    assert list(rows["tag:cert"]) == ["1", "1", "2", "2"]

    r = ev.run_evaluation(manifest=manifest, robustness_limit=0, progress=False, n_boot=20)
    by_shot = r["pipeline"]["by_tag"]["shot"]
    # _FakeService grades a found card 1, a fallback card its true grade.
    assert by_shot["careful"]["exact"] == 1.0 and by_shot["casual"]["exact"] == 0.0
    assert set(r["pipeline"]["by_tag"]["in_db_split"]) == {"absent", "train"}
    assert r["overall_ci"]["n_groups"] == 2


# --------------------------------------------------------------------------- #
# Perturbations: swap_faces and --perturbations
# --------------------------------------------------------------------------- #
class _ByteService(_FakeService):
    """Grade 10 when the front upload is a sample's own front file, else 5."""

    fronts: set[bytes] = set()

    def predict(self, front: bytes, back: bytes):
        return _FakePrediction(10 if front in self.fronts else 5, True)


def _jpeg_samples(tmp_path, n=3):
    import cv2

    import ml.eval.evaluate as ev

    rng = np.random.default_rng(0)
    samples = []
    for i in range(n):
        paths = []
        for side in ("front", "back"):
            path = tmp_path / f"{i}_{side}.jpg"
            assert cv2.imwrite(str(path), rng.integers(0, 256, (40, 28, 3), dtype=np.uint8))
            paths.append(path)
        samples.append(ev.Sample(str(i), 10, *paths, "card"))
    return samples


def test_swap_faces_swaps_the_upload_bytes():
    from ml.eval.evaluate import ALL_PERTURBATIONS, PERTURBATIONS, apply_perturbation

    assert apply_perturbation("swap_faces", b"front", b"back") == (b"back", b"front")
    assert "swap_faces" in ALL_PERTURBATIONS and "swap_faces" not in PERTURBATIONS
    assert set(PERTURBATIONS) < set(ALL_PERTURBATIONS)


def test_robustness_runs_swap_faces(tmp_path):
    import ml.eval.evaluate as ev

    samples = _jpeg_samples(tmp_path)
    service = _ByteService()
    service.fronts = {s.front.read_bytes() for s in samples}
    clean = ev.predict_all(service, samples, progress=False)
    assert list(clean["cert"]) == ["0", "1", "2"] and (clean["pred_int"] == 10).all()
    out = ev.robustness(service, samples, clean, ["swap_faces"])
    assert list(out) == ["swap_faces"]
    r = out["swap_faces"]
    assert r["n"] == 3 and r["grade_flip_rate"] == 1.0 and r["mean_abs_soft_shift"] == 5.0
    assert r["pred_int"] == [5, 5, 5]


@pytest.mark.parametrize(
    "text, expected",
    [
        (None, None),
        ("", None),
        ("rotate_180,swap_faces", ["rotate_180", "swap_faces"]),
        (" swap_faces , blur,swap_faces", ["swap_faces", "blur"]),
    ],
)
def test_parse_perturbations(text, expected):
    from ml.eval.evaluate import parse_perturbations

    assert parse_perturbations(text) == expected


def test_parse_perturbations_rejects_unknown_names():
    from ml.eval.evaluate import parse_perturbations

    with pytest.raises(ValueError, match=r"unknown perturbation\(s\) rotate_90, nope; known: blur"):
        parse_perturbations("rotate_180,rotate_90,nope")


def test_run_evaluation_default_runs_every_perturbation(tmp_path, monkeypatch):
    import backend.app.ml.inference as inference
    import ml.eval.evaluate as ev

    samples = _jpeg_samples(tmp_path, n=2)
    monkeypatch.setattr(_ByteService, "fronts", {s.front.read_bytes() for s in samples})
    monkeypatch.setattr(inference, "GradeInferenceService", _ByteService)
    monkeypatch.setattr(ev, "load_samples", lambda split, limit=None: samples)
    every = ev.run_evaluation(progress=False, n_boot=0)
    assert list(every["robustness"]) == list(ev.ALL_PERTURBATIONS)
    picked = ev.run_evaluation(progress=False, n_boot=0, perturbations=["swap_faces"])
    assert list(picked["robustness"]) == ["swap_faces"]
    assert [p["cert"] for p in picked["predictions"]] == ["0", "1"]
    with pytest.raises(ValueError, match="unknown perturbation"):
        ev.run_evaluation(progress=False, n_boot=0, perturbations=["sideways"])


def test_cli_perturbations_option(monkeypatch):
    from typer.testing import CliRunner

    import ml.eval.evaluate as ev

    seen = {}

    class StopError(Exception):
        pass

    def fake_run(**kwargs):
        seen.update(kwargs)
        raise StopError

    monkeypatch.setattr(ev, "run_evaluation", fake_run)
    runner = CliRunner()
    bad = runner.invoke(ev.app, ["--perturbations", "rotate_180,flip"])
    assert bad.exit_code == 2 and not seen
    assert "unknown perturbation(s) flip" in bad.output
    runner.invoke(ev.app, ["--perturbations", "rotate_180,swap_faces"])
    assert seen["perturbations"] == ["rotate_180", "swap_faces"]
    seen.clear()
    runner.invoke(ev.app, [])
    assert seen["perturbations"] is None  # all, as before


# --------------------------------------------------------------------------- #
# Face guard probabilities per cert
# --------------------------------------------------------------------------- #
class _GuardService(_FakeService):
    """Sample 0 gets a guard decision that rotated its front; sample 1 has no guard."""

    def predict(self, front: bytes, back: bytes):
        from backend.app.ml.face_guard import GuardDecision

        p = _FakePrediction(10, True)
        p.notes = []
        p.guard_decision = None
        if front.startswith(b"0"):
            p.notes = ["front photo was upside down; rotated it"]
            p.guard_decision = GuardDecision(False, (True, False), p.notes, (0.01, 0.98), (0.995, 0.2))
        return p


def test_report_and_robustness_carry_guard_probabilities(tmp_path):
    import ml.eval.evaluate as ev

    samples = []
    for i in range(2):
        f = tmp_path / f"{i}.jpg"
        f.write_bytes(f"{i}".encode())
        samples.append(ev.Sample(str(i), 10, f, f, "card"))
    service = _GuardService()
    rows = ev.predict_all(service, samples, progress=False)
    assert rows["p_flip"].shape == (2, 2) and np.isnan(rows["p_flip"][1]).all()
    report = ev.build_report(rows, n_boot=0)
    first, second = report["predictions"]
    assert first["p_back"] == [0.01, 0.98] and first["p_flip"] == [0.995, 0.2]
    assert first["notes"] == "front photo was upside down; rotated it"
    assert "p_back" not in second and "p_flip" not in second
    # Each sample's front and back are one file, so swap_faces keeps the bytes.
    out = ev.robustness(service, samples, rows, ["swap_faces"])["swap_faces"]
    assert out["guard"] == [{"p_back": [0.01, 0.98], "p_flip": [0.995, 0.2]}, {}]
    assert out["notes"] == ["front photo was upside down; rotated it", ""]
