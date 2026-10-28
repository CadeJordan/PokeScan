"""Re-test of 419a2d4: rot180 feature cache, guard training leakage, evaluate.

tmp_path and synthetic data only; nothing under data/ is read or written.
The defects found here (once strict xfails) are fixed; the tests now pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.training import train_face_guard as tfg

torch = pytest.importorskip("torch")


def _rows(tmp_path, n=3):
    from ml.tests.test_probe_cache import _write_crop

    rows = []
    for i in range(n):
        front, back = f"crops/images/c{i}_front.jpg", f"crops/images/c{i}_back.jpg"
        _write_crop(tmp_path / front, 2 * i)
        _write_crop(tmp_path / back, 2 * i + 1)
        rows.append({"cert_number": f"c{i}", "cropped_front_path": front, "cropped_back_path": back})
    return rows


def _rot(rows, tmp_path, cache, factory, **kw):
    from ml.training import train_probe

    return train_probe.extract_features(
        rows, tmp_path, cache, model_factory=factory, transform=tfg.rot180, variant="rot180", **kw
    )


# --------------------------------------------------------------------------- #
# rot180 cache: invalidated like the main cache, never confused with it
# --------------------------------------------------------------------------- #
def test_rot180_cache_reextracts_a_rewritten_crop_only(tmp_path):
    from ml.tests.test_probe_cache import Factory, _bump_mtime, _write_crop

    rows = _rows(tmp_path)
    cache = tmp_path / "cache" / "region_feats_rot180.npz"
    first = _rot(rows, tmp_path, cache, Factory())
    _write_crop(tmp_path / rows[1]["cropped_back_path"], 99)
    _bump_mtime(tmp_path / rows[1]["cropped_back_path"])
    again = Factory()
    second = _rot(rows, tmp_path, cache, again)
    assert again.images == 2  # exactly cert c1, both faces
    np.testing.assert_array_equal(second[[0, 2]], first[[0, 2]])
    assert not np.array_equal(second[1], first[1])


def test_rot180_cache_is_discarded_on_a_feature_spec_bump(tmp_path, monkeypatch):
    from backend.app.ml import features
    from ml.tests.test_probe_cache import Factory

    rows = _rows(tmp_path)
    cache = tmp_path / "cache" / "region_feats_rot180.npz"
    _rot(rows, tmp_path, cache, Factory())
    monkeypatch.setitem(features.FEATURE_SPEC, "version", features.FEATURE_SPEC["version"] + 1)
    again = Factory()
    _rot(rows, tmp_path, cache, again)
    assert again.images == 6


def test_train_probe_never_reuses_a_rot180_cache(tmp_path):
    """The main (variant-less) reader discards a rot180 file, so pointing
    train_probe at the wrong file can't feed it rotated features."""
    from ml.tests.test_probe_cache import Factory
    from ml.training import train_probe

    rows = _rows(tmp_path)
    cache = tmp_path / "cache" / "region_feats_rot180.npz"
    rot = _rot(rows, tmp_path, cache, Factory())
    plain = Factory()
    up = train_probe.extract_features(rows, tmp_path, cache, model_factory=plain, save=False)
    assert plain.images == 6 and not np.allclose(up, rot)


def test_save_false_on_a_missing_main_cache_writes_nothing(tmp_path):
    from ml.tests.test_probe_cache import Factory
    from ml.training import train_probe

    rows = _rows(tmp_path)
    cache = tmp_path / "cache" / "region_feats.npz"
    train_probe.extract_features(rows, tmp_path, cache, model_factory=Factory(), save=False)
    assert not cache.exists() and not cache.parent.exists()


# --------------------------------------------------------------------------- #
# Training: test rows never reach the CV folds or the fitted probes
# --------------------------------------------------------------------------- #
def _synthetic(n, seed):
    from ml.tests.test_face_guard_training import _synthetic as synth

    return synth(n, seed)


def test_test_rows_never_reach_cv_or_the_final_fit(monkeypatch):
    up, rot, groups = _synthetic(200, 7)
    fit = np.arange(200) < 150
    seen_x = []
    real_fit = tfg.fit_logistic

    def spy(x, y, c=tfg.C):
        seen_x.append(np.asarray(x))
        return real_fit(x, y, c)

    monkeypatch.setattr(tfg, "fit_logistic", spy)
    # Mark every test face with a value no fit face has.
    up = up.copy()
    rot = rot.copy()
    up[~fit] = 1e6
    rot[~fit] = 1e6
    tfg.train_guard(up, rot, groups, fit, ~fit)
    assert seen_x and all((x < 1e5).all() for x in seen_x)


def test_cv_folds_are_card_grouped_across_certs_of_one_card(monkeypatch):
    """Several certs share a card: every face of every cert of a card is in one fold."""
    from sklearn.model_selection import StratifiedGroupKFold

    up, rot, _ = _synthetic(120, 8)
    groups = np.array([f"card{i // 3}" for i in range(120)])  # 3 certs per card
    seen = []
    real_split = StratifiedGroupKFold.split

    def split(self, x, y, g):
        for tr, te in real_split(self, x, y, g):
            seen.append(set(g[tr]) & set(g[te]))
            yield tr, te

    monkeypatch.setattr(StratifiedGroupKFold, "split", split)
    tfg.train_guard(up, rot, groups, np.ones(120, bool), np.zeros(120, bool))
    assert len(seen) == 3 * 5 and not any(seen)


def test_pair_metrics_index_the_right_faces():
    # side X is [up fronts, rot fronts, up backs, rot backs]; pairs use p[:n] and p[2n:3n].
    n = 4
    p = np.concatenate([np.full(n, 0.01), np.full(n, 0.5), np.full(n, 0.99), np.full(n, 0.5)])
    m = tfg.pair_side_metrics(p[:n], p[2 * n : 3 * n], 0.9)
    assert m["false_swap_rate"] == 0.0 and m["swapped_pairs_caught"] == 1.0
    assert m["clean_same_side_note_rate"] == 0.0


# --------------------------------------------------------------------------- #
# Evaluate
# --------------------------------------------------------------------------- #
def test_image_perturbation_reencodes_both_faces_and_pair_one_does_not(tmp_path):
    import cv2

    import ml.eval.evaluate as ev

    rng = np.random.default_rng(0)
    ok, f = cv2.imencode(".jpg", rng.integers(0, 256, (40, 28, 3), dtype=np.uint8))
    ok2, b = cv2.imencode(".jpg", rng.integers(0, 256, (40, 28, 3), dtype=np.uint8))
    front, back = f.tobytes(), b.tobytes()
    pf, pb = ev.apply_perturbation("rotate_180", front, back)
    assert cv2.imdecode(np.frombuffer(pf, np.uint8), cv2.IMREAD_COLOR).shape == (40, 28, 3)
    assert pf != front and pb != back
    assert ev.apply_perturbation("swap_faces", front, back) == (back, front)


@pytest.mark.parametrize("text", [" , ,", ",", "   ", " ,	, "])
def test_parse_perturbations_with_no_name_is_refused_like_an_unknown_name(text):
    """Only commas/whitespace used to parse to [] and run every perturbation."""
    import ml.eval.evaluate as ev

    with pytest.raises(ValueError, match=r"unknown perturbation\(s\) .*; known: blur"):
        ev.parse_perturbations(text)


def test_cli_turns_a_perturbation_list_with_no_name_into_a_usage_error(monkeypatch):
    from typer.testing import CliRunner

    import ml.eval.evaluate as ev

    monkeypatch.setattr(ev, "run_evaluation", lambda **kw: pytest.fail("must not evaluate"))
    result = CliRunner().invoke(ev.app, ["--perturbations", " , ,"])
    assert result.exit_code == 2


def _patched_eval(tmp_path, monkeypatch, found):
    """run_evaluation over 3 fake certs; `found[i]` is cert i's front quad flag."""
    import backend.app.ml.inference as inference
    import ml.eval.evaluate as ev
    from ml.tests.test_eval_metrics import _FakePrediction, _FakeService, _jpeg_samples

    samples = _jpeg_samples(tmp_path, n=len(found))
    by_front = {s.front.read_bytes(): (i, s) for i, s in enumerate(samples)}

    class Svc(_FakeService):
        def predict(self, front, back):
            i, _ = by_front.get(front, by_front.get(back, (0, None)))
            p = _FakePrediction(10 - i, found[i])
            p.notes = ["note"] if i == 1 else []
            return p

    monkeypatch.setattr(inference, "GradeInferenceService", Svc)
    monkeypatch.setattr(ev, "load_samples", lambda split, limit=None: samples)
    return ev


def test_report_per_cert_fields_and_notes(tmp_path, monkeypatch):
    ev = _patched_eval(tmp_path, monkeypatch, [True, True, True])
    r = ev.run_evaluation(progress=False, n_boot=0, perturbations=["swap_faces"])
    preds = r["predictions"]
    assert [p["cert"] for p in preds] == ["0", "1", "2"]
    assert [p["pred_int"] for p in preds] == [10, 9, 8]
    assert "notes" not in preds[0] and preds[1]["notes"] == "note"
    rob = r["robustness"]["swap_faces"]
    assert rob["pred_int"] == [p["pred_int"] for p in preds] and rob["n_with_notes"] == 1


def test_robustness_per_cert_lists_line_up_with_predictions_under_exclude_fallback(
    tmp_path, monkeypatch
):
    ev = _patched_eval(tmp_path, monkeypatch, [False, True, True])
    r = ev.run_evaluation(progress=False, n_boot=0, perturbations=["swap_faces"], exclude_fallback=True)
    certs = [p["cert"] for p in r["predictions"]]
    rob = r["robustness"]["swap_faces"]
    assert len(rob["pred_int"]) == len(certs)
    assert certs == ["1", "2"] and rob["cert"] == certs and rob["n"] == 2
    assert rob["pred_int"] == [p["pred_int"] for p in r["predictions"]]


def test_robustness_cert_list_names_each_entry_without_exclude_fallback(tmp_path, monkeypatch):
    ev = _patched_eval(tmp_path, monkeypatch, [False, True, True])
    r = ev.run_evaluation(progress=False, n_boot=0, perturbations=["swap_faces"], robustness_limit=2)
    rob = r["robustness"]["swap_faces"]
    assert rob["cert"] == ["0", "1"] == [p["cert"] for p in r["predictions"][:2]]
    assert rob["n"] == len(rob["pred_int"]) == len(rob["pred_soft"]) == 2


def test_robustness_refuses_clean_rows_of_other_certs(tmp_path, monkeypatch):
    ev = _patched_eval(tmp_path, monkeypatch, [True, True, True])
    samples = ev.load_samples("test")
    import backend.app.ml.inference as inference

    svc = inference.GradeInferenceService()
    clean = ev.predict_all(svc, samples[1:], progress=False)
    with pytest.raises(ValueError, match="not the same certs"):
        ev.robustness(svc, samples[:2], clean, ["swap_faces"])
