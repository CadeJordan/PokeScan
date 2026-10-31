"""Retest of the full-resolution pilot (`ml.eval.resolution_pilot`): gaps left by
`test_resolution_pilot.py`.

- the bootstrap resamples whole card groups and gives every arm the same resample;
- AUC 10 vs 9 is scored on the 9s and 10s only, with 10 as the positive class;
- the decision rule at its boundaries;
- `probe.cross_validate` uses the same card-grouped folds whatever the features;
- `evaluate` end to end on a synthetic DB + caches in tmp_path: test-split certs
  never enter it, the arms share certs / labels / groups / folds, and they differ
  only in the corner + edge regions.

Synthetic data in tmp_path only; heavy parts (feature extraction, the head's CV)
are stubbed.
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.app.ml.features import CORNER_SLICE, EDGE_SLICE, REGION_NAMES
from backend.app.ml.grade_head import PROBE_NAMES, decode
from ml.eval import resolution_pilot as rp


# --------------------------------------------------------------------------- #
# Bootstrap: whole card groups, paired across arms
# --------------------------------------------------------------------------- #
def _arm(y, rng, noise):
    soft = y + rng.normal(0, noise, len(y))
    hard = np.clip(np.round(soft), 1, 10).astype(int)
    return {"soft": soft, "hard": hard, "corners": soft + 0.1, "edges": soft - 0.1}


def test_bootstrap_resamples_whole_groups_and_pairs_the_arms(monkeypatch):
    rng = np.random.default_rng(0)
    groups = np.repeat([f"g{i}" for i in range(40)], rng.integers(1, 4, 40))
    y = rng.choice([9, 10], len(groups))
    preds = {a: _arm(y, rng, n) for a, n in (("A", 1.0), ("B", 0.5), ("C", 0.2))}
    seen: list[tuple[int, np.ndarray]] = []
    real = rp.arm_metrics

    def spy(yy, p, idx=None):
        if idx is not None:
            seen.append((id(p), np.sort(idx)))
        return real(yy, p, idx)

    monkeypatch.setattr(rp, "arm_metrics", spy)
    n_boot = 30
    rp.paired_group_bootstrap(y, groups, preds, [("B", "A"), ("C", "A"), ("C", "B")], n_boot=n_boot)
    assert len(seen) == 3 * n_boot  # each arm scored once per resample
    for b in range(n_boot):
        block = seen[3 * b : 3 * b + 3]
        assert {pid for pid, _ in block} == {id(p) for p in preds.values()}
        for _, idx in block[1:]:
            np.testing.assert_array_equal(idx, block[0][1])  # paired: the same rows for every arm
        idx = block[0][1]
        # Whole groups: each drawn group's members appear the same number of times.
        for g in np.unique(groups[idx]):
            members = np.flatnonzero(groups == g)
            counts = [int((idx == m).sum()) for m in members]
            assert len(set(counts)) == 1 and counts[0] >= 1


def test_bootstrap_matches_an_independent_reimplementation():
    rng = np.random.default_rng(1)
    groups = np.repeat([f"g{i}" for i in range(60)], 2)
    y = rng.choice([9, 10], len(groups))
    preds = {"A": _arm(y, rng, 1.0), "B": _arm(y, rng, 0.4)}
    out = rp.paired_group_bootstrap(y, groups, preds, [("B", "A")], n_boot=200, seed=7)

    _, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]
    r = np.random.default_rng(7)
    diffs = []
    for _ in range(200):
        idx = np.concatenate([members[g] for g in r.integers(0, len(members), len(members))])
        top = y[idx] >= 9
        pos = y[idx][top] == 10
        diffs.append(rp.auc(pos, preds["B"]["soft"][idx][top]) - rp.auc(pos, preds["A"]["soft"][idx][top]))
    res = out["pairs"]["B-A"]["auc_10_vs_9"]
    np.testing.assert_allclose(res["ci95"], np.percentile(diffs, [2.5, 97.5]))
    assert res["p_le_0"] == pytest.approx(np.mean(np.array(diffs) <= 0))
    full = rp.auc(y == 10, preds["B"]["soft"]) - rp.auc(y == 10, preds["A"]["soft"])
    assert res["diff"] == pytest.approx(full)


# --------------------------------------------------------------------------- #
# Metrics: subset and direction
# --------------------------------------------------------------------------- #
def test_auc_10_vs_9_uses_only_9s_and_10s_with_10_positive():
    y = np.array([10, 10, 10, 9, 9, 9, 5, 3])
    soft = np.array([9.9, 9.8, 9.7, 9.1, 9.2, 9.3, 9.95, 9.99])  # the low grades would break a ranking
    p = {"soft": soft, "hard": np.array([10, 10, 10, 9, 9, 9, 10, 10]), "corners": soft, "edges": -soft}
    m = rp.arm_metrics(y, p)
    assert m["auc_10_vs_9"] == 1.0
    assert m["auc_10_vs_9_corners"] == 1.0
    assert m["auc_10_vs_9_edges"] == 0.0  # a reversed score is 0, not folded to 1
    assert m["recall_9"] == 1.0
    assert m["mae"] == pytest.approx((5 + 7) / 8)


def test_arm_predictions_take_the_corners_and_edges_probe_columns():
    rng = np.random.default_rng(0)
    cum = np.sort(rng.random((5, 9)), axis=1)[:, ::-1]
    regions = rng.random((5, len(PROBE_NAMES))) + 8
    p = rp.arm_predictions(cum, regions)
    np.testing.assert_array_equal(p["corners"], regions[:, PROBE_NAMES.index("corners")])
    np.testing.assert_array_equal(p["edges"], regions[:, PROBE_NAMES.index("edges")])
    np.testing.assert_allclose(p["soft"], [decode(c)[0] for c in cum])
    np.testing.assert_array_equal(p["hard"], [decode(c)[1] for c in cum])


# --------------------------------------------------------------------------- #
# Decision rule boundaries (spec: Steps 4 + "Evaluation, fixed in advance")
# --------------------------------------------------------------------------- #
def _boot(ca, ca_lo, ba):
    return {
        "pairs": {
            "C-A": {"auc_10_vs_9": {"diff": ca, "ci95": [ca_lo, ca + 0.05], "p_le_0": 0.0}},
            "B-A": {"auc_10_vs_9": {"diff": ba, "ci95": [ba - 0.03, ba + 0.03], "p_le_0": 0.5}},
        }
    }


@pytest.mark.parametrize(
    ("ca", "ca_lo", "ba", "call"),
    [
        (0.03, 0.001, 0.0, "go_download"),  # exactly +0.03 meets the bar
        (0.0299, 0.001, 0.0, "stop"),
        (0.04, 0.0, 0.0, "stop"),  # a CI touching 0 doesn't exclude it
        (0.04, 0.001, 0.02, "go_no_download"),  # B - A exactly half of C - A
        (0.04, 0.001, 0.0199, "go_download"),
        (0.017, -0.010, 0.027, "stop"),  # the pilot's numbers: B - A is irrelevant without the go
        (-0.05, -0.08, 0.0, "stop"),
    ],
)
def test_decide_boundaries(ca, ca_lo, ba, call):
    assert rp.decide(_boot(ca, ca_lo, ba))["call"] == call


# --------------------------------------------------------------------------- #
# The head's CV: the folds don't depend on the features
# --------------------------------------------------------------------------- #
def test_cross_validate_folds_depend_only_on_labels_and_groups(monkeypatch):
    from ml.training import probe

    rng = np.random.default_rng(0)
    groups = np.repeat([f"g{i}" for i in range(30)], 2)
    y = np.repeat(rng.choice([9, 10], 30), 2)
    calls: list[np.ndarray] = []

    class Head:
        def region_grades(self, feats):
            return np.full((len(feats), len(PROBE_NAMES)), 9.5)

        meta = SimpleNamespace(cum_probs=lambda r: np.full((len(r), 9), 0.5))

    def fake_fit(feats, yy, gg, metadata=None):
        calls.append(np.sort(feats[:, 0, 0, 0].astype(int)))  # slot 0 holds the row index
        return Head()

    monkeypatch.setattr(probe, "fit_grade_head", fake_fit)
    idx = np.arange(len(y), dtype=float)[:, None, None, None]
    feats_a = np.broadcast_to(idx, (len(y), 2, 11, 3)).copy()
    feats_b = feats_a.copy()
    feats_b[:, :, 2:10] = rng.normal(size=feats_b[:, :, 2:10].shape)  # other corner / edge features
    probe.cross_validate(feats_a, y, groups, n_folds=5, seed=0)
    first = list(calls)
    calls.clear()
    probe.cross_validate(feats_b, y, groups, n_folds=5, seed=0)
    assert len(first) == len(calls) == 5
    for a, b in zip(first, calls, strict=True):
        np.testing.assert_array_equal(a, b)
        test = np.setdiff1d(np.arange(len(y)), a)
        assert not set(groups[a]) & set(groups[test])  # card-grouped


# --------------------------------------------------------------------------- #
# evaluate, end to end on synthetic data
# --------------------------------------------------------------------------- #
D = 4
FACES = ("front", "back")


def _feat(cert: str, salt: int = 0) -> np.ndarray:
    return np.random.default_rng(int(cert) * 7 + salt).normal(size=(2, len(REGION_NAMES), D)).astype(np.float16)


def _build_world(tmp_path: Path, stale_c_cert: bool = False):
    """A DB with train / val / test certs, the pilot folder's caches, and dummy photos."""
    data = tmp_path / "data"
    for d in ("images", "crops", "models"):
        (data / d).mkdir(parents=True)
    db = data / "pokescan.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE certs (cert_number TEXT PRIMARY KEY, grade_int INTEGER, split TEXT, "
        "brand_title TEXT, subject TEXT, card_number TEXT, front_path TEXT, back_path TEXT, "
        "cropped_front_path TEXT, cropped_back_path TEXT, is_pokemon INTEGER, has_images INTEGER, "
        "crop_found_front INTEGER, crop_found_back INTEGER)"
    )
    rows = []
    n = 0
    # Pilot candidates: 9s and 10s in train / val, a few groups shared.
    for grade in (9, 10):
        for i in range(14):
            n += 1
            rows.append((f"{1000 + n}", grade, "train" if i % 4 else "val", f"P{n % 11}"))
    # Extra train+val certs (B only), incl. low grades.
    for i in range(16):
        n += 1
        rows.append((f"{1000 + n}", (8, 8, 9, 10)[i % 4], "train", f"X{i % 9}"))
    # Test-split certs: must never be evaluated.
    test_certs = []
    for i in range(6):
        n += 1
        rows.append((f"{1000 + n}", (9, 10)[i % 2], "test", f"T{i}"))
        test_certs.append(f"{1000 + n}")
    quad = [[10.0, 10.0], [90.0, 10.0], [90.0, 130.0], [10.0, 130.0]]
    quad_cache = {}
    for cert, grade, split, group in rows:
        paths = {f: f"images/{cert}_{f}.jpg" for f in FACES}
        for f in FACES:
            (data / paths[f]).write_bytes(b"x" + cert.encode())
            quad_cache[paths[f]] = {"sig": "", "found": True, "rotation": 0, "src_wh": [100, 140], "quad": quad, "precrop_mad": 1.0}
        conn.execute(
            "INSERT INTO certs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (cert, grade, split, "SET", group, "1", paths["front"], paths["back"],
             f"crops/{paths['front']}", f"crops/{paths['back']}", 1, 1, 1, 1),
        )
    conn.commit()
    conn.close()

    pdir = data / "images_full"
    (pdir / "cache").mkdir(parents=True)
    pilot = [r for r in rows if r[2] in ("train", "val") and r[0] <= f"{1000 + 28}"]
    # A test cert slipped into the selection is still never evaluated.
    with open(pdir / "pilot_certs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(rp.SELECTION_FIELDS)
        for cert, grade, split, group in [*pilot, rows[-1]]:
            w.writerow([cert, grade, split, f"SET|{group}|1"])
    rp.write_json(pdir / "cache" / "quads.json", quad_cache)
    spec = rp.TileSpec(224)
    all_certs = [r[0] for r in rows]
    tv_certs = [r[0] for r in rows if r[2] != "test"]
    pilot_certs = [r[0] for r in pilot]
    dropped_c = pilot_certs[3]  # no C tiles: must leave every pilot arm
    bad_original = pilot_certs[5]  # original fails the check (stale C entry if `stale_c_cert`)
    c_certs = [c for c in pilot_certs if c != dropped_c and (stale_c_cert or c != bad_original)]
    # Downloaded originals + manifest; tile keys and checks carry the files' real signatures.
    entries = {}
    for c in pilot_certs:
        for f in FACES:
            (pdir / rp.original_name(c, f)).write_bytes(b"o" + c.encode())
            entries[(c, f)] = {"cert": c, "face": f, "url": f"https://cdn.example/{c}_{f}.jpg",
                               "file": rp.original_name(c, f), "bytes": 1 + len(c), "width": 1430,
                               "height": 2400, "status": "ok"}
    rp.write_manifest(pdir / "manifest.csv", entries)
    checks = {f"{c}_{f}": {"problem": None, "mad": 2.0, "sig": rp._file_sig(pdir / rp.original_name(c, f))}
              for c in pilot_certs for f in FACES}
    checks[f"{bad_original}_front"] = {**checks[f"{bad_original}_front"], "problem": "original_mismatch", "mad": 30.0}
    rp.write_json(pdir / "cache" / "original_checks.json", checks)

    def b_key(c):
        photos = [data / f"images/{c}_{f}.jpg" for f in FACES]
        return (c, *[p.relative_to(data).as_posix() for p in photos], *[rp._file_sig(p) for p in photos])

    def c_key(c):
        origs = [pdir / rp.original_name(c, f) for f in FACES]
        return (c, *[o.name for o in origs], *[rp._file_sig(o) if o.exists() else "0:0" for o in origs])

    def tiles(certs, salt, key):
        keys = np.array([key(c) for c in certs])
        feats = np.stack([_feat(c, salt)[:, CORNER_SLICE.start : EDGE_SLICE.stop] for c in certs])
        np.savez(pdir / "cache" / f"tiles_{'b' if salt == 1 else 'c'}_224.npz",
                 tile_spec_hash=np.array(spec.hash()), keys=keys, feats=feats)

    tiles(all_certs, 1, b_key)  # B: includes the test certs, which evaluate must ignore
    tiles([*c_certs, *test_certs], 2, c_key)
    settings = SimpleNamespace(data_dir=data, db_path=db, models_dir=data / "models")
    return SimpleNamespace(
        data=data, pdir=pdir, settings=settings, test_certs=set(test_certs), tv_certs=tv_certs,
        pilot_certs=pilot_certs, dropped_c=dropped_c, bad_original=bad_original,
    )


def _run_evaluate(tmp_path, monkeypatch, world):
    from ml.training import probe, train_probe

    seen_rows: list[str] = []

    def fake_extract(rows, data_dir, cache, save=True, **_):
        assert save is False  # arm A's cache is read only
        seen_rows.extend(r["cert_number"] for r in rows)
        return np.stack([_feat(r["cert_number"]) for r in rows]).astype(np.float32)

    cv_calls: list[dict] = []

    def fake_cv(feats, y, groups, n_folds=5, seed=0):
        cv_calls.append({"feats": feats.copy(), "y": y.copy(), "groups": groups.copy(),
                         "folds": probe._folds(y, groups, n_folds, seed)})
        s = 1 / (1 + np.exp(-feats[:, 0, 2, 0].astype(float)))
        cum = np.ones((len(y), 9))
        cum[:, 8] = s
        regions = np.stack([9 + s, 9 + 1 / (1 + np.exp(-feats[:, 0, 6, 0])), np.full(len(y), 9.5), np.full(len(y), 9.5)], 1)
        return cum, regions

    monkeypatch.setattr(rp, "get_settings", lambda: world.settings)
    monkeypatch.setattr(train_probe, "extract_features", fake_extract)
    monkeypatch.setattr(probe, "cross_validate", fake_cv)
    out = tmp_path / "report" / "report.json"
    rp.evaluate(tile=224, pilot_dir=world.pdir, folds=5, n_boot=40, out=out)
    return json.loads(out.read_text()), cv_calls, seen_rows


def test_evaluate_end_to_end_same_certs_and_folds_no_test_split(tmp_path, monkeypatch):
    world = _build_world(tmp_path)
    report, cv_calls, seen_rows = _run_evaluate(tmp_path, monkeypatch, world)

    assert not set(seen_rows) & world.test_certs  # arm A never extracted for a test cert
    pilot_eval = [c for c in world.pilot_certs if c not in (world.dropped_c, world.bad_original)]
    assert report["pilot"]["n"] == len(pilot_eval)
    assert report["trainval"]["n"] == len(world.tv_certs)
    assert len(cv_calls) == 5  # pilot A, B, C; train+val A, B
    for block in (cv_calls[:3], cv_calls[3:]):
        ref = block[0]
        for call in block[1:]:
            np.testing.assert_array_equal(call["y"], ref["y"])
            np.testing.assert_array_equal(call["groups"], ref["groups"])
            for (tr1, te1), (tr2, te2) in zip(call["folds"], ref["folds"], strict=True):
                np.testing.assert_array_equal(tr1, tr2)
                np.testing.assert_array_equal(te1, te2)
            # Only the corner and edge regions differ from arm A; CLS, mean, centre are A's.
            np.testing.assert_array_equal(call["feats"][:, :, :2], ref["feats"][:, :, :2])
            np.testing.assert_array_equal(call["feats"][:, :, 10:], ref["feats"][:, :, 10:])
            assert not np.array_equal(call["feats"][:, :, 2:10], ref["feats"][:, :, 2:10])
    # Arm A is the cached features; B / C the tiles of the same certs, in the same order.
    a, b, c = (cv_calls[i]["feats"] for i in range(3))
    certs = [cc for cc in world.tv_certs if cc in pilot_eval]
    np.testing.assert_array_equal(a, np.stack([_feat(x) for x in certs]).astype(np.float32))
    np.testing.assert_array_equal(b[:, :, 2:10], np.stack([_feat(x, 1)[:, 2:10] for x in certs]).astype(np.float32))
    np.testing.assert_array_equal(c[:, :, 2:10], np.stack([_feat(x, 2)[:, 2:10] for x in certs]).astype(np.float32))
    assert report["decision"]["call"] == rp.decide(report["pilot"]["bootstrap"])["call"]
    assert set(report["pilot"]["bootstrap"]["pairs"]) == {"B-A", "C-A", "C-B"}
    assert set(report["trainval"]["bootstrap"]["pairs"]) == {"B-A"}
    # Nothing written outside the pilot folder and the report.
    assert sorted(p.name for p in (world.data / "models").rglob("*")) == []


def test_evaluate_drops_a_failed_original_even_with_a_current_c_tile(tmp_path, monkeypatch):
    world = _build_world(tmp_path, stale_c_cert=True)
    report, _, _ = _run_evaluate(tmp_path, monkeypatch, world)
    assert world.bad_original in report["checks"]["pilot_dropped"]
    assert report["pilot"]["n"] == len(world.pilot_certs) - 2


def test_evaluate_drops_stale_tiles_and_quad_problems_from_every_arm(tmp_path, monkeypatch):
    world = _build_world(tmp_path)
    p = world.pilot_certs
    replaced_photo, bad_quad, replaced_orig = p[0], p[1], p[2]
    # A stored photo replaced after B was extracted: its B tile key is stale.
    (world.data / f"images/{replaced_photo}_front.jpg").write_bytes(b"replaced photo")
    # A quad that no longer passes the precrop check, though both tiles are cached.
    quads = rp.read_json(world.pdir / "cache" / "quads.json")
    quads[f"images/{bad_quad}_back.jpg"]["precrop_mad"] = 30.0
    rp.write_json(world.pdir / "cache" / "quads.json", quads)
    # An original replaced after C was extracted and checked: check and C key are stale.
    (world.pdir / rp.original_name(replaced_orig, "front")).write_bytes(b"replaced original")
    report, cv_calls, _ = _run_evaluate(tmp_path, monkeypatch, world)

    dropped = report["checks"]["pilot_dropped"]
    assert dropped[replaced_photo] == "b:no_tiles"
    assert dropped[bad_quad] == "b:back:precrop_mismatch"
    assert dropped[replaced_orig] == "front:original_check_stale"
    assert dropped[world.dropped_c] == "c:no_tiles"
    assert dropped[world.bad_original] == "front:original_mismatch"
    assert report["checks"]["stale_tiles_skipped"]["b"] == 1
    gone = {replaced_photo, bad_quad, replaced_orig, world.dropped_c, world.bad_original}
    kept = [c for c in world.tv_certs if c in world.pilot_certs and c not in gone]
    assert report["pilot"]["n"] == len(kept) == len(world.pilot_certs) - 5
    a = np.stack([_feat(c).astype(np.float32) for c in kept])
    for call in cv_calls[:3]:  # A, B and C all on the same remaining certs
        np.testing.assert_array_equal(call["feats"][:, :, :2], a[:, :, :2])
    # Train+val loses the two certs without current B tiles.
    assert report["trainval"]["n"] == len(world.tv_certs) - 2


# --------------------------------------------------------------------------- #
# The published report (read only; skipped when it isn't there)
# --------------------------------------------------------------------------- #
REPORT = rp.REPORT_PATH


@pytest.mark.skipif(not REPORT.exists(), reason="no pilot report")
def test_published_report_is_consistent_with_the_rule():
    r = json.loads(REPORT.read_text())
    assert r["decision"]["call"] == rp.decide(r["pilot"]["bootstrap"])["call"]
    pilot = r["pilot"]
    assert pilot["n"] == pilot["n9"] + pilot["n10"]
    for arm in pilot["arms"].values():
        assert arm["n"] == pilot["n"]
    pairs = pilot["bootstrap"]["pairs"]
    for name, (x, z) in {"B-A": ("B", "A"), "C-A": ("C", "A"), "C-B": ("C", "B")}.items():
        for m in rp.METRICS:
            want = pilot["arms"][x][m] - pilot["arms"][z][m]
            assert pairs[name][m]["diff"] == pytest.approx(want, abs=1e-12)
            lo, hi = pairs[name][m]["ci95"]
            assert lo <= hi


# --------------------------------------------------------------------------- #
# Output-path guard
# --------------------------------------------------------------------------- #
def test_check_output_dir_refuses_the_db_file(tmp_path):
    data = tmp_path / "data"
    with pytest.raises(ValueError):
        rp.check_output_dir(data / "pokescan.sqlite", data)


# --------------------------------------------------------------------------- #
# Re-test of the fixes (2026-10-31): gaps
# --------------------------------------------------------------------------- #
def test_evaluate_drops_a_stale_c_tile_whose_new_original_passed_the_check(tmp_path, monkeypatch):
    world = _build_world(tmp_path)
    cert = world.pilot_certs[8]
    # The original was re-downloaded and re-checked (ok), but C was not re-extracted.
    o = world.pdir / rp.original_name(cert, "front")
    o.write_bytes(b"re-downloaded original")
    checks = rp.read_json(world.pdir / "cache" / "original_checks.json")
    checks[f"{cert}_front"]["sig"] = rp._file_sig(o)
    rp.write_json(world.pdir / "cache" / "original_checks.json", checks)
    report, cv_calls, _ = _run_evaluate(tmp_path, monkeypatch, world)
    assert report["checks"]["pilot_dropped"][cert] == "c:no_tiles"
    assert report["checks"]["stale_tiles_skipped"] == {"b": 0, "c": 1}
    assert report["pilot"]["n"] == len(world.pilot_certs) - 3  # + dropped_c, bad_original
    assert report["trainval"]["n"] == len(world.tv_certs)  # B is unaffected
    for call in cv_calls[:3]:
        assert len(call["y"]) == report["pilot"]["n"]


def test_evaluate_drops_a_missing_check_a_failed_download_and_a_cert_that_left_trainval(tmp_path, monkeypatch):
    world = _build_world(tmp_path)
    no_check, failed_dl, left = world.pilot_certs[9], world.pilot_certs[10], world.pilot_certs[11]
    checks = rp.read_json(world.pdir / "cache" / "original_checks.json")
    del checks[f"{no_check}_back"]
    rp.write_json(world.pdir / "cache" / "original_checks.json", checks)
    entries = rp.read_manifest(world.pdir / "manifest.csv")
    entries[(failed_dl, "front")]["status"] = "failed"
    rp.write_manifest(world.pdir / "manifest.csv", entries)
    with sqlite3.connect(world.settings.db_path) as conn:
        conn.execute("UPDATE certs SET split='test' WHERE cert_number=?", (left,))
    report, cv_calls, seen_rows = _run_evaluate(tmp_path, monkeypatch, world)
    dropped = report["checks"]["pilot_dropped"]
    assert dropped[no_check] == "back:original_check_missing"
    assert dropped[failed_dl] == "c:front:download_failed"
    assert dropped[left] == "not_trainval"
    assert left not in seen_rows  # arm A never extracted for it
    assert report["pilot"]["n"] == len(world.pilot_certs) - 5
    assert report["trainval"]["n"] == len(world.tv_certs) - 1


@pytest.mark.skipif(os.name != "nt", reason="Windows path forms")
@pytest.mark.parametrize("form", ["verbatim", "unc_admin_share", "trailing_space_dir"])
def test_check_output_dir_refuses_other_spellings_of_the_data_folder(tmp_path, form):
    data = tmp_path / "data"
    (data / "models").mkdir(parents=True)
    (data / "pokescan.sqlite").write_bytes(b"db")
    r = str(data.resolve()).replace(os.sep, "/")
    path = {
        "verbatim": "//?/" + r + "/pokescan.sqlite",  # the \\?\ prefix
        "unc_admin_share": "//localhost/" + r[0] + "$" + r[2:] + "/pokescan.sqlite",
        # resolves to the verbatim form of data/models/x.json: a write into a protected folder
        "trailing_space_dir": r + "/models /x.json",
    }[form]
    with pytest.raises(ValueError):
        rp.check_output_dir(Path(path), data)


def _sensitive_requests(client_kwargs, handler):
    import httpx

    sent: list[list[str]] = []

    def record(request):
        sent.append(sorted(k.lower() for k in request.headers if k.lower() in rp.SENSITIVE_HEADERS))
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(record), follow_redirects=True, **client_kwargs)
    with client as c, contextlib.suppress(RuntimeError):  # raising after the fact is too late
        rp.fetch(c, "https://cdn.example/1_f.jpg", sleep=lambda s: None)
    return [s for s in sent if s]


def test_fetch_header_audit_is_case_insensitive():
    import httpx

    ok = lambda r: httpx.Response(200, content=b"x")  # noqa: E731
    for name in ("AUTHORIZATION", "authorization", "X-Api-KEY", "COOKIE", "Proxy-Authorization"):
        assert _sensitive_requests({"headers": {name: "secret"}}, ok) == []


@pytest.mark.parametrize("case", ["client_auth", "request_hook", "cookie_set_on_redirect"])
def test_fetch_never_sends_a_forbidden_header_added_at_send_time(case):
    import httpx

    def redirect(request):
        if request.url.path == "/1_f.jpg":
            return httpx.Response(302, headers={"Location": "/moved.jpg", "Set-Cookie": "s=1; Path=/"})
        return httpx.Response(200, content=b"x")

    kwargs, handler = {
        "client_auth": ({"auth": ("user", "secret")}, lambda r: httpx.Response(200, content=b"x")),
        "request_hook": (
            {"event_hooks": {"request": [lambda r: r.headers.__setitem__("Authorization", "bearer t")]}},
            lambda r: httpx.Response(200, content=b"x"),
        ),
        "cookie_set_on_redirect": ({}, redirect),
    }[case]
    assert _sensitive_requests(kwargs, handler) == []
