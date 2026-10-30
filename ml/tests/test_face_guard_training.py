"""train_face_guard: probe fitting, thresholds and gates, and the rotated-feature cache."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.ml.face_guard import CHECKS, FaceGuard
from backend.app.ml.features import REGION_NAMES
from ml.training import train_face_guard as tfg

D = 4


def test_fit_logistic_folds_the_scaler():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    x = rng.normal(5.0, [1.0, 10.0, 0.1], size=(200, 3))
    y = (x[:, 0] + 0.1 * x[:, 1] > 5.5).astype(int)
    probe = tfg.fit_logistic(x, y)
    scaler = StandardScaler().fit(x)
    ref = LogisticRegression(C=tfg.C, max_iter=5000).fit(scaler.transform(x), y)
    np.testing.assert_allclose(probe.prob(x), ref.predict_proba(scaler.transform(x))[:, 1], atol=1e-9)


def _synthetic(n, seed, back_flip_signal=True):
    """(up, rot, groups): side in dim 0 of CLS, flip in dim 1 (not for backs if disabled)."""
    rng = np.random.default_rng(seed)
    up = rng.normal(size=(n, 2, len(REGION_NAMES), D)).astype(np.float32)
    up[:, 0, 0, 0] -= 4.0  # fronts
    up[:, 1, 0, 0] += 4.0  # backs
    rot = up + rng.normal(scale=0.1, size=up.shape).astype(np.float32)
    rot[:, 0, 0, 1] += 6.0
    if back_flip_signal:
        rot[:, 1, 0, 1] += 6.0
    groups = np.array([f"card{i // 2}" for i in range(n)])  # two certs per card
    return up, rot, groups


def test_separable_faces_enable_every_check():
    up, rot, groups = _synthetic(400, 0)
    fit = np.arange(400) < 300
    guard, report = tfg.train_guard(up, rot, groups, fit, ~fit)
    assert guard.enabled == CHECKS
    assert guard.thresholds == {n: 0.9 for n in CHECKS}  # lowest grid value that passes
    assert report["disabled_reasons"] == {}
    for n in CHECKS:
        assert report["cv_grouped"][n]["auc"] > 0.999
        assert report["test"][n]["n"] in (2 * 100, 4 * 100)
    pairs = report["cv_grouped"]["side"]["pairs"]["0.9"]
    assert pairs["false_swap_rate"] == 0.0 and pairs["swapped_pairs_caught"] > 0.95


def test_check_that_misses_its_gate_is_disabled_with_a_reason():
    up, rot, groups = _synthetic(300, 1, back_flip_signal=False)
    fit = np.arange(300) < 240
    guard, report = tfg.train_guard(up, rot, groups, fit, ~fit)
    assert guard.enabled == ("side", "front_upright")
    reason = report["disabled_reasons"]["back_upright"]
    assert "CV AUC" in reason and "< 0.99" in reason
    assert guard.metadata["disabled_reasons"] == report["disabled_reasons"]


def test_test_split_never_moves_thresholds_or_cv():
    up, rot, groups = _synthetic(300, 2)
    fit = np.arange(300) < 240
    a, ra = tfg.train_guard(up, rot, groups, fit, ~fit)
    up2, rot2 = up.copy(), rot.copy()
    up2[~fit], rot2[~fit] = rot[~fit], up[~fit]  # scramble the test faces
    b, rb = tfg.train_guard(up2, rot2, groups, fit, ~fit)
    assert a.thresholds == b.thresholds and a.enabled == b.enabled
    assert ra["cv_grouped"] == rb["cv_grouped"]
    assert ra["test"] != rb["test"]


def test_cv_folds_keep_a_cards_faces_together():
    # Both orientations and both faces of a card must share a fold, or the
    # probe could score by recognising the card.
    from sklearn.model_selection import StratifiedGroupKFold

    up, rot, groups = _synthetic(60, 3)
    d = tfg.face_sets(up, rot, groups)["side"]
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=tfg.SEED)
    assert len(d["y"]) == 4 * 60
    for tr, te in skf.split(np.zeros(len(d["y"])), d["y"], d["groups"]):
        assert not set(d["groups"][tr]) & set(d["groups"][te])


def test_choose_threshold_and_gate():
    m = {"auc": 0.9995, "error_at_0.5": 0.002, "at_threshold": {
        "0.9": {"clean_fpr": 0.004}, "0.95": {"clean_fpr": 0.002}, "0.99": {"clean_fpr": 0.0}}}
    assert tfg.choose_threshold(m) == 0.95
    assert tfg.gate("side", m, 0.95) is None
    worse = {**m, "error_at_0.5": 0.004, "at_threshold": {t: {"clean_fpr": 0.01} for t in m["at_threshold"]}}
    t = tfg.choose_threshold(worse)
    assert t is None
    reason = tfg.gate("side", worse, t)
    assert "CV error" in reason and "no threshold" in reason


def test_false_positive_on_clean_faces():
    p = np.array([0.96, 0.04, 0.5])
    y = np.array([0, 1, 0])  # a front read as a back, a back read as a front, an unsure front
    np.testing.assert_array_equal(tfg.false_positive("side", p, y, 0.95), [True, True, False])
    np.testing.assert_array_equal(tfg.false_positive("front_upright", p, y, 0.95), [True, False, False])


def test_written_guard_loads(tmp_path):
    up, rot, groups = _synthetic(200, 4)
    fit = np.arange(200) < 160
    guard, _ = tfg.train_guard(up, rot, groups, fit, ~fit)
    path = tmp_path / "face_guard.npz"
    guard.save(path)
    loaded = FaceGuard.load(path)
    assert loaded.enabled == guard.enabled and loaded.thresholds == guard.thresholds
    # Fronts upright read as fronts and upright; a rotated front reads as flipped.
    assert loaded.decide(up[0]).notes == []
    assert loaded.decide(np.stack([rot[0, 0], up[0, 1]])).flip == (True, False)


# --------------------------------------------------------------------------- #
# Rotated-feature cache: train_probe.extract_features keying, with a variant
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch")


def test_rot180_cache_is_its_own_variant(tmp_path):
    from ml.tests.test_probe_cache import Factory, _write_crop
    from ml.training import train_probe

    rows = []
    for i in range(3):
        front, back = f"crops/images/c{i}_front.jpg", f"crops/images/c{i}_back.jpg"
        _write_crop(tmp_path / front, 2 * i)
        _write_crop(tmp_path / back, 2 * i + 1)
        rows.append({"cert_number": f"c{i}", "cropped_front_path": front, "cropped_back_path": back})
    cache = tmp_path / "cache" / "region_feats.npz"
    rot_cache = tmp_path / "cache" / "region_feats_rot180.npz"

    def extract(path, factory, **kw):
        return train_probe.extract_features(rows, tmp_path, path, model_factory=factory, **kw)

    up = extract(cache, Factory())
    before = cache.read_bytes()
    first = Factory()
    rot = extract(rot_cache, first, transform=tfg.rot180, variant="rot180")
    assert first.images == 6 and not np.allclose(up, rot)
    with np.load(rot_cache, allow_pickle=False) as z:
        assert str(z["variant"]) == "rot180"
    again = Factory()
    np.testing.assert_array_equal(extract(rot_cache, again, transform=tfg.rot180, variant="rot180"), rot)
    assert again.calls == 0
    # A cache of another variant is never reused (here: the upright cache read as rot180).
    other = Factory()
    extract(cache, other, transform=tfg.rot180, variant="rot180", save=False)
    assert other.images == 6
    # save=False never writes: the upright cache is byte-identical.
    assert cache.read_bytes() == before
    with np.load(cache, allow_pickle=False) as z:
        assert "variant" not in z.files  # the main cache format is unchanged


# --------------------------------------------------------------------------- #
# Served-path features: the DB photos through /api/grade's preprocessing
# --------------------------------------------------------------------------- #
def _photo(scene_seed: int, back: bool, size=(1200, 1000), orientation: int | None = None) -> bytes:
    """A synthetic phone-style photo of a card front (or back) as JPEG bytes.

    With `orientation`, the pixels are stored turned so that the EXIF tag
    turns them upright again (only 6 is used: stored 90 degrees CCW).
    """
    import io

    from PIL import Image

    from backend.tests.helpers.synth import make_card, make_scene

    card = make_card(seed=scene_seed, border_rgb=(40, 80, 200)) if back else None
    rgb = make_scene(seed=scene_seed, size=size, card=card).bgr[:, :, ::-1]
    exif = Image.Exif()
    if orientation is not None:
        assert orientation == 6
        rgb = np.rot90(rgb, 1)
        exif[0x0112] = orientation
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="JPEG", quality=92, exif=exif)
    return buf.getvalue()


def _served_rows(data_dir, n=3, n_test=0):
    """Rows with original photos (images/) and precrops (crops/) under data_dir.

    Cert c0's photos are 1500 px tall with an EXIF orientation of 6, so the
    served path's transpose and 1024 px cap both act on them.
    """
    from ml.tests.test_probe_cache import _write_crop

    rows = []
    for i in range(n):
        row = {
            "cert_number": f"c{i}",
            "grade_int": 9,
            "split": "test" if i < n_test else "train",
            "brand_title": "set",
            "subject": f"card{i}",
            "card_number": "1",
        }
        for side in ("front", "back"):
            photo = f"images/c{i}_{side}.jpg"
            kw = {"size": (1500, 1250), "orientation": 6} if i == 0 else {}
            (data_dir / photo).parent.mkdir(parents=True, exist_ok=True)
            (data_dir / photo).write_bytes(_photo(10 + i, side == "back", **kw))
            crop = f"crops/images/c{i}_{side}.jpg"
            _write_crop(data_dir / crop, 2 * i + (side == "back"))
            row[f"{side}_path"], row[f"cropped_{side}_path"] = photo, crop
        rows.append(row)
    return rows


class _TinyExtractor:
    """The service's extractor slot, running the test Tiny model on what it is fed."""

    session = None

    def __init__(self):
        from ml.tests.test_probe_cache import Tiny

        self.model = Tiny().eval()
        self.inputs: list[tuple[np.ndarray, np.ndarray]] = []
        self.outputs: list[np.ndarray] = []

    def __call__(self, front_rgb, back_rgb):
        from backend.app.ml.features import to_input

        self.inputs.append((front_rgb, back_rgb))
        x = torch.from_numpy(np.stack([to_input(front_rgb), to_input(back_rgb)]))
        with torch.no_grad():
            out = self.model(x).numpy()
        self.outputs.append(out)
        return out


def _served_service(tmp_path):
    from backend.tests.test_inference import _probe_service, _zero_head

    feats = np.zeros((2, len(REGION_NAMES), D), np.float32)
    svc = _probe_service(tmp_path, feats, _zero_head(feats))
    svc._extractor = _TinyExtractor()
    svc._guard = None  # one extraction per request
    return svc


def test_served_features_equal_what_the_service_extracts(tmp_path):
    """Training-side served features of an upload == the service's, upright and rotate_180."""
    from backend.app.ml.inference import detect_face
    from ml.eval.evaluate import apply_perturbation
    from ml.tests.test_probe_cache import Factory

    rows = _served_rows(tmp_path)
    crop = detect_face((tmp_path / rows[0]["front_path"]).read_bytes(), "front")
    assert crop.found_quad and crop.scale < 1.0  # EXIF transpose, then the 1024 px cap

    up, rot = tfg.load_face_features("served", rows, tmp_path, tmp_path / "cache", model_factory=Factory())
    svc = _served_service(tmp_path)
    for i, r in enumerate(rows):
        front, back = ((tmp_path / r[k]).read_bytes() for k in tfg.SERVED_PATHS)
        for want, pair, load in (
            (up[i], (front, back), tfg.served_upright),
            (rot[i], apply_perturbation("rotate_180", front, back), tfg.served_rot180),
        ):
            svc.predict(*pair)
            got = svc._extractor.outputs[-1]
            # float16 cache, torch on the training device vs CPU here
            np.testing.assert_allclose(want, got, rtol=2e-3, atol=2e-3 * np.abs(got).max())
            # The images fed to the extractor are bit-identical.
            for face, key in enumerate(tfg.SERVED_PATHS):
                np.testing.assert_array_equal(load(tmp_path / r[key]), svc._extractor.inputs[-1][face])
    # The rotated set is the photo rotated before detection, not the crop rotated after.
    assert not np.allclose(up, rot)


def test_served_rotation_is_the_eval_rotate_180():
    from ml.eval.evaluate import apply_perturbation, perturb_image

    front, back = _photo(1, False), _photo(2, True)
    assert apply_perturbation("rotate_180", front, back) == (
        perturb_image("rotate_180", front),
        perturb_image("rotate_180", back),
    )


def test_served_cache_keys_on_the_original_photos(tmp_path, monkeypatch):
    from backend.app.ml import features
    from ml.tests.test_probe_cache import Factory, _bump_mtime, _write_crop
    from ml.training import train_probe

    rows = _served_rows(tmp_path, n=3)
    cache_dir = tmp_path / "cache"

    def load(factory):
        return tfg.load_face_features("served", rows, tmp_path, cache_dir, model_factory=factory)

    first = Factory()
    up, rot = load(first)
    assert first.calls == 2 and first.images == 2 * 2 * 3  # both caches, every face
    for (name, variant), want in zip(tfg.CACHES["served"], ("served", "served_rot180"), strict=True):
        assert variant == want
        with np.load(cache_dir / name, allow_pickle=False) as z:
            assert str(z["variant"]) == variant
            assert str(z["feature_spec_hash"]) == features.feature_spec_hash()
            assert [tuple(k[:3]) for k in z["keys"]] == [
                (r["cert_number"], r["front_path"], r["back_path"]) for r in rows
            ]
    again = Factory()
    a2, r2 = load(again)
    assert again.calls == 0
    np.testing.assert_array_equal(a2, up)
    np.testing.assert_array_equal(r2, rot)

    # A rewritten precrop doesn't concern the served caches.
    crop = tmp_path / rows[1]["cropped_back_path"]
    _write_crop(crop, 77)
    _bump_mtime(crop)
    untouched = Factory()
    load(untouched)
    assert untouched.calls == 0

    # A touched original photo re-extracts that cert only, in both caches.
    _bump_mtime(tmp_path / rows[1]["back_path"])
    touched = Factory()
    a3, r3 = load(touched)
    assert touched.calls == 2 and touched.images == 2 * 2
    for i in (0, 2):
        np.testing.assert_array_equal(a3[i], up[i])
        np.testing.assert_array_equal(r3[i], rot[i])
    np.testing.assert_allclose(a3[1], up[1], rtol=2e-3, atol=2e-3)  # same bytes, same features

    # A cache of the other served variant is never reused.
    other = Factory()
    up_name, rot_name = (n for n, _ in tfg.CACHES["served"])
    train_probe.extract_features(
        rows, tmp_path, cache_dir / up_name, model_factory=other, path_keys=tfg.SERVED_PATHS,
        load=tfg.served_rot180, variant="served_rot180", save=False,
    )
    assert other.images == 2 * 3

    # Another feature spec discards both.
    monkeypatch.setitem(features.FEATURE_SPEC, "version", features.FEATURE_SPEC["version"] + 1)
    bumped = Factory()
    load(bumped)
    assert bumped.images == 2 * 2 * 3


def test_main_records_the_feature_source_and_never_rewrites_the_precrop_caches(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from ml.tests.test_probe_cache import Tiny
    from ml.training import train_probe

    rows = _served_rows(tmp_path, n=8, n_test=2)
    models = tmp_path / "models"
    cache_dir = models / "cache"
    factory = lambda: Tiny().eval()  # noqa: E731
    # The precrop caches as the 2026-10-28 training left them.
    train_probe.extract_features(rows, tmp_path, cache_dir / "region_feats.npz", model_factory=factory)
    tfg.precrop_features(rows, tmp_path, cache_dir, save_rot=True, model_factory=factory)
    old = {
        name: ((cache_dir / name).read_bytes(), (cache_dir / name).stat().st_mtime_ns)
        for name in ("region_feats.npz", "region_feats_rot180.npz")
    }
    settings = SimpleNamespace(db_path=tmp_path / "t.sqlite", data_dir=tmp_path, models_dir=models)
    monkeypatch.setattr(tfg, "get_settings", lambda: settings)
    monkeypatch.setattr(tfg, "load_rows", lambda _db: rows)
    monkeypatch.setattr(tfg, "build_region_model", factory)

    def run(features, out, save=True):
        report = tmp_path / f"report_{features}_{save}.json"
        tfg.main(cv_folds=2, out=out, features=features, save=save, report=report, workers=2)
        return json.loads(report.read_text())

    guard_path = models / "face_guard.npz"
    report = run("served", guard_path)
    sidecar = json.loads(guard_path.with_suffix(".json").read_text())
    assert sidecar["feature_source"] == "served" and report["feature_source"] == "served"
    assert sidecar["feature_caches"] == ["region_feats_served.npz", "region_feats_served_rot180.npz"]
    assert "rotate_180" in sidecar["feature_source_note"]
    assert FaceGuard.load(guard_path).metadata["feature_source"] == "served"
    assert set(report["transfer_from_precrop"]) == set(CHECKS)
    assert report["saved"] is True
    assert (cache_dir / "region_feats_served.npz").exists()
    assert (cache_dir / "region_feats_served_rot180.npz").exists()

    # --no-save writes no guard; the precrop source is still available.
    skipped = models / "skipped.npz"
    assert run("served", skipped, save=False)["saved"] is False and not skipped.exists()
    precrop_path = models / "face_guard_precrop.npz"
    report = run("precrop", precrop_path)
    assert json.loads(precrop_path.with_suffix(".json").read_text())["feature_source"] == "precrop"
    assert "transfer_from_precrop" not in report

    # Neither precrop cache was ever rewritten.
    for name, (data, mtime) in old.items():
        assert (cache_dir / name).read_bytes() == data
        assert (cache_dir / name).stat().st_mtime_ns == mtime


def test_extract_features_defaults_and_workers_are_unchanged(tmp_path):
    """The default call still reads the precrops; `workers` doesn't change the result."""
    from ml.tests.test_probe_cache import Factory
    from ml.training import train_probe

    rows = _served_rows(tmp_path, n=3)
    one = train_probe.extract_features(rows, tmp_path, tmp_path / "a.npz", model_factory=Factory())
    many = train_probe.extract_features(rows, tmp_path, tmp_path / "b.npz", model_factory=Factory(), workers=3)
    np.testing.assert_array_equal(one, many)
    for name in ("a.npz", "b.npz"):
        with np.load(tmp_path / name, allow_pickle=False) as z:
            assert "variant" not in z.files
            assert [tuple(k[:3]) for k in z["keys"]] == [
                (r["cert_number"], r["cropped_front_path"], r["cropped_back_path"]) for r in rows
            ]


def test_unknown_feature_source_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown feature source 'raw'"):
        tfg.load_face_features("raw", [], tmp_path, tmp_path)


def test_transfer_metrics_scores_the_destination_out_of_fold():
    up, rot, groups = _synthetic(200, 5)
    # Destination: the same faces, but every upright front looks a little flipped.
    dst_up = up.copy()
    dst_up[:, 0, 0, 1] += 6.0
    out = tfg.transfer_metrics(up, rot, dst_up, rot, groups, cv_folds=5)
    assert set(out) == set(CHECKS)
    fu = out["front_upright"]
    assert fu["source_threshold"] == 0.9 and fu["source_cv"]["at_threshold"]["0.9"]["clean_fpr"] == 0.0
    assert fu["dest_clean_fpr_at_source_threshold"] > 0.5  # the shift shows up as false rotations
    assert out["back_upright"]["dest_clean_fpr_at_source_threshold"] == 0.0
    # Same faces, same folds: on identical features transfer equals the source's own CV.
    same = tfg.transfer_metrics(up, rot, up, rot, groups, cv_folds=5)
    for name in CHECKS:
        assert same[name]["source_cv"] == same[name]["dest_cv"]
    with pytest.raises(ValueError, match="not the same faces"):
        tfg.transfer_metrics(up, rot, up[:100], rot[:100], groups)


@pytest.mark.model
def test_served_features_match_the_real_service_on_db_photos(tmp_path):
    """On real DB photos (read only), training-side served features (torch) match
    what the running service's ONNX extractor sees, upright and rotate_180."""
    from backend.app.core.config import get_settings
    from backend.app.ml.inference import GradeInferenceService
    from ml.eval.evaluate import apply_perturbation
    from ml.training import train_probe

    s = get_settings()
    if not s.db_path.exists():
        pytest.skip("no DB")
    rows = train_probe.load_rows(s.db_path)
    rows = [r for r in rows if r["split"] == "train"][:2]
    if not rows or not all((s.data_dir / r[k]).exists() for r in rows for k in tfg.SERVED_PATHS):
        pytest.skip("original photos missing")
    svc = GradeInferenceService()
    if not svc.ready or svc.kind != "probe":
        pytest.skip(f"probe model not loaded: {svc.metadata.get('probe_load_error')}")
    try:
        up, rot = tfg.load_face_features("served", rows, s.data_dir, tmp_path / "cache")
    except Exception as exc:  # noqa: BLE001 - timm weights not cached offline
        pytest.skip(f"region model unavailable: {exc}")
    seen = []
    real = svc._extractor
    svc._extractor = lambda f, b: seen.append(real(f, b)) or seen[-1]
    svc._guard = None
    for i, r in enumerate(rows):
        front, back = ((s.data_dir / r[k]).read_bytes() for k in tfg.SERVED_PATHS)
        for want, pair in ((up[i], (front, back)), (rot[i], apply_perturbation("rotate_180", front, back))):
            svc.predict(*pair)
            got = seen[-1]
            assert np.linalg.norm(got - want) / np.linalg.norm(got) < 2e-3
