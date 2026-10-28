"""train_probe's feature cache and artifact order, with a tiny stand-in for DINOv2.

The cache must re-extract a cert whose crop file changed, discard everything
when the feature spec changes, and keep only the current rows. `main` must
export the feature graph before it writes the head.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend.app.ml import features
from backend.app.ml.features import FEAT_H, FEAT_W, REGION_NAMES, feature_spec_hash
from ml.training import train_probe

torch = pytest.importorskip("torch")

D = 4


class Tiny(torch.nn.Module):
    """(B, 3, FEAT_H, FEAT_W) -> (B, n_regions, D), a fixed function of the pixels."""

    def __init__(self) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.proj = torch.nn.Parameter(torch.randn(3 * 4, len(REGION_NAMES) * D, generator=g))
        self.images = 0
        self.seen: tuple[int, ...] | None = None

    def forward(self, x):
        self.images += x.shape[0]
        self.seen = tuple(x.shape[1:])
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, (2, 2)).flatten(1)
        return (pooled @ self.proj).reshape(x.shape[0], len(REGION_NAMES), D)


class Factory:
    """model_factory that counts how often it's called and how many images it saw."""

    def __init__(self) -> None:
        self.calls = 0
        self.models: list[Tiny] = []

    def __call__(self) -> Tiny:
        self.calls += 1
        self.models.append(Tiny().eval())
        return self.models[-1]

    @property
    def images(self) -> int:
        return sum(m.images for m in self.models)


def _write_crop(path, seed):
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = rng.integers(0, 256, (84, 60, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), cv2.resize(img, (600, 840), interpolation=cv2.INTER_NEAREST))


def _bump_mtime(path):
    # Make the rewrite visible whatever the filesystem's timestamp granularity.
    ns = path.stat().st_mtime_ns + 2_000_000_000
    os.utime(path, ns=(ns, ns))


@pytest.fixture
def crops(tmp_path):
    """(data_dir, rows) for 4 certs with front/back JPEG crops under tmp_path."""
    rows = []
    for i in range(4):
        cert = f"c{i}"
        front, back = f"crops/images/{cert}_front.jpg", f"crops/images/{cert}_back.jpg"
        _write_crop(tmp_path / front, 2 * i)
        _write_crop(tmp_path / back, 2 * i + 1)
        rows.append({"cert_number": cert, "cropped_front_path": front, "cropped_back_path": back})
    return tmp_path, rows


def _extract(crops, factory, rows=None):
    data_dir, all_rows = crops
    cache = data_dir / "models" / "cache" / "region_feats.npz"
    return train_probe.extract_features(rows or all_rows, data_dir, cache, model_factory=factory), cache


def test_unchanged_files_extract_nothing(crops):
    first = Factory()
    a, _ = _extract(crops, first)
    assert first.calls == 1 and first.images == 2 * 4
    assert first.models[0].seen == (3, FEAT_H, FEAT_W)
    assert a.shape == (4, 2, len(REGION_NAMES), D) and a.dtype == np.float32
    second = Factory()
    b, _ = _extract(crops, second)
    assert second.calls == 0
    np.testing.assert_array_equal(a, b)


def test_rewritten_crop_reextracts_only_that_cert(crops):
    a, _ = _extract(crops, Factory())
    data_dir, rows = crops
    path = data_dir / rows[2]["cropped_back_path"]
    _write_crop(path, 99)
    _bump_mtime(path)
    factory = Factory()
    b, _ = _extract(crops, factory)
    assert factory.calls == 1 and factory.images == 2  # one cert, both faces
    assert not np.allclose(a[2, 1], b[2, 1])
    np.testing.assert_allclose(a[2, 0], b[2, 0], rtol=2e-3, atol=2e-3)  # front re-extracted, same
    for i in (0, 1, 3):
        np.testing.assert_array_equal(a[i], b[i])
    again = Factory()
    np.testing.assert_array_equal(_extract(crops, again)[0], b)
    assert again.calls == 0


def test_touched_crop_reextracts(crops):
    # Same bytes, new mtime: the signature changes, so the cert is re-extracted.
    _extract(crops, Factory())
    data_dir, rows = crops
    _bump_mtime(data_dir / rows[0]["cropped_front_path"])
    factory = Factory()
    _extract(crops, factory)
    assert factory.images == 2


def test_feature_spec_change_discards_cache(crops, monkeypatch, caplog):
    _, cache = _extract(crops, Factory())
    monkeypatch.setitem(features.FEATURE_SPEC, "version", features.FEATURE_SPEC["version"] + 1)
    factory = Factory()
    with caplog.at_level("INFO", logger="train_probe"):
        _extract(crops, factory)
    assert factory.images == 2 * 4
    assert any("feature_spec_hash" in r.getMessage() for r in caplog.records)
    with np.load(cache, allow_pickle=False) as z:
        assert str(z["feature_spec_hash"]) == feature_spec_hash()


def test_cache_without_spec_hash_is_discarded(crops):
    # The pre-check cache format: backbone + (cert, front, back) keys, no hash.
    data_dir, rows = crops
    cache = data_dir / "models" / "cache" / "region_feats.npz"
    cache.parent.mkdir(parents=True)
    keys = [(r["cert_number"], r["cropped_front_path"], r["cropped_back_path"]) for r in rows]
    np.savez(
        cache,
        backbone=np.array(features.BACKBONE),
        keys=np.array(keys),
        feats=np.zeros((len(rows), 2, len(REGION_NAMES), D), np.float16),
    )
    factory = Factory()
    feats, _ = _extract(crops, factory)
    assert factory.images == 2 * 4
    assert np.abs(feats).sum() > 0


def test_dropped_certs_leave_the_saved_cache(crops):
    _, cache = _extract(crops, Factory())
    _, rows = crops
    factory = Factory()
    _extract(crops, factory, rows=rows[1:3])
    assert factory.calls == 0
    with np.load(cache, allow_pickle=False) as z:
        assert sorted(k[0] for k in z["keys"]) == ["c1", "c2"]
        assert z["keys"].shape == (2, 5) and z["feats"].shape[0] == 2


def test_export_onnx_writes_feature_spec_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "build_region_model", lambda pretrained=True: Tiny().eval())
    path = tmp_path / "card_features.onnx"
    features.export_onnx(path)
    assert path.exists()
    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert sidecar["feature_spec_hash"] == feature_spec_hash()
    assert sidecar["feature_spec"] == features.FEATURE_SPEC
    assert "exported_at" in sidecar


# --------------------------------------------------------------------------- #
# train_probe.main: feature graph before head
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_training(tmp_path, monkeypatch):
    """main() over synthetic rows/features with a tmp_path models dir; returns that dir."""
    rng = np.random.default_rng(0)
    n = 240
    latent = rng.normal(size=n)
    grade = np.clip(np.round(8.5 + 1.2 * latent), 7, 10).astype(int)
    direction = rng.normal(size=(2, len(REGION_NAMES), D))
    feats = latent[:, None, None, None] * direction + rng.normal(size=(n, 2, len(REGION_NAMES), D))
    rows = [
        {
            "cert_number": f"c{i}",
            "grade_int": int(grade[i]),
            "split": "test" if i % 5 == 0 else "train",
            "brand_title": "set",
            "subject": f"card{i // 2}",
            "card_number": "1",
        }
        for i in range(n)
    ]
    models = tmp_path / "models"
    settings = SimpleNamespace(db_path=tmp_path / "t.sqlite", data_dir=tmp_path, models_dir=models)
    monkeypatch.setattr(train_probe, "get_settings", lambda: settings)
    monkeypatch.setattr(train_probe, "load_rows", lambda _db: rows)
    monkeypatch.setattr(train_probe, "extract_features", lambda *a, **k: feats.astype(np.float32))
    monkeypatch.setattr(train_probe, "REPORT_PATH", tmp_path / "probe_report.json")
    return models


def _fake_export(calls):
    def export(path, pretrained=True):
        calls.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"onnx")
        path.with_suffix(".json").write_text(json.dumps({"feature_spec_hash": feature_spec_hash()}))

    return export


def test_failed_export_writes_no_head(fake_training, monkeypatch):
    def boom(path, pretrained=True):
        raise RuntimeError("export failed")

    monkeypatch.setattr(train_probe, "export_onnx", boom)
    with pytest.raises(RuntimeError, match="export failed"):
        train_probe.main(force_onnx=False, cv_folds=0)
    assert not (fake_training / "grade_head.npz").exists()
    assert not (fake_training / "grade_head.json").exists()


def test_head_records_feature_spec_and_export_follows_the_sidecar(fake_training, monkeypatch):
    calls = []
    monkeypatch.setattr(train_probe, "export_onnx", _fake_export(calls))
    train_probe.main(force_onnx=False, cv_folds=0)
    assert len(calls) == 1  # no graph yet
    meta = json.loads((fake_training / "grade_head.json").read_text())
    assert meta["feature_spec_hash"] == feature_spec_hash()
    assert meta["feature_spec"] == features.FEATURE_SPEC
    assert {"kind", "backbone", "regions", "n_fit", "trained_at", "test"} <= set(meta)

    train_probe.main(force_onnx=False, cv_folds=0)
    assert len(calls) == 1  # sidecar matches: no re-export
    train_probe.main(force_onnx=True, cv_folds=0)
    assert len(calls) == 2
    sidecar = fake_training / "card_features.json"
    sidecar.write_text(json.dumps({"feature_spec_hash": "000000000000"}))
    train_probe.main(force_onnx=False, cv_folds=0)
    assert len(calls) == 3  # sidecar from another spec
    sidecar.unlink()
    train_probe.main(force_onnx=False, cv_folds=0)
    assert len(calls) == 4  # graph without a sidecar (exported before the check)
