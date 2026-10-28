"""Step 2/3 training setup: sampler power, augmentation, QWK monitoring, threshold offsets."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from backend.app.ml.model import NUM_THRESHOLDS, GradeModel, decode_corn_logits
from ml.data.dataset import _build_transforms, _photometric_transforms, sampler_weights
from ml.training.calibrate import (
    _hard_grades,
    apply_calibration,
    fit_temperature_on_logits,
    fit_threshold_offsets,
    offset_metrics,
)
from ml.training.train import checkpoint_filename, monitor_settings

# Train grade distribution from the plan (10:1089 ... 1:1).
TRAIN_COUNTS = {10: 1089, 9: 515, 8: 140, 7: 40, 6: 26, 5: 20, 4: 5, 3: 6, 2: 2, 1: 1}


def _grades_from_counts(counts: dict[int, int]) -> list[int]:
    return [g for g, n in counts.items() for _ in range(n)]


# --------------------------------------------------------------------------- #
# Sampler
# --------------------------------------------------------------------------- #
def _class_mass(grades: list[int], w: torch.Tensor) -> dict[int, float]:
    g = np.asarray(grades)
    w = w.numpy()
    return {k: float(w[g == k].sum() / w.sum()) for k in sorted(set(grades))}


def test_sampler_power_zero_is_uniform():
    grades = _grades_from_counts(TRAIN_COUNTS)
    w = sampler_weights(grades, 0.0)
    assert torch.allclose(w, torch.ones_like(w))


def test_sampler_power_one_balances_classes():
    grades = _grades_from_counts(TRAIN_COUNTS)
    mass = _class_mass(grades, sampler_weights(grades, 1.0))
    assert all(m == pytest.approx(0.1) for m in mass.values())


def test_sampler_power_half_is_inverse_sqrt():
    grades = _grades_from_counts(TRAIN_COUNTS)
    w = sampler_weights(grades, 0.5)
    i10, i1 = grades.index(10), grades.index(1)
    assert (w[i1] / w[i10]).item() == pytest.approx(np.sqrt(1089))
    mass = _class_mass(grades, w)
    # Grade 10 still dominates, but far less than with no reweighting.
    assert mass[10] == max(mass.values())
    assert mass[10] < 1089 / len(grades)


def test_sampler_power_monotone_in_rare_class_mass():
    grades = _grades_from_counts(TRAIN_COUNTS)
    m = [_class_mass(grades, sampler_weights(grades, p))[1] for p in (0.0, 0.5, 1.0)]
    assert m[0] < m[1] < m[2]


def test_sampler_rejects_negative_power():
    with pytest.raises(ValueError):
        sampler_weights([1, 2], -0.1)


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
FORBIDDEN = ("Blur", "Noise", "Crop", "Dropout", "Elastic", "Distortion", "Sharpen", "Compression")


def _img(seed=0, h=168, w=120):  # 5:7 card aspect, like the precrops
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_photometric_block_has_no_condition_changing_ops():
    names = [type(t).__name__ for t in _photometric_transforms(2.0)]
    assert names and not any(bad in n for n in names for bad in FORBIDDEN)
    assert _photometric_transforms(0.0) == []


@pytest.mark.parametrize("strength", [0.0, 1.0, 2.0])
def test_train_transform_shape_dtype(strength):
    tf = _build_transforms(84, 60, train=True, aug_strength=strength)
    for seed in range(5):
        x = tf(image=_img(seed))["image"]
        assert x.shape == (3, 84, 60)
        assert x.dtype == torch.float32
        assert torch.isfinite(x).all()


def test_val_transform_is_deterministic():
    tf = _build_transforms(84, 60, train=False, aug_strength=2.0)
    img = _img(1)
    a, b = tf(image=img)["image"], tf(image=img)["image"]
    assert a.shape == (3, 84, 60) and a.dtype == torch.float32
    assert torch.equal(a, b)


def test_train_transform_is_random():
    tf = _build_transforms(84, 60, train=True, aug_strength=1.0)
    img = _img(2)
    outs = [tf(image=img)["image"] for _ in range(6)]
    assert any(not torch.equal(outs[0], o) for o in outs[1:])


# --------------------------------------------------------------------------- #
# Monitor config
# --------------------------------------------------------------------------- #
def test_monitor_defaults_to_qwk_max():
    assert monitor_settings({}) == ("val/qwk", "max")
    assert monitor_settings({"monitor": "val/mae"}) == ("val/mae", "min")
    assert monitor_settings({"monitor": "val/mae", "monitor_mode": "max"}) == ("val/mae", "max")
    with pytest.raises(ValueError):
        monitor_settings({"monitor_mode": "up"})


def test_checkpoint_filename_includes_metric_tag():
    assert checkpoint_filename("val/qwk") == "best-{epoch:02d}-qwk{val/qwk:.3f}"


def test_all_configs_have_new_keys():
    from pathlib import Path

    import yaml

    cfg_dir = Path(__file__).resolve().parents[1] / "training" / "configs"
    for name in ("mvp", "mvp_8gb", "multitask", "smoke"):
        cfg = yaml.safe_load((cfg_dir / f"{name}.yaml").read_text())
        assert cfg["sampler_power"] == 0.5, name
        assert cfg["aug_strength"] == 1.0, name
        assert monitor_settings(cfg) == ("val/qwk", "max"), name


# --------------------------------------------------------------------------- #
# Lightning QWK metric (tiny CPU model)
# --------------------------------------------------------------------------- #
class _TinyCards(torch.utils.data.Dataset):
    def __init__(self, n=12, multitask=False):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, 3, 32, 32, generator=g)
        self.grades = torch.tensor([10, 9, 8, 7] * (n // 4))
        self.multitask = multitask

    def __len__(self):
        return len(self.grades)

    def __getitem__(self, i):
        item = {"front": self.x[i], "back": self.x[i], "grade": self.grades[i]}
        if self.multitask:
            item.update(corners=torch.tensor(8.0), edges=torch.tensor(float("nan")),
                        surface=torch.tensor(9.0))
        return item


@pytest.mark.parametrize("multitask", [False, True])
def test_lightning_logs_val_and_test_qwk(multitask, tmp_path):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint

    from ml.training.lightning_module import GradeLightningModule, MultiTaskGradeLightningModule

    cls = MultiTaskGradeLightningModule if multitask else GradeLightningModule
    module = cls(backbone="resnet18", pretrained=False, max_epochs=1)
    loader = torch.utils.data.DataLoader(_TinyCards(multitask=multitask), batch_size=4)
    ckpt = ModelCheckpoint(dirpath=tmp_path, filename=checkpoint_filename("val/qwk"),
                           monitor="val/qwk", mode="max", auto_insert_metric_name=False)
    trainer = pl.Trainer(
        max_epochs=1, accelerator="cpu", logger=False, callbacks=[ckpt],
        enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
    )
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    qwk = trainer.callback_metrics["val/qwk"].item()
    assert -1.0 <= qwk <= 1.0
    assert ckpt.best_model_path and "qwk" in ckpt.best_model_path
    res = trainer.test(module, dataloaders=loader, verbose=False)
    assert "test/qwk" in res[0]


def test_qwk_buffer_matches_direct_computation():
    from ml.eval.evaluate import quadratic_weighted_kappa
    from ml.training.lightning_module import GradeLightningModule

    m = GradeLightningModule(backbone="resnet18", pretrained=False)
    logged = {}
    m.log = lambda name, value, **kw: logged.__setitem__(name, value)
    m.on_validation_epoch_start()
    t = torch.tensor([10, 9, 8, 10, 7, 6])
    p = torch.tensor([10, 10, 8, 9, 7, 8])
    m._qwk_accumulate("val", p[:3], t[:3])
    m._qwk_accumulate("val", p[3:], t[3:])
    m._qwk_accumulate("train", p, t)  # ignored
    m.on_validation_epoch_end()
    assert logged["val/qwk"] == pytest.approx(quadratic_weighted_kappa(t.numpy(), p.numpy()))


# --------------------------------------------------------------------------- #
# Threshold offsets
# --------------------------------------------------------------------------- #
def _biased_logits(n=600, shift=-1.5, seed=0, alpha=2.0, noise=0.8):
    """Realistic CORN-like logits: a noisy latent score per card,
    ``z_k = alpha * (score - (k + 1.5)) + shift``. A negative ``shift`` pushes
    every threshold down -> the model grades systematically too low (like the
    deployed v4, bias -0.72)."""
    rng = np.random.default_rng(seed)
    probs = np.array([TRAIN_COUNTS[g] for g in range(1, 11)], float)
    grades = rng.choice(np.arange(1, 11), n, p=probs / probs.sum())
    score = grades + rng.normal(0, noise, n)
    cut = np.arange(NUM_THRESHOLDS) + 1.5
    logits = alpha * (score[:, None] - cut[None, :]) + shift
    return logits.astype(np.float32), grades


def test_hard_grades_match_decode():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 3, (200, NUM_THRESHOLDS))
    ref = decode_corn_logits(torch.as_tensor(z)).predicted_grade.numpy()
    np.testing.assert_array_equal(_hard_grades(z), ref)


@pytest.mark.parametrize("objective", ["mae_qwk", "mae", "qwk"])
def test_offsets_remove_bias_and_improve_metrics(objective):
    logits, grades = _biased_logits()
    res = fit_threshold_offsets(logits, grades, temperature=1.0, objective=objective)
    assert len(res.offsets) == NUM_THRESHOLDS
    b, a = res.before, res.after
    assert b["bias"] < -0.5
    if objective in ("mae", "mae_qwk"):
        assert a["mae"] < b["mae"] - 0.2
    if objective in ("qwk", "mae_qwk"):
        assert a["qwk"] > b["qwk"]
    assert abs(a["bias"]) < 0.25
    # Offsets mostly push thresholds back up.
    assert np.mean(res.offsets) > 0


def test_offsets_hold_on_heldout_data():
    logits, grades = _biased_logits(seed=0)
    res = fit_threshold_offsets(logits, grades)
    lt, gt = _biased_logits(seed=1)
    before = offset_metrics(lt / 1.0, gt)
    after = offset_metrics(apply_calibration(lt, res.temperature, res.offsets), gt)
    assert after["mae"] < before["mae"]
    assert abs(after["bias"]) < abs(before["bias"])


def test_offsets_noop_on_unbiased_data():
    logits, grades = _biased_logits(shift=0.0)
    res = fit_threshold_offsets(logits, grades)
    assert res.after["mae"] <= res.before["mae"] + 1e-9


def test_refit_temperature_keeps_hard_grades():
    logits, grades = _biased_logits()
    a = fit_threshold_offsets(logits, grades, refit_temperature=False)
    b = fit_threshold_offsets(logits, grades, refit_temperature=True)
    za = apply_calibration(logits, a.temperature, a.offsets)
    zb = apply_calibration(logits, b.temperature, b.offsets)
    np.testing.assert_array_equal(_hard_grades(za), _hard_grades(zb))
    assert "nll" in b.after and "nll" not in a.after


def test_temperature_fit_on_logits_reduces_nll():
    logits, grades = _biased_logits(shift=0.0)
    # Over-confident: the same noisy logits blown up 10x -> T must come out > 1.
    cal = fit_temperature_on_logits(torch.as_tensor(logits * 10), torch.as_tensor(grades))
    assert cal.nll_after <= cal.nll_before + 1e-6
    assert cal.temperature > 1.0


# --------------------------------------------------------------------------- #
# Export module
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return GradeModel(backbone="resnet18", pretrained=False).eval()


def test_export_module_applies_temperature_and_offsets(tiny_model):
    from ml.training.export import CalibratedExportModule

    front, back = torch.randn(2, 3, 64, 48), torch.randn(2, 3, 64, 48)
    offsets = [0.5, -0.25, 0.0, 1.0, 2.0, -1.0, 0.3, 0.7, 1.5]
    mod = CalibratedExportModule(tiny_model, 2.0, offsets).eval()
    with torch.no_grad():
        raw = tiny_model(front, back)
        out = mod(front, back)
    torch.testing.assert_close(out, raw / 2.0 + torch.tensor(offsets))
    assert "offsets" in dict(mod.named_buffers())
    # Without offsets -> identical to old behaviour.
    with torch.no_grad():
        torch.testing.assert_close(CalibratedExportModule(tiny_model, 2.0)(front, back), raw / 2.0)
    with pytest.raises(ValueError):
        CalibratedExportModule(tiny_model, 1.0, [0.0, 1.0])


def test_export_module_offsets_survive_torchscript_and_onnx(tiny_model, tmp_path):
    from ml.training.export import CalibratedExportModule, export_onnx

    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    offsets = [float(v) for v in np.linspace(-1, 1, NUM_THRESHOLDS)]
    mod = CalibratedExportModule(tiny_model, 1.5, offsets).eval()
    front, back = torch.randn(2, 3, 64, 48), torch.randn(2, 3, 64, 48)
    with torch.no_grad():
        expected = mod(front, back).numpy()
        traced = torch.jit.trace(mod, (front[:1], back[:1]), strict=False)
        np.testing.assert_allclose(traced(front, back).numpy(), expected, atol=1e-5)

    path = tmp_path / "m.onnx"
    export_onnx(mod, path, 64, 48)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [o.name for o in sess.get_outputs()] == ["corn_logits"]
    got = sess.run(["corn_logits"], {"front": front.numpy(), "back": back.numpy()})[0]
    np.testing.assert_allclose(got, expected, atol=1e-4)
