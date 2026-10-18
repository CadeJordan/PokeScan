"""CORN helpers and GradeModel (backend/app/ml/model.py)."""

from __future__ import annotations

import pytest
import torch

from backend.app.ml.model import (
    NUM_THRESHOLDS,
    GradeModel,
    corn_loss,
    decode_corn_logits,
    grade_to_targets,
)

GRADES = torch.arange(1, 11)


def test_grade_to_targets_values():
    t = grade_to_targets(torch.tensor([1, 5, 10]))
    assert t.shape == (3, NUM_THRESHOLDS)
    assert t[0].sum() == 0
    assert t[1].tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 0]
    assert t[2].sum() == 9


def test_grade_to_targets_clamps():
    t = grade_to_targets(torch.tensor([0, 11]))
    assert t[0].sum() == 0 and t[1].sum() == 9


def test_targets_decode_roundtrip():
    # Perfectly confident logits built from targets decode back to the grade.
    logits = (grade_to_targets(GRADES) * 2 - 1) * 20
    out = decode_corn_logits(logits)
    assert out.predicted_grade.tolist() == GRADES.tolist()
    torch.testing.assert_close(out.expected_grade, GRADES.float(), atol=1e-4, rtol=0)
    assert (out.confidence > 0.999).all()


def test_decode_enforces_monotone_probs():
    logits = torch.tensor([[3.0, -3.0, 3.0, 3.0, -3.0, 3.0, 3.0, 3.0, 3.0]])
    out = decode_corn_logits(logits)
    p = out.rank_probs[0]
    assert (p[1:] <= p[:-1]).all()
    assert out.predicted_grade.item() == 2


def test_decode_confidence_bounds():
    out = decode_corn_logits(torch.randn(64, NUM_THRESHOLDS) * 3)
    assert ((out.confidence >= 0) & (out.confidence <= 1)).all()
    assert ((out.expected_grade >= 1) & (out.expected_grade <= 10)).all()
    assert ((out.predicted_grade >= 1) & (out.predicted_grade <= 10)).all()


def test_corn_loss_prefers_correct_logits():
    good = (grade_to_targets(GRADES) * 2 - 1) * 5
    bad = -good
    assert corn_loss(good, GRADES) >= 0
    assert corn_loss(good, GRADES) < corn_loss(bad, GRADES)


def test_corn_loss_ignores_inactive_thresholds():
    # Grade 1: only threshold 0 is active; changing later logits must not matter.
    a = torch.full((1, NUM_THRESHOLDS), -5.0)
    b = a.clone()
    b[0, 3:] = 50.0
    g = torch.tensor([1])
    assert corn_loss(a, g).item() == pytest.approx(corn_loss(b, g).item())


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return GradeModel(backbone="resnet18", pretrained=False, dropout=0.0)


def test_forward_shape(tiny_model):
    x = torch.randn(2, 3, 112, 80)
    logits = tiny_model(x, x)
    assert logits.shape == (2, NUM_THRESHOLDS)
    out = tiny_model.eval().predict(x, x)
    assert out.predicted_grade.shape == (2,)


def test_front_weight_blend(tiny_model):
    tiny_model.eval()
    f, b = torch.randn(1, 3, 112, 80), torch.randn(1, 3, 112, 80)
    with torch.no_grad():
        fused = 0.7 * tiny_model._embed(f) + 0.3 * tiny_model._embed(b)
        torch.testing.assert_close(tiny_model(f, b), tiny_model.fc(fused))


def test_gradients_reach_shared_backbone(tiny_model):
    tiny_model.train()
    tiny_model.zero_grad()
    x = torch.randn(2, 3, 112, 80)
    corn_loss(tiny_model(x, x), torch.tensor([3, 9])).backward()
    grads = [p.grad for p in tiny_model.backbone.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("w", [0.0, 1.0, -0.1])
def test_front_weight_validation(w):
    with pytest.raises(ValueError):
        GradeModel(backbone="resnet18", pretrained=False, front_weight=w)
