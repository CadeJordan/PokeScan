"""Independent retest of guard-skip-orientation-without-card.

`FaceGuard.decide(feats, found_quad)` must never rotate a face with no card
outline (the flags follow a swap), and must decide every face that has a card
exactly as the pre-change guard (df2a561) did. Synthetic features and
hand-built guards only; the old module is loaded from git into tmp_path.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import backend.app.ml.inference as inference_mod
from backend.app.ml.face_guard import CHECKS, NOTE_FLIPPED, NOTE_SWAPPED, FaceGuard, LogisticProbe
from backend.app.ml.features import REGION_NAMES
from backend.app.ml.preprocess import detect_and_crop
from backend.tests.test_face_guard import THRESHOLDS, _feats, _guard, _logit, _service
from backend.tests.test_face_guard import blank as _blank_fixture
from backend.tests.test_face_guard import pair as _pair_fixture
from backend.tests.test_retest_reject_uploads import CARD_NOT_FOUND, _post

pair = _pair_fixture
blank = _blank_fixture

D = 4
REPO = Path(__file__).resolve().parents[2]
OLD_COMMIT = "df2a561"


# --------------------------------------------------------------------------- #
# The pre-change guard, loaded from git
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def old_face_guard(tmp_path_factory):
    try:
        src = subprocess.run(
            ["git", "show", f"{OLD_COMMIT}:backend/app/ml/face_guard.py"],
            cwd=REPO, capture_output=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover - no git
        pytest.skip(f"can't read {OLD_COMMIT} from git: {exc}")
    path = tmp_path_factory.mktemp("old_guard") / "old_face_guard.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location("old_face_guard", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look their module up here
    try:
        spec.loader.exec_module(mod)
        assert "found_quad" not in src.decode()  # really the old version
        yield mod
    finally:
        sys.modules.pop(spec.name, None)


def _random_guards(mod, rng, n):
    """`n` pairs (new, old) of identical guards with random weights/thresholds/checks."""
    out = []
    for _ in range(n):
        probes, old_probes = {}, {}
        for name in CHECKS:
            w = rng.normal(0.0, 1.0, 2 * D)
            b = float(rng.normal(0.0, 1.0))
            probes[name] = LogisticProbe(w.copy(), b)
            old_probes[name] = mod.LogisticProbe(w.copy(), b)
        thresholds = {
            name: float(rng.choice([0.9, 0.95, 0.99, rng.uniform(0.501, 0.999)])) for name in CHECKS
        }
        enabled = tuple(c for c in CHECKS if rng.random() < 0.7)
        out.append(
            (
                FaceGuard(probes, dict(thresholds), enabled),
                mod.FaceGuard(old_probes, dict(thresholds), enabled),
            )
        )
    return out


def _random_feats(rng, guard: FaceGuard) -> np.ndarray:
    f = rng.normal(0.0, 3.0, (2, len(REGION_NAMES), D)).astype(np.float32)
    if rng.random() < 0.2:
        # Put the side/flip logits right at a threshold, to hit the > edge.
        f[:, :2, :] = 0.0
        w = guard.probes["front_upright"].weight
        k = int(np.argmax(np.abs(w)))
        t = guard.thresholds["front_upright"]
        target = (_logit(t) - guard.probes["front_upright"].bias) / w[k]
        if k < D:
            f[0, 0, k] = target
        else:
            f[0, 1, k - D] = target
    return f


def _fields(d):
    return (d.swap, tuple(d.flip), list(d.notes), d.p_back, d.p_flip, d.same_side)


def test_faces_with_a_card_are_decided_exactly_as_before(old_face_guard):
    rng = np.random.default_rng(20261030)
    seen = {"swap": 0, "same": 0, "flip": 0, "none": 0}
    for new, old in _random_guards(old_face_guard, rng, 60):
        for _ in range(25):
            feats = _random_feats(rng, new)
            want = old.decide(feats)
            for got in (new.decide(feats), new.decide(feats, (True, True)), new.decide(feats, [True, True])):
                assert _fields(got) == _fields(want)
                assert got.no_card == (False, False)
            # check_orientation too (its tuple grew a 4th element)
            o_flip, o_notes, o_p = old.check_orientation(feats)
            n_flip, n_notes, n_p, n_nc = new.check_orientation(feats)
            assert (n_flip, n_notes, n_p, n_nc) == (o_flip, o_notes, o_p, (False, False))
            seen["swap"] += want.swap
            seen["same"] += want.same_side is not None
            seen["flip"] += any(want.flip)
            seen["none"] += not want.notes
    assert all(v > 20 for v in seen.values()), seen  # every branch was exercised


@pytest.mark.parametrize("quads", [(False, True), (True, False), (False, False)])
def test_no_card_faces_only_lose_their_flip(old_face_guard, quads):
    """Against the old guard: same swap, P(back), P(flip), same_side; a face's flip
    and its note are dropped exactly when its (post-swap) card is missing."""
    rng = np.random.default_rng(hash(quads) % 2**32)
    skipped = 0
    for new, old in _random_guards(old_face_guard, rng, 40):
        for _ in range(25):
            feats = _random_feats(rng, new)
            want = old.decide(feats)
            got = new.decide(feats, quads)
            assert (got.swap, got.p_back, got.p_flip, got.same_side) == (
                want.swap, want.p_back, want.p_flip, want.same_side,
            )
            graded = quads[::-1] if want.swap else quads
            if want.same_side is not None:
                assert got.flip == (False, False) and got.no_card == (False, False)
                assert got.notes == want.notes
                continue
            enabled = tuple(c in new.enabled for c in ("front_upright", "back_upright"))
            assert got.no_card == tuple(e and not q for e, q in zip(enabled, graded, strict=True))
            assert got.flip == tuple(f and q for f, q in zip(want.flip, graded, strict=True))
            dropped = [NOTE_FLIPPED.format(face) for face, f, q in zip(("front", "back"), want.flip, graded, strict=True) if f and not q]
            assert got.notes == [n for n in want.notes if n not in dropped]
            skipped += len(dropped)
            assert all(type(v) is bool for v in got.flip + got.no_card)
    assert skipped > 10


# --------------------------------------------------------------------------- #
# Edge cases on decide
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "found",
    [
        [False, True],
        (np.bool_(False), np.bool_(True)),
        np.array([False, True]),
        (0, 1),
    ],
    ids=["list", "np.bool_", "ndarray", "ints"],
)
def test_found_quad_types_with_a_swap(found):
    # Uploaded front: no card, reads as a back. After the swap it is the graded
    # back; the graded front (uploaded back, card found) is flipped.
    feats = _feats(side=(10.0, -10.0), flip=(10.0, 10.0))
    d = _guard().decide(feats, found)
    assert d.swap and d.flip == (True, False) and d.no_card == (False, True)
    assert all(type(v) is bool for v in d.flip + d.no_card)
    assert d == _guard().decide(feats, (False, True))


@pytest.mark.parametrize("found", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("side", [(-10.0, -10.0), (10.0, 10.0)])
def test_same_side_pair_records_no_skip(found, side):
    """A same-side pair skips the orientation check for its own reason: no flip,
    P(flip) None and no_card all False, whatever the card flags."""
    d = _guard().decide(_feats(side=side, flip=(10.0, 10.0)), found)
    assert d.same_side is not None and d.p_flip is None
    assert d.flip == (False, False) and d.no_card == (False, False)


@pytest.mark.parametrize("found", [(False, False), (False, True), (True, False)])
def test_orientation_checks_disabled_records_no_skip(found):
    d = _guard(enabled=("side",)).decide(_feats(side=(10.0, -10.0), flip=(10.0, 10.0)), found)
    assert d.swap and d.p_flip is None and d.flip == (False, False) and d.no_card == (False, False)


def test_only_back_check_enabled():
    feats = _feats(flip=(10.0, 10.0))
    g = _guard(enabled=("side", "back_upright"))
    assert g.decide(feats, (True, True)).flip == (False, True)
    d = g.decide(feats, (False, False))
    assert d.flip == (False, False) and d.no_card == (False, True) and d.notes == []
    d = g.decide(feats, (False, True))
    assert d.flip == (False, True) and d.no_card == (False, False)


def test_without_side_check_the_flags_are_not_swapped():
    # Rows read as swapped, but the side check is off: flags stay per uploaded row.
    feats = _feats(side=(10.0, -10.0), flip=(10.0, 10.0))
    d = _guard(enabled=("front_upright", "back_upright")).decide(feats, (False, True))
    assert not d.swap and d.flip == (False, True) and d.no_card == (True, False)


@pytest.mark.parametrize("delta", [-1e-6, 0.0, 1e-6])
def test_threshold_edge_is_unchanged_with_a_card_and_never_flips_without(delta):
    t = THRESHOLDS["front_upright"]
    feats = _feats(flip=(_logit(t) + delta, -10.0))
    with_card = _guard().decide(feats, (True, True))
    p = with_card.p_flip[0]
    assert with_card.flip[0] == (p > t)
    without = _guard().decide(feats, (False, True))
    assert without.flip == (False, False) and without.p_flip == with_card.p_flip


# --------------------------------------------------------------------------- #
# Service: no rotation of a no-card face, in every combination
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("flip", [(-10.0, -10.0), (10.0, -10.0), (-10.0, 10.0), (10.0, 10.0)])
@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("front_card, back_card", [(True, True), (False, True), (True, False), (False, False)])
def test_service_never_rotates_a_face_without_a_card(
    tmp_path, pair, blank, monkeypatch, front_card, back_card, swap, flip
):
    rotated = []
    real_rotate = inference_mod._rotate_crop

    def spy(crop):
        rotated.append(crop.found_quad)
        return real_rotate(crop)

    monkeypatch.setattr(inference_mod, "_rotate_crop", spy)
    side = (10.0, -10.0) if swap else (-10.0, 10.0)
    first, second = _feats(side=side, flip=flip), _feats(tag=4.0)
    svc = _service(tmp_path, _guard(), first, second)
    front = pair[0] if front_card else blank
    back = pair[1] if back_card else blank
    pred = svc.predict(front, back)

    assert all(rotated), "a crop with no card outline was rotated"
    graded_quads = (back_card, front_card) if swap else (front_card, back_card)
    # After a swap row 1's flip logit is the graded front's.
    graded_flip = (flip[1] > 0, flip[0] > 0) if swap else (flip[0] > 0, flip[1] > 0)
    want_flip = tuple(f and q for f, q in zip(graded_flip, graded_quads, strict=True))
    d = pred.guard_decision
    assert d.swap == swap and d.flip == want_flip
    assert d.no_card == tuple(not q for q in graded_quads)
    assert len(rotated) == sum(want_flip)
    assert svc._extractor.calls == 1 + any(want_flip)
    want_notes = ([NOTE_SWAPPED] if swap else []) + [
        NOTE_FLIPPED.format(face) for face, f in zip(("front", "back"), want_flip, strict=True) if f
    ]
    assert pred.notes == want_notes
    assert (pred.found_quad_front, pred.found_quad_back) == graded_quads
    # rejection() still names the uploaded faces.
    missing = [f for f, ok in zip(("front", "back"), (front_card, back_card), strict=True) if not ok]
    if missing:
        assert pred.rejection() == ("card_not_found", f"{' and '.join(missing)}: {CARD_NOT_FOUND}")
    else:
        assert pred.rejection() is None
    # A no-card face is graded on its unrotated crop.
    uploads = (back, front) if swap else (front, back)
    for i in range(2):
        if not graded_quads[i]:
            np.testing.assert_array_equal(pred.crops[i].image, detect_and_crop(uploads[i]).image)


def test_no_card_flipped_front_grades_like_the_unguarded_service(tmp_path, pair, blank):
    """With nothing to swap, a guarded no-card prediction equals the unguarded one
    (guard_decision isn't part of its value)."""
    feats = _feats(flip=(10.0, 10.0))
    guarded = _service(tmp_path, _guard(), feats, _feats(tag=8.0)).predict(blank, blank)
    plain = _service(tmp_path, None, feats).predict(blank, blank)
    assert guarded == plain
    assert guarded.guard_decision.no_card == (True, True)
    assert guarded.guard_decision.p_flip[0] > 0.99 and guarded.guard_decision.p_flip[1] > 0.99


@pytest.mark.parametrize("which", ["front", "back", "both"])
def test_api_response_for_a_flipped_no_card_face_is_the_unguarded_422(
    tmp_path, pair, blank, monkeypatch, which
):
    front = blank if which in ("front", "both") else pair[0]
    back = blank if which in ("back", "both") else pair[1]
    feats = _feats(flip=(10.0, 10.0))
    r_guarded = _post(_service(tmp_path, _guard(), feats, _feats(tag=2.0)), monkeypatch, front, back)
    r_plain = _post(_service(tmp_path, None, feats), monkeypatch, front, back)
    assert r_guarded.status_code == 422 and r_plain.status_code == 422
    assert r_guarded.json() == r_plain.json()
    faces = "front and back" if which == "both" else which
    assert r_guarded.json()["detail"] == f"{faces}: {CARD_NOT_FOUND}"


def test_api_still_rotates_a_flipped_front_with_a_card(tmp_path, pair, monkeypatch):
    r = _post(
        _service(tmp_path, _guard(), _feats(flip=(10.0, -10.0)), _feats(tag=2.0)), monkeypatch, *pair
    )
    assert r.status_code == 200
    assert NOTE_FLIPPED.format("front") in r.json()["notes"]


def test_service_passes_the_crops_found_quad_to_decide(tmp_path, pair, blank, monkeypatch):
    seen = []
    real = FaceGuard.decide

    def spy(self, feats, found_quad=(True, True)):
        seen.append(tuple(found_quad))
        return real(self, feats, found_quad)

    monkeypatch.setattr(FaceGuard, "decide", spy)
    for f, b, want in [(pair[0], pair[1], (True, True)), (blank, pair[1], (False, True)), (pair[0], blank, (True, False))]:
        _service(tmp_path, _guard(), _feats()).predict(f, b)
        assert seen[-1] == want
