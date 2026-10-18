"""Regression gate: the exported model must not get worse on the test split.

Compares against ml/eval/baseline.json (pin it with
`python -m ml.eval.evaluate --save-baseline`). Without a baseline, falls back
to absolute floors. POKESCAN_EVAL_LIMIT=N evaluates only the first N cards.
"""

from __future__ import annotations

import json
import os

import pytest

from ml.eval.evaluate import BASELINE_PATH, compare_to_baseline, run_evaluation

pytestmark = [pytest.mark.model, pytest.mark.slow]

# Used only when no baseline.json has been pinned yet.
FLOORS = {"mae": 1.0, "off_by_one": 0.80, "qwk": 0.40, "ece": 0.15}


@pytest.fixture(scope="module")
def report(models_dir):
    from backend.app.core.config import get_settings

    if not get_settings().db_path.exists():
        pytest.skip("no SQLite DB")
    limit = int(os.environ.get("POKESCAN_EVAL_LIMIT", "0")) or None
    try:
        return run_evaluation(split="test", limit=limit, robustness_limit=0, progress=False)
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_no_regression(report):
    if BASELINE_PATH.exists():
        problems = compare_to_baseline(report, json.loads(BASELINE_PATH.read_text()))
    else:
        o = {**report["overall"], "ece": report["calibration"]["ece"]}
        problems = [
            f"{k}: {o[k]:.4f} vs floor {v}"
            for k, v in FLOORS.items()
            if (o[k] > v if k in ("mae", "ece") else o[k] < v)
        ]
    assert not problems, "; ".join(problems)


def test_beats_trivial_baseline(report):
    model = report["overall"]
    majority = report["baselines"]["majority_grade"]
    assert model["mae"] < majority["mae"]
    assert model["qwk"] > majority["qwk"] + 0.1


def test_detection_rate_on_real_photos(report):
    assert report["pipeline"]["quad_found_front"] >= 0.9
    assert report["pipeline"]["quad_found_back"] >= 0.9


# Leakage guard. A test input where no card outline is found is fed to the
# model as the padded raw slab photo, whose PSA label prints the grade, so
# every fallback can inflate the score. The target is zero, but the detector
# still misses 5 of the 393 test certs (1.3%; one is a die-cut card), as
# recorded in certs.crop_found_* by
# precrop.py. Allow at most 1.5% so a detector regression fails this test;
# lower the bound as detection improves.
MAX_FALLBACK_FRAC = 0.015


def test_no_label_leak_through_fallback(report):
    n = report["overall"]["n"]
    n_fallback = report["pipeline"]["n_fallback"]
    assert n_fallback <= MAX_FALLBACK_FRAC * n, (
        f"{n_fallback}/{n} test inputs fell back to the raw slab photo (label visible)"
    )
