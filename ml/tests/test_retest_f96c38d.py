"""Re-test of f96c38d: evaluate robustness / perturbation parsing, refresh docs.

tmp_path and synthetic data only; nothing under data/ is read or written.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
import numpy as np
import pytest

from ml.tests.test_retest_419a2d4 import _patched_eval


# --------------------------------------------------------------------------- #
# parse_perturbations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text, expected",
    [
        (None, None),
        ("", None),
        ("rotate_180,", ["rotate_180"]),
        (" , rotate_180 , ,", ["rotate_180"]),
        ("  swap_faces  ", ["swap_faces"]),
    ],
)
def test_parse_perturbations_accepts_stray_commas_around_a_name(text, expected):
    import ml.eval.evaluate as ev

    assert ev.parse_perturbations(text) == expected


@pytest.mark.parametrize("text", ["\t", "\n", " ", ",,,"])
def test_parse_perturbations_with_no_name_raises(text):
    import ml.eval.evaluate as ev

    with pytest.raises(ValueError, match="unknown perturbation"):
        ev.parse_perturbations(text)


def test_cli_empty_perturbations_still_means_all(monkeypatch):
    from typer.testing import CliRunner

    import ml.eval.evaluate as ev

    seen = {}

    class StopError(Exception):
        pass

    def fake_run(**kw):
        seen.update(kw)
        raise StopError

    monkeypatch.setattr(ev, "run_evaluation", fake_run)
    result = CliRunner().invoke(ev.app, ["--perturbations", ""])
    assert isinstance(result.exception, StopError)
    assert seen["perturbations"] is None


# --------------------------------------------------------------------------- #
# robustness: clean rows must be the same certs, in order
# --------------------------------------------------------------------------- #
def _svc_and_samples(tmp_path, monkeypatch, found):
    ev = _patched_eval(tmp_path, monkeypatch, found)
    import backend.app.ml.inference as inference

    return ev, inference.GradeInferenceService(), ev.load_samples("test")


def test_robustness_refuses_clean_rows_in_another_order(tmp_path, monkeypatch):
    ev, svc, samples = _svc_and_samples(tmp_path, monkeypatch, [True, True, True])
    clean = ev.predict_all(svc, samples[::-1], progress=False)
    with pytest.raises(ValueError, match="not the same certs"):
        ev.robustness(svc, samples, clean, ["swap_faces"])


def test_robustness_refuses_clean_rows_shorter_than_the_samples(tmp_path, monkeypatch):
    ev, svc, samples = _svc_and_samples(tmp_path, monkeypatch, [True, True, True])
    clean = ev.predict_all(svc, samples[:2], progress=False)
    with pytest.raises(ValueError, match="not the same certs"):
        ev.robustness(svc, samples, clean, ["swap_faces"])


def test_robustness_accepts_a_longer_clean_run_and_non_str_certs(tmp_path, monkeypatch):
    ev, svc, samples = _svc_and_samples(tmp_path, monkeypatch, [True, True, True])
    clean = ev.predict_all(svc, samples, progress=False)
    clean["cert"] = np.array([int(c) for c in clean["cert"]])
    out = ev.robustness(svc, samples[:2], clean, ["swap_faces"])["swap_faces"]
    assert out["cert"] == ["0", "1"] and out["n"] == 2


def test_without_exclude_fallback_robustness_is_the_old_call(tmp_path, monkeypatch):
    """No flag: robustness(samples[:n], rows) exactly as before the fix, fallback
    certs included, plus the `cert` list."""
    ev, svc, samples = _svc_and_samples(tmp_path, monkeypatch, [False, True, False])
    r = ev.run_evaluation(
        progress=False, n_boot=0, perturbations=["swap_faces", "rotate_180"], robustness_limit=2
    )
    old = ev.robustness(svc, samples[:2], ev.predict_all(svc, samples, progress=False),
                        ["swap_faces", "rotate_180"])
    assert r["robustness"] == old
    assert r["robustness"]["swap_faces"]["cert"] == ["0", "1"]
    assert r["meta"]["exclude_fallback"] is False


def test_exclude_fallback_robustness_limit_above_the_kept_certs(tmp_path, monkeypatch):
    ev, svc, samples = _svc_and_samples(tmp_path, monkeypatch, [True, False, True])
    r = ev.run_evaluation(
        progress=False, n_boot=0, perturbations=["swap_faces"], robustness_limit=100,
        exclude_fallback=True,
    )
    rob = r["robustness"]["swap_faces"]
    assert rob["cert"] == ["0", "2"] == [p["cert"] for p in r["predictions"]]
    assert rob["n"] == 2 and r["pipeline"]["excluded_from_metrics"] == 1
    # clean MAE is over the kept certs only (grades 10 and 8 vs true 10).
    assert rob["mae"] - rob["mae_delta"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# --refresh warning vs collect._process_cert
# --------------------------------------------------------------------------- #
def test_refresh_with_one_failed_download_keeps_urls_and_the_saved_face(tmp_path):
    """What a refresh writes when the images call succeeds but one download fails.

    The README's warning lists front_url/back_url and both paths as cleared for
    "an images call (or an image download)" failure; for a download failure the
    URLs and the downloaded face's path are in fact kept, only the failed face's
    path and has_images are reset.
    """
    from ml.data import collect
    from ml.data.psa_client import PSACertImages
    from ml.tests.test_collect_rescan import _details

    ok, jpg = cv2.imencode(".jpg", np.full((40, 28, 3), 128, np.uint8))
    assert ok

    class PSA:
        async def get_cert_details(self, cert):
            return _details(str(cert))

        async def get_cert_images(self, cert):
            return PSACertImages(str(cert), "https://example.invalid/f.jpg", "https://example.invalid/b.jpg")

        async def download_image_bytes(self, url):
            return jpg.tobytes() if url.endswith("f.jpg") else None

    images = tmp_path / "images"
    images.mkdir()
    row = asyncio.run(collect._process_cert(PSA(), 12345678, images, asyncio.Semaphore(1)))
    assert row["has_images"] == 0
    assert row["front_url"] and row["back_url"]
    assert Path(row["front_path"]) == Path("images/12345678_front.jpg") and row["back_path"] is None
