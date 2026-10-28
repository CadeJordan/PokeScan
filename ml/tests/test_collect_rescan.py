"""ml/data/collect.py re-scans: upserting an already-collected cert again must
not reset the columns later pipeline steps own (`split` above all), and a
re-scan skips certs already collected with images unless `--refresh`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from ml.data import collect, collect_all
from ml.data.collect import _details_row
from ml.data.db import connect, update_factor_subgrades, update_precrop_paths, upsert_cert
from ml.data.psa_client import PSACertDetails, PSACertImages


def _details(cert="12345678"):
    return PSACertDetails(
        cert_number=cert,
        brand_title="POKEMON BASE SET",
        subject="CHARIZARD-HOLO",
        year="1999",
        card_number="4",
        category="TCG Cards",
        grade_label="MINT 9",
        grade_int=9,
        is_dual_cert=False,
        raw={"CertNumber": cert},
    )


def _scanned_row(details):
    """The row `_process_cert` upserts once both images were saved."""
    row = _details_row(details)
    row.update(
        front_url="https://example.invalid/f.jpg",
        back_url="https://example.invalid/b.jpg",
        front_path=f"images/{details.cert_number}_front.jpg",
        back_path=f"images/{details.cert_number}_back.jpg",
        has_images=1,
    )
    return row


def _get(conn, cert):
    return conn.execute("SELECT * FROM certs WHERE cert_number=?", (cert,)).fetchone()


def test_rescan_keeps_assigned_split(tmp_path):
    db = tmp_path / "certs.sqlite"
    details = _details()
    with connect(db) as conn:
        upsert_cert(conn, _scanned_row(details))
        assert _get(conn, "12345678")["split"] is None  # fresh cert: unassigned

        conn.execute("UPDATE certs SET split='test' WHERE cert_number='12345678'")
        update_precrop_paths(
            conn, "12345678",
            cropped_front_path="crops/f.jpg", cropped_back_path="crops/b.jpg",
            found_front=True, found_back=True, kind_front="card", kind_back="card",
        )
        update_factor_subgrades(conn, "12345678", corners=8.0, edges=9.0, surface=7.5)

        upsert_cert(conn, _scanned_row(details))  # re-scan of the same cert range
        row = _get(conn, "12345678")

    assert row["split"] == "test"
    assert row["cropped_front_path"] == "crops/f.jpg"
    assert row["crop_found_front"] == 1
    assert row["crop_kind_back"] == "card"
    assert row["corners_subgrade"] == 8.0
    assert row["has_images"] == 1


def test_details_row_omits_pipeline_owned_columns():
    row = _details_row(_details())
    for col in ("split", "split_legacy", "cropped_front_path", "crop_found_front",
                "corners_subgrade", "precrop_at", "factors_computed_at"):
        assert col not in row


# --------------------------------------------------------------------------- #
# A re-scan skips certs already collected with images (smallfix 11)
# --------------------------------------------------------------------------- #
class _FakePSA:
    """Stands in for PSAClient: no network, records every call in `calls`."""

    calls: list = []

    def __init__(self, token, daily_limit=100):
        self._daily_limit = daily_limit

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    @property
    def calls_remaining(self):
        return max(0, self._daily_limit - len(self.calls))

    async def get_cert_details(self, cert_number):
        self.calls.append(("details", str(cert_number)))
        return _details(str(cert_number))

    async def get_cert_images(self, cert_number):
        self.calls.append(("images", str(cert_number)))
        # An empty images response is what used to wipe a collected cert's paths.
        return PSACertImages(str(cert_number), None, None)

    async def download_image_bytes(self, url):
        self.calls.append(("download", url))
        return None


@pytest.fixture
def fake_psa(tmp_path, monkeypatch):
    """A tmp_path DB + images dir with the PSA client stubbed; returns the DB path."""
    settings = SimpleNamespace(
        db_path=tmp_path / "certs.sqlite",
        images_dir=tmp_path / "images",
        psa_api_tokens={0: "token"},
        get_psa_token=lambda index=None: "token",
    )
    monkeypatch.setattr(collect, "get_settings", lambda: settings)
    monkeypatch.setattr(collect_all, "get_settings", lambda: settings)
    monkeypatch.setattr(_FakePSA, "calls", [])
    monkeypatch.setattr(collect, "PSAClient", _FakePSA)
    return settings.db_path


def _collected(db, cert):
    """Store `cert` as fully collected: images, split, crops and sub-grades."""
    with connect(db) as conn:
        upsert_cert(conn, _scanned_row(_details(cert)))
        conn.execute("UPDATE certs SET split='test' WHERE cert_number=?", (cert,))
        update_precrop_paths(
            conn, cert,
            cropped_front_path="crops/f.jpg", cropped_back_path="crops/b.jpg",
            found_front=True, found_back=True, kind_front="card", kind_back="card",
        )
        update_factor_subgrades(conn, cert, corners=8.0, edges=9.0, surface=7.5)


def _collect(*args):
    return CliRunner().invoke(collect.app, ["--start-cert", "12345678", *args])


def _db_row(db, cert):
    with connect(db) as conn:
        return dict(_get(conn, cert))


def test_rescan_skips_cert_collected_with_images(fake_psa, caplog):
    _collected(fake_psa, "12345678")
    before = _db_row(fake_psa, "12345678")
    with caplog.at_level("INFO", logger="collect"):
        result = _collect("--count", "1")
    assert result.exit_code == 0, result.output
    assert _FakePSA.calls == []  # no quota spent on it
    assert _db_row(fake_psa, "12345678") == before
    assert before["front_path"] == "images/12345678_front.jpg"
    assert before["split"] == "test" and before["cropped_front_path"] == "crops/f.jpg"
    assert any("skipped 1 certs" in r.getMessage() for r in caplog.records)


def test_rescan_with_refresh_refetches_collected_cert(fake_psa):
    _collected(fake_psa, "12345678")
    result = _collect("--count", "1", "--refresh")
    assert result.exit_code == 0, result.output
    assert ("details", "12345678") in _FakePSA.calls
    assert ("images", "12345678") in _FakePSA.calls
    row = _db_row(fake_psa, "12345678")
    # A deliberate re-fetch writes what PSA returned (here: no images) ...
    assert row["has_images"] == 0 and row["front_path"] is None
    # ... and still leaves the pipeline-owned columns alone.
    assert row["split"] == "test" and row["corners_subgrade"] == 8.0


def test_rescan_refetches_cert_without_images(fake_psa):
    with connect(fake_psa) as conn:
        upsert_cert(conn, _details_row(_details("12345678")))  # has_images=0
        upsert_cert(conn, _scanned_row(_details("12345679")))  # has_images=1
    result = _collect("--count", "2")
    assert result.exit_code == 0, result.output
    assert {c for kind, c in _FakePSA.calls if kind == "details"} == {"12345678"}
    assert ("images", "12345678") in _FakePSA.calls
    assert _db_row(fake_psa, "12345679")["has_images"] == 1


def test_skipped_certs_do_not_count_towards_the_404_abort(fake_psa, monkeypatch):
    # A range that starts with collected certs is an issued range: the misses
    # after them must not trip the "zero hits" abort.
    for cert in ("12345678", "12345679"):
        _collected(fake_psa, cert)
    monkeypatch.setattr(_FakePSA, "get_cert_details", _miss)
    result = _collect("--count", "10", "--abort-after-misses", "2", "--concurrency", "1")
    assert result.exit_code == 0, result.output
    assert len(_FakePSA.calls) == 8  # every uncollected cert was tried


async def _miss(self, cert_number):
    self.calls.append(("details", str(cert_number)))
    return None


def test_collect_all_forwards_refresh(fake_psa, monkeypatch):
    seen = []

    async def fake_run(**kwargs):
        seen.append(kwargs["refresh"])

    monkeypatch.setattr(collect_all, "_run", fake_run)
    for args, want in (([], False), (["--refresh"], True)):
        seen.clear()
        result = CliRunner().invoke(collect_all.app, ["--starts", "1", "--tokens", "0", *args])
        assert result.exit_code == 0, result.output
        assert seen == [want]
