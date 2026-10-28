"""ml/data/collect.py re-scans: upserting an already-collected cert again must
not reset the columns later pipeline steps own (`split` above all)."""

from __future__ import annotations

from ml.data.collect import _details_row
from ml.data.db import connect, update_factor_subgrades, update_precrop_paths, upsert_cert
from ml.data.psa_client import PSACertDetails


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
