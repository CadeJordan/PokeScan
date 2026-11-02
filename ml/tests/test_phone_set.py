"""The phone-photo eval set helper (ml/eval/phone_set.py), on tmp folders only."""

from __future__ import annotations

import csv
import io
import sqlite3

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from ml.eval.evaluate import load_manifest
from ml.eval.phone_set import (
    MANIFEST_COLUMNS,
    PhoneSetError,
    app,
    build_manifest,
    lookup_splits,
    parse_photo_name,
    write_manifest,
)


def _image_bytes(fmt: str = "JPEG", seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (48, 32, 3), np.uint8)).save(buf, format=fmt)
    return buf.getvalue()


# The first bytes of an iPhone HEIC file: an ISO-BMFF `ftyp` box, brand heic.
_HEIC = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"\x00" * 64


def _make_set(tmp_path, cards="cert,grade,notes\n111,9,\n222,10,upside down casual\n", photos=None):
    """A valid set by default: both shots of 111, the careful pair of 222."""
    folder = tmp_path / "phone_eval"
    (folder / "photos").mkdir(parents=True)
    (folder / "cards.csv").write_text(cards, encoding="utf-8")
    if photos is None:
        photos = [
            "111_careful_front.jpg", "111_careful_back.JPG",
            "111_casual_front.jpeg", "111_casual_back.png",
            "222_Careful_Front.webp", "222_careful_back.jpg",
        ]
    fmts = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}
    for i, name in enumerate(photos):
        ext = name[name.rfind("."):].lower()
        (folder / "photos" / name).write_bytes(_image_bytes(fmts.get(ext, "JPEG"), seed=i))
    return folder


def _make_db(path, rows):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE certs (cert_number TEXT PRIMARY KEY, split TEXT)")
    conn.executemany("INSERT INTO certs VALUES (?, ?)", rows)
    conn.commit()
    conn.close()
    return path


def _problems(folder, db_path=None) -> list[str]:
    with pytest.raises(PhoneSetError) as exc:
        build_manifest(folder, db_path)
    return exc.value.problems


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("82001234_careful_front.jpg", ("82001234", "careful", "front")),
        ("82001234_CASUAL_Back.JPEG", ("82001234", "casual", "back")),
        ("1_careful_back.webp", ("1", "careful", "back")),
        ("82001234_careful_front.heic", None),   # format, not name: see below
        ("82001234_sloppy_front.jpg", None),
        ("82001234_careful_side.jpg", None),
        ("abc_careful_front.jpg", None),
        ("82001234_careful_front_2.jpg", None),
        ("IMG_0001.jpg", None),
    ],
)
def test_parse_photo_name(name, expected):
    assert parse_photo_name(name) == expected


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_build_pairs_photos_and_writes_manifest(tmp_path):
    folder = _make_set(tmp_path)
    db = _make_db(tmp_path / "db.sqlite", [("111", "train"), ("999", "test")])
    result = build_manifest(folder, db)
    assert [r["id"] for r in result.rows] == ["111_careful", "111_casual", "222_careful"]
    first = result.rows[0]
    assert first == {
        "front": "photos/111_careful_front.jpg",
        "back": "photos/111_careful_back.JPG",
        "grade": "9",
        "id": "111_careful",
        "shot": "careful",
        "cert": "111",
        "in_db_split": "train",
    }
    assert result.rows[2]["in_db_split"] == "absent"
    assert result.rows[2]["front"] == "photos/222_Careful_Front.webp"
    assert any("222" in w and "only the careful pair" in w for w in result.warnings)

    before = sorted(p.name for p in folder.rglob("*"))
    path = write_manifest(folder, result.rows)
    assert path == folder / "manifest.csv"
    assert sorted(p.name for p in folder.rglob("*")) == sorted([*before, "manifest.csv"])
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames) == MANIFEST_COLUMNS
        assert list(reader) == result.rows

    # The evaluator reads it back, extra columns as tags.
    samples = {s.cert: s for s in load_manifest(path)}
    assert samples["111_casual"].front == folder / "photos" / "111_casual_front.jpeg"
    assert samples["111_casual"].tags == {"shot": "casual", "cert": "111", "in_db_split": "train"}
    assert samples["222_careful"].grade == 10


def test_build_db_lookup_is_read_only(tmp_path):
    folder = _make_set(tmp_path)
    db = _make_db(tmp_path / "db.sqlite", [("111", "test"), ("222", None)])
    before = db.read_bytes()
    rows = build_manifest(folder, db).rows
    assert {r["cert"]: r["in_db_split"] for r in rows} == {"111": "test", "222": "unassigned"}
    assert db.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["db.sqlite", "phone_eval"]


@pytest.mark.parametrize("dirname", ["a b", "a#b", "a%20b", "pct%41", "café & co"])
def test_db_path_with_uri_characters_opens_that_file(tmp_path, dirname):
    # Unescaped, '#' and '%' in the path open the wrong file or fail.
    (tmp_path / "a b").mkdir()
    _make_db(tmp_path / "a b" / "db.sqlite", [("111", "val")])
    d = tmp_path / dirname
    d.mkdir(exist_ok=True)
    db = _make_db(d / "db.sqlite", [("111", "test")]) if dirname != "a b" else d / "db.sqlite"
    expected = "val" if dirname == "a b" else "test"
    splits, warnings = lookup_splits(db, ["111"])
    assert splits == {"111": expected} and not warnings


def test_excel_bom_crlf_cards_csv_and_header_whitespace(tmp_path):
    folder = _make_set(tmp_path)
    (folder / "cards.csv").write_bytes(b"\xef\xbb\xbf Cert , GRADE \r\n111,9\r\n222,10\r\n")
    rows = build_manifest(folder, None).rows
    assert {r["cert"]: r["grade"] for r in rows} == {"111": "9", "222": "10"}


def test_non_utf8_cards_csv_is_a_problem(tmp_path):
    folder = _make_set(tmp_path)
    (folder / "cards.csv").write_bytes("cert,grade,notes\n111,9,café\n".encode("cp1252"))
    with pytest.raises(PhoneSetError, match="CSV UTF-8"):
        build_manifest(folder, None)


def test_non_ascii_digits_dont_parse():
    assert parse_photo_name("١٢٣_careful_front.jpg") is None


def test_missing_db_is_unknown_with_warning(tmp_path):
    folder = _make_set(tmp_path)
    result = build_manifest(folder, tmp_path / "nope.sqlite")
    assert {r["in_db_split"] for r in result.rows} == {"unknown"}
    assert any("DB not found" in w for w in result.warnings)
    assert not (tmp_path / "nope.sqlite").exists()


def test_db_without_certs_table_is_unknown_with_warning(tmp_path):
    folder = _make_set(tmp_path)
    db = tmp_path / "empty.sqlite"
    sqlite3.connect(db).close()
    splits, warnings = lookup_splits(db, ["111"])
    assert splits == {"111": "unknown"} and warnings
    result = build_manifest(folder, db)
    assert {r["in_db_split"] for r in result.rows} == {"unknown"}
    assert any("unreadable" in w for w in result.warnings)


def test_card_without_photos_warns_not_fails(tmp_path):
    folder = _make_set(tmp_path, cards="cert,grade\n111,9\n222,10\n333,8\n")
    result = build_manifest(folder, None)
    assert "333" not in {r["cert"] for r in result.rows}
    assert any("333" in w and "no photos" in w for w in result.warnings)


def test_ignores_os_litter(tmp_path):
    folder = _make_set(tmp_path)
    (folder / "photos" / "Thumbs.db").write_bytes(b"x")
    (folder / "photos" / ".DS_Store").write_bytes(b"x")
    assert len(build_manifest(folder, None).rows) == 3


# --------------------------------------------------------------------------- #
# Every problem, listed at once and naming the file
# --------------------------------------------------------------------------- #
def test_unparseable_name(tmp_path):
    folder = _make_set(tmp_path)
    (folder / "photos" / "IMG_0001.jpg").write_bytes(_image_bytes())
    (problem,) = _problems(folder)
    assert problem.startswith("IMG_0001.jpg:") and "doesn't parse" in problem


def test_cert_not_in_cards(tmp_path):
    folder = _make_set(tmp_path)
    (folder / "photos" / "555_careful_front.jpg").write_bytes(_image_bytes())
    (folder / "photos" / "555_careful_back.jpg").write_bytes(_image_bytes())
    problems = _problems(folder)
    assert len(problems) == 2
    assert all("not in cards.csv" in p and "555" in p for p in problems)


@pytest.mark.parametrize("drop", ["111_careful_back.JPG", "111_casual_front.jpeg"])
def test_missing_face_of_a_shot(tmp_path, drop):
    folder = _make_set(tmp_path)
    (folder / "photos" / drop).unlink()
    (problem,) = _problems(folder)
    _, shot, face = parse_photo_name(drop)
    assert f"no {face} photo for cert 111 {shot}" in problem
    assert problem.startswith(f"111_{shot}_")  # names the photo that is there


@pytest.mark.parametrize("grade", ["0", "11", "8.5", "nine", ""])
def test_grade_outside_range(tmp_path, grade):
    folder = _make_set(tmp_path, cards=f"cert,grade\n111,{grade}\n222,10\n")
    (problem,) = _problems(folder)
    assert problem.startswith("cards.csv:2:") and "111" in problem and "grade" in problem


def test_duplicate_cert(tmp_path):
    folder = _make_set(tmp_path, cards="cert,grade\n111,9\n222,10\n111,8\n")
    (problem,) = _problems(folder)
    assert problem == "cards.csv:4: duplicate cert 111"


def test_duplicate_photo(tmp_path):
    folder = _make_set(tmp_path)
    (folder / "photos" / "111_careful_front.png").write_bytes(_image_bytes("PNG"))
    (problem,) = _problems(folder)
    assert "duplicate front photo for cert 111 careful" in problem


def test_heic_extension_is_an_error(tmp_path):
    folder = _make_set(tmp_path)
    (folder / "photos" / "222_casual_front.heic").write_bytes(_HEIC)
    (folder / "photos" / "222_casual_back.HEIC").write_bytes(_HEIC)
    problems = _problems(folder)
    assert len(problems) == 2
    assert all("not accepted" in p and ".heic" in p.lower() for p in problems)
    assert problems[0].startswith("222_casual_")


@pytest.mark.parametrize("content", [_HEIC, b"not an image", b""], ids=["heic", "junk", "empty"])
def test_unreadable_file_is_an_error(tmp_path, content):
    folder = _make_set(tmp_path)
    (folder / "photos" / "111_casual_back.png").write_bytes(content)
    (problem,) = _problems(folder)
    assert problem.startswith("111_casual_back.png:") and "not a readable JPEG/PNG/WebP" in problem


def test_other_pil_format_is_an_error(tmp_path):
    """A TIFF/BMP PIL could open is still refused, like the API refuses it."""
    folder = _make_set(tmp_path)
    (folder / "photos" / "111_casual_back.png").write_bytes(_image_bytes("BMP"))
    (problem,) = _problems(folder)
    assert "not a readable JPEG/PNG/WebP" in problem


def test_every_problem_listed_at_once(tmp_path):
    folder = _make_set(tmp_path, cards="cert,grade\n111,12\n222,10\n222,9\n")
    (folder / "photos" / "IMG_1.jpg").write_bytes(_image_bytes())
    (folder / "photos" / "777_casual_front.jpg").write_bytes(b"junk")
    problems = _problems(folder)
    text = "\n".join(problems)
    for needle in ("cards.csv:2:", "cards.csv:4: duplicate cert 222", "IMG_1.jpg:",
                   "777_casual_front.jpg: not a readable", "777_casual_front.jpg: cert 777 not in",
                   "no back photo for cert 777 casual"):
        assert needle in text, needle
    assert len(problems) == 6


def test_missing_cards_and_photos(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    problems = _problems(folder)
    assert any(p.startswith("cards.csv:") for p in problems)
    assert any(p.startswith("photos/:") for p in problems)


def test_cards_csv_missing_columns(tmp_path):
    folder = _make_set(tmp_path, cards="number,grade\n111,9\n")
    problems = _problems(folder)
    assert "cards.csv: missing columns ['cert']" in problems


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_build(tmp_path):
    folder = _make_set(tmp_path)
    db = _make_db(tmp_path / "db.sqlite", [])
    res = CliRunner().invoke(app, ["build", str(folder), "--db-path", str(db)])
    assert res.exit_code == 0, res.output
    assert "3 pairs, 2 cards" in res.output
    assert (folder / "manifest.csv").is_file()


def test_cli_build_fails_and_writes_nothing(tmp_path):
    folder = _make_set(tmp_path)
    (folder / "photos" / "x.heic").write_bytes(_HEIC)
    res = CliRunner().invoke(app, ["build", str(folder), "--db-path", str(tmp_path / "none.sqlite")])
    assert res.exit_code == 1
    assert "x.heic" in res.output and "manifest.csv not written" in res.output
    assert not (folder / "manifest.csv").exists()
