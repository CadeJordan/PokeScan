"""Build the manifest of the phone-photo evaluation set (intent/phone-photo-eval-set).

    python -m ml.eval.phone_set build data/phone_eval

The folder holds the owner's photos of their own slabs and a hand-written
`cards.csv` (`cert,grade[,notes]`, the cert and grade on the PSA label):

    photos/<cert>_<shot>_<face>.jpg    shot: careful | casual, face: front | back
    cards.csv
    manifest.csv                       written here

`build` pairs the photos by name, checks them, and writes `manifest.csv` with
columns `front,back,grade,id,shot,cert,in_db_split` for
`python -m ml.eval.evaluate --manifest`. Every problem is collected and reported
at once (naming the file), and then nothing is written. `in_db_split` is the
cert's split in the DB (train / val / test, absent, or unknown if the DB can't be read), looked up read-only, so
the report can keep leaked cards out of the headline number. Nothing but
`manifest.csv` is written.
"""

from __future__ import annotations

import csv
import io
import re
import sqlite3
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

import typer

from backend.app.ml.preprocess import ImageDecodeError, decode_image

SHOTS = ("careful", "casual")
FACES = ("front", "back")
EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
MANIFEST_COLUMNS = ("front", "back", "grade", "id", "shot", "cert", "in_db_split")
# Files a phone or OS drops next to photos; skipped rather than reported.
_IGNORED = {"thumbs.db", "desktop.ini"}
_NAME = re.compile(r"^(?P<cert>[0-9]+)_(?P<shot>[a-z]+)_(?P<face>[a-z]+)$", re.IGNORECASE | re.ASCII)


class PhoneSetError(ValueError):
    """The folder isn't a valid set; `problems` lists each one, file named."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__(f"{len(problems)} problem(s):\n" + "\n".join(f"  - {p}" for p in problems))


@dataclass(slots=True)
class BuildResult:
    rows: list[dict[str, str]]
    warnings: list[str] = field(default_factory=list)


def parse_photo_name(name: str) -> tuple[str, str, str] | None:
    """`(cert, shot, face)` from `<cert>_<shot>_<face>.<ext>`, any case; None if it doesn't parse."""
    p = Path(name)
    m = _NAME.match(p.stem)
    if not m or p.suffix.lower() not in EXTENSIONS:
        return None
    shot, face = m["shot"].lower(), m["face"].lower()
    if shot not in SHOTS or face not in FACES:
        return None
    return m["cert"], shot, face


def read_cards(path: Path) -> tuple[dict[str, int | None], list[str]]:
    """`{cert: grade}` from cards.csv, plus the problems found in it.

    A cert whose grade is a problem maps to None, so its photos aren't also
    reported as "not in cards.csv".
    """
    problems: list[str] = []
    cards: dict[str, int | None] = {}
    if not path.is_file():
        return cards, [f"{path.name}: not found"]
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return cards, [f"{path.name}: not UTF-8; save it as \"CSV UTF-8\""]
    with io.StringIO(text, newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.strip().lower() for c in (reader.fieldnames or []) if c}
        missing = {"cert", "grade"} - cols
        if missing:
            return cards, [f"{path.name}: missing columns {sorted(missing)}"]
        for i, raw in enumerate(reader, start=2):
            row = {k.strip().lower(): (v or "").strip() for k, v in raw.items() if k}
            cert, grade_text = row.get("cert", ""), row.get("grade", "")
            if not cert and not grade_text:
                continue  # blank line
            where = f"{path.name}:{i}"
            if not (cert.isascii() and cert.isdigit()):
                problems.append(f"{where}: cert {cert!r} is not a number")
                continue
            if cert in cards:
                problems.append(f"{where}: duplicate cert {cert}")
                continue
            try:
                grade_f = float(grade_text)
            except ValueError:
                grade_f = float("nan")
            if not (1 <= grade_f <= 10 and grade_f == int(grade_f)):
                problems.append(f"{where}: cert {cert}: grade {grade_text!r} not a whole grade in 1..10")
                cards[cert] = None
                continue
            cards[cert] = int(grade_f)
    return cards, problems


def _check_image(path: Path) -> str | None:
    """None if the served decode opens it (JPEG/PNG/WebP), else the reason."""
    try:
        decode_image(path.read_bytes())
    except ImageDecodeError as exc:
        return f"not a readable JPEG/PNG/WebP ({exc})"
    except OSError as exc:  # the read itself
        return f"can't read ({exc})"
    return None


def lookup_splits(db_path: Path | None, certs: list[str]) -> tuple[dict[str, str], list[str]]:
    """Each cert's DB split (train / val / test / ...; "absent" when not in the DB).

    The DB is opened read-only (`mode=ro`). A cert in the DB with no split is
    "unassigned". A missing or unreadable DB makes every cert "unknown", with a
    warning: never "absent", which the report treats as unseen by training.
    """
    unknown = dict.fromkeys(certs, "unknown")
    if db_path is None or not Path(db_path).is_file():
        return unknown, [f"DB not found ({db_path}): in_db_split is 'unknown' for every cert"]
    out = dict.fromkeys(certs, "absent")
    uri = "file:" + urllib.parse.quote(Path(db_path).resolve().as_posix(), safe="/:") + "?mode=ro"
    conn = None
    try:
        conn = sqlite3.connect(uri, uri=True)
        for i in range(0, len(certs), 500):  # stay under SQLite's variable limit
            chunk = certs[i : i + 500]
            marks = ",".join("?" * len(chunk))
            for cert, split in conn.execute(
                f"SELECT cert_number, split FROM certs WHERE cert_number IN ({marks})", chunk
            ):
                out[cert] = split or "unassigned"
    except sqlite3.Error as exc:
        return unknown, [f"DB {db_path} unreadable ({exc}): in_db_split is 'unknown' for every cert"]
    finally:
        if conn is not None:
            conn.close()
    return out, []


def build_manifest(folder: Path, db_path: Path | None) -> BuildResult:
    """Pair and check the photos in `folder`; raises PhoneSetError listing every problem.

    Writes nothing: `write_manifest` does.
    """
    folder = Path(folder)
    cards, problems = read_cards(folder / "cards.csv")
    photos_dir = folder / "photos"
    # (cert, shot) -> {face: file}
    pairs: dict[tuple[str, str], dict[str, Path]] = {}
    if not photos_dir.is_dir():
        problems.append(f"{photos_dir.name}/: not found")
        files = []
    else:
        files = sorted(p for p in photos_dir.iterdir() if p.is_file())
    for p in files:
        if p.name.startswith(".") or p.name.lower() in _IGNORED:
            continue
        parsed = parse_photo_name(p.name)
        if parsed is None:
            if p.suffix.lower() not in EXTENSIONS and _NAME.match(p.stem):
                problems.append(f"{p.name}: format {p.suffix or '(none)'} not accepted "
                                f"(JPEG/PNG/WebP only: {', '.join(EXTENSIONS)})")
            else:
                problems.append(f"{p.name}: name doesn't parse as "
                                f"<cert>_<{'|'.join(SHOTS)}>_<{'|'.join(FACES)}>.jpg")
            continue
        cert, shot, face = parsed
        bad = _check_image(p)
        if bad:
            problems.append(f"{p.name}: {bad}")
        if cert not in cards:
            problems.append(f"{p.name}: cert {cert} not in cards.csv")
        faces = pairs.setdefault((cert, shot), {})
        if face in faces:
            problems.append(f"{p.name}: duplicate {face} photo for cert {cert} {shot} "
                            f"(also {faces[face].name})")
            continue
        faces[face] = p
    for (cert, shot), faces in sorted(pairs.items()):
        for face in FACES:
            if face not in faces:
                have = next(iter(faces.values()))
                problems.append(f"{have.name}: no {face} photo for cert {cert} {shot}")
    if problems:
        raise PhoneSetError(problems)

    warnings: list[str] = []
    for cert in sorted(cards):
        shots = [s for s in SHOTS if (cert, s) in pairs]
        if not shots:
            warnings.append(f"cards.csv: cert {cert} has no photos (left out)")
        elif len(shots) < len(SHOTS):
            warnings.append(f"cert {cert}: only the {shots[0]} pair")
    splits, db_warnings = lookup_splits(db_path, sorted({c for c, _ in pairs}))
    warnings += db_warnings
    rows = [
        {
            "front": faces["front"].relative_to(folder).as_posix(),
            "back": faces["back"].relative_to(folder).as_posix(),
            "grade": str(cards[cert]),
            "id": f"{cert}_{shot}",
            "shot": shot,
            "cert": cert,
            "in_db_split": splits[cert],
        }
        for (cert, shot), faces in sorted(pairs.items())
    ]
    return BuildResult(rows, warnings)


def write_manifest(folder: Path, rows: list[dict[str, str]]) -> Path:
    path = Path(folder) / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


app = typer.Typer(add_completion=False)


@app.callback()
def _cli() -> None:
    """Phone-photo evaluation set (intent/phone-photo-eval-set)."""


@app.command()
def build(
    folder: Path = typer.Argument(..., help="Folder with cards.csv and photos/."),
    db_path: Path | None = typer.Option(None, help="DB for the in_db_split lookup (default: settings)."),
) -> None:
    """Check the photos and write <folder>/manifest.csv."""
    from collections import Counter

    if db_path is None:
        from backend.app.core.config import get_settings

        db_path = get_settings().db_path
    try:
        result = build_manifest(folder, db_path)
    except PhoneSetError as exc:
        typer.echo(f"{folder}: {exc}\nmanifest.csv not written.", err=True)
        raise typer.Exit(1) from exc
    for w in result.warnings:
        typer.echo(f"warning: {w}", err=True)
    if not result.rows:
        typer.echo(f"{folder}: no photo pairs; manifest.csv not written.", err=True)
        raise typer.Exit(1)
    path = write_manifest(folder, result.rows)
    rows = result.rows
    typer.echo(f"wrote {path}: {len(rows)} pairs, {len({r['cert'] for r in rows})} cards")
    for col in ("shot", "in_db_split"):
        counts = Counter(r[col] for r in rows)
        typer.echo(f"  {col}: " + "  ".join(f"{k} {n}" for k, n in sorted(counts.items())))


if __name__ == "__main__":
    app()
