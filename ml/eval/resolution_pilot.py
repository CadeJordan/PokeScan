"""Full-resolution pilot: does more resolution separate PSA 9 from 10?

Spec: `intent/full-resolution-pilot/intent.md`. Three arms on the same certs
and card-grouped folds; only the corner and edge region features differ (CLS,
mean and centre are today's features in every arm):

- A: today's features (the probe's feature cache, read only).
- B: high-res tiles of the corner / edge regions, warped from the stored
  1024 px photo with the precrop quad.
- C: the same tiles, warped from the PSA original (re-fetched from the CDN)
  with the precrop quad scaled to the original's size.

    python -m ml.eval.resolution_pilot integrity --out data/images_full/integrity_before.json
    python -m ml.eval.resolution_pilot select            # pilot_certs.csv (DB read-only)
    python -m ml.eval.resolution_pilot step0             # 5 originals + header audit
    python -m ml.eval.resolution_pilot download          # the pilot originals + manifest.csv
    python -m ml.eval.resolution_pilot quads             # precrop quads of train+val photos
    python -m ml.eval.resolution_pilot extract --arm b   # tiles, all train+val
    python -m ml.eval.resolution_pilot extract --arm c   # tiles, pilot originals
    python -m ml.eval.resolution_pilot evaluate          # CV + bootstrap -> report.json

Everything is written under `data/images_full/` (gitignored) and the report
under `test-artifacts/eval/resolution_pilot/`. The DB is opened read-only and
nothing under `data/images`, `data/crops` or `data/models` is written.

Tile geometry (fixed in the spec before any CV ran): a region's card area is
exactly today's patch block (`features.region_means`), in normalised card
coordinates. Corner blocks become one square `tile` x `tile` tile; edge strips
are cut along their length into `round(length / tile)` equal pieces, each
resampled to `tile` (along) x `edge_depth` (deep). Every tile is sampled
straight from the source photo through the precrop quad's homography (one
resampling), after an INTER_AREA pre-shrink when the source is denser than
the tile canvas. A tile's feature is its mean DINOv2 patch token; a region's
is the mean over its tiles.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import logging
import math
import os
import sqlite3
import subprocess
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import cv2
import httpx
import numpy as np
import typer
from rich.logging import RichHandler

from backend.app.core.config import PROJECT_ROOT, get_settings
from backend.app.ml.features import (
    BACKBONE,
    CORNER_PATCHES,
    CORNER_SLICE,
    EDGE_PATCHES,
    EDGE_SLICE,
    FEAT_H,
    FEAT_W,
    GRID_H,
    GRID_W,
    PATCH,
    REGION_NAMES,
)
from backend.app.ml.preprocess import CARD_H, CARD_W, decode_image, detect_and_crop
from ml.data.groups import card_group_key

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True)])
log = logging.getLogger("resolution_pilot")

app = typer.Typer(add_completion=False)

PILOT_DIR = PROJECT_ROOT / "data" / "images_full"
REPORT_PATH = PROJECT_ROOT / "test-artifacts" / "eval" / "resolution_pilot" / "report.json"
TRAINVAL = ("train", "val")
FACES = ("front", "back")
# Folders under data/ the pilot must never write to.
PROTECTED_DIRS = ("images", "crops", "models")
TILE_REGIONS = REGION_NAMES[CORNER_SLICE.start : EDGE_SLICE.stop]
STORED_LONG_EDGE = 1024  # `ml.data.image_io.IMAGE_MAX_LONG_EDGE`
# Data-quality gates (fixed before any CV): mean absolute difference in grey
# levels between (a) today's re-warp of the stored photo and the stored precrop
# JPEG, (b) the original shrunk to the stored size and the stored photo.
MAX_PRECROP_MAD = 8.0
MAX_ORIGINAL_MAD = 8.0
SENSITIVE_HEADERS = ("authorization", "proxy-authorization", "cookie", "x-api-key")
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # as `features.to_input`
_STD = np.array([0.229, 0.224, 0.225], np.float32)


# --------------------------------------------------------------------------- #
# Paths, safety, small helpers
# --------------------------------------------------------------------------- #
def _within(path: Path, folder: Path) -> bool:
    """Whether `path` is `folder` or lies inside it, by path or by file identity.

    Identity (`os.path.samefile` on each existing ancestor) catches spellings
    that resolve to a different string: verbatim (long-path) and UNC forms, or a
    trailing-space folder name Windows strips.
    """
    if path == folder or folder in path.parents:
        return True
    if not folder.exists():
        return False
    for p in (path, *path.parents):
        try:
            if p.exists() and os.path.samefile(p, folder):
                return True
        except (OSError, ValueError):
            continue
    return False


def check_output_dir(path: Path, data_dir: Path) -> Path:
    """`path`, resolved, unless it lies inside the data folder outside `images_full/`.

    Both paths are resolved first (`..`, relative paths, links; Windows paths
    compare case-insensitively) and then compared by file identity too, so the
    DB, `backups/`, any file directly in the data folder and the protected
    folders are refused however the path is spelled.
    """
    out = Path(path).resolve()
    data = Path(data_dir).resolve()
    allowed = data / PILOT_DIR.name
    if _within(out, data) and not _within(out, allowed):
        raise ValueError(f"refusing to write {out}: only {allowed} may be written under {data}")
    return out


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_jsonable))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def record_timing(pilot_dir: Path, name: str, seconds: float, **extra) -> None:
    path = pilot_dir / "timings.json"
    t = read_json(path, {})
    t[name] = {"seconds": round(seconds, 1), "at": _now(), **extra}
    write_json(path, t)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _file_sig(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def integrity_snapshot(data_dir: Path, db_path: Path) -> dict:
    """sha256 / size / mtime of the DB and file counts + bytes of the protected folders."""
    st = db_path.stat()
    out: dict = {
        "at": _now(),
        "db": {
            "path": str(db_path),
            "bytes": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "sha256": sha256_file(db_path),
        },
        "dirs": {},
    }
    for name in PROTECTED_DIRS:
        files = [p for p in (data_dir / name).rglob("*") if p.is_file()]
        out["dirs"][name] = {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}
    return out


# --------------------------------------------------------------------------- #
# Pilot selection (DB read-only, train+val only)
# --------------------------------------------------------------------------- #
CANDIDATE_SQL = (
    "SELECT cert_number, grade_int, split, brand_title, subject, card_number, "
    "front_path, back_path, cropped_front_path, cropped_back_path, front_url, back_url, "
    "crop_kind_front, crop_kind_back FROM certs "
    "WHERE is_pokemon=1 AND has_images=1 AND grade_int IN (9, 10) "
    "AND split IN ('train', 'val') "
    "AND crop_found_front=1 AND crop_found_back=1 "
    "AND cropped_front_path IS NOT NULL AND cropped_back_path IS NOT NULL "
    "AND front_url IS NOT NULL AND back_url IS NOT NULL "
    "AND front_url != '' AND back_url != '' "
    "ORDER BY cert_number"
)


def connect_ro(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def load_candidates(db_path: Path) -> list[dict]:
    """PSA 9 / 10 train+val certs with both original URLs and a usable precrop."""
    conn = connect_ro(db_path)
    try:
        rows = [dict(r) for r in conn.execute(CANDIDATE_SQL)]
    finally:
        conn.close()
    return [r for r in rows if r["split"] in TRAINVAL]


def group_of(row: dict) -> str:
    return card_group_key(row["brand_title"] or "", row["subject"] or "", row["card_number"] or "")


def select_pilot(rows: Sequence[dict], n_per_grade: int = 300, seed: int = 0) -> list[dict]:
    """`n_per_grade` PSA 9s and 10s from train+val, spread over card groups.

    Groups are visited in a seeded random order taking one cert each, then a
    second cert from groups that have one, and so on, so a card repeats only
    once every group has been used.
    """
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for grade in (9, 10):
        by_group: dict[str, list[dict]] = {}
        for r in rows:
            if r["grade_int"] == grade and r["split"] in TRAINVAL:
                by_group.setdefault(group_of(r), []).append(r)
        keys = sorted(by_group)
        queues = []
        for i in rng.permutation(len(keys)):
            q = sorted(by_group[keys[i]], key=lambda r: r["cert_number"])
            queues.append([q[j] for j in rng.permutation(len(q))])
        picked: list[dict] = []
        depth = 0
        while len(picked) < n_per_grade and any(len(q) > depth for q in queues):
            for q in queues:
                if len(q) > depth:
                    picked.append(q[depth])
                    if len(picked) == n_per_grade:
                        break
            depth += 1
        if len(picked) < n_per_grade:
            log.warning("grade %d: only %d candidates", grade, len(picked))
        out.extend(picked)
    return sorted(out, key=lambda r: r["cert_number"])


SELECTION_FIELDS = ("cert_number", "grade_int", "split", "group")


def write_selection(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, SELECTION_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: r[k] for k in SELECTION_FIELDS[:3]}, "group": group_of(r)})


def read_selection(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return [r["cert_number"] for r in csv.DictReader(f)]


def pilot_rows(db_path: Path, pilot_dir: Path) -> list[dict]:
    """The selected certs' current DB rows; refuses any cert no longer in train+val."""
    certs = read_selection(pilot_dir / "pilot_certs.csv")
    by_cert = {r["cert_number"]: r for r in load_candidates(db_path)}
    missing = [c for c in certs if c not in by_cert]
    if missing:
        raise RuntimeError(f"{len(missing)} pilot certs are no longer train+val candidates: {missing[:5]}")
    return [by_cert[c] for c in certs]


# --------------------------------------------------------------------------- #
# Download (unauthenticated CDN GETs) + manifest
# --------------------------------------------------------------------------- #
MANIFEST_FIELDS = (
    "cert",
    "face",
    "url",
    "file",
    "bytes",
    "sha256",
    "width",
    "height",
    "exif_orientation",
    "status",
    "reason",
    "fetched_at",
)


def read_manifest(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["cert"], r["face"]): r for r in csv.DictReader(f)}


def write_manifest(path: Path, entries: dict[tuple[str, str], dict]) -> None:
    """Atomic rewrite, rows sorted by (cert, face)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, MANIFEST_FIELDS)
        w.writeheader()
        for key in sorted(entries):
            w.writerow({k: entries[key].get(k, "") for k in MANIFEST_FIELDS})
    os.replace(tmp, path)


def cdn_client(transport: httpx.BaseTransport | None = None, timeout: float = 60.0) -> httpx.Client:
    """A bare client like `PSAClient.download_image_bytes` uses: no token, no headers."""
    return httpx.Client(timeout=timeout, follow_redirects=True, transport=transport)


def request_audit(request: httpx.Request) -> dict:
    headers = {k.lower(): v for k, v in request.headers.items()}
    return {
        "url_host": request.url.host,
        "url_has_query": bool(request.url.query),
        "headers": headers,
        "sensitive": sorted(h for h in headers if h in SENSITIVE_HEADERS),
    }


@dataclass
class Fetch:
    data: bytes | None
    reason: str
    status_code: int | None
    attempts: int
    request: dict | None
    response_headers: dict | None


def fetch(
    client: httpx.Client,
    url: str,
    retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    max_redirects: int = 5,
) -> Fetch:
    """GET `url`; retries transport errors, 429 and 5xx with backoff.

    The pilot must never send a token to the CDN. A client that could add
    headers at send time (auth, request hooks, a cookie jar) is refused up
    front. Each hop, redirects included, is built and audited before it is
    sent with no auth and no redirect-following, and raises if it would carry
    an auth-like header. Cookies the CDN sets are dropped after every hop.
    """
    if client.auth is not None or client.event_hooks.get("request") or len(client.cookies):
        raise RuntimeError(f"refusing to fetch {url}: the client has auth, request hooks or cookies")
    last = Fetch(None, "not attempted", None, 0, None, None)
    for attempt in range(1, retries + 1):
        target = url
        try:
            for _ in range(max_redirects + 1):
                req = client.build_request("GET", target)
                audit = request_audit(req)
                if audit["sensitive"]:
                    raise RuntimeError(f"request to {target} would carry {audit['sensitive']}")
                resp = client.send(req, auth=None, follow_redirects=False)
                client.cookies.clear()
                if not (resp.is_redirect and "Location" in resp.headers):
                    break
                target = str(req.url.join(resp.headers["Location"]))
        except httpx.HTTPError as exc:
            last = Fetch(None, f"transport: {type(exc).__name__}: {exc}", None, attempt, None, None)
            if attempt < retries:
                sleep(2.0 * attempt)
            continue
        rh = {k.lower(): v for k, v in resp.headers.items()}
        if resp.status_code == 200:
            return Fetch(resp.content, "", 200, attempt, audit, rh)
        last = Fetch(None, f"http {resp.status_code}", resp.status_code, attempt, audit, rh)
        if (resp.status_code == 429 or resp.status_code >= 500) and attempt < retries:
            try:
                wait = float(resp.headers.get("Retry-After", 2.0 * attempt))
            except ValueError:
                wait = 2.0 * attempt
            sleep(min(wait, 30.0))
            continue
        break
    return last


def image_info(data: bytes) -> dict:
    """Pixel size after EXIF orientation (the frame ingest stored), format, orientation."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        im.load()
        try:
            orientation = int(im.getexif().get(0x0112, 1) or 1)
        except Exception:  # noqa: BLE001 - unreadable EXIF: upright, as `decode_image`
            orientation = 1
        w, h = im.size
        if orientation in (5, 6, 7, 8):
            w, h = h, w
        return {"width": w, "height": h, "format": im.format, "exif_orientation": orientation}


def original_name(cert: str, face: str) -> str:
    return f"{cert}_{face}.jpg"


def download_plan(rows: Iterable[dict]) -> list[tuple[str, str, str]]:
    """(cert, face, url) for both faces of each row, train+val only."""
    plan = []
    for r in rows:
        if r["split"] not in TRAINVAL:
            raise ValueError(f"cert {r['cert_number']} is in split {r['split']!r}")
        plan.append((r["cert_number"], "front", r["front_url"]))
        plan.append((r["cert_number"], "back", r["back_url"]))
    return plan


def download_originals(
    plan: Sequence[tuple[str, str, str]],
    out_dir: Path,
    client: httpx.Client,
    rate: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    on_fetch: Callable[[tuple[str, str, str], Fetch], None] | None = None,
) -> dict:
    """Fetch each (cert, face, url) into `out_dir`, keeping `out_dir/manifest.csv`.

    An entry whose manifest row is ok and whose file still has that sha256 is
    skipped (no request); a file without a manifest row (killed between the
    write and the manifest) is re-hashed and recorded. Failed entries are
    retried on the next run. Requests start at most `rate` per second.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    entries = read_manifest(manifest_path)
    stats = {"planned": len(plan), "fetched": 0, "skipped": 0, "recovered": 0, "failed": 0, "bytes": 0}
    interval = 1.0 / rate if rate > 0 else 0.0
    last_start: float | None = None
    for item in plan:
        cert, face, url = item
        dest = out_dir / original_name(cert, face)
        row = entries.get((cert, face))
        row_ok = bool(row and row["status"] == "ok" and row["url"] == url)
        if dest.exists() and row_ok and row["sha256"] == sha256_file(dest):
            stats["skipped"] += 1
            continue
        if dest.exists() and not row_ok:
            data = dest.read_bytes()
            try:
                info = image_info(data)
            except Exception:  # noqa: BLE001 - unreadable leftover: fetch again
                info = None
            if info is not None:
                entries[(cert, face)] = _ok_row(cert, face, url, dest, data, info)
                write_manifest(manifest_path, entries)
                stats["recovered"] += 1
                continue
        if last_start is not None and interval:
            wait = last_start + interval - clock()
            if wait > 0:
                sleep(wait)
        last_start = clock()
        res = fetch(client, url, sleep=sleep)
        if on_fetch is not None:
            on_fetch(item, res)
        info = None
        reason = res.reason
        if res.data is not None:
            try:
                info = image_info(res.data)
            except Exception as exc:  # noqa: BLE001
                reason = f"undecodable: {type(exc).__name__}: {exc}"
        if info is None:
            entries[(cert, face)] = {
                "cert": cert,
                "face": face,
                "url": url,
                "file": "",
                "bytes": len(res.data) if res.data else "",
                "status": "failed",
                "reason": reason,
                "fetched_at": _now(),
            }
            stats["failed"] += 1
            log.warning("%s %s failed: %s", cert, face, reason)
        else:
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(res.data)
            os.replace(tmp, dest)
            entries[(cert, face)] = _ok_row(cert, face, url, dest, res.data, info)
            stats["fetched"] += 1
            stats["bytes"] += len(res.data)
        write_manifest(manifest_path, entries)
    return stats


def _ok_row(cert: str, face: str, url: str, dest: Path, data: bytes, info: dict) -> dict:
    return {
        "cert": cert,
        "face": face,
        "url": url,
        "file": dest.name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "width": info["width"],
        "height": info["height"],
        "exif_orientation": info["exif_orientation"],
        "status": "ok",
        "reason": "",
        "fetched_at": _now(),
    }


def manifest_summary(entries: dict[tuple[str, str], dict]) -> dict:
    ok = [r for r in entries.values() if r["status"] == "ok"]
    failed = [r for r in entries.values() if r["status"] != "ok"]
    sizes = np.array([int(r["bytes"]) for r in ok], float)
    longs = np.array([max(int(r["width"]), int(r["height"])) for r in ok], float)
    out = {
        "files_ok": len(ok),
        "files_failed": len(failed),
        "failed": [{k: r[k] for k in ("cert", "face", "reason")} for r in failed],
        "total_bytes": int(sizes.sum()),
    }
    if len(ok):
        out["bytes_median"] = float(np.median(sizes))
        out["bytes_p5_p95"] = [float(x) for x in np.percentile(sizes, [5, 95])]
        out["long_edge_median"] = float(np.median(longs))
        out["long_edge_min_max"] = [float(longs.min()), float(longs.max())]
        out["sizes_px"] = _counts(f"{r['width']}x{r['height']}" for r in ok)
    return out


def _counts(values: Iterable) -> dict:
    out: dict = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# --------------------------------------------------------------------------- #
# Tile geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Box:
    """Normalised card rectangle: u left->right, v top->bottom, both in [0, 1]."""

    u0: float
    u1: float
    v0: float
    v1: float


@dataclass(frozen=True)
class Tile:
    region: str
    box: Box
    out_w: int
    out_h: int


@dataclass(frozen=True)
class TileSpec:
    """`tile`: side of a corner tile in px (a multiple of the 14 px patch)."""

    tile: int = 224

    def __post_init__(self) -> None:
        if self.tile % PATCH or self.tile < 2 * PATCH:
            raise ValueError(f"tile must be a multiple of {PATCH} >= {2 * PATCH}")

    @property
    def px_per_feat_px(self) -> float:
        """Tile px per px of today's FEAT_W x FEAT_H feature input."""
        return self.tile / (CORNER_PATCHES * PATCH)

    @property
    def edge_depth(self) -> int:
        """Edge tile depth: the strip's depth at the corner scale, rounded to whole patches."""
        return PATCH * max(1, round(EDGE_PATCHES * self.px_per_feat_px))

    @property
    def canvas(self) -> tuple[float, float]:
        """(W, H) of the whole card at the tile scale."""
        return FEAT_W * self.px_per_feat_px, FEAT_H * self.px_per_feat_px

    def to_dict(self) -> dict:
        tiles = region_tiles(self)
        return {
            "tile": self.tile,
            "edge_depth": self.edge_depth,
            "px_per_feature_px": self.px_per_feat_px,
            "canvas_wh": [round(x, 1) for x in self.canvas],
            "tiles_per_region": {r: sum(t.region == r for t in tiles) for r in TILE_REGIONS},
            "tile_shapes_hw": sorted({(t.out_h, t.out_w) for t in tiles}),
            "backbone": BACKBONE,
            "interp": "INTER_LINEAR warp, INTER_AREA pre-shrink",
            "version": 1,
        }

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def region_cells() -> dict[str, tuple[int, int, int, int]]:
    """(row0, row1, col0, col1) patch blocks, half-open, exactly as `features.region_means`."""
    k, e, gh, gw = CORNER_PATCHES, EDGE_PATCHES, GRID_H, GRID_W
    return {
        "corner_tl": (0, k, 0, k),
        "corner_tr": (0, k, gw - k, gw),
        "corner_br": (gh - k, gh, gw - k, gw),
        "corner_bl": (gh - k, gh, 0, k),
        "edge_top": (0, e, k, gw - k),
        "edge_right": (k, gh - k, gw - e, gw),
        "edge_bottom": (gh - e, gh, k, gw - k),
        "edge_left": (k, gh - k, 0, e),
    }


def region_boxes() -> dict[str, Box]:
    """The same blocks as card fractions (the patch grid tiles FEAT_W x FEAT_H exactly)."""
    assert GRID_W * PATCH == FEAT_W and GRID_H * PATCH == FEAT_H
    return {
        name: Box(c0 / GRID_W, c1 / GRID_W, r0 / GRID_H, r1 / GRID_H)
        for name, (r0, r1, c0, c1) in region_cells().items()
    }


def region_tiles(spec: TileSpec) -> list[Tile]:
    f, t, d = spec.px_per_feat_px, spec.tile, spec.edge_depth
    tiles: list[Tile] = []
    for name, b in region_boxes().items():
        if name.startswith("corner"):
            tiles.append(Tile(name, b, t, t))
        elif name in ("edge_top", "edge_bottom"):
            n = max(1, round((b.u1 - b.u0) * FEAT_W * f / t))
            for i in range(n):
                u0 = b.u0 + (b.u1 - b.u0) * i / n
                u1 = b.u0 + (b.u1 - b.u0) * (i + 1) / n
                tiles.append(Tile(name, Box(u0, u1, b.v0, b.v1), t, d))
        else:
            n = max(1, round((b.v1 - b.v0) * FEAT_H * f / t))
            for i in range(n):
                v0 = b.v0 + (b.v1 - b.v0) * i / n
                v1 = b.v0 + (b.v1 - b.v0) * (i + 1) / n
                tiles.append(Tile(name, Box(b.u0, b.u1, v0, v1), d, t))
    return tiles


def crop_to_source(quad: np.ndarray) -> np.ndarray:
    """3x3 map from 600x840 crop pixels to source pixels: the inverse of the precrop warp."""
    dst = np.array([[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]], np.float32)
    return cv2.getPerspectiveTransform(dst, np.asarray(quad, np.float32).reshape(4, 2))


def tile_to_source(quad: np.ndarray, box: Box, out_w: int, out_h: int) -> np.ndarray:
    """3x3 map from tile pixels to source pixels.

    Card fraction u covers crop pixel x in [u*CARD_W - 0.5, ...]: pixel 0's
    centre sits on the quad corner, as in the precrop warp, and the INTER_AREA
    resize to FEAT_W keeps fractions of the pixel extent.
    """
    sx = (box.u1 - box.u0) * CARD_W / out_w
    sy = (box.v1 - box.v0) * CARD_H / out_h
    a = np.array(
        [
            [sx, 0.0, box.u0 * CARD_W - 0.5 + 0.5 * sx],
            [0.0, sy, box.v0 * CARD_H - 0.5 + 0.5 * sy],
            [0.0, 0.0, 1.0],
        ]
    )
    return crop_to_source(quad).astype(np.float64) @ a


def sample_tile(src: np.ndarray, quad: np.ndarray, tile: Tile) -> np.ndarray:
    m = tile_to_source(quad, tile.box, tile.out_w, tile.out_h)
    return cv2.warpPerspective(
        src, m, (tile.out_w, tile.out_h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    )


def scale_quad(quad: np.ndarray, from_wh: tuple[int, int], to_wh: tuple[int, int]) -> np.ndarray:
    """Quad in an image of size `from_wh` -> the same points in a resize to `to_wh`.

    Resizes map pixel extents, so x + 0.5 scales (pixel centres are offset by half a pixel).
    """
    s = np.array([to_wh[0] / from_wh[0], to_wh[1] / from_wh[1]], np.float64)
    return ((np.asarray(quad, np.float64).reshape(4, 2) + 0.5) * s - 0.5).astype(np.float32)


def quad_height(quad: np.ndarray) -> float:
    q = np.asarray(quad, np.float64).reshape(4, 2)
    return float((np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2)


def prepare_source(src: np.ndarray, quad: np.ndarray, spec: TileSpec) -> tuple[np.ndarray, np.ndarray]:
    """Shrink (INTER_AREA) a source denser than the tile canvas so the warp never aliases."""
    ratio = quad_height(quad) / spec.canvas[1]
    if ratio <= 1.0:
        return src, np.asarray(quad, np.float32)
    h, w = src.shape[:2]
    nw, nh = max(1, round(w / ratio)), max(1, round(h / ratio))
    small = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
    return small, scale_quad(quad, (w, h), (nw, nh))


def face_tiles(src: np.ndarray, quad: np.ndarray, spec: TileSpec) -> list[tuple[int, np.ndarray]]:
    """(index into TILE_REGIONS, uint8 RGB tile) for every tile of one face."""
    src, quad = prepare_source(src, quad, spec)
    index = {r: i for i, r in enumerate(TILE_REGIONS)}
    return [(index[t.region], sample_tile(src, quad, t)) for t in region_tiles(spec)]


def suggest_tile(card_height_px: float) -> int:
    """224 or 448, whichever is closer (in log scale) to a corner block's size in source px."""
    corner = card_height_px * CORNER_PATCHES / GRID_H
    return min((224, 448), key=lambda t: abs(math.log(corner / t)))


# --------------------------------------------------------------------------- #
# Sources: stored photos (+ precrop quad), originals (+ scaled quad)
# --------------------------------------------------------------------------- #
def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def stored_rgb(photo: Path) -> np.ndarray:
    """The stored photo in the frame `detect_and_crop` saw (EXIF + cap; a no-op at 1024 px)."""
    bgr, _ = decode_image(photo.read_bytes())
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def original_rgb(path: Path) -> np.ndarray:
    """A downloaded original, upright per EXIF and RGB, as ingest saw it before the resize."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im.load()
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")
        return np.array(im)


def stored_quad(photo: Path, crop: Path) -> dict:
    """Re-detect the precrop quad on the stored photo and check it against the stored crop."""
    c = detect_and_crop(photo.read_bytes())
    h, w = c.source_bgr.shape[:2]
    rec: dict = {
        "sig": _file_sig(photo),
        "found": bool(c.found_quad),
        "rotation": int(c.rotation),
        "src_wh": [int(w), int(h)],
        "quad": c.quad.astype(float).round(3).tolist() if c.found_quad else None,
        "precrop_mad": None,
    }
    if crop.exists():
        ref = read_rgb(crop)
        if ref.shape == c.image.shape:
            rec["precrop_mad"] = float(cv2.absdiff(ref, c.image).mean())
    return rec


def quad_problem(rec: dict | None, max_mad: float = MAX_PRECROP_MAD) -> str | None:
    if rec is None:
        return "no_quad_record"
    if not rec["found"]:
        return "no_quad"
    if rec["rotation"]:
        return "rotated"
    if rec["precrop_mad"] is None:
        return "precrop_unreadable"
    if rec["precrop_mad"] > max_mad:
        return "precrop_mismatch"
    return None


def original_check(orig: np.ndarray, stored: np.ndarray, max_mad: float = MAX_ORIGINAL_MAD) -> dict:
    """Is `orig` the image the stored 1024 px photo was made from (same frame)?"""
    oh, ow = orig.shape[:2]
    sh, sw = stored.shape[:2]
    out: dict = {"orig_wh": [ow, oh], "stored_wh": [sw, sh], "mad": None, "problem": None}
    if max(ow, oh) <= max(sw, sh):
        out["problem"] = "not_larger"
        return out
    s = STORED_LONG_EDGE / max(ow, oh)
    if abs(round(ow * s) - sw) > 1 or abs(round(oh * s) - sh) > 1:
        out["problem"] = "size_mismatch"
        return out
    small = cv2.resize(orig, (sw, sh), interpolation=cv2.INTER_AREA)
    g1 = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(stored, cv2.COLOR_RGB2GRAY)
    out["mad"] = float(cv2.absdiff(g1, g2).mean())
    if out["mad"] > max_mad:
        out["problem"] = "original_mismatch"
    return out


def compute_quads(
    items: Sequence[tuple[Path, Path]],
    data_dir: Path,
    cache_path: Path,
    workers: int = 8,
    save_every: int = 200,
) -> dict[str, dict]:
    """Quad records keyed by the photo's path relative to `data_dir`, cached by file signature."""
    cache = read_json(cache_path, {})
    todo = []
    for photo, crop in items:
        key = photo.relative_to(data_dir).as_posix()
        if key in cache and cache[key]["sig"] == _file_sig(photo):
            continue
        todo.append((key, photo, crop))
    log.info("quads: %d cached, %d to detect", len(items) - len(todo), len(todo))
    if todo:
        from tqdm import tqdm

        with ThreadPoolExecutor(workers) as pool:
            futures = [(key, pool.submit(stored_quad, photo, crop)) for key, photo, crop in todo]
            for i, (key, fut) in enumerate(tqdm(futures, desc="quads")):
                cache[key] = fut.result()
                if (i + 1) % save_every == 0:
                    write_json(cache_path, cache)
        write_json(cache_path, cache)
    return cache


# --------------------------------------------------------------------------- #
# Tile features
# --------------------------------------------------------------------------- #
def build_tile_model(pretrained: bool = True):
    """torch module: (B, 3, H, W) normalised tiles, any multiple of 14 -> (B, D) mean patch token."""
    import timm
    import torch

    class TileModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = timm.create_model(
                BACKBONE, pretrained=pretrained, num_classes=0, dynamic_img_size=True
            )

        def forward(self, x):
            tokens = self.backbone.forward_features(x)
            return tokens[:, self.backbone.num_prefix_tokens :].mean(1)

    return TileModel().eval()


def tiles_to_tensor(tiles: Sequence[np.ndarray], device):
    """uint8 RGB (H, W, 3) tiles -> normalised (B, 3, H, W) float32, as `features.to_input`."""
    import torch

    x = torch.from_numpy(np.stack(tiles)).to(device).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.from_numpy(_MEAN).to(device)[None, :, None, None]
    std = torch.from_numpy(_STD).to(device)[None, :, None, None]
    return ((x - mean) / std).contiguous()


def run_tiles(model, device, tiles: Sequence[np.ndarray], max_tokens: int = 32768) -> np.ndarray:
    """(n, D) mean patch tokens for same-shaped tiles, batched by token count."""
    import torch

    h, w = tiles[0].shape[:2]
    per = (h // PATCH) * (w // PATCH) + 5
    bs = max(1, max_tokens // per)
    out = []
    with torch.inference_mode():
        for s in range(0, len(tiles), bs):
            out.append(model(tiles_to_tensor(tiles[s : s + bs], device)).float().cpu().numpy())
    return np.concatenate(out)


def region_features(
    faces: Sequence[list[tuple[int, np.ndarray]]], model, device, max_tokens: int = 32768
) -> np.ndarray:
    """(n_faces, len(TILE_REGIONS), D): per face, the mean tile feature of each region."""
    by_shape: dict[tuple[int, int], list[tuple[int, int, np.ndarray]]] = {}
    for fi, tiles in enumerate(faces):
        for ri, t in tiles:
            by_shape.setdefault(t.shape[:2], []).append((fi, ri, t))
    sums: np.ndarray | None = None
    counts = np.zeros((len(faces), len(TILE_REGIONS)))
    for group in by_shape.values():
        feats = run_tiles(model, device, [t for _, _, t in group], max_tokens)
        if sums is None:
            sums = np.zeros((len(faces), len(TILE_REGIONS), feats.shape[1]))
        for (fi, ri, _), v in zip(group, feats, strict=True):
            sums[fi, ri] += v
            counts[fi, ri] += 1
    if sums is None or (counts == 0).any():
        raise ValueError("every face needs at least one tile per region")
    return (sums / counts[..., None]).astype(np.float32)


# One cert's job: cache key + a loader returning [(rgb, quad) front, (rgb, quad) back].
TileJob = tuple[tuple[str, ...], Callable[[], list[tuple[np.ndarray, np.ndarray]]]]


def extract_tile_features(
    jobs: Sequence[TileJob],
    spec: TileSpec,
    cache: Path,
    model_factory=build_tile_model,
    chunk: int = 16,
    workers: int = 4,
    max_tokens: int = 32768,
    save_every: int = 20,
) -> np.ndarray:
    """(N, 2, len(TILE_REGIONS), D) float32, reusing cached certs (key + tile spec hash)."""
    import torch

    spec_hash = spec.hash()
    cached: dict[tuple[str, ...], np.ndarray] = {}
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            if str(z["tile_spec_hash"]) == spec_hash:
                cached = dict(zip(map(tuple, z["keys"]), z["feats"], strict=True))
            else:
                log.info("tile cache %s is from another tile spec: re-extracting", cache)
    keys = [tuple(k) for k, _ in jobs]
    todo = [i for i, k in enumerate(keys) if k not in cached]
    log.info("tiles %s: %d cached, %d to extract", cache.name, len(keys) - len(todo), len(todo))

    def save() -> None:
        have = [k for k in keys if k in cached]
        if not have:
            return
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.stem + ".tmp.npz")
        np.savez(
            tmp,
            tile_spec_hash=np.array(spec_hash),
            tile_spec=np.array(json.dumps(spec.to_dict())),
            keys=np.array(have),
            feats=np.stack([cached[k] for k in have]),
        )
        os.replace(tmp, cache)

    if todo:
        from tqdm import tqdm

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model_factory().to(device)

        def load(i: int) -> list[list[tuple[int, np.ndarray]]]:
            return [face_tiles(rgb, quad, spec) for rgb, quad in jobs[i][1]()]

        # Certs load in threads while the GPU runs; at most 2 chunks are held.
        with ThreadPoolExecutor(max(1, workers)) as pool:
            pending: deque = deque()
            queue = iter(todo)
            for i in islice(queue, 2 * chunk):
                pending.append((i, pool.submit(load, i)))
            bar = tqdm(total=len(todo), desc="tiles")
            batches = 0
            while pending:
                idx, faces = [], []
                while pending and len(idx) < chunk:
                    i, fut = pending.popleft()
                    idx.append(i)
                    faces.extend(fut.result())
                    nxt = next(queue, None)
                    if nxt is not None:
                        pending.append((nxt, pool.submit(load, nxt)))
                feats = region_features(faces, model, device, max_tokens)
                for j, i in enumerate(idx):
                    cached[keys[i]] = feats[2 * j : 2 * j + 2].astype(np.float16)
                bar.update(len(idx))
                batches += 1
                if batches % save_every == 0:
                    save()
            bar.close()
        save()
    return np.stack([cached[k] for k in keys]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Evaluation: CV per arm, metrics, paired card-group bootstrap, decision
# --------------------------------------------------------------------------- #
def auc(pos: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney rank sum (ties averaged), nan with one class."""
    from scipy.stats import rankdata

    pos = np.asarray(pos, bool)
    n1 = int(pos.sum())
    n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def with_tiles(a_feats: np.ndarray, tiles: np.ndarray) -> np.ndarray:
    """Arm A's features with the corner + edge regions replaced by tile features."""
    out = a_feats.copy()
    out[:, :, CORNER_SLICE.start : EDGE_SLICE.stop] = tiles
    return out


def arm_predictions(cum: np.ndarray, regions: np.ndarray) -> dict[str, np.ndarray]:
    from backend.app.ml.grade_head import PROBE_NAMES, decode

    decoded = [decode(c) for c in cum]
    return {
        "soft": np.array([d[0] for d in decoded], float),
        "hard": np.array([d[1] for d in decoded], int),
        "corners": regions[:, PROBE_NAMES.index("corners")],
        "edges": regions[:, PROBE_NAMES.index("edges")],
    }


METRICS = ("auc_10_vs_9", "auc_10_vs_9_corners", "auc_10_vs_9_edges", "mae", "qwk", "recall_9")


def arm_metrics(y: np.ndarray, p: dict[str, np.ndarray], idx: np.ndarray | None = None) -> dict:
    from ml.eval.evaluate import quadratic_weighted_kappa

    if idx is not None:
        y = y[idx]
        p = {k: v[idx] for k, v in p.items()}
    top = y >= 9
    is10 = y[top] == 10
    nine = y == 9
    return {
        "auc_10_vs_9": auc(is10, p["soft"][top]),
        "auc_10_vs_9_corners": auc(is10, p["corners"][top]),
        "auc_10_vs_9_edges": auc(is10, p["edges"][top]),
        "mae": float(np.abs(p["hard"] - y).mean()),
        "qwk": quadratic_weighted_kappa(y, p["hard"]),
        "recall_9": float((p["hard"][nine] == 9).mean()) if nine.any() else float("nan"),
    }


def describe_arm(y: np.ndarray, p: dict[str, np.ndarray]) -> dict:
    out = arm_metrics(y, p)
    out["mae_soft"] = float(np.abs(p["soft"] - y).mean())
    out["recall_10"] = float((p["hard"][y == 10] == 10).mean()) if (y == 10).any() else float("nan")
    out["n"] = int(len(y))
    out["n9"] = int((y == 9).sum())
    out["n10"] = int((y == 10).sum())
    out["pred_counts"] = {str(int(g)): int(c) for g, c in zip(*np.unique(p["hard"], return_counts=True), strict=True)}
    return out


def paired_group_bootstrap(
    y: np.ndarray,
    groups: np.ndarray,
    preds: dict[str, dict[str, np.ndarray]],
    pairs: Sequence[tuple[str, str]],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap over card groups: every arm is scored on the same resamples.

    For each pair (x, y) and metric: the full-sample difference x - y, the 95%
    percentile CI of the resampled differences and the share of resamples with
    a difference <= 0.
    """
    _, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]
    rng = np.random.default_rng(seed)
    arms = sorted({a for pair in pairs for a in pair})
    boot = {a: {m: np.empty(n_boot) for m in METRICS} for a in arms}
    for b in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        for a in arms:
            for m, v in arm_metrics(y, preds[a], idx).items():
                boot[a][m][b] = v
    full = {a: arm_metrics(y, preds[a]) for a in arms}
    out: dict = {"n_boot": n_boot, "seed": seed, "n_groups": len(members), "pairs": {}}
    for x, z in pairs:
        res = {}
        for m in METRICS:
            d = boot[x][m] - boot[z][m]
            d = d[~np.isnan(d)]
            lo, hi = np.percentile(d, [2.5, 97.5]) if len(d) else (float("nan"), float("nan"))
            res[m] = {
                "diff": full[x][m] - full[z][m],
                "ci95": [float(lo), float(hi)],
                "p_le_0": float((d <= 0).mean()) if len(d) else float("nan"),
            }
        out["pairs"][f"{x}-{z}"] = res
    return out


GO_MIN_GAIN = 0.03
B_SHARE_FOR_NO_DOWNLOAD = 0.5


def decide(pilot_boot: dict) -> dict:
    """The spec's default go / no-go rule on the pilot's AUC 10 vs 9 (head).

    Go (download) when C - A >= +0.03 and its 95% CI excludes 0; go with no
    download when B - A is then at least half of C - A ("B already gets most
    of that gain"); stop otherwise.
    """
    ca = pilot_boot["pairs"]["C-A"]["auc_10_vs_9"]
    ba = pilot_boot["pairs"]["B-A"]["auc_10_vs_9"]
    gain_ok = ca["diff"] >= GO_MIN_GAIN
    ci_ok = ca["ci95"][0] > 0
    if gain_ok and ci_ok:
        call = "go_no_download" if ba["diff"] >= B_SHARE_FOR_NO_DOWNLOAD * ca["diff"] else "go_download"
    else:
        call = "stop"
    return {
        "call": call,
        "rule": (
            f"go if C-A AUC10v9 >= {GO_MIN_GAIN} and its 95% CI excludes 0; then no download "
            f"if B-A >= {B_SHARE_FOR_NO_DOWNLOAD} x (C-A); else stop"
        ),
        "c_minus_a": ca,
        "b_minus_a": ba,
        "c_minus_a_meets_gain": gain_ok,
        "c_minus_a_ci_excludes_0": ci_ok,
    }


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _settings_and_dir(pilot_dir: Path | None) -> tuple[object, Path]:
    s = get_settings()
    out = check_output_dir(pilot_dir or PILOT_DIR, s.data_dir)
    return s, out


@app.command()
def integrity(out: Path = typer.Option(..., help="JSON file to write the snapshot to.")) -> None:
    """Snapshot the DB (sha256/size/mtime) and the protected folders' file counts."""
    s = get_settings()
    snap = integrity_snapshot(s.data_dir, s.db_path)
    write_json(check_output_dir(out, s.data_dir), snap)
    log.info("integrity -> %s: %s", out, json.dumps(snap))


@app.command()
def select(
    n_per_grade: int = typer.Option(300, help="PSA 9s and 10s each."),
    seed: int = typer.Option(0),
    pilot_dir: Path | None = typer.Option(None),
    force: bool = typer.Option(False, help="Overwrite an existing selection."),
) -> None:
    """Pick the pilot certs (train+val only) into pilot_certs.csv."""
    s, out = _settings_and_dir(pilot_dir)
    path = out / "pilot_certs.csv"
    if path.exists() and not force:
        raise SystemExit(f"{path} exists (use --force to reselect)")
    cands = load_candidates(s.db_path)
    rows = select_pilot(cands, n_per_grade, seed)
    write_selection(path, rows)
    n_groups = len({group_of(r) for r in rows})
    log.info(
        "selected %d certs (%d groups) from %d candidates -> %s",
        len(rows), n_groups, len(cands), path,
    )


@app.command()
def step0(
    n: int = typer.Option(5, help="Originals to fetch."),
    pilot_dir: Path | None = typer.Option(None),
    rate: float = typer.Option(2.0, help="Max requests per second."),
) -> None:
    """Fetch the first `n` pilot originals; record request/response headers and pixel sizes."""
    s, out = _settings_and_dir(pilot_dir)
    rows = pilot_rows(s.db_path, out)
    plan = download_plan(rows)[:n]
    seen: dict = {}

    def on_fetch(item, res: Fetch) -> None:
        seen[item] = res

    t0 = time.monotonic()
    with cdn_client() as client:
        stats = download_originals(plan, out, client, rate=rate, on_fetch=on_fetch)
    by_cert = {r["cert_number"]: r for r in rows}
    entries = read_manifest(out / "manifest.csv")
    images = []
    for cert, face, url in plan:
        r = by_cert[cert]
        m = entries[(cert, face)]
        res = seen.get((cert, face, url))
        rec: dict = {"cert": cert, "face": face, "url": url, "status": m["status"]}
        if res is not None:
            rec["request"] = res.request
            rec["response_status"] = res.status_code
            rec["response_headers"] = res.response_headers
            rec["rate_limit_like_headers"] = sorted(
                h for h in (res.response_headers or {}) if any(k in h for k in ("ratelimit", "rate-limit", "quota", "remaining"))
            )
        if m["status"] == "ok":
            photo = s.data_dir / r[f"{face}_path"]
            q = stored_quad(photo, s.data_dir / r[f"cropped_{face}_path"])
            ow, oh = int(m["width"]), int(m["height"])
            rec.update(bytes=int(m["bytes"]), original_wh=[ow, oh], stored_wh=q["src_wh"])
            if q["found"]:
                big = scale_quad(np.array(q["quad"]), tuple(q["src_wh"]), (ow, oh))
                rec["card_height_stored_px"] = round(quad_height(np.array(q["quad"])), 1)
                rec["card_height_original_px"] = round(quad_height(big), 1)
                rec["corner_block_original_px"] = round(quad_height(big) * CORNER_PATCHES / GRID_H, 1)
        images.append(rec)
    heights = [i["card_height_original_px"] for i in images if "card_height_original_px" in i]
    report = {
        "at": _now(),
        "stats": stats,
        "seconds": round(time.monotonic() - t0, 1),
        "images": images,
        "any_sensitive_header_sent": any(i.get("request", {}).get("sensitive") for i in images),
        "psa_api_calls_made": 0,
        "suggested_tile": suggest_tile(float(np.median(heights))) if heights else None,
    }
    write_json(out / "step0.json", report)
    log.info("step0 -> %s", out / "step0.json")


@app.command()
def download(
    pilot_dir: Path | None = typer.Option(None),
    rate: float = typer.Option(2.0, help="Max requests per second."),
) -> None:
    """Fetch both originals of every pilot cert (resumable)."""
    s, out = _settings_and_dir(pilot_dir)
    plan = download_plan(pilot_rows(s.db_path, out))
    t0 = time.monotonic()
    with cdn_client() as client:
        stats = download_originals(plan, out, client, rate=rate)
    secs = time.monotonic() - t0
    record_timing(out, "download", secs, **stats)
    log.info("download: %s in %.0fs; %s", stats, secs, manifest_summary(read_manifest(out / "manifest.csv")))


def original_sizes(rows: Sequence[dict], entries: dict, quad_cache: dict, data_dir: Path) -> dict:
    """Card height / corner block in original px over every downloaded pilot face, and T."""
    heights, kinds = [], []
    for r in rows:
        for face in FACES:
            m = entries.get((r["cert_number"], face))
            rec = _quad_for(quad_cache, data_dir, data_dir / r[f"{face}_path"])
            if not (m and m["status"] == "ok" and rec and rec["found"]):
                continue
            ow, oh = int(m["width"]), int(m["height"])
            heights.append(quad_height(scale_quad(np.array(rec["quad"]), tuple(rec["src_wh"]), (ow, oh))))
            kinds.append(f"{ow}x{oh}")
    h = np.array(heights, float)
    corner = h * CORNER_PATCHES / GRID_H
    long_edges = _counts(max(int(k.split("x")[0]), int(k.split("x")[1])) for k in kinds)
    return {
        "faces": len(h),
        "card_height_px": {
            "median": float(np.median(h)),
            "p5_p95": [float(x) for x in np.percentile(h, [5, 95])],
        },
        "corner_block_px": {
            "median": float(np.median(corner)),
            "p5_p95": [float(x) for x in np.percentile(corner, [5, 95])],
        },
        "original_long_edge_counts": long_edges,
        "rule": "T in {224, 448} closest in log scale to the median corner block",
        "tile": suggest_tile(float(np.median(h))),
    }


@app.command()
def sizes(pilot_dir: Path | None = typer.Option(None)) -> None:
    """Measure the card size in the downloaded originals and apply the tile-size rule."""
    s, out = _settings_and_dir(pilot_dir)
    res = original_sizes(
        pilot_rows(s.db_path, out),
        read_manifest(out / "manifest.csv"),
        read_json(out / "cache" / "quads.json", {}),
        s.data_dir,
    )
    res["at"] = _now()
    write_json(out / "sizes.json", res)
    log.info("sizes: %s", json.dumps(res))


def trainval_rows(db_path: Path) -> list:
    from ml.training.train_probe import load_rows

    return [r for r in load_rows(db_path) if r["split"] in TRAINVAL]


@app.command()
def quads(
    pilot_dir: Path | None = typer.Option(None),
    workers: int = typer.Option(8),
) -> None:
    """Re-detect the precrop quad on every train+val stored photo (both faces)."""
    s, out = _settings_and_dir(pilot_dir)
    rows = trainval_rows(s.db_path)
    items = [
        (s.data_dir / r[f"{face}_path"], s.data_dir / r[f"cropped_{face}_path"])
        for r in rows
        for face in FACES
    ]
    t0 = time.monotonic()
    cache = compute_quads(items, s.data_dir, out / "cache" / "quads.json", workers)
    problems = _counts(quad_problem(cache[p.relative_to(s.data_dir).as_posix()]) for p, _ in items)
    record_timing(out, "quads", time.monotonic() - t0, faces=len(items), problems=problems)
    log.info("quads: %d faces, problems %s", len(items), problems)


def _quad_for(cache: dict, data_dir: Path, photo: Path) -> dict | None:
    return cache.get(photo.relative_to(data_dir).as_posix())


def build_jobs(
    arm: str, rows: Sequence, data_dir: Path, pilot_dir: Path, quad_cache: dict
) -> tuple[list[TileJob], dict[str, str]]:
    """Tile jobs for usable certs of `rows`, plus {cert: reason} for the dropped ones."""
    entries = read_manifest(pilot_dir / "manifest.csv") if arm == "c" else {}
    jobs: list[TileJob] = []
    dropped: dict[str, str] = {}
    for r in rows:
        cert = r["cert_number"]
        photos = [data_dir / r[f"{face}_path"] for face in FACES]
        recs = [_quad_for(quad_cache, data_dir, p) for p in photos]
        problem = next((f"{face}:{p}" for face, rec in zip(FACES, recs, strict=True) if (p := quad_problem(rec))), None)
        if problem:
            dropped[cert] = problem
            continue
        if arm == "b":
            key = (cert, *[p.relative_to(data_dir).as_posix() for p in photos], *[_file_sig(p) for p in photos])

            def loader(photos=photos, recs=recs):
                return [(stored_rgb(p), np.array(q["quad"], np.float32)) for p, q in zip(photos, recs, strict=True)]
        else:
            origs = [pilot_dir / original_name(cert, face) for face in FACES]
            bad = next(
                (f"{face}:download_{(entries.get((cert, face)) or {}).get('status', 'missing')}"
                 for face, o in zip(FACES, origs, strict=True)
                 if not (o.exists() and (entries.get((cert, face)) or {}).get("status") == "ok")),
                None,
            )
            if bad:
                dropped[cert] = bad
                continue
            key = (cert, *[o.name for o in origs], *[_file_sig(o) for o in origs])

            def loader(origs=origs, recs=recs):
                out = []
                for o, q in zip(origs, recs, strict=True):
                    big = original_rgb(o)
                    h, w = big.shape[:2]
                    out.append((big, scale_quad(np.array(q["quad"]), tuple(q["src_wh"]), (w, h))))
                return out
        jobs.append((key, loader))
    return jobs, dropped


def original_checks(rows: Sequence, data_dir: Path, pilot_dir: Path, cache_path: Path, workers: int = 4) -> dict:
    """Per pilot face: is the original the stored photo's source (size + MAD)? Cached."""
    cache = read_json(cache_path, {})
    entries = read_manifest(pilot_dir / "manifest.csv")
    todo = []
    for r in rows:
        for face in FACES:
            o = pilot_dir / original_name(r["cert_number"], face)
            key = f"{r['cert_number']}_{face}"
            m = entries.get((r["cert_number"], face))
            if not (o.exists() and m and m["status"] == "ok"):
                continue
            if key in cache and cache[key].get("sig") == _file_sig(o):
                continue
            todo.append((key, o, data_dir / r[f"{face}_path"]))

    def check(item):
        key, o, photo = item
        rec = original_check(original_rgb(o), stored_rgb(photo))
        rec["sig"] = _file_sig(o)
        return key, rec

    if todo:
        with ThreadPoolExecutor(workers) as pool:
            for key, rec in pool.map(check, todo):
                cache[key] = rec
        write_json(cache_path, cache)
    return cache


@app.command()
def extract(
    arm: str = typer.Option(..., help="b (stored photos, all train+val) or c (pilot originals)."),
    tile: int = typer.Option(224, help="Corner tile side in px (fixed in the spec)."),
    pilot_dir: Path | None = typer.Option(None),
    workers: int = typer.Option(4),
    limit: int | None = typer.Option(None, help="Only the first N certs (smoke run)."),
) -> None:
    """Extract tile features for arm B (all train+val) or C (pilot originals)."""
    arm = arm.lower()
    if arm not in ("b", "c"):
        raise typer.BadParameter("arm must be b or c")
    s, out = _settings_and_dir(pilot_dir)
    spec = TileSpec(tile)
    quad_cache = read_json(out / "cache" / "quads.json", {})
    if not quad_cache:
        raise SystemExit("run `quads` first")
    if arm == "b":
        rows = trainval_rows(s.db_path)
    else:
        pilot = set(read_selection(out / "pilot_certs.csv"))
        rows = [r for r in trainval_rows(s.db_path) if r["cert_number"] in pilot]
        # Reasons are recorded per cert in original_checks.json; a cert whose
        # original fails the check is dropped from C (and so from every arm).
        checks = original_checks(rows, s.data_dir, out, out / "cache" / "original_checks.json", workers)
        bad = {k.rsplit("_", 1)[0] for k, v in checks.items() if v["problem"]}
        rows = [r for r in rows if r["cert_number"] not in bad]
    if limit:
        rows = rows[:limit]
    jobs, dropped = build_jobs(arm, rows, s.data_dir, out, quad_cache)
    t0 = time.monotonic()
    cache = out / "cache" / f"tiles_{arm}_{spec.tile}.npz"
    feats = extract_tile_features(jobs, spec, cache, workers=workers)
    secs = time.monotonic() - t0
    record_timing(out, f"extract_{arm}_{spec.tile}", secs, certs=len(jobs), dropped=len(dropped))
    log.info("arm %s: %s for %d certs in %.0fs (%d dropped: %s)", arm, feats.shape, len(jobs), secs, len(dropped), _counts(dropped.values()))


def current_tile_keys(
    arm: str, rows: Sequence, data_dir: Path, pilot_dir: Path, quad_cache: dict
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """{cert: the tile cache key its current source files give}, plus {cert: reason} dropped."""
    jobs, dropped = build_jobs(arm, rows, data_dir, pilot_dir, quad_cache)
    return {key[0]: key for key, _ in jobs}, dropped


def _load_tiles(path: Path, spec: TileSpec, keys: dict[str, tuple[str, ...]]) -> tuple[dict[str, np.ndarray], int]:
    """{cert: (2, 8, D)} from a tile cache (must match the tile spec), plus the stale count.

    Only certs in `keys` whose cached key (source paths + file signatures) is
    still `keys[cert]` are returned; an entry from a since-replaced file is stale.
    """
    with np.load(path, allow_pickle=False) as z:
        if str(z["tile_spec_hash"]) != spec.hash():
            raise SystemExit(f"{path} is from another tile spec")
        out, stale = {}, 0
        for k, f in zip(z["keys"], z["feats"], strict=True):
            key = tuple(str(x) for x in k)
            if keys.get(key[0]) == key:
                out[key[0]] = f.astype(np.float32)
            elif key[0] in keys:
                stale += 1
        return out, stale


def _cv(feats: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int) -> tuple[dict, float]:
    from ml.training.probe import cross_validate

    t0 = time.monotonic()
    cum, regions = cross_validate(feats, y, groups, n_folds=folds, seed=0)
    return arm_predictions(cum, regions), time.monotonic() - t0


@app.command()
def evaluate(
    tile: int = typer.Option(224),
    pilot_dir: Path | None = typer.Option(None),
    folds: int = typer.Option(5),
    n_boot: int = typer.Option(1000),
    out: Path = typer.Option(REPORT_PATH),
) -> None:
    """Card-grouped CV of arms A/B/C on the pilot (+ A/B on all train+val), bootstrap, report."""
    from ml.training.probe import card_groups
    from ml.training.train_probe import extract_features

    s, pdir = _settings_and_dir(pilot_dir)
    out = check_output_dir(out, s.data_dir)
    spec = TileSpec(tile)
    t_start = time.monotonic()
    rows = trainval_rows(s.db_path)
    quad_cache = read_json(pdir / "cache" / "quads.json", {})
    pilot = read_selection(pdir / "pilot_certs.csv")
    selection = set(pilot)
    checks = read_json(pdir / "cache" / "original_checks.json", {})
    # Arm A: today's features, from the probe's cache, read only (save=False).
    a_cache = s.models_dir / "cache" / "region_feats.npz"
    a_all = extract_features(rows, s.data_dir, a_cache, save=False)
    a_by_cert = {r["cert_number"]: a_all[i] for i, r in enumerate(rows)}
    # B / C tiles count only when cached for the current source files (a quad
    # or download problem drops the cert from the keys), and C only when both
    # originals passed the check on those same files.
    b_keys, b_dropped = current_tile_keys("b", rows, s.data_dir, pdir, quad_cache)
    pilot_rows_tv = [r for r in rows if r["cert_number"] in selection]
    c_keys, c_dropped = current_tile_keys("c", pilot_rows_tv, s.data_dir, pdir, quad_cache)
    original_dropped: dict[str, str] = {}
    for cert, key in c_keys.items():
        for face, sig in zip(FACES, key[-len(FACES) :], strict=True):
            rec = checks.get(f"{cert}_{face}")
            if rec is None:
                original_dropped.setdefault(cert, f"{face}:original_check_missing")
            elif rec.get("sig") != sig:
                original_dropped.setdefault(cert, f"{face}:original_check_stale")
            elif rec["problem"]:
                original_dropped.setdefault(cert, f"{face}:{rec['problem']}")
    c_keys = {c: k for c, k in c_keys.items() if c not in original_dropped}
    b_tiles, b_stale = _load_tiles(pdir / "cache" / f"tiles_b_{spec.tile}.npz", spec, b_keys)
    c_tiles, c_stale = _load_tiles(pdir / "cache" / f"tiles_c_{spec.tile}.npz", spec, c_keys)

    # A reproduction check: 8 pilot certs re-extracted from scratch, never cached.
    first8 = set(pilot[:8])
    check_rows = [r for r in rows if r["cert_number"] in first8]
    fresh = extract_features(check_rows, s.data_dir, pdir / "cache" / "a_check_never_written.npz", save=False)
    cached = np.stack([a_by_cert[r["cert_number"]] for r in check_rows])
    a_repro = {
        "n_certs": len(check_rows),
        "max_abs_diff": float(np.abs(fresh - cached).max()),
        "max_rel_diff": float((np.abs(fresh - cached) / (np.abs(cached) + 1e-3)).max()),
        "cosine_min": float(
            np.min(np.sum(fresh * cached, -1) / (np.linalg.norm(fresh, axis=-1) * np.linalg.norm(cached, axis=-1)))
        ),
    }

    def subset(certs: Sequence[str]):
        keep = set(certs)
        rs = [r for r in rows if r["cert_number"] in keep]
        y = np.array([r["grade_int"] for r in rs], int)
        g = card_groups(
            [r["brand_title"] or "" for r in rs],
            [r["subject"] or "" for r in rs],
            [r["card_number"] or "" for r in rs],
        )
        a = np.stack([a_by_cert[r["cert_number"]] for r in rs])
        return rs, y, g, a

    # Pilot: certs with A, B and C features; a cert dropped from any arm leaves them all.
    tv_certs = {r["cert_number"] for r in rows}

    def drop_reason(c: str) -> str | None:
        if c not in tv_certs:
            return "not_trainval"
        if c in b_dropped:
            return f"b:{b_dropped[c]}"
        if c in c_dropped:
            return f"c:{c_dropped[c]}"
        if c in original_dropped:
            return original_dropped[c]
        if c not in b_tiles:
            return "b:no_tiles"
        if c not in c_tiles:
            return "c:no_tiles"
        return None

    pilot_dropped = {c: r for c in pilot if (r := drop_reason(c))}
    pilot_set = [c for c in pilot if c not in pilot_dropped]
    prs, py, pg, pa = subset(pilot_set)
    pcerts = [r["cert_number"] for r in prs]
    arms = {
        "A": pa,
        "B": with_tiles(pa, np.stack([b_tiles[c] for c in pcerts])),
        "C": with_tiles(pa, np.stack([c_tiles[c] for c in pcerts])),
    }
    preds, cv_secs = {}, {}
    for name, f in arms.items():
        preds[name], cv_secs[name] = _cv(f, py, pg, folds)
        log.info("pilot %s: %s", name, describe_arm(py, preds[name]))
    pilot_boot = paired_group_bootstrap(py, pg, preds, [("B", "A"), ("C", "A"), ("C", "B")], n_boot)

    # Train+val: A and B on every cert with B features.
    tv_set = [r["cert_number"] for r in rows if r["cert_number"] in b_tiles]
    trs, ty, tg, ta = subset(tv_set)
    tcerts = [r["cert_number"] for r in trs]
    tv_arms = {"A": ta, "B": with_tiles(ta, np.stack([b_tiles[c] for c in tcerts]))}
    tv_preds, tv_secs = {}, {}
    for name, f in tv_arms.items():
        tv_preds[name], tv_secs[name] = _cv(f, ty, tg, folds)
        log.info("train+val %s: %s", name, describe_arm(ty, tv_preds[name]))
    tv_boot = paired_group_bootstrap(ty, tg, tv_preds, [("B", "A")], n_boot)

    entries = read_manifest(pdir / "manifest.csv")
    mads = np.array([v["mad"] for v in checks.values() if v.get("mad") is not None], float)
    pre_mads = np.array([v["precrop_mad"] for v in quad_cache.values() if v.get("precrop_mad") is not None], float)
    report = {
        "generated_at": _now(),
        "git_commit": git_commit(),
        "spec": "intent/full-resolution-pilot/intent.md",
        "tile_spec": spec.to_dict(),
        "tile_spec_hash": spec.hash(),
        "cv": {"folds": folds, "seed": 0, "head": "probe.cross_validate (whole head refit per arm)"},
        "gates": {"max_precrop_mad": MAX_PRECROP_MAD, "max_original_mad": MAX_ORIGINAL_MAD},
        "step0": read_json(pdir / "step0.json"),
        "original_sizes": read_json(pdir / "sizes.json"),
        "redetect_sample": (read_json(pdir / "redetect_sample.json") or {}).get("summary"),
        "download": {**manifest_summary(entries), "timing": read_json(pdir / "timings.json", {}).get("download")},
        "checks": {
            "precrop_mad_median_p99_max": [float(np.median(pre_mads)), float(np.percentile(pre_mads, 99)), float(pre_mads.max())] if len(pre_mads) else None,
            "original_mad_median_p99_max": [float(np.median(mads)), float(np.percentile(mads, 99)), float(mads.max())] if len(mads) else None,
            "original_problems": _counts(v["problem"] for v in checks.values() if v["problem"]),
            "trainval_b_dropped": _counts(r.split(":", 1)[1] for r in b_dropped.values()),
            "pilot_dropped": pilot_dropped,
            "stale_tiles_skipped": {"b": b_stale, "c": c_stale},
            "a_reproduction": a_repro,
        },
        "pilot": {
            "n_selected": len(pilot),
            "n": len(pcerts),
            "n9": int((py == 9).sum()),
            "n10": int((py == 10).sum()),
            "n_groups": int(len(np.unique(pg))),
            "arms": {k: describe_arm(py, v) for k, v in preds.items()},
            "bootstrap": pilot_boot,
            "cv_seconds": {k: round(v, 1) for k, v in cv_secs.items()},
        },
        "trainval": {
            "n": len(tcerts),
            "n9": int((ty == 9).sum()),
            "n10": int((ty == 10).sum()),
            "n_groups": int(len(np.unique(tg))),
            "arms": {k: describe_arm(ty, v) for k, v in tv_preds.items()},
            "bootstrap": tv_boot,
            "cv_seconds": {k: round(v, 1) for k, v in tv_secs.items()},
        },
        "decision": decide(pilot_boot),
        "notes": [
            "The pilot set holds only PSA 9s and 10s, so its MAE is about the 9/10 error rate "
            "(plus any prediction below 9) and its QWK is close to a 2-class kappa; neither is "
            "comparable with all-grade numbers. AUC 10 vs 9 is the primary metric.",
            "The head's soft grade can't rank 9 vs 10 better than the region scores allow; "
            "corners/edges AUCs are the OOF soft grades of those region probes.",
        ],
        "timings": read_json(pdir / "timings.json", {}),
        "integrity_before": read_json(pdir / "integrity_before.json"),
        "integrity_after": integrity_snapshot(s.data_dir, s.db_path),
    }
    report["evaluate_seconds"] = round(time.monotonic() - t_start, 1)
    write_json(out, report)
    record_timing(pdir, "evaluate", time.monotonic() - t_start)
    log.info("report -> %s; decision %s", out, report["decision"]["call"])


@app.command("redetect-sample")
def redetect_sample(
    n: int = typer.Option(20, help="Pilot faces to re-detect on the original."),
    pilot_dir: Path | None = typer.Option(None),
) -> None:
    """Assumption check: does the scaled 1024 px quad land where detection on the original does?"""
    from backend.app.ml.preprocess import _find_card_quad

    s, out = _settings_and_dir(pilot_dir)
    quad_cache = read_json(out / "cache" / "quads.json", {})
    entries = read_manifest(out / "manifest.csv")
    rows = pilot_rows(s.db_path, out)
    results = []
    for r in rows:
        for face in FACES:
            if len(results) >= n:
                break
            m = entries.get((r["cert_number"], face))
            rec = _quad_for(quad_cache, s.data_dir, s.data_dir / r[f"{face}_path"])
            if not (m and m["status"] == "ok" and rec and rec["found"]):
                continue
            o = out / original_name(r["cert_number"], face)
            big = original_rgb(o)
            h, w = big.shape[:2]
            scaled = scale_quad(np.array(rec["quad"]), tuple(rec["src_wh"]), (w, h))
            res: dict = {"cert": r["cert_number"], "face": face, "orig_wh": [w, h]}
            # (a) detection on the full-size original (tuned for ~1024 px; may miss)
            found = _find_card_quad(cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
            if found is not None:
                d = np.linalg.norm(found[0] - scaled, axis=1)
                res["fullres_corner_dist_px"] = [round(float(x), 2) for x in d]
            # (b) `detect_and_crop` on the original's bytes (decoded to 1024 px, no JPEG pass)
            c = detect_and_crop(o.read_bytes())
            if c.found_quad:
                q = scale_quad(c.quad, (c.source_bgr.shape[1], c.source_bgr.shape[0]), (w, h))
                res["redetect_1024_corner_dist_px"] = [round(float(x), 2) for x in np.linalg.norm(q - scaled, axis=1)]
            res["px_per_stored_px"] = round(w / rec["src_wh"][0], 3)
            results.append(res)
    summary = {}
    for k in ("fullres_corner_dist_px", "redetect_1024_corner_dist_px"):
        d = np.array([x for r in results for x in r.get(k, [])], float)
        if len(d):
            summary[k] = {
                "faces": sum(k in r for r in results),
                "median": float(np.median(d)),
                "p90": float(np.percentile(d, 90)),
                "max": float(d.max()),
            }
    write_json(out / "redetect_sample.json", {"at": _now(), "summary": summary, "faces": results})
    log.info("redetect sample: %s", summary)


if __name__ == "__main__":
    app()
