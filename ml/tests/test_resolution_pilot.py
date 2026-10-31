"""Full-resolution pilot (`ml.eval.resolution_pilot`): tile geometry, quad
scaling, the CDN download + manifest, resume, test-split exclusion and the
evaluation helpers. Synthetic images in tmp_path only; HTTP is mocked."""

from __future__ import annotations

import hashlib
import io
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
import pytest
from PIL import Image

from backend.app.ml.features import (
    CORNER_PATCHES,
    EDGE_PATCHES,
    FEAT_H,
    FEAT_W,
    GRID_H,
    GRID_W,
    PATCH,
    region_means,
    to_input,
)
from backend.app.ml.preprocess import CARD_H, CARD_W, detect_and_crop, encode_jpeg
from backend.tests.helpers.synth import corner_error, make_scene
from ml.eval import resolution_pilot as rp

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------------- #
# Tile geometry
# --------------------------------------------------------------------------- #
def test_region_cells_are_the_region_means_blocks():
    patches = np.random.default_rng(0).normal(size=(1, GRID_H, GRID_W, 3))
    means = region_means(patches)
    cells = rp.region_cells()
    assert tuple(cells) == rp.TILE_REGIONS
    for i, (r0, r1, c0, c1) in enumerate(cells.values()):
        np.testing.assert_allclose(means[i][0], patches[0, r0:r1, c0:c1].mean((0, 1)))


def test_corner_boxes_are_5x5_patch_blocks():
    for name, b in rp.region_boxes().items():
        if name.startswith("corner"):
            assert b.u1 - b.u0 == pytest.approx(CORNER_PATCHES / GRID_W)
            assert b.v1 - b.v0 == pytest.approx(CORNER_PATCHES / GRID_H)
            # Same card area in today's feature pixels: 70 x 70.
            assert (b.u1 - b.u0) * FEAT_W == pytest.approx(CORNER_PATCHES * PATCH)
            assert (b.v1 - b.v0) * FEAT_H == pytest.approx(CORNER_PATCHES * PATCH)


@pytest.mark.parametrize(("tile", "depth", "n_long", "n_side"), [(224, 140, 4, 7), (448, 266, 4, 7)])
def test_tile_spec_sizes(tile, depth, n_long, n_side):
    spec = rp.TileSpec(tile)
    assert spec.edge_depth == depth
    tiles = rp.region_tiles(spec)
    count = {r: sum(t.region == r for t in tiles) for r in rp.TILE_REGIONS}
    assert count["edge_top"] == count["edge_bottom"] == n_long
    assert count["edge_left"] == count["edge_right"] == n_side
    assert all(count[r] == 1 for r in rp.TILE_REGIONS if r.startswith("corner"))
    for t in tiles:
        assert t.out_w % PATCH == 0 and t.out_h % PATCH == 0
        if t.region.startswith("corner"):
            assert (t.out_w, t.out_h) == (tile, tile)
        elif t.region in ("edge_top", "edge_bottom"):
            assert (t.out_w, t.out_h) == (tile, depth)
        else:
            assert (t.out_w, t.out_h) == (depth, tile)


def test_tile_spec_rejects_non_patch_sizes():
    with pytest.raises(ValueError):
        rp.TileSpec(200)
    with pytest.raises(ValueError):
        rp.TileSpec(14)


@pytest.mark.parametrize("tile", [224, 448])
def test_edge_tiles_partition_their_strip(tile):
    boxes = rp.region_boxes()
    tiles = rp.region_tiles(rp.TileSpec(tile))
    for name in ("edge_top", "edge_right", "edge_bottom", "edge_left"):
        strip = boxes[name]
        mine = [t.box for t in tiles if t.region == name]
        horizontal = name in ("edge_top", "edge_bottom")
        lo, hi = ("u0", "u1") if horizontal else ("v0", "v1")
        across = ("v0", "v1") if horizontal else ("u0", "u1")
        mine.sort(key=lambda b, lo=lo: getattr(b, lo))
        assert getattr(mine[0], lo) == pytest.approx(getattr(strip, lo))
        assert getattr(mine[-1], hi) == pytest.approx(getattr(strip, hi))
        for a, b in zip(mine, mine[1:], strict=False):
            assert getattr(a, hi) == pytest.approx(getattr(b, lo))  # no gap, no overlap
        for b in mine:
            for k in across:  # the full 3-patch depth of the strip
                assert getattr(b, k) == pytest.approx(getattr(strip, k))
        depth = (strip.v1 - strip.v0) * FEAT_H if horizontal else (strip.u1 - strip.u0) * FEAT_W
        assert depth == pytest.approx(EDGE_PATCHES * PATCH)


def _coordinate_scene(margin=(50, 60)):
    """Float image whose channels are the 600x840 crop coordinates (x, y, 0) of each
    pixel, with the card quad offset by `margin` so border samples stay in the image."""
    mx, my = margin
    h, w = CARD_H + 2 * my, CARD_W + 2 * mx
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32) - mx, np.arange(h, dtype=np.float32) - my)
    src = np.dstack([xs, ys, np.zeros_like(xs)])
    quad = np.array([[mx, my], [mx + CARD_W - 1, my], [mx + CARD_W - 1, my + CARD_H - 1], [mx, my + CARD_H - 1]], np.float32)
    return src, quad


@pytest.mark.parametrize("tile", [224, 448])
def test_each_tile_samples_exactly_its_card_area(tile):
    """The tile's pixel extent maps onto [u0, u1] x [v0, v1] of the 600x840 crop's pixel
    extent, i.e. the card area of today's patch block (feature px = crop extent x 434/600)."""
    src, quad = _coordinate_scene()
    for t in rp.region_tiles(rp.TileSpec(tile)):
        img = rp.sample_tile(src, quad, t)
        assert img.shape[:2] == (t.out_h, t.out_w)
        sx = (t.box.u1 - t.box.u0) * CARD_W / t.out_w
        sy = (t.box.v1 - t.box.v0) * CARD_H / t.out_h
        x_edges = (img[0, 0, 0] - sx / 2, img[0, -1, 0] + sx / 2)
        y_edges = (img[0, 0, 1] - sy / 2, img[-1, 0, 1] + sy / 2)
        np.testing.assert_allclose(x_edges, (t.box.u0 * CARD_W - 0.5, t.box.u1 * CARD_W - 0.5), atol=0.05)
        np.testing.assert_allclose(y_edges, (t.box.v0 * CARD_H - 0.5, t.box.v1 * CARD_H - 0.5), atol=0.05)
        # Axis-aligned and uniform inside the tile.
        np.testing.assert_allclose(np.diff(img[0, :, 0]), sx, atol=0.05)
        np.testing.assert_allclose(np.diff(img[:, 0, 1]), sy, atol=0.05)


def test_corner_tile_matches_feature_pixels_of_the_crop():
    """Sampling the corner from the photo equals cropping today's feature image of the
    same card, resized: the two paths agree on a smooth card."""
    yy, xx = np.mgrid[0:CARD_H, 0:CARD_W].astype(np.float32)
    card = np.dstack([128 + 80 * np.sin(xx / 37.0), 128 + 80 * np.cos(yy / 53.0), np.full_like(xx, 90)])
    card = card.astype(np.uint8)
    src = cv2.copyMakeBorder(card, 40, 40, 30, 30, cv2.BORDER_REPLICATE)
    quad = np.array([[30, 40], [30 + CARD_W - 1, 40], [30 + CARD_W - 1, 40 + CARD_H - 1], [30, 40 + CARD_H - 1]], np.float32)
    spec = rp.TileSpec(224)
    tile = next(t for t in rp.region_tiles(spec) if t.region == "corner_tl")
    got = rp.sample_tile(src, quad, tile).astype(float)
    feat = cv2.resize(card, (FEAT_W, FEAT_H), interpolation=cv2.INTER_AREA)
    k = CORNER_PATCHES * PATCH
    ref = cv2.resize(feat[:k, :k], (224, 224), interpolation=cv2.INTER_LINEAR).astype(float)
    assert np.abs(got - ref).mean() < 3.0


def test_scale_quad_maps_pixel_extents():
    w, h, big_w, big_h = 610, 1024, 1430, 2400
    extent = np.array([[-0.5, -0.5], [w - 0.5, -0.5], [w - 0.5, h - 0.5], [-0.5, h - 0.5]])
    big = rp.scale_quad(extent, (w, h), (big_w, big_h))
    np.testing.assert_allclose(
        big, [[-0.5, -0.5], [big_w - 0.5, -0.5], [big_w - 0.5, big_h - 0.5], [-0.5, big_h - 0.5]], atol=1e-4
    )
    back = rp.scale_quad(big, (big_w, big_h), (w, h))
    np.testing.assert_allclose(back, extent, atol=1e-4)


def test_quad_found_at_1024_scales_onto_the_card_in_the_original():
    """Ingest-like: a 1430x2400 'original' is LANCZOS-shrunk to 1024 px, the quad is
    found there and scaled back; it lands on the true card corners."""
    scene = make_scene(seed=3, condition="clean", size=(2400, 1430))
    pil = Image.fromarray(cv2.cvtColor(scene.bgr, cv2.COLOR_BGR2RGB))
    s = 1024 / 2400
    stored = np.array(pil.resize((round(1430 * s), 1024), Image.LANCZOS))
    crop = detect_and_crop(stored)
    assert crop.found_quad
    big = rp.scale_quad(crop.quad, (stored.shape[1], stored.shape[0]), (1430, 2400))
    # The synthetic quad marks the card's outer pixel edges; 3 original px ~ 1.3 stored px.
    assert corner_error(big, scene.quad) < 3.0 / s


@pytest.mark.parametrize("tile", [56, 224])
def test_tiles_from_original_match_tiles_from_stored_photo(tile):
    """Same card, two resolutions: tiles sampled with the scaled quad agree on smooth
    content (tile 56 exercises the INTER_AREA pre-shrink of a dense source)."""
    big_h, big_w = 2000, 1200
    yy, xx = np.mgrid[0:big_h, 0:big_w].astype(np.float32)
    orig = np.dstack(
        [
            128 + 70 * np.sin(xx / 90.0) * np.cos(yy / 120.0),
            128 + 70 * np.cos(xx / 140.0),
            128 + 70 * np.sin((xx + yy) / 200.0),
        ]
    ).astype(np.uint8)
    quad_big = np.array([[150, 260], [1050, 240], [1070, 1500], [140, 1520]], np.float32)
    s = 1024 / big_h
    stored = np.array(Image.fromarray(orig).resize((round(big_w * s), 1024), Image.LANCZOS))
    quad_small = rp.scale_quad(quad_big, (big_w, big_h), (stored.shape[1], stored.shape[0]))
    spec = rp.TileSpec(tile)
    a = rp.face_tiles(orig, quad_big, spec)
    b = rp.face_tiles(stored, quad_small, spec)
    assert [i for i, _ in a] == [i for i, _ in b]
    for (_, ta), (_, tb) in zip(a, b, strict=True):
        assert ta.shape == tb.shape
        assert np.abs(ta.astype(float) - tb.astype(float)).mean() < 2.0


def test_prepare_source_shrinks_only_dense_sources():
    spec = rp.TileSpec(56)  # canvas height 481.6 px
    src = np.zeros((1200, 800, 3), np.uint8)
    dense = np.array([[100, 100], [700, 100], [700, 1060], [100, 1060]], np.float32)  # 960 px tall
    small, q = rp.prepare_source(src, dense, spec)
    ratio = 960 / spec.canvas[1]
    assert small.shape[:2] == (round(1200 / ratio), round(800 / ratio))
    assert rp.quad_height(q) == pytest.approx(spec.canvas[1], rel=0.01)
    sparse = dense / 3
    same, q2 = rp.prepare_source(src, sparse, spec)
    assert same is src
    np.testing.assert_array_equal(q2, sparse)


def test_suggest_tile():
    assert rp.suggest_tile(1500) == 224  # corner block ~174 px
    assert rp.suggest_tile(3050) == 448  # ~355 px


def test_original_sizes_scale_the_stored_card_height(tmp_path):
    data = tmp_path / "data"
    quad = [[100.0, 180.0], [560.0, 180.0], [560.0, 840.0], [100.0, 840.0]]  # 660 px card
    rows = [_row(str(i), 9, "train", i) for i in range(3)]
    cache = {f"images/{r['cert_number']}_{f}.jpg": {"found": True, "quad": quad, "src_wh": [610, 1024]} for r in rows for f in ("front", "back")}
    big = {"0": (1430, 2400), "1": (1430, 2400), "2": (2807, 4742)}
    entries = {
        (c, f): {"status": "ok", "width": str(w), "height": str(h)} for c, (w, h) in big.items() for f in ("front", "back")
    }
    entries[("2", "back")] = {"status": "failed"}
    res = rp.original_sizes(rows, entries, cache, data)
    assert res["faces"] == 5
    assert res["card_height_px"]["median"] == pytest.approx(660 * 2400 / 1024, rel=0.01)
    assert res["original_long_edge_counts"] == {2400: 4, 4742: 1}
    assert res["tile"] == 224


# --------------------------------------------------------------------------- #
# Quads and source checks
# --------------------------------------------------------------------------- #
def test_stored_quad_matches_the_precrop_it_came_from(tmp_path):
    scene = make_scene(seed=1, condition="clean", size=(1024, 620))
    photo = tmp_path / "images" / "1_front.jpg"
    photo.parent.mkdir()
    photo.write_bytes(encode_jpeg(cv2.cvtColor(scene.bgr, cv2.COLOR_BGR2RGB), quality=95))
    crop = tmp_path / "crops" / "1_front.jpg"
    crop.parent.mkdir()
    crop.write_bytes(encode_jpeg(detect_and_crop(photo.read_bytes()).image))
    rec = rp.stored_quad(photo, crop)
    assert rec["found"] and rec["rotation"] == 0 and rec["src_wh"] == [620, 1024]
    assert rec["precrop_mad"] < 3.0
    assert rp.quad_problem(rec) is None
    # A precrop that isn't today's warp of this photo (here: upside down) is flagged.
    crop.write_bytes(encode_jpeg(np.ascontiguousarray(detect_and_crop(photo.read_bytes()).image[::-1])))
    assert rp.quad_problem(rp.stored_quad(photo, crop)) == "precrop_mismatch"


@pytest.mark.parametrize(
    ("rec", "problem"),
    [
        (None, "no_quad_record"),
        ({"found": False, "rotation": 0, "precrop_mad": 1.0}, "no_quad"),
        ({"found": True, "rotation": 90, "precrop_mad": 1.0}, "rotated"),
        ({"found": True, "rotation": 0, "precrop_mad": None}, "precrop_unreadable"),
        ({"found": True, "rotation": 0, "precrop_mad": 30.0}, "precrop_mismatch"),
        ({"found": True, "rotation": 0, "precrop_mad": 2.0}, None),
    ],
)
def test_quad_problem(rec, problem):
    assert rp.quad_problem(rec) == problem


def test_original_check():
    yy, xx = np.mgrid[0:2400, 0:1430].astype(np.float32)
    orig = np.dstack([yy / 2400 * 255, 128 + 100 * np.sin(xx / 60.0), xx / 1430 * 255]).astype(np.uint8)
    s = 1024 / 2400
    stored = np.array(Image.fromarray(orig).resize((round(1430 * s), 1024), Image.LANCZOS))
    ok = rp.original_check(orig, stored)
    assert ok["problem"] is None and ok["mad"] < 2.0
    other = np.ascontiguousarray(orig[::-1])
    assert rp.original_check(other, stored)["problem"] == "original_mismatch"
    assert rp.original_check(orig[:, :1200], stored)["problem"] == "size_mismatch"
    assert rp.original_check(stored, stored)["problem"] == "not_larger"


# --------------------------------------------------------------------------- #
# Selection: train+val only, spread over card groups
# --------------------------------------------------------------------------- #
def _row(cert, grade, split, group, url=True):
    return {
        "cert_number": cert,
        "grade_int": grade,
        "split": split,
        "brand_title": "SET",
        "subject": f"CARD{group}",
        "card_number": "1",
        "front_path": f"images/{cert}_front.jpg",
        "back_path": f"images/{cert}_back.jpg",
        "cropped_front_path": f"crops/images/{cert}_front.jpg",
        "cropped_back_path": f"crops/images/{cert}_back.jpg",
        "front_url": f"https://cdn.example/{cert}_f.jpg" if url else None,
        "back_url": f"https://cdn.example/{cert}_b.jpg" if url else None,
    }


def test_select_pilot_is_train_val_only_and_spread_over_groups():
    rows = []
    n = 0
    for grade in (9, 10):
        for g in range(8):
            for _ in range(3):
                n += 1
                rows.append(_row(f"{n:04d}", grade, ("train", "val", "test")[n % 3], g))
    picked = rp.select_pilot(rows, n_per_grade=6, seed=0)
    assert all(r["split"] in ("train", "val") for r in picked)
    for grade in (9, 10):
        mine = [r for r in picked if r["grade_int"] == grade]
        assert len(mine) == 6
        assert len({rp.group_of(r) for r in mine}) == 6  # one per group before repeats
    assert picked == rp.select_pilot(rows, n_per_grade=6, seed=0)
    assert [r["cert_number"] for r in picked] == sorted(r["cert_number"] for r in picked)


def _make_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE certs (cert_number TEXT PRIMARY KEY, grade_int INTEGER, split TEXT, "
        "brand_title TEXT, subject TEXT, card_number TEXT, front_path TEXT, back_path TEXT, "
        "cropped_front_path TEXT, cropped_back_path TEXT, front_url TEXT, back_url TEXT, "
        "crop_kind_front TEXT, crop_kind_back TEXT, is_pokemon INTEGER, has_images INTEGER, "
        "crop_found_front INTEGER, crop_found_back INTEGER)"
    )
    specs = [
        ("1", 9, "train", True, 1),
        ("2", 10, "val", True, 1),
        ("3", 10, "test", True, 1),  # test split: never a candidate
        ("4", 9, "train", False, 1),  # no original URLs
        ("5", 8, "train", True, 1),  # not a 9 / 10
        ("6", 9, "train", True, 0),  # no usable precrop
        ("7", 9, None, True, 1),  # unassigned
    ]
    for cert, grade, split, url, found in specs:
        r = _row(cert, grade, split, 0, url)
        conn.execute(
            "INSERT INTO certs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                cert, grade, split, r["brand_title"], r["subject"], r["card_number"],
                r["front_path"], r["back_path"], r["cropped_front_path"], r["cropped_back_path"],
                r["front_url"], r["back_url"], "card", "card", 1, 1, found, found,
            ),
        )
    conn.commit()
    conn.close()


def test_load_candidates_excludes_test_split_and_never_writes(tmp_path):
    db = tmp_path / "pokescan.sqlite"
    _make_db(db)
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    rows = rp.load_candidates(db)
    assert [r["cert_number"] for r in rows] == ["1", "2"]
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


def test_pilot_rows_and_plan_refuse_test_certs(tmp_path):
    db = tmp_path / "pokescan.sqlite"
    _make_db(db)
    pilot = tmp_path / "pilot"
    rp.write_selection(pilot / "pilot_certs.csv", [_row("1", 9, "train", 0), _row("3", 10, "test", 0)])
    with pytest.raises(RuntimeError, match="no longer"):
        rp.pilot_rows(db, pilot)
    with pytest.raises(ValueError, match="test"):
        rp.download_plan([_row("3", 10, "test", 0)])
    rp.write_selection(pilot / "pilot_certs.csv", [_row("1", 9, "train", 0), _row("2", 10, "val", 0)])
    plan = rp.download_plan(rp.pilot_rows(db, pilot))
    assert [(c, f) for c, f, _ in plan] == [("1", "front"), ("1", "back"), ("2", "front"), ("2", "back")]


def test_check_output_dir_refuses_protected_folders(tmp_path):
    data = tmp_path / "data"
    for bad in (data / "images", data / "crops" / "x", data / "models" / "cache", data):
        with pytest.raises(ValueError):
            rp.check_output_dir(bad, data)
    assert rp.check_output_dir(data / "images_full", data) == (data / "images_full").resolve()


def test_check_output_dir_refuses_everything_in_data_but_images_full(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "images_full").mkdir(parents=True)
    (data / "pokescan.sqlite").write_bytes(b"db")
    bad = [
        data / "pokescan.sqlite",
        data / "other.json",
        data / "backups" / "x.json",
        data / "images_full" / ".." / "pokescan.sqlite",
        data / "images_full" / ".." / "models" / "x.json",
        data / "images_fullx" / "x.json",
    ]
    if os.name == "nt":  # Windows paths are case-insensitive
        bad += [tmp_path / "DATA" / "POKESCAN.SQLITE", tmp_path / "Data" / "Images_Full" / ".." / "Crops"]
    for p in bad:
        with pytest.raises(ValueError):
            rp.check_output_dir(p, data)
    monkeypatch.chdir(data / "images_full")
    with pytest.raises(ValueError):
        rp.check_output_dir(Path("..") / "pokescan.sqlite", data)  # relative
    ok = data / "images_full" / "cache" / "x.json"
    assert rp.check_output_dir(ok, data) == ok.resolve()
    assert rp.check_output_dir(Path("integrity.json"), data) == (data / "images_full" / "integrity.json").resolve()
    assert rp.check_output_dir(tmp_path / "test-artifacts" / "r.json", data) == (tmp_path / "test-artifacts" / "r.json").resolve()


def test_evaluate_refuses_an_out_path_in_data_before_any_work(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "images_full").mkdir(parents=True)
    db = data / "pokescan.sqlite"
    db.write_bytes(b"db")
    monkeypatch.setattr(rp, "get_settings", lambda: SimpleNamespace(data_dir=data, db_path=db, models_dir=data / "models"))
    monkeypatch.setattr(rp, "trainval_rows", lambda *_: pytest.fail("evaluate did work before checking --out"))
    with pytest.raises(ValueError):
        rp.evaluate(tile=224, pilot_dir=data / "images_full", folds=5, n_boot=10, out=db)
    assert db.read_bytes() == b"db"


# --------------------------------------------------------------------------- #
# Download + manifest (mocked HTTP)
# --------------------------------------------------------------------------- #
def _jpeg(w, h, seed):
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


class FakeCDN:
    def __init__(self, missing=(), flaky=()):
        self.missing, self.flaky = set(missing), dict.fromkeys(flaky, 1)
        self.requests: list[httpx.Request] = []
        self.images = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        if url in self.missing:
            return httpx.Response(404)
        if self.flaky.get(url):
            self.flaky[url] -= 1
            return httpx.Response(503)
        data = self.images.setdefault(url, _jpeg(64 + len(self.images), 96, len(self.images)))
        return httpx.Response(200, content=data, headers={"content-type": "application/octet-stream"})


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def _plan():
    return [
        ("1", "front", "https://cdn.example/1_f.jpg"),
        ("1", "back", "https://cdn.example/1_b.jpg"),
        ("2", "front", "https://cdn.example/2_f.jpg"),
        ("2", "back", "https://cdn.example/2_b.jpg"),
    ]


def test_download_writes_files_and_manifest(tmp_path):
    cdn = FakeCDN(missing={"https://cdn.example/2_b.jpg"}, flaky={"https://cdn.example/1_b.jpg"})
    clock = FakeClock()
    with rp.cdn_client(transport=httpx.MockTransport(cdn)) as client:
        stats = rp.download_originals(_plan(), tmp_path, client, rate=2.0, sleep=clock.sleep, clock=clock)
    assert stats == {"planned": 4, "fetched": 3, "skipped": 0, "recovered": 0, "failed": 1, "bytes": stats["bytes"]}
    m = rp.read_manifest(tmp_path / "manifest.csv")
    assert set(m) == {("1", "front"), ("1", "back"), ("2", "front"), ("2", "back")}
    for (cert, face), row in m.items():
        if (cert, face) == ("2", "back"):
            assert row["status"] == "failed" and row["reason"] == "http 404" and row["file"] == ""
            assert not (tmp_path / "2_back.jpg").exists()
            continue
        f = tmp_path / row["file"]
        assert row["status"] == "ok" and row["reason"] == ""
        assert row["sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()
        assert int(row["bytes"]) == f.stat().st_size
        with Image.open(f) as im:
            assert (int(row["width"]), int(row["height"])) == im.size
    # No token or cookie ever reaches the CDN.
    assert all("authorization" not in r.headers and "cookie" not in r.headers for r in cdn.requests)
    # 503 retried once; requests spaced by >= 1/rate (the fake clock only moves by sleeping).
    assert len(cdn.requests) == 5
    assert sum(s for s in clock.sleeps) >= 0.5 * 3
    assert not list(tmp_path.glob("*.part")) and not list(tmp_path.glob("*.tmp"))


def test_download_resumes_without_refetching(tmp_path):
    cdn = FakeCDN(missing={"https://cdn.example/2_b.jpg"})
    clock = FakeClock()
    with rp.cdn_client(transport=httpx.MockTransport(cdn)) as client:
        rp.download_originals(_plan(), tmp_path, client, sleep=clock.sleep, clock=clock)
        first = len(cdn.requests)
        cdn.missing.clear()
        stats = rp.download_originals(_plan(), tmp_path, client, sleep=clock.sleep, clock=clock)
    assert stats["skipped"] == 3 and stats["fetched"] == 1 and stats["failed"] == 0
    assert len(cdn.requests) == first + 1
    assert str(cdn.requests[-1].url).endswith("2_b.jpg")
    assert rp.read_manifest(tmp_path / "manifest.csv")[("2", "back")]["status"] == "ok"


def test_download_recovers_unlisted_file_and_refetches_corrupt_one(tmp_path):
    cdn = FakeCDN()
    clock = FakeClock()
    with rp.cdn_client(transport=httpx.MockTransport(cdn)) as client:
        rp.download_originals(_plan(), tmp_path, client, sleep=clock.sleep, clock=clock)
        n = len(cdn.requests)
        # A file written but never recorded (killed between write and manifest).
        entries = rp.read_manifest(tmp_path / "manifest.csv")
        del entries[("1", "front")]
        rp.write_manifest(tmp_path / "manifest.csv", entries)
        # A file changed on disk after it was recorded.
        (tmp_path / "2_front.jpg").write_bytes(_jpeg(10, 10, 99))
        stats = rp.download_originals(_plan(), tmp_path, client, sleep=clock.sleep, clock=clock)
    assert stats["recovered"] == 1 and stats["fetched"] == 1 and stats["skipped"] == 2
    assert len(cdn.requests) == n + 1 and str(cdn.requests[-1].url).endswith("2_f.jpg")
    m = rp.read_manifest(tmp_path / "manifest.csv")
    for (cert, face), row in m.items():
        assert row["sha256"] == hashlib.sha256((tmp_path / f"{cert}_{face}.jpg").read_bytes()).hexdigest()


def test_fetch_refuses_to_send_a_token():
    cdn = FakeCDN()
    with httpx.Client(transport=httpx.MockTransport(cdn), headers={"Authorization": "bearer secret"}) as c, pytest.raises(RuntimeError, match="authorization"):
        rp.fetch(c, "https://cdn.example/1_f.jpg", sleep=lambda s: None)


@pytest.mark.parametrize(
    "client_kwargs",
    [
        {"headers": {"Authorization": "bearer secret"}},
        {"headers": {"X-API-Key": "secret"}},
        {"headers": {"Cookie": "session=secret"}},
        {"cookies": {"session": "secret"}},
    ],
)
def test_fetch_never_sends_a_request_with_a_forbidden_header(client_kwargs):
    calls: list[httpx.Request] = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=_jpeg(8, 8, 0))

    with httpx.Client(transport=httpx.MockTransport(handler), **client_kwargs) as c, pytest.raises(RuntimeError):
        rp.fetch(c, "https://cdn.example/1_f.jpg", sleep=lambda s: None)
    assert calls == []  # audited before the transport ever sees it


def test_undecodable_download_is_failed_not_saved(tmp_path):
    def handler(request):
        return httpx.Response(200, content=b"<html>not an image</html>")

    with rp.cdn_client(transport=httpx.MockTransport(handler)) as client:
        stats = rp.download_originals(_plan()[:1], tmp_path, client, sleep=lambda s: None)
    row = rp.read_manifest(tmp_path / "manifest.csv")[("1", "front")]
    assert stats["failed"] == 1 and row["status"] == "failed" and row["reason"].startswith("undecodable")
    assert not (tmp_path / "1_front.jpg").exists()


def test_manifest_summary_lists_failures():
    entries = {
        ("1", "front"): {"cert": "1", "face": "front", "status": "ok", "bytes": "100", "width": "10", "height": "20"},
        ("1", "back"): {"cert": "1", "face": "back", "status": "failed", "reason": "http 404"},
    }
    s = rp.manifest_summary(entries)
    assert s["files_ok"] == 1 and s["files_failed"] == 1 and s["total_bytes"] == 100
    assert s["failed"] == [{"cert": "1", "face": "back", "reason": "http 404"}]


# --------------------------------------------------------------------------- #
# Tile features (stub model)
# --------------------------------------------------------------------------- #
def test_tiles_to_tensor_matches_to_input():
    img = np.random.default_rng(0).integers(0, 255, (FEAT_H, FEAT_W, 3), dtype=np.uint8)
    got = rp.tiles_to_tensor([img], "cpu")[0].numpy()
    np.testing.assert_allclose(got, to_input(img), atol=1e-6)


class MeanPixel(torch.nn.Module):
    """(B, 3, H, W) -> (B, 3): the mean normalised pixel, a stand-in for DINOv2."""

    def forward(self, x):
        return x.mean((2, 3))


def test_region_features_average_tiles_per_region():
    rng = np.random.default_rng(0)
    faces = [
        [(0, rng.integers(0, 255, (28, 28, 3), dtype=np.uint8))]
        + [(ri, rng.integers(0, 255, (14, 28, 3), dtype=np.uint8)) for ri in range(1, 8) for _ in range(2)],
        [(ri, rng.integers(0, 255, (28, 14, 3), dtype=np.uint8)) for ri in range(8)],
    ]
    out = rp.region_features(faces, MeanPixel(), "cpu")
    assert out.shape == (2, 8, 3)
    for fi, face in enumerate(faces):
        for ri in range(8):
            want = np.mean(
                [rp.tiles_to_tensor([t], "cpu")[0].numpy().mean((1, 2)) for r, t in face if r == ri], axis=0
            )
            np.testing.assert_allclose(out[fi, ri], want, atol=1e-5)


def test_region_features_need_every_region():
    face = [(0, np.zeros((28, 28, 3), np.uint8))]
    with pytest.raises(ValueError):
        rp.region_features([face], MeanPixel(), "cpu")


def test_extract_tile_features_caches_by_key_and_spec(tmp_path):
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 255, (300, 220, 3), dtype=np.uint8) for _ in range(6)]
    quad = np.array([[20, 30], [200, 30], [200, 280], [20, 280]], np.float32)
    calls = []

    def factory():
        calls.append(1)
        return MeanPixel()

    def jobs(n):
        return [((str(i), "f", "b", "1", "1"), lambda i=i: [(imgs[2 * i], quad), (imgs[2 * i + 1], quad)]) for i in range(n)]

    cache = tmp_path / "cache" / "tiles.npz"
    spec = rp.TileSpec(28)
    a = rp.extract_tile_features(jobs(2), spec, cache, model_factory=factory, chunk=1, workers=2)
    assert a.shape == (2, 2, 8, 3) and len(calls) == 1
    b = rp.extract_tile_features(jobs(3), spec, cache, model_factory=factory, chunk=2, workers=1)
    assert len(calls) == 2  # only cert 2 extracted
    np.testing.assert_allclose(b[:2], a, atol=1e-2)  # float16 cache
    rp.extract_tile_features(jobs(3), spec, cache, model_factory=factory)
    assert len(calls) == 2  # all cached
    rp.extract_tile_features(jobs(3), rp.TileSpec(42), cache, model_factory=factory)
    assert len(calls) == 3  # another tile spec: re-extracted


def test_with_tiles_replaces_only_corners_and_edges():
    a = np.zeros((3, 2, 11, 4), np.float32)
    t = np.ones((3, 2, 8, 4), np.float32)
    out = rp.with_tiles(a, t)
    assert (out[:, :, 2:10] == 1).all() and (out[:, :, :2] == 0).all() and (out[:, :, 10] == 0).all()
    assert (a == 0).all()


# --------------------------------------------------------------------------- #
# Metrics, bootstrap, decision
# --------------------------------------------------------------------------- #
def test_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.random(200) < 0.4
    s = np.round(rng.normal(size=200) + y, 1)  # ties
    assert rp.auc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert np.isnan(rp.auc(np.ones(5, bool), np.arange(5)))


def _preds(y, noise, seed):
    rng = np.random.default_rng(seed)
    soft = y + rng.normal(0, noise, len(y))
    hard = np.clip(np.round(soft), 9, 10).astype(int)
    return {"soft": soft, "hard": hard, "corners": soft, "edges": soft}


def test_paired_group_bootstrap():
    rng = np.random.default_rng(0)
    y = rng.choice([9, 10], 300)
    groups = np.array([f"g{i // 2}" for i in range(300)])
    good, bad = _preds(y, 0.2, 1), _preds(y, 3.0, 2)
    out = rp.paired_group_bootstrap(y, groups, {"A": bad, "B": bad, "C": good}, [("B", "A"), ("C", "A")], n_boot=100)
    assert out["n_groups"] == 150
    same = out["pairs"]["B-A"]["auc_10_vs_9"]
    assert same["diff"] == 0 and same["ci95"] == [0.0, 0.0]
    better = out["pairs"]["C-A"]["auc_10_vs_9"]
    assert better["diff"] > 0.2 and better["ci95"][0] > 0 and better["p_le_0"] == 0
    assert set(out["pairs"]["C-A"]) == set(rp.METRICS)


def _boot(ca, ca_lo, ba):
    return {
        "pairs": {
            "C-A": {"auc_10_vs_9": {"diff": ca, "ci95": [ca_lo, ca + 0.05], "p_le_0": 0.0}},
            "B-A": {"auc_10_vs_9": {"diff": ba, "ci95": [ba - 0.03, ba + 0.03], "p_le_0": 0.5}},
        }
    }


@pytest.mark.parametrize(
    ("ca", "ca_lo", "ba", "call"),
    [
        (0.05, 0.01, 0.0, "go_download"),
        (0.05, 0.01, 0.03, "go_no_download"),
        (0.05, -0.01, 0.0, "stop"),
        (0.02, 0.005, 0.0, "stop"),
    ],
)
def test_decide(ca, ca_lo, ba, call):
    assert rp.decide(_boot(ca, ca_lo, ba))["call"] == call
