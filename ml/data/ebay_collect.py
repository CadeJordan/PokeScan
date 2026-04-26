"""Harvest PSA-graded Pokemon listings from eBay.

For each grade we want, we run a Browse-API search like
    "PSA 9 pokemon"
filter to the Trading Card Singles category, and walk the results pages.
Titles are parsed for an unambiguous PSA grade (the one we asked for, with
no conflicting numbers), and listings that look like multi-card lots are
skipped. Then for each surviving listing we:

  1. Fetch full item detail (gives us all image URLs).
  2. Rewrite each URL to the largest available variant (s-l1600).
  3. Download every candidate image.
  4. Pass the candidates through ``side_classifier.pick_front_back`` to
     identify which is the front and which is the back.
  5. Run both chosen images through ``quality_filter.assess_image``
     (Laplacian-variance blur + min-resolution).
  6. Upsert the surviving pair into the SQLite ``certs`` table with
     ``source='ebay'``.

Usage:
    python -m ml.data.ebay_collect --grades 8 9 10 --per-grade 500
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from backend.app.core.config import get_settings
from ml.data.db import cert_exists, connect, pokemon_cert_count, upsert_cert
from ml.data.ebay_client import EBayClient, EBayItemSummary, upscale_ebay_image_url
from ml.data.quality_filter import assess_image
from ml.data.side_classifier import pick_front_back

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True)])
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("ebay-collect")

app = typer.Typer(add_completion=False, help="Harvest PSA-graded Pokemon listings from eBay.")

# Trading Card Singles - eBay leaf category for individual cards in the US.
# Lots and accessories live under different categories so we lock to this one.
EBAY_CATEGORY_TCG_SINGLES = "183454"

# Words that signal the listing is a lot of multiple cards, not a single graded
# slab. We err on the side of skipping; false-positives are cheaper than
# bad training data.
_LOT_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\blot\b",
        r"\bbundle\b",
        r"\bcollection\b",
        r"\bcomplete set\b",
        r"\bset of\b",
        r"\bx\s*\d{2,}\b",       # x10, x100, x1000
        r"\(\s*\d{2,}\s*\)",     # (10), (100)
        r"\bpack\b",
        r"\bbooster\b",
        r"\bsealed\b",
    )
]

# A title that mentions a *different* PSA grade than the one we asked for is
# almost certainly a lot or comparison listing.
_PSA_GRADE_RE = re.compile(r"\bPSA\s*(\d{1,2}(?:\.\d)?)\b", re.IGNORECASE)
_BGS_OR_CGC_RE = re.compile(r"\b(BGS|CGC|SGC)\s*\d", re.IGNORECASE)
_POKEMON_RE = re.compile(r"\bpok[eé]mon\b", re.IGNORECASE)


@dataclass(slots=True)
class ListingDecision:
    keep: bool
    grade: int | None
    reason: str


def _classify_listing(title: str, expected_grade: int) -> ListingDecision:
    if not _POKEMON_RE.search(title):
        return ListingDecision(False, None, "not pokemon")
    if _BGS_OR_CGC_RE.search(title):
        return ListingDecision(False, None, "different grader")
    grades_in_title = _PSA_GRADE_RE.findall(title)
    if not grades_in_title:
        return ListingDecision(False, None, "no PSA grade in title")

    parsed = []
    for g in grades_in_title:
        try:
            parsed.append(int(round(float(g))))
        except ValueError:
            continue
    if not parsed:
        return ListingDecision(False, None, "unparseable grade")

    unique_grades = {g for g in parsed if 1 <= g <= 10}
    if len(unique_grades) > 1:
        return ListingDecision(False, None, f"multiple grades in title: {sorted(unique_grades)}")
    grade = next(iter(unique_grades))
    if grade != expected_grade:
        return ListingDecision(False, None, f"unexpected grade {grade} (asked for {expected_grade})")

    for pat in _LOT_PATTERNS:
        if pat.search(title):
            return ListingDecision(False, None, f"looks like a lot ({pat.pattern})")
    return ListingDecision(True, grade, "ok")


# Cap on how many candidate images to download per listing. Most graded
# listings have 2-6 images; we hard-cap to bound network usage on the
# occasional 12-image listing.
_MAX_CANDIDATES = 8


async def _resolve_candidate_urls(
    client: EBayClient, summary: EBayItemSummary
) -> list[str]:
    """Get the full deduplicated list of image URLs for a listing,
    upscaled to the largest available eBay variant."""
    seen: set[str] = set()
    urls: list[str] = []

    def _add(url: str | None) -> None:
        if not url:
            return
        upscaled = upscale_ebay_image_url(url)
        if upscaled not in seen:
            seen.add(upscaled)
            urls.append(upscaled)

    for url in summary.all_images:
        _add(url)

    # The per-item endpoint usually exposes more images than the summary.
    detail = await client.get_item(summary.item_id)
    if detail is not None:
        _add((detail.get("image") or {}).get("imageUrl"))
        for img in detail.get("additionalImages") or []:
            _add(img.get("imageUrl"))

    return urls[:_MAX_CANDIDATES]


async def _ingest_item(
    client: EBayClient,
    summary: EBayItemSummary,
    grade: int,
    images_dir: Path,
) -> tuple[dict | None, str]:
    """Download candidates, classify front/back, quality-check, and return a
    DB row dict plus a reason string for skip diagnostics."""
    candidate_urls = await _resolve_candidate_urls(client, summary)
    if len(candidate_urls) < 2:
        return None, "fewer than 2 images"

    candidate_bytes: list[bytes] = []
    candidate_urls_kept: list[str] = []
    for url in candidate_urls:
        data = await client.download_image_bytes(url)
        if data is None:
            continue
        candidate_bytes.append(data)
        candidate_urls_kept.append(url)
    if len(candidate_bytes) < 2:
        return None, "fewer than 2 downloadable images"

    pick = pick_front_back(candidate_bytes)
    if pick is None:
        return None, "no clear back side"
    front_idx, back_idx = pick

    cert_key = f"ebay:{summary.item_id}"
    front_path = images_dir / f"{summary.item_id}_front.jpg"
    back_path = images_dir / f"{summary.item_id}_back.jpg"
    front_path.parent.mkdir(parents=True, exist_ok=True)
    front_path.write_bytes(candidate_bytes[front_idx])
    back_path.write_bytes(candidate_bytes[back_idx])

    front_q = assess_image(front_path)
    back_q = assess_image(back_path)
    if not (front_q.ok and back_q.ok):
        front_path.unlink(missing_ok=True)
        back_path.unlink(missing_ok=True)
        bad = "front: " + front_q.reason if not front_q.ok else "back: " + back_q.reason
        return None, f"quality reject ({bad})"

    images_root = images_dir.parent
    row = {
        "cert_number": cert_key,
        "source": "ebay",
        "brand_title": "Pokemon",
        "subject": summary.title[:200],
        "year": None,
        "card_number": None,
        "category": "Pokemon",
        "grade_label": f"PSA {grade}",
        "grade_int": grade,
        "is_pokemon": 1,
        "front_url": candidate_urls_kept[front_idx],
        "back_url": candidate_urls_kept[back_idx],
        "front_path": str(front_path.relative_to(images_root)),
        "back_path": str(back_path.relative_to(images_root)),
        "has_images": 1,
        "split": None,
        "raw_json": json.dumps(
            {
                "item_id": summary.item_id,
                "item_web_url": summary.item_web_url,
                "title": summary.title,
                "all_images": candidate_urls_kept,
                "front_index": front_idx,
                "back_index": back_idx,
                "blur_front": front_q.blur_score,
                "blur_back": back_q.blur_score,
            }
        ),
    }
    return row, "ok"


async def _harvest_grade(
    client: EBayClient,
    conn,
    grade: int,
    target: int,
    images_dir: Path,
    progress: Progress,
    task_id: int,
) -> tuple[int, dict[str, int]]:
    """Harvest up to `target` listings for the given grade.

    Returns ``(kept, skip_reasons)`` where ``skip_reasons`` is a histogram
    of why listings were dropped, useful for tuning the filters.
    """
    query = f"PSA {grade} pokemon"
    filters = [
        f"categoryIds:{{{EBAY_CATEGORY_TCG_SINGLES}}}",
        # Slabs are usually listed as Used (graded slabs aren't "New")
        # but sellers vary - leave open and rely on title parsing.
    ]
    kept = 0
    skip_reasons: dict[str, int] = {}
    offset = 0
    page_size = 200

    def _bump(reason: str) -> None:
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    while kept < target:
        items, total = await client.search_items(
            query=query,
            category_ids=[EBAY_CATEGORY_TCG_SINGLES],
            limit=page_size,
            offset=offset,
            filter_clauses=filters,
        )
        if not items:
            log.info("grade %d: no more results at offset %d (total=%d)", grade, offset, total)
            break

        for summary in items:
            if kept >= target:
                break
            decision = _classify_listing(summary.title, grade)
            if not decision.keep:
                _bump(f"title: {decision.reason}")
                continue
            if cert_exists(conn, f"ebay:{summary.item_id}"):
                _bump("already in db")
                continue
            row, reason = await _ingest_item(client, summary, grade, images_dir)
            if row is None:
                _bump(reason)
                continue
            upsert_cert(conn, row)
            kept += 1
            progress.update(task_id, advance=1)

        offset += page_size
        if offset >= total:
            break
        # eBay caps offset+limit at 10,000 for unauthorized searches.
        if offset >= 10_000:
            log.info("grade %d: reached eBay's 10k offset cap", grade)
            break

    return kept, skip_reasons


async def _run(grades: list[int], per_grade: int) -> None:
    settings = get_settings()
    if not (settings.ebay_app_id and settings.ebay_cert_id):
        raise typer.BadParameter(
            "EBAY_APP_ID / EBAY_CERT_ID missing - set them in .env (see .env.example)."
        )

    images_dir = settings.images_dir
    images_dir.mkdir(parents=True, exist_ok=True)

    progress = Progress(
        TextColumn("[bold blue]grade {task.fields[grade]}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )

    async with EBayClient(
        app_id=settings.ebay_app_id,
        cert_id=settings.ebay_cert_id,
        scope=settings.ebay_oauth_scope,
        oauth_url=settings.ebay_oauth_url,
        api_base_url=settings.ebay_api_base_url,
        marketplace=settings.ebay_marketplace,
    ) as client:
        with connect(settings.db_path) as conn, progress:
            for g in grades:
                task_id = progress.add_task(
                    "harvest", total=per_grade, grade=g
                )
                kept, skip_reasons = await _harvest_grade(
                    client, conn, g, per_grade, images_dir, progress, task_id
                )
                total_skipped = sum(skip_reasons.values())
                log.info("grade %d: kept=%d skipped=%d", g, kept, total_skipped)
                if skip_reasons:
                    summary_pairs = sorted(
                        skip_reasons.items(), key=lambda kv: kv[1], reverse=True
                    )
                    summary_str = ", ".join(f"{r}={n}" for r, n in summary_pairs[:8])
                    log.info("  skip reasons: %s", summary_str)

            n_pokemon = pokemon_cert_count(conn)
            log.info("done. pokemon certs with images in DB: %d", n_pokemon)


@app.command()
def main(
    grades: list[int] = typer.Option(
        [6, 7, 8, 9, 10],
        "--grades",
        "-g",
        help="Grades to harvest (1-10). Repeat the flag or pass multiple values.",
    ),
    per_grade: int = typer.Option(
        500, help="Target number of single-card listings to keep per grade."
    ),
) -> None:
    bad = [g for g in grades if not 1 <= g <= 10]
    if bad:
        raise typer.BadParameter(f"grades must be in 1..10, got {bad}")
    asyncio.run(_run(grades, per_grade))


if __name__ == "__main__":
    app()
