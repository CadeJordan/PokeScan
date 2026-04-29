"""Cert harvester: walks a range of PSA cert numbers, filters to Pokemon, and
saves images + metadata.

Usage:
    python -m ml.data.collect --start-cert 90000000 --count 5000

Quota-aware: stops cleanly when the local daily quota is exhausted so you can
resume tomorrow without losing progress.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import typer
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from backend.app.core.config import get_settings
from ml.data.db import connect, pokemon_cert_count, upsert_cert
from ml.data.image_io import save_compressed_jpeg
from ml.data.psa_client import PSAClient, PSAQuotaExhaustedError, PSARateLimitError

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True)])
# httpx logs every request at INFO; with hundreds of 404s in a row that's
# noise that drowns out the useful warnings. Bump it down to WARNING.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("collect")

app = typer.Typer(add_completion=False, help="Harvest PSA certs into local SQLite + images.")


async def _process_cert(
    psa: PSAClient,
    cert_number: int,
    images_dir: Path,
    sem: asyncio.Semaphore,
) -> dict | None:
    async with sem:
        details = await psa.get_cert_details(cert_number)
        if details is None:
            return None
        row: dict = {
            "cert_number": details.cert_number,
            "brand_title": details.brand_title,
            "subject": details.subject,
            "year": details.year,
            "card_number": details.card_number,
            "category": details.category,
            "grade_label": details.grade_label,
            "grade_int": details.grade_int,
            "is_pokemon": int(details.is_pokemon),
            "front_url": None,
            "back_url": None,
            "front_path": None,
            "back_path": None,
            "has_images": 0,
            "split": None,
            "raw_json": json.dumps(details.raw),
        }
        if not details.is_pokemon or details.grade_int is None:
            return row

        images = await psa.get_cert_images(cert_number)
        if images is None:
            return row
        row["front_url"] = images.front_url
        row["back_url"] = images.back_url

        front_path = images_dir / f"{details.cert_number}_front.jpg"
        back_path = images_dir / f"{details.cert_number}_back.jpg"

        got_front = False
        if images.front_url:
            raw = await psa.download_image_bytes(images.front_url)
            if raw is not None:
                save_compressed_jpeg(raw, front_path)
                got_front = True

        got_back = False
        if images.back_url:
            raw = await psa.download_image_bytes(images.back_url)
            if raw is not None:
                save_compressed_jpeg(raw, back_path)
                got_back = True

        if got_front:
            row["front_path"] = str(front_path.relative_to(images_dir.parent))
        if got_back:
            row["back_path"] = str(back_path.relative_to(images_dir.parent))
        row["has_images"] = int(got_front and got_back)
        return row


async def _run(
    start_cert: int,
    count: int,
    concurrency: int,
    daily_limit: int,
    abort_after_misses: int,
    token_index: int | None,
) -> None:
    settings = get_settings()
    try:
        token = settings.get_psa_token(token_index)
    except (RuntimeError, IndexError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    available = sorted(settings.psa_api_tokens)
    label = f"PSA_API_TOKEN{token_index}" if token_index is not None else "PSA_API_TOKEN (default)"
    log.info("using %s (available indices: %s)", label, available or "[]")

    sem = asyncio.Semaphore(concurrency)
    # PSAClient is an *async* context manager, while `connect()` returns a
    # sync sqlite context manager. They cannot be combined in a single
    # `async with` line, so we enter them separately.
    async with PSAClient(token, daily_limit=daily_limit) as psa:
        with connect(settings.db_path) as conn:
            progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
            )
            with progress:
                task = progress.add_task("scanning certs", total=count)
                inflight: list[asyncio.Task] = []
                quota_exhausted = False
                consecutive_404s = 0
                pokemon_seen = 0
                hits_seen = 0

                async def _drain(limit: int = 0) -> None:
                    nonlocal quota_exhausted, consecutive_404s, pokemon_seen, hits_seen
                    while inflight and len(inflight) > limit:
                        done, _pending = await asyncio.wait(
                            inflight, return_when=asyncio.FIRST_COMPLETED
                        )
                        for t in done:
                            inflight.remove(t)
                            try:
                                row = t.result()
                            except (PSARateLimitError, PSAQuotaExhaustedError):
                                quota_exhausted = True
                                continue
                            except Exception as exc:  # noqa: BLE001
                                log.error("cert task failed: %s", exc)
                                continue
                            if row is None:
                                consecutive_404s += 1
                            else:
                                consecutive_404s = 0
                                hits_seen += 1
                                if row.get("is_pokemon"):
                                    pokemon_seen += 1
                                upsert_cert(conn, row)
                            progress.advance(task)
                        if quota_exhausted:
                            return

                async def _cancel_inflight() -> None:
                    if not inflight:
                        return
                    log.info("cancelling %d in-flight tasks", len(inflight))
                    for t in inflight:
                        t.cancel()
                    # Swallow cancellation + already-raised quota errors so
                    # asyncio doesn't print "Task exception was never retrieved".
                    await asyncio.gather(*inflight, return_exceptions=True)
                    inflight.clear()

                try:
                    for cert in range(start_cert, start_cert + count):
                        if quota_exhausted or psa.calls_remaining <= 0:
                            log.warning("local quota exhausted before reaching %d", cert)
                            break
                        inflight.append(
                            asyncio.create_task(
                                _process_cert(psa, cert, settings.images_dir, sem)
                            )
                        )
                        if len(inflight) >= concurrency * 4:
                            await _drain(concurrency * 2)
                        if (
                            abort_after_misses > 0
                            and hits_seen == 0
                            and consecutive_404s >= abort_after_misses
                        ):
                            log.error(
                                "%d consecutive 404s and zero hits - aborting to "
                                "preserve quota. Cert range %d+ is probably above "
                                "the issued range. Try a lower --start-cert; recent "
                                "Pokemon certs are typically in the 70M-100M range.",
                                consecutive_404s, start_cert,
                            )
                            break
                    await _drain(0)
                finally:
                    await _cancel_inflight()

                if quota_exhausted:
                    log.warning("daily quota exhausted - stopped cleanly")

            n_pokemon = pokemon_cert_count(conn)
            log.info(
                "done. cert hits this run: %d (pokemon: %d). pokemon w/ images in DB: %d",
                hits_seen, pokemon_seen, n_pokemon,
            )


@app.command()
def main(
    start_cert: int = typer.Option(..., help="First cert number to scan."),
    count: int = typer.Option(1000, help="How many cert numbers to scan."),
    concurrency: int = typer.Option(4, help="Concurrent in-flight API requests."),
    daily_limit: int = typer.Option(100, help="Local PSA daily quota cap."),
    abort_after_misses: int = typer.Option(
        25,
        help="If we hit this many consecutive 404s with zero data, bail out to "
        "preserve quota. Set to 0 to disable.",
    ),
    token_index: int | None = typer.Option(
        None,
        "--token-index",
        "-t",
        help="Which PSA_API_TOKEN<N> from .env to use (e.g. 0, 1, 2). "
        "Omit to use the lowest-indexed token (or legacy PSA_API_TOKEN).",
    ),
) -> None:
    asyncio.run(
        _run(start_cert, count, concurrency, daily_limit, abort_after_misses, token_index)
    )


if __name__ == "__main__":
    app()
