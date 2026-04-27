"""Async client for the PSA Public API.

Endpoints used:
- GET /cert/GetByCertNumber/{cert}        - card metadata + holistic grade
- GET /cert/GetImagesByCertNumber/{cert}  - list of {ImageURL, IsFrontImage}

Free tier: 100 calls/day. We respect that with an in-process daily counter and
exponential-backoff retries on 429/5xx. Each card costs 2 calls (metadata + images)
plus 2 image downloads, but image downloads hit the CDN, not the API.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)


class PSAAPIError(RuntimeError):
    pass


class PSARateLimitError(PSAAPIError):
    pass


@dataclass(slots=True)
class PSACertImages:
    cert_number: str
    front_url: str | None
    back_url: str | None


@dataclass(slots=True)
class PSACertDetails:
    cert_number: str
    brand_title: str
    subject: str
    year: str | None
    card_number: str | None
    category: str | None
    grade_label: str            # e.g. "GEM MT 10"
    grade_int: int | None       # 1..10 parsed from grade_label
    is_dual_cert: bool
    raw: dict[str, Any]

    @property
    def is_pokemon(self) -> bool:
        haystack = " ".join(
            filter(None, [self.brand_title, self.category, self.subject])
        ).lower()
        return "pokemon" in haystack or "pokémon" in haystack


_GRADE_PARSE_TABLE = {
    "PR 1": 1, "PR1": 1,
    "FR 1.5": 1, "GD 2": 2, "GD2": 2,
    "VG 3": 3, "VG3": 3,
    "VG-EX 4": 4, "VGEX 4": 4,
    "EX 5": 5, "EX5": 5,
    "EX-MT 6": 6, "EXMT 6": 6,
    "NM 7": 7, "NM7": 7,
    "NM-MT 8": 8, "NMMT 8": 8,
    "MINT 9": 9, "MT 9": 9,
    "GEM MT 10": 10, "GEM-MT 10": 10, "GEMMT 10": 10,
}


def parse_grade(grade_label: str | None) -> int | None:
    """Parse a PSA grade string like 'GEM MT 10' -> 10. Returns None if unparseable."""
    if not grade_label:
        return None
    g = grade_label.strip().upper()
    if g in _GRADE_PARSE_TABLE:
        return _GRADE_PARSE_TABLE[g]
    # Fall back to the trailing integer if present.
    for token in reversed(g.replace("-", " ").split()):
        try:
            v = int(float(token))
            if 1 <= v <= 10:
                return v
        except ValueError:
            continue
    return None


class PSAClient:
    """Async PSA Public API client with retries + daily-quota tracking."""

    def __init__(
        self,
        token: str,
        base_url: str = "https://api.psacard.com/publicapi",
        daily_limit: int = 100,
        timeout_s: float = 30.0,
    ) -> None:
        if not token:
            raise ValueError("PSA_API_TOKEN is empty - set it in .env")
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._daily_limit = daily_limit
        self._calls_today = 0
        self._counter_day = datetime.now(tz=UTC).date()
        self._lock = asyncio.Lock()
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"bearer {token}"},
            timeout=timeout_s,
        )

    async def __aenter__(self) -> PSAClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    @property
    def calls_remaining(self) -> int:
        return max(0, self._daily_limit - self._calls_today)

    async def _bump_quota(self) -> None:
        async with self._lock:
            today = datetime.now(tz=UTC).date()
            if today != self._counter_day:
                self._counter_day = today
                self._calls_today = 0
            if self._calls_today >= self._daily_limit:
                raise PSARateLimitError(
                    f"Local daily quota exhausted ({self._daily_limit}); "
                    "wait until 00:00 UTC or use a paid token."
                )
            self._calls_today += 1

    async def _get_json(self, path: str) -> Any:
        await self._bump_quota()
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type((httpx.TransportError, PSARateLimitError)),
            before_sleep=before_sleep_log(log, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                resp = await self._client.get(path)
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", "5"))
                    log.warning("PSA 429 - sleeping %.1fs", retry_after)
                    await asyncio.sleep(retry_after)
                    raise PSARateLimitError("429 from upstream")
                if resp.status_code >= 500:
                    raise httpx.TransportError(f"upstream {resp.status_code}")
                if resp.status_code == 404:
                    return None
                if resp.status_code != 200:
                    raise PSAAPIError(f"{resp.status_code}: {resp.text[:200]}")
                if "application/json" not in resp.headers.get("content-type", ""):
                    raise PSAAPIError(f"non-JSON response from {path}")
                return resp.json()
        return None

    async def get_cert_details(self, cert_number: str | int) -> PSACertDetails | None:
        data = await self._get_json(f"/cert/GetByCertNumber/{cert_number}")
        if data is None:
            return None
        # PSA wraps the payload under "PSACert" (and sometimes "DNACert")
        cert = data.get("PSACert") or {}
        if not cert:
            return None
        grade_label = cert.get("CardGrade") or cert.get("Grade") or ""
        return PSACertDetails(
            cert_number=str(cert.get("CertNumber") or cert_number),
            # PSA's payload uses `Brand` for some categories and `BrandTitle`
            # for others - check both so we don't lose the most useful field.
            brand_title=str(cert.get("BrandTitle") or cert.get("Brand") or ""),
            subject=str(cert.get("Subject") or ""),
            year=str(cert.get("Year") or "") or None,
            card_number=str(cert.get("CardNumber") or "") or None,
            category=str(cert.get("Category") or "") or None,
            grade_label=grade_label,
            grade_int=parse_grade(grade_label),
            is_dual_cert=bool(data.get("DNACert")),
            raw=cert,
        )

    async def get_cert_images(self, cert_number: str | int) -> PSACertImages | None:
        data = await self._get_json(f"/cert/GetImagesByCertNumber/{cert_number}")
        if not data:
            return PSACertImages(str(cert_number), None, None)
        front, back = None, None
        for item in data:
            url = item.get("ImageURL")
            if not url:
                continue
            if item.get("IsFrontImage"):
                front = url
            else:
                back = url
        return PSACertImages(str(cert_number), front, back)

    async def download_image_bytes(self, url: str) -> bytes | None:
        """Download an image URL into memory. Returns None on failure.

        Image URLs are CDN, not the API, so they don't count toward the
        daily quota.
        """
        async with httpx.AsyncClient(timeout=60.0) as cdn:
            try:
                resp = await cdn.get(url)
            except httpx.HTTPError as exc:
                log.warning("image download %s failed: %s", url, exc)
                return None
        if resp.status_code != 200:
            log.warning("image download %s -> %s", url, resp.status_code)
            return None
        return resp.content

    async def download_image(self, url: str, dest: Path) -> bool:
        """Download an image URL straight to disk (uncompressed).

        Prefer ``download_image_bytes`` + ``image_io.save_compressed_jpeg``
        for ingest, which downsamples + recompresses to manageable sizes.
        """
        data = await self.download_image_bytes(url)
        if data is None:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True
