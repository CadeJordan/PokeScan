"""Async client for the eBay Buy/Browse API.

We use the **client_credentials** OAuth flow (no end-user login required)
which is sufficient for `item_summary/search` and `item/{itemId}`. The
token is fetched lazily on first use and refreshed when it nears expiry.

Docs:
- OAuth:   https://developer.ebay.com/api-docs/static/oauth-client-credentials-grant.html
- Browse:  https://developer.ebay.com/api-docs/buy/browse/resources/methods
"""

from __future__ import annotations

import base64
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import httpx
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)


# eBay serves images at fixed-name variants like `s-l140.jpg` (140px),
# `s-l500.jpg`, `s-l1200.jpg`, `s-l1600.jpg` (largest typically available).
# The Browse API often returns whichever variant the seller uploaded;
# rewriting the URL gives the largest variant for free.
_EBAY_IMG_VARIANT_RE = re.compile(r"/s-l\d+(\.\w+)$")


def upscale_ebay_image_url(url: str, target: str = "s-l1600") -> str:
    """Rewrite an eBay image URL to its largest variant.

    Falls through unchanged for non-eBay URLs.
    """
    if not url or "ebayimg.com" not in url:
        return url
    return _EBAY_IMG_VARIANT_RE.sub(rf"/{target}\1", url)


class EBayAPIError(RuntimeError):
    pass


@dataclass(slots=True)
class EBayItemSummary:
    item_id: str
    title: str
    image_url: str | None
    additional_image_urls: list[str] = field(default_factory=list)
    item_web_url: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def all_images(self) -> list[str]:
        """Primary image first, then additional images, deduplicated."""
        seen: set[str] = set()
        out: list[str] = []
        for url in [self.image_url, *self.additional_image_urls]:
            if url and url not in seen:
                seen.add(url)
                out.append(url)
        return out


class EBayClient:
    def __init__(
        self,
        app_id: str,
        cert_id: str,
        scope: str = "https://api.ebay.com/oauth/api_scope",
        oauth_url: str = "https://api.ebay.com/identity/v1/oauth2/token",
        api_base_url: str = "https://api.ebay.com/buy/browse/v1",
        marketplace: str = "EBAY_US",
        timeout_s: float = 30.0,
    ) -> None:
        if not app_id or not cert_id:
            raise ValueError("EBAY_APP_ID / EBAY_CERT_ID missing - set them in .env")
        self._app_id = app_id
        self._cert_id = cert_id
        self._scope = scope
        self._oauth_url = oauth_url
        self._api_base_url = api_base_url.rstrip("/")
        self._marketplace = marketplace
        self._token: str | None = None
        self._token_expiry: float = 0.0
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def __aenter__(self) -> "EBayClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    async def _ensure_token(self) -> str:
        # Refresh 60s before expiry to avoid edge cases.
        if self._token and time.time() < self._token_expiry - 60:
            return self._token
        creds = base64.b64encode(f"{self._app_id}:{self._cert_id}".encode()).decode()
        resp = await self._client.post(
            self._oauth_url,
            headers={
                "Authorization": f"Basic {creds}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "client_credentials", "scope": self._scope},
        )
        if resp.status_code != 200:
            raise EBayAPIError(f"oauth {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        self._token = body["access_token"]
        self._token_expiry = time.time() + float(body.get("expires_in", 7200))
        log.info("eBay OAuth token acquired (expires in %ss)", body.get("expires_in"))
        return self._token

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(httpx.TransportError),
            before_sleep=before_sleep_log(log, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                token = await self._ensure_token()
                resp = await self._client.get(
                    f"{self._api_base_url}{path}",
                    params=params,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "X-EBAY-C-MARKETPLACE-ID": self._marketplace,
                        "Accept": "application/json",
                    },
                )
                if resp.status_code == 401:
                    self._token = None
                    raise httpx.TransportError("token expired")
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", "30"))
                    log.warning("eBay 429 - sleeping %.1fs", retry_after)
                    raise httpx.TransportError(f"429: retry in {retry_after}")
                if resp.status_code >= 500:
                    raise httpx.TransportError(f"upstream {resp.status_code}")
                if resp.status_code != 200:
                    raise EBayAPIError(f"{resp.status_code}: {resp.text[:300]}")
                return resp.json()
        return None

    # --- search ---------------------------------------------------------------

    async def search_items(
        self,
        query: str,
        *,
        category_ids: Iterable[str] | None = None,
        limit: int = 200,
        offset: int = 0,
        filter_clauses: Iterable[str] | None = None,
    ) -> tuple[list[EBayItemSummary], int]:
        """Run a Browse search; return (item_summaries, total_results).

        `limit` is capped at 200 by eBay. `filter_clauses` should be
        eBay-Browse-style strings, e.g. 'conditions:{NEW|USED}'.
        """
        params: dict[str, Any] = {"q": query, "limit": min(int(limit), 200), "offset": int(offset)}
        if category_ids:
            params["category_ids"] = ",".join(category_ids)
        if filter_clauses:
            params["filter"] = ",".join(filter_clauses)

        data = await self._get("/item_summary/search", params=params)
        if data is None:
            return [], 0

        items = []
        for it in data.get("itemSummaries") or []:
            items.append(
                EBayItemSummary(
                    item_id=str(it.get("itemId") or it.get("legacyItemId") or ""),
                    title=str(it.get("title") or ""),
                    image_url=(it.get("image") or {}).get("imageUrl"),
                    additional_image_urls=[
                        i.get("imageUrl")
                        for i in (it.get("additionalImages") or [])
                        if i.get("imageUrl")
                    ],
                    item_web_url=it.get("itemWebUrl"),
                    raw=it,
                )
            )
        total = int(data.get("total") or 0)
        return items, total

    async def get_item(self, item_id: str) -> dict[str, Any] | None:
        """Full item detail, including all `additionalImages`.

        item_summary/search only returns up to ~12 additional images and may
        omit some; the per-item detail endpoint gives the full list.
        """
        data = await self._get(f"/item/{item_id}", params=None)
        return data

    async def download_image(self, url: str, dest: Path) -> bool:
        data = await self.download_image_bytes(url)
        if data is None:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True

    async def download_image_bytes(self, url: str) -> bytes | None:
        try:
            resp = await self._client.get(url, follow_redirects=True)
        except httpx.HTTPError as exc:
            log.warning("image download %s failed: %s", url, exc)
            return None
        if resp.status_code != 200:
            log.warning("image download %s -> %s", url, resp.status_code)
            return None
        return resp.content
