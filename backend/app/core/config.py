"""Application settings, loaded from environment / .env."""

from __future__ import annotations

import contextlib
import os
import re
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_TOKEN_INDEX_RE = re.compile(r"^PSA_API_TOKEN(\d+)$")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Single legacy token. PSA_API_TOKEN0..N are loaded separately via
    # `psa_api_tokens` so each can be selected by index for daily rotation.
    psa_api_token: str = Field(default="", description="PSA Public API bearer token (legacy / single)")
    psa_api_base_url: str = "https://api.psacard.com/publicapi"

    ebay_app_id: str = Field(default="", description="eBay App ID (client_id)")
    ebay_cert_id: str = Field(default="", description="eBay Cert ID (client secret)")
    ebay_oauth_scope: str = "https://api.ebay.com/oauth/api_scope"
    ebay_oauth_url: str = "https://api.ebay.com/identity/v1/oauth2/token"
    ebay_api_base_url: str = "https://api.ebay.com/buy/browse/v1"
    ebay_marketplace: str = "EBAY_US"

    data_dir: Path = PROJECT_ROOT / "data"
    images_dir: Path = PROJECT_ROOT / "data" / "images"
    crops_dir: Path = PROJECT_ROOT / "data" / "crops"
    db_path: Path = PROJECT_ROOT / "data" / "pokescan.sqlite"
    models_dir: Path = PROJECT_ROOT / "data" / "models"
    # The probe grade model served by the API (see ml/inference.py).
    grade_head_path: Path = PROJECT_ROOT / "data" / "models" / "grade_head.npz"
    feature_model_path: Path = PROJECT_ROOT / "data" / "models" / "card_features.onnx"

    # Grades run at once per API process (`api/grade.py`); a request over the
    # limit waits up to `grade_queue_timeout_s` for a slot, then gets a 503.
    # Measured (intent/archive/serving-hardening): each extra concurrent grade adds only
    # ~20-25 MB, and 1 at a time costs ~15% throughput, so the default is 4.
    # Running uvicorn with `--workers N` multiplies it.
    grade_concurrency: int = Field(default=4, ge=1)
    grade_queue_timeout_s: float = Field(default=30.0, gt=0)

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000"
    # Optional regex of extra allowed origins (e.g. Vercel preview URLs:
    # `https://pokescan[-a-z0-9]*\.vercel\.app`), matched against the whole
    # origin. Empty means only the exact `cors_origins` list.
    cors_origin_regex: str = ""
    # Optional shared code: when set, /api/grade needs it in the `X-Access-Code`
    # header (401 otherwise). Empty means open, as before (local dev, tests).
    access_code: str = ""
    # Per-IP rate limits on /api/grade (`core/rate_limit.py`), in this process's
    # memory. Wrong (or missing) access codes per minute before the IP gets 429s,
    # even with the right code; only applies when `access_code` is set (0 = off).
    code_attempts_per_minute: int = Field(default=5, ge=0)
    # Grades per hour per IP (requests that reach a grading slot with a readable
    # image); 0 = off, so local dev and tests are unlimited. The deploy sets 30.
    grades_per_hour: int = Field(default=0, ge=0)
    # Take the client IP from the right-most `X-Forwarded-For` entry (the one
    # Cloud Run's front end appends) instead of the TCP peer. Only turn on
    # behind a proxy that appends it, or clients can pick their own IP.
    trust_forwarded_for: bool = False

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def psa_api_tokens(self) -> dict[int, str]:
        """All PSA tokens defined in the environment / .env, keyed by index.

        Reads any var matching ``PSA_API_TOKEN<N>`` from both the OS
        environment and the project ``.env`` file (env vars win on conflict).
        Indices may be sparse - we only return what's actually defined.
        """
        sources: dict[str, str | None] = {}
        # .env values first (lowest precedence).
        with contextlib.suppress(OSError):
            sources.update(dotenv_values(PROJECT_ROOT / ".env"))
        # Real env vars override.
        sources.update(os.environ)

        out: dict[int, str] = {}
        for key, raw in sources.items():
            m = _TOKEN_INDEX_RE.match(key)
            if not m or raw is None:
                continue
            cleaned = raw.strip()
            if cleaned:
                out[int(m.group(1))] = cleaned
        return out

    def get_psa_token(self, index: int | None = None) -> str:
        """Return one PSA token. ``index`` selects ``PSA_API_TOKEN<index>``;
        if omitted, returns the lowest-indexed token, or the legacy
        ``PSA_API_TOKEN`` value if no indexed tokens exist."""
        tokens = self.psa_api_tokens
        if index is not None:
            if index in tokens:
                return tokens[index]
            available = sorted(tokens) if tokens else []
            raise IndexError(
                f"PSA_API_TOKEN{index} not set. Available indices: {available}"
            )
        if tokens:
            return tokens[min(tokens)]
        if self.psa_api_token:
            return self.psa_api_token.strip()
        raise RuntimeError(
            "No PSA API token found. Define PSA_API_TOKEN0..N (or the "
            "legacy PSA_API_TOKEN) in your .env."
        )

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.images_dir, self.crops_dir, self.models_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
