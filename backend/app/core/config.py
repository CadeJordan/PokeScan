"""Application settings, loaded from environment / .env."""

from __future__ import annotations

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
    db_path: Path = PROJECT_ROOT / "data" / "pokescan.sqlite"
    models_dir: Path = PROJECT_ROOT / "data" / "models"
    active_model_path: Path = PROJECT_ROOT / "data" / "models" / "grade_model.onnx"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000"

    wandb_project: str = "pokescan"
    wandb_mode: str = "disabled"

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
        try:
            sources.update(dotenv_values(PROJECT_ROOT / ".env"))
        except OSError:
            pass
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
        for p in (self.data_dir, self.images_dir, self.models_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
