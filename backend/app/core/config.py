"""Application settings, loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    psa_api_token: str = Field(default="", description="PSA Public API bearer token")
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

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.images_dir, self.models_dir):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
