"""FastAPI entrypoint.

Run:
    uvicorn backend.app.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.grade import router as grade_router
from backend.app.core.config import get_settings
from backend.app.ml.inference import get_inference_service

log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    log.info("starting PokeScan API on %s:%d", settings.api_host, settings.api_port)
    get_inference_service()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="PokeScan",
        description="Predict the PSA grade a Pokemon card will receive.",
        version="0.1.0",
        lifespan=_lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list or ["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(grade_router, prefix="/api")
    return app


app = create_app()
