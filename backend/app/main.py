"""FastAPI entrypoint.

Run:
    uvicorn backend.app.main:app --reload
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.grade import router as grade_router
from backend.app.core.access import PROTECTED_PATHS, AccessCodeMiddleware
from backend.app.core.config import get_settings
from backend.app.core.rate_limit import Clock, GradeRateLimitMiddleware, SlidingWindowLimiter
from backend.app.ml.inference import get_inference_service

log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    log.info("starting PokeScan API on %s:%d", settings.api_host, settings.api_port)
    get_inference_service()
    yield


def create_app(clock: Clock | None = None) -> FastAPI:
    """Build the app. `clock` drives the rate limiters (tests pass a fake one)."""
    settings = get_settings()
    app = FastAPI(
        title="PokeScan",
        description="Predict the PSA grade a Pokemon card will receive.",
        version="0.1.0",
        lifespan=_lifespan,
    )
    # Middleware added later wraps what was added before: the order of
    # requests is CORS -> access code (+ wrong-code limit) -> grade limit -> app.
    # CORS stays outermost, so 401s and 429s carry its headers.
    if settings.grades_per_hour > 0:
        limit = settings.grades_per_hour
        app.add_middleware(
            GradeRateLimitMiddleware,
            limiter=SlidingWindowLimiter(limit, 3600.0, clock=clock),
            trust_forwarded_for=settings.trust_forwarded_for,
            paths=PROTECTED_PATHS,
            detail=f"grade limit reached ({limit} per hour), try again later",
        )
    # Stripped: a stray space or newline in the env var would otherwise lock
    # everyone out (header values arrive trimmed); blank means no code.
    access_code = settings.access_code.strip()
    if access_code:
        attempts = settings.code_attempts_per_minute
        app.add_middleware(
            AccessCodeMiddleware,
            code=access_code,
            attempts=SlidingWindowLimiter(attempts, 60.0, clock=clock) if attempts > 0 else None,
            trust_forwarded_for=settings.trust_forwarded_for,
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list or ["*"],
        allow_origin_regex=settings.cors_origin_regex or None,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        # A browser client can read when to retry a 503 "server busy" or a 429.
        expose_headers=["Retry-After"],
    )
    app.include_router(grade_router, prefix="/api")
    return app


app = create_app()
