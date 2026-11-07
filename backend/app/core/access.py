"""Shared access code for /api/grade (`settings.access_code`, intent/cloud-run-deploy).

A plain ASGI middleware rather than a route dependency: FastAPI parses the
multipart body before it resolves dependencies, and a missing or wrong code
must be refused before the upload is read or a grading slot is taken.

It also limits wrong codes per client IP (`settings.code_attempts_per_minute`,
intent/rate-limits-and-photo-picker): each missing or wrong code counts as one
failure, and an IP at the limit gets a 429 for every protected request, even
with the right code, until its oldest failure leaves the window.
"""

from __future__ import annotations

import hmac

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from backend.app.core.rate_limit import SlidingWindowLimiter, client_ip, too_many_requests

ACCESS_CODE_HEADER = b"x-access-code"
PROTECTED_PATHS = frozenset({"/api/grade"})
TOO_MANY_ATTEMPTS = "too many attempts, wait a minute"


class AccessCodeMiddleware:
    """401 "access code required" for a protected path without the right code.

    Every other path (e.g. /api/health) passes through. Add it before
    `CORSMiddleware`, so CORS stays outermost: preflights are answered there and
    the 401/429 carry the CORS headers a browser needs to read them.

    With `attempts` (a limiter over client IPs), a locked-out IP gets a 429
    with `Retry-After` before its code is even compared.
    """

    def __init__(
        self,
        app: ASGIApp,
        code: str,
        attempts: SlidingWindowLimiter | None = None,
        trust_forwarded_for: bool = False,
    ) -> None:
        if not code:
            raise ValueError("AccessCodeMiddleware needs a non-empty code")
        self.app = app
        self._code = code.encode("utf-8")
        self.attempts = attempts
        self.trust_forwarded_for = trust_forwarded_for

    def _allowed(self, scope: Scope) -> bool:
        given = next((v for k, v in scope["headers"] if k.lower() == ACCESS_CODE_HEADER), None)
        return given is not None and hmac.compare_digest(given, self._code)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] not in PROTECTED_PATHS:
            await self.app(scope, receive, send)
            return
        ip = client_ip(scope, self.trust_forwarded_for) if self.attempts is not None else None
        if self.attempts is not None:
            wait = self.attempts.retry_after(ip)
            if wait is not None:
                await too_many_requests(TOO_MANY_ATTEMPTS, wait)(scope, receive, send)
                return
        if not self._allowed(scope):
            if self.attempts is not None:
                self.attempts.record(ip)
            response = JSONResponse({"detail": "access code required"}, status_code=401)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
