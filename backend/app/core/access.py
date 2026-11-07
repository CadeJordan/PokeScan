"""Shared access code for /api/grade (`settings.access_code`, intent/cloud-run-deploy).

A plain ASGI middleware rather than a route dependency: FastAPI parses the
multipart body before it resolves dependencies, and a missing or wrong code
must be refused before the upload is read or a grading slot is taken.
"""

from __future__ import annotations

import hmac

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

ACCESS_CODE_HEADER = b"x-access-code"
PROTECTED_PATHS = frozenset({"/api/grade"})


class AccessCodeMiddleware:
    """401 "access code required" for a protected path without the right code.

    Every other path (e.g. /api/health) passes through. Add it before
    `CORSMiddleware`, so CORS stays outermost: preflights are answered there and
    the 401 carries the CORS headers a browser needs to read it.
    """

    def __init__(self, app: ASGIApp, code: str) -> None:
        if not code:
            raise ValueError("AccessCodeMiddleware needs a non-empty code")
        self.app = app
        self._code = code.encode("utf-8")

    def _allowed(self, scope: Scope) -> bool:
        given = next((v for k, v in scope["headers"] if k.lower() == ACCESS_CODE_HEADER), None)
        return given is not None and hmac.compare_digest(given, self._code)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] == "http"
            and scope["path"] in PROTECTED_PATHS
            and not self._allowed(scope)
        ):
            response = JSONResponse({"detail": "access code required"}, status_code=401)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)
