"""Per-IP rate limits for /api/grade (intent/archive/rate-limits-and-photo-picker).

Two limits, both in this process's memory (Cloud Run runs one instance with one
worker, so one counter sees every request; counts reset on a restart):

- wrong access codes, `settings.code_attempts_per_minute` per IP
  (`AccessCodeMiddleware` in `core/access.py`);
- grades, `settings.grades_per_hour` per IP (`GradeRateLimitMiddleware` here).

Both refuse with a 429 + `Retry-After` before the upload is read or a grading
slot is taken: they're plain ASGI middleware for the reason in `access.py`.

The client IP is `scope["client"]`, or with `settings.trust_forwarded_for` the
right-most `X-Forwarded-For` entry (the one Cloud Run's front end appends; a
client can only forge entries to its left). An IPv6 client is counted by its
/64: one connection usually gets a whole /64, so per-address counting would let
it rotate addresses past both limits.
"""

from __future__ import annotations

import ipaddress
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Hashable

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

FORWARDED_FOR_HEADER = b"x-forwarded-for"
UNKNOWN_CLIENT = "unknown"
# Key in the ASGI scope under which `GradeRateLimitMiddleware` hands the
# endpoint its `GradeTicket`.
GRADE_TICKET_SCOPE_KEY = "pokescan.grade_ticket"

Clock = Callable[[], float]


def _ip_key(ip: str) -> str:
    """An IPv6 address as its /64 network; anything else (IPv4, junk) unchanged."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return ip
    if addr.version == 6:
        return str(ipaddress.ip_network(f"{addr}/64", strict=False))
    return ip


def client_ip(scope: Scope, trust_forwarded_for: bool) -> str:
    """The key to count a request against: its IP, or an IPv6 IP's /64.

    With `trust_forwarded_for`, the right-most `X-Forwarded-For` entry (header
    lines joined in order). Otherwise, or with no usable entry, the TCP peer;
    with no peer either, one shared "unknown" key. The header is never read
    when `trust_forwarded_for` is off.
    """
    if trust_forwarded_for:
        values = [v.decode("latin-1") for k, v in scope.get("headers", ()) if k.lower() == FORWARDED_FOR_HEADER]
        entries = [e.strip() for e in ",".join(values).split(",")]
        entries = [e for e in entries if e]
        if entries:
            return _ip_key(entries[-1])
    client = scope.get("client")
    if client and client[0]:
        return _ip_key(str(client[0]))
    return UNKNOWN_CLIENT


class SlidingWindowLimiter:
    """At most `limit` events per key in any `window_s` seconds. Thread-safe.

    Each key keeps the times of its events inside the window (never more than
    `limit` of them: `try_acquire` doesn't record a refused event, and callers
    of `record` check `retry_after` first). Expired times are dropped whenever a
    key is touched, and every key is swept once per window, so memory is bounded
    by the keys active in the last window.
    """

    def __init__(self, limit: int, window_s: float, clock: Clock | None = None) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if window_s <= 0:
            raise ValueError("window_s must be > 0")
        self.limit = limit
        self.window_s = window_s
        self._clock: Clock = clock or time.monotonic
        self._events: dict[Hashable, deque[float]] = {}
        self._lock = threading.Lock()
        self._last_sweep = self._clock()

    def __len__(self) -> int:
        """Keys currently held (for tests and debugging)."""
        with self._lock:
            return len(self._events)

    # -- internals (lock held) ------------------------------------------------
    def _prune(self, key: Hashable, now: float) -> deque[float] | None:
        events = self._events.get(key)
        if events is None:
            return None
        cutoff = now - self.window_s
        while events and events[0] <= cutoff:
            events.popleft()
        if not events:
            del self._events[key]
            return None
        return events

    def _sweep(self, now: float) -> None:
        if now - self._last_sweep < self.window_s:
            return
        self._last_sweep = now
        for key in list(self._events):
            self._prune(key, now)

    def _wait(self, events: deque[float] | None, now: float) -> int | None:
        """Whole seconds (>= 1) until one more event fits, or None if it fits now."""
        if events is None or len(events) < self.limit:
            return None
        # The event that must expire is the limit-th most recent one.
        expires = events[-self.limit] + self.window_s
        return max(1, math.ceil(expires - now))

    # -- public API ---------------------------------------------------------
    def retry_after(self, key: Hashable) -> int | None:
        """Seconds to wait if `key` is at its limit, else None. Records nothing."""
        with self._lock:
            now = self._clock()
            self._sweep(now)
            return self._wait(self._prune(key, now), now)

    def record(self, key: Hashable) -> float:
        """Count one event for `key` now; returns its timestamp."""
        with self._lock:
            now = self._clock()
            self._sweep(now)
            events = self._prune(key, now)
            if events is None:
                events = self._events[key] = deque()
            events.append(now)
            return now

    def try_acquire(self, key: Hashable) -> tuple[float | None, int | None]:
        """Atomically count one event if it fits.

        Returns `(stamp, None)` when counted (pass `stamp` to `release` to undo
        it) or `(None, retry_after_s)` when `key` is at its limit.
        """
        with self._lock:
            now = self._clock()
            self._sweep(now)
            events = self._prune(key, now)
            wait = self._wait(events, now)
            if wait is not None:
                return None, wait
            if events is None:
                events = self._events[key] = deque()
            events.append(now)
            return now, None

    def release(self, key: Hashable, stamp: float) -> None:
        """Undo one event counted at `stamp` (no-op if it already expired)."""
        with self._lock:
            events = self._events.get(key)
            if events is None:
                return
            try:
                events.remove(stamp)
            except ValueError:
                return
            if not events:
                del self._events[key]


def too_many_requests(detail: str, retry_after_s: int) -> JSONResponse:
    return JSONResponse(
        {"detail": detail},
        status_code=429,
        headers={"Retry-After": str(max(1, int(retry_after_s)))},
    )


class GradeTicket:
    """One request's reservation against the grade limit.

    `GradeRateLimitMiddleware` reserves it before the request runs and releases
    it afterwards unless the endpoint called `count()`: `grade.py` does that
    once the request holds a grading slot.
    """

    __slots__ = ("counted",)

    def __init__(self) -> None:
        self.counted = False

    def count(self) -> None:
        self.counted = True


def grade_ticket(scope: Scope) -> GradeTicket | None:
    """The request's `GradeTicket`, or None when the grade limit is off."""
    ticket = scope.get(GRADE_TICKET_SCOPE_KEY)
    return ticket if isinstance(ticket, GradeTicket) else None


class GradeRateLimitMiddleware:
    """429 "grade limit reached" once an IP has used its grades for the window.

    Sits inside `AccessCodeMiddleware`, so a request refused for its code never
    reaches it. For each request to a limited path it atomically reserves one
    grade for the client IP, or answers 429 at once (before the upload is read
    or a slot is taken) when the IP is at its limit. The reservation stands only
    if the endpoint marked the request's `GradeTicket` as counted (it reached a
    grading slot); otherwise (413/415/empty 400/422 validation, 503 busy or not
    loaded, an error) it's released when the request ends.
    Reserving up front, rather than checking now and counting later, keeps the
    limit exact under concurrent requests.
    """

    def __init__(
        self,
        app: ASGIApp,
        limiter: SlidingWindowLimiter,
        trust_forwarded_for: bool,
        paths: frozenset[str],
        detail: str,
    ) -> None:
        self.app = app
        self.limiter = limiter
        self.trust_forwarded_for = trust_forwarded_for
        self.paths = paths
        self.detail = detail

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] not in self.paths:
            await self.app(scope, receive, send)
            return
        key = client_ip(scope, self.trust_forwarded_for)
        stamp, wait = self.limiter.try_acquire(key)
        if stamp is None:
            await too_many_requests(self.detail, wait or 1)(scope, receive, send)
            return
        ticket = GradeTicket()
        scope[GRADE_TICKET_SCOPE_KEY] = ticket
        try:
            await self.app(scope, receive, send)
        finally:
            if not ticket.counted:
                self.limiter.release(key, stamp)
