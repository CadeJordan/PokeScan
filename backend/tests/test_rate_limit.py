"""Per-IP rate limits on /api/grade (intent/rate-limits-and-photo-picker).

- `CODE_ATTEMPTS_PER_MINUTE` (default 5, only with `ACCESS_CODE`): missing or
  wrong codes per IP per minute; past it every /api/grade request from the IP is
  a 429, even with the right code, until the window clears.
- `GRADES_PER_HOUR` (default 0 = off): grades per IP per hour. Only requests
  that reach a grading slot count.
- `TRUST_FORWARDED_FOR` (default off): the client IP is the right-most
  `X-Forwarded-For` entry rather than the TCP peer. IPv6 clients count per /64.

Every 429 is sent before the upload is read or a grading slot is taken, carries
`Retry-After` and the CORS headers; /api/health is never limited.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator

import anyio
import pytest
from fastapi.testclient import TestClient

import backend.app.api.grade as grade_mod
import backend.app.main as main_mod
from backend.app.core.access import AccessCodeMiddleware
from backend.app.core.config import Settings
from backend.app.core.rate_limit import (
    GradeRateLimitMiddleware,
    SlidingWindowLimiter,
    client_ip,
)
from backend.app.ml.inference import GradePrediction
from backend.app.ml.preprocess import ImageDecodeError
from backend.tests.conftest import FakeService

CODE = "pikachu-teal-river"
ORIGIN = "https://pkmn-scan.vercel.app"
TOO_MANY = {"detail": "too many attempts, wait a minute"}
GRADE_LIMIT = {"detail": "grade limit reached (30 per hour), try again later"}


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


def _files(image: bytes, back: bytes | None = None) -> dict:
    return {
        "front": ("front.jpg", image, "image/jpeg"),
        "back": ("back.jpg", image if back is None else back, "image/jpeg"),
    }


@pytest.fixture(autouse=True)
def _no_centering(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the real centering measurement (~0.7 s a grade): these tests grade a lot."""
    import backend.app.ml.centering as centering

    monkeypatch.setattr(centering, "compute_centering_subgrade", lambda *_faces: None)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def make_client(
    fake_service: FakeService, monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> Iterator[Callable[..., TestClient]]:
    """A test client for an app built with the given settings (no .env) and the fake clock."""
    monkeypatch.setattr(grade_mod, "get_inference_service", lambda: fake_service)
    monkeypatch.setattr(main_mod, "get_inference_service", lambda: fake_service)
    clients: list[TestClient] = []

    def make(client: tuple[str, int] = ("203.0.113.7", 5000), **settings: object) -> TestClient:
        s = Settings(_env_file=None, **settings)
        monkeypatch.setattr(main_mod, "get_settings", lambda: s)
        grade_mod.reset_grade_limiter()
        c = TestClient(main_mod.create_app(clock=clock), client=client)
        c.__enter__()
        clients.append(c)
        return c

    yield make
    for c in clients:
        c.__exit__(None, None, None)
    grade_mod.reset_grade_limiter()


def _grade(c: TestClient, image: bytes, code: str | None = None, xff: str | None = None, **kw):
    headers = dict(kw.pop("headers", {}))
    if code is not None:
        headers["X-Access-Code"] = code
    if xff is not None:
        headers["X-Forwarded-For"] = xff
    return c.post("/api/grade", files=kw.pop("files", None) or _files(image), headers=headers, **kw)


def _retry_after(r) -> int:
    value = int(r.headers["retry-after"])
    assert value >= 1
    return value


# --------------------------------------------------------------------------- #
# Settings and defaults
# --------------------------------------------------------------------------- #
def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.code_attempts_per_minute == 5
    assert s.grades_per_hour == 0
    assert s.trust_forwarded_for is False


def test_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("CODE_ATTEMPTS_PER_MINUTE", "7")
    monkeypatch.setenv("GRADES_PER_HOUR", "30")
    monkeypatch.setenv("TRUST_FORWARDED_FOR", "true")
    s = Settings(_env_file=None)
    assert (s.code_attempts_per_minute, s.grades_per_hour, s.trust_forwarded_for) == (7, 30, True)


def test_defaults_add_no_limit(make_client, fake_service, scene_jpeg):
    """ACCESS_CODE unset, GRADES_PER_HOUR 0: no limiter at all, as before."""
    c = make_client()
    classes = {m.cls for m in c.app.user_middleware}
    assert GradeRateLimitMiddleware not in classes
    assert AccessCodeMiddleware not in classes
    for _ in range(40):
        assert _grade(c, scene_jpeg).status_code == 200
    assert len(fake_service.calls) == 40


def test_code_attempts_zero_turns_the_wrong_code_limit_off(make_client, scene_jpeg):
    c = make_client(access_code=CODE, code_attempts_per_minute=0)
    for _ in range(20):
        assert _grade(c, scene_jpeg, "wrong").status_code == 401
    assert _grade(c, scene_jpeg, CODE).status_code == 200


# --------------------------------------------------------------------------- #
# Wrong codes
# --------------------------------------------------------------------------- #
def test_sixth_wrong_code_attempt_is_429_even_with_the_right_code(make_client, fake_service, scene_jpeg, clock):
    c = make_client(access_code=CODE)
    # A missing code counts too (a first visit sends one request without it).
    assert _grade(c, scene_jpeg).status_code == 401
    for i in range(4):
        clock.advance(1)
        r = _grade(c, scene_jpeg, f"wrong{i}")
        assert r.status_code == 401
    clock.advance(1)
    for code in ("wrong", CODE, None):
        r = _grade(c, scene_jpeg, code)
        assert r.status_code == 429, code
        assert r.json() == TOO_MANY
        # The oldest failure (t=0) leaves the window at t=60; now is t=5.
        assert _retry_after(r) == 55
    assert fake_service.calls == []


def test_wrong_code_lockout_clears_after_the_window(make_client, fake_service, scene_jpeg, clock):
    c = make_client(access_code=CODE)
    for _ in range(5):
        assert _grade(c, scene_jpeg, "wrong").status_code == 401
    assert _grade(c, scene_jpeg, CODE).status_code == 429
    clock.advance(59.5)
    r = _grade(c, scene_jpeg, CODE)
    assert r.status_code == 429
    assert _retry_after(r) == 1  # rounded up, never 0
    clock.advance(0.5)
    assert _grade(c, scene_jpeg, CODE).status_code == 200
    assert len(fake_service.calls) == 1


def test_lockout_does_not_extend_itself(make_client, scene_jpeg, clock):
    """429s don't count as failures: the IP is free once its 5 failures expire."""
    c = make_client(access_code=CODE)
    for _ in range(5):
        _grade(c, scene_jpeg, "wrong")
    for _ in range(10):
        clock.advance(5)
        assert _grade(c, scene_jpeg, "wrong").status_code == 429
    clock.advance(10)
    assert _grade(c, scene_jpeg, CODE).status_code == 200


def test_right_codes_do_not_count(make_client, scene_jpeg):
    c = make_client(access_code=CODE)
    for _ in range(4):
        _grade(c, scene_jpeg, "wrong")
    for _ in range(10):
        assert _grade(c, scene_jpeg, CODE).status_code == 200
    assert _grade(c, scene_jpeg, "wrong").status_code == 401
    assert _grade(c, scene_jpeg, CODE).status_code == 429


def test_wrong_code_limit_is_per_ip(make_client, scene_jpeg):
    a = make_client(access_code=CODE, client=("198.51.100.1", 1))
    for _ in range(5):
        _grade(a, scene_jpeg, "wrong")
    assert _grade(a, scene_jpeg, CODE).status_code == 429
    # A second client of the same app (shared limiter), another IP.
    b = TestClient(a.app, client=("198.51.100.2", 1))
    with b:
        assert _grade(b, scene_jpeg, CODE).status_code == 200
        assert _grade(b, scene_jpeg, "wrong").status_code == 401


def test_health_is_never_limited(make_client, scene_jpeg):
    c = make_client(access_code=CODE, grades_per_hour=1)
    for _ in range(5):
        _grade(c, scene_jpeg, "wrong")
    assert _grade(c, scene_jpeg, CODE).status_code == 429
    for _ in range(20):
        r = c.get("/api/health")
        assert r.status_code == 200
        assert "retry-after" not in r.headers


def test_wrong_code_429_carries_cors_headers(make_client, scene_jpeg):
    c = make_client(access_code=CODE, cors_origins=ORIGIN)
    for _ in range(5):
        _grade(c, scene_jpeg, "wrong", headers={"Origin": ORIGIN})
    r = _grade(c, scene_jpeg, CODE, headers={"Origin": ORIGIN})
    assert r.status_code == 429
    assert r.headers["access-control-allow-origin"] == ORIGIN
    assert "retry-after" in r.headers["access-control-expose-headers"].lower()
    # Preflights are answered by CORS and never limited.
    pre = c.options(
        "/api/grade",
        headers={"Origin": ORIGIN, "Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": "x-access-code"},
    )
    assert pre.status_code == 200


# --------------------------------------------------------------------------- #
# Client IP and X-Forwarded-For
# --------------------------------------------------------------------------- #
def _scope(xff: list[bytes] | None = None, client: tuple[str, int] | None = ("10.0.0.1", 1)) -> dict:
    headers = [(b"host", b"test")] + [(b"x-forwarded-for", v) for v in xff or []]
    return {"type": "http", "headers": headers, "client": client}


@pytest.mark.parametrize(
    ("xff", "trust", "expected"),
    [
        (None, False, "10.0.0.1"),
        ([b"1.1.1.1"], False, "10.0.0.1"),  # never trusted when off
        (None, True, "10.0.0.1"),  # trusted but absent: the peer
        ([b"1.1.1.1"], True, "1.1.1.1"),
        ([b"6.6.6.6, 1.1.1.1"], True, "1.1.1.1"),  # spoofed left-most entry ignored
        ([b"6.6.6.6,1.1.1.1 "], True, "1.1.1.1"),
        ([b"6.6.6.6", b"1.1.1.1"], True, "1.1.1.1"),  # several header lines, in order
        ([b"1.1.1.1, "], True, "1.1.1.1"),  # empty entries skipped
        ([b" , "], True, "10.0.0.1"),
        ([b"2001:db8::1"], True, "2001:db8::/64"),  # IPv6: its /64
        ([b"2001:db8:0:0:ffff:1:2:3"], True, "2001:db8::/64"),
        ([b"2001:db8:0:1::1"], True, "2001:db8:0:1::/64"),
        ([b"not-an-ip"], True, "not-an-ip"),  # junk kept as is
    ],
)
def test_client_ip(xff, trust, expected):
    assert client_ip(_scope(xff), trust) == expected


def test_client_ip_ipv6_peer_is_keyed_by_its_64():
    assert client_ip(_scope(None, client=("2001:db8::abcd", 1)), False) == "2001:db8::/64"


def test_ipv6_addresses_in_one_64_share_the_limits(make_client, scene_jpeg):
    """Rotating addresses inside one /64 doesn't escape either limit."""
    c = make_client(access_code=CODE, grades_per_hour=1, trust_forwarded_for=True)
    for i in range(5):
        assert _grade(c, scene_jpeg, "wrong", xff=f"2001:db8::{i + 1:x}").status_code == 401
    assert _grade(c, scene_jpeg, CODE, xff="2001:db8::ff").status_code == 429
    # Another /64 is someone else; it gets one grade, then its other addresses are refused.
    assert _grade(c, scene_jpeg, CODE, xff="2001:db8:0:1::1").status_code == 200
    r = _grade(c, scene_jpeg, CODE, xff="2001:db8:0:1::2")
    assert r.status_code == 429
    assert r.json() == {"detail": "grade limit reached (1 per hour), try again later"}


def test_client_ip_unknown_without_a_peer():
    assert client_ip(_scope(None, client=None), False) == "unknown"
    assert client_ip(_scope(None, client=None), True) == "unknown"


def test_spoofed_left_most_entry_does_not_escape_the_limit(make_client, scene_jpeg):
    """What Cloud Run sends when a client forges the header: `<forged>, <real>`."""
    c = make_client(access_code=CODE, trust_forwarded_for=True)
    for i in range(5):
        assert _grade(c, scene_jpeg, "wrong", xff=f"192.0.2.{i}, 203.0.113.50").status_code == 401
    r = _grade(c, scene_jpeg, CODE, xff="192.0.2.99, 203.0.113.50")
    assert r.status_code == 429
    assert r.json() == TOO_MANY
    # Another real client IP (right-most entry) isn't locked out.
    assert _grade(c, scene_jpeg, CODE, xff="192.0.2.1, 203.0.113.51").status_code == 200


def test_forwarded_for_ignored_when_not_trusted(make_client, scene_jpeg):
    """With the setting off, a varying header (even its right-most entry) doesn't split the count."""
    c = make_client(access_code=CODE)
    for i in range(5):
        assert _grade(c, scene_jpeg, "wrong", xff=f"192.0.2.{i}").status_code == 401
    assert _grade(c, scene_jpeg, CODE, xff="192.0.2.200").status_code == 429


def test_trusted_header_separates_ips(make_client, scene_jpeg):
    c = make_client(access_code=CODE, trust_forwarded_for=True)
    for _ in range(5):
        _grade(c, scene_jpeg, "wrong", xff="203.0.113.50")
    assert _grade(c, scene_jpeg, CODE, xff="203.0.113.50").status_code == 429
    # Same TCP peer, another forwarded IP.
    assert _grade(c, scene_jpeg, CODE, xff="203.0.113.60").status_code == 200


# --------------------------------------------------------------------------- #
# Grades per hour
# --------------------------------------------------------------------------- #
def test_31st_grade_in_an_hour_is_429_and_clears_after_the_window(make_client, fake_service, scene_jpeg, clock):
    c = make_client(grades_per_hour=30)
    for _ in range(30):
        assert _grade(c, scene_jpeg).status_code == 200
        clock.advance(10)
    # The first grade (t=0) leaves the window at t=3600; now is t=300.
    r = _grade(c, scene_jpeg)
    assert r.status_code == 429
    assert r.json() == GRADE_LIMIT
    assert _retry_after(r) == 3300
    assert len(fake_service.calls) == 30
    clock.advance(3300)
    assert _grade(c, scene_jpeg).status_code == 200
    # Only the oldest left the window: the next one is refused again.
    r = _grade(c, scene_jpeg)
    assert r.status_code == 429
    assert _retry_after(r) == 10
    assert len(fake_service.calls) == 31


def test_grade_limit_is_per_ip_and_honours_trusted_forwarded_for(make_client, scene_jpeg):
    c = make_client(grades_per_hour=2, trust_forwarded_for=True)
    for _ in range(2):
        assert _grade(c, scene_jpeg, xff="192.0.2.1, 203.0.113.50").status_code == 200
    assert _grade(c, scene_jpeg, xff="192.0.2.2, 203.0.113.50").status_code == 429
    assert _grade(c, scene_jpeg, xff="203.0.113.51").status_code == 200


def test_grade_limit_detail_names_the_configured_limit(make_client, scene_jpeg):
    c = make_client(grades_per_hour=1)
    assert _grade(c, scene_jpeg).status_code == 200
    r = _grade(c, scene_jpeg)
    assert r.json() == {"detail": "grade limit reached (1 per hour), try again later"}


def _no_quad() -> GradePrediction:
    pred = FakeService().prediction
    pred.found_quad_front = False
    return pred


def test_refused_requests_do_not_use_grade_allowance(make_client, fake_service, scene_jpeg, monkeypatch, clock):
    """413, 415, empty 400, 422 validation, 401 and 429 cost nothing."""
    c = make_client(access_code=CODE, grades_per_hour=2)
    monkeypatch.setattr(grade_mod, "MAX_BYTES", 1000)
    assert _grade(c, scene_jpeg, CODE).status_code == 413
    monkeypatch.setattr(grade_mod, "MAX_BYTES", 12 * 1024 * 1024)
    bad_type = {"front": ("f.gif", scene_jpeg, "image/gif"), "back": ("b.jpg", scene_jpeg, "image/jpeg")}
    assert _grade(c, scene_jpeg, CODE, files=bad_type).status_code == 415
    assert _grade(c, scene_jpeg, CODE, files=_files(scene_jpeg, b"")).status_code == 400
    missing = {"front": ("f.jpg", scene_jpeg, "image/jpeg")}
    assert _grade(c, scene_jpeg, CODE, files=missing).status_code == 422
    for _ in range(2):  # 2 < the 5 wrong-code attempts
        assert _grade(c, scene_jpeg, "wrong").status_code == 401
    # The full allowance is still there.
    assert _grade(c, scene_jpeg, CODE).status_code == 200
    assert _grade(c, scene_jpeg, CODE).status_code == 200
    assert _grade(c, scene_jpeg, CODE).status_code == 429
    # 429s don't count either (else the wait would grow): one window later, free.
    for _ in range(5):
        assert _grade(c, scene_jpeg, CODE).status_code == 429
    clock.advance(3600)
    assert _grade(c, scene_jpeg, CODE).status_code == 200


def test_model_not_loaded_503_does_not_use_grade_allowance(make_client, fake_service, scene_jpeg):
    c = make_client(grades_per_hour=1)
    fake_service.ready = False
    for _ in range(3):
        assert _grade(c, scene_jpeg).status_code == 503
    fake_service.ready = True
    assert _grade(c, scene_jpeg).status_code == 200
    assert _grade(c, scene_jpeg).status_code == 429


def test_server_busy_503_does_not_use_grade_allowance(raw_app, fake_service, scene_jpeg):
    """No slot within the queue timeout: a 503 that did no work, so it's free."""
    app = raw_app(grades_per_hour=1, grade_concurrency=1, grade_queue_timeout_s=0.1)

    async def main() -> None:
        limiter = grade_mod.grade_limiter()
        holder = object()
        await limiter.acquire_on_behalf_of(holder)
        try:
            for _ in range(3):
                assert (await _drive(app, scene_jpeg, None))[0] == 503
        finally:
            limiter.release_on_behalf_of(holder)
        assert (await _drive(app, scene_jpeg, None))[0] == 200
        assert (await _drive(app, scene_jpeg, None))[0] == 429

    anyio.run(main)
    assert len(fake_service.calls) == 1


def test_health_stays_open_when_the_grade_limit_is_used_up(make_client, scene_jpeg):
    c = make_client(grades_per_hour=1)
    assert _grade(c, scene_jpeg).status_code == 200
    assert _grade(c, scene_jpeg).status_code == 429
    for _ in range(5):
        r = c.get("/api/health")
        assert r.status_code == 200
        assert "retry-after" not in r.headers


@pytest.mark.parametrize(
    ("error", "prediction", "expected"),
    [
        (None, _no_quad(), 422),
        (RuntimeError("inference failed"), None, 500),
        (ImageDecodeError("bad", face="back"), None, 400),
    ],
    ids=["rejected_422", "runtime_500", "unreadable_400"],
)
def test_graded_failures_use_grade_allowance(make_client, fake_service, scene_jpeg, error, prediction, expected):
    """A request that held a grading slot counts, whatever the outcome. An
    unreadable image too: it held the slot while the other face was decoded."""
    c = make_client(grades_per_hour=1)
    fake_service.error = error
    if prediction is not None:
        fake_service.prediction = prediction
    assert _grade(c, scene_jpeg).status_code == expected
    fake_service.error = None
    assert _grade(c, scene_jpeg).status_code == 429


def test_grade_429_carries_cors_headers(make_client, scene_jpeg):
    c = make_client(grades_per_hour=1, cors_origins=ORIGIN)
    assert _grade(c, scene_jpeg, headers={"Origin": ORIGIN}).status_code == 200
    r = _grade(c, scene_jpeg, headers={"Origin": ORIGIN})
    assert r.status_code == 429
    assert r.headers["access-control-allow-origin"] == ORIGIN
    assert int(r.headers["retry-after"]) >= 1


def test_concurrent_requests_cannot_exceed_the_grade_limit(make_client, fake_service, scene_jpeg):
    """The reservation is taken atomically before the request runs."""
    c = make_client(grades_per_hour=3, grade_concurrency=8)
    statuses: list[int] = []
    lock = threading.Lock()

    def post() -> None:
        r = _grade(c, scene_jpeg)
        with lock:
            statuses.append(r.status_code)

    threads = [threading.Thread(target=post) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(statuses) == [200] * 3 + [429] * 7
    assert len(fake_service.calls) == 3


# --------------------------------------------------------------------------- #
# A 429 reads no body and takes no grading slot (driven as raw ASGI)
# --------------------------------------------------------------------------- #
def _asgi_request(image: bytes, code: bytes | None, client=("203.0.113.7", 1)):
    boundary = "pokescanboundary"
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
        f'filename="{name}.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()
        + image
        + b"\r\n"
        for name in ("front", "back")
    ]
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    headers = [
        (b"host", b"test"),
        (b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
        (b"content-length", str(len(body)).encode()),
    ]
    if code is not None:
        headers.append((b"x-access-code", code))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/grade",
        "raw_path": b"/api/grade",
        "root_path": "",
        "query_string": b"",
        "headers": headers,
        "client": client,
        "server": ("test", 80),
    }
    return scope, body


async def _drive(app, image: bytes, code: bytes | None) -> tuple[int, int]:
    """(status, number of `receive` calls) for one request."""
    scope, body = _asgi_request(image, code)
    reads = 0
    sent: list[dict] = []

    async def receive() -> dict:
        nonlocal reads
        reads += 1
        if reads == 1:
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        sent.append(message)

    await app(scope, receive, send)
    assert sent[0]["type"] == "http.response.start"
    return sent[0]["status"], reads


@pytest.fixture
def raw_app(fake_service: FakeService, monkeypatch: pytest.MonkeyPatch, clock: FakeClock):
    def make(**settings: object):
        monkeypatch.setattr(grade_mod, "get_inference_service", lambda: fake_service)
        s = Settings(_env_file=None, **settings)
        monkeypatch.setattr(main_mod, "get_settings", lambda: s)
        monkeypatch.setattr(grade_mod, "get_settings", lambda: s)
        grade_mod.reset_grade_limiter()
        return main_mod.create_app(clock=clock)

    yield make
    grade_mod.reset_grade_limiter()


def test_wrong_code_429_reads_no_body(raw_app, fake_service, scene_jpeg):
    app = raw_app(access_code=CODE)

    async def main() -> None:
        for _ in range(5):
            assert await _drive(app, scene_jpeg, b"wrong") == (401, 0)
        assert await _drive(app, scene_jpeg, CODE.encode()) == (429, 0)

    anyio.run(main)
    assert fake_service.calls == []


def test_grade_429_reads_no_body_and_takes_no_slot(raw_app, fake_service, scene_jpeg):
    """With the only slot held by someone else, an over-limit request is a 429 at
    once: not a 503 after waiting for the slot, and its body is never read."""
    app = raw_app(access_code=CODE, grades_per_hour=1, grade_concurrency=1, grade_queue_timeout_s=0.5)

    async def main() -> None:
        status, reads = await _drive(app, scene_jpeg, CODE.encode())
        assert status == 200
        assert reads >= 1
        limiter = grade_mod.grade_limiter()
        holder = object()
        await limiter.acquire_on_behalf_of(holder)
        try:
            with anyio.fail_after(0.4):  # shorter than the queue timeout
                assert await _drive(app, scene_jpeg, CODE.encode()) == (429, 0)
            stats = limiter.statistics()
            assert stats.borrowed_tokens == 1
            assert stats.tasks_waiting == 0
        finally:
            limiter.release_on_behalf_of(holder)

    anyio.run(main)
    assert len(fake_service.calls) == 1


# --------------------------------------------------------------------------- #
# The limiter itself
# --------------------------------------------------------------------------- #
def test_limiter_rejects_bad_parameters():
    with pytest.raises(ValueError):
        SlidingWindowLimiter(0, 60)
    with pytest.raises(ValueError):
        SlidingWindowLimiter(1, 0)


def test_limiter_try_acquire_release_and_retry_after(clock):
    lim = SlidingWindowLimiter(2, 60, clock=clock)
    s1, w = lim.try_acquire("a")
    assert s1 is not None and w is None
    clock.advance(10)
    s2, _ = lim.try_acquire("a")
    assert s2 is not None
    assert lim.try_acquire("a") == (None, 50)
    assert lim.retry_after("a") == 50
    lim.release("a", s2)
    assert lim.retry_after("a") is None
    s3, _ = lim.try_acquire("a")
    assert s3 is not None
    clock.advance(50)  # s1 expires
    assert lim.retry_after("a") is None
    lim.release("a", s1)  # already expired: a no-op
    assert lim.retry_after("b") is None


def test_limiter_memory_is_bounded(clock):
    """Keys are dropped once their events expire: memory follows the active IPs."""
    lim = SlidingWindowLimiter(5, 60, clock=clock)
    for i in range(1000):
        lim.record(f"10.0.{i // 256}.{i % 256}")
    assert len(lim) == 1000
    clock.advance(61)
    lim.record("fresh")  # a sweep is due: every expired key goes
    assert len(lim) == 1
    # A key never holds more than its limit when used as the middleware does.
    for _ in range(100):
        if lim.retry_after("busy") is None:
            lim.record("busy")
        lim.try_acquire("busy2")
    assert len(lim._events["busy"]) == 5
    assert len(lim._events["busy2"]) == 5


def test_limiter_is_thread_safe(clock):
    lim = SlidingWindowLimiter(500, 60, clock=clock)
    granted: list[float] = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(200):
            stamp, _ = lim.try_acquire("k")
            if stamp is not None:
                with lock:
                    granted.append(stamp)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(granted) == 500
    assert len(lim._events["k"]) == 500
