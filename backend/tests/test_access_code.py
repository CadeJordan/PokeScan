"""Deploy settings (intent/cloud-run-deploy): `CORS_ORIGIN_REGEX` and `ACCESS_CODE`.

Both are off by default. With `ACCESS_CODE` set, /api/grade needs the
`X-Access-Code` header and refuses a missing or wrong one with a 401 before the
upload is read or a grading slot is taken; /api/health stays open.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import anyio
import pytest
from fastapi.testclient import TestClient

import backend.app.api.grade as grade_mod
import backend.app.main as main_mod
from backend.app.core.access import AccessCodeMiddleware
from backend.app.core.config import Settings
from backend.tests.conftest import FakeService

CODE = "pikachu-teal-river"
LOCAL = "http://localhost:3000"
PROD = "https://pokescan.vercel.app"
PREVIEW = "https://pokescan-git-feature-x-cadejordan.vercel.app"
REGEX = r"https://pokescan[-a-z0-9]*\.vercel\.app"


def _files(image: bytes) -> dict:
    return {
        "front": ("front.jpg", image, "image/jpeg"),
        "back": ("back.jpg", image, "image/jpeg"),
    }


@pytest.fixture
def make_client(
    fake_service: FakeService, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[..., TestClient]]:
    """A test client for an app built with the given settings (no .env)."""
    monkeypatch.setattr(grade_mod, "get_inference_service", lambda: fake_service)
    monkeypatch.setattr(main_mod, "get_inference_service", lambda: fake_service)
    clients: list[TestClient] = []

    def make(**settings: object) -> TestClient:
        s = Settings(_env_file=None, **settings)
        monkeypatch.setattr(main_mod, "get_settings", lambda: s)
        grade_mod.reset_grade_limiter()
        c = TestClient(main_mod.create_app())
        c.__enter__()
        clients.append(c)
        return c

    yield make
    for c in clients:
        c.__exit__(None, None, None)
    grade_mod.reset_grade_limiter()


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
def test_settings_default_off():
    s = Settings(_env_file=None)
    assert s.access_code == ""
    assert s.cors_origin_regex == ""


def test_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", CODE)
    monkeypatch.setenv("CORS_ORIGIN_REGEX", REGEX)
    s = Settings(_env_file=None)
    assert s.access_code == CODE
    assert s.cors_origin_regex == REGEX


def test_no_code_set_grade_is_open(make_client, fake_service, scene_jpeg):
    c = make_client()
    r = c.post("/api/grade", files=_files(scene_jpeg))
    assert r.status_code == 200, r.text
    assert len(fake_service.calls) == 1
    # A stray header is ignored when no code is configured.
    r = c.post("/api/grade", files=_files(scene_jpeg), headers={"X-Access-Code": "anything"})
    assert r.status_code == 200


def test_no_code_set_has_no_access_middleware(make_client):
    c = make_client()
    assert all(m.cls is not AccessCodeMiddleware for m in c.app.user_middleware)


def test_configured_code_is_stripped(make_client, scene_jpeg):
    c = make_client(access_code=f"  {CODE}\n")
    r = c.post("/api/grade", files=_files(scene_jpeg), headers={"X-Access-Code": CODE})
    assert r.status_code == 200, r.text


def test_whitespace_only_code_means_no_code(make_client, scene_jpeg):
    c = make_client(access_code=" \n")
    assert all(m.cls is not AccessCodeMiddleware for m in c.app.user_middleware)
    assert c.post("/api/grade", files=_files(scene_jpeg)).status_code == 200


def test_middleware_refuses_empty_code():
    with pytest.raises(ValueError):
        AccessCodeMiddleware(app=None, code="")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# ACCESS_CODE
# --------------------------------------------------------------------------- #
def test_code_missing_is_401(make_client, fake_service, scene_jpeg):
    c = make_client(access_code=CODE)
    r = c.post("/api/grade", files=_files(scene_jpeg))
    assert r.status_code == 401
    assert r.json() == {"detail": "access code required"}
    assert fake_service.calls == []


@pytest.mark.parametrize(
    "given", ["wrong", "", CODE.upper(), CODE + " ", " " + CODE, CODE[:-1], CODE + "x"]
)
def test_code_wrong_is_401(make_client, fake_service, scene_jpeg, given):
    c = make_client(access_code=CODE)
    r = c.post("/api/grade", files=_files(scene_jpeg), headers={"X-Access-Code": given})
    assert r.status_code == 401
    assert r.json() == {"detail": "access code required"}
    assert fake_service.calls == []


def test_code_right_is_200(make_client, fake_service, scene_jpeg):
    c = make_client(access_code=CODE)
    r = c.post("/api/grade", files=_files(scene_jpeg), headers={"X-Access-Code": CODE})
    assert r.status_code == 200, r.text
    assert r.json()["grade_int"] == 9
    assert fake_service.calls == [(scene_jpeg, scene_jpeg)]


def test_header_name_is_case_insensitive(make_client, scene_jpeg):
    c = make_client(access_code=CODE)
    r = c.post("/api/grade", files=_files(scene_jpeg), headers={"x-access-code": CODE})
    assert r.status_code == 200


def test_health_stays_open_and_does_not_leak_the_code(make_client, fake_service):
    c = make_client(access_code=CODE)
    r = c.get("/api/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True
    assert CODE not in r.text


def test_401_takes_no_grading_slot(make_client, scene_jpeg):
    c = make_client(access_code=CODE, grade_concurrency=1)
    for _ in range(3):
        assert c.post("/api/grade", files=_files(scene_jpeg)).status_code == 401
    stats = grade_mod.grade_limiter().statistics()
    assert stats.borrowed_tokens == 0
    assert stats.tasks_waiting == 0


@pytest.mark.parametrize(("header", "status"), [(None, 401), (b"wrong", 401), (CODE.encode(), 200)])
def test_401_is_sent_before_the_upload_is_read(fake_service, monkeypatch, scene_jpeg, header, status):
    """Drive the ASGI app directly: a refused request never calls `receive`."""
    monkeypatch.setattr(grade_mod, "get_inference_service", lambda: fake_service)
    s = Settings(_env_file=None, access_code=CODE)
    monkeypatch.setattr(main_mod, "get_settings", lambda: s)
    grade_mod.reset_grade_limiter()
    app = main_mod.create_app()

    boundary = "pokescanboundary"
    parts = []
    for name in ("front", "back"):
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
            f'filename="{name}.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()
            + scene_jpeg
            + b"\r\n"
        )
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    headers = [
        (b"host", b"test"),
        (b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
        (b"content-length", str(len(body)).encode()),
    ]
    if header is not None:
        headers.append((b"x-access-code", header))
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
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }
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

    try:
        anyio.run(app, scope, receive, send)
    finally:
        grade_mod.reset_grade_limiter()
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == status
    if status == 401:
        assert reads == 0
        assert fake_service.calls == []
    else:
        assert reads >= 1
        assert len(fake_service.calls) == 1


# --------------------------------------------------------------------------- #
# CORS_ORIGIN_REGEX
# --------------------------------------------------------------------------- #
def _preflight(c: TestClient, origin: str, headers: str | None = None):
    h = {"Origin": origin, "Access-Control-Request-Method": "POST"}
    if headers is not None:
        h["Access-Control-Request-Headers"] = headers
    return c.options("/api/grade", headers=h)


def test_regex_allows_a_preview_origin(make_client):
    c = make_client(cors_origins=f"{LOCAL},{PROD}", cors_origin_regex=REGEX)
    for origin in (PREVIEW, PROD, LOCAL):
        r = _preflight(c, origin)
        assert r.status_code == 200, origin
        assert r.headers["access-control-allow-origin"] == origin


@pytest.mark.parametrize(
    "origin",
    [
        "https://evil.vercel.app",
        "https://pokescan.vercel.app.evil.com",  # the regex must match the whole origin
        "http://pokescan.vercel.app",
        "https://xpokescan.vercel.app",
        "https://pokescan-preview.vercel.app:8443",
    ],
)
def test_regex_rejects_other_origins(make_client, origin):
    c = make_client(cors_origins=LOCAL, cors_origin_regex=REGEX)
    r = _preflight(c, origin)
    assert r.status_code == 400
    assert "access-control-allow-origin" not in r.headers


def test_no_regex_means_exact_origins_only(make_client):
    c = make_client(cors_origins=f"{LOCAL},{PROD}")
    assert _preflight(c, PROD).status_code == 200
    assert _preflight(c, PREVIEW).status_code == 400


def test_preflight_allows_the_access_code_header(make_client):
    c = make_client(cors_origins=PROD, cors_origin_regex=REGEX, access_code=CODE)
    for origin in (PROD, PREVIEW):
        r = _preflight(c, origin, headers="x-access-code")
        # The preflight is answered by CORS, never refused for the missing code.
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == origin
        assert "x-access-code" in r.headers["access-control-allow-headers"].lower()


def test_browser_can_read_the_401(make_client, scene_jpeg):
    """The 401 carries the CORS headers, so the frontend sees the status."""
    c = make_client(cors_origins=PROD, cors_origin_regex=REGEX, access_code=CODE)
    for origin in (PROD, PREVIEW):
        r = c.post("/api/grade", files=_files(scene_jpeg), headers={"Origin": origin})
        assert r.status_code == 401
        assert r.headers["access-control-allow-origin"] == origin
