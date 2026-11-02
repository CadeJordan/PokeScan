"""Bounded grading (intent/archive/serving-hardening): /api/grade holds one of
`settings.grade_concurrency` tokens for its predict + centering, waits up to
`grade_queue_timeout_s` for one, and answers 503 + Retry-After past that.

Runs the app in-process through httpx's ASGI transport with a fake service
whose `predict` blocks until the test releases it, so no real model is used.
"""

from __future__ import annotations

import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

import anyio
import httpx
import pytest

import backend.app.api.grade as grade_mod
import backend.app.main as main_mod
from backend.app.core.config import Settings
from backend.app.ml.inference import GradePrediction
from backend.app.ml.preprocess import ImageDecodeError
from backend.tests.conftest import FakeService

TIMEOUT_S = 1.0  # queue timeout in these tests: a leaked token shows up as a quick 503
# Queue timeout for a request that must wait out another's whole grade (predict +
# real centering), which can take over a second on a slow CI runner.
WAIT_S = 10.0
GATE_S = 10.0  # a gated predict never blocks a broken test for longer than this


@dataclass
class GatedService(FakeService):
    """`predict` waits for the test to open its gate (one gate per call) when `hold`."""

    hold: bool = True
    entered: int = 0
    gates: list[threading.Event] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def predict(self, front: bytes, back: bytes) -> GradePrediction:
        with self.lock:
            gate = threading.Event()
            self.gates.append(gate)
            self.entered += 1
        if self.hold:
            gate.wait(GATE_S)
        return super().predict(front, back)

    def release(self, i: int) -> None:
        self.gates[i].set()

    def release_all(self) -> None:
        for gate in self.gates:
            gate.set()


@pytest.fixture
def service() -> GatedService:
    svc = GatedService()
    yield svc
    svc.release_all()


@pytest.fixture
def limit(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    def configure(concurrency: int = 1, timeout_s: float = TIMEOUT_S, reset: bool = True) -> None:
        settings = Settings(
            _env_file=None, grade_concurrency=concurrency, grade_queue_timeout_s=timeout_s
        )
        monkeypatch.setattr(grade_mod, "get_settings", lambda: settings)
        if reset:  # the queue timeout is read per request, so it can change mid-test
            grade_mod.reset_grade_limiter()

    configure()
    yield configure
    grade_mod.reset_grade_limiter()


@pytest.fixture
async def client(
    service: GatedService, limit, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[httpx.AsyncClient]:
    monkeypatch.setattr(grade_mod, "get_inference_service", lambda: service)
    transport = httpx.ASGITransport(app=main_mod.create_app(), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _files(image: bytes, back: bytes | None = None) -> dict:
    return {
        "front": ("front.jpg", image, "image/jpeg"),
        "back": ("back.jpg", image if back is None else back, "image/jpeg"),
    }


async def _post(client: httpx.AsyncClient, image: bytes) -> httpx.Response:
    return await client.post("/api/grade", files=_files(image))


async def _until(cond: Callable[[], bool], timeout_s: float = 5.0) -> None:
    with anyio.fail_after(timeout_s):
        while not cond():
            await anyio.sleep(0.01)


def _stats():
    return grade_mod.grade_limiter().statistics()


async def test_over_limit_request_waits_then_times_out(client, service, limit, scene_jpeg):
    limit(timeout_s=WAIT_S)
    results: dict[str, httpx.Response] = {}

    async def post(name: str) -> None:
        results[name] = await client.post("/api/grade", files=_files(scene_jpeg))

    async with anyio.create_task_group() as tg:
        tg.start_soon(post, "a")
        await _until(lambda: service.entered == 1)
        # Second request queues for the token without touching the model.
        tg.start_soon(post, "b")
        await _until(lambda: _stats().tasks_waiting == 1)
        assert service.entered == 1
        service.release(0)
        await _until(lambda: service.entered == 2)  # b got the token when a finished
        assert results["a"].status_code == 200

        # Third request: b holds the only token past the queue timeout.
        limit(timeout_s=TIMEOUT_S, reset=False)
        t0 = time.perf_counter()
        r = await client.post("/api/grade", files=_files(scene_jpeg))
        waited = time.perf_counter() - t0
        assert r.status_code == 503
        assert r.json()["detail"] == "server busy, try again"
        assert int(r.headers["Retry-After"]) > 0
        assert waited >= TIMEOUT_S * 0.9
        assert service.entered == 2  # the timed-out request did no model work
        service.release(1)
    assert results["b"].status_code == 200
    assert _stats().borrowed_tokens == 0


async def test_concurrency_setting_allows_that_many_grades(client, service, limit, scene_jpeg):
    limit(concurrency=2, timeout_s=WAIT_S)
    async with anyio.create_task_group() as tg:
        for _ in range(3):
            tg.start_soon(_post, client, scene_jpeg)
        await _until(lambda: service.entered == 2 and _stats().tasks_waiting == 1)
        service.release_all()
        await _until(lambda: service.entered == 3)
        service.release_all()
    assert _stats().borrowed_tokens == 0


async def test_health_and_upload_checks_answer_while_grading(client, service, scene_jpeg):
    async with anyio.create_task_group() as tg:
        tg.start_soon(_post, client, scene_jpeg)
        await _until(lambda: service.entered == 1)
        with anyio.fail_after(3.0):  # health would block until GATE_S if it waited
            health = await client.get("/api/health")
            # An oversized upload is refused before it would wait for a token.
            big = await client.post(
                "/api/grade", files=_files(scene_jpeg, b"\0" * (grade_mod.MAX_BYTES + 1))
            )
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True
        assert big.status_code == 413
        assert _stats().borrowed_tokens == 1
        service.release_all()


def _no_quad() -> GradePrediction:
    pred = FakeService().prediction
    pred.found_quad_front = False
    return pred


@pytest.mark.parametrize(
    ("error", "prediction", "expected"),
    [
        (ImageDecodeError("bad", face="front"), None, 400),
        (None, _no_quad(), 422),
        (RuntimeError("inference failed"), None, 500),
        (ValueError("unexpected"), None, 500),  # unhandled: the server's own 500
    ],
    ids=["decode_400", "rejected_422", "runtime_500", "unhandled_500"],
)
async def test_token_released_on_error(client, service, scene_jpeg, error, prediction, expected):
    service.hold = False
    service.error = error
    if prediction is not None:
        service.prediction = prediction
    r = await client.post("/api/grade", files=_files(scene_jpeg))
    assert r.status_code == expected
    assert _stats().borrowed_tokens == 0
    # The next request gets the token at once.
    service.error = None
    service.prediction = FakeService().prediction
    # With the short queue timeout, a leaked token would give a 503 here.
    r = await client.post("/api/grade", files=_files(scene_jpeg))
    assert r.status_code == 200


async def test_token_released_when_centering_fails(client, service, scene_jpeg, monkeypatch):
    import backend.app.ml.centering as centering

    def boom(*_a):
        raise RuntimeError("centering exploded")

    monkeypatch.setattr(centering, "compute_centering_subgrade", boom)
    service.hold = False
    r = await client.post("/api/grade", files=_files(scene_jpeg))
    assert r.status_code == 200
    assert r.json()["sub_grades"]["centering"] is None
    assert _stats().borrowed_tokens == 0


async def test_cancel_during_grade_keeps_token_until_predict_ends(client, service, scene_jpeg):
    """A cancelled request keeps its token while its predict thread still runs,
    then gives it back. (Starlette doesn't cancel a plain endpoint on a client
    disconnect; this covers the harder case of a cancelled task.)"""
    scope = anyio.CancelScope()
    done = anyio.Event()

    async def post() -> None:
        with scope:
            await client.post("/api/grade", files=_files(scene_jpeg))
        done.set()

    async with anyio.create_task_group() as tg:
        tg.start_soon(post)
        await _until(lambda: service.entered == 1)
        scope.cancel()
        await anyio.sleep(0.1)
        assert not done.is_set()
        assert _stats().borrowed_tokens == 1  # predict is still running
        service.release(0)
        with anyio.fail_after(5):
            await done.wait()
    assert scope.cancelled_caught
    assert _stats().borrowed_tokens == 0
    service.hold = False
    # With the short queue timeout, a leaked token would give a 503 here.
    r = await client.post("/api/grade", files=_files(scene_jpeg))
    assert r.status_code == 200


async def test_cancel_while_queued_leaves_no_waiter(client, service, scene_jpeg):
    scope = anyio.CancelScope()

    async def queued() -> None:
        with scope:
            await client.post("/api/grade", files=_files(scene_jpeg))

    async with anyio.create_task_group() as tg:
        tg.start_soon(_post, client, scene_jpeg)
        await _until(lambda: service.entered == 1)
        tg.start_soon(queued)
        await _until(lambda: _stats().tasks_waiting == 1)
        scope.cancel()
        await _until(lambda: _stats().tasks_waiting == 0)
        assert _stats().borrowed_tokens == 1
        service.release_all()
    assert service.entered == 1  # the cancelled request never ran the model
    assert _stats().borrowed_tokens == 0


def test_default_settings(monkeypatch):
    monkeypatch.delenv("GRADE_CONCURRENCY", raising=False)
    monkeypatch.delenv("GRADE_QUEUE_TIMEOUT_S", raising=False)
    settings = Settings(_env_file=None)
    assert settings.grade_concurrency == 4
    assert settings.grade_queue_timeout_s == 30.0


@pytest.mark.parametrize("bad", [{"grade_concurrency": 0}, {"grade_queue_timeout_s": 0}])
def test_settings_reject_non_positive_limits(bad):
    with pytest.raises(ValueError):
        Settings(_env_file=None, **bad)


def test_default_limiter_uses_settings(monkeypatch):
    settings = Settings(_env_file=None, grade_concurrency=3)
    monkeypatch.setattr(grade_mod, "get_settings", lambda: settings)
    grade_mod.reset_grade_limiter()
    try:
        assert grade_mod.grade_limiter().total_tokens == 3
    finally:
        grade_mod.reset_grade_limiter()


async def test_browser_can_read_retry_after(client):
    origins = main_mod.get_settings().cors_origins_list
    origin = origins[0] if origins else "http://example.test"
    r = await client.get("/api/health", headers={"Origin": origin})
    assert "retry-after" in r.headers["access-control-expose-headers"].lower()
