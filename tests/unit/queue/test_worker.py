from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from webapp.queue import worker as worker_module


@pytest.fixture
def worker_health_env(monkeypatch):
    monkeypatch.setenv("DOCUSCOPE_RQ_ENABLED", "1")
    monkeypatch.setenv("DOCUSCOPE_RQ_QUEUE", "docuscope")
    monkeypatch.setenv("DOCUSCOPE_RQ_WORKER_NAME", "worker-1")
    connection = SimpleNamespace(ping=lambda: True)
    monkeypatch.setattr(worker_module, "get_redis_connection", lambda: connection)
    return connection


def _worker(
    *,
    state: str = "idle",
    queue_name: str = "docuscope",
    heartbeat_age_seconds: int = 0,
):
    return SimpleNamespace(
        name="worker-1",
        state=state,
        queues=[SimpleNamespace(name=queue_name)],
        last_heartbeat=(
            datetime.now(timezone.utc)
            - timedelta(seconds=heartbeat_age_seconds)
        ),
    )


def test_worker_health_accepts_fresh_expected_registration(
    worker_health_env,
    monkeypatch,
):
    monkeypatch.setattr(
        worker_module.Worker,
        "all",
        lambda **_kwargs: [_worker()],
    )

    worker_module.check_worker_health(max_heartbeat_age_seconds=90)


def test_worker_health_default_allows_rq_idle_heartbeat_cadence(
    worker_health_env,
    monkeypatch,
):
    monkeypatch.delenv("DOCUSCOPE_RQ_HEALTH_MAX_AGE_SECONDS", raising=False)
    monkeypatch.setattr(
        worker_module.Worker,
        "all",
        lambda **_kwargs: [
            _worker(
                heartbeat_age_seconds=worker_module.DEFAULT_WORKER_TTL - 1
            )
        ],
    )

    worker_module.check_worker_health()


def test_worker_health_rejects_redis_failure(worker_health_env, monkeypatch):
    monkeypatch.setattr(worker_health_env, "ping", lambda: False)

    with pytest.raises(RuntimeError, match="Redis did not respond"):
        worker_module.check_worker_health()


@pytest.mark.parametrize(
    ("workers", "message"),
    [
        ([], "is not registered"),
        ([_worker(state="suspended")], "unhealthy state"),
        ([_worker(queue_name="other")], "is not subscribed"),
        ([_worker(heartbeat_age_seconds=91)], "heartbeat is"),
    ],
)
def test_worker_health_rejects_unhealthy_registration(
    worker_health_env,
    monkeypatch,
    workers,
    message,
):
    monkeypatch.setattr(
        worker_module.Worker,
        "all",
        lambda **_kwargs: workers,
    )

    with pytest.raises(RuntimeError, match=message):
        worker_module.check_worker_health(max_heartbeat_age_seconds=90)
