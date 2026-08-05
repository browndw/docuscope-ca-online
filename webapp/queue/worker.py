"""RQ worker entrypoint for Redis-backed background jobs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import socket

from rq import SimpleWorker, Worker
from rq.defaults import DEFAULT_WORKER_TTL

from webapp.persistence import initialize_database_schema
from webapp.queue.client import get_queue, get_redis_connection
from webapp.queue.config import get_redis_queue_config


def get_worker_name() -> str:
    """Return the stable name shared by the worker and its health probe."""

    return os.environ.get("DOCUSCOPE_RQ_WORKER_NAME", "").strip() or socket.gethostname()


def check_worker_health(max_heartbeat_age_seconds: int | None = None) -> None:
    """Raise when Redis or this worker's RQ registration is unhealthy."""

    config = get_redis_queue_config()
    if not config.enabled:
        raise RuntimeError("Redis/RQ queueing is disabled.")

    connection = get_redis_connection()
    if not connection.ping():
        raise RuntimeError("Redis did not respond to PING.")

    worker_name = get_worker_name()
    worker = next(
        (
            candidate
            for candidate in Worker.all(connection=connection)
            if candidate.name == worker_name
        ),
        None,
    )
    if worker is None:
        raise RuntimeError(f"RQ worker {worker_name!r} is not registered.")

    state = getattr(worker.state, "value", worker.state)
    if state not in {"started", "idle", "busy"}:
        raise RuntimeError(f"RQ worker {worker_name!r} has unhealthy state {state!r}.")

    queue_names = {queue.name for queue in worker.queues}
    if config.queue_name not in queue_names:
        raise RuntimeError(
            f"RQ worker {worker_name!r} is not subscribed to {config.queue_name!r}."
        )

    max_age = max_heartbeat_age_seconds or int(
        os.environ.get(
            "DOCUSCOPE_RQ_HEALTH_MAX_AGE_SECONDS",
            str(DEFAULT_WORKER_TTL),
        )
    )
    last_heartbeat = worker.last_heartbeat
    if last_heartbeat is None:
        raise RuntimeError(f"RQ worker {worker_name!r} has no heartbeat.")
    if last_heartbeat.tzinfo is None:
        last_heartbeat = last_heartbeat.replace(tzinfo=timezone.utc)
    heartbeat_age = (datetime.now(timezone.utc) - last_heartbeat).total_seconds()
    if heartbeat_age > max_age:
        raise RuntimeError(
            f"RQ worker {worker_name!r} heartbeat is {heartbeat_age:.1f}s old."
        )


def run_worker() -> None:
    """Run an RQ worker for the configured queue."""

    config = get_redis_queue_config()
    if not config.enabled:
        raise SystemExit(
            "Redis/RQ queueing is disabled. Set DOCUSCOPE_RQ_ENABLED=1 to run the worker."
        )

    initialize_database_schema()
    connection = get_redis_connection()
    simple_worker_setting = os.environ.get(
        "DOCUSCOPE_RQ_SIMPLE_WORKER",
        "1",
    ).strip().lower()
    use_simple_worker = simple_worker_setting in {
        "1",
        "true",
        "yes",
        "on",
    }
    worker_cls = SimpleWorker if use_simple_worker else Worker
    worker = worker_cls(
        [get_queue()],
        connection=connection,
        name=get_worker_name(),
    )
    worker.work()


def main(argv: list[str] | None = None) -> None:
    """Run the worker or its container health probe."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--health-check", action="store_true")
    args = parser.parse_args(argv)
    if args.health_check:
        try:
            check_worker_health()
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        return
    run_worker()


if __name__ == "__main__":
    main()
