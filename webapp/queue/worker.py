"""RQ worker entrypoint for Redis-backed background jobs."""

from __future__ import annotations

import os

from rq import SimpleWorker, Worker

from webapp.persistence import initialize_database_schema
from webapp.queue.client import get_queue, get_redis_connection
from webapp.queue.config import get_redis_queue_config


def main() -> None:
    """Run an RQ worker for the configured queue."""

    config = get_redis_queue_config()
    if not config.enabled:
        raise SystemExit(
            "Redis/RQ queueing is disabled. Set DOCUSCOPE_RQ_ENABLED=1 to run the worker."
        )

    initialize_database_schema()
    connection = get_redis_connection()
    use_simple_worker = os.environ.get("DOCUSCOPE_RQ_SIMPLE_WORKER", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    worker_cls = SimpleWorker if use_simple_worker else Worker
    worker = worker_cls([get_queue()], connection=connection)
    worker.work()


if __name__ == "__main__":
    main()