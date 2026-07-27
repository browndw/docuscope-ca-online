"""Configuration helpers for optional Redis/RQ background jobs."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class RedisQueueConfig:
    """Resolved Redis/RQ configuration."""

    enabled: bool
    redis_url: str
    queue_name: str
    job_timeout: int
    result_ttl: int


def get_redis_queue_config() -> RedisQueueConfig:
    """Return queue settings derived from environment variables."""

    return RedisQueueConfig(
        enabled=os.environ.get("DOCUSCOPE_RQ_ENABLED", "0") == "1",
        redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        queue_name=os.environ.get("DOCUSCOPE_RQ_QUEUE", "docuscope"),
        job_timeout=int(os.environ.get("DOCUSCOPE_RQ_JOB_TIMEOUT", "600")),
        result_ttl=int(os.environ.get("DOCUSCOPE_RQ_RESULT_TTL", "86400")),
    )