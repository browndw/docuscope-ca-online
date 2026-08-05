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
    max_retries: int
    retry_interval_seconds: int


def get_redis_queue_config() -> RedisQueueConfig:
    """Return queue settings derived from environment variables."""

    return RedisQueueConfig(
        enabled=os.environ.get("DOCUSCOPE_RQ_ENABLED", "0") == "1",
        redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        queue_name=os.environ.get("DOCUSCOPE_RQ_QUEUE", "docuscope"),
        job_timeout=int(os.environ.get("DOCUSCOPE_RQ_JOB_TIMEOUT", "600")),
        result_ttl=int(os.environ.get("DOCUSCOPE_RQ_RESULT_TTL", "86400")),
        # Bounded retries so a transient Redis/filesystem blip doesn't strand
        # a student on a permanently failed job; 0 disables retries entirely.
        max_retries=int(os.environ.get("DOCUSCOPE_RQ_MAX_RETRIES", "2")),
        retry_interval_seconds=int(
            os.environ.get("DOCUSCOPE_RQ_RETRY_INTERVAL_SECONDS", "15")
        ),
    )
