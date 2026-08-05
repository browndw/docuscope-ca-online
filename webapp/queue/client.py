"""Queue client helpers for Redis/RQ smoke tests and future background jobs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json

from redis import Redis
from rq import Queue, Retry
from rq.job import Callback

from webapp.persistence import ArtifactIdentity, registry_service
from webapp.persistence.registry import (
    build_shared_collocation_identity,
    build_shared_keyness_identity,
    build_shared_keyness_parts_identity,
    build_shared_ngram_identity,
    get_pipeline_version,
)
from webapp.corpus_paths import is_builtin_corpus_ref, make_portable_corpus_path
from webapp.queue.config import RedisQueueConfig, get_redis_queue_config


RQ_SMOKE_ARTIFACT_TYPE = "rq_smoke_result"
INTERNAL_TARGET_ARTIFACT_TYPE = "internal_target_ready"
CONTROL_PLANE_FAILURE_CALLBACK = Callback(
    "webapp.queue.tasks.mark_control_plane_job_failed"
)


@dataclass(frozen=True)
class QueueSmokeEnqueueResult:
    """Result returned when a queue smoke job is requested."""

    state: str
    control_plane_job_id: int | None
    rq_job_id: str | None = None
    artifact_id: int | None = None


@dataclass(frozen=True)
class QueueInternalTargetEnqueueResult:
    """Result returned when a built-in target prep job is requested."""

    state: str
    control_plane_job_id: int | None
    rq_job_id: str | None = None
    artifact_id: int | None = None


@dataclass(frozen=True)
class QueueKeynessEnqueueResult:
    """Result returned when a built-in keyness job is requested."""

    state: str
    control_plane_job_id: int | None
    rq_job_id: str | None = None
    artifact_id: int | None = None


@dataclass(frozen=True)
class QueueCollocationEnqueueResult:
    """Result returned when a built-in collocation job is requested."""

    state: str
    control_plane_job_id: int | None
    rq_job_id: str | None = None
    artifact_id: int | None = None


@dataclass(frozen=True)
class QueueKeynessPartsEnqueueResult:
    """Result returned when a built-in corpus-parts keyness job is requested."""

    state: str
    control_plane_job_id: int | None
    rq_job_id: str | None = None
    artifact_id: int | None = None


@dataclass(frozen=True)
class QueueNgramEnqueueResult:
    """Result returned when a built-in n-gram/cluster job is requested."""

    state: str
    control_plane_job_id: int | None
    rq_job_id: str | None = None
    artifact_id: int | None = None


@dataclass(frozen=True)
class QueuePlotbotEnqueueResult:
    """Result returned when a Plotbot generation job is requested."""

    state: str
    rq_job_id: str


def _hash_payload(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_redis_connection() -> Redis:
    """Return the cached Redis client for the current queue configuration.

    Cached per process so callers share one pooled connection instead of
    opening a new client (and socket) on every enqueue/status-poll call.
    """

    return _build_redis_connection(get_redis_queue_config().redis_url)


@lru_cache(maxsize=1)
def _build_redis_connection(redis_url: str) -> Redis:
    return Redis.from_url(redis_url)


def get_queue() -> Queue:
    """Return the configured RQ queue."""

    config = get_redis_queue_config()
    return Queue(config.queue_name, connection=get_redis_connection())


def get_plotbot_queue() -> Queue:
    """Return the configured RQ queue for built-in-only Plotbot jobs."""

    config = get_redis_queue_config()
    return Queue(config.plotbot_queue_name, connection=get_redis_connection())


def _build_job_retry(config: RedisQueueConfig) -> Retry | None:
    """Return the bounded retry policy for queued jobs, or None to disable retries."""

    if config.max_retries <= 0:
        return None
    return Retry(max=config.max_retries, interval=config.retry_interval_seconds)


def _normalize_rq_status(status: object) -> str:
    """Convert RQ status values into a lowercase string for stable checks."""

    value = getattr(status, "value", status)
    return str(value).strip().lower()


def build_queue_smoke_identity(
    request_key: str,
    requester_principal_id: str | None = None,
) -> ArtifactIdentity:
    """Build a normalized identity for a small registry-backed smoke test."""

    selector_payload = {
        "request_type": "rq_smoke_test",
        "request_key": request_key,
    }
    parameter_payload = {
        "job_kind": "smoke",
    }
    return ArtifactIdentity(
        artifact_type=RQ_SMOKE_ARTIFACT_TYPE,
        scope="private" if requester_principal_id else "public",
        owner_principal_id=requester_principal_id,
        selector_hash=_hash_payload(selector_payload),
        selector_payload=selector_payload,
        parameter_hash=_hash_payload(parameter_payload),
        parameter_payload=parameter_payload,
        pipeline_version=get_pipeline_version(),
        model_version="rq-smoke-v1",
    )


def build_internal_target_identity(
    corpus_path: str,
    requester_principal_id: str | None = None,
) -> ArtifactIdentity:
    """Build a normalized identity for built-in target preparation work."""

    portable_corpus_path = make_portable_corpus_path(corpus_path)

    selector_payload = {
        "request_type": "internal_target_prepare",
        "corpus_path": portable_corpus_path,
    }
    parameter_payload = {
        "job_kind": "internal_target_prepare",
        "warm_shared_frequency": True,
    }
    return ArtifactIdentity(
        artifact_type=INTERNAL_TARGET_ARTIFACT_TYPE,
        scope="private" if requester_principal_id else "public",
        owner_principal_id=requester_principal_id,
        selector_hash=_hash_payload(selector_payload),
        selector_payload=selector_payload,
        parameter_hash=_hash_payload(parameter_payload),
        parameter_payload=parameter_payload,
        pipeline_version=get_pipeline_version(),
        model_version="internal-target-v1",
    )


def enqueue_registry_smoke_test(
    request_key: str = "local-smoke",
    requester_principal_id: str | None = None,
) -> QueueSmokeEnqueueResult:
    """Enqueue a small background job through Redis/RQ and the control plane."""

    config = get_redis_queue_config()
    if not config.enabled:
        raise RuntimeError(
            "Redis/RQ queueing is disabled. Set DOCUSCOPE_RQ_ENABLED=1 to enable it."
        )

    identity = build_queue_smoke_identity(request_key, requester_principal_id)
    reservation = registry_service.reserve_artifact(identity)

    if reservation.state == "ready" and reservation.artifact is not None:
        return QueueSmokeEnqueueResult(
            state="ready",
            control_plane_job_id=None,
            artifact_id=reservation.artifact.artifact_id,
        )

    if reservation.job is None:
        raise RuntimeError("Artifact reservation did not return a job to enqueue.")

    queue = get_queue()
    rq_job_id = f"rq-smoke-{reservation.job.job_id}"
    existing_job = queue.fetch_job(rq_job_id)
    existing_status = (
        _normalize_rq_status(existing_job.get_status())
        if existing_job is not None
        else ""
    )
    if reservation.state == "pending" and existing_job is not None:
        return QueueSmokeEnqueueResult(
            state="pending",
            control_plane_job_id=reservation.job.job_id,
            rq_job_id=existing_job.id,
            artifact_id=reservation.artifact.artifact_id if reservation.artifact else None,
        )

    if existing_job is not None and existing_status in {
        "queued",
        "started",
        "scheduled",
        "deferred",
    }:
        return QueueSmokeEnqueueResult(
            state="queued",
            control_plane_job_id=reservation.job.job_id,
            rq_job_id=existing_job.id,
            artifact_id=reservation.artifact.artifact_id if reservation.artifact else None,
        )

    rq_job = queue.enqueue(
        "webapp.queue.tasks.run_registry_smoke_test",
        reservation.job.job_id,
        job_id=rq_job_id,
        job_timeout=config.job_timeout,
        result_ttl=config.result_ttl,
        retry=_build_job_retry(config),
        on_failure=CONTROL_PLANE_FAILURE_CALLBACK,
    )
    return QueueSmokeEnqueueResult(
        state="queued",
        control_plane_job_id=reservation.job.job_id,
        rq_job_id=rq_job.id,
        artifact_id=reservation.artifact.artifact_id if reservation.artifact else None,
    )


def enqueue_internal_target_preparation(
    corpus_path: str,
    requester_principal_id: str | None = None,
) -> QueueInternalTargetEnqueueResult:
    """Enqueue preparation work for a built-in internal target corpus."""

    config = get_redis_queue_config()
    if not config.enabled:
        raise RuntimeError(
            "Redis/RQ queueing is disabled. Set DOCUSCOPE_RQ_ENABLED=1 to enable it."
        )

    portable_corpus_path = make_portable_corpus_path(corpus_path)
    identity = build_internal_target_identity(portable_corpus_path, requester_principal_id)
    reservation = registry_service.reserve_artifact(identity)

    if reservation.state == "ready" and reservation.artifact is not None:
        return QueueInternalTargetEnqueueResult(
            state="ready",
            control_plane_job_id=None,
            artifact_id=reservation.artifact.artifact_id,
        )

    if reservation.job is None:
        raise RuntimeError("Internal target reservation did not return a job to enqueue.")

    queue = get_queue()
    rq_job_id = f"internal-target-{reservation.job.job_id}"
    existing_job = queue.fetch_job(rq_job_id)
    existing_status = (
        _normalize_rq_status(existing_job.get_status())
        if existing_job is not None
        else ""
    )
    if reservation.state == "pending" and existing_job is not None:
        return QueueInternalTargetEnqueueResult(
            state="pending",
            control_plane_job_id=reservation.job.job_id,
            rq_job_id=existing_job.id,
            artifact_id=reservation.artifact.artifact_id if reservation.artifact else None,
        )

    if existing_job is not None and existing_status in {
        "queued",
        "started",
        "scheduled",
        "deferred",
    }:
        return QueueInternalTargetEnqueueResult(
            state="queued",
            control_plane_job_id=reservation.job.job_id,
            rq_job_id=existing_job.id,
            artifact_id=reservation.artifact.artifact_id if reservation.artifact else None,
        )

    rq_job = queue.enqueue(
        "webapp.queue.tasks.run_internal_target_preparation",
        reservation.job.job_id,
        portable_corpus_path,
        job_id=rq_job_id,
        job_timeout=config.job_timeout,
        result_ttl=config.result_ttl,
        retry=_build_job_retry(config),
        on_failure=CONTROL_PLANE_FAILURE_CALLBACK,
    )
    return QueueInternalTargetEnqueueResult(
        state="queued",
        control_plane_job_id=reservation.job.job_id,
        rq_job_id=rq_job.id,
        artifact_id=reservation.artifact.artifact_id if reservation.artifact else None,
    )


def enqueue_keyness_preparation(
    target_source: str,
    reference_source: str,
    threshold: float,
    swap_target: bool,
) -> QueueKeynessEnqueueResult:
    """Enqueue shared keyness generation for built-in corpus comparisons."""

    config = get_redis_queue_config()
    if not config.enabled:
        raise RuntimeError(
            "Redis/RQ queueing is disabled. Set DOCUSCOPE_RQ_ENABLED=1 to enable it."
        )

    identity = build_shared_keyness_identity(
        target_source=target_source,
        reference_source=reference_source,
        threshold=threshold,
        swap_target=swap_target,
    )

    reservation = registry_service.reserve_artifact(identity)

    if reservation.state == "ready" and reservation.artifact is not None:
        return QueueKeynessEnqueueResult(
            state="ready",
            control_plane_job_id=None,
            artifact_id=reservation.artifact.artifact_id,
        )

    if reservation.job is None:
        raise RuntimeError("Keyness reservation did not return a job to enqueue.")

    queue = get_queue()
    rq_job_id = f"keyness-{reservation.job.job_id}"
    existing_job = queue.fetch_job(rq_job_id)
    existing_status = (
        _normalize_rq_status(existing_job.get_status())
        if existing_job is not None
        else ""
    )
    if reservation.state == "pending" and existing_job is not None:
        return QueueKeynessEnqueueResult(
            state="pending",
            control_plane_job_id=reservation.job.job_id,
            rq_job_id=existing_job.id,
            artifact_id=reservation.artifact.artifact_id if reservation.artifact else None,
        )

    if existing_job is not None and existing_status in {
        "queued",
        "started",
        "scheduled",
        "deferred",
    }:
        return QueueKeynessEnqueueResult(
            state="queued",
            control_plane_job_id=reservation.job.job_id,
            rq_job_id=existing_job.id,
            artifact_id=reservation.artifact.artifact_id if reservation.artifact else None,
        )

    rq_job = queue.enqueue(
        "webapp.queue.tasks.run_keyness_preparation",
        reservation.job.job_id,
        target_source,
        reference_source,
        threshold,
        swap_target,
        job_id=rq_job_id,
        job_timeout=config.job_timeout,
        result_ttl=config.result_ttl,
        retry=_build_job_retry(config),
        on_failure=CONTROL_PLANE_FAILURE_CALLBACK,
    )
    return QueueKeynessEnqueueResult(
        state="queued",
        control_plane_job_id=reservation.job.job_id,
        rq_job_id=rq_job.id,
        artifact_id=reservation.artifact.artifact_id if reservation.artifact else None,
    )


def enqueue_collocation_preparation(
    target_source: str,
    node_word: str,
    node_tag: str | None,
    to_left: int,
    to_right: int,
    stat_mode: str,
    count_by: str,
) -> QueueCollocationEnqueueResult:
    """Enqueue shared collocation generation for a built-in target corpus."""

    config = get_redis_queue_config()
    if not config.enabled:
        raise RuntimeError(
            "Redis/RQ queueing is disabled. Set DOCUSCOPE_RQ_ENABLED=1 to enable it."
        )

    identity = build_shared_collocation_identity(
        target_source=target_source,
        node_word=node_word,
        node_tag=node_tag,
        to_left=to_left,
        to_right=to_right,
        stat_mode=stat_mode,
        count_by=count_by,
    )

    reservation = registry_service.reserve_artifact(identity)

    if reservation.state == "ready" and reservation.artifact is not None:
        return QueueCollocationEnqueueResult(
            state="ready",
            control_plane_job_id=None,
            artifact_id=reservation.artifact.artifact_id,
        )

    if reservation.job is None:
        raise RuntimeError("Collocation reservation did not return a job to enqueue.")

    queue = get_queue()
    rq_job_id = f"collocation-{reservation.job.job_id}"
    existing_job = queue.fetch_job(rq_job_id)
    existing_status = (
        _normalize_rq_status(existing_job.get_status())
        if existing_job is not None
        else ""
    )
    if reservation.state == "pending" and existing_job is not None:
        return QueueCollocationEnqueueResult(
            state="pending",
            control_plane_job_id=reservation.job.job_id,
            rq_job_id=existing_job.id,
            artifact_id=reservation.artifact.artifact_id if reservation.artifact else None,
        )

    if existing_job is not None and existing_status in {
        "queued",
        "started",
        "scheduled",
        "deferred",
    }:
        return QueueCollocationEnqueueResult(
            state="queued",
            control_plane_job_id=reservation.job.job_id,
            rq_job_id=existing_job.id,
            artifact_id=reservation.artifact.artifact_id if reservation.artifact else None,
        )

    rq_job = queue.enqueue(
        "webapp.queue.tasks.run_collocation_preparation",
        reservation.job.job_id,
        target_source,
        node_word,
        node_tag,
        to_left,
        to_right,
        stat_mode,
        count_by,
        job_id=rq_job_id,
        job_timeout=config.job_timeout,
        result_ttl=config.result_ttl,
        retry=_build_job_retry(config),
        on_failure=CONTROL_PLANE_FAILURE_CALLBACK,
    )
    return QueueCollocationEnqueueResult(
        state="queued",
        control_plane_job_id=reservation.job.job_id,
        rq_job_id=rq_job.id,
        artifact_id=reservation.artifact.artifact_id if reservation.artifact else None,
    )


def enqueue_keyness_parts_preparation(
    target_source: str,
    target_categories: list[str],
    reference_categories: list[str],
    threshold: float,
    swap_target: bool,
) -> QueueKeynessPartsEnqueueResult:
    """Enqueue shared keyness generation for built-in corpus part comparisons."""

    config = get_redis_queue_config()
    if not config.enabled:
        raise RuntimeError(
            "Redis/RQ queueing is disabled. Set DOCUSCOPE_RQ_ENABLED=1 to enable it."
        )

    identity = build_shared_keyness_parts_identity(
        target_source=target_source,
        target_categories=target_categories,
        reference_categories=reference_categories,
        threshold=threshold,
        swap_target=swap_target,
    )

    reservation = registry_service.reserve_artifact(identity)

    if reservation.state == "ready" and reservation.artifact is not None:
        return QueueKeynessPartsEnqueueResult(
            state="ready",
            control_plane_job_id=None,
            artifact_id=reservation.artifact.artifact_id,
        )

    if reservation.job is None:
        raise RuntimeError("Keyness-parts reservation did not return a job to enqueue.")

    queue = get_queue()
    rq_job_id = f"keyness-parts-{reservation.job.job_id}"
    existing_job = queue.fetch_job(rq_job_id)
    existing_status = (
        _normalize_rq_status(existing_job.get_status())
        if existing_job is not None
        else ""
    )
    if reservation.state == "pending" and existing_job is not None:
        return QueueKeynessPartsEnqueueResult(
            state="pending",
            control_plane_job_id=reservation.job.job_id,
            rq_job_id=existing_job.id,
            artifact_id=reservation.artifact.artifact_id if reservation.artifact else None,
        )

    if existing_job is not None and existing_status in {
        "queued",
        "started",
        "scheduled",
        "deferred",
    }:
        return QueueKeynessPartsEnqueueResult(
            state="queued",
            control_plane_job_id=reservation.job.job_id,
            rq_job_id=existing_job.id,
            artifact_id=reservation.artifact.artifact_id if reservation.artifact else None,
        )

    rq_job = queue.enqueue(
        "webapp.queue.tasks.run_keyness_parts_preparation",
        reservation.job.job_id,
        target_source,
        target_categories,
        reference_categories,
        threshold,
        swap_target,
        job_id=rq_job_id,
        job_timeout=config.job_timeout,
        result_ttl=config.result_ttl,
        retry=_build_job_retry(config),
        on_failure=CONTROL_PLANE_FAILURE_CALLBACK,
    )
    return QueueKeynessPartsEnqueueResult(
        state="queued",
        control_plane_job_id=reservation.job.job_id,
        rq_job_id=rq_job.id,
        artifact_id=reservation.artifact.artifact_id if reservation.artifact else None,
    )


def enqueue_ngram_preparation(
    target_source: str,
    analysis_type: str,
    ngram_span: int,
    count_by: str,
    from_anchor: str | None = None,
    node_word: str | None = None,
    tag: str | None = None,
    position: int | None = None,
    search_type: str | None = None,
) -> QueueNgramEnqueueResult:
    """Enqueue shared n-gram/cluster generation for a built-in target corpus."""

    config = get_redis_queue_config()
    if not config.enabled:
        raise RuntimeError(
            "Redis/RQ queueing is disabled. Set DOCUSCOPE_RQ_ENABLED=1 to enable it."
        )

    identity = build_shared_ngram_identity(
        target_source=target_source,
        analysis_type=analysis_type,
        ngram_span=ngram_span,
        count_by=count_by,
        from_anchor=from_anchor,
        node_word=node_word,
        tag=tag,
        position=position,
        search_type=search_type,
    )
    reservation = registry_service.reserve_artifact(identity)

    if reservation.state == "ready" and reservation.artifact is not None:
        return QueueNgramEnqueueResult(
            state="ready",
            control_plane_job_id=None,
            artifact_id=reservation.artifact.artifact_id,
        )

    if reservation.job is None:
        raise RuntimeError("N-gram reservation did not return a job to enqueue.")

    queue = get_queue()
    rq_job_id = f"ngram-{reservation.job.job_id}"
    existing_job = queue.fetch_job(rq_job_id)
    existing_status = (
        _normalize_rq_status(existing_job.get_status())
        if existing_job is not None
        else ""
    )
    if reservation.state == "pending" and existing_job is not None:
        return QueueNgramEnqueueResult(
            state="pending",
            control_plane_job_id=reservation.job.job_id,
            rq_job_id=existing_job.id,
            artifact_id=reservation.artifact.artifact_id if reservation.artifact else None,
        )

    if existing_job is not None and existing_status in {
        "queued",
        "started",
        "scheduled",
        "deferred",
    }:
        return QueueNgramEnqueueResult(
            state="queued",
            control_plane_job_id=reservation.job.job_id,
            rq_job_id=existing_job.id,
            artifact_id=reservation.artifact.artifact_id if reservation.artifact else None,
        )

    rq_job = queue.enqueue(
        "webapp.queue.tasks.run_ngram_preparation",
        reservation.job.job_id,
        target_source,
        analysis_type,
        ngram_span,
        count_by,
        from_anchor,
        node_word,
        tag,
        position,
        search_type,
        job_id=rq_job_id,
        job_timeout=config.job_timeout,
        result_ttl=config.result_ttl,
        retry=_build_job_retry(config),
        on_failure=CONTROL_PLANE_FAILURE_CALLBACK,
    )
    return QueueNgramEnqueueResult(
        state="queued",
        control_plane_job_id=reservation.job.job_id,
        rq_job_id=rq_job.id,
        artifact_id=reservation.artifact.artifact_id if reservation.artifact else None,
    )


def enqueue_plotbot_generation(
    dataframe_records: list[dict[str, object]],
    source_refs: list[str],
    plot_lib: str,
    user_input: str,
    llm_params: dict[str, object],
    schema: str | None = None,
    code_chunk: str | None = None,
    cached_code: str | None = None,
    api_key: str = "",
    requester_principal_id: str = "anonymous",
    model_version: str = "plotbot-v1",
) -> QueuePlotbotEnqueueResult:
    """Enqueue a TTL-backed Plotbot job for built-in-only corpus data."""

    portable_sources = [
        make_portable_corpus_path(source) for source in source_refs if source
    ]
    if not portable_sources or not all(
        is_builtin_corpus_ref(source) for source in portable_sources
    ):
        raise ValueError(
            "Queued Plotbot generation requires built-in-only corpus data."
        )

    config = get_redis_queue_config()
    if not config.enabled:
        raise RuntimeError(
            "Redis/RQ queueing is disabled. Set DOCUSCOPE_RQ_ENABLED=1 to enable it."
        )

    queue = get_plotbot_queue()
    request_hash = _hash_payload({
        "table_hash": _hash_payload({"records": dataframe_records}),
        "source_refs": sorted(portable_sources),
        "plot_lib": plot_lib,
        "user_input": user_input,
        "llm_params": llm_params,
        "schema": schema or "",
        "code_chunk_hash": _hash_payload({"code_chunk": code_chunk or ""}),
        "cached_code_hash": _hash_payload({"cached_code": cached_code or ""}),
        "requester_hash": _hash_payload({
            "requester": requester_principal_id or "anonymous"
        }),
        "model_version": model_version,
    })
    rq_job_id = f"plotbot-{request_hash[:32]}"
    existing_job = queue.fetch_job(rq_job_id)
    existing_status = (
        _normalize_rq_status(existing_job.get_status(refresh=True))
        if existing_job is not None
        else ""
    )
    if existing_status == "finished" and isinstance(existing_job.result, dict):
        return QueuePlotbotEnqueueResult(
            state="ready",
            rq_job_id=existing_job.id,
        )

    if existing_job is not None and existing_status in {
        "queued",
        "started",
        "scheduled",
        "deferred",
    }:
        return QueuePlotbotEnqueueResult(
            state="pending",
            rq_job_id=existing_job.id,
        )

    if existing_job is not None:
        existing_job.delete()

    rq_job = queue.enqueue(
        "webapp.queue.tasks.run_plotbot_generation",
        dataframe_records,
        plot_lib,
        user_input,
        llm_params,
        schema,
        code_chunk,
        cached_code,
        api_key,
        job_id=rq_job_id,
        job_timeout=config.job_timeout,
        result_ttl=config.result_ttl,
        retry=_build_job_retry(config),
    )
    return QueuePlotbotEnqueueResult(
        state="queued",
        rq_job_id=rq_job.id,
    )
