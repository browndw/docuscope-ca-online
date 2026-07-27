"""Reusable shared-artifact workflow helpers."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

from webapp.persistence.registry import ArtifactIdentity, ArtifactRegistryService


@dataclass(frozen=True)
class SharedArtifactDecision:
    """Decision returned by shared-artifact reservation helpers."""

    state: str
    job_id: int | None = None
    payload: Any = None


class SharedArtifactWorkflow:
    """Reusable control flow for shared artifact reuse and coordination."""

    def __init__(self, registry: ArtifactRegistryService, logger) -> None:
        self._registry = registry
        self._logger = logger

    def load_ready(
        self,
        identity: ArtifactIdentity | None,
        loader: Callable[[Any], Any],
        *,
        cache_name: str,
    ) -> tuple[Any, Any] | None:
        """Load a ready artifact and its payload if it exists."""

        if identity is None:
            self._logger.debug(
                f"Shared {cache_name} cache bypassed: no built-in corpus identity available"
            )
            return None

        try:
            artifact = self._registry.find_ready_artifact(identity)
            if artifact is None:
                self._logger.info(
                    f"Shared {cache_name} cache miss for selector=%s params=%s",
                    identity.selector_hash,
                    identity.parameter_hash,
                )
                return None

            payload = loader(artifact)
            self._logger.info(
                f"Shared {cache_name} cache hit for artifact_id=%s selector=%s",
                artifact.artifact_id,
                identity.selector_hash,
            )
            return artifact, payload
        except Exception as exc:
            self._logger.warning(f"Shared {cache_name} cache load failed: {exc}")
            return None

    def reserve(
        self,
        identity: ArtifactIdentity | None,
        *,
        cache_name: str,
        ready_loader: Callable[[], Any | None] | None = None,
        poll_attempts: int = 0,
        poll_interval_seconds: float = 0.0,
    ) -> SharedArtifactDecision:
        """Reserve an artifact identity or return a ready/pending decision."""

        if identity is None:
            return SharedArtifactDecision("bypass")

        try:
            reservation = self._registry.reserve_artifact(identity)
        except Exception as exc:
            self._logger.warning(f"Shared {cache_name} cache reservation failed: {exc}")
            return SharedArtifactDecision("bypass")

        if reservation.state == "reserved":
            if reservation.job is not None:
                self._registry.mark_job_running(reservation.job.job_id)
            self._logger.info(
                f"Reserved shared {cache_name} artifact selector=%s for computation",
                identity.selector_hash,
            )
            return SharedArtifactDecision(
                "reserved",
                job_id=reservation.job.job_id if reservation.job is not None else None,
            )

        if reservation.state == "ready" and ready_loader is not None:
            payload = ready_loader()
            if payload is not None:
                self._logger.info(
                    f"Shared {cache_name} artifact became ready during "
                    f"reservation selector=%s",
                    identity.selector_hash,
                )
                return SharedArtifactDecision("ready", payload=payload)

        if reservation.state == "pending":
            self._logger.info(
                f"Shared {cache_name} artifact already pending selector=%s",
                identity.selector_hash,
            )
            if ready_loader is not None and poll_attempts > 0:
                for _ in range(poll_attempts):
                    time.sleep(poll_interval_seconds)
                    payload = ready_loader()
                    if payload is not None:
                        return SharedArtifactDecision("ready", payload=payload)
            return SharedArtifactDecision("pending")

        return SharedArtifactDecision("bypass")

    def store(
        self,
        identity: ArtifactIdentity | None,
        job_id: int | None,
        *,
        cache_name: str,
        store_func: Callable[[ArtifactIdentity], Any],
    ) -> Any | None:
        """Store an artifact and complete or fail the associated job."""

        if identity is None:
            self._logger.debug(
                f"Shared {cache_name} cache store skipped: "
                f"no built-in corpus identity available"
            )
            return None

        try:
            artifact = store_func(identity)
            if job_id is not None:
                self._registry.mark_job_completed(job_id, artifact.artifact_id)
            self._logger.info(
                f"Stored shared {cache_name} artifact_id=%s selector=%s",
                artifact.artifact_id,
                identity.selector_hash,
            )
            return artifact
        except Exception as exc:
            if job_id is not None:
                self._registry.mark_job_failed(job_id, str(exc))
            self._logger.warning(f"Shared {cache_name} cache store failed: {exc}")
            return None
