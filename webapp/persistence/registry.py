"""Artifact registry services for shared and private analytical artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from webapp.persistence.database import create_session_factory
from webapp.persistence.models import ArtifactJob, ArtifactRecord


ARTIFACT_STORE_ROOT = Path("webapp/_artifacts")
KEYNESS_ARTIFACT_TYPE = "keyness_bundle"
KEYNESS_PARTS_ARTIFACT_TYPE = "keyness_parts_bundle"
FREQUENCY_ARTIFACT_TYPE = "frequency_bundle"
COLLOCATION_ARTIFACT_TYPE = "collocation_bundle"
NGRAM_ARTIFACT_TYPE = "ngram_bundle"
JSON_ARTIFACT_FILENAME = "payload.json"


@dataclass(frozen=True)
class ArtifactIdentity:
    """Normalized identity for an artifact registry entry."""

    artifact_type: str
    scope: str
    selector_hash: str
    selector_payload: dict[str, Any]
    parameter_hash: str
    parameter_payload: dict[str, Any]
    pipeline_version: str
    model_version: str
    owner_principal_id: str | None = None


@dataclass(frozen=True)
class ReservationResult:
    """Result of attempting to reserve an artifact identity for computation."""

    state: str
    artifact: ArtifactRecord | None
    job: ArtifactJob | None = None


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _hash_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _normalize_source_path(raw_path: str) -> str:
    path = Path(raw_path).resolve()
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def get_pipeline_version() -> str:
    try:
        from importlib.metadata import version
        return version("docuscope-ca-online")
    except Exception:
        return "0.0.0+local"


@lru_cache(maxsize=64)
def _load_cached_keyness_bundle(
    storage_uri: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load a keyness bundle once per process and reuse it across requests."""

    artifact_dir = Path(storage_uri)
    return (
        pl.read_parquet(artifact_dir / "kw_pos.parquet"),
        pl.read_parquet(artifact_dir / "kw_ds.parquet"),
        pl.read_parquet(artifact_dir / "kt_pos.parquet"),
        pl.read_parquet(artifact_dir / "kt_ds.parquet"),
    )


@lru_cache(maxsize=64)
def _load_cached_frequency_bundle(
    storage_uri: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load a frequency bundle once per process and reuse it across requests."""

    artifact_dir = Path(storage_uri)
    return (
        pl.read_parquet(artifact_dir / "ft_pos.parquet"),
        pl.read_parquet(artifact_dir / "ft_ds.parquet"),
    )


@lru_cache(maxsize=64)
def _load_cached_collocation_bundle(storage_uri: str) -> pl.DataFrame:
    """Load a collocation bundle once per process and reuse it across requests."""

    artifact_dir = Path(storage_uri)
    return pl.read_parquet(artifact_dir / "collocations.parquet")


def build_shared_keyness_identity(
    target_source: str,
    reference_source: str,
    threshold: float,
    swap_target: bool,
    model_version: str = "preprocessed",
) -> ArtifactIdentity:
    """Build a normalized identity for a shared built-in keyness artifact."""

    selector_payload = {
        "comparison_type": "built_in_corpora",
        "target": {"source": _normalize_source_path(target_source)},
        "reference": {"source": _normalize_source_path(reference_source)},
    }
    parameter_payload = {
        "threshold": threshold,
        "swap_target": swap_target,
    }
    return ArtifactIdentity(
        artifact_type=KEYNESS_ARTIFACT_TYPE,
        scope="public",
        selector_hash=_hash_payload(selector_payload),
        selector_payload=selector_payload,
        parameter_hash=_hash_payload(parameter_payload),
        parameter_payload=parameter_payload,
        pipeline_version=get_pipeline_version(),
        model_version=model_version,
    )


def build_shared_keyness_parts_identity(
    target_source: str,
    target_categories: list[str],
    reference_categories: list[str],
    threshold: float,
    swap_target: bool,
    model_version: str = "preprocessed",
) -> ArtifactIdentity:
    """Build a normalized identity for a shared built-in corpus-parts keyness artifact."""

    selector_payload = {
        "comparison_type": "built_in_corpus_parts",
        "target": {"source": _normalize_source_path(target_source)},
    }
    parameter_payload = {
        "target_categories": sorted(str(category) for category in target_categories),
        "reference_categories": sorted(str(category) for category in reference_categories),
        "threshold": threshold,
        "swap_target": swap_target,
    }
    return ArtifactIdentity(
        artifact_type=KEYNESS_PARTS_ARTIFACT_TYPE,
        scope="public",
        selector_hash=_hash_payload(selector_payload),
        selector_payload=selector_payload,
        parameter_hash=_hash_payload(parameter_payload),
        parameter_payload=parameter_payload,
        pipeline_version=get_pipeline_version(),
        model_version=model_version,
    )


def build_shared_frequency_identity(
    target_source: str,
    model_version: str = "preprocessed",
) -> ArtifactIdentity:
    """Build a normalized identity for a shared built-in frequency artifact."""

    selector_payload = {
        "analysis_type": "built_in_frequency",
        "target": {"source": _normalize_source_path(target_source)},
    }
    parameter_payload = {"count_by": "both"}
    return ArtifactIdentity(
        artifact_type=FREQUENCY_ARTIFACT_TYPE,
        scope="public",
        selector_hash=_hash_payload(selector_payload),
        selector_payload=selector_payload,
        parameter_hash=_hash_payload(parameter_payload),
        parameter_payload=parameter_payload,
        pipeline_version=get_pipeline_version(),
        model_version=model_version,
    )


def build_shared_collocation_identity(
    target_source: str,
    node_word: str,
    node_tag: str | None,
    to_left: int,
    to_right: int,
    stat_mode: str,
    count_by: str,
    model_version: str = "preprocessed",
) -> ArtifactIdentity:
    """Build a normalized identity for a shared built-in collocation artifact."""

    selector_payload = {
        "analysis_type": "built_in_collocations",
        "target": {"source": _normalize_source_path(target_source)},
    }
    parameter_payload = {
        "node_word": node_word.strip(),
        "node_tag": node_tag,
        "to_left": to_left,
        "to_right": to_right,
        "stat_mode": stat_mode,
        "count_by": count_by,
    }
    return ArtifactIdentity(
        artifact_type=COLLOCATION_ARTIFACT_TYPE,
        scope="public",
        selector_hash=_hash_payload(selector_payload),
        selector_payload=selector_payload,
        parameter_hash=_hash_payload(parameter_payload),
        parameter_payload=parameter_payload,
        pipeline_version=get_pipeline_version(),
        model_version=model_version,
    )


def build_shared_ngram_identity(
    target_source: str,
    analysis_type: str,
    ngram_span: int,
    count_by: str,
    from_anchor: str | None = None,
    node_word: str | None = None,
    tag: str | None = None,
    position: int | None = None,
    search_type: str | None = None,
    model_version: str = "preprocessed",
) -> ArtifactIdentity:
    """Build a normalized identity for shared built-in n-gram/cluster artifacts."""

    selector_payload = {
        "analysis_type": "built_in_ngrams_clusters",
        "target": {"source": _normalize_source_path(target_source)},
    }
    parameter_payload = {
        "analysis_type": analysis_type,
        "ngram_span": ngram_span,
        "count_by": count_by,
        "from_anchor": from_anchor,
        "node_word": node_word.strip() if isinstance(node_word, str) else node_word,
        "tag": tag,
        "position": position,
        "search_type": search_type,
    }
    return ArtifactIdentity(
        artifact_type=NGRAM_ARTIFACT_TYPE,
        scope="public",
        selector_hash=_hash_payload(selector_payload),
        selector_payload=selector_payload,
        parameter_hash=_hash_payload(parameter_payload),
        parameter_payload=parameter_payload,
        pipeline_version=get_pipeline_version(),
        model_version=model_version,
    )


class ArtifactRegistryService:
    """Small registry service around artifact and job models."""

    def __init__(self) -> None:
        self._session_factory = create_session_factory()

    def _artifact_storage_exists(self, artifact: ArtifactRecord) -> bool:
        """Return True when an artifact points to a materialized storage location."""

        if not artifact.storage_uri:
            return False
        return Path(artifact.storage_uri).exists()

    def find_ready_artifact(self, identity: ArtifactIdentity) -> ArtifactRecord | None:
        """Return a ready artifact matching the normalized identity."""

        with self._session_factory() as session:
            stmt = select(ArtifactRecord).where(
                ArtifactRecord.artifact_type == identity.artifact_type,
                ArtifactRecord.scope == identity.scope,
                ArtifactRecord.selector_hash == identity.selector_hash,
                ArtifactRecord.parameter_hash == identity.parameter_hash,
                ArtifactRecord.pipeline_version == identity.pipeline_version,
                ArtifactRecord.model_version == identity.model_version,
                ArtifactRecord.owner_principal_id == identity.owner_principal_id,
                ArtifactRecord.status == "ready",
            )
            artifact = session.execute(stmt).scalar_one_or_none()
            if artifact is None:
                return None

            if not self._artifact_storage_exists(artifact):
                artifact.status = "failed"
                artifact.last_accessed_at = datetime.now(timezone.utc)
                session.commit()
                return None

            artifact.last_accessed_at = datetime.now(timezone.utc)
            artifact.access_count += 1
            session.commit()
            session.refresh(artifact)
            return artifact

    def _find_active_job(
        self,
        session,
        identity: ArtifactIdentity,
        artifact_id: int | None,
    ) -> ArtifactJob | None:
        """Return the newest pending or running job for an artifact identity."""

        stmt = select(ArtifactJob).where(
            ArtifactJob.artifact_type == identity.artifact_type,
            ArtifactJob.scope == identity.scope,
            ArtifactJob.selector_hash == identity.selector_hash,
            ArtifactJob.parameter_hash == identity.parameter_hash,
            ArtifactJob.pipeline_version == identity.pipeline_version,
            ArtifactJob.model_version == identity.model_version,
            ArtifactJob.status.in_(("pending", "running")),
        )
        if artifact_id is not None:
            stmt = stmt.where(ArtifactJob.artifact_id == artifact_id)
        stmt = stmt.order_by(ArtifactJob.created_at.desc())
        return session.execute(stmt).scalars().first()

    def _create_pending_job(
        self,
        session,
        identity: ArtifactIdentity,
        artifact_id: int,
    ) -> ArtifactJob:
        """Create a pending job row linked to an artifact."""

        job = ArtifactJob(
            artifact_id=artifact_id,
            artifact_type=identity.artifact_type,
            scope=identity.scope,
            requester_principal_id=identity.owner_principal_id,
            selector_hash=identity.selector_hash,
            selector_payload=identity.selector_payload,
            parameter_hash=identity.parameter_hash,
            parameter_payload=identity.parameter_payload,
            pipeline_version=identity.pipeline_version,
            model_version=identity.model_version,
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
        session.add(job)
        session.flush()
        return job

    def reserve_artifact(self, identity: ArtifactIdentity) -> ReservationResult:
        """Reserve an artifact identity for computation or return existing state."""

        now = datetime.now(timezone.utc)

        with self._session_factory() as session:
            stmt = select(ArtifactRecord).where(
                ArtifactRecord.artifact_type == identity.artifact_type,
                ArtifactRecord.scope == identity.scope,
                ArtifactRecord.selector_hash == identity.selector_hash,
                ArtifactRecord.parameter_hash == identity.parameter_hash,
                ArtifactRecord.pipeline_version == identity.pipeline_version,
                ArtifactRecord.model_version == identity.model_version,
                ArtifactRecord.owner_principal_id == identity.owner_principal_id,
            )
            artifact = session.execute(stmt).scalar_one_or_none()
            if artifact is not None:
                if artifact.status == "ready":
                    if not self._artifact_storage_exists(artifact):
                        artifact.status = "pending"
                        artifact.storage_uri = ""
                        artifact.last_accessed_at = now
                        job = self._create_pending_job(
                            session,
                            identity,
                            artifact.artifact_id,
                        )
                        session.commit()
                        session.refresh(artifact)
                        session.refresh(job)
                        return ReservationResult("reserved", artifact, job)

                    artifact.last_accessed_at = now
                    artifact.access_count += 1
                    session.commit()
                    session.refresh(artifact)
                    return ReservationResult("ready", artifact)

                if artifact.status == "pending":
                    job = self._find_active_job(session, identity, artifact.artifact_id)
                    if job is not None:
                        return ReservationResult("pending", artifact, job)

                    # Recover from orphaned pending artifacts where the active
                    # control-plane job row is missing (for example, after an
                    # interrupted enqueue path). Recreate the pending job so
                    # callers can enqueue work again.
                    artifact.last_accessed_at = now
                    recovered_job = self._create_pending_job(
                        session,
                        identity,
                        artifact.artifact_id,
                    )
                    session.commit()
                    session.refresh(artifact)
                    session.refresh(recovered_job)
                    return ReservationResult("reserved", artifact, recovered_job)

                artifact.status = "pending"
                artifact.storage_uri = ""
                artifact.last_accessed_at = now
                job = self._create_pending_job(session, identity, artifact.artifact_id)
                session.commit()
                session.refresh(artifact)
                session.refresh(job)
                return ReservationResult("reserved", artifact, job)

            artifact = ArtifactRecord(
                artifact_type=identity.artifact_type,
                scope=identity.scope,
                owner_principal_id=identity.owner_principal_id,
                selector_hash=identity.selector_hash,
                selector_payload=identity.selector_payload,
                pipeline_version=identity.pipeline_version,
                model_version=identity.model_version,
                parameter_hash=identity.parameter_hash,
                parameter_payload=identity.parameter_payload,
                storage_uri="",
                status="pending",
                created_at=now,
                last_accessed_at=now,
                access_count=0,
            )
            session.add(artifact)
            session.flush()
            job = self._create_pending_job(session, identity, artifact.artifact_id)

            try:
                session.commit()
                session.refresh(artifact)
                session.refresh(job)
                return ReservationResult("reserved", artifact, job)
            except IntegrityError:
                session.rollback()
                artifact = session.execute(stmt).scalar_one_or_none()
                if artifact is None:
                    return ReservationResult("pending", None, None)
                job = self._find_active_job(
                    session,
                    identity,
                    artifact.artifact_id,
                )
                return ReservationResult(artifact.status, artifact, job)

    def create_job(self, identity: ArtifactIdentity) -> ArtifactJob:
        """Register a generation job for an artifact identity."""

        with self._session_factory() as session:
            job = ArtifactJob(
                artifact_type=identity.artifact_type,
                scope=identity.scope,
                requester_principal_id=identity.owner_principal_id,
                selector_hash=identity.selector_hash,
                selector_payload=identity.selector_payload,
                parameter_hash=identity.parameter_hash,
                parameter_payload=identity.parameter_payload,
                pipeline_version=identity.pipeline_version,
                model_version=identity.model_version,
                status="completed",
                created_at=datetime.now(timezone.utc),
                started_at=datetime.now(timezone.utc),
                finished_at=datetime.now(timezone.utc),
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            return job

    def mark_job_running(self, job_id: int, worker_id: str | None = None) -> None:
        """Mark a pending job as running."""

        with self._session_factory() as session:
            job = session.get(ArtifactJob, job_id)
            if job is None:
                return
            job.status = "running"
            job.worker_id = worker_id
            job.started_at = datetime.now(timezone.utc)
            session.commit()

    def mark_job_completed(self, job_id: int, artifact_id: int) -> None:
        """Mark a job as completed and link it to the ready artifact."""

        with self._session_factory() as session:
            job = session.get(ArtifactJob, job_id)
            if job is None:
                return
            job.artifact_id = artifact_id
            job.status = "completed"
            job.finished_at = datetime.now(timezone.utc)
            session.commit()

    def mark_job_failed(self, job_id: int, failure_reason: str) -> None:
        """Mark a job as failed and release its artifact reservation."""

        with self._session_factory() as session:
            job = session.get(ArtifactJob, job_id)
            if job is None:
                return

            job.status = "failed"
            job.failure_reason = failure_reason
            job.finished_at = datetime.now(timezone.utc)

            if job.artifact_id is not None:
                artifact = session.get(ArtifactRecord, job.artifact_id)
                if artifact is not None and artifact.status == "pending":
                    artifact.status = "failed"

            session.commit()

    def get_job_by_id(self, job_id: int) -> ArtifactJob | None:
        """Return a job by primary key without mutating its state."""

        with self._session_factory() as session:
            return session.get(ArtifactJob, job_id)

    def get_artifact_by_id(self, artifact_id: int) -> ArtifactRecord | None:
        """Return an artifact by primary key without changing its state."""

        with self._session_factory() as session:
            return session.get(ArtifactRecord, artifact_id)

    def store_json_artifact(
        self,
        identity: ArtifactIdentity,
        payload: dict[str, Any],
    ) -> ArtifactRecord:
        """Persist a small JSON payload in the artifact store and registry."""

        artifact_dir = ARTIFACT_STORE_ROOT / identity.artifact_type / identity.selector_hash
        artifact_dir = artifact_dir / identity.parameter_hash
        artifact_dir.mkdir(parents=True, exist_ok=True)

        payload_path = artifact_dir / JSON_ARTIFACT_FILENAME
        payload_path.write_text(
            json.dumps(payload, sort_keys=True, indent=2),
            encoding="utf-8",
        )

        storage_uri = artifact_dir.as_posix()
        now = datetime.now(timezone.utc)

        with self._session_factory() as session:
            stmt = select(ArtifactRecord).where(
                ArtifactRecord.artifact_type == identity.artifact_type,
                ArtifactRecord.scope == identity.scope,
                ArtifactRecord.selector_hash == identity.selector_hash,
                ArtifactRecord.parameter_hash == identity.parameter_hash,
                ArtifactRecord.pipeline_version == identity.pipeline_version,
                ArtifactRecord.model_version == identity.model_version,
                ArtifactRecord.owner_principal_id == identity.owner_principal_id,
            )
            artifact = session.execute(stmt).scalar_one_or_none()
            if artifact is None:
                artifact = ArtifactRecord(
                    artifact_type=identity.artifact_type,
                    scope=identity.scope,
                    owner_principal_id=identity.owner_principal_id,
                    selector_hash=identity.selector_hash,
                    selector_payload=identity.selector_payload,
                    pipeline_version=identity.pipeline_version,
                    model_version=identity.model_version,
                    parameter_hash=identity.parameter_hash,
                    parameter_payload=identity.parameter_payload,
                    storage_uri=storage_uri,
                    status="ready",
                    created_at=now,
                    last_accessed_at=now,
                    access_count=1,
                )
                session.add(artifact)
            else:
                artifact.storage_uri = storage_uri
                artifact.status = "ready"
                artifact.last_accessed_at = now
                artifact.access_count = max(artifact.access_count, 0) + 1

            session.commit()
            session.refresh(artifact)
            return artifact

    def load_json_artifact(self, artifact: ArtifactRecord) -> dict[str, Any]:
        """Load a JSON artifact payload from the artifact store."""

        payload_path = Path(artifact.storage_uri) / JSON_ARTIFACT_FILENAME
        return json.loads(payload_path.read_text(encoding="utf-8"))

    def store_keyness_bundle(
        self,
        identity: ArtifactIdentity,
        keyness_frames: dict[str, pl.DataFrame],
    ) -> ArtifactRecord:
        """Persist a keyness bundle in the artifact store and register it."""

        artifact_dir = ARTIFACT_STORE_ROOT / identity.artifact_type / identity.selector_hash
        artifact_dir = artifact_dir / identity.parameter_hash
        artifact_dir.mkdir(parents=True, exist_ok=True)

        file_map = {
            "kw_pos": "kw_pos.parquet",
            "kw_ds": "kw_ds.parquet",
            "kt_pos": "kt_pos.parquet",
            "kt_ds": "kt_ds.parquet",
        }
        for key, filename in file_map.items():
            keyness_frames[key].write_parquet(artifact_dir / filename)

        storage_uri = artifact_dir.as_posix()
        now = datetime.now(timezone.utc)

        with self._session_factory() as session:
            stmt = select(ArtifactRecord).where(
                ArtifactRecord.artifact_type == identity.artifact_type,
                ArtifactRecord.scope == identity.scope,
                ArtifactRecord.selector_hash == identity.selector_hash,
                ArtifactRecord.parameter_hash == identity.parameter_hash,
                ArtifactRecord.pipeline_version == identity.pipeline_version,
                ArtifactRecord.model_version == identity.model_version,
                ArtifactRecord.owner_principal_id == identity.owner_principal_id,
            )
            artifact = session.execute(stmt).scalar_one_or_none()
            if artifact is None:
                artifact = ArtifactRecord(
                    artifact_type=identity.artifact_type,
                    scope=identity.scope,
                    owner_principal_id=identity.owner_principal_id,
                    selector_hash=identity.selector_hash,
                    selector_payload=identity.selector_payload,
                    pipeline_version=identity.pipeline_version,
                    model_version=identity.model_version,
                    parameter_hash=identity.parameter_hash,
                    parameter_payload=identity.parameter_payload,
                    storage_uri=storage_uri,
                    status="ready",
                    created_at=now,
                    last_accessed_at=now,
                    access_count=0,
                )
                session.add(artifact)
            else:
                artifact.storage_uri = storage_uri
                artifact.status = "ready"
                artifact.selector_payload = identity.selector_payload
                artifact.parameter_payload = identity.parameter_payload
                artifact.last_accessed_at = now

            session.commit()
            session.refresh(artifact)
            return artifact

    def load_keyness_bundle(self, artifact: ArtifactRecord) -> dict[str, pl.DataFrame]:
        """Load a keyness bundle from parquet files in the artifact store."""

        kw_pos, kw_ds, kt_pos, kt_ds = _load_cached_keyness_bundle(artifact.storage_uri)
        return {
            "kw_pos": kw_pos.clone(),
            "kw_ds": kw_ds.clone(),
            "kt_pos": kt_pos.clone(),
            "kt_ds": kt_ds.clone(),
        }

    def store_keyness_parts_bundle(
        self,
        identity: ArtifactIdentity,
        keyness_frames: dict[str, pl.DataFrame],
        metadata: dict[str, Any],
    ) -> ArtifactRecord:
        """Persist a corpus-parts keyness bundle and metadata in the artifact store."""

        artifact_dir = ARTIFACT_STORE_ROOT / identity.artifact_type / identity.selector_hash
        artifact_dir = artifact_dir / identity.parameter_hash
        artifact_dir.mkdir(parents=True, exist_ok=True)

        file_map = {
            "kw_pos_cp": "kw_pos.parquet",
            "kw_ds_cp": "kw_ds.parquet",
            "kt_pos_cp": "kt_pos.parquet",
            "kt_ds_cp": "kt_ds.parquet",
        }
        for key, filename in file_map.items():
            keyness_frames[key].write_parquet(artifact_dir / filename)

        metadata_path = artifact_dir / JSON_ARTIFACT_FILENAME
        metadata_path.write_text(
            json.dumps(metadata, sort_keys=True, indent=2),
            encoding="utf-8",
        )

        storage_uri = artifact_dir.as_posix()
        now = datetime.now(timezone.utc)

        with self._session_factory() as session:
            stmt = select(ArtifactRecord).where(
                ArtifactRecord.artifact_type == identity.artifact_type,
                ArtifactRecord.scope == identity.scope,
                ArtifactRecord.selector_hash == identity.selector_hash,
                ArtifactRecord.parameter_hash == identity.parameter_hash,
                ArtifactRecord.pipeline_version == identity.pipeline_version,
                ArtifactRecord.model_version == identity.model_version,
                ArtifactRecord.owner_principal_id == identity.owner_principal_id,
            )
            artifact = session.execute(stmt).scalar_one_or_none()
            if artifact is None:
                artifact = ArtifactRecord(
                    artifact_type=identity.artifact_type,
                    scope=identity.scope,
                    owner_principal_id=identity.owner_principal_id,
                    selector_hash=identity.selector_hash,
                    selector_payload=identity.selector_payload,
                    pipeline_version=identity.pipeline_version,
                    model_version=identity.model_version,
                    parameter_hash=identity.parameter_hash,
                    parameter_payload=identity.parameter_payload,
                    storage_uri=storage_uri,
                    status="ready",
                    created_at=now,
                    last_accessed_at=now,
                    access_count=0,
                )
                session.add(artifact)
            else:
                artifact.storage_uri = storage_uri
                artifact.status = "ready"
                artifact.selector_payload = identity.selector_payload
                artifact.parameter_payload = identity.parameter_payload
                artifact.last_accessed_at = now

            session.commit()
            session.refresh(artifact)
            return artifact

    def load_keyness_parts_bundle(self, artifact: ArtifactRecord) -> dict[str, Any]:
        """Load a corpus-parts keyness bundle from the artifact store."""

        kw_pos, kw_ds, kt_pos, kt_ds = _load_cached_keyness_bundle(artifact.storage_uri)
        metadata_path = Path(artifact.storage_uri) / JSON_ARTIFACT_FILENAME
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {
            "kw_pos_cp": kw_pos.clone(),
            "kw_ds_cp": kw_ds.clone(),
            "kt_pos_cp": kt_pos.clone(),
            "kt_ds_cp": kt_ds.clone(),
            "metadata": metadata,
        }

    def store_frequency_bundle(
        self,
        identity: ArtifactIdentity,
        frequency_frames: dict[str, pl.DataFrame],
    ) -> ArtifactRecord:
        """Persist a frequency bundle in the artifact store and register it."""

        artifact_dir = ARTIFACT_STORE_ROOT / identity.artifact_type / identity.selector_hash
        artifact_dir = artifact_dir / identity.parameter_hash
        artifact_dir.mkdir(parents=True, exist_ok=True)

        file_map = {
            "ft_pos": "ft_pos.parquet",
            "ft_ds": "ft_ds.parquet",
        }
        for key, filename in file_map.items():
            frequency_frames[key].write_parquet(artifact_dir / filename)

        storage_uri = artifact_dir.as_posix()
        now = datetime.now(timezone.utc)

        with self._session_factory() as session:
            stmt = select(ArtifactRecord).where(
                ArtifactRecord.artifact_type == identity.artifact_type,
                ArtifactRecord.scope == identity.scope,
                ArtifactRecord.selector_hash == identity.selector_hash,
                ArtifactRecord.parameter_hash == identity.parameter_hash,
                ArtifactRecord.pipeline_version == identity.pipeline_version,
                ArtifactRecord.model_version == identity.model_version,
                ArtifactRecord.owner_principal_id == identity.owner_principal_id,
            )
            artifact = session.execute(stmt).scalar_one_or_none()
            if artifact is None:
                artifact = ArtifactRecord(
                    artifact_type=identity.artifact_type,
                    scope=identity.scope,
                    owner_principal_id=identity.owner_principal_id,
                    selector_hash=identity.selector_hash,
                    selector_payload=identity.selector_payload,
                    pipeline_version=identity.pipeline_version,
                    model_version=identity.model_version,
                    parameter_hash=identity.parameter_hash,
                    parameter_payload=identity.parameter_payload,
                    storage_uri=storage_uri,
                    status="ready",
                    created_at=now,
                    last_accessed_at=now,
                    access_count=0,
                )
                session.add(artifact)
            else:
                artifact.storage_uri = storage_uri
                artifact.status = "ready"
                artifact.selector_payload = identity.selector_payload
                artifact.parameter_payload = identity.parameter_payload
                artifact.last_accessed_at = now

            session.commit()
            session.refresh(artifact)
            return artifact

    def load_frequency_bundle(self, artifact: ArtifactRecord) -> dict[str, pl.DataFrame]:
        """Load a frequency bundle from parquet files in the artifact store."""

        ft_pos, ft_ds = _load_cached_frequency_bundle(artifact.storage_uri)
        return {
            "ft_pos": ft_pos,
            "ft_ds": ft_ds,
        }

    def store_collocation_bundle(
        self,
        identity: ArtifactIdentity,
        collocations: pl.DataFrame,
    ) -> ArtifactRecord:
        """Persist a collocation bundle in the artifact store and register it."""

        artifact_dir = ARTIFACT_STORE_ROOT / identity.artifact_type / identity.selector_hash
        artifact_dir = artifact_dir / identity.parameter_hash
        artifact_dir.mkdir(parents=True, exist_ok=True)

        collocations.write_parquet(artifact_dir / "collocations.parquet")

        storage_uri = artifact_dir.as_posix()
        now = datetime.now(timezone.utc)

        with self._session_factory() as session:
            stmt = select(ArtifactRecord).where(
                ArtifactRecord.artifact_type == identity.artifact_type,
                ArtifactRecord.scope == identity.scope,
                ArtifactRecord.selector_hash == identity.selector_hash,
                ArtifactRecord.parameter_hash == identity.parameter_hash,
                ArtifactRecord.pipeline_version == identity.pipeline_version,
                ArtifactRecord.model_version == identity.model_version,
                ArtifactRecord.owner_principal_id == identity.owner_principal_id,
            )
            artifact = session.execute(stmt).scalar_one_or_none()
            if artifact is None:
                artifact = ArtifactRecord(
                    artifact_type=identity.artifact_type,
                    scope=identity.scope,
                    owner_principal_id=identity.owner_principal_id,
                    selector_hash=identity.selector_hash,
                    selector_payload=identity.selector_payload,
                    pipeline_version=identity.pipeline_version,
                    model_version=identity.model_version,
                    parameter_hash=identity.parameter_hash,
                    parameter_payload=identity.parameter_payload,
                    storage_uri=storage_uri,
                    status="ready",
                    created_at=now,
                    last_accessed_at=now,
                    access_count=0,
                )
                session.add(artifact)
            else:
                artifact.storage_uri = storage_uri
                artifact.status = "ready"
                artifact.selector_payload = identity.selector_payload
                artifact.parameter_payload = identity.parameter_payload
                artifact.last_accessed_at = now

            session.commit()
            session.refresh(artifact)
            return artifact

    def load_collocation_bundle(self, artifact: ArtifactRecord) -> dict[str, pl.DataFrame]:
        """Load a collocation bundle from parquet files in the artifact store."""

        return {"collocations": _load_cached_collocation_bundle(artifact.storage_uri).clone()}

    def store_ngram_bundle(
        self,
        identity: ArtifactIdentity,
        ngrams: pl.DataFrame,
    ) -> ArtifactRecord:
        """Persist an n-gram/cluster bundle in the artifact store and register it."""

        artifact_dir = ARTIFACT_STORE_ROOT / identity.artifact_type / identity.selector_hash
        artifact_dir = artifact_dir / identity.parameter_hash
        artifact_dir.mkdir(parents=True, exist_ok=True)

        ngrams.write_parquet(artifact_dir / "ngrams.parquet")

        storage_uri = artifact_dir.as_posix()
        now = datetime.now(timezone.utc)

        with self._session_factory() as session:
            stmt = select(ArtifactRecord).where(
                ArtifactRecord.artifact_type == identity.artifact_type,
                ArtifactRecord.scope == identity.scope,
                ArtifactRecord.selector_hash == identity.selector_hash,
                ArtifactRecord.parameter_hash == identity.parameter_hash,
                ArtifactRecord.pipeline_version == identity.pipeline_version,
                ArtifactRecord.model_version == identity.model_version,
                ArtifactRecord.owner_principal_id == identity.owner_principal_id,
            )
            artifact = session.execute(stmt).scalar_one_or_none()
            if artifact is None:
                artifact = ArtifactRecord(
                    artifact_type=identity.artifact_type,
                    scope=identity.scope,
                    owner_principal_id=identity.owner_principal_id,
                    selector_hash=identity.selector_hash,
                    selector_payload=identity.selector_payload,
                    pipeline_version=identity.pipeline_version,
                    model_version=identity.model_version,
                    parameter_hash=identity.parameter_hash,
                    parameter_payload=identity.parameter_payload,
                    storage_uri=storage_uri,
                    status="ready",
                    created_at=now,
                    last_accessed_at=now,
                    access_count=0,
                )
                session.add(artifact)
            else:
                artifact.storage_uri = storage_uri
                artifact.status = "ready"
                artifact.selector_payload = identity.selector_payload
                artifact.parameter_payload = identity.parameter_payload
                artifact.last_accessed_at = now

            session.commit()
            session.refresh(artifact)
            return artifact

    def load_ngram_bundle(self, artifact: ArtifactRecord) -> dict[str, pl.DataFrame]:
        """Load an n-gram/cluster bundle from parquet files in the artifact store."""

        return {"ngrams": pl.read_parquet(Path(artifact.storage_uri) / "ngrams.parquet")}

    def load_artifact_payload(self, artifact: ArtifactRecord) -> dict[str, pl.DataFrame]:
        """Load artifact payload based on artifact type."""

        if artifact.artifact_type == KEYNESS_ARTIFACT_TYPE:
            return self.load_keyness_bundle(artifact)
        if artifact.artifact_type == KEYNESS_PARTS_ARTIFACT_TYPE:
            return self.load_keyness_parts_bundle(artifact)
        if artifact.artifact_type == FREQUENCY_ARTIFACT_TYPE:
            return self.load_frequency_bundle(artifact)
        if artifact.artifact_type == COLLOCATION_ARTIFACT_TYPE:
            return self.load_collocation_bundle(artifact)
        if artifact.artifact_type == NGRAM_ARTIFACT_TYPE:
            return self.load_ngram_bundle(artifact)
        raise ValueError(f"Unsupported artifact type: {artifact.artifact_type}")


registry_service = ArtifactRegistryService()
