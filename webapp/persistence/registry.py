"""Artifact registry services for shared and private analytical artifacts."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterator

import polars as pl
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from webapp.persistence.database import create_session_factory
from webapp.persistence.models import ArtifactJob, ArtifactRecord
from webapp.corpus_paths import (
    is_builtin_corpus_ref,
    make_portable_corpus_path,
)


ARTIFACT_STORE_ROOT = Path("webapp/_artifacts")
ARTIFACT_STORE_ROOT_ENV = "DOCUSCOPE_ARTIFACT_STORE_ROOT"
KEYNESS_ARTIFACT_TYPE = "keyness_bundle"
KEYNESS_PARTS_ARTIFACT_TYPE = "keyness_parts_bundle"
FREQUENCY_ARTIFACT_TYPE = "frequency_bundle"
COLLOCATION_ARTIFACT_TYPE = "collocation_bundle"
NGRAM_ARTIFACT_TYPE = "ngram_bundle"
JSON_ARTIFACT_FILENAME = "payload.json"
MODEL_METADATA_FILENAME = "meta.json"
PUBLIC_OWNER_PRINCIPAL_ID = "__public__"
PRIVATE_ARTIFACT_TTL_HOURS_ENV = "DOCUSCOPE_PRIVATE_ARTIFACT_TTL_HOURS"


def _artifact_expiry(identity: "ArtifactIdentity", now: datetime) -> datetime | None:
    """Return expiry for private compatibility artifacts only."""

    if identity.scope != "private":
        return None
    return now + timedelta(
        hours=int(os.getenv(PRIVATE_ARTIFACT_TTL_HOURS_ENV, "24"))
    )


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

    def __post_init__(self) -> None:
        """Normalize ownership so public identities are unique in PostgreSQL."""

        if self.scope == "public":
            if self.owner_principal_id not in (None, PUBLIC_OWNER_PRINCIPAL_ID):
                raise ValueError("Public artifacts cannot have a private owner.")
            object.__setattr__(
                self,
                "owner_principal_id",
                PUBLIC_OWNER_PRINCIPAL_ID,
            )
        elif not self.owner_principal_id:
            raise ValueError("Private artifacts require an owner principal.")


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
    portable_path = make_portable_corpus_path(raw_path)
    if portable_path != raw_path:
        return portable_path

    path = Path(raw_path).resolve(strict=False)
    try:
        return path.relative_to(Path.cwd().resolve(strict=False)).as_posix()
    except ValueError:
        return path.as_posix()


def _require_builtin_source(raw_path: str) -> str:
    """Return a portable built-in source or reject private/session data."""

    portable_path = make_portable_corpus_path(raw_path)
    if not is_builtin_corpus_ref(portable_path):
        raise ValueError(
            "Durable shared artifacts require a built-in corpus source."
        )
    return portable_path


@lru_cache(maxsize=4)
def _load_model_fingerprint(model_dir: str) -> str:
    """Return a stable fingerprint from one bundled spaCy model's metadata."""

    metadata_path = Path(model_dir) / MODEL_METADATA_FILENAME
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unable to resolve model metadata from {metadata_path}."
        ) from exc

    name = str(metadata.get("name", "")).strip()
    version = str(metadata.get("version", "")).strip()
    build = str(metadata.get("spacy_git_version", "")).strip()
    if not name or not version:
        raise ValueError(
            f"Model metadata at {metadata_path} must define name and version."
        )

    fingerprint = f"{name}@{version}"
    return f"{fingerprint}+{build}" if build else fingerprint


def get_builtin_model_fingerprint(source: str) -> str:
    """Resolve the model fingerprint associated with a built-in corpus."""

    portable_source = _require_builtin_source(source)
    relative_parts = Path(
        portable_source.removeprefix("builtin:")
    ).parts
    if not relative_parts:
        raise ValueError(f"Invalid built-in corpus reference: {portable_source}")

    dictionary_family = relative_parts[0]
    project_root = Path(__file__).resolve().parents[2]
    model_dirs = {
        "ld": project_root / "webapp" / "_models" / "en_docusco_spacy",
        "cd": project_root / "webapp" / "_models" / "en_docusco_spacy_cd",
    }
    model_dir = model_dirs.get(dictionary_family)
    if model_dir is None:
        raise ValueError(
            f"Unknown built-in corpus model family: {dictionary_family}"
        )
    return _load_model_fingerprint(str(model_dir))


def _resolve_shared_model_version(
    sources: list[str],
    model_version: str | None,
) -> str:
    """Return the metadata-derived model fingerprint for built-in sources."""

    builtin_sources = [_require_builtin_source(source) for source in sources]
    fingerprints = {
        get_builtin_model_fingerprint(source) for source in builtin_sources
    }
    if len(fingerprints) != 1:
        raise ValueError(
            "Shared artifacts require built-in corpora produced by the same model."
        )
    fingerprint = fingerprints.pop()
    if model_version is not None and model_version.strip() != fingerprint:
        raise ValueError(
            "model_version must match the bundled model metadata fingerprint."
        )
    return fingerprint


def get_pipeline_version() -> str:
    try:
        from importlib.metadata import version
        return version("docuscope-ca-online")
    except Exception:
        return "0.0.0+local"


def get_artifact_store_root() -> Path:
    """Return the configured root for persisted artifact payload files."""

    configured_root = os.getenv(ARTIFACT_STORE_ROOT_ENV, "").strip()
    if configured_root:
        return Path(configured_root)
    return ARTIFACT_STORE_ROOT


def _build_artifact_dir(identity: ArtifactIdentity) -> Path:
    """Return the payload directory for an artifact identity."""

    identity_hash = _hash_payload({
        "artifact_type": identity.artifact_type,
        "scope": identity.scope,
        "owner_principal_id": identity.owner_principal_id,
        "selector_hash": identity.selector_hash,
        "parameter_hash": identity.parameter_hash,
        "pipeline_version": identity.pipeline_version,
        "model_version": identity.model_version,
    })
    return (
        get_artifact_store_root() /
        identity.artifact_type /
        identity_hash
    )


def _publish_staged_dir(staging_dir: Path, final_dir: Path) -> None:
    """Atomically swap a fully-written staging directory into its final path.

    Using a same-filesystem rename means readers only ever see either the
    previous complete artifact or the new complete one, never a partially
    written directory (e.g. if a worker crashes mid-write).
    """

    if final_dir.exists():
        stale_dir = final_dir.with_name(f"{final_dir.name}.stale-{uuid.uuid4().hex}")
        os.replace(final_dir, stale_dir)
        try:
            os.replace(staging_dir, final_dir)
        finally:
            shutil.rmtree(stale_dir, ignore_errors=True)
    else:
        os.replace(staging_dir, final_dir)


@contextmanager
def _staged_artifact_dir(final_dir: Path) -> Iterator[Path]:
    """Yield a temporary directory to write artifact files into, then publish it.

    On success, the staging directory is atomically renamed to `final_dir`.
    On failure, the partially written staging directory is discarded and the
    previous artifact at `final_dir` (if any) is left untouched.
    """

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(
        prefix=f".{final_dir.name}.staging-", dir=final_dir.parent
    ))
    try:
        yield staging_dir
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    else:
        _publish_staged_dir(staging_dir, final_dir)


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
    model_version: str | None = None,
) -> ArtifactIdentity:
    """Build a normalized identity for a shared built-in keyness artifact."""

    target_source = _require_builtin_source(target_source)
    reference_source = _require_builtin_source(reference_source)
    resolved_model_version = _resolve_shared_model_version(
        [target_source, reference_source],
        model_version,
    )
    selector_payload = {
        "comparison_type": "built_in_corpora",
        "target": {"source": target_source},
        "reference": {"source": reference_source},
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
        model_version=resolved_model_version,
    )


def build_shared_keyness_parts_identity(
    target_source: str,
    target_categories: list[str],
    reference_categories: list[str],
    threshold: float,
    swap_target: bool,
    model_version: str | None = None,
) -> ArtifactIdentity:
    """Build a normalized identity for a shared built-in corpus-parts keyness artifact."""

    target_source = _require_builtin_source(target_source)
    resolved_model_version = _resolve_shared_model_version(
        [target_source],
        model_version,
    )
    selector_payload = {
        "comparison_type": "built_in_corpus_parts",
        "target": {"source": target_source},
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
        model_version=resolved_model_version,
    )


def build_shared_frequency_identity(
    target_source: str,
    model_version: str | None = None,
) -> ArtifactIdentity:
    """Build a normalized identity for a shared built-in frequency artifact."""

    target_source = _require_builtin_source(target_source)
    resolved_model_version = _resolve_shared_model_version(
        [target_source],
        model_version,
    )
    selector_payload = {
        "analysis_type": "built_in_frequency",
        "target": {"source": target_source},
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
        model_version=resolved_model_version,
    )


def build_shared_collocation_identity(
    target_source: str,
    node_word: str,
    node_tag: str | None,
    to_left: int,
    to_right: int,
    stat_mode: str,
    count_by: str,
    model_version: str | None = None,
) -> ArtifactIdentity:
    """Build a normalized identity for a shared built-in collocation artifact."""

    target_source = _require_builtin_source(target_source)
    resolved_model_version = _resolve_shared_model_version(
        [target_source],
        model_version,
    )
    selector_payload = {
        "analysis_type": "built_in_collocations",
        "target": {"source": target_source},
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
        model_version=resolved_model_version,
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
    model_version: str | None = None,
) -> ArtifactIdentity:
    """Build a normalized identity for shared built-in n-gram/cluster artifacts."""

    target_source = _require_builtin_source(target_source)
    resolved_model_version = _resolve_shared_model_version(
        [target_source],
        model_version,
    )
    selector_payload = {
        "analysis_type": "built_in_ngrams_clusters",
        "target": {"source": target_source},
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
        model_version=resolved_model_version,
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
                        artifact.expires_at = _artifact_expiry(identity, now)
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
                    artifact.expires_at = _artifact_expiry(identity, now)
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
                    artifact.expires_at = _artifact_expiry(identity, now)
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
                artifact.expires_at = _artifact_expiry(identity, now)
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
                expires_at=_artifact_expiry(identity, now),
            )
            session.add(artifact)

            try:
                session.flush()
                job = self._create_pending_job(
                    session,
                    identity,
                    artifact.artifact_id,
                )
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

    def record_job_retry(self, job_id: int, failure_reason: str) -> None:
        """Record a failed attempt without releasing the active reservation."""

        with self._session_factory() as session:
            job = session.get(ArtifactJob, job_id)
            if job is None or job.status in {"completed", "failed"}:
                return
            job.retry_count += 1
            job.failure_reason = failure_reason
            job.finished_at = None
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

    def get_job_by_id_internal(self, job_id: int) -> ArtifactJob | None:
        """Return a job by primary key without mutating its state."""

        with self._session_factory() as session:
            return session.get(ArtifactJob, job_id)

    def get_public_job_by_id(self, job_id: int) -> ArtifactJob | None:
        """Return a job only when it belongs to the public artifact workflow."""

        job = self.get_job_by_id_internal(job_id)
        return job if job is not None and job.scope == "public" else None

    def get_job_by_id_for_principal(
        self,
        job_id: int,
        requester_principal_id: str,
    ) -> ArtifactJob | None:
        """Return a public job or a private job owned by the requester."""

        job = self.get_job_by_id_internal(job_id)
        if job is None or job.scope == "public":
            return job
        if requester_principal_id and (
            job.requester_principal_id == requester_principal_id
        ):
            return job
        if requester_principal_id and job.artifact_id is not None:
            with self._session_factory() as session:
                sharing_principal_id = session.scalar(
                    select(ArtifactRecord.sharing_principal_id).where(
                        ArtifactRecord.artifact_id == job.artifact_id
                    )
                )
            if sharing_principal_id == requester_principal_id:
                return job
        return None

    def get_artifact_by_id_internal(self, artifact_id: int) -> ArtifactRecord | None:
        """Return an artifact by primary key when its ready storage is usable."""

        with self._session_factory() as session:
            artifact = session.get(ArtifactRecord, artifact_id)
            if artifact is None:
                return None

            if artifact.status == "ready" and not self._artifact_storage_exists(artifact):
                artifact.status = "failed"
                artifact.last_accessed_at = datetime.now(timezone.utc)
                session.commit()
                return None

            return artifact

    def get_public_artifact_by_id(self, artifact_id: int) -> ArtifactRecord | None:
        """Return an artifact only when it is explicitly public."""

        artifact = self.get_artifact_by_id_internal(artifact_id)
        return artifact if artifact is not None and artifact.scope == "public" else None

    def get_artifact_by_id_for_principal(
        self,
        artifact_id: int,
        requester_principal_id: str,
    ) -> ArtifactRecord | None:
        """Return a public artifact or a private artifact accessible to requester."""

        artifact = self.get_artifact_by_id_internal(artifact_id)
        if artifact is None or artifact.scope == "public":
            return artifact
        if requester_principal_id and requester_principal_id in {
            artifact.owner_principal_id,
            artifact.sharing_principal_id,
        }:
            return artifact
        return None

    def store_json_artifact(
        self,
        identity: ArtifactIdentity,
        payload: dict[str, Any],
    ) -> ArtifactRecord:
        """Persist a small JSON payload in the artifact store and registry."""

        artifact_dir = _build_artifact_dir(identity)
        with _staged_artifact_dir(artifact_dir) as staging_dir:
            payload_path = staging_dir / JSON_ARTIFACT_FILENAME
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

        artifact_dir = _build_artifact_dir(identity)
        file_map = {
            "kw_pos": "kw_pos.parquet",
            "kw_ds": "kw_ds.parquet",
            "kt_pos": "kt_pos.parquet",
            "kt_ds": "kt_ds.parquet",
        }
        with _staged_artifact_dir(artifact_dir) as staging_dir:
            for key, filename in file_map.items():
                keyness_frames[key].write_parquet(staging_dir / filename)

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

        artifact_dir = _build_artifact_dir(identity)
        file_map = {
            "kw_pos_cp": "kw_pos.parquet",
            "kw_ds_cp": "kw_ds.parquet",
            "kt_pos_cp": "kt_pos.parquet",
            "kt_ds_cp": "kt_ds.parquet",
        }
        with _staged_artifact_dir(artifact_dir) as staging_dir:
            for key, filename in file_map.items():
                keyness_frames[key].write_parquet(staging_dir / filename)

            metadata_path = staging_dir / JSON_ARTIFACT_FILENAME
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

        artifact_dir = _build_artifact_dir(identity)
        file_map = {
            "ft_pos": "ft_pos.parquet",
            "ft_ds": "ft_ds.parquet",
        }
        with _staged_artifact_dir(artifact_dir) as staging_dir:
            for key, filename in file_map.items():
                frequency_frames[key].write_parquet(staging_dir / filename)

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

        artifact_dir = _build_artifact_dir(identity)
        with _staged_artifact_dir(artifact_dir) as staging_dir:
            collocations.write_parquet(staging_dir / "collocations.parquet")

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

        collocations = _load_cached_collocation_bundle(artifact.storage_uri).clone()
        return {"collocations": collocations}

    def store_ngram_bundle(
        self,
        identity: ArtifactIdentity,
        ngrams: pl.DataFrame,
    ) -> ArtifactRecord:
        """Persist an n-gram/cluster bundle in the artifact store and register it."""

        artifact_dir = _build_artifact_dir(identity)
        with _staged_artifact_dir(artifact_dir) as staging_dir:
            ngrams.write_parquet(staging_dir / "ngrams.parquet")

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
