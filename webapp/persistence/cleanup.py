"""Scheduled cleanup for control-plane rows and shared artifact payloads."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import signal
from threading import Event

from sqlalchemy import delete, exists, func, or_, select

from webapp.persistence.database import (
    create_session_factory,
    initialize_database_schema,
)
from webapp.persistence.models import (
    AccessLog,
    ArtifactJob,
    ArtifactRecord,
    ConfigAuditLog,
    SessionRecord,
    UserQuery,
)
from webapp.persistence.registry import get_artifact_store_root
from webapp.utilities.storage.postgres_session_backend import PostgresSessionBackend


ACTIVE_JOB_STATUSES = ("pending", "running")
TERMINAL_JOB_STATUSES = ("completed", "failed")


@dataclass(frozen=True)
class CleanupResult:
    """Counts removed, or eligible when dry-run mode is enabled."""

    dry_run: bool
    sessions: int
    user_queries: int
    artifacts: int
    artifact_jobs: int
    access_logs: int
    config_audit_logs: int
    payload_directories: int


def _env_days(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _safe_payload_path(storage_uri: str) -> Path | None:
    """Return a root-confined artifact payload path, or None when unsafe."""

    if not storage_uri:
        return None
    root = get_artifact_store_root().resolve(strict=False)
    path = Path(storage_uri).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError:
        return None
    if path == root:
        return None
    if not path.exists():
        return None
    return path


def run_cleanup(
    *,
    dry_run: bool = False,
    now: datetime | None = None,
) -> CleanupResult:
    """Apply one conservative retention pass and return cleanup metrics."""

    initialize_database_schema()
    session_factory = create_session_factory()
    current_time = now or datetime.now(timezone.utc)
    private_cutoff = current_time - timedelta(
        hours=int(os.environ.get("DOCUSCOPE_PRIVATE_ARTIFACT_TTL_HOURS", "24"))
    )
    terminal_job_cutoff = current_time - timedelta(
        days=_env_days("DOCUSCOPE_TERMINAL_JOB_RETENTION_DAYS", 7)
    )
    access_log_cutoff = current_time - timedelta(
        days=_env_days("DOCUSCOPE_ACCESS_LOG_RETENTION_DAYS", 30)
    )
    config_audit_cutoff = current_time - timedelta(
        days=_env_days("DOCUSCOPE_CONFIG_AUDIT_RETENTION_DAYS", 90)
    )
    query_cutoff = current_time - timedelta(days=7)

    active_job_exists = exists().where(
        ArtifactJob.artifact_id == ArtifactRecord.artifact_id,
        ArtifactJob.status.in_(ACTIVE_JOB_STATUSES),
    )
    artifact_eligible = (
        or_(
            ArtifactRecord.expires_at <= current_time,
            (
                (ArtifactRecord.scope == "private")
                & (ArtifactRecord.created_at <= private_cutoff)
            ),
        )
        & ~active_job_exists
    )

    with session_factory() as session:
        expired_session_ids = session.scalars(
            select(SessionRecord.session_id).where(
                SessionRecord.expires_at <= current_time
            )
        ).all()
        query_count = session.scalar(
            select(func.count()).select_from(UserQuery).where(
                or_(
                    UserQuery.query_timestamp < query_cutoff,
                    UserQuery.session_id.in_(expired_session_ids),
                )
            )
        ) or 0
        artifacts = session.scalars(
            select(ArtifactRecord).where(artifact_eligible)
        ).all()
        artifact_ids = [artifact.artifact_id for artifact in artifacts]
        payload_paths: dict[Path, str] = {}
        for artifact in artifacts:
            path = _safe_payload_path(artifact.storage_uri)
            if path is None:
                continue
            surviving_reference = session.scalar(
                select(func.count()).select_from(ArtifactRecord).where(
                    ArtifactRecord.storage_uri == artifact.storage_uri,
                    ArtifactRecord.artifact_id.not_in(artifact_ids),
                )
            )
            if not surviving_reference:
                payload_paths[path] = artifact.storage_uri

        linked_job_count = 0
        if artifact_ids:
            linked_job_count = session.scalar(
                select(func.count()).select_from(ArtifactJob).where(
                    ArtifactJob.artifact_id.in_(artifact_ids)
                )
            ) or 0
        old_terminal_job_count = session.scalar(
            select(func.count()).select_from(ArtifactJob).where(
                ArtifactJob.status.in_(TERMINAL_JOB_STATUSES),
                ArtifactJob.finished_at <= terminal_job_cutoff,
                or_(
                    ArtifactJob.artifact_id.is_(None),
                    ArtifactJob.artifact_id.not_in(artifact_ids),
                ),
            )
        ) or 0
        access_log_count = session.scalar(
            select(func.count()).select_from(AccessLog).where(
                AccessLog.timestamp < access_log_cutoff
            )
        ) or 0
        config_audit_count = session.scalar(
            select(func.count()).select_from(ConfigAuditLog).where(
                ConfigAuditLog.updated_at < config_audit_cutoff
            )
        ) or 0

        if not dry_run:
            if artifact_ids:
                session.execute(
                    delete(ArtifactJob).where(ArtifactJob.artifact_id.in_(artifact_ids))
                )
            session.execute(
                delete(ArtifactJob).where(
                    ArtifactJob.status.in_(TERMINAL_JOB_STATUSES),
                    ArtifactJob.finished_at <= terminal_job_cutoff,
                )
            )
            session.execute(delete(ArtifactRecord).where(artifact_eligible))
            session.execute(
                delete(AccessLog).where(AccessLog.timestamp < access_log_cutoff)
            )
            session.execute(
                delete(ConfigAuditLog).where(
                    ConfigAuditLog.updated_at < config_audit_cutoff
                )
            )
            session.commit()

    sessions_removed = len(expired_session_ids)
    if not dry_run:
        sessions_removed = PostgresSessionBackend().cleanup_expired_sessions()

    payloads_removed = 0
    if not dry_run:
        with session_factory() as session:
            for path, storage_uri in payload_paths.items():
                surviving_reference = session.scalar(
                    select(func.count()).select_from(ArtifactRecord).where(
                        ArtifactRecord.storage_uri == storage_uri
                    )
                )
                if surviving_reference:
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                    payloads_removed += 1
                elif path.exists():
                    path.unlink()
                    payloads_removed += 1

    return CleanupResult(
        dry_run=dry_run,
        sessions=sessions_removed,
        user_queries=int(query_count),
        artifacts=len(artifact_ids),
        artifact_jobs=int(linked_job_count + old_terminal_job_count),
        access_logs=int(access_log_count),
        config_audit_logs=int(config_audit_count),
        payload_directories=(len(payload_paths) if dry_run else payloads_removed),
    )


def main(argv: list[str] | None = None) -> None:
    """Run one cleanup pass or a periodic cleanup process."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=int(os.environ.get("DOCUSCOPE_CLEANUP_INTERVAL_SECONDS", "3600")),
    )
    args = parser.parse_args(argv)
    stop_event = Event()
    signal.signal(signal.SIGTERM, lambda *_args: stop_event.set())
    signal.signal(signal.SIGINT, lambda *_args: stop_event.set())

    while True:
        print(json.dumps(asdict(run_cleanup(dry_run=args.dry_run))), flush=True)
        if args.once or stop_event.wait(args.interval_seconds):
            return


if __name__ == "__main__":
    main()
