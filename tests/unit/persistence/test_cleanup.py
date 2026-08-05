from datetime import datetime, timedelta, timezone
from pathlib import Path
import pickle

from sqlalchemy import func, select

from webapp.persistence import Base
from webapp.persistence import cleanup as cleanup_module
from webapp.persistence.database import build_engine, create_session_factory
from webapp.persistence.models import (
    AccessLog,
    ArtifactJob,
    ArtifactRecord,
    ConfigAuditLog,
    SessionRecord,
    UserQuery,
)


def _artifact(
    *,
    selector_hash: str,
    scope: str,
    storage_uri: Path,
    created_at: datetime,
    expires_at: datetime | None = None,
) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_type="cleanup_test",
        scope=scope,
        owner_principal_id="__public__" if scope == "public" else "user-1",
        selector_hash=selector_hash,
        selector_payload={"selector": selector_hash},
        parameter_hash="params",
        parameter_payload={},
        pipeline_version="test",
        model_version="test",
        storage_uri=str(storage_uri),
        status="ready",
        created_at=created_at,
        expires_at=expires_at,
    )


def test_cleanup_enforces_retention_and_payload_safety(tmp_path, monkeypatch):
    database_path = tmp_path / "cleanup.db"
    artifact_root = tmp_path / "artifacts"
    session_root = tmp_path / "sessions"
    outside_path = tmp_path / "outside"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setenv("DOCUSCOPE_ARTIFACT_STORE_ROOT", str(artifact_root))
    monkeypatch.setenv("DOCUSCOPE_SESSION_ARTIFACT_ROOT", str(session_root))
    build_engine.cache_clear()
    create_session_factory.cache_clear()

    Base.metadata.create_all(build_engine())

    now = datetime.now(timezone.utc)
    old = now - timedelta(days=10)
    recent = now - timedelta(hours=1)
    removable_path = artifact_root / "private-expired"
    explicit_path = artifact_root / "public-explicit-expiry"
    shared_path = artifact_root / "shared"
    for path in (removable_path, explicit_path, shared_path, outside_path):
        path.mkdir(parents=True)
        (path / "payload.json").write_text("{}", encoding="utf-8")

    session_artifact = session_root / "expired" / "ds_tokens.gz"
    session_artifact.parent.mkdir(parents=True)
    session_artifact.write_bytes(b"private")
    session_data = {
        "target": {
            "_artifact_refs": {
                "ds_tokens": {
                    "storage_type": "gzip_pickle",
                    "path": str(session_artifact),
                }
            }
        }
    }

    session_factory = create_session_factory()
    with session_factory() as session:
        expired_private = _artifact(
            selector_hash="expired-private",
            scope="private",
            storage_uri=removable_path,
            created_at=old,
        )
        explicit_public = _artifact(
            selector_hash="explicit-public",
            scope="public",
            storage_uri=explicit_path,
            created_at=recent,
            expires_at=now - timedelta(seconds=1),
        )
        current_public = _artifact(
            selector_hash="current-public",
            scope="public",
            storage_uri=shared_path,
            created_at=old,
        )
        expired_shared_private = _artifact(
            selector_hash="shared-private",
            scope="private",
            storage_uri=shared_path,
            created_at=old,
        )
        active_private = _artifact(
            selector_hash="active-private",
            scope="private",
            storage_uri=artifact_root / "active-private",
            created_at=old,
        )
        unsafe_private = _artifact(
            selector_hash="unsafe-private",
            scope="private",
            storage_uri=outside_path,
            created_at=old,
        )
        session.add_all(
            [
                expired_private,
                explicit_public,
                current_public,
                expired_shared_private,
                active_private,
                unsafe_private,
            ]
        )
        session.flush()
        session.add_all(
            [
                ArtifactJob(
                    artifact_id=expired_private.artifact_id,
                    artifact_type="cleanup_test",
                    scope="private",
                    requester_principal_id="user-1",
                    selector_hash="expired-private",
                    selector_payload={},
                    parameter_hash="params",
                    parameter_payload={},
                    pipeline_version="test",
                    model_version="test",
                    status="completed",
                    created_at=old,
                    finished_at=old,
                ),
                ArtifactJob(
                    artifact_id=active_private.artifact_id,
                    artifact_type="cleanup_test",
                    scope="private",
                    requester_principal_id="user-1",
                    selector_hash="active-private",
                    selector_payload={},
                    parameter_hash="params",
                    parameter_payload={},
                    pipeline_version="test",
                    model_version="test",
                    status="running",
                    created_at=old,
                    started_at=old,
                ),
                ArtifactJob(
                    artifact_type="old-terminal",
                    scope="public",
                    selector_hash="old-terminal",
                    selector_payload={},
                    parameter_hash="params",
                    parameter_payload={},
                    pipeline_version="test",
                    model_version="test",
                    status="failed",
                    created_at=old,
                    finished_at=old,
                ),
                SessionRecord(
                    session_id="expired-session",
                    user_id="user-1",
                    data=pickle.dumps(session_data),
                    created_at=old,
                    updated_at=old,
                    expires_at=now - timedelta(seconds=1),
                    size_bytes=1,
                ),
                UserQuery(
                    user_id="hashed-user",
                    session_id="expired-session",
                    query_timestamp=recent,
                ),
                UserQuery(
                    user_id="hashed-user",
                    session_id="active-session",
                    query_timestamp=old,
                ),
                AccessLog(
                    email="user@example.edu",
                    access_granted=True,
                    timestamp=now - timedelta(days=31),
                ),
                ConfigAuditLog(
                    config_key="test",
                    updated_at=now - timedelta(days=91),
                ),
            ]
        )
        session.commit()

    dry_run = cleanup_module.run_cleanup(dry_run=True, now=now)
    assert dry_run.sessions == 1
    assert dry_run.user_queries == 2
    assert dry_run.artifacts == 4
    assert dry_run.artifact_jobs == 2
    assert dry_run.access_logs == 1
    assert dry_run.config_audit_logs == 1
    assert dry_run.payload_directories == 2
    assert session_artifact.exists()
    assert removable_path.exists()

    result = cleanup_module.run_cleanup(now=now)
    assert result == dry_run.__class__(
        dry_run=False,
        sessions=1,
        user_queries=2,
        artifacts=4,
        artifact_jobs=2,
        access_logs=1,
        config_audit_logs=1,
        payload_directories=2,
    )
    assert not session_artifact.exists()
    assert not removable_path.exists()
    assert not explicit_path.exists()
    assert shared_path.exists()
    assert outside_path.exists()

    with session_factory() as session:
        remaining_selectors = set(
            session.scalars(select(ArtifactRecord.selector_hash)).all()
        )
        assert remaining_selectors == {"current-public", "active-private"}
        assert session.scalar(select(func.count()).select_from(SessionRecord)) == 0
        assert session.scalar(select(func.count()).select_from(UserQuery)) == 0
        assert session.scalar(select(func.count()).select_from(AccessLog)) == 0
        assert session.scalar(select(func.count()).select_from(ConfigAuditLog)) == 0

    build_engine.cache_clear()
    create_session_factory.cache_clear()
