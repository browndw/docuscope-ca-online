"""Opt-in integration smoke tests for the Postgres control plane."""

from __future__ import annotations

import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from sqlalchemy import delete
from sqlalchemy import func, select


pytestmark = pytest.mark.integration


def _postgres_integration_enabled() -> bool:
    return os.getenv("DOCUSCOPE_POSTGRES_INTEGRATION", "").strip() == "1"


@pytest.mark.skipif(
    not _postgres_integration_enabled(),
    reason="Set DOCUSCOPE_POSTGRES_INTEGRATION=1 to run real Postgres smoke tests.",
)
def test_postgres_control_plane_session_and_registry_smoke(tmp_path, monkeypatch):
    """Exercise schema, session envelopes, artifact reservation, and job completion."""

    from webapp.persistence.database import (
        create_session_factory,
        build_engine,
        initialize_database_schema,
    )
    from webapp.persistence.models import ArtifactJob, ArtifactRecord, SessionRecord
    from webapp.persistence.registry import ArtifactIdentity, ArtifactRegistryService
    from webapp.utilities.storage.postgres_session_backend import PostgresSessionBackend

    build_engine.cache_clear()
    create_session_factory.cache_clear()
    initialize_database_schema()

    run_id = uuid.uuid4().hex
    session_id = f"pg-smoke-{run_id}"
    selector_hash = f"selector-{run_id}"
    parameter_hash = "params"
    artifact_dir: Path | None = None
    session_artifact_root = tmp_path / "session-corpora"
    session_artifact_dir = session_artifact_root / run_id / "target"
    session_artifact_dir.mkdir(parents=True, exist_ok=True)
    session_artifact_path = session_artifact_dir / "ds_tokens.gz"
    session_sidecar_path = session_artifact_dir / "metadata_descriptor.json"
    session_artifact_path.write_bytes(b"payload")
    session_sidecar_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("DOCUSCOPE_SESSION_ARTIFACT_ROOT", str(session_artifact_root))

    session_factory = create_session_factory()

    try:
        backend = PostgresSessionBackend()
        session_payload = {
            "session": {
                "has_target": True,
                "target_db": f"target-{run_id}",
            },
            "target": {
                "_artifact_refs": {
                    "ds_tokens": {
                        "storage_type": "gzip_pickle",
                        "path": str(session_artifact_path),
                    }
                }
            },
        }
        assert backend.save_session(
            session_id,
            session_payload,
            user_id="pg-smoke@example.test",
        ) is True
        assert backend.load_session(session_id) == session_payload
        assert backend.delete_session(session_id) is True
        assert not session_artifact_path.exists()
        assert not session_sidecar_path.exists()
        assert backend.load_session(session_id) is None

        registry = ArtifactRegistryService()
        identity = ArtifactIdentity(
            artifact_type="pg_smoke_json",
            scope="private",
            owner_principal_id=f"owner-{run_id}",
            selector_hash=selector_hash,
            selector_payload={"run_id": run_id},
            parameter_hash=parameter_hash,
            parameter_payload={"kind": "postgres-smoke"},
            pipeline_version="test",
            model_version="test",
        )

        reservation = registry.reserve_artifact(identity)
        assert reservation.state == "reserved"
        assert reservation.job is not None

        artifact = registry.store_json_artifact(
            identity,
            {"ok": True, "run_id": run_id},
        )
        artifact_dir = Path(artifact.storage_uri)
        registry.mark_job_completed(reservation.job.job_id, artifact.artifact_id)

        job = registry.get_job_by_id_internal(reservation.job.job_id)
        assert job is not None
        assert job.status == "completed"
        assert job.artifact_id == artifact.artifact_id

        assert registry.load_json_artifact(artifact) == {"ok": True, "run_id": run_id}

        ready_artifact = registry.find_ready_artifact(identity)
        assert ready_artifact is not None
        assert ready_artifact.artifact_id == artifact.artifact_id

    finally:
        with session_factory() as session:
            session.execute(
                delete(ArtifactJob).where(ArtifactJob.selector_hash == selector_hash)
            )
            session.execute(
                delete(ArtifactRecord).where(
                    ArtifactRecord.selector_hash == selector_hash
                )
            )
            session.execute(
                delete(SessionRecord).where(SessionRecord.session_id == session_id)
            )
            session.commit()

        if artifact_dir is not None:
            shutil.rmtree(artifact_dir, ignore_errors=True)

        build_engine.cache_clear()
        create_session_factory.cache_clear()


@pytest.mark.skipif(
    not _postgres_integration_enabled(),
    reason="Set DOCUSCOPE_POSTGRES_INTEGRATION=1 to run real Postgres smoke tests.",
)
def test_concurrent_public_reservations_share_one_artifact_and_job():
    """Prove PostgreSQL uniqueness serializes a cold public reservation race."""

    from webapp.persistence.database import create_session_factory, build_engine
    from webapp.persistence.models import ArtifactJob, ArtifactRecord
    from webapp.persistence.registry import ArtifactIdentity, ArtifactRegistryService

    build_engine.cache_clear()
    create_session_factory.cache_clear()
    run_id = uuid.uuid4().hex
    selector_hash = f"concurrent-selector-{run_id}"
    identity = ArtifactIdentity(
        artifact_type="concurrent_public_smoke",
        scope="public",
        selector_hash=selector_hash,
        selector_payload={"run_id": run_id},
        parameter_hash="params",
        parameter_payload={"kind": "concurrent-postgres-smoke"},
        pipeline_version="test",
        model_version="test",
    )
    session_factory = create_session_factory()

    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(
                executor.map(
                    lambda _request: ArtifactRegistryService().reserve_artifact(
                        identity
                    ),
                    range(8),
                )
            )

        artifact_ids = {
            result.artifact.artifact_id
            for result in results
            if result.artifact is not None
        }
        job_ids = {
            result.job.job_id
            for result in results
            if result.job is not None
        }
        assert artifact_ids and len(artifact_ids) == 1
        assert job_ids and len(job_ids) == 1
        assert sum(result.state == "reserved" for result in results) == 1
        assert all(result.state in {"reserved", "pending"} for result in results)

        with session_factory() as session:
            artifact_count = session.scalar(
                select(func.count()).select_from(ArtifactRecord).where(
                    ArtifactRecord.selector_hash == selector_hash
                )
            )
            active_job_count = session.scalar(
                select(func.count()).select_from(ArtifactJob).where(
                    ArtifactJob.selector_hash == selector_hash,
                    ArtifactJob.status.in_(("pending", "running")),
                )
            )
        assert artifact_count == 1
        assert active_job_count == 1
    finally:
        with session_factory() as session:
            session.execute(
                delete(ArtifactJob).where(ArtifactJob.selector_hash == selector_hash)
            )
            session.execute(
                delete(ArtifactRecord).where(
                    ArtifactRecord.selector_hash == selector_hash
                )
            )
            session.commit()
        build_engine.cache_clear()
        create_session_factory.cache_clear()
