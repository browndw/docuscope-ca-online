"""Integration tests for control-plane Alembic migrations."""

from __future__ import annotations

from alembic import command
from alembic.config import Config
import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError


pytestmark = pytest.mark.integration
BASELINE_REVISION = "20260608_0001"
HEAD_REVISION = "20260805_01"


def _alembic_config(database_url: str) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def _create_legacy_artifact_schema(engine) -> None:
    metadata = sa.MetaData()
    artifacts = sa.Table(
        "artifacts",
        metadata,
        sa.Column(
            "artifact_id",
            sa.Integer,
            primary_key=True,
            autoincrement=True,
        ),
        sa.Column("artifact_type", sa.String(64), nullable=False),
        sa.Column("scope", sa.String(16), nullable=False),
        sa.Column("owner_principal_id", sa.String(255), nullable=True),
        sa.Column("selector_hash", sa.String(128), nullable=False),
        sa.Column("parameter_hash", sa.String(128), nullable=False),
        sa.Column("pipeline_version", sa.String(64), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.UniqueConstraint(
            "artifact_type",
            "scope",
            "selector_hash",
            "parameter_hash",
            "pipeline_version",
            "model_version",
            "owner_principal_id",
            name="uq_artifact_identity",
        ),
    )
    artifact_jobs = sa.Table(
        "artifact_jobs",
        metadata,
        sa.Column(
            "job_id",
            sa.Integer,
            primary_key=True,
            autoincrement=True,
        ),
        sa.Column("artifact_id", sa.Integer, nullable=True),
    )
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(artifacts.insert(), [
            {
                "artifact_type": "frequency_bundle",
                "scope": "public",
                "owner_principal_id": None,
                "selector_hash": "selector",
                "parameter_hash": "params",
                "pipeline_version": "pipeline",
                "model_version": "model",
                "status": "pending",
            },
            {
                "artifact_type": "frequency_bundle",
                "scope": "public",
                "owner_principal_id": None,
                "selector_hash": "selector",
                "parameter_hash": "params",
                "pipeline_version": "pipeline",
                "model_version": "model",
                "status": "ready",
            },
            {
                "artifact_type": "legacy_private",
                "scope": "private",
                "owner_principal_id": None,
                "selector_hash": "private-selector",
                "parameter_hash": "params",
                "pipeline_version": "pipeline",
                "model_version": "model",
                "status": "ready",
            },
        ])
        connection.execute(artifact_jobs.insert(), [
            {"artifact_id": 1},
            {"artifact_id": 2},
        ])


def test_public_owner_migration_consolidates_legacy_duplicates(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'migration.db'}"
    engine = create_engine(database_url)
    _create_legacy_artifact_schema(engine)
    config = _alembic_config(database_url)

    command.stamp(config, BASELINE_REVISION)
    command.upgrade(config, "head")

    with engine.connect() as connection:
        public_rows = connection.execute(text("""
            SELECT artifact_id, owner_principal_id, status
            FROM artifacts
            WHERE scope = 'public'
        """)).mappings().all()
        assert public_rows == [{
            "artifact_id": 2,
            "owner_principal_id": "__public__",
            "status": "ready",
        }]

        job_artifact_ids = connection.execute(text("""
            SELECT artifact_id FROM artifact_jobs ORDER BY job_id
        """)).scalars().all()
        assert job_artifact_ids == [2, 2]

        private_owner = connection.execute(text("""
            SELECT owner_principal_id FROM artifacts WHERE scope = 'private'
        """)).scalar_one()
        assert private_owner == "__legacy_private__:3"

        revision = connection.execute(text("""
            SELECT version_num FROM alembic_version
        """)).scalar_one()
        assert revision == HEAD_REVISION

    owner_column = next(
        column
        for column in inspect(engine).get_columns("artifacts")
        if column["name"] == "owner_principal_id"
    )
    assert owner_column["nullable"] is False

    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(text("""
            INSERT INTO artifacts (
                artifact_type, scope, owner_principal_id, selector_hash,
                parameter_hash, pipeline_version, model_version, status
            ) VALUES (
                'frequency_bundle', 'public', '__public__', 'selector', 'params',
                'pipeline', 'model', 'ready'
            )
        """))

    engine.dispose()
