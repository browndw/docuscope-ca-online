"""Fix public artifact uniqueness.

Revision ID: 20260805_01
Revises: 20260608_0001
Create Date: 2026-08-05
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260805_01"
down_revision: Union[str, None] = "20260608_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

PUBLIC_OWNER_PRINCIPAL_ID = "__public__"
IDENTITY_COLUMNS = (
    "artifact_type",
    "scope",
    "selector_hash",
    "parameter_hash",
    "pipeline_version",
    "model_version",
    "owner_principal_id",
)


def _artifact_table() -> sa.TableClause:
    return sa.table(
        "artifacts",
        sa.column("artifact_id", sa.Integer()),
        sa.column("artifact_type", sa.String()),
        sa.column("scope", sa.String()),
        sa.column("selector_hash", sa.String()),
        sa.column("parameter_hash", sa.String()),
        sa.column("pipeline_version", sa.String()),
        sa.column("model_version", sa.String()),
        sa.column("owner_principal_id", sa.String()),
        sa.column("status", sa.String()),
    )


def _artifact_job_table() -> sa.TableClause:
    return sa.table(
        "artifact_jobs",
        sa.column("artifact_id", sa.Integer()),
    )


def _deduplicate_public_artifacts(connection: sa.Connection) -> None:
    artifacts = _artifact_table()
    artifact_jobs = _artifact_job_table()
    rows = connection.execute(sa.select(artifacts)).mappings().all()
    grouped_rows: dict[tuple[object, ...], list[dict]] = defaultdict(list)

    for row in rows:
        if row["scope"] != "public":
            continue
        identity_key = (
            row["artifact_type"],
            row["scope"],
            row["selector_hash"],
            row["parameter_hash"],
            row["pipeline_version"],
            row["model_version"],
            PUBLIC_OWNER_PRINCIPAL_ID,
        )
        grouped_rows[identity_key].append(row)

    for duplicate_rows in grouped_rows.values():
        if len(duplicate_rows) < 2:
            continue
        ordered_rows = sorted(
            duplicate_rows,
            key=lambda row: (
                row["status"] != "ready",
                row["artifact_id"],
            ),
        )
        keeper_id = ordered_rows[0]["artifact_id"]
        duplicate_ids = [row["artifact_id"] for row in ordered_rows[1:]]
        connection.execute(
            sa.update(artifact_jobs)
            .where(artifact_jobs.c.artifact_id.in_(duplicate_ids))
            .values(artifact_id=keeper_id)
        )
        connection.execute(
            sa.delete(artifacts).where(artifacts.c.artifact_id.in_(duplicate_ids))
        )


def upgrade() -> None:
    connection = op.get_bind()
    artifacts = _artifact_table()

    with op.batch_alter_table("artifacts") as batch_op:
        batch_op.drop_constraint("uq_artifact_identity", type_="unique")

    _deduplicate_public_artifacts(connection)
    connection.execute(
        sa.update(artifacts)
        .where(artifacts.c.scope == "public")
        .values(owner_principal_id=PUBLIC_OWNER_PRINCIPAL_ID)
    )

    legacy_private_rows = connection.execute(
        sa.select(artifacts.c.artifact_id).where(
            artifacts.c.owner_principal_id.is_(None)
        )
    ).scalars()
    for artifact_id in legacy_private_rows:
        connection.execute(
            sa.update(artifacts)
            .where(artifacts.c.artifact_id == artifact_id)
            .values(owner_principal_id=f"__legacy_private__:{artifact_id}")
        )

    with op.batch_alter_table("artifacts") as batch_op:
        batch_op.alter_column(
            "owner_principal_id",
            existing_type=sa.String(length=255),
            nullable=False,
        )
        batch_op.create_unique_constraint(
            "uq_artifact_identity",
            IDENTITY_COLUMNS,
        )


def downgrade() -> None:
    connection = op.get_bind()
    artifacts = _artifact_table()

    with op.batch_alter_table("artifacts") as batch_op:
        batch_op.drop_constraint("uq_artifact_identity", type_="unique")
        batch_op.alter_column(
            "owner_principal_id",
            existing_type=sa.String(length=255),
            nullable=True,
        )

    connection.execute(
        sa.update(artifacts)
        .where(
            artifacts.c.scope == "public",
            artifacts.c.owner_principal_id == PUBLIC_OWNER_PRINCIPAL_ID,
        )
        .values(owner_principal_id=None)
    )

    with op.batch_alter_table("artifacts") as batch_op:
        batch_op.create_unique_constraint(
            "uq_artifact_identity",
            IDENTITY_COLUMNS,
        )
