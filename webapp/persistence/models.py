"""Initial control-plane models for artifacts and jobs."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from webapp.persistence.database import Base


class ArtifactRecord(Base):
    """Registry entry for a stored analytical artifact."""

    __tablename__ = "artifacts"
    __table_args__ = (
        UniqueConstraint(
            "artifact_type",
            "scope",
            "selector_hash",
            "parameter_hash",
            "pipeline_version",
            "model_version",
            "owner_principal_id",
            name="uq_artifact_identity",
        ),
        Index("ix_artifacts_scope_status", "scope", "status"),
    )

    artifact_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    scope: Mapped[str] = mapped_column(String(16), nullable=False)
    owner_principal_id: Mapped[str] = mapped_column(String(255), nullable=False)
    sharing_principal_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    selector_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    selector_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    pipeline_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    parameter_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    parameter_payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    storage_uri: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )
    last_accessed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    access_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class ArtifactJob(Base):
    """Job entry for generating or refreshing an artifact."""

    __tablename__ = "artifact_jobs"
    __table_args__ = (Index("ix_artifact_jobs_status_created", "status", "created_at"),)

    job_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    artifact_id: Mapped[int | None] = mapped_column(
        ForeignKey("artifacts.artifact_id"),
        nullable=True,
    )
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    scope: Mapped[str] = mapped_column(String(16), nullable=False)
    requester_principal_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    selector_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    selector_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    parameter_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    parameter_payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    pipeline_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    worker_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class SessionRecord(Base):
    """Durable lightweight session envelope stored in the control plane."""

    __tablename__ = "sessions"
    __table_args__ = (
        Index("ix_sessions_expires", "expires_at"),
        Index("ix_sessions_user", "user_id"),
        Index("ix_sessions_updated", "updated_at"),
    )

    session_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    access_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class UserQuery(Base):
    """Quota and assistant-query accounting row."""

    __tablename__ = "user_queries"
    __table_args__ = (
        Index("ix_queries_user_time", "user_id", "query_timestamp"),
        Index("ix_queries_session", "session_id"),
    )

    query_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    session_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    query_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )
    assistant_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    message_content: Mapped[str | None] = mapped_column(Text, nullable=True)


class UserRole(Base):
    """Role definition and permission set for authorization."""

    __tablename__ = "user_roles"

    role_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    permissions: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )


class AuthorizedUser(Base):
    """User authorization and role assignment."""

    __tablename__ = "authorized_users"
    __table_args__ = (
        Index("ix_authorized_users_role_active", "role", "active"),
    )

    email: Mapped[str] = mapped_column(String(255), primary_key=True)
    role: Mapped[str] = mapped_column(
        ForeignKey("user_roles.role_name"),
        nullable=False,
        default="user",
    )
    permissions: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    added_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )
    last_accessed: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class AccessLog(Base):
    """Authorization access attempt audit log."""

    __tablename__ = "access_log"
    __table_args__ = (
        Index("ix_access_log_email_timestamp", "email", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    page: Mapped[str | None] = mapped_column(String(255), nullable=True)
    required_role: Mapped[str | None] = mapped_column(String(64), nullable=True)
    required_permission: Mapped[str | None] = mapped_column(String(128), nullable=True)
    access_granted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)


class RuntimeConfigRecord(Base):
    """Runtime configuration override stored in the control plane."""

    __tablename__ = "runtime_config"
    __table_args__ = (
        Index("ix_runtime_config_updated", "updated_at"),
    )

    config_key: Mapped[str] = mapped_column(String(255), primary_key=True)
    config_value: Mapped[str] = mapped_column(Text, nullable=False)
    config_type: Mapped[str] = mapped_column(String(32), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )
    updated_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)


class ConfigAuditLog(Base):
    """Audit log for runtime configuration changes."""

    __tablename__ = "config_audit_log"
    __table_args__ = (
        Index("ix_config_audit_updated", "updated_at"),
        Index("ix_config_audit_key_updated", "config_key", "updated_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    config_key: Mapped[str] = mapped_column(String(255), nullable=False)
    old_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    new_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
    )
    updated_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    instance_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
