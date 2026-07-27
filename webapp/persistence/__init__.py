"""Persistence foundation for the refactored control plane."""

from webapp.persistence.config import DatabaseConfig, get_database_config
from webapp.persistence.database import (
    Base,
    build_engine,
    create_session_factory,
    initialize_database_schema,
)
from webapp.persistence.models import (
    AccessLog,
    ArtifactJob,
    ArtifactRecord,
    AuthorizedUser,
    ConfigAuditLog,
    RuntimeConfigRecord,
    UserRole,
)
from webapp.persistence.registry import (
    ArtifactIdentity,
    ArtifactRegistryService,
    ReservationResult,
    build_shared_collocation_identity,
    build_shared_frequency_identity,
    build_shared_keyness_identity,
    build_shared_keyness_parts_identity,
    build_shared_ngram_identity,
    registry_service,
)
from webapp.persistence.workflows import SharedArtifactDecision, SharedArtifactWorkflow

__all__ = [
    "ArtifactJob",
    "ArtifactIdentity",
    "ArtifactRecord",
    "AccessLog",
    "AuthorizedUser",
    "ConfigAuditLog",
    "ArtifactRegistryService",
    "Base",
    "DatabaseConfig",
    "ReservationResult",
    "SharedArtifactDecision",
    "SharedArtifactWorkflow",
    "RuntimeConfigRecord",
    "UserRole",
    "build_shared_collocation_identity",
    "build_shared_frequency_identity",
    "build_shared_keyness_identity",
    "build_shared_keyness_parts_identity",
    "build_shared_ngram_identity",
    "build_engine",
    "create_session_factory",
    "get_database_config",
    "initialize_database_schema",
    "registry_service",
]
