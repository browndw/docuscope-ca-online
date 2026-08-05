"""Unit tests for bootstrap administrator provisioning."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from webapp.persistence.authorization_service import AuthorizationService
from webapp.persistence.database import build_engine, create_session_factory
from webapp.persistence.models import AuthorizedUser


@pytest.fixture
def authorization_service(tmp_path, monkeypatch):
    database_path = tmp_path / "authorization.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path}")
    build_engine.cache_clear()
    create_session_factory.cache_clear()
    yield AuthorizationService()
    build_engine.cache_clear()
    create_session_factory.cache_clear()


def _roles() -> dict[str, dict[str, object]]:
    return {
        "admin": {
            "description": "Administrator",
            "permissions": ["user_management"],
        },
        "user": {
            "description": "User",
            "permissions": ["corpus_analysis"],
        },
    }


def test_initialize_defaults_records_normalized_bootstrap_provenance(
    authorization_service,
):
    authorization_service.initialize_defaults(
        _roles(),
        "  ADMIN@Example.Test  ",
    )

    with authorization_service._session_factory() as session:
        administrator = session.get(AuthorizedUser, "admin@example.test")

    assert administrator is not None
    assert administrator.role == "admin"
    assert administrator.added_by == "system"
    assert administrator.added_at is not None
    assert administrator.active is True


def test_initialize_defaults_does_not_replace_existing_active_admin(
    authorization_service,
):
    authorization_service.initialize_defaults(
        _roles(),
        "first-admin@example.test",
    )
    authorization_service.initialize_defaults(
        _roles(),
        "second-admin@example.test",
    )

    with authorization_service._session_factory() as session:
        administrators = session.scalars(
            select(AuthorizedUser).where(
                AuthorizedUser.role == "admin",
                AuthorizedUser.active.is_(True),
            )
        ).all()

    assert [administrator.email for administrator in administrators] == [
        "first-admin@example.test"
    ]
