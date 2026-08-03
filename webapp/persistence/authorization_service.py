"""Postgres-backed authorization service."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select

from webapp.persistence.database import create_session_factory, initialize_database_schema
from webapp.persistence.models import AccessLog, AuthorizedUser, UserRole
from webapp.utilities.configuration.logging_config import get_logger

logger = get_logger()


class AuthorizationService:
    """Role, user, and access-log operations backed by the control plane."""

    def __init__(self) -> None:
        initialize_database_schema()
        self._session_factory = create_session_factory()

    def initialize_defaults(
        self,
        default_roles: dict[str, dict[str, Any]],
        default_admin: str | None,
    ) -> None:
        """Create default roles and the first admin when needed."""

        with self._session_factory() as session:
            for role_name, role_data in default_roles.items():
                role = session.get(UserRole, role_name)
                if role is None:
                    role = UserRole(role_name=role_name)
                    session.add(role)
                role.description = role_data["description"]
                role.permissions = list(role_data["permissions"])

            session.flush()

            if default_admin:
                admin_count = session.execute(
                    select(func.count()).select_from(AuthorizedUser).where(
                        AuthorizedUser.role == "admin",
                        AuthorizedUser.active.is_(True),
                    )
                ).scalar_one()
                if admin_count == 0:
                    admin_email = self._normalize_email(default_admin)
                    admin = session.get(AuthorizedUser, admin_email)
                    if admin is None:
                        admin = AuthorizedUser(email=admin_email)
                        session.add(admin)
                    admin.role = "admin"
                    admin.added_by = "system"
                    admin.active = True

            session.commit()

    def get_user_role(self, email: str) -> str | None:
        """Return the active role for an email address."""

        with self._session_factory() as session:
            user = session.get(AuthorizedUser, self._normalize_email(email))
            if user is None or not user.active:
                return None
            return user.role

    def get_user_permissions(self, email: str) -> list[str]:
        """Return combined role and user-specific permissions."""

        email = self._normalize_email(email)
        with self._session_factory() as session:
            user = session.get(AuthorizedUser, email)
            if user is None or not user.active:
                return []

            role = session.get(UserRole, user.role)
            role_permissions = role.permissions if role is not None else []
            user_permissions = user.permissions or []
            return sorted(set(role_permissions + user_permissions))

    def add_authorized_user(
        self,
        email: str,
        role: str,
        added_by: str | None,
        permissions: list[str] | None,
    ) -> bool:
        """Add or reactivate an authorized user."""

        email = self._normalize_email(email)
        with self._session_factory() as session:
            if session.get(UserRole, role) is None:
                return False
            user = session.get(AuthorizedUser, email)
            if user is None:
                user = AuthorizedUser(email=email)
                session.add(user)
            user.role = role
            user.permissions = permissions
            user.added_by = added_by
            user.added_at = datetime.now(timezone.utc)
            user.active = True
            session.commit()
            return True

    def remove_authorized_user(self, email: str) -> bool:
        """Deactivate an authorized user."""

        with self._session_factory() as session:
            user = session.get(AuthorizedUser, self._normalize_email(email))
            if user is None:
                return False
            user.active = False
            session.commit()
            return True

    def update_user_role(
        self,
        email: str,
        new_role: str,
        updated_by: str | None,
    ) -> bool:
        """Update the role for an active user."""

        with self._session_factory() as session:
            if session.get(UserRole, new_role) is None:
                return False
            user = session.get(AuthorizedUser, self._normalize_email(email))
            if user is None or not user.active:
                return False
            user.role = new_role
            user.added_by = updated_by
            user.added_at = datetime.now(timezone.utc)
            session.commit()
            return True

    def list_authorized_users(self) -> list[dict[str, Any]]:
        """List active authorized users ordered by creation time."""

        with self._session_factory() as session:
            rows = session.execute(
                select(AuthorizedUser)
                .where(AuthorizedUser.active.is_(True))
                .order_by(AuthorizedUser.added_at.desc())
            ).scalars()
            return [
                {
                    "email": user.email,
                    "role": user.role,
                    "added_by": user.added_by,
                    "added_at": user.added_at,
                    "last_accessed": user.last_accessed,
                    "active": user.active,
                }
                for user in rows
            ]

    def update_last_accessed(self, email: str) -> None:
        """Record the latest access timestamp for an active user."""

        with self._session_factory() as session:
            user = session.get(AuthorizedUser, self._normalize_email(email))
            if user is None or not user.active:
                return
            user.last_accessed = datetime.now(timezone.utc)
            session.commit()

    def log_access_attempt(
        self,
        email: str,
        page: str | None,
        required_role: str | None,
        required_permission: str | None,
        granted: bool,
    ) -> None:
        """Persist an authorization audit event."""

        with self._session_factory() as session:
            session.add(
                AccessLog(
                    email=self._normalize_email(email),
                    page=page,
                    required_role=required_role,
                    required_permission=required_permission,
                    access_granted=granted,
                )
            )
            session.commit()

    @staticmethod
    def _normalize_email(email: str) -> str:
        return email.lower().strip() if email else ""
