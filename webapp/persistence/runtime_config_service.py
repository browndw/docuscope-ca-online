"""Postgres-backed runtime configuration service."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, select

from webapp.persistence.database import create_session_factory, initialize_database_schema
from webapp.persistence.models import ConfigAuditLog, RuntimeConfigRecord


class RuntimeConfigService:
    """Runtime configuration override operations backed by the control plane."""

    def __init__(self) -> None:
        initialize_database_schema()
        self._session_factory = create_session_factory()

    def get_value(self, config_key: str) -> Any | None:
        """Return a typed runtime override value."""

        with self._session_factory() as session:
            record = session.get(RuntimeConfigRecord, config_key)
            if record is None:
                return None
            return self._deserialize_value(record.config_value, record.config_type)

    def set_value(
        self,
        config_key: str,
        value: Any,
        updated_by: str | None = None,
        description: str | None = None,
        instance_id: str | None = None,
    ) -> Any | None:
        """Create or update a runtime override and audit the change."""

        value_str, value_type = self._serialize_value(value)
        now = datetime.now(timezone.utc)

        with self._session_factory() as session:
            record = session.get(RuntimeConfigRecord, config_key)
            old_value = None
            if record is None:
                record = RuntimeConfigRecord(config_key=config_key)
                session.add(record)
            else:
                old_value = self._deserialize_value(record.config_value, record.config_type)

            record.config_value = value_str
            record.config_type = value_type
            record.updated_at = now
            record.updated_by = updated_by
            record.description = description

            session.add(
                ConfigAuditLog(
                    config_key=config_key,
                    old_value=str(old_value) if old_value is not None else None,
                    new_value=value_str,
                    updated_at=now,
                    updated_by=updated_by,
                    instance_id=instance_id,
                )
            )
            session.commit()
            return old_value

    def clear_value(
        self,
        config_key: str,
        updated_by: str | None = None,
        instance_id: str | None = None,
    ) -> Any | None:
        """Remove a runtime override and audit the clear operation."""

        now = datetime.now(timezone.utc)

        with self._session_factory() as session:
            record = session.get(RuntimeConfigRecord, config_key)
            old_value = None
            if record is not None:
                old_value = self._deserialize_value(record.config_value, record.config_type)
                session.execute(
                    delete(RuntimeConfigRecord).where(
                        RuntimeConfigRecord.config_key == config_key,
                    )
                )

            session.add(
                ConfigAuditLog(
                    config_key=config_key,
                    old_value=str(old_value) if old_value is not None else None,
                    new_value="CLEARED",
                    updated_at=now,
                    updated_by=updated_by,
                    instance_id=instance_id,
                )
            )
            session.commit()
            return old_value

    def list_overrides(self) -> dict[str, dict[str, Any]]:
        """Return all runtime overrides keyed by config key."""

        with self._session_factory() as session:
            records = session.execute(
                select(RuntimeConfigRecord).order_by(RuntimeConfigRecord.updated_at.desc())
            ).scalars()
            return {
                record.config_key: {
                    "value": self._deserialize_value(
                        record.config_value,
                        record.config_type,
                    ),
                    "updated_at": record.updated_at,
                    "updated_by": record.updated_by,
                    "description": record.description,
                }
                for record in records
            }

    def get_audit_log(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent runtime configuration audit events."""

        with self._session_factory() as session:
            rows = session.execute(
                select(ConfigAuditLog)
                .order_by(ConfigAuditLog.updated_at.desc())
                .limit(limit)
            ).scalars()
            return [
                {
                    "config_key": row.config_key,
                    "old_value": row.old_value,
                    "new_value": row.new_value,
                    "updated_at": row.updated_at,
                    "updated_by": row.updated_by,
                    "instance_id": row.instance_id,
                }
                for row in rows
            ]

    @staticmethod
    def _serialize_value(value: Any) -> tuple[str, str]:
        if isinstance(value, bool):
            return str(value), "bool"
        if isinstance(value, int):
            return str(value), "int"
        if isinstance(value, float):
            return str(value), "float"
        return str(value), "str"

    @staticmethod
    def _deserialize_value(value_str: str, value_type: str) -> Any:
        if value_type == "bool":
            return value_str.lower() == "true"
        if value_type == "int":
            return int(value_str)
        if value_type == "float":
            return float(value_str)
        return value_str