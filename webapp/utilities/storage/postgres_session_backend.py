"""Postgres-backed session storage for the shared control plane."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pickle
from typing import Any, Dict, Optional

from sqlalchemy import delete, func, select

from webapp.persistence.database import create_session_factory, initialize_database_schema
from webapp.persistence.models import SessionRecord, UserQuery
from webapp.utilities.configuration.logging_config import get_logger


logger = get_logger()


class PostgresSessionBackend:
    """Session backend backed by the SQLAlchemy/Postgres control plane."""

    def __init__(self) -> None:
        initialize_database_schema()
        self._session_factory = create_session_factory()

    def _session_artifact_root(self) -> Path:
        """Return the root directory used for session-scoped corpus artifacts."""

        return Path("webapp/_session") / "corpora"

    def _collect_session_artifact_paths(self, data: Dict[str, Any]) -> list[Path]:
        """Collect session-scoped file-backed artifact paths from persisted data."""

        artifact_paths: list[Path] = []
        artifact_root = self._session_artifact_root()

        for corpus_key in ("target", "reference"):
            corpus_state = data.get(corpus_key)
            if not isinstance(corpus_state, dict):
                continue

            refs = corpus_state.get("_artifact_refs")
            if not isinstance(refs, dict):
                continue

            for ref in refs.values():
                if not isinstance(ref, dict):
                    continue
                if ref.get("storage_type") != "gzip_pickle":
                    continue

                path_value = ref.get("path")
                if not path_value:
                    continue

                artifact_path = Path(path_value)
                try:
                    resolved_path = artifact_path.resolve(strict=False)
                    resolved_root = artifact_root.resolve(strict=False)
                    resolved_path.relative_to(resolved_root)
                except Exception:
                    continue

                artifact_paths.append(artifact_path)

        deduped_paths: list[Path] = []
        seen_paths: set[Path] = set()
        for path in artifact_paths:
            if path in seen_paths:
                continue
            seen_paths.add(path)
            deduped_paths.append(path)
        return deduped_paths

    def _cleanup_session_artifacts(self, artifact_paths: list[Path]) -> None:
        """Remove session-scoped artifact files and empty parent directories."""

        artifact_root = self._session_artifact_root()

        for artifact_path in artifact_paths:
            try:
                if artifact_path.exists():
                    artifact_path.unlink()

                sidecar_path = artifact_path.with_name("metadata_descriptor.json")
                if sidecar_path.exists():
                    sidecar_path.unlink()

                parent = artifact_path.parent
                resolved_root = artifact_root.resolve(strict=False)
                while parent.exists() and parent != artifact_root:
                    try:
                        parent.resolve(strict=False).relative_to(resolved_root)
                    except Exception:
                        break

                    try:
                        parent.rmdir()
                    except OSError:
                        break
                    parent = parent.parent
            except Exception as exc:
                logger.warning(
                    f"Failed to clean session artifact {artifact_path}: {exc}"
                )

    def save_session(
        self,
        session_id: str,
        data: Dict[str, Any],
        user_id: str = None,
    ) -> bool:
        """Save a lightweight session envelope to the control plane."""

        try:
            from webapp.utilities.storage.cache_management import persistent_hash

            serialized_data = pickle.dumps(data)
            stored_user_id = persistent_hash(user_id) if user_id else None
            expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
            now = datetime.now(timezone.utc)

            with self._session_factory() as session:
                record = session.get(SessionRecord, session_id)
                if record is None:
                    record = SessionRecord(
                        session_id=session_id,
                        user_id=stored_user_id,
                        data=serialized_data,
                        created_at=now,
                        updated_at=now,
                        expires_at=expires_at,
                        size_bytes=len(serialized_data),
                        access_count=1,
                    )
                    session.add(record)
                else:
                    record.user_id = stored_user_id
                    record.data = serialized_data
                    record.updated_at = now
                    record.expires_at = expires_at
                    record.size_bytes = len(serialized_data)
                    record.access_count += 1

                session.commit()
                return True

        except Exception as exc:
            logger.error(f"Failed to save session {session_id}: {exc}")
            return False

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a session envelope from the control plane."""

        try:
            now = datetime.now(timezone.utc)
            with self._session_factory() as session:
                record = session.get(SessionRecord, session_id)
                if record is None:
                    return None
                if record.expires_at is not None and record.expires_at <= now:
                    return None

                data = pickle.loads(record.data)
                record.access_count += 1
                record.updated_at = now
                session.commit()
                return data

        except Exception as exc:
            logger.error(f"Failed to load session {session_id}: {exc}")
            return None

    def delete_session(self, session_id: str, user_id: str = None) -> bool:
        """Delete a session envelope and associated query rows."""

        try:
            artifact_paths: list[Path] = []
            with self._session_factory() as session:
                record = session.get(SessionRecord, session_id)
                if record is not None:
                    artifact_paths = self._collect_session_artifact_paths(
                        pickle.loads(record.data)
                    )
                    session.delete(record)

                session.execute(
                    delete(UserQuery).where(UserQuery.session_id == session_id)
                )
                session.commit()

            self._cleanup_session_artifacts(artifact_paths)
            return True

        except Exception as exc:
            logger.error(f"Failed to delete session {session_id}: {exc}")
            return False

    def get_user_query_count_24h(self, user_id: str) -> int:
        """Return the user's assistant query count in the previous 24 hours."""

        try:
            from webapp.utilities.storage.cache_management import persistent_hash

            hashed_user_id = persistent_hash(user_id)
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

            with self._session_factory() as session:
                stmt = select(func.count()).select_from(UserQuery).where(
                    UserQuery.user_id == hashed_user_id,
                    UserQuery.query_timestamp >= cutoff,
                )
                return int(session.execute(stmt).scalar_one())

        except Exception as exc:
            logger.error(f"Failed to get query count for user {user_id}: {exc}")
            return 0

    def log_user_query(
        self,
        user_id: str,
        session_id: str,
        assistant_type: str = None,
        message_content: str = None,
    ) -> bool:
        """Log an assistant query for quota and usage accounting."""

        try:
            from webapp.utilities.storage.cache_management import persistent_hash

            query = UserQuery(
                user_id=persistent_hash(user_id),
                session_id=session_id,
                query_timestamp=datetime.now(timezone.utc),
                assistant_type=assistant_type,
                message_content=message_content,
            )
            with self._session_factory() as session:
                session.add(query)
                session.commit()
            return True

        except Exception as exc:
            logger.error(f"Failed to log user query: {exc}")
            return False

    def cleanup_expired_sessions(self) -> int:
        """Clean expired sessions and old query logs."""

        try:
            now = datetime.now(timezone.utc)
            old_query_cutoff = now - timedelta(days=7)
            artifact_paths: list[Path] = []

            with self._session_factory() as session:
                expired_records = session.execute(
                    select(SessionRecord).where(SessionRecord.expires_at <= now)
                ).scalars().all()
                expired_session_ids = [record.session_id for record in expired_records]
                for record in expired_records:
                    artifact_paths.extend(
                        self._collect_session_artifact_paths(pickle.loads(record.data))
                    )
                    session.delete(record)

                if expired_session_ids:
                    session.execute(
                        delete(UserQuery).where(
                            UserQuery.session_id.in_(expired_session_ids)
                        )
                    )
                session.execute(
                    delete(UserQuery).where(UserQuery.query_timestamp < old_query_cutoff)
                )
                session.commit()

            self._cleanup_session_artifacts(artifact_paths)
            return len(expired_session_ids)

        except Exception as exc:
            logger.error(f"Failed to cleanup expired sessions: {exc}")
            return 0

    def get_session_stats(self) -> Dict[str, Any]:
        """Return control-plane session and query statistics."""

        try:
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(hours=24)
            with self._session_factory() as session:
                active_sessions = session.execute(
                    select(func.count()).select_from(SessionRecord).where(
                        SessionRecord.expires_at > now
                    )
                ).scalar_one()
                total_sessions = session.execute(
                    select(func.count()).select_from(SessionRecord)
                ).scalar_one()
                avg_size, max_size = session.execute(
                    select(
                        func.avg(SessionRecord.size_bytes),
                        func.max(SessionRecord.size_bytes),
                    )
                ).one()
                queries_24h = session.execute(
                    select(func.count()).select_from(UserQuery).where(
                        UserQuery.query_timestamp >= cutoff
                    )
                ).scalar_one()

            return {
                "active_sessions": int(active_sessions),
                "total_sessions": int(total_sessions),
                "avg_session_size_bytes": float(avg_size or 0),
                "max_session_size_bytes": int(max_size or 0),
                "queries_last_24h": int(queries_24h),
                "database_size_bytes": 0,
                "backend_type": "postgres",
            }

        except Exception as exc:
            logger.error(f"Failed to get session stats: {exc}")
            return {}

    def get_query_usage_stats_24h(self) -> Dict[str, int]:
        """Return aggregate assistant-query usage for the previous 24 hours."""

        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            with self._session_factory() as session:
                total_queries = session.execute(
                    select(func.count()).select_from(UserQuery).where(
                        UserQuery.query_timestamp >= cutoff
                    )
                ).scalar_one()
                total_users = session.execute(
                    select(func.count(func.distinct(UserQuery.user_id))).where(
                        UserQuery.query_timestamp >= cutoff
                    )
                ).scalar_one()

            return {
                "total_queries_24h": int(total_queries),
                "total_users_with_queries": int(total_users),
            }

        except Exception as exc:
            logger.error(f"Failed to get query usage stats: {exc}")
            return {
                "total_queries_24h": 0,
                "total_users_with_queries": 0,
            }

    def health_check(self) -> Dict[str, Any]:
        """Perform a lightweight health check for the Postgres backend."""

        stats = self.get_session_stats()
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_healthy": bool(stats),
            "backend_type": "postgres",
            "database_size_bytes": stats.get("database_size_bytes", 0),
            "active_sessions": stats.get("active_sessions", 0),
        }

    def close(self) -> None:
        """Compatibility hook for the backend factory."""

        return None
