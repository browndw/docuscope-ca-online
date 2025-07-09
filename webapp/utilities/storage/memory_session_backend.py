"""
In-Memory Session Backend for Desktop Mode

This module implements a lightweight, in-memory-only session backend
specifically designed for desktop mode where database persistence
is unnecessary and would create storage bloat.

Features:
- Pure in-memory storage (no SQLite databases)
- Session-scoped data isolation
- Minimal memory footprint
- No background threads or cleanup processes
- No analytics or health monitoring overhead
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

from webapp.config.unified import get_config
from webapp.utilities.configuration.logging_config import get_logger

logger = get_logger()


class InMemorySessionBackend:
    """
    Lightweight in-memory session backend for desktop mode.

    This backend stores all session data in memory only, with no
    persistence, making it ideal for single-user desktop deployments
    where database bloat is a concern.
    """

    def __init__(self):
        """Initialize the in-memory session backend."""
        # Core session storage
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._session_metadata: Dict[str, Dict[str, Any]] = {}

        # Simple cache storage
        self._cache: Dict[str, Dict[str, Any]] = {}

        # User query tracking (in-memory only)
        self._user_queries: Dict[str, list] = {}

        # Configuration
        self._session_timeout_hours = get_config(
            'session_timeout_hours', 'session', 24
            )

    def save_session(
            self,
            session_id: str,
            data: Dict[str, Any],
            user_id: str = None
    ) -> bool:
        """
        Save session data in memory.

        Parameters
        ----------
        session_id : str
            Unique session identifier
        data : Dict[str, Any]
            Session data to store
        user_id : str, optional
            User identifier (not used in desktop mode)

        Returns
        -------
        bool
            True if saved successfully
        """
        try:
            # Calculate expiration
            expires_at = datetime.now(timezone.utc) + timedelta(hours=self._session_timeout_hours)  # noqa: E501

            # Store session data
            self._sessions[session_id] = data.copy()

            # Store metadata
            self._session_metadata[session_id] = {
                'user_id': user_id or 'desktop_user',
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc),
                'expires_at': expires_at,
                'access_count': self._session_metadata.get(
                    session_id, {}).get('access_count', 0) + 1,
                'size_estimate': len(str(data))  # Simple size estimate
            }
            return True

        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            return False

    def load_session(
            self,
            session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load session data from memory.

        Parameters
        ----------
        session_id : str
            Session identifier to load
        user_id : str, optional
            User identifier (not used in desktop mode)

        Returns
        -------
        Optional[Dict[str, Any]]
            Session data or None if not found/expired
        """
        try:
            # Check if session exists
            if session_id not in self._sessions:
                return None

            # Check expiration
            metadata = self._session_metadata.get(session_id, {})
            expires_at = metadata.get('expires_at')
            if expires_at and datetime.now(timezone.utc) > expires_at:
                # Session expired, clean it up
                self._cleanup_session(session_id)
                return None

            # Update access count and last access time
            if session_id in self._session_metadata:
                self._session_metadata[session_id]['access_count'] += 1
                self._session_metadata[session_id]['last_accessed'] = datetime.now(timezone.utc)  # noqa: E501

            logger.debug(f"Loaded session {session_id} from memory")
            return self._sessions[session_id].copy()

        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def delete_session(self, session_id: str, user_id: str = None) -> bool:
        """Delete session from memory."""
        try:
            self._cleanup_session(session_id)
            logger.debug(f"Deleted session {session_id} from memory")
            return True

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    def _cleanup_session(self, session_id: str) -> None:
        """Clean up a single session from memory."""
        self._sessions.pop(session_id, None)
        self._session_metadata.pop(session_id, None)

    def get_user_query_count_24h(self, user_id: str) -> int:
        """Get user query count for the last 24 hours."""
        try:
            if user_id not in self._user_queries:
                return 0

            # Filter queries from last 24 hours
            twenty_four_hours_ago = datetime.now(timezone.utc) - timedelta(hours=24)
            recent_queries = [
                q for q in self._user_queries[user_id]
                if q.get('timestamp', datetime.min) >= twenty_four_hours_ago
            ]

            return len(recent_queries)

        except Exception as e:
            logger.error(f"Failed to get query count for user {user_id}: {e}")
            return 0

    def log_user_query(
            self,
            user_id: str,
            session_id: str,
            assistant_type: str = None,
            message_content: str = None
    ) -> bool:
        """Log user query in memory."""
        try:
            if user_id not in self._user_queries:
                self._user_queries[user_id] = []

            query_record = {
                'session_id': session_id,
                'timestamp': datetime.now(timezone.utc),
                'assistant_type': assistant_type,
                # Don't store full message content to save memory
                'has_content': bool(message_content)
            }

            self._user_queries[user_id].append(query_record)

            # Keep only last 100 queries per user to prevent memory bloat
            if len(self._user_queries[user_id]) > 100:
                self._user_queries[user_id] = self._user_queries[user_id][-100:]

            return True

        except Exception as e:
            logger.error(f"Failed to log user query: {e}")
            return False

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions from memory."""
        try:
            current_time = datetime.now(timezone.utc)
            expired_sessions = []

            for session_id, metadata in self._session_metadata.items():
                expires_at = metadata.get('expires_at')
                if expires_at and current_time > expires_at:
                    expired_sessions.append(session_id)

            # Clean up expired sessions
            for session_id in expired_sessions:
                self._cleanup_session(session_id)

            if expired_sessions:
                logger.info(
                    f"Cleaned up {len(expired_sessions)} expired sessions from memory"
                    )

            return len(expired_sessions)

        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        try:
            current_time = datetime.now(timezone.utc)
            active_sessions = 0
            total_size_estimate = 0
            total_queries = 0

            # Count active sessions
            for session_id, metadata in self._session_metadata.items():
                expires_at = metadata.get('expires_at')
                if expires_at and current_time <= expires_at:
                    active_sessions += 1

                total_size_estimate += metadata.get('size_estimate', 0)

            # Count recent queries
            for user_queries in self._user_queries.values():
                total_queries += len(user_queries)

            return {
                'active_sessions': active_sessions,
                'total_sessions': len(self._sessions),
                'queries_last_24h': total_queries,  # Approximate
                'total_size_bytes': total_size_estimate,
                'avg_session_size_bytes': total_size_estimate / max(len(self._sessions), 1),
                'max_session_size_bytes': max(
                    (m.get('size_estimate', 0) for m in self._session_metadata.values()),
                    default=0
                ),
                'backend_type': 'in_memory'
            }

        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return {
                'active_sessions': 0,
                'total_sessions': 0,
                'queries_last_24h': 0,
                'total_size_bytes': 0,
                'backend_type': 'in_memory',
                'error': str(e)
            }

    # Cache management methods (in-memory)
    def cache_set(
            self,
            key: str,
            value: Any,
            ttl_seconds: int = 3600
    ) -> bool:
        """Set cache value in memory."""
        try:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)

            self._cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': datetime.now(timezone.utc)
            }

            return True

        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False

    def cache_get(self, key: str) -> Optional[Any]:
        """Get cache value from memory."""
        try:
            if key not in self._cache:
                return None

            cache_entry = self._cache[key]

            # Check expiration
            if datetime.now(timezone.utc) > cache_entry['expires_at']:
                del self._cache[key]
                return None

            return cache_entry['value']

        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return None

    def cache_delete(self, key: str) -> bool:
        """Delete cache key from memory."""
        try:
            self._cache.pop(key, None)
            return True

        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False

    def cache_cleanup(self) -> int:
        """Clean up expired cache entries."""
        try:
            current_time = datetime.now(timezone.utc)
            expired_keys = [
                key for key, entry in self._cache.items()
                if current_time > entry['expires_at']
            ]

            for key in expired_keys:
                del self._cache[key]

            return len(expired_keys)

        except Exception as e:
            logger.error(f"Failed to cleanup cache: {e}")
            return 0

    def health_check(self) -> Dict[str, Any]:
        """Perform health check (always healthy for in-memory backend)."""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'overall_healthy': True,
            'backend_type': 'in_memory',
            'memory_usage': {
                'sessions_count': len(self._sessions),
                'cache_count': len(self._cache),
                'users_tracked': len(self._user_queries)
            }
        }

    def close(self):
        """Clean up resources (clear memory)."""
        self._sessions.clear()
        self._session_metadata.clear()
        self._cache.clear()
        self._user_queries.clear()
        logger.info("Closed in-memory session backend")
