"""
SQLite-based session storage backend with WAL mode and connection pooling.

This module provides persistent session storage using SQLite with optimizations
for concurrent access during high-traffic scenarios (classroom usage).
"""

import sqlite3
import pickle
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import contextmanager
import queue

from webapp.config.unified import get_config
from webapp.utilities.configuration.logging_config import get_logger

logger = get_logger()


class SQLiteConnectionPool:
    """
    Connection pool for SQLite with WAL mode support.
    Optimized for concurrent classroom usage.
    """

    def __init__(self, db_path: str, pool_size: int = 10):
        self.db_path = db_path
        self.pool_size = pool_size
        self.pool = queue.Queue(maxsize=pool_size)
        self.lock = threading.Lock()
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize the connection pool with WAL mode enabled."""
        for _ in range(self.pool_size):
            conn = self._create_connection()
            self.pool.put(conn)

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with optimized settings."""
        conn = sqlite3.Connection(
            self.db_path,
            timeout=30.0,  # 30 second timeout for classroom scenarios
            check_same_thread=False
        )

        # Enable WAL mode for concurrent access
        conn.execute("PRAGMA journal_mode=WAL")

        # Performance optimizations for concurrent access
        conn.execute("PRAGMA synchronous=NORMAL")  # Balance safety/performance
        conn.execute("PRAGMA cache_size=10000")    # 40MB cache
        conn.execute("PRAGMA temp_store=MEMORY")   # Use memory for temp tables
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory map

        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys=ON")

        return conn

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        conn = None
        try:
            # Try to get connection from pool with timeout
            conn = self.pool.get(timeout=10.0)
            yield conn
        except queue.Empty:
            # Pool exhausted, create temporary connection
            logger.warning("Connection pool exhausted, creating temporary connection")
            conn = self._create_connection()
            yield conn
        finally:
            if conn:
                try:
                    # Return connection to pool if it's healthy
                    if self.pool.qsize() < self.pool_size:
                        self.pool.put(conn)
                    else:
                        conn.close()
                except Exception:
                    # If pool is full or connection is bad, close it
                    conn.close()


class SQLiteSessionBackend:
    """
    SQLite-based session storage with connection pooling and WAL mode.
    Optimized for educational deployment scenarios.
    """

    def __init__(self, storage_path: Optional[str] = None):
        # Get storage path from configuration if not provided
        if storage_path is None:
            # Try sqlite_db_path first, then fall back to storage_path + sessions.db
            sqlite_db_path = get_config('sqlite_db_path', 'session', None)
            if sqlite_db_path:
                self.db_path = sqlite_db_path
                self.storage_path = Path(sqlite_db_path).parent
            else:
                storage_dir = get_config(
                    'storage_path', 'session', 'webapp/_session'
                )
                self.storage_path = Path(storage_dir)
                self.db_path = str(self.storage_path / "sessions.db")
        else:
            self.storage_path = Path(storage_path)
            self.db_path = str(self.storage_path / "sessions.db")

        # Ensure directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Connection pool for concurrent access
        pool_size = get_config('connection_pool_size', 'session', 10)
        self.pool = SQLiteConnectionPool(self.db_path, pool_size)

        # Initialize database schema
        self._initialize_database()

        # Background cleanup thread
        self._start_cleanup_thread()

    def _initialize_database(self):
        """Initialize database schema with proper indexes."""
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()

            # Sessions table for core session data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    data BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    size_bytes INTEGER,
                    access_count INTEGER DEFAULT 0
                )
            """)

            # User queries table for quota tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    query_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    assistant_type TEXT,
                    message_content TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)

            # Performance indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_expires "
                "ON sessions(expires_at)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_user "
                "ON sessions(user_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_updated "
                "ON sessions(updated_at)"
            )
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_queries_user_time
                ON user_queries(user_id, query_timestamp)
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_queries_session "
                "ON user_queries(session_id)"
            )

            # Trigger to update updated_at timestamp
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS update_sessions_timestamp
                AFTER UPDATE ON sessions
                BEGIN
                    UPDATE sessions SET updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = NEW.session_id;
                END
            """)

            conn.commit()

    def save_session(self, session_id: str, data: Dict[str, Any],
                     user_id: str = None) -> bool:
        """
        Save session data to SQLite.

        Parameters
        ----------
        session_id : str
            Unique session identifier
        data : Dict[str, Any]
            Session data to store
        user_id : str, optional
            User identifier for the session

        Returns
        -------
        bool
            True if saved successfully
        """
        try:
            # Serialize data
            serialized_data = pickle.dumps(data)
            size_bytes = len(serialized_data)

            # Calculate expiration (24 hours from now)
            expires_at = datetime.now(timezone.utc) + timedelta(hours=24)

            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                # Upsert session data
                cursor.execute("""
                    INSERT OR REPLACE INTO sessions
                    (session_id, user_id, data, expires_at, size_bytes, access_count)
                    VALUES (?, ?, ?, ?, ?,
                        COALESCE(
                            (SELECT access_count FROM sessions WHERE session_id = ?), 0
                        ) + 1)
                """, (session_id, user_id, serialized_data, expires_at,
                      size_bytes, session_id))

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            return False

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load session data from SQLite.

        Parameters
        ----------
        session_id : str
            Session identifier to load

        Returns
        -------
        Optional[Dict[str, Any]]
            Session data or None if not found/expired
        """
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                # Load session with expiration check
                cursor.execute("""
                    SELECT data, expires_at FROM sessions
                    WHERE session_id = ? AND expires_at > CURRENT_TIMESTAMP
                """, (session_id,))

                result = cursor.fetchone()
                if result:
                    serialized_data, expires_at = result
                    data = pickle.loads(serialized_data)

                    # Update access count
                    cursor.execute("""
                        UPDATE sessions SET access_count = access_count + 1
                        WHERE session_id = ?
                    """, (session_id,))
                    conn.commit()

                    return data

                return None

        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def delete_session(self, session_id: str) -> bool:
        """Delete session and associated data."""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                # Delete associated query records first
                cursor.execute(
                    "DELETE FROM user_queries WHERE session_id = ?", (session_id,)
                )

                # Delete session
                cursor.execute(
                    "DELETE FROM sessions WHERE session_id = ?", (session_id,)
                )

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    def get_user_query_count_24h(self, user_id: str) -> int:
        """
        Get user query count for the last 24 hours from SQLite.
        This replaces the Firestore query for quota checking.

        Parameters
        ----------
        user_id : str
            User identifier (will be hashed for consistency)

        Returns
        -------
        int
            Number of queries in the last 24 hours
        """
        try:
            # Hash user_id for consistency with existing system
            from webapp.utilities.storage.cache_management import persistent_hash
            hashed_user_id = persistent_hash(user_id)

            # Calculate 24 hours ago
            twenty_four_hours_ago = datetime.now(timezone.utc) - timedelta(hours=24)

            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT COUNT(*) FROM user_queries
                    WHERE user_id = ? AND query_timestamp >= ?
                """, (hashed_user_id, twenty_four_hours_ago))

                count = cursor.fetchone()[0]
                return count

        except Exception as e:
            logger.error(f"Failed to get query count for user {user_id}: {e}")
            return 0  # Fail-safe: allow usage

    def log_user_query(self, user_id: str, session_id: str,
                       assistant_type: str = None,
                       message_content: str = None) -> bool:
        """
        Log a user query for quota tracking.

        Parameters
        ----------
        user_id : str
            User identifier
        session_id : str
            Session identifier
        assistant_type : str, optional
            Type of assistant ('plotbot', 'pandasai', etc.)
        message_content : str, optional
            Query content (for research if enabled)

        Returns
        -------
        bool
            True if logged successfully
        """
        try:
            from webapp.utilities.storage.cache_management import persistent_hash
            hashed_user_id = persistent_hash(user_id)

            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO user_queries
                    (user_id, session_id, assistant_type, message_content)
                    VALUES (?, ?, ?, ?)
                """, (hashed_user_id, session_id, assistant_type, message_content))

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to log user query: {e}")
            return False

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and old query logs."""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                # Get expired session IDs first
                cursor.execute(
                    "SELECT session_id FROM sessions "
                    "WHERE expires_at <= CURRENT_TIMESTAMP"
                )
                expired_sessions = [row[0] for row in cursor.fetchall()]

                # Delete expired session queries
                if expired_sessions:
                    placeholders = ','.join('?' * len(expired_sessions))
                    cursor.execute(
                        f"DELETE FROM user_queries "
                        f"WHERE session_id IN ({placeholders})",
                        expired_sessions
                    )

                # Delete expired sessions
                cursor.execute(
                    "DELETE FROM sessions WHERE expires_at <= CURRENT_TIMESTAMP"
                )
                sessions_deleted = cursor.rowcount

                # Clean up old query logs (keep 7 days for research)
                seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
                cursor.execute(
                    "DELETE FROM user_queries WHERE query_timestamp < ?",
                    (seven_days_ago,)
                )
                queries_deleted = cursor.rowcount

                conn.commit()

                if sessions_deleted > 0 or queries_deleted > 0:
                    logger.info(
                        f"Cleaned up {sessions_deleted} expired sessions "
                        f"and {queries_deleted} old queries"
                    )

                return sessions_deleted

        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0

    def get_session_stats(self) -> Dict[str, Any]:
        """Get database statistics for monitoring."""
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()

                # Session statistics
                cursor.execute(
                    "SELECT COUNT(*) FROM sessions WHERE expires_at > CURRENT_TIMESTAMP"
                )
                active_sessions = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM sessions")
                total_sessions = cursor.fetchone()[0]

                cursor.execute("SELECT AVG(size_bytes), MAX(size_bytes) FROM sessions")
                avg_size, max_size = cursor.fetchone()

                # Query statistics
                cursor.execute("""
                    SELECT COUNT(*) FROM user_queries
                    WHERE query_timestamp >= datetime('now', '-24 hours')
                """)
                queries_24h = cursor.fetchone()[0]

                # Database size
                cursor.execute("""
                    SELECT page_count * page_size as size
                    FROM pragma_page_count(), pragma_page_size()
                """)
                db_size = cursor.fetchone()[0]

                return {
                    'active_sessions': active_sessions,
                    'total_sessions': total_sessions,
                    'avg_session_size_bytes': avg_size or 0,
                    'max_session_size_bytes': max_size or 0,
                    'queries_last_24h': queries_24h,
                    'database_size_bytes': db_size,
                    'pool_size': self.pool.pool_size,
                    'pool_available': self.pool.pool.qsize()
                }

        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return {}

    def _start_cleanup_thread(self):
        """Start background thread for periodic cleanup."""
        def cleanup_worker():
            while True:
                try:
                    # Clean up every hour
                    time.sleep(3600)
                    self.cleanup_expired_sessions()
                except Exception as e:
                    logger.error(f"Background cleanup failed: {e}")

        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()


# Global session backend instance
_session_backend = None
_backend_lock = threading.Lock()


def get_session_backend() -> SQLiteSessionBackend:
    """Get the global session backend instance (singleton)."""
    global _session_backend

    if _session_backend is None:
        with _backend_lock:
            if _session_backend is None:
                storage_path = get_config(
                    'storage_path', 'session', 'webapp/_session'
                )
                _session_backend = SQLiteSessionBackend(storage_path)

    return _session_backend
