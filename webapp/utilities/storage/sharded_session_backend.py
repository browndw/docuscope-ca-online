"""
Sharded SQLite Session Backend for Enterprise Scale

This module implements a session backend that uses the ShardedDatabaseManager
to distribute session data across multiple SQLite databases for enterprise
scale deployments (500+ concurrent users).
"""

import pickle
import hashlib
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

from webapp.config.unified import get_config
from webapp.utilities.configuration.logging_config import get_logger
from webapp.utilities.storage.sqlite_sharding import get_sharded_db_manager

logger = get_logger()


class ShardedSQLiteSessionBackend:
    """
    Enterprise-scale session backend using sharded SQLite databases.

    This backend distributes session data across multiple SQLite databases
    based on user hash, providing horizontal scaling capabilities.
    """

    def __init__(self):
        """Initialize the sharded session backend."""
        self.shard_manager = get_sharded_db_manager()

        # Initialize database schemas for all shards
        self._initialize_all_schemas()

        # Start background cleanup
        self._start_cleanup_thread()

        # Health monitoring
        self._last_health_check = datetime.now(timezone.utc)
        self._health_check_interval = get_config(
            'pool_health_check_interval', 'session', 30
        )

    def _initialize_all_schemas(self):
        """Initialize database schemas for all shards and shared databases."""
        # Initialize session table schema for each shard directly
        for shard_id in range(self.shard_manager.shard_count):
            try:
                # Access session pool directly to ensure we initialize this specific shard
                session_pool = self.shard_manager.session_pools[shard_id]
                with session_pool.get_connection() as conn:
                    self._create_session_schema(conn)

                # Access cache pool directly to ensure we initialize this specific shard
                cache_pool = self.shard_manager.cache_pools[shard_id]
                with cache_pool.get_connection() as conn:
                    self._create_cache_schema(conn)

            except Exception as e:
                logger.error(f"Failed to initialize shard {shard_id}: {e}")

        # Initialize shared database schemas
        try:
            with self.shard_manager.get_analytics_connection() as conn:
                self._create_analytics_schema(conn)

            with self.shard_manager.get_health_connection() as conn:
                self._create_health_schema(conn)

        except Exception as e:
            logger.error(f"Failed to initialize shared databases: {e}")

    def _create_session_schema(
            self,
            conn
    ):
        """Create session tables and indexes for a shard."""
        cursor = conn.cursor()

        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                data BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                size_bytes INTEGER DEFAULT 0,
                access_count INTEGER DEFAULT 0,
                checksum TEXT
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
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            )
        """)

        # Performance indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_user_expires
            ON sessions(user_id, expires_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_expires
            ON sessions(expires_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_queries_user_time
            ON user_queries(user_id, query_timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_queries_session
            ON user_queries(session_id)
        """)

        # Automatic timestamp update trigger
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS update_sessions_timestamp
            AFTER UPDATE ON sessions
            BEGIN
                UPDATE sessions SET updated_at = CURRENT_TIMESTAMP
                WHERE session_id = NEW.session_id;
            END
        """)

        conn.commit()

    def _create_cache_schema(
            self,
            conn
    ):
        """Create cache tables for a shard."""
        cursor = conn.cursor()

        # Cache table with TTL support
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_data (
                cache_key TEXT PRIMARY KEY,
                data BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                size_bytes INTEGER DEFAULT 0,
                access_count INTEGER DEFAULT 0,
                tags TEXT
            )
        """)

        # Cache performance indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_expires
            ON cache_data(expires_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_tags
            ON cache_data(tags)
        """)

        conn.commit()

    def _create_analytics_schema(
            self,
            conn
    ):
        """Create analytics tables in shared database."""
        cursor = conn.cursor()

        # Session analytics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                shard_id INTEGER,
                active_sessions INTEGER,
                total_sessions INTEGER,
                queries_per_hour INTEGER,
                avg_session_size INTEGER,
                max_session_size INTEGER
            )
        """)

        # System performance metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                shard_id INTEGER,
                tags TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_analytics_timestamp
            ON session_analytics(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_name_time
            ON performance_metrics(metric_name, timestamp)
        """)

        conn.commit()

    def _create_health_schema(
            self,
            conn
    ):
        """Create health monitoring tables."""
        cursor = conn.cursor()

        # Shard health status
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shard_health (
                shard_id INTEGER PRIMARY KEY,
                is_healthy BOOLEAN DEFAULT TRUE,
                last_check TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                error_count INTEGER DEFAULT 0,
                last_error TEXT,
                connection_pool_size INTEGER DEFAULT 0,
                active_connections INTEGER DEFAULT 0
            )
        """)

        # Health check history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                shard_id INTEGER,
                check_type TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                response_time_ms REAL,
                error_message TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_health_checks_time
            ON health_checks(timestamp)
        """)

        conn.commit()

    def save_session(
            self,
            session_id: str,
            data: Dict[str, Any],
            user_id: str = None
    ) -> bool:
        """
        Save session data to appropriate shard based on user ID.

        Parameters
        ----------
        session_id : str
            Unique session identifier
        data : Dict[str, Any]
            Session data to store
        user_id : str, optional
            User identifier for sharding

        Returns
        -------
        bool
            True if saved successfully
        """
        try:
            # Use session_id as fallback if no user_id
            shard_key = user_id or session_id

            # Serialize and validate data
            serialized_data = pickle.dumps(data)
            size_bytes = len(serialized_data)

            # Generate checksum for integrity
            checksum = hashlib.md5(serialized_data).hexdigest()

            # Calculate expiration
            session_timeout_hours = get_config('session_timeout_hours', 'session', 24)
            expires_at = datetime.now(timezone.utc) + timedelta(hours=session_timeout_hours)

            # Save to appropriate shard
            with self.shard_manager.get_session_connection(shard_key) as (conn, shard_id):
                cursor = conn.cursor()

                cursor.execute
                (
                    """
                    INSERT OR REPLACE INTO sessions
                    (session_id, user_id, data, expires_at, size_bytes, checksum, access_count)
                    VALUES (?, ?, ?, ?, ?, ?,
                        COALESCE((SELECT access_count FROM sessions WHERE session_id = ?), 0) + 1)
                    """,  # noqa: E501
                    (
                        session_id, user_id, serialized_data, expires_at,
                        size_bytes, checksum, session_id
                    )
                )

                conn.commit()

            # Log to analytics (async)
            self._log_session_analytics('session_save', shard_id, size_bytes)
            return True

        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            return False

    def load_session(
            self,
            session_id: str,
            user_id: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load session data from appropriate shard.

        Parameters
        ----------
        session_id : str
            Session identifier to load
        user_id : str, optional
            User identifier for sharding

        Returns
        -------
        Optional[Dict[str, Any]]
            Session data or None if not found/expired
        """
        try:
            shard_key = user_id or session_id

            with self.shard_manager.get_session_connection(shard_key) as (conn, shard_id):
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT data, checksum, size_bytes FROM sessions
                    WHERE session_id = ? AND expires_at > CURRENT_TIMESTAMP
                """, (session_id,))

                result = cursor.fetchone()
                if result:
                    serialized_data, stored_checksum, size_bytes = result

                    # Verify data integrity
                    current_checksum = hashlib.md5(serialized_data).hexdigest()
                    if current_checksum != stored_checksum:
                        logger.warning(f"Checksum mismatch for session {session_id}")
                        return None

                    # Deserialize data
                    data = pickle.loads(serialized_data)

                    # Update access count
                    cursor.execute("""
                        UPDATE sessions SET access_count = access_count + 1
                        WHERE session_id = ?
                    """, (session_id,))
                    conn.commit()

                    # Log analytics
                    self._log_session_analytics('session_load', shard_id, size_bytes)

                    logger.debug(f"Loaded session {session_id} from shard {shard_id}")
                    return data

            return None

        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def delete_session(
            self,
            session_id: str,
            user_id: str = None
    ) -> bool:
        """Delete session from appropriate shard."""
        try:
            shard_key = user_id or session_id

            with self.shard_manager.get_session_connection(shard_key) as (conn, shard_id):
                cursor = conn.cursor()

                # Delete associated queries first
                cursor.execute(
                    "DELETE FROM user_queries WHERE session_id = ?",
                    (session_id,)
                    )

                # Delete session
                cursor.execute(
                    "DELETE FROM sessions WHERE session_id = ?",
                    (session_id,)
                    )

                conn.commit()

            logger.debug(f"Deleted session {session_id} from shard {shard_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    def get_user_query_count_24h(
            self,
            user_id: str
    ) -> int:
        """Get user query count across all shards for the last 24 hours."""
        try:
            from webapp.utilities.storage.cache_management import persistent_hash
            hashed_user_id = persistent_hash(user_id)

            twenty_four_hours_ago = datetime.now(timezone.utc) - timedelta(hours=24)
            total_count = 0

            # Check primary shard for this user
            with self.shard_manager.get_session_connection(user_id) as (conn, shard_id):
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) FROM user_queries
                    WHERE user_id = ? AND query_timestamp >= ?
                """, (hashed_user_id, twenty_four_hours_ago))

                count = cursor.fetchone()[0]
                total_count += count

            return total_count

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
        """Log user query to appropriate shard."""
        try:
            from webapp.utilities.storage.cache_management import persistent_hash
            hashed_user_id = persistent_hash(user_id)

            with self.shard_manager.get_session_connection(user_id) as (conn, shard_id):
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
        """Clean up expired sessions across all shards."""
        total_cleaned = 0

        for shard_id in range(self.shard_manager.shard_count):
            try:
                temp_user = f"cleanup_shard_{shard_id}"

                with self.shard_manager.get_session_connection(temp_user) as (conn, _):
                    cursor = conn.cursor()

                    # Get expired session IDs first
                    cursor.execute("""
                        SELECT session_id FROM sessions
                        WHERE expires_at <= CURRENT_TIMESTAMP
                    """)
                    expired_sessions = [row[0] for row in cursor.fetchall()]

                    # Delete expired session queries
                    if expired_sessions:
                        placeholders = ','.join('?' * len(expired_sessions))
                        cursor.execute(
                            f"DELETE FROM user_queries WHERE session_id IN ({placeholders})",  # noqa: E501
                            expired_sessions
                        )

                    # Delete expired sessions
                    cursor.execute(
                        "DELETE FROM sessions WHERE expires_at <= CURRENT_TIMESTAMP"
                        )
                    sessions_deleted = cursor.rowcount

                    # Clean up old queries (keep 7 days)
                    seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
                    cursor.execute(
                        "DELETE FROM user_queries WHERE query_timestamp < ?",
                        (seven_days_ago,)
                    )

                    conn.commit()
                    total_cleaned += sessions_deleted

                    if sessions_deleted > 0:
                        logger.info(
                            f"Cleaned {sessions_deleted} sessions from shard {shard_id}"
                            )

            except Exception as e:
                logger.error(f"Failed to cleanup shard {shard_id}: {e}")

        return total_cleaned

    def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics across all shards."""
        total_stats = {
            'active_sessions': 0,
            'total_sessions': 0,
            'queries_last_24h': 0,
            'total_size_bytes': 0,
            'shards': {},
            'shard_count': self.shard_manager.shard_count
        }

        for shard_id in range(self.shard_manager.shard_count):
            try:
                temp_user = f"stats_shard_{shard_id}"

                with self.shard_manager.get_session_connection(temp_user) as (conn, _):
                    cursor = conn.cursor()

                    # Session counts
                    cursor.execute("""
                        SELECT COUNT(*) FROM sessions WHERE expires_at > CURRENT_TIMESTAMP
                    """)
                    active_sessions = cursor.fetchone()[0]

                    cursor.execute("SELECT COUNT(*) FROM sessions")
                    total_sessions = cursor.fetchone()[0]

                    cursor.execute("SELECT SUM(size_bytes) FROM sessions")
                    shard_size = cursor.fetchone()[0] or 0

                    # Query counts
                    cursor.execute("""
                        SELECT COUNT(*) FROM user_queries
                        WHERE query_timestamp >= datetime('now', '-24 hours')
                    """)
                    queries_24h = cursor.fetchone()[0]

                    shard_stats = {
                        'active_sessions': active_sessions,
                        'total_sessions': total_sessions,
                        'queries_24h': queries_24h,
                        'size_bytes': shard_size,
                        'healthy': self.shard_manager.shard_health.get(shard_id, True)
                    }

                    total_stats['shards'][shard_id] = shard_stats
                    total_stats['active_sessions'] += active_sessions
                    total_stats['total_sessions'] += total_sessions
                    total_stats['queries_last_24h'] += queries_24h
                    total_stats['total_size_bytes'] += shard_size

            except Exception as e:
                logger.error(f"Failed to get stats for shard {shard_id}: {e}")
                total_stats['shards'][shard_id] = {'error': str(e)}

        return total_stats

    def get_health_connection(self):
        """
        Get a connection to the health database.

        This method exposes the health database connection from the underlying
        shard manager, allowing external systems (like runtime config) to
        access the shared health database.

        Returns:
            A connection context manager for the health database
        """
        return self.shard_manager.get_health_connection()

    def _log_session_analytics(
            self,
            operation: str,
            shard_id: int,
            size_bytes: int = 0
    ) -> None:
        """Log session operation to analytics database (non-blocking)."""
        try:
            with self.shard_manager.get_analytics_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO performance_metrics
                    (metric_name, metric_value, shard_id, tags)
                    VALUES (?, ?, ?, ?)
                """, (operation, size_bytes, shard_id, 'session_operation'))

                conn.commit()

        except Exception as e:
            # Don't fail the main operation if analytics fails
            logger.debug(f"Failed to log analytics: {e}")

    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_worker():
            while True:
                try:
                    # Clean up every hour
                    time.sleep(3600)
                    cleaned = self.cleanup_expired_sessions()
                    if cleaned > 0:
                        logger.info(
                            f"Background cleanup removed {cleaned} expired sessions"
                            )

                except Exception as e:
                    logger.error(f"Background cleanup failed: {e}")

        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()

    def health_check(self) -> Dict[str, Any]:
        """Perform health check across all shards."""
        now = datetime.now(timezone.utc)

        # Only run health check if enough time has passed
        if (now - self._last_health_check).seconds < self._health_check_interval:
            return self.shard_manager.get_shard_statistics()

        self._last_health_check = now
        health_report = {
            'timestamp': now.isoformat(),
            'overall_healthy': True,
            'shards': {}
        }

        for shard_id in range(self.shard_manager.shard_count):
            try:
                start_time = time.time()
                temp_user = f"health_check_{shard_id}"

                with self.shard_manager.get_session_connection(temp_user) as (conn, _):
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.fetchone()

                response_time = (time.time() - start_time) * 1000

                shard_health = {
                    'healthy': True,
                    'response_time_ms': response_time,
                    'last_check': now.isoformat()
                }

                health_report['shards'][shard_id] = shard_health

                # Log to health database
                with self.shard_manager.get_health_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO health_checks
                        (shard_id, check_type, success, response_time_ms)
                        VALUES (?, ?, ?, ?)
                    """, (shard_id, 'connection_test', True, response_time))
                    conn.commit()

            except Exception as e:
                logger.error(f"Health check failed for shard {shard_id}: {e}")

                health_report['shards'][shard_id] = {
                    'healthy': False,
                    'error': str(e),
                    'last_check': now.isoformat()
                }
                health_report['overall_healthy'] = False

                # Mark shard as unhealthy
                with self.shard_manager._health_lock:
                    self.shard_manager.shard_health[shard_id] = False

        return health_report

    # Cache management methods for the sharded backend
    def cache_set(
            self,
            key: str,
            value: Any,
            ttl_seconds: int = 3600
    ) -> bool:
        """Set cache value in appropriate shard."""
        try:
            serialized_value = pickle.dumps(value)
            size_bytes = len(serialized_value)
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)

            with self.shard_manager.get_cache_connection(key) as (conn, shard_id):
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO cache_data
                    (cache_key, data, expires_at, size_bytes, access_count)
                    VALUES (?, ?, ?, ?, 1)
                """, (key, serialized_value, expires_at, size_bytes))

                conn.commit()

            return True

        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False

    def cache_get(
            self,
            key: str
    ) -> Optional[Any]:
        """Get cache value from appropriate shard."""
        try:
            with self.shard_manager.get_cache_connection(key) as (conn, shard_id):
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT data FROM cache_data
                    WHERE cache_key = ? AND expires_at > CURRENT_TIMESTAMP
                """, (key,))

                result = cursor.fetchone()
                if result:
                    # Update access count
                    cursor.execute("""
                        UPDATE cache_data SET access_count = access_count + 1
                        WHERE cache_key = ?
                    """, (key,))
                    conn.commit()

                    return pickle.loads(result[0])

            return None

        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return None

    def cache_delete(self, key: str) -> bool:
        """Delete cache key from appropriate shard."""
        try:
            with self.shard_manager.get_cache_connection(key) as (conn, shard_id):
                cursor = conn.cursor()
                cursor.execute("DELETE FROM cache_data WHERE cache_key = ?", (key,))
                conn.commit()

            return True

        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False

    def cache_cleanup(self) -> int:
        """Clean up expired cache entries across all shards."""
        total_cleaned = 0

        for shard_id in range(self.shard_manager.shard_count):
            try:
                temp_key = f"cleanup_cache_shard_{shard_id}"

                with self.shard_manager.get_cache_connection(temp_key) as (conn, _):
                    cursor = conn.cursor()
                    cursor.execute(
                        "DELETE FROM cache_data WHERE expires_at <= CURRENT_TIMESTAMP"
                        )
                    cleaned = cursor.rowcount
                    conn.commit()

                    total_cleaned += cleaned

                    if cleaned > 0:
                        logger.info(
                            f"Cleaned {cleaned} cache entries from shard {shard_id}"
                            )

            except Exception as e:
                logger.error(f"Failed to cleanup cache shard {shard_id}: {e}")

        return total_cleaned
