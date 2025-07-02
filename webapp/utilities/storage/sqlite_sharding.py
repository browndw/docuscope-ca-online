"""
Enterprise Database Sharding for SQLite
Handles 500+ concurrent users across multiple database shards.
"""

import hashlib
import threading
from typing import Dict, Any
from pathlib import Path
from contextlib import contextmanager

from webapp.config.unified import get_config
from webapp.utilities.configuration.logging_config import get_logger
from webapp.utilities.storage.sqlite_session_backend import SQLiteConnectionPool

logger = get_logger()


class ShardedDatabaseManager:
    """
    Manages multiple SQLite database shards for enterprise scale.
    Distributes load across multiple databases based on user hash.
    """

    def __init__(self):
        # Check if enterprise mode is enabled (desktop_mode = false)
        desktop_mode = get_config('desktop_mode', 'global', True)

        if desktop_mode:
            # Desktop mode: minimal sharding for compatibility
            self.shard_count = 1
            self.enable_sharding = False
            logger.info("Desktop mode: using single database (no sharding)")
        else:
            # Enterprise mode: full sharding enabled
            self.shard_count = get_config('shard_count', 'session', 8)
            self.enable_sharding = True

        self.shard_strategy = get_config('shard_strategy', 'session', 'user_hash')

        # Connection pools for each shard
        self.session_pools: Dict[int, SQLiteConnectionPool] = {}
        self.cache_pools: Dict[int, SQLiteConnectionPool] = {}

        # Shared databases (not sharded)
        self.analytics_pool = None
        self.health_pool = None

        self._initialize_shards()
        self._initialize_shared_databases()

        # Shard health tracking
        self.shard_health = {i: True for i in range(self.shard_count)}
        self._health_lock = threading.RLock()

    def _initialize_shards(self):
        """Initialize connection pools for all shards."""
        if not self.enable_sharding:
            # Fallback to single database
            self.shard_count = 1

        sessions_pattern = get_config(
            'sessions_shard_pattern',
            'session',
            'webapp/_session/sessions_shard_{}.db'
        )
        cache_pattern = get_config(
            'cache_shard_pattern',
            'session',
            'webapp/_session/cache_shard_{}.db'
        )

        pool_size = get_config('connection_pool_size', 'session', 20)
        cache_pool_size = get_config('cache_pool_size', 'session', 15)

        for shard_id in range(self.shard_count):
            # Session database pools
            session_db_path = sessions_pattern.format(shard_id)
            Path(session_db_path).parent.mkdir(parents=True, exist_ok=True)
            self.session_pools[shard_id] = SQLiteConnectionPool(
                session_db_path, pool_size
            )

            # Cache database pools
            cache_db_path = cache_pattern.format(shard_id)
            Path(cache_db_path).parent.mkdir(parents=True, exist_ok=True)
            self.cache_pools[shard_id] = SQLiteConnectionPool(
                cache_db_path, cache_pool_size
            )

    def _initialize_shared_databases(self):
        """Initialize shared databases that are not sharded."""
        analytics_path = get_config(
            'analytics_db_path',
            'session',
            'webapp/_session/analytics.db'
        )
        health_path = get_config(
            'health_db_path',
            'session',
            'webapp/_session/health.db'
        )

        analytics_pool_size = get_config('analytics_pool_size', 'session', 8)
        health_pool_size = get_config('health_pool_size', 'session', 3)

        self.analytics_pool = SQLiteConnectionPool(analytics_path, analytics_pool_size)
        self.health_pool = SQLiteConnectionPool(health_path, health_pool_size)

    def get_shard_id(self, user_id: str) -> int:
        """
        Determine which shard to use for a given user.

        Parameters
        ----------
        user_id : str
            User identifier

        Returns
        -------
        int
            Shard ID (0 to shard_count-1)
        """
        if not self.enable_sharding or self.shard_count == 1:
            return 0

        if self.shard_strategy == "user_hash":
            # Consistent hashing based on user ID
            hash_value = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
            return hash_value % self.shard_count
        elif self.shard_strategy == "round_robin":
            # Simple round-robin (not persistent across restarts)
            return hash(user_id) % self.shard_count
        else:
            # Default to hash-based
            return hash(user_id) % self.shard_count

    @contextmanager
    def get_session_connection(self, user_id: str):
        """Get a database connection for session data."""
        shard_id = self.get_shard_id(user_id)

        # Check shard health
        with self._health_lock:
            if not self.shard_health.get(shard_id, True):
                # Use fallback shard if primary is unhealthy
                shard_id = (shard_id + 1) % self.shard_count
                logger.warning(f"Using fallback shard {shard_id} for user {user_id}")

        try:
            with self.session_pools[shard_id].get_connection() as conn:
                yield conn, shard_id
        except Exception as e:
            # Mark shard as unhealthy
            with self._health_lock:
                self.shard_health[shard_id] = False
            logger.error(f"Shard {shard_id} failed: {e}")
            raise

    @contextmanager
    def get_cache_connection(self, cache_key: str):
        """Get a database connection for cache data."""
        # Use cache key to determine shard (for even distribution)
        shard_id = hash(cache_key) % self.shard_count

        try:
            with self.cache_pools[shard_id].get_connection() as conn:
                yield conn, shard_id
        except Exception as e:
            logger.error(f"Cache shard {shard_id} failed: {e}")
            raise

    @contextmanager
    def get_analytics_connection(self):
        """Get connection to shared analytics database."""
        with self.analytics_pool.get_connection() as conn:
            yield conn

    @contextmanager
    def get_health_connection(self):
        """Get connection to health monitoring database."""
        with self.health_pool.get_connection() as conn:
            yield conn

    def get_shard_statistics(self) -> Dict[str, Any]:
        """Get statistics about shard usage and health."""
        stats = {
            'shard_count': self.shard_count,
            'sharding_enabled': self.enable_sharding,
            'shard_strategy': self.shard_strategy,
            'shard_health': self.shard_health.copy(),
            'pool_stats': {}
        }

        for shard_id in range(self.shard_count):
            session_pool = self.session_pools[shard_id]
            cache_pool = self.cache_pools[shard_id]

            stats['pool_stats'][f'shard_{shard_id}'] = {
                'session_pool_available': session_pool.pool.qsize(),
                'session_pool_size': session_pool.pool_size,
                'cache_pool_available': cache_pool.pool.qsize(),
                'cache_pool_size': cache_pool.pool_size,
                'healthy': self.shard_health.get(shard_id, True)
            }

        return stats

    def health_check_all_shards(self) -> bool:
        """Perform health check on all shards."""
        all_healthy = True

        for shard_id in range(self.shard_count):
            try:
                with self.session_pools[shard_id].get_connection() as conn:
                    conn.execute("SELECT 1").fetchone()
                with self.cache_pools[shard_id].get_connection() as conn:
                    conn.execute("SELECT 1").fetchone()

                with self._health_lock:
                    self.shard_health[shard_id] = True

            except Exception as e:
                logger.error(f"Health check failed for shard {shard_id}: {e}")
                with self._health_lock:
                    self.shard_health[shard_id] = False
                all_healthy = False

        return all_healthy

    def backup_shard(self, shard_id: int, backup_path: str) -> bool:
        """Create backup of specific shard."""
        try:
            # SQLite backup using VACUUM INTO
            with self.session_pools[shard_id].get_connection() as conn:
                conn.execute(f"VACUUM INTO '{backup_path}_sessions.db'")

            with self.cache_pools[shard_id].get_connection() as conn:
                conn.execute(f"VACUUM INTO '{backup_path}_cache.db'")

            logger.info(f"Backup completed for shard {shard_id}")
            return True

        except Exception as e:
            logger.error(f"Backup failed for shard {shard_id}: {e}")
            return False


# Global sharded database manager
_sharded_db_manager = None
_manager_lock = threading.Lock()


def get_sharded_db_manager() -> ShardedDatabaseManager:
    """Get the global sharded database manager (singleton)."""
    global _sharded_db_manager

    if _sharded_db_manager is None:
        with _manager_lock:
            if _sharded_db_manager is None:
                _sharded_db_manager = ShardedDatabaseManager()

    return _sharded_db_manager
