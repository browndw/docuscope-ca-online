"""
Runtime Configuration Management

This module provides dynamic configuration capabilities that persist across
horizontal scaling using the shared health database. Settings can be toggled
at runtime without application restart.

Note: Runtime configuration is automatically disabled in desktop mode to
maintain lightweight architecture and avoid database dependencies.
"""

import threading
import time
from typing import Dict, Any, Optional

from webapp.config.static_config import get_static_value
from webapp.config.unified import get_config
from webapp.utilities.configuration.logging_config import get_logger
from webapp.utilities.storage.backend_factory import get_session_backend

logger = get_logger()


class RuntimeConfigManager:
    """
    Manages runtime configuration overrides with horizontal scaling support.

    Uses the shared health database to ensure configuration changes
    are visible across all application instances.
    """

    def __init__(self):
        """Initialize the runtime configuration manager."""
        self._cache = {}
        self._cache_timestamp = {}
        self._lock = threading.RLock()
        self._cache_ttl = 5.0  # Cache for 5 seconds to reduce DB hits
        self._initialized = False

        # Check if running in desktop mode - disable runtime config for
        # lightweight operation
        self._desktop_mode = get_config('desktop_mode', 'global')
        if self._desktop_mode:
            self._initialized = True  # Mark as initialized to skip database operations

    def _ensure_initialized(self):
        """Ensure the runtime config tables exist in the health database."""
        if self._initialized:
            return

        # Skip initialization in desktop mode
        if self._desktop_mode:
            self._initialized = True
            return

        try:
            backend = get_session_backend()
            with backend.get_health_connection() as conn:
                cursor = conn.cursor()

                # Create runtime configuration table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS runtime_config (
                        config_key TEXT PRIMARY KEY,
                        config_value TEXT NOT NULL,
                        config_type TEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_by TEXT,
                        description TEXT
                    )
                """)

                # Create configuration audit log
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS config_audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        config_key TEXT NOT NULL,
                        old_value TEXT,
                        new_value TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_by TEXT,
                        instance_id TEXT
                    )
                """)

                conn.commit()
                self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize runtime config tables: {e}")
            self._initialized = False

    def _get_from_database(self, config_key: str) -> Optional[Any]:
        """Get configuration value from shared database."""
        # Skip database operations in desktop mode
        if self._desktop_mode:
            return None

        try:
            backend = get_session_backend()
            with backend.get_health_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT config_value, config_type FROM runtime_config "
                    "WHERE config_key = ?",
                    (config_key,)
                )
                result = cursor.fetchone()

                if result:
                    value_str, value_type = result
                    # Convert back to appropriate type
                    if value_type == 'bool':
                        return value_str.lower() == 'true'
                    elif value_type == 'int':
                        return int(value_str)
                    elif value_type == 'float':
                        return float(value_str)
                    else:
                        return value_str

                return None

        except Exception as e:
            logger.error(f"Failed to get runtime config {config_key}: {e}")
            return None

    def _set_in_database(self, config_key: str, value: Any,
                         updated_by: str = None, description: str = None):
        """Set configuration value in shared database."""
        # Skip database operations in desktop mode
        if self._desktop_mode:
            logger.debug(f"Skipping runtime config set in desktop mode: {config_key}")
            return

        try:
            backend = get_session_backend()
            with backend.get_health_connection() as conn:
                cursor = conn.cursor()

                # Get old value for audit
                old_value = self._get_from_database(config_key)

                # Determine value type and convert to string
                if isinstance(value, bool):
                    value_str, value_type = str(value), 'bool'
                elif isinstance(value, int):
                    value_str, value_type = str(value), 'int'
                elif isinstance(value, float):
                    value_str, value_type = str(value), 'float'
                else:
                    value_str, value_type = str(value), 'str'

                # Insert or update configuration
                cursor.execute("""
                    INSERT OR REPLACE INTO runtime_config
                    (config_key, config_value, config_type, updated_at,
                     updated_by, description)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
                """, (config_key, value_str, value_type, updated_by, description))

                # Log the change for audit
                cursor.execute("""
                    INSERT INTO config_audit_log
                    (config_key, old_value, new_value, updated_by, instance_id)
                    VALUES (?, ?, ?, ?, ?)
                """, (config_key, str(old_value) if old_value is not None else None,
                      value_str, updated_by, f"container-{time.time()}"))

                conn.commit()

                # Update local cache
                with self._lock:
                    self._cache[config_key] = value
                    self._cache_timestamp[config_key] = time.time()

                logger.info(f"Runtime config updated: {config_key} = {value} "
                            f"by {updated_by}")

        except Exception as e:
            logger.error(f"Failed to set runtime config {config_key}: {e}")

    def get_config_value(self, key: str, default: Any, section: str) -> Any:
        """
        Get configuration value with runtime override priority.

        Priority order:
        1. Runtime override (from shared database)
        2. TOML default value

        Parameters
        ----------
        key : str
            Configuration key
        default : Any
            Default value if not found
        section : str
            Configuration section

        Returns
        -------
        Any
            Configuration value
        """
        self._ensure_initialized()

        config_key = f"{section}.{key}"

        # Check cache first (with TTL)
        with self._lock:
            if (
                config_key in self._cache and
                config_key in self._cache_timestamp and
                time.time() - self._cache_timestamp[config_key] < self._cache_ttl
            ):
                return self._cache[config_key]

        # Check database for runtime override
        runtime_value = self._get_from_database(config_key)
        if runtime_value is not None:
            # Update cache
            with self._lock:
                self._cache[config_key] = runtime_value
                self._cache_timestamp[config_key] = time.time()
            return runtime_value

        # Fall back to TOML default
        toml_value = get_static_value(key, section, default)

        # Cache the TOML value briefly to reduce lookups
        with self._lock:
            self._cache[config_key] = toml_value
            self._cache_timestamp[config_key] = time.time()

        return toml_value

    def set_runtime_override(
            self,
            key: str,
            section: str,
            value: Any,
            updated_by: str = None,
            description: str = None
    ) -> None:
        """
        Set a runtime configuration override.

        Parameters
        ----------
        key : str
            Configuration key
        section : str
            Configuration section
        value : Any
            New value
        updated_by : str, optional
            User who made the change
        description : str, optional
            Description of the change
        """
        self._ensure_initialized()

        # Skip runtime overrides in desktop mode
        if self._desktop_mode:
            logger.debug(
                f"Skipping runtime override for {section}.{key} "
                "in desktop mode"
            )
            return

        config_key = f"{section}.{key}"
        self._set_in_database(config_key, value, updated_by, description)

    def clear_runtime_override(
            self,
            key: str,
            section: str,
            updated_by: str = None
    ) -> None:
        """Clear a runtime configuration override, reverting to TOML default."""
        self._ensure_initialized()

        # Skip runtime override clearing in desktop mode
        if self._desktop_mode:
            logger.debug(
                f"Skipping runtime override clear for {section}.{key} "
                "in desktop mode"
            )
            return

        config_key = f"{section}.{key}"

        try:
            backend = get_session_backend()
            with backend.get_health_connection() as conn:
                cursor = conn.cursor()

                # Get current value for audit
                old_value = self._get_from_database(config_key)

                # Remove from database
                cursor.execute(
                    "DELETE FROM runtime_config WHERE config_key = ?",
                    (config_key,)
                )

                # Log the change
                cursor.execute(
                    """
                    INSERT INTO config_audit_log
                    (config_key, old_value, new_value, updated_by, instance_id)
                    VALUES (?, ?, 'CLEARED', ?, ?)
                    """,
                    (config_key, str(old_value) if old_value is not None else None,
                     updated_by, f"container-{time.time()}"
                     )
                    )

                conn.commit()

                # Clear from cache
                with self._lock:
                    self._cache.pop(config_key, None)
                    self._cache_timestamp.pop(config_key, None)

                logger.info(f"Runtime config cleared: {config_key} by {updated_by}")

        except Exception as e:
            logger.error(f"Failed to clear runtime config {config_key}: {e}")

    def get_all_overrides(self) -> Dict[str, Any]:
        """Get all current runtime configuration overrides."""
        self._ensure_initialized()

        # Return empty dict in desktop mode - no runtime overrides needed
        if self._desktop_mode:
            return {}

        try:
            backend = get_session_backend()
            with backend.get_health_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT config_key, config_value, config_type, updated_at,
                           updated_by, description
                    FROM runtime_config
                    ORDER BY updated_at DESC
                    """
                    )

                overrides = {}
                for row in cursor.fetchall():
                    key, value_str, value_type, updated_at, updated_by, desc = row

                    # Convert back to appropriate type
                    if value_type == 'bool':
                        value = value_str.lower() == 'true'
                    elif value_type == 'int':
                        value = int(value_str)
                    elif value_type == 'float':
                        value = float(value_str)
                    else:
                        value = value_str

                    overrides[key] = {
                        'value': value,
                        'updated_at': updated_at,
                        'updated_by': updated_by,
                        'description': desc
                    }

                return overrides

        except Exception as e:
            logger.error(f"Failed to get runtime config overrides: {e}")
            return {}

    def get_audit_log(self, limit: int = 50) -> list:
        """Get configuration change audit log."""
        self._ensure_initialized()

        # Return empty list in desktop mode - no audit logging needed
        if self._desktop_mode:
            return []

        try:
            backend = get_session_backend()
            with backend.get_health_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT config_key, old_value, new_value, updated_at,
                           updated_by, instance_id
                    FROM config_audit_log
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (limit,))

                return [
                    {
                        'config_key': row[0],
                        'old_value': row[1],
                        'new_value': row[2],
                        'updated_at': row[3],
                        'updated_by': row[4],
                        'instance_id': row[5]
                    }
                    for row in cursor.fetchall()
                ]

        except Exception as e:
            logger.error(f"Failed to get config audit log: {e}")
            return []

    # Convenience methods for specific configurations
    def toggle_firestore_collection(
            self,
            enabled: bool,
            updated_by: str = None
    ) -> None:
        """Toggle Firestore research data collection."""
        self.set_runtime_override(
            'cache_mode', 'cache', enabled,
            updated_by=updated_by,
            description="Firestore research data collection toggle"
        )

    def is_firestore_enabled(self) -> bool:
        """Check if Firestore research data collection is enabled."""
        return self.get_config_value('cache_mode', False, 'cache')

    def clear_firestore_override(
            self,
            updated_by: str = None
    ) -> None:
        """Clear Firestore collection override, reverting to TOML default."""
        self.clear_runtime_override('cache_mode', 'cache', updated_by=updated_by)


# Global runtime configuration manager instance
runtime_config = RuntimeConfigManager()


def get_runtime_config_value(
        key: str,
        default: Any,
        section: str
) -> Any:
    """
    Convenience function to get configuration value with runtime override support.

    DEPRECATED: Use webapp.config.unified.get_config() instead.
    """
    return runtime_config.get_config_value(key, default, section)
