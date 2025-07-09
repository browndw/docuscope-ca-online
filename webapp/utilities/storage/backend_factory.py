"""
Session Backend Factory

This module provides a factory for creating session backends without
direct imports, helping to avoid circular dependencies.

The factory automatically selects the appropriate backend based on deployment mode:
- desktop_mode = true: In-memory storage (no database bloat for desktop users)
- desktop_mode = false: Sharded SQLite databases (enterprise scale)
"""

from typing import Optional
from webapp.config.unified import get_config
from webapp.utilities.storage.sqlite_session_backend import (
    SQLiteSessionBackend
)


class SessionBackendFactory:
    """Factory for creating session storage backends."""

    _instance = None
    _backend_cache = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_backend(self, backend_type: Optional[str] = None):
        """
        Get a session backend instance.

        Automatically selects appropriate backend based on deployment mode:
        - desktop_mode = true: In-memory storage (no database files)
        - desktop_mode = false: Sharded SQLite (enterprise scale)

        Parameters
        ----------
        backend_type : str, optional
            Type of backend to create. If None, auto-selects based on desktop_mode.

        Returns
        -------
        Backend instance
        """
        if backend_type is None:
            backend_type = get_config('backend', 'session', 'sqlite')

        # Auto-select backend based on desktop_mode
        desktop_mode = get_config('desktop_mode', 'global', True)

        # Use appropriate backend for deployment mode:
        # - desktop_mode = true: In-memory backend (no database bloat)
        # - desktop_mode = false: Sharded SQLite backend (enterprise scale)
        if desktop_mode and backend_type == 'sqlite':
            backend_type = 'memory'
        elif not desktop_mode and backend_type == 'sqlite':
            backend_type = 'sharded_sqlite'
        # Log startup mode information (one-time only)
        if not hasattr(self, '_startup_logged'):
            from webapp.utilities.configuration.logging_config import get_logger
            logger = get_logger()

            mode_text = "Desktop Mode" if desktop_mode else "Enterprise Mode"
            backend_text = backend_type.title().replace('_', ' ')

            logger.info(f"DocuScope CA starting in {mode_text} with {backend_text} backend")
            self._startup_logged = True
        # Return cached instance if available
        if backend_type in self._backend_cache:
            return self._backend_cache[backend_type]

        # Create new backend instance
        if backend_type == 'memory':
            # Lightweight in-memory backend for desktop mode
            from webapp.utilities.storage.memory_session_backend import (
                InMemorySessionBackend
            )
            backend = InMemorySessionBackend()
            self._backend_cache[backend_type] = backend
            return backend
        elif backend_type == 'sqlite':
            # Standard SQLite backend for legacy/testing use
            backend = SQLiteSessionBackend()
            self._backend_cache[backend_type] = backend
            return backend
        elif backend_type == 'sharded_sqlite':
            # Enterprise sharded SQLite backend
            try:
                from webapp.utilities.storage.sharded_session_backend import (
                    ShardedSQLiteSessionBackend
                )
                backend = ShardedSQLiteSessionBackend()
                self._backend_cache[backend_type] = backend
                return backend
            except ImportError as e:
                # Fallback to single SQLite if sharded backend unavailable
                from webapp.utilities.configuration.logging_config import get_logger
                logger = get_logger()
                logger.warning(f"Sharded backend unavailable, falling back to SQLite: {e}")
                return self.get_backend('sqlite')
        else:
            raise ValueError(
                f"Unknown backend type: {backend_type}. "
                "Supported types: 'memory', 'sqlite', 'sharded_sqlite'"
            )

    def clear_cache(self):
        """Clear the backend cache."""
        # Close any existing backends
        for backend in self._backend_cache.values():
            if hasattr(backend, 'close'):
                backend.close()
        self._backend_cache.clear()


# Global factory instance
backend_factory = SessionBackendFactory()


def get_session_backend(backend_type: Optional[str] = None):
    """
    Convenience function to get a session backend.

    Parameters
    ----------
    backend_type : str, optional
        Type of backend to create. If None, uses configuration.

    Returns
    -------
    Backend instance
    """
    return backend_factory.get_backend(backend_type)
