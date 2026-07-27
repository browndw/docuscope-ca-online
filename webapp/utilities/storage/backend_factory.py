"""
Session Backend Factory

This module provides a factory for creating session backends without
direct imports, helping to avoid circular dependencies.

The factory automatically selects the appropriate backend based on deployment mode:
- desktop_mode = true: In-memory storage (no database bloat for desktop users)
- desktop_mode = false: Postgres control-plane storage
"""

from typing import Optional
from webapp.config.unified import get_config
from webapp.utilities.storage.postgres_session_backend import (
    PostgresSessionBackend
)
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
        - desktop_mode = false: Postgres control-plane storage

        Parameters
        ----------
        backend_type : str, optional
            Type of backend to create. If None, auto-selects based on desktop_mode.

        Returns
        -------
        Backend instance
        """
        if backend_type is None:
            backend_type = get_config('backend', 'session', 'postgres')

        # Auto-select backend based on desktop_mode
        desktop_mode = get_config('desktop_mode', 'global', True)

        # Use appropriate backend for deployment mode:
        # - desktop_mode = true: In-memory backend (no database bloat)
        # - desktop_mode = false: Postgres backend by default
        if desktop_mode and backend_type in {'sqlite', 'postgres'}:
            backend_type = 'memory'
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
            # Transitional SQLite backend for local tests and migration support
            backend = SQLiteSessionBackend()
            self._backend_cache[backend_type] = backend
            return backend
        elif backend_type == 'postgres':
            backend = PostgresSessionBackend()
            self._backend_cache[backend_type] = backend
            return backend
        else:
            raise ValueError(
                f"Unknown backend type: {backend_type}. "
                "Supported types: 'memory', 'sqlite', 'postgres'"
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
