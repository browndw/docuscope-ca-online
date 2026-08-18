"""
Session Backend Factory

This module provides a factory for creating session backends without
direct imports, helping to avoid circular dependencies.

The factory automatically selects the appropriate backend based on deployment mode:
- desktop_mode = true: In-memory storage (no database bloat for desktop users)
- desktop_mode = false: Postgres control-plane storage
"""

import os
from typing import Optional

from sqlalchemy.engine import make_url

from webapp.config.unified import config
from webapp.persistence.config import get_database_config
from webapp.utilities.configuration.logging_config import get_logger
from webapp.utilities.storage.postgres_session_backend import (
    PostgresSessionBackend
)
from webapp.utilities.storage.sqlite_session_backend import (
    SQLiteSessionBackend
)


logger = get_logger()
LOCAL_DATABASE_HOSTS = {None, "", "localhost", "127.0.0.1", "::1"}


def _create_memory_backend():
    """Create the lightweight backend without eager storage imports."""

    from webapp.utilities.storage.memory_session_backend import (
        InMemorySessionBackend,
    )

    return InMemorySessionBackend()


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
        auto_selected = backend_type is None
        if auto_selected:
            # Backend selection is a bootstrap decision. Do not consult the
            # Postgres-backed runtime config before Postgres availability has
            # been established and local desktop fallback can be activated.
            backend_type = config.get_static('backend', 'session', 'postgres')

        # Resolve the configured mode without consulting Postgres-backed runtime
        # overrides; this backend must be established before those are available.
        desktop_mode = config.is_desktop_mode()

        # Use appropriate backend for deployment mode:
        # - desktop_mode = true: In-memory backend (no database bloat)
        # - desktop_mode = false: Postgres backend by default
        if desktop_mode and backend_type in {'sqlite', 'postgres'}:
            backend_type = 'memory'
        # Return cached instance if available
        if backend_type in self._backend_cache:
            return self._backend_cache[backend_type]

        # Create new backend instance
        if backend_type == 'memory':
            # Lightweight in-memory backend for desktop mode
            backend = _create_memory_backend()
        elif backend_type == 'sqlite':
            # Transitional SQLite backend for local tests and migration support
            backend = SQLiteSessionBackend()
        elif backend_type == 'postgres':
            try:
                backend = PostgresSessionBackend()
            except Exception as exc:
                if not auto_selected or not self._allow_local_desktop_fallback():
                    raise
                config.activate_desktop_fallback(type(exc).__name__)
                backend_type = 'memory'
                backend = _create_memory_backend()
                logger.warning(
                    "Local Postgres is unavailable; DocuScope CA is falling back "
                    "to Desktop Mode with in-memory session storage. Session data "
                    "will last only for this app process. Start the Docker stack "
                    "for enterprise persistence."
                )
        else:
            raise ValueError(
                f"Unknown backend type: {backend_type}. "
                "Supported types: 'memory', 'sqlite', 'postgres'"
            )

        self._backend_cache[backend_type] = backend
        self._log_startup_mode(backend_type)
        return backend

    def _allow_local_desktop_fallback(self) -> bool:
        """Return whether automatic local Postgres failure may use memory."""

        if os.getenv("DOCUSCOPE_DISABLE_DESKTOP_FALLBACK", "").strip() == "1":
            return False
        database_host = make_url(get_database_config().url).host
        return database_host in LOCAL_DATABASE_HOSTS

    def _log_startup_mode(self, backend_type: str) -> None:
        """Log the effective process mode once after backend creation."""

        if hasattr(self, '_startup_logged'):
            return
        mode_text = "Desktop Mode" if config.is_desktop_mode() else "Enterprise Mode"
        backend_text = backend_type.title().replace('_', ' ')
        logger.info(
            "DocuScope CA starting in {} with {} backend",
            mode_text,
            backend_text,
        )
        self._startup_logged = True

    def clear_cache(self):
        """Clear the backend cache."""
        # Close any existing backends
        for backend in self._backend_cache.values():
            if hasattr(backend, 'close'):
                backend.close()
        self._backend_cache.clear()
        if hasattr(self, '_startup_logged'):
            del self._startup_logged


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
