"""
Session persistence layer for integrating SQLite backend with Streamlit session state.

This module provides automatic session persistence by hooking into Streamlit's
session state management and syncing with the SQLite session backend.
"""

import streamlit as st
from typing import Dict, Any
from datetime import datetime, timezone
import hashlib
import json

from webapp.utilities.configuration.logging_config import get_logger

logger = get_logger()


def _get_session_backend():
    """Get the session backend using factory pattern."""
    # Import backend factory only when needed to avoid circular imports
    from webapp.utilities.storage.backend_factory import get_session_backend
    return get_session_backend()


class SessionPersistenceManager:
    """
    Manages automatic session persistence between Streamlit session state and SQLite.
    """

    def __init__(self):
        """Initialize the session persistence manager."""
        self._backend = None
        self._session_cache = {}
        self._last_sync = {}

    @property
    def backend(self):
        """Get the session backend, initializing if needed."""
        if self._backend is None:
            try:
                self._backend = _get_session_backend()
            except Exception as e:
                logger.error(f"Failed to initialize session backend: {e}")
                return None
        return self._backend

    def get_user_id(self) -> str:
        """
        Get user ID from Streamlit context or generate anonymous ID.

        Returns
        -------
        str
            User identifier
        """
        try:
            # Try to get user email from Streamlit
            if hasattr(st, 'user') and st.user and hasattr(st.user, 'email'):
                return st.user.email

            # Fallback to session-based anonymous ID
            if 'anonymous_user_id' not in st.session_state:
                # Generate consistent anonymous ID based on session
                session_info = str(st.session_state.get('session_id', 'anonymous'))
                st.session_state['anonymous_user_id'] = hashlib.md5(
                    session_info.encode()
                ).hexdigest()[:16]

            return f"anon_{st.session_state['anonymous_user_id']}"

        except Exception:
            return "anonymous_user"

    def load_session(self, session_id: str) -> bool:
        """
        Load session data from SQLite into Streamlit session state.

        Parameters
        ----------
        session_id : str
            Session identifier

        Returns
        -------
        bool
            True if session was loaded successfully
        """
        if self.backend is None:
            return False

        try:
            # Check if already loaded and current
            if (session_id in self._session_cache and
                    session_id in st.session_state and
                    self._is_session_current(session_id)):
                return True

            # Load from SQLite
            session_data = self.backend.load_session(session_id)
            if session_data:
                # Restore to Streamlit session state
                st.session_state[session_id] = session_data
                self._session_cache[session_id] = self._hash_session_data(session_data)
                self._last_sync[session_id] = datetime.now(timezone.utc)

                logger.info(f"Loaded session {session_id} from SQLite")
                return True

        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")

        return False

    def save_session(self, session_id: str, force: bool = False) -> bool:
        """
        Save session data from Streamlit session state to SQLite.

        Parameters
        ----------
        session_id : str
            Session identifier
        force : bool
            Force save even if data hasn't changed

        Returns
        -------
        bool
            True if session was saved successfully
        """
        if self.backend is None:
            return False

        try:
            # Check if session exists in Streamlit
            if session_id not in st.session_state:
                return False

            session_data = st.session_state[session_id]

            # Check if data has changed (unless forced)
            if not force and self._is_session_current(session_id):
                return True

            # Get user ID
            user_id = self.get_user_id()

            # Save to SQLite
            success = self.backend.save_session(session_id, session_data, user_id)

            if success:
                # Update cache
                self._session_cache[session_id] = self._hash_session_data(session_data)
                self._last_sync[session_id] = datetime.now(timezone.utc)
                return True

        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")

        return False

    def auto_save_session(self, session_id: str) -> bool:
        """
        Automatically save session if it has been modified.

        Parameters
        ----------
        session_id : str
            Session identifier

        Returns
        -------
        bool
            True if session was saved or is current
        """
        if session_id not in st.session_state:
            return False

        # Only save if data has changed
        if not self._is_session_current(session_id):
            return self.save_session(session_id)

        return True

    def delete_session(self, session_id: str) -> bool:
        """
        Delete session from both SQLite and Streamlit state.

        Parameters
        ----------
        session_id : str
            Session identifier

        Returns
        -------
        bool
            True if session was deleted successfully
        """
        success = True

        try:
            # Remove from SQLite
            if self.backend:
                self.backend.delete_session(session_id)

            # Remove from Streamlit state
            if session_id in st.session_state:
                del st.session_state[session_id]

            # Clean up cache
            self._session_cache.pop(session_id, None)
            self._last_sync.pop(session_id, None)

            logger.info(f"Deleted session {session_id}")

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            success = False

        return success

    def _hash_session_data(self, data: Dict[str, Any]) -> str:
        """
        Generate hash of session data for change detection.

        Parameters
        ----------
        data : Dict[str, Any]
            Session data

        Returns
        -------
        str
            Hash of the data
        """
        try:
            # Convert data to JSON string (sorted for consistency)
            json_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.md5(json_str.encode()).hexdigest()
        except Exception:
            # Fallback to timestamp if hashing fails
            return str(datetime.now(timezone.utc).timestamp())

    def _is_session_current(self, session_id: str) -> bool:
        """
        Check if session data in memory matches cached version.

        Parameters
        ----------
        session_id : str
            Session identifier

        Returns
        -------
        bool
            True if session is current (no changes)
        """
        if session_id not in st.session_state:
            return False

        if session_id not in self._session_cache:
            return False

        current_hash = self._hash_session_data(st.session_state[session_id])
        return current_hash == self._session_cache[session_id]


# Global persistence manager instance
_persistence_manager = None


def get_persistence_manager() -> SessionPersistenceManager:
    """Get the global session persistence manager."""
    global _persistence_manager
    if _persistence_manager is None:
        _persistence_manager = SessionPersistenceManager()
    return _persistence_manager


# Public API functions

def load_persistent_session(session_id: str) -> bool:
    """
    Load session from persistent storage.

    Parameters
    ----------
    session_id : str
        Session identifier

    Returns
    -------
    bool
        True if loaded successfully
    """
    return get_persistence_manager().load_session(session_id)


def save_persistent_session(session_id: str, force: bool = False) -> bool:
    """
    Save session to persistent storage.

    Parameters
    ----------
    session_id : str
        Session identifier
    force : bool
        Force save even if unchanged

    Returns
    -------
    bool
        True if saved successfully
    """
    return get_persistence_manager().save_session(session_id, force)


def auto_persist_session(session_id: str) -> bool:
    """
    Automatically persist session if changed.

    Parameters
    ----------
    session_id : str
        Session identifier

    Returns
    -------
    bool
        True if persisted or current
    """
    return get_persistence_manager().auto_save_session(session_id)


def delete_persistent_session(session_id: str) -> bool:
    """
    Delete session from persistent storage.

    Parameters
    ----------
    session_id : str
        Session identifier

    Returns
    -------
    bool
        True if deleted successfully
    """
    return get_persistence_manager().delete_session(session_id)
