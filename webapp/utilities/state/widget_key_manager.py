"""
Centralized Widget Key Manager

This module provides a centralized system for managing widget keys and session state
with user/session scoping, selective persistence, and safe cleanup operations.

Key Features:
- Explicit registration of persistent widget keys
- User/session scoping for all widget keys
- Safe cleanup and lifecycle management
- Utility functions for safe get/set/delete operations
- Batch cleanup operations
"""

import streamlit as st
from typing import Any, Dict, List, Set

# Import centralized logging configuration
from webapp.utilities.configuration.logging_config import get_logger, setup_utility_logging

# Set up logging
logger = get_logger()


class WidgetKeyManager:
    """
    Centralized manager for widget keys and session state with user/session scoping.

    This class ensures:
    - All widget keys are properly scoped to user/session
    - Only explicitly registered keys are persisted
    - Safe cleanup operations that prevent KeyErrors
    - Consistent lifecycle management
    """

    def __init__(self):
        self._persistent_keys: Set[str] = set()
        self._session_prefix = self._get_session_prefix()

    def _get_session_prefix(self) -> str:
        """Get the session prefix for scoping keys."""
        # Use session_state's built-in session ID if available
        if hasattr(st, 'session_state') and hasattr(st.session_state, '_session_id'):
            return f"session_{st.session_state._session_id}"

        # Fallback to a simple session identifier
        if 'session_id' not in st.session_state:
            import uuid
            st.session_state.session_id = str(uuid.uuid4())[:8]

        return f"session_{st.session_state.session_id}"

    def register_persistent_key(self, key: str) -> None:
        """
        Register a widget key for persistence.

        Args:
            key: The base widget key (without session scoping)
        """
        self._persistent_keys.add(key)

    def register_persistent_keys(self, keys: List[str]) -> None:
        """
        Register multiple widget keys for persistence.

        Args:
            keys: List of base widget keys (without session scoping)
        """
        for key in keys:
            self.register_persistent_key(key)

    def is_persistent(self, key: str) -> bool:
        """
        Check if a key is registered for persistence.

        Args:
            key: The base widget key (without session scoping)

        Returns:
            True if the key is registered for persistence
        """
        return key in self._persistent_keys

    def get_scoped_key(self, key: str) -> str:
        """
        Get the session-scoped version of a widget key.

        Args:
            key: The base widget key

        Returns:
            Session-scoped key
        """
        return f"{self._session_prefix}_{key}"

    def set_widget_state(self, key: str, value: Any, persist: bool = False) -> None:
        """
        Set a widget state value with optional persistence registration.

        Args:
            key: The base widget key
            value: The value to set
            persist: Whether to register this key for persistence
        """
        if persist:
            self.register_persistent_key(key)

        scoped_key = self.get_scoped_key(key)
        st.session_state[scoped_key] = value

    def get_widget_state(self, key: str, default: Any = None) -> Any:
        """
        Get a widget state value safely.

        Args:
            key: The base widget key
            default: Default value if key doesn't exist

        Returns:
            The widget state value or default
        """
        scoped_key = self.get_scoped_key(key)
        return st.session_state.get(scoped_key, default)

    def has_widget_state(self, key: str) -> bool:
        """
        Check if a widget state exists.

        Args:
            key: The base widget key

        Returns:
            True if the widget state exists
        """
        scoped_key = self.get_scoped_key(key)
        return scoped_key in st.session_state

    def delete_widget_state(self, key: str, safe: bool = True) -> bool:
        """
        Delete a widget state safely.

        Args:
            key: The base widget key
            safe: If True, doesn't raise KeyError if key doesn't exist

        Returns:
            True if key was deleted, False if it didn't exist
        """
        scoped_key = self.get_scoped_key(key)

        if safe:
            if scoped_key in st.session_state:
                del st.session_state[scoped_key]
                logger.debug(f"Deleted widget state: {scoped_key}")
                return True
            return False
        else:
            del st.session_state[scoped_key]
            logger.debug(f"Deleted widget state: {scoped_key}")
            return True

    def cleanup_widget_states(self, keys: List[str], safe: bool = True) -> Dict[str, bool]:
        """
        Batch cleanup of widget states.

        Args:
            keys: List of base widget keys to delete
            safe: If True, doesn't raise KeyError if keys don't exist

        Returns:
            Dict mapping keys to whether they were successfully deleted
        """
        results = {}
        for key in keys:
            try:
                results[key] = self.delete_widget_state(key, safe=safe)
            except KeyError as e:
                logger.error(f"Failed to delete widget state {key}: {e}")
                results[key] = False

        return results

    def cleanup_all_session_widgets(self) -> int:
        """
        Clean up all widget states for the current session.

        Returns:
            Number of keys deleted
        """
        session_keys = [k for k in st.session_state.keys()
                        if k.startswith(self._session_prefix)]

        deleted_count = 0
        for key in session_keys:
            try:
                del st.session_state[key]
                deleted_count += 1
            except KeyError:
                logger.warning(f"Failed to delete session widget: {key}")
        return deleted_count

    def get_persistent_widget_states(self) -> Dict[str, Any]:
        """
        Get all persistent widget states for the current session.

        Returns:
            Dict mapping base keys to their values for persistent widgets
        """
        persistent_states = {}

        for base_key in self._persistent_keys:
            scoped_key = self.get_scoped_key(base_key)
            if scoped_key in st.session_state:
                persistent_states[base_key] = st.session_state[scoped_key]

        return persistent_states

    def get_all_session_widget_states(self) -> Dict[str, Any]:
        """
        Get all widget states for the current session.

        Returns:
            Dict mapping scoped keys to their values
        """
        return {k: v for k, v in st.session_state.items()
                if k.startswith(self._session_prefix)}

    def get_widget_statistics(self) -> Dict[str, int]:
        """
        Get statistics about widget states.

        Returns:
            Dict with counts of total, persistent, and non-persistent widgets
        """
        all_session_widgets = self.get_all_session_widget_states()
        persistent_widgets = self.get_persistent_widget_states()

        return {
            'total_session_widgets': len(all_session_widgets),
            'persistent_widgets': len(persistent_widgets),
            'non_persistent_widgets': len(all_session_widgets) - len(persistent_widgets),
            'registered_persistent_keys': len(self._persistent_keys)
        }


# Global instance for use throughout the application
widget_key_manager = WidgetKeyManager()


# Convenience functions for easier migration from existing code
def register_persistent_widget(key: str) -> None:
    """Register a widget key for persistence (convenience function)."""
    widget_key_manager.register_persistent_key(key)


def register_persistent_widgets(keys: List[str]) -> None:
    """Register multiple widget keys for persistence (convenience function)."""
    widget_key_manager.register_persistent_keys(keys)


def set_widget_state(key: str, value: Any, persist: bool = False) -> None:
    """Set widget state with optional persistence (convenience function)."""
    widget_key_manager.set_widget_state(key, value, persist=persist)


def get_widget_state(key: str, default: Any = None) -> Any:
    """Get widget state safely (convenience function)."""
    return widget_key_manager.get_widget_state(key, default)


def has_widget_state(key: str) -> bool:
    """Check if widget state exists (convenience function)."""
    return widget_key_manager.has_widget_state(key)


def delete_widget_state(key: str, safe: bool = True) -> bool:
    """Delete widget state safely (convenience function)."""
    return widget_key_manager.delete_widget_state(key, safe=safe)


def cleanup_widget_states(keys: List[str], safe: bool = True) -> Dict[str, bool]:
    """Batch cleanup of widget states (convenience function)."""
    return widget_key_manager.cleanup_widget_states(keys, safe=safe)


# Set up logging for state management utilities
setup_utility_logging("state")


def create_persist_function(user_session_id: str = None):
    """
    Create a persist function replacement that handles both key generation and persistence.
    This provides backward compatibility with the legacy persist() function pattern.

    This function replicates the full behavior of the legacy persist() function:
    - Registers keys for persistence
    - Stores current widget values in session state
    - Restores persisted values on subsequent calls
    - Auto-detects page name for persistence namespace

    Parameters
    ----------
    user_session_id : str, optional
        The user session ID for scoping

    Returns
    -------
    callable
        Function that generates persistent widget keys with automatic registration
        and value persistence/restoration
    """
    def persist_func(key: str, session_id: str):
        # Register the key for persistence
        widget_key_manager.register_persistent_key(key)

        # Get the scoped key for the widget
        scoped_key = widget_key_manager.get_scoped_key(key)

        # Auto-detect app/page name for persistence namespace
        import inspect
        import pathlib
        app_name = None

        try:
            frame = inspect.currentframe().f_back.f_back  # Go up two levels from the caller
            caller_file = None

            # Walk up the call stack to find the first file that looks like a page
            while frame:
                file_path = frame.f_globals.get('__file__')
                if file_path:
                    path_obj = pathlib.Path(file_path)
                    filename = path_obj.stem

                    # If it's in pages directory or starts with a number, it's likely a page
                    if (
                        'pages' in path_obj.parts or
                        (filename and filename[0].isdigit()) or
                        path_obj.parent.name == 'pages'
                    ):
                        caller_file = file_path
                        break

                frame = frame.f_back

            # If we found a page file, use it; otherwise use the immediate caller
            if caller_file:
                app_name = pathlib.Path(caller_file).stem
            else:
                # Fallback to immediate caller
                frame = inspect.currentframe().f_back.f_back
                if frame and frame.f_globals.get('__file__'):
                    app_name = pathlib.Path(frame.f_globals['__file__']).stem
                else:
                    app_name = "unknown_app"
        except Exception:
            app_name = "unknown_app"

        # Create persistence namespace
        _PERSIST_STATE_KEY = f"{app_name}_PERSIST"

        # Ensure session structure exists
        if session_id not in st.session_state:
            st.session_state[session_id] = {}

        # Ensure persistence storage exists
        if _PERSIST_STATE_KEY not in st.session_state[session_id]:
            st.session_state[session_id][_PERSIST_STATE_KEY] = {}

        # Initialize key if not exists
        if key not in st.session_state[session_id][_PERSIST_STATE_KEY]:
            st.session_state[session_id][_PERSIST_STATE_KEY][key] = None

        # Get persisted value
        persisted_value = st.session_state[session_id][_PERSIST_STATE_KEY][key]

        # If there's a persisted value and no current widget value, restore it
        if persisted_value is not None and scoped_key not in st.session_state:
            st.session_state[scoped_key] = persisted_value

        # If widget has a current value, store it for persistence
        if scoped_key in st.session_state:
            persistence_store = st.session_state[session_id][_PERSIST_STATE_KEY]
            persistence_store[key] = st.session_state[scoped_key]

        return scoped_key

    return persist_func
