"""
Widget state management utilities.

This module provides functions for managing Streamlit widget states,
form controls, and UI element state persistence.
"""
import inspect
import pathlib
import streamlit as st
from typing import Any, Dict, Optional


def get_widget_state(key: str, default: Any = None) -> Any:
    """
    Get widget state from session state with fallback to default.

    Parameters
    ----------
    key : str
        The session state key to retrieve.
    default : Any, optional
        Default value to return if key doesn't exist.

    Returns
    -------
    Any
        The widget state value or default.
    """
    return st.session_state.get(key, default)


def set_widget_state(key: str, value: Any) -> None:
    """
    Set widget state in session state.

    Parameters
    ----------
    key : str
        The session state key to set.
    value : Any
        The value to store.
    """
    st.session_state[key] = value


def clear_widget_state(key: str) -> None:
    """
    Clear specific widget state from session state.

    Parameters
    ----------
    key : str
        The session state key to clear.
    """
    if key in st.session_state:
        del st.session_state[key]


def reset_form_state(form_keys: list) -> None:
    """
    Reset multiple form widget states.

    Parameters
    ----------
    form_keys : list
        List of session state keys to reset.
    """
    for key in form_keys:
        clear_widget_state(key)


def preserve_widget_state(widget_map: Dict[str, Any]) -> None:
    """
    Preserve multiple widget states in session state.

    Parameters
    ----------
    widget_map : Dict[str, Any]
        Dictionary mapping widget keys to their values.
    """
    for key, value in widget_map.items():
        set_widget_state(key, value)


def get_form_state(keys: list) -> Dict[str, Any]:
    """
    Get the current state of multiple form widgets.

    Parameters
    ----------
    keys : list
        List of session state keys to retrieve.

    Returns
    -------
    Dict[str, Any]
        Dictionary of key-value pairs for current widget states.
    """
    return {key: get_widget_state(key) for key in keys}


def validate_required_fields(required_fields: Dict[str, str]) -> Optional[str]:
    """
    Validate that required form fields are filled.

    Parameters
    ----------
    required_fields : Dict[str, str]
        Dictionary mapping field keys to field names for display.

    Returns
    -------
    Optional[str]
        Error message if validation fails, None if all fields are valid.
    """
    missing_fields = []

    for key, field_name in required_fields.items():
        value = get_widget_state(key)

        # Check if field is empty or None
        if value is None or (isinstance(value, str) and not value.strip()):
            missing_fields.append(field_name)
        elif isinstance(value, list) and len(value) == 0:
            missing_fields.append(field_name)

    if missing_fields:
        return f"Please fill in the following required fields: {', '.join(missing_fields)}"

    return None


def persist(
        key: str,
        session_id: str,
        app_name: str = None
        ) -> str:
    """
    Persist a widget state across sessions.
    This function checks if the key exists in the session state,
    and if not, initializes it with None.
    If the key exists, it updates the session state with the current value.

    Parameters
    ----------
    key : str
        The key to persist in the session state.
    session_id : str
        The session ID for the current user session.
    app_name : str, optional
        The name of the application, used to create a unique session state key.
        If not provided, will auto-detect from the calling file's name.

    Returns
    -------
    str
        The key that was persisted.
    """
    if app_name is None:
        # Auto-detect app name from calling file
        import inspect
        import pathlib
        try:
            frame = inspect.currentframe().f_back
            caller_file = frame.f_globals.get('__file__')
            if caller_file:
                app_name = pathlib.Path(caller_file).stem
            else:
                # Fallback to a generic app name if detection fails
                app_name = "unknown_app"
                st.warning(
                    "Could not auto-detect page name for widget persistence. "
                    "Using fallback name. Some widget states may not persist correctly.",
                    icon=":material/warning:"
                )
        except Exception:
            # Fallback to a generic app name if any error occurs
            app_name = "unknown_app"
            st.warning(
                "Could not auto-detect page name for widget persistence. "
                "Using fallback name. Some widget states may not persist correctly.",
                icon=":material/warning:"
            )

    _PERSIST_STATE_KEY = f"{app_name}_PERSIST"
    if _PERSIST_STATE_KEY not in st.session_state[session_id].keys():
        st.session_state[session_id][_PERSIST_STATE_KEY] = {}
        st.session_state[session_id][_PERSIST_STATE_KEY][key] = None

    if key in st.session_state:
        st.session_state[session_id][_PERSIST_STATE_KEY][key] = st.session_state[key]  # noqa: E501

    return key


def load_widget_state(
        session_id: str,
        app_name: str = None
        ) -> None:
    """
    Load persistent widget state from the session state.
    This function checks if the persistent state key exists in the session state,
    and if it does, it loads the values into the current session state.
    If the key does not exist, it initializes the persistent state with None.

    Parameters
    ----------
    session_id : str
        The session ID for the current user session.
    app_name : str, optional
        The name of the application, used to create a unique session state key.
        If not provided, will auto-detect from the calling file's name.

    Returns
    -------
    None
    """
    if app_name is None:
        # Auto-detect app name from calling file
        try:
            frame = inspect.currentframe().f_back
            caller_file = frame.f_globals.get('__file__')
            if caller_file:
                app_name = pathlib.Path(caller_file).stem
            else:
                # Fallback to a generic app name if detection fails
                app_name = "unknown_app"
                st.warning(
                    "Could not auto-detect page name for widget state. "
                    "Using fallback name. Some widget states may not persist correctly.",
                    icon=":material/warning:"
                )
        except Exception:
            # Fallback to a generic app name if any error occurs
            app_name = "unknown_app"
            st.warning(
                "Could not auto-detect page name for widget state. "
                "Using fallback name. Some widget states may not persist correctly.",
                icon=":material/warning:"
            )

    _PERSIST_STATE_KEY = f"{app_name}_PERSIST"
    """Load persistent widget state."""
    try:
        if _PERSIST_STATE_KEY in st.session_state[session_id]:
            for key in st.session_state[session_id][_PERSIST_STATE_KEY]:
                if st.session_state[session_id][_PERSIST_STATE_KEY][key] is not None:  # noqa: E501
                    if key not in st.session_state:
                        st.session_state[key] = st.session_state[session_id][_PERSIST_STATE_KEY][key]  # noqa: E501
    except KeyError:
        # Session state structure might be corrupted or missing
        # Silently skip loading to avoid crashing the app
        pass
    except Exception:
        # Any other unexpected error
        # Silently skip to avoid disrupting the user experience
        pass


class WidgetStateManager:
    """
    Context manager for widget state management.

    Provides a clean interface for managing widget states within
    a specific context or form.
    """

    def __init__(self, namespace: str = ""):
        """
        Initialize the widget state manager.

        Parameters
        ----------
        namespace : str, optional
            Namespace prefix for widget keys to avoid conflicts.
        """
        self.namespace = f"{namespace}_" if namespace else ""
        self._original_states = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Optionally restore original states on exit
        pass

    def get(self, key: str, default: Any = None) -> Any:
        """Get widget state with namespace prefix."""
        return get_widget_state(f"{self.namespace}{key}", default)

    def set(self, key: str, value: Any) -> None:
        """Set widget state with namespace prefix."""
        set_widget_state(f"{self.namespace}{key}", value)

    def clear(self, key: str) -> None:
        """Clear widget state with namespace prefix."""
        clear_widget_state(f"{self.namespace}{key}")

    def reset_all(self, keys: list) -> None:
        """Reset all namespaced widget states."""
        namespaced_keys = [f"{self.namespace}{key}" for key in keys]
        reset_form_state(namespaced_keys)
