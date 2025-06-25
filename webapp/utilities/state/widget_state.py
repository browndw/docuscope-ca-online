"""
Widget state management utilities.

This module provides functions for managing Streamlit widget states,
form controls, and UI element state persistence.

Note: For new code, consider using the widget_key_manager module which provides
centralized, session-scoped widget management with better safety guarantees.
"""
import streamlit as st
from typing import Any, Dict


def get_widget_state(key: str, default: Any = None) -> Any:
    """
    Get a widget state value with default fallback.
    
    Parameters
    ----------
    key : str
        The widget state key
    default : Any
        Default value if key doesn't exist
        
    Returns
    -------
    Any
        The widget state value or default
    """
    return st.session_state.get(key, default)


def set_widget_state(key: str, value: Any) -> None:
    """
    Set a widget state value.
    
    Parameters
    ----------
    key : str
        The widget state key
    value : Any
        The value to set
    """
    st.session_state[key] = value


def clear_widget_state(key: str) -> None:
    """
    Clear a specific widget state.
    
    Parameters
    ----------
    key : str
        The widget state key to clear
    """
    if key in st.session_state:
        del st.session_state[key]


def reset_form_state(keys: list) -> None:
    """
    Reset multiple widget states to None.
    
    Parameters
    ----------
    keys : list
        List of widget state keys to reset
    """
    for key in keys:
        if key in st.session_state:
            st.session_state[key] = None


def preserve_widget_state(
    keys: list,
    session_id: str,
    namespace: str = "preserved"
) -> None:
    """
    Preserve widget states in session storage for later restoration.
    
    Parameters
    ----------
    keys : list
        List of widget state keys to preserve
    session_id : str
        Session ID for storage
    namespace : str
        Namespace for preservation
    """
    if session_id not in st.session_state:
        st.session_state[session_id] = {}
    
    preservation_key = f"{namespace}_preserved_state"
    if preservation_key not in st.session_state[session_id]:
        st.session_state[session_id][preservation_key] = {}
    
    for key in keys:
        if key in st.session_state:
            st.session_state[session_id][preservation_key][key] = st.session_state[key]


def get_form_state(keys: list) -> Dict[str, Any]:
    """
    Get current form state for specified keys.
    
    Parameters
    ----------
    keys : list
        List of widget state keys to retrieve
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of current form state
    """
    return {key: st.session_state.get(key) for key in keys}


def validate_required_fields(required_fields: list) -> tuple:
    """
    Validate that required form fields are filled.
    
    Parameters
    ----------
    required_fields : list
        List of required widget state keys
        
    Returns
    -------
    tuple
        (is_valid: bool, missing_fields: list)
    """
    missing_fields = []
    for field in required_fields:
        value = st.session_state.get(field)
        if value is None or value == "" or value == []:
            missing_fields.append(field)
    
    return len(missing_fields) == 0, missing_fields


def safe_clear_widget_state(keys: list) -> None:
    """
    Safely clear widget states without raising errors.
    
    Parameters
    ----------
    keys : list
        List of widget state keys to clear
    """
    for key in keys:
        try:
            if key in st.session_state:
                del st.session_state[key]
        except Exception:
            # Silently continue if there's an issue clearing a specific key
            continue


def safe_clear_widget_states(keys: list) -> Dict[str, bool]:
    """
    Safely clear multiple widget states without raising KeyErrors.

    Parameters
    ----------
    keys : list
        List of session state keys to clear.

    Returns
    -------
    Dict[str, bool]
        Dictionary mapping keys to whether they were successfully deleted.
    """
    results = {}
    for key in keys:
        try:
            if key in st.session_state:
                del st.session_state[key]
                results[key] = True
            else:
                results[key] = False
        except Exception:
            results[key] = False
    return results


class WidgetStateManager:
    """
    Context manager for widget state management.

    Provides a clean interface for managing widget states within
    a specific namespace and handles cleanup automatically.
    """

    def __init__(self, namespace: str = "default"):
        """Initialize widget state manager with namespace."""
        self.namespace = f"{namespace}_"
        self.managed_keys = set()

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
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
