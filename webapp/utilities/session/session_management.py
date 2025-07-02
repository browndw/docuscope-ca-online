"""
Session management utilities for corpus analysis application.

This module provides functions for initializing, updating, and managing
session state across the application.
"""

import streamlit as st
from webapp.utilities.session.session_persistence import (
    load_persistent_session,
    auto_persist_session
)


def ensure_session_loaded(session_id: str) -> bool:
    """
    Ensure session is loaded from persistent storage if it exists.

    Parameters
    ----------
    session_id : str
        The session ID to ensure is loaded

    Returns
    -------
    bool
        True if session is available (loaded or already in memory)
    """
    # If session already exists in memory, we're good
    if session_id in st.session_state:
        return True

    # Try to load from persistent storage
    if load_persistent_session(session_id):
        return True

    # Session doesn't exist anywhere
    return False


def persist_session_changes(session_id: str) -> bool:
    """
    Persist session changes to storage if needed.

    Parameters
    ----------
    session_id : str
        The session ID to persist

    Returns
    -------
    bool
        True if session was persisted or is current
    """
    if session_id in st.session_state:
        return auto_persist_session(session_id)
    return False


def init_ai_assist(
        session_id: str
        ) -> None:
    """
    Initialize AI assistant-related session state for a specific session ID.

    Parameters
    ----------
    session_id : str
        The session ID for which the AI assistant state is to be initialized.

    Returns
    -------
    None
    """
    # Ensure session is loaded before initializing
    ensure_session_loaded(session_id)

    # Initialize if not already present
    if session_id not in st.session_state:
        st.session_state[session_id] = {}

    if "messages" not in st.session_state[session_id]:
        st.session_state[session_id]["messages"] = [
            {"role": "assistant",
             "content": "Hello, what can I do for you today?"}
        ]

    if "plot_intent" not in st.session_state[session_id]:
        st.session_state[session_id]["plot_intent"] = False

    # Persist the initialization
    persist_session_changes(session_id)


# update_session function moved to session_core.py to eliminate duplication


# get_corpus_categories function moved to session_core.py to eliminate duplication


def validate_session_state(user_session_id: str) -> bool:
    """
    Validate that session state contains required data structures.
    This function matches the legacy behavior and checks for basic requirements.

    Parameters
    ----------
    user_session_id : str
        The session ID to validate.

    Returns
    -------
    bool
        True if session state is valid, False otherwise.
    """
    try:
        # Check that session exists (matching legacy behavior)
        if user_session_id not in st.session_state:
            return False

        # Check for basic metadata structure (like legacy version)
        # This is the minimum requirement for most UI functions
        required_keys = ['metadata_target']
        for key in required_keys:
            if key not in st.session_state[user_session_id]:
                return False

        return True

    except Exception:
        return False


def validate_session_structure(user_session_id: str, required_keys: list) -> bool:
    """
    Lightweight validation for session state structure.

    Parameters
    ----------
    user_session_id : str
        The session ID to validate
    required_keys : list
        List of keys that must exist in the session

    Returns
    -------
    bool
        True if session structure is valid, False otherwise
    """
    try:
        if user_session_id not in st.session_state:
            return False

        session_data = st.session_state[user_session_id]
        return all(key in session_data for key in required_keys)
    except Exception:
        return False


def ensure_session_key(user_session_id: str, key: str, default_value=None) -> None:
    """
    Ensure a session key exists with a default value.

    Parameters
    ----------
    user_session_id : str
        The session ID
    key : str
        The key to ensure exists
    default_value : any, optional
        Default value if key doesn't exist
    """
    try:
        if user_session_id not in st.session_state:
            st.session_state[user_session_id] = {}

        if key not in st.session_state[user_session_id]:
            st.session_state[user_session_id][key] = default_value
    except Exception:
        # Silently handle errors to avoid disrupting the application
        pass


def get_session_value(user_session_id: str, key: str, default=None):
    """
    Safely get a value from session state.

    Parameters
    ----------
    user_session_id : str
        The session ID
    key : str
        The key to retrieve
    default : any, optional
        Default value if key doesn't exist

    Returns
    -------
    any
        The session value or default
    """
    try:
        return st.session_state.get(user_session_id, {}).get(key, default)
    except Exception:
        return default


def generate_temp(states: dict, session_id: str) -> None:
    """
    Initialize session states with the given states for a specific session ID.

    Parameters
    ----------
    states : dict
        A dictionary of key-value pairs representing
        the states to be initialized.
    session_id : str
        The session ID for which the states are to be initialized.

    Returns
    -------
    None
    """
    # Ensure session is loaded first
    ensure_session_loaded(session_id)

    if session_id not in st.session_state:
        st.session_state[session_id] = {}

    changes_made = False
    for key, value in states:
        if key not in st.session_state[session_id]:
            st.session_state[session_id][key] = value
            changes_made = True

    # Persist changes if any were made
    if changes_made:
        persist_session_changes(session_id)


# init_session function moved to session_core.py to eliminate duplication


def get_or_init_user_session() -> tuple[str, dict]:
    """
    Ensure a user session exists and return its ID and session dict.

    Returns
    -------
    tuple[str, dict]
        The user session ID and the session dictionary.
    """
    user_session = st.runtime.scriptrunner_utils.script_run_context.get_script_run_ctx()
    user_session_id = user_session.session_id

    if user_session_id not in st.session_state:
        st.session_state[user_session_id] = {}

    try:
        session_raw = st.session_state[user_session_id]["session"]
        # Handle both DataFrame and dict cases (unified session management)
        if hasattr(session_raw, 'to_dict') and hasattr(session_raw, 'columns'):
            # It's a Polars DataFrame (has both to_dict and columns attributes)
            session = session_raw.to_dict(as_series=False)
        else:
            # It's already a dictionary or other object
            session = session_raw if isinstance(session_raw, dict) else {}
    except KeyError:
        from webapp.utilities.session.session_core import init_session
        init_session(user_session_id)
        session_raw = st.session_state[user_session_id]["session"]
        if hasattr(session_raw, 'to_dict') and hasattr(session_raw, 'columns'):
            session = session_raw.to_dict(as_series=False)
        else:
            session = session_raw if isinstance(session_raw, dict) else {}

    return user_session_id, session
