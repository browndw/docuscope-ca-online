"""
Common imports and utilities.

Provides consolidated imports and utility functions to reduce
repetitive import statements and improve maintainability.
"""

# Essential external libraries
import streamlit as st

# Core application utilities
from webapp.utilities.core import app_core
from webapp.utilities.error_handling import error_handler

# Session and state management essentials
from webapp.utilities.session import get_or_init_user_session
# Corpus management
from webapp.utilities.corpus import get_corpus_data_manager
# Menu system
from webapp.menu import menu, require_login


def setup_page(title: str, icon: str, widgets: list = None) -> tuple:
    """
    Standard page setup with common configuration.

    Parameters
    ----------
    title : str
        Page title
    icon : str
        Page icon
    widgets : list, optional
        List of persistent widget keys

    Returns
    -------
    tuple
        (user_session_id, session)
    """
    # Set page config
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide"
    )

    # Register widgets if provided
    if widgets:
        app_core.register_page_widgets(widgets)

    # Standard menu and auth
    menu()
    require_login()

    # Initialize session
    user_session_id, session = get_or_init_user_session()

    return user_session_id, session


def ensure_corpus_loaded(
        user_session_id: str,
        corpus_type: str = "target"
) -> bool:
    """
    Check if corpus is loaded and ready with proper error handling.

    Parameters
    ----------
    user_session_id : str
        Session ID
    corpus_type : str
        Corpus type to check

    Returns
    -------
    bool
        True if corpus is ready
    """
    try:
        manager = get_corpus_data_manager(user_session_id, corpus_type)
        return manager.is_ready()
    except Exception as e:
        error_handler.handle_error(
            e, f"Corpus validation for {corpus_type}",
            severity="error", show_user=False
        )
        st.error("Unable to validate corpus data. Please reload your corpus.")
        return False


def safe_corpus_operation(
        operation_name: str
) -> callable:
    """
    Decorator for corpus operations with standardized error handling.

    Parameters
    ----------
    operation_name : str
        Name of the operation for error messages
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Don't log user data issues, only system issues
                if any(indicator in str(e).lower() for indicator in
                       ['encoding', 'format', 'empty', 'malformed']):
                    st.error(
                        f"Data format issue in {operation_name}. "
                        "Please check your corpus data and try again."
                    )
                else:
                    error_handler.handle_error(e, operation_name)
                return None
        return wrapper
    return decorator
