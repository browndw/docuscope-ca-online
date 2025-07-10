"""
Resources for storage and caching in the web application.

This module provides functions for handling persistent storage of messages,
plots, and user logins in a Firestore database. It includes utilities for
generating persistent hashes, adding messages and plots to the database,
and tracking user logins. It also includes a function to count user queries
in the last 24 hours to help manage query limits and quotas.
"""

import hashlib
import streamlit as st
from datetime import datetime, timezone
from google.cloud import firestore
from google.oauth2 import service_account

from webapp.config.unified import get_config
from webapp.utilities.storage.backend_factory import get_session_backend

# Import centralized logging configuration and logger
from webapp.utilities.configuration.logging_config import (
    get_logger, setup_utility_logging
)

logger = get_logger()

# Set up logging for storage utilities
setup_utility_logging("storage")

# Use fallback-aware config to prevent initialization with invalid secrets
DESKTOP = get_config('desktop_mode', 'global', True)


def should_store_to_firestore(enable_firestore: bool = None) -> bool:
    """
    Check if data should be stored to Firestore for research purposes.

    Parameters
    ----------
    enable_firestore : bool, optional
        Override for Firestore storage. If None, uses configuration with fallback logic.

    Returns
    -------
    bool
        True if Firestore storage is enabled
    """
    if enable_firestore is not None:
        return enable_firestore

    # Import here to avoid circular imports
    try:
        from webapp.config.config_utils import get_runtime_setting
        return get_runtime_setting('cache_mode', False, 'cache')
    except ImportError:
        # Fallback to static config if runtime config not available
        return get_config('cache_mode', 'cache', False)


if DESKTOP is False:
    # Set up the Google Cloud Firestore credentials
    try:
        key_dict = st.secrets["firestore"]["key_dict"]
        creds = service_account.Credentials.from_service_account_info(key_dict)
    except FileNotFoundError:
        creds = None


# Functions for handling states and files.
def persistent_hash(input_string,
                    algorithm='sha256'):
    """
    Generates a persistent hash of a string using the specified algorithm.

    Parameters
    ----------
        input_string: The string to hash.
        algorithm: The hashing algorithm to use (e.g., 'md5', 'sha256').

    Returns
    -------
        A hexadecimal string representing the hash of the input string.
    """
    hasher = hashlib.new(algorithm)
    hasher.update(input_string.encode('utf-8'))
    return hasher.hexdigest()


def add_message(user_id: str,
                session_id: str,
                assistant_id: int,
                role: str,
                message_idx: int,
                message: str,
                enable_firestore: bool = None):
    """
    Adds a message to the Firestore database.

    Parameters
    ----------
        user_id: str
            The ID of the user sending or receiving the message.
        session_id: str
            The ID of the session associated with the message.
        assistant_id: int
            The ID of the assistant involved in the conversation.
        role: str
            The role of the message sender ('user' or 'assistant').
        message: str
            The content of the message.
        enable_firestore: bool, optional
            Whether to store to Firestore. If None, uses configuration.

    Returns
    -------
        None
    """
    # Check if Firestore storage is enabled
    if not should_store_to_firestore(enable_firestore):
        logger.debug("Firestore storage disabled - skipping message storage")
        return

    if DESKTOP:
        logger.debug("Desktop mode - skipping Firestore message storage")
        return

    timestamp = datetime.now(timezone.utc)
    user_id = persistent_hash(user_id)

    # Generate a unique document ID based on user_id, timestamp, and role
    # Note: The role is converted to an integer (0 or 1) for the ID
    # to ensure uniqueness for nearly simultaneous assignment to Firestore
    if role == "user":
        type = 0
    else:
        type = 1
    doc_id = (
        user_id[:12] +
        "-" +
        timestamp.strftime("%Y%m%d%H%M%S") +
        "-" +
        str(type)
        )

    # Create a Firestore client and add the message
    try:
        db = firestore.Client(credentials=creds, project="docuscope-ca-data")
        doc_ref = db.collection('messages').document(doc_id)
        doc_ref.set({
            'user_id': user_id,
            'session_id': session_id,
            'time_stamp': timestamp,
            'assistant_id': assistant_id,
            'role': role,
            'message_idx': message_idx,
            'message': message
        })
    except Exception as e:
        logger.error(f"Failed to add message to Firestore: {e}")


def add_plot(user_id: str,
             session_id: str,
             assistant_id: int,
             message_idx: int,
             plot_library: str,
             plot_svg: str,
             enable_firestore: bool = None) -> None:
    """
    Adds a plot array to the Firestore database.

    Parameters
    ----------
        user_id: str
            The ID of the user sending or receiving the message.
        session_id: str
            The ID of the session associated with the message.
        assistant_id: int
            The ID of the assistant involved in the conversation.
        message_idx: int
            Index of the message in the conversation.
        plot_library: str
            The plotting library used.
        plot_svg: str
            The SVG content of the plot.
        enable_firestore: bool, optional
            Whether to store to Firestore. If None, uses configuration.

    Returns
    -------
        None
    """
    # Check if Firestore storage is enabled
    if not should_store_to_firestore(enable_firestore):
        return

    if DESKTOP:
        return

    timestamp = datetime.now(timezone.utc)
    user_id = persistent_hash(user_id)
    type = 1

    # Generate a unique document ID based on user_id, timestamp, and role
    # Note: The role is converted to an integer (0 or 1) for the ID
    # to ensure uniqueness for nearly simultaneous assignment to Firestore

    doc_id = (
        user_id[:12] +
        "-" +
        timestamp.strftime("%Y%m%d%H%M%S") +
        "-" +
        str(type)
        )

    # Create a Firestore client and add the message
    try:
        db = firestore.Client(credentials=creds, project="docuscope-ca-data")
        doc_ref = db.collection('plots').document(doc_id)
        doc_ref.set({
            'user_id': user_id,
            'session_id': session_id,
            'time_stamp': timestamp,
            'assistant_id': assistant_id,
            'message_idx': message_idx,
            'plot_library': plot_library,
            'plot_svg': plot_svg
        })
    except Exception as e:
        logger.error(f"Failed to add plot to Firestore: {e}")


def add_login(user_id: str,
              session_id: str,
              enable_firestore: bool = None):
    """
    Adds a user login instance to the Firestore database.

    Parameters
    ----------
        user_id: The ID of the user.
        session_id: The ID of the session.
        enable_firestore: bool, optional
            Whether to store to Firestore. If None, uses configuration.

    Returns
    -------
        None
    """
    # Check if Firestore storage is enabled
    if not should_store_to_firestore(enable_firestore):
        logger.debug("Firestore storage disabled - skipping login storage")
        return

    if DESKTOP:
        logger.debug("Desktop mode - skipping Firestore login storage")
        return

    timestamp = datetime.now(timezone.utc)
    from_cmu = user_id.endswith(".cmu.edu")
    user_id = persistent_hash(user_id)

    doc_id = (
        user_id[:12] +
        "-" +
        timestamp.strftime("%Y%m%d%H%M%S")
        )
    try:
        db = firestore.Client(credentials=creds, project="docuscope-ca-data")
        doc_ref = db.collection('users').document(doc_id)
        doc_ref.set({
            'user_id': user_id,
            'from_cmu': from_cmu,
            'session_id': session_id,
            'time_stamp': timestamp
        })
    except Exception as e:
        logger.error(f"Failed to add login to Firestore: {e}")


def get_query_count(user_id):
    """
    Get the count of user queries in the last 24 hours from SQLite.

    This function now uses local SQLite storage for instant quota checking
    instead of querying Firestore, providing better performance and
    eliminating API costs for quota management.

    Parameters
    ----------
    user_id : str
        The user ID to check queries for

    Returns
    -------
    int
        Number of user queries in the last 24 hours
    """
    try:
        # Use SQLite for quota checking (fast, local, always available)

        backend = get_session_backend()
        count = backend.get_user_query_count_24h(user_id)
        return count

    except Exception as e:
        logger.error(f"Failed to get query count from SQLite: {e}")
        return 0  # Fail-safe: allow usage when quota check fails


# Enhanced functions that use SQLite + optional Firestore

def log_user_query_local(user_id: str, session_id: str, assistant_type: str = None,
                         message_content: str = None) -> bool:
    """
    Log user query to local SQLite for quota tracking.

    This function should be called whenever a user makes a query to any AI assistant.
    It provides instant quota tracking without external API dependencies.

    Parameters
    ----------
    user_id : str
        The user ID making the query
    session_id : str
        The session ID for the query
    assistant_type : str, optional
        Type of assistant ('plotbot', 'pandasai', etc.)
    message_content : str, optional
        The actual query content

    Returns
    -------
    bool
        True if logged successfully
    """
    try:

        backend = get_session_backend()
        success = backend.log_user_query(
            user_id, session_id, assistant_type, message_content
        )

        if success:
            logger.debug(f"Logged user query for {user_id} in session {session_id}")

        return success

    except Exception as e:
        logger.error(f"Failed to log user query locally: {e}")
        return False


def add_message_enhanced(user_id: str, session_id: str, assistant_id: int,
                         role: str, message_idx: int, message: str,
                         enable_firestore: bool = None):
    """
    Enhanced message logging that uses SQLite + optional Firestore.

    This function logs to SQLite for quota tracking and optionally to Firestore
    for research data collection if enabled in configuration.

    Parameters
    ----------
    user_id : str
        The ID of the user sending or receiving the message
    session_id : str
        The ID of the session associated with the message
    assistant_id : int
        The ID of the assistant involved in the conversation
    role : str
        The role of the message sender ('user' or 'assistant')
    message_idx : int
        Index of the message in the conversation
    message : str
        The content of the message
    enable_firestore : bool, optional
        Whether to store to Firestore. If None, uses configuration.

    Returns
    -------
    None
    """
    # Always log to SQLite for quota tracking (if it's a user query)
    if role == "user":
        assistant_type = f"assistant_{assistant_id}"  # Map assistant_id to type
        log_user_query_local(user_id, session_id, assistant_type, message)

    # Optionally log to Firestore for research data
    firestore_enabled = get_config('enabled', 'firestore', False)
    if (enable_firestore or firestore_enabled) and not DESKTOP and creds is not None:
        try:
            # Use the original Firestore logging function
            add_message(
                user_id, session_id, assistant_id, role, message_idx, message,
                enable_firestore
            )
        except Exception as e:
            logger.warning(f"Failed to log to Firestore (research data): {e}")
            # Don't fail the operation if Firestore logging fails


def get_session_analytics() -> dict:
    """
    Get session and usage analytics from SQLite.

    Returns comprehensive statistics about application usage
    for monitoring and capacity planning.

    Returns
    -------
    dict
        Analytics data including session counts, query patterns, etc.
    """
    try:

        backend = get_session_backend()
        stats = backend.get_session_stats()

        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'database_stats': stats,
            'firestore_enabled': get_config('enabled', 'firestore', False),
            'desktop_mode': DESKTOP
        }

    except Exception as e:
        logger.error(f"Failed to get session analytics: {e}")
        return {}
