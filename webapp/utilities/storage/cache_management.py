"""
Resources for managing storage and caching in the web application.

This module provides functions for handling persistent storage of messages,
plots, and user logins in a Firestore database. It includes utilities for
generating persistent hashes, adding messages and plots to the database,
and tracking user logins. It also includes a function to count user queries
in the last 24 hours to help manage query limits and quotas.
"""

import hashlib
import streamlit as st
from datetime import datetime, timedelta
from google.cloud import firestore
from google.oauth2 import service_account

from webapp.utilities.core import safe_config_value

# Import centralized logging configuration and logger
from webapp.utilities.configuration.logging_config import get_logger, setup_utility_logging

logger = get_logger()

# Set up logging for storage utilities
setup_utility_logging("storage")

DESKTOP = safe_config_value('desktop_mode', config_type='global')

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
                message: str):
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

    Returns
    -------
        None
    """
    timestamp = datetime.now()
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
        logger(f"Failed to add message to Firestore: {e}")


def add_plot(user_id: str,
             session_id: str,
             assistant_id: int,
             message_idx: int,
             plot_library: str,
             plot_svg: str) -> None:
    """
    Adds a plot arry to the Firestore database.

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

    Returns
    -------
        None
    """
    timestamp = datetime.now()
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
        logger(f"Failed to add plot to Firestore: {e}")


def add_login(user_id: str,
              session_id: str):
    """
    Adds a user login instance to the Firestore database.

    Parameters
    ----------
        user_id: The ID of the user.
        session_id: The ID of the session.

    Returns
    -------
        None
    """
    timestamp = datetime.now()
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
        logger(f"Failed to add login to Firestore: {e}")


def get_query_count(user_id):
    """
    Get the count of user queries in the last 24 hours.

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
        # Only proceed if we're in online mode and have credentials
        if DESKTOP or creds is None:
            return 0

        db = firestore.Client(credentials=creds, project="docuscope-ca-data")
        collection_ref = db.collection("messages")
        timestamp = datetime.now()
        hashed_user_id = persistent_hash(user_id)

        # Calculate the timestamp for 24 hours ago
        last_24_hours = timestamp - timedelta(hours=24)

        # Create a query using the newer filter syntax
        query = (
            collection_ref
            .where(filter=firestore.FieldFilter("user_id", "==", hashed_user_id))
            .where(filter=firestore.FieldFilter("role", "==", "user"))
            .where(filter=firestore.FieldFilter("time_stamp", ">=", last_24_hours))
        )

        docs = query.get()
        count = len(docs)

        return count

    except Exception as e:
        logger(f"Failed to get query count from Firestore: {e}")
        return 0
