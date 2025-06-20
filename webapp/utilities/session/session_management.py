# Copyright (C) 2025 David West Brown

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Session management utilities for corpus analysis application.

This module provides functions for initializing, updating, and managing
session state across the application.
"""

import polars as pl
import streamlit as st

from webapp.utilities.data import get_doc_cats
from webapp.config.session_keys import SessionKeys


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
    if "messages" not in st.session_state[session_id]:
        st.session_state[session_id]["messages"] = [
            {"role": "assistant",
             "content": "Hello, what can I do for you today?"}
        ]

    if "plot_intent" not in st.session_state[session_id]:
        st.session_state[session_id]["plot_intent"] = False


def update_session(
        key: str,
        value: any,
        session_id: str
        ) -> None:
    """
    Update a specific key-value pair in the session state
    for a given session ID.

    Parameters
    ----------
    key : str
        The key in the session state to update.
    value : any
        The value to assign to the specified key.
    session_id : str
        The session ID for which the session state is to be updated.

    Returns
    -------
    None
    """
    session = st.session_state[session_id]["session"]
    session = session.to_dict(as_series=False)
    session[key] = value
    df = pl.from_dict(session)
    st.session_state[session_id]["session"] = df


def get_corpus_categories(doc_ids: list, user_session_id: str) -> tuple[list, int]:
    """Get document categories with user-scoped caching."""
    cache_key = f"corpus_categories_{user_session_id}"

    # Check if already cached in user's session
    if cache_key in st.session_state.get(user_session_id, {}):
        return st.session_state[user_session_id][cache_key]

    # Calculate and cache in user session
    doc_cats = get_doc_cats(doc_ids)
    unique_count = len(set(doc_cats)) if doc_cats else 0
    result = (doc_cats, unique_count)

    # Store in user's session state
    if user_session_id not in st.session_state:
        st.session_state[user_session_id] = {}
    st.session_state[user_session_id][cache_key] = result

    return result


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
    if session_id not in st.session_state:
        st.session_state[session_id] = {}
    for key, value in states:
        if key not in st.session_state[session_id]:
            st.session_state[session_id][key] = value


def init_session(session_id: str) -> None:
    """
    Initialize a new session with default values for a specific session ID.

    Parameters
    ----------
    session_id : str
        The session ID for which the session is to be initialized.

    Returns
    -------
    None
    """
    # Create session dictionary with scalar values (matching legacy exactly)
    session = {
        SessionKeys.HAS_TARGET: False,
        SessionKeys.TARGET_DB: '',
        SessionKeys.HAS_META: False,
        SessionKeys.HAS_REFERENCE: False,
        SessionKeys.REFERENCE_DB: '',
        SessionKeys.FREQ_TABLE: False,
        SessionKeys.TAGS_TABLE: False,
        SessionKeys.KEYNESS_TABLE: False,
        SessionKeys.NGRAMS: False,
        SessionKeys.KWIC: False,
        SessionKeys.KEYNESS_PARTS: False,
        SessionKeys.DTM: False,
        SessionKeys.PCA: False,
        SessionKeys.COLLOCATIONS: False,
        SessionKeys.DOC: False,
    }
    df = pl.from_dict(session)

    if session_id not in st.session_state:
        st.session_state[session_id] = {}

    st.session_state[session_id]["session"] = df


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
        session = pl.DataFrame.to_dict(
            st.session_state[user_session_id]["session"], as_series=False
        )
    except KeyError:
        init_session(user_session_id)
        session = pl.DataFrame.to_dict(
            st.session_state[user_session_id]["session"], as_series=False
        )

    return user_session_id, session
