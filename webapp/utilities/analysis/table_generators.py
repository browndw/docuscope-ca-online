"""
Table generation functions for corpus analysis.

This module provides functions for generating various types of tables
from corpus data, including tags tables, frequency tables, and metadata tables.
"""

import streamlit as st
from webapp.config.session_keys import (
    SessionKeys, CorpusKeys, TargetKeys, WarningKeys
)
from webapp.utilities.session import update_session


def generate_tags_table(user_session_id: str) -> None:
    """
    Load tags tables for the target corpus.

    Parameters
    ----------
    user_session_id : str
        The session ID for the user.

    Returns
    -------
    None
    """
    # --- Try to get the target tokens table ---
    try:
        tok_pl = st.session_state[user_session_id][CorpusKeys.TARGET][TargetKeys.DS_TOKENS]
    except KeyError:
        st.session_state[user_session_id][WarningKeys.TAGS] = (
            "Tags table cannot be generated: no tokens found in the target corpus.",
            ":material/info:"
        )
        return

    if tok_pl is None or getattr(tok_pl, "height", 0) == 0:
        st.session_state[user_session_id][WarningKeys.TAGS] = (
            "Tags table cannot be generated: no tokens found in the target corpus.",
            ":material/info:"
        )
        return

    update_session('tags_table', True, user_session_id)
    st.session_state[user_session_id][WarningKeys.TAGS] = None
    st.rerun()


def load_metadata(corpus_type: str, session_id: str) -> dict:
    """
    Load metadata for the specified corpus type from the session state.

    Parameters
    ----------
    corpus_type : str
        The type of corpus for which metadata is to be loaded.
        Should be either 'target' or 'reference'.
    session_id : str
        The session ID for which the metadata is to be loaded.

    Returns
    -------
    dict
        A dictionary containing the metadata for the specified corpus type.
    """
    if corpus_type == "target":
        table_name = SessionKeys.METADATA_TARGET
    elif corpus_type == "reference":
        table_name = SessionKeys.METADATA_REFERENCE
    else:
        raise ValueError("corpus_type must be 'target' or 'reference'")

    metadata = st.session_state[session_id][table_name]
    metadata = metadata.to_dict(as_series=False)
    return metadata


def update_metadata(corpus_type: str, key: str, value, session_id: str) -> None:
    """
    Update metadata for the specified corpus type in the session state.

    Parameters
    ----------
    corpus_type : str
        The type of corpus for which metadata is to be updated.
        Should be either 'target' or 'reference'.
    key : str
        The key in the metadata dictionary to update.
    value : any
        The value to assign to the specified key in the metadata dictionary.
    session_id : str
        The session ID for which the metadata is to be updated.

    Returns
    -------
    None
        The function updates the metadata in the session state.
    """
    if corpus_type == "target":
        table_name = SessionKeys.METADATA_TARGET
    elif corpus_type == "reference":
        table_name = SessionKeys.METADATA_REFERENCE
    else:
        raise ValueError("corpus_type must be 'target' or 'reference'")

    metadata_table = st.session_state[session_id][table_name]
    st.session_state[session_id][table_name] = metadata_table.with_columns(
        **{key: value}
    )
