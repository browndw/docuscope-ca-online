"""
Table generation functions for corpus analysis.

This module provides functions for generating various types of tables
from corpus data, including tags tables, frequency tables, and metadata tables.
"""

import streamlit as st
from webapp.utilities.state import (
    CorpusKeys, TargetKeys, WarningKeys, SessionKeys
)
from webapp.utilities.core import app_core
from webapp.utilities.corpus import get_corpus_data


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
        tok_pl = get_corpus_data(user_session_id, CorpusKeys.TARGET, TargetKeys.DS_TOKENS)
    except (KeyError, ValueError):
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

    app_core.session_manager.update_session_state(
        user_session_id, SessionKeys.TAGS_TABLE, True
    )
    st.session_state[user_session_id][WarningKeys.TAGS] = None
    st.rerun()


# Note: load_metadata and update_metadata functions have been consolidated
# into webapp.utilities.session.metadata_handlers to eliminate duplication.
# Use the standardized versions from webapp.utilities.session import instead.
