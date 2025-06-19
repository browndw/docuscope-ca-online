"""
Document processing utilities for single document analysis.

This module provides functions for processing individual documents
and generating HTML representations with various tag highlighting.
"""

import streamlit as st
import polars as pl
from webapp.utilities.session import update_session
# TODO: Migrate html_build_pl function from legacy module
# from webapp.utilities_legacy.analysis_legacy import html_build_pl


def generate_document_html(
        user_session_id: str,
        doc_key: str
        ) -> None:
    """
    Process a single document and generate HTML representations.

    Parameters
    ----------
    user_session_id : str
        The session ID for the user.
    doc_key : str
        The document key or identifier.

    Returns
    -------
    None
    """
    # --- Check if target corpus is loaded ---
    session = pl.DataFrame.to_dict(
        st.session_state[user_session_id]["session"], as_series=False
    )
    if session.get('has_target', [False])[0] is False:
        st.session_state[user_session_id]["doc_warning"] = (
            "No target corpus loaded. Please load a document first.",
            ":material/info:"
        )
        return

    # --- Try to get the target tokens table ---
    try:
        tok_pl = st.session_state[user_session_id]["target"]["ds_tokens"]
    except KeyError:
        st.session_state[user_session_id]["doc_warning"] = (
            "No tokens found in the target corpus.",
            ":material/info:"
        )
        return

    if tok_pl is None or getattr(tok_pl, "height", 0) == 0:
        st.session_state[user_session_id]["doc_warning"] = (
            "No tokens found in the target corpus.",
            ":material/info:"
        )
        return

    # --- Generate HTML representations ---
    # TODO: Implement html_build_pl function or import from legacy
    try:
        # doc_pos, doc_simple, doc_ds = html_build_pl(tok_pl, doc_key)
        # Temporary placeholder until html_build_pl is migrated
        st.session_state[user_session_id]["doc_warning"] = (
            "Document HTML generation not yet implemented in new structure.",
            ":material/info:"
        )
        return
    except Exception as e:
        st.session_state[user_session_id]["doc_warning"] = (
            f"Failed to process document: {e}",
            ":material/sentiment_stressed:"
        )
        return

    # TODO: Uncomment when html_build_pl is available
    # --- Save results ---
    # st.session_state[user_session_id]["target"]["doc_pos"] = doc_pos
    # st.session_state[user_session_id]["target"]["doc_simple"] = doc_simple
    # st.session_state[user_session_id]["target"]["doc_ds"] = doc_ds

    # update_session('doc', True, user_session_id)
    # st.session_state[user_session_id]["doc_warning"] = None
    # st.success('Document processed!')
    # st.rerun()
