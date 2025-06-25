"""
Modular UI utilities for download pages.

This module provides shared utilities for download pages (13_download_corpus.py and
14_download_tagged_files.py) to ensure consistency and maintainability.
"""

import streamlit as st
from webapp.utilities.state import (
    SessionKeys, WarningKeys
)
from webapp.utilities.analysis import generate_tags_table
from webapp.utilities.ui import sidebar_action_button
from webapp.utilities.session.session_core import safe_session_get


def render_download_page_header(title: str, help_url: str, description: str = None) -> None:
    """
    Render the standard header for download pages.

    Parameters
    ----------
    title : str
        The page title to display
    help_url : str
        URL for the help link in sidebar
    description : str, optional
        Custom description text. If None, uses a default description.
    """
    st.markdown(
        body=f"## {title}",
        help=description or (
            "This page allows you to download files from your corpus analysis "
            "for offline use or integration with other tools."
        )
    )

    st.sidebar.link_button(
        label="Help",
        url=help_url,
        icon=":material/help:"
    )


def render_data_loading_interface(user_session_id: str, session: dict) -> None:
    """
    Render the interface for loading data when tables are not available.

    Parameters
    ----------
    user_session_id : str
        The user session identifier
    session : dict
        The session state dictionary
    """
    st.sidebar.markdown(
        """
        ### Load tables
        Use the button to load corpus tables.
        """,
        help=(
            "For tables to be loaded, you must first process a target corpus "
            "using: **:material/database: Manage Corpus Data**"
        )
    )

    sidebar_action_button(
        button_label="Load Data",
        button_icon=":material/manufacturing:",
        preconditions=[
            safe_session_get(session, SessionKeys.HAS_TARGET, False),
        ],
        action=lambda: generate_tags_table(user_session_id),
        spinner_message="Loading data..."
    )

    # Display warning if exists
    if st.session_state[user_session_id].get(WarningKeys.TAGS):
        msg, icon = st.session_state[user_session_id][WarningKeys.TAGS]
        st.warning(msg, icon=icon)

    st.sidebar.markdown("---")


def render_corpus_selection() -> str:
    """
    Render corpus selection radio buttons.

    Returns
    -------
    str
        Selected corpus type ("Target" or "Reference")
    """
    return st.radio(
        "Choose a corpus",
        ["Target", "Reference"],
        captions=[
            "",
            """You can only download reference
            corpus data if you've processed one.
            """
        ]
    )


def render_data_type_selection() -> str:
    """
    Render data type selection radio buttons.

    Returns
    -------
    str
        Selected data type ("Corpus file only" or "All of the processed data")
    """
    return st.radio(
        "Choose the data to download",
        ["Corpus file only", "All of the processed data"],
        captions=[
            """This is the option you want
            if you're planning to save your corpus
            for future analysis using this tool.
            """,
            """This is the option you want
            if you're planning to explore your data
            outside of the tool, in coding environments
            like R or Python.
            The data include the token file,
            frequency tables, and document-term-matrices.
            """
        ]
    )


def render_format_selection() -> str:
    """
    Render file format selection radio buttons.

    Returns
    -------
    str
        Selected format ("CSV" or "PARQUET")
    """
    return st.radio(
        "Select a file format",
        ["CSV", "PARQUET"],
        horizontal=True
    )


def render_tagset_selection() -> str:
    """
    Render tagset selection for tagged file downloads.

    Returns
    -------
    str
        Selected tagset ("pos" or "ds")
    """
    st.sidebar.markdown("### Tagset to embed")
    download_radio = st.sidebar.radio(
        "Select tagset:",
        ("Parts-of-Speech", "DocuScope"),
        horizontal=True
    )

    return 'pos' if download_radio == 'Parts-of-Speech' else 'ds'


def render_download_button(
    label: str,
    data: bytes,
    filename: str,
    mime_type: str,
    icon: str = ":material/download:",
    message: str = None
) -> None:
    """
    Render a standardized download button in the sidebar.

    Parameters
    ----------
    label : str
        Button label text
    data : bytes
        File data to download
    filename : str
        Name for the downloaded file
    mime_type : str
        MIME type of the file
    icon : str, optional
        Icon for the button
    message : str, optional
        Message to display above the button
    """
    if message:
        st.sidebar.markdown(f"#### {message}")

    st.sidebar.download_button(
        label=label,
        icon=icon,
        data=data,
        file_name=filename,
        mime=mime_type,
    )
    st.sidebar.markdown("---")


def check_reference_corpus_availability(session: dict) -> bool:
    """
    Check if reference corpus is available and show error if not.

    Parameters
    ----------
    session : dict
        The session state dictionary

    Returns
    -------
    bool
        True if reference corpus is available, False otherwise
    """
    if not safe_session_get(session, SessionKeys.HAS_REFERENCE, False):
        st.error(
            """
            It doesn't look like you've loaded a reference corpus yet.
            You can do this from **Manage Corpus Data**.
            """,
            icon=":material/sentiment_stressed:"
        )
        return False
    return True


def get_corpus_data(user_session_id: str, corpus_type: str, data_key: str):
    """
    Get corpus data from session state using the new corpus manager.

    Parameters
    ----------
    user_session_id : str
        The user session identifier
    corpus_type : str
        Either "target" or "reference"
    data_key : str
        The key for the specific data to retrieve

    Returns
    -------
    Data from the corpus session
    """
    from webapp.utilities.corpus import get_corpus_data
    return get_corpus_data(user_session_id, corpus_type.lower(), data_key)
