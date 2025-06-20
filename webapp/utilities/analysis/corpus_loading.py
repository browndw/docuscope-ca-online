"""
Corpus loading utilities for data processing and validation.

This module provides functions for loading, validating, and processing
corpus data from various sources including internal databases and user uploads.
"""

import pathlib
import streamlit as st
from lingua import LanguageDetectorBuilder
from webapp.config.session_keys import SessionKeys

# Ensure project root is in sys.path
project_root = pathlib.Path(__file__).parent.parents[2].resolve()

CORPUS_DIR = project_root.joinpath("webapp/_corpora")

# Warning constants
WARNING_CORRUPT_TARGET = 10
WARNING_CORRUPT_REFERENCE = 11
WARNING_DUPLICATE_REFERENCE = 21
WARNING_EXCLUDED_TARGET = 40
WARNING_EXCLUDED_REFERENCE = 41


@st.cache_resource(show_spinner=False)
def load_detector():
    """
    Load and cache the language detector.

    Returns
    -------
    LanguageDetector
        Configured language detector instance.
    """
    detector = LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()  # noqa: E501
    return detector


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
