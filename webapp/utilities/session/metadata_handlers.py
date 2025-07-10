"""
Metadata handling utilities for corpus analysis.

This module provides functions for initializing, loading, and updating
metadata for target and reference corpora.
"""

import streamlit as st
import polars as pl
from webapp.utilities.session.session_core import (
    get_corpus_categories
)
from webapp.utilities.state import (
    SessionKeys, CorpusKeys,
    MetadataKeys
)


# Constants for metadata validation
MIN_CATEGORIES = 2
MAX_CATEGORIES = 20


# init_metadata_target function moved to session_core.py to eliminate duplication


# init_metadata_reference function moved to session_core.py to eliminate duplication


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

    metadata_raw = st.session_state[session_id][table_name]

    # Handle both DataFrame and dict cases (unified session management)
    if hasattr(metadata_raw, 'to_dict') and hasattr(metadata_raw, 'columns'):
        # It's a Polars DataFrame (has both to_dict and columns attributes)
        metadata = metadata_raw.to_dict(as_series=False)
    else:
        # It's already a dictionary or other object
        metadata = metadata_raw if isinstance(metadata_raw, dict) else {}

    return metadata


def update_metadata(
        corpus_type: str,
        key: str,
        value: any,
        session_id: str
        ) -> None:
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

    metadata_raw = st.session_state[session_id][table_name]

    # Handle both DataFrame and dict cases (unified session management)
    if hasattr(metadata_raw, 'to_dict') and hasattr(metadata_raw, 'columns'):
        # It's a Polars DataFrame (has both to_dict and columns attributes)
        metadata = metadata_raw.to_dict(as_series=False)
        was_dataframe = True
    else:
        # It's already a dictionary or other object
        metadata = metadata_raw.copy() if isinstance(metadata_raw, dict) else {}
        was_dataframe = False

    # Update the metadata dictionary
    if key == MetadataKeys.DOCCATS:
        metadata[MetadataKeys.DOCCATS] = [{'cats': value}]
    elif key == MetadataKeys.COLLOCATIONS:
        # Store collocation parameters dictionary directly
        metadata[MetadataKeys.COLLOCATIONS] = value
    elif key == MetadataKeys.KEYNESS_PARTS:
        metadata[MetadataKeys.KEYNESS_PARTS] = {'temp': [value]}
    elif key == MetadataKeys.VARIANCE:
        metadata[MetadataKeys.VARIANCE] = [{'temp': value}]
    else:
        metadata[key] = value

    # Store back in the same format it was in originally
    if was_dataframe:
        # Convert back to DataFrame and store
        df = pl.from_dict(metadata, strict=False)
        st.session_state[session_id][table_name] = df
    else:
        # Store as dictionary
        st.session_state[session_id][table_name] = metadata


def handle_target_metadata_processing(metadata_target: dict, user_session_id: str) -> None:
    """Handle target corpus metadata processing with validation."""
    st.sidebar.markdown('### Target corpus metadata:')
    load_cats = st.sidebar.radio(
        "Do you have categories in your file names to process?",
        ("No", "Yes"),
        horizontal=True,
        help=(
            "Metadata can be encoded into your file names, "
            "which can be used for further analysis. "
            "The tool can detect information that comes before "
            "the first underscore in the file name, and will "
            "use that information to assign categories to your "
            "documents. For example, if your file names are "
            "`cat1_doc1.txt`, `cat2_doc2.txt`, etc., "
            "the tool will assign `cat1` and `cat2` as categories. "
        )
    )

    if load_cats == 'Yes':
        if st.sidebar.button(
            label="Process Document Metadata",
            icon=":material/manufacturing:"
        ):
            with st.spinner('Processing metadata...'):
                try:
                    docids_data = metadata_target.get(MetadataKeys.DOCIDS, [{}])
                    if isinstance(docids_data, list) and len(docids_data) > 0:
                        doc_ids = docids_data[0].get('ids', [])
                    else:
                        doc_ids = []

                    if not doc_ids:
                        st.sidebar.error("No document IDs found to process.")
                        return

                    # Use cached function for efficiency
                    doc_cats, unique_count = get_corpus_categories(doc_ids, user_session_id)

                    if MIN_CATEGORIES <= unique_count <= MAX_CATEGORIES:
                        update_metadata(
                            CorpusKeys.TARGET,
                            MetadataKeys.DOCCATS,
                            doc_cats,
                            user_session_id)
                        # Use the session update function to ensure consistent format
                        from webapp.utilities.session.session_core import update_session
                        update_session(SessionKeys.HAS_META, True, user_session_id)
                        st.sidebar.success(
                            f"Successfully processed {unique_count} document categories!"
                        )
                        st.rerun()
                    else:
                        st.sidebar.error(
                            f"Found {unique_count} categories. "
                            f"Please ensure you have between {MIN_CATEGORIES} "
                            f"and {MAX_CATEGORIES} categories."
                        )
                except Exception as e:
                    st.sidebar.error(f"Error processing metadata: {str(e)}")
                    st.sidebar.exception(e)

    st.sidebar.markdown("---")
