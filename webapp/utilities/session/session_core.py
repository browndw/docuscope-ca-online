"""
Core session management functions migrated from legacy handlers.

This module contains the core session initialization, update, and management
functions that were originally in handlers.py.
"""

import polars as pl
import streamlit as st

from webapp.config.session_keys import SessionKeys, MetadataKeys
from webapp.utilities.data import get_doc_cats


def init_session(session_id: str) -> None:
    """
    Initialize the session state with default values for a specific session ID.

    Parameters
    ----------
    session_id : str
        The session ID for which the session state is to be initialized.

    Returns
    -------
    None
    """
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
    st.session_state[session_id]["session"] = df


def update_session(key: str, value: any, session_id: str) -> None:
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
    
    # Cache the result in user session
    if user_session_id not in st.session_state:
        st.session_state[user_session_id] = {}
    st.session_state[user_session_id][cache_key] = result
    
    return result


def init_metadata_target(session_id: str) -> None:
    """
    Initialize the metadata for the target corpus in the session state.

    Parameters
    ----------
    session_id : str
        The session ID for which the metadata is to be initialized.

    Returns
    -------
    None
    """
    
    df = st.session_state[session_id]["target"]["ds_tokens"]
    tags_to_check = df.get_column("ds_tag").to_list()
    tags = [
        'Actors', 'Organization', 'Planning', 'Sentiment', 'Signposting', 'Stance'
    ]
    model = 'Common Dictionary' if any(tag in item for item in tags_to_check for tag in tags) else 'Large Dictionary'  # noqa: E501
    ds_tags = df.get_column("ds_tag").unique().to_list()
    tags_pos = df.get_column("pos_tag").unique().to_list()
    if "Untagged" in ds_tags:
        ds_tags.remove("Untagged")
    if "Y" in tags_pos:
        tags_pos.remove("Y")
    temp_metadata_target = {
        MetadataKeys.TOKENS_POS: df.group_by(
            ["doc_id", "pos_id", "pos_tag"]
        ).agg(pl.col("token").str.concat("")).filter(pl.col("pos_tag") != "Y").height,
        MetadataKeys.TOKENS_DS: df.group_by(
            ["doc_id", "ds_id", "ds_tag"]
        ).agg(pl.col("token").str.concat("")).filter(
            ~(pl.col("token").str.contains("^[[[:punct:]] ]+$") & pl.col("ds_tag").str.contains("Untagged"))  # noqa: E501
        ).height,
        MetadataKeys.NDOCS: len(df.get_column("doc_id").unique().to_list()),
        MetadataKeys.TAG_MODEL: model,
        MetadataKeys.TAGS_DS: ds_tags,
        MetadataKeys.TAGS_POS: tags_pos
    }
    metadata_target = pl.from_dict(temp_metadata_target)
    st.session_state[session_id]["metadata_target"] = metadata_target


def init_metadata_reference(session_id: str) -> None:
    """
    Initialize the metadata for the reference corpus in the session state.

    Parameters
    ----------
    session_id : str
        The session ID for which the reference metadata is to be initialized.

    Returns
    -------
    None
    """
    # Import here to avoid circular import
    
    df = st.session_state[session_id]["reference"]["ds_tokens"]
    temp_metadata_reference = {
        MetadataKeys.TOKENS_POS: df.group_by(
            ["doc_id", "pos_id", "pos_tag"]
        ).agg(pl.col("token").str.concat("")).filter(pl.col("pos_tag") != "Y").height,
        MetadataKeys.TOKENS_DS: df.group_by(
            ["doc_id", "ds_id", "ds_tag"]
        ).agg(pl.col("token").str.concat("")).filter(
            ~(pl.col("token").str.contains("^[[[:punct:]] ]+$") & pl.col("ds_tag").str.contains("Untagged"))  # noqa: E501
        ).height,
        MetadataKeys.NDOCS: len(df.get_column("doc_id").unique().to_list()),
    }
    metadata_reference = pl.from_dict(temp_metadata_reference)
    st.session_state[session_id]["metadata_reference"] = metadata_reference
