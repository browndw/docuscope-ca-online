"""
This module contains the core session initialization, update, and management.
"""

import polars as pl
import streamlit as st

from webapp.utilities.state import SessionKeys, MetadataKeys
from webapp.utilities.common import get_doc_cats
from webapp.utilities.session.session_persistence import (
    load_persistent_session,
    auto_persist_session
)


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
    # First try to load from persistent storage
    if load_persistent_session(session_id):
        return  # Session was loaded from storage

    # If no existing session, create new one
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

    # Initialize session state
    if session_id not in st.session_state:
        st.session_state[session_id] = {}
    st.session_state[session_id]["session"] = df

    # Persist the initial session
    auto_persist_session(session_id)


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
    session_raw = st.session_state[session_id]["session"]

    # Handle both DataFrame and dict cases (unified session management)
    if hasattr(session_raw, 'to_dict') and hasattr(session_raw, 'columns'):
        # It's a Polars DataFrame (has both to_dict and columns attributes)
        session = session_raw.to_dict(as_series=False)
        was_dataframe = True
    else:
        # It's already a dictionary or other object
        session = session_raw.copy() if isinstance(session_raw, dict) else {}
        was_dataframe = False

    # Update the session dictionary
    session[key] = value

    # Store back in the same format it was in originally
    if was_dataframe:
        # Convert back to DataFrame and store
        df = pl.from_dict(session)
        st.session_state[session_id]["session"] = df
    else:
        # Store as dictionary
        st.session_state[session_id]["session"] = session

    # Persist the session changes
    auto_persist_session(session_id)


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

    # Cache in user session
    if user_session_id not in st.session_state:
        st.session_state[user_session_id] = {}
    st.session_state[user_session_id][cache_key] = result

    # Persist the session with new cache data
    auto_persist_session(user_session_id)

    return result


def safe_session_get(session: dict, key: str, default=None):
    """
    Safely get a value from session dict, handling both list and scalar formats.

    When session data comes from DataFrame.to_dict(), values are lists.
    When session data is already a dict, values are scalars.
    This function normalizes access to always return the scalar value.

    Parameters
    ----------
    session : dict
        The session dictionary
    key : str
        The key to access
    default : any
        Default value if key not found

    Returns
    -------
    any
        The scalar value from the session
    """
    value = session.get(key, default)

    # If it's a list (from DataFrame conversion), return first element
    if isinstance(value, list) and len(value) > 0:
        return value[0]

    # If it's already a scalar or empty list, return as-is
    return value if not isinstance(value, list) else default


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
        MetadataKeys.MODEL: model,
        MetadataKeys.DOCIDS: {'ids': sorted(df.get_column("doc_id").unique().to_list())},
        MetadataKeys.TAGS_DS: {'tags': sorted(ds_tags)},
        MetadataKeys.TAGS_POS: {'tags': sorted(tags_pos)},
        MetadataKeys.DOCCATS: {'cats': ''},
        MetadataKeys.COLLOCATIONS: {'temp': ''},
        MetadataKeys.KEYNESS_PARTS: {'temp': ''},
        MetadataKeys.VARIANCE: {'temp': ''},
    }
    df = pl.from_dict(temp_metadata_target, strict=False)
    st.session_state[session_id]["metadata_target"] = df

    # Persist the session with new metadata
    auto_persist_session(session_id)


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
    df = st.session_state[session_id]["reference"]["ds_tokens"]
    tags_to_check = df.get_column("ds_tag").to_list()
    tags = [
        'Actors',
        'Organization',
        'Planning',
        'Sentiment',
        'Signposting',
        'Stance'
    ]
    model = 'Common Dictionary' if any(tag in item for item in tags_to_check for tag in tags) else 'Large Dictionary'  # noqa: E501
    ds_tags = df.get_column("ds_tag").unique().to_list()
    tags_pos = df.get_column("pos_tag").unique().to_list()
    if "Untagged" in ds_tags:
        ds_tags.remove("Untagged")
    if "Y" in tags_pos:
        tags_pos.remove("Y")

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
        MetadataKeys.MODEL: model,
        MetadataKeys.DOCIDS: {'ids': sorted(df.get_column("doc_id").unique().to_list())},
        MetadataKeys.TAGS_DS: {'tags': sorted(ds_tags)},
        MetadataKeys.TAGS_POS: {'tags': sorted(tags_pos)},
        MetadataKeys.DOCCATS: False,
        MetadataKeys.COLLOCATIONS: {'temp': ''},
        MetadataKeys.KEYNESS_PARTS: {'temp': ''},
        MetadataKeys.VARIANCE: {'temp': ''},
    }
    df = pl.from_dict(temp_metadata_reference, strict=False)
    st.session_state[session_id]["metadata_reference"] = df

    # Persist the session with new metadata
    auto_persist_session(session_id)
