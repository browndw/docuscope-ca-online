"""
This module contains the core session initialization, update, and management.
"""

from functools import lru_cache
from pathlib import Path
import gzip
import json
import pickle

import polars as pl
import streamlit as st
from time import perf_counter

from webapp.utilities.corpus import get_corpus_manager
from webapp.utilities.state import (
    CorpusPersistencePolicy,
    MetadataKeys,
    SessionKeys,
)
from webapp.utilities.common import get_doc_cats
from webapp.utilities.configuration.logging_config import get_logger
from webapp.utilities.session.session_persistence import (
    load_persistent_session,
    auto_persist_session,
    mark_session_dirty,
)


COMMON_DICTIONARY_TAGS = {
    'Actors', 'Organization', 'Planning', 'Sentiment', 'Signposting', 'Stance'
}
METADATA_DESCRIPTOR_FILE = "metadata_descriptor.json"
SLOW_SESSION_UPDATE_MS = 25
SLOW_METADATA_DESCRIPTOR_MS = 50
logger = get_logger()


def _log_slow_session_update(
    session_id: str,
    key: str,
    start_time: float,
    persist_ms: float,
) -> None:
    """Log slow session-state mutations on the analysis hot path."""

    elapsed_ms = (perf_counter() - start_time) * 1000
    if elapsed_ms >= SLOW_SESSION_UPDATE_MS:
        logger.warning(
            (
                "Slow session update session={} key={} persist_ms={:.2f} "
                "duration_ms={:.2f}"
            ),
            session_id,
            key,
            persist_ms,
            elapsed_ms,
        )


def _log_slow_metadata_descriptor(
    session_id: str,
    corpus_type: str,
    source: str,
    start_time: float,
) -> None:
    """Log slow metadata descriptor resolution on the corpus-load path."""

    elapsed_ms = (perf_counter() - start_time) * 1000
    if elapsed_ms >= SLOW_METADATA_DESCRIPTOR_MS:
        logger.warning(
            (
                "Slow metadata descriptor session={} corpus_type={} source={} "
                "duration_ms={:.2f}"
            ),
            session_id,
            corpus_type,
            source,
            elapsed_ms,
        )


def build_corpus_metadata_descriptor(df: pl.DataFrame) -> dict:
    """Build the minimal eager metadata contract shared across corpus sources."""

    ds_tag_series = df.get_column("ds_tag")
    pos_tag_series = df.get_column("pos_tag")
    doc_id_series = df.get_column("doc_id")

    ds_tags = sorted(
        tag for tag in ds_tag_series.unique().to_list()
        if tag != "Untagged"
    )
    tags_pos = sorted(
        tag for tag in pos_tag_series.unique().to_list()
        if tag != "Y"
    )
    doc_ids = sorted(doc_id_series.unique().to_list())

    model = (
        'Common Dictionary'
        if any(tag in ds_tag for ds_tag in ds_tags for tag in COMMON_DICTIONARY_TAGS)
        else 'Large Dictionary'
    )

    return {
        MetadataKeys.TOKENS_POS: df.group_by(
            ["doc_id", "pos_id", "pos_tag"]
        ).agg(pl.col("token").str.join("")).filter(
            pl.col("pos_tag") != "Y"
        ).height,
        MetadataKeys.TOKENS_DS: df.group_by(
            ["doc_id", "ds_id", "ds_tag"]
        ).agg(pl.col("token").str.join("")).filter(
            ~(
                pl.col("token").str.contains("^[[[:punct:]] ]+$")
                & pl.col("ds_tag").str.contains("Untagged")
            )
        ).height,
        MetadataKeys.NDOCS: len(doc_ids),
        MetadataKeys.MODEL: model,
        MetadataKeys.DOCIDS: {'ids': doc_ids},
        MetadataKeys.TAGS_DS: {'tags': ds_tags},
        MetadataKeys.TAGS_POS: {'tags': tags_pos},
    }


def write_metadata_descriptor_sidecar(
    corpus_dir: str | Path,
    metadata: dict | None = None,
    df: pl.DataFrame | None = None,
) -> Path:
    """Write a metadata descriptor sidecar inside one corpus storage boundary."""

    if metadata is None:
        if df is None:
            raise ValueError("metadata or df is required to write a descriptor sidecar")
        metadata = build_corpus_metadata_descriptor(df)

    sidecar_path = Path(corpus_dir) / METADATA_DESCRIPTOR_FILE
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar_path, "w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=2, sort_keys=True)
        file_handle.write("\n")

    return sidecar_path


@lru_cache(maxsize=32)
def _load_metadata_descriptor_from_json(path: str, modified_at_ns: int) -> dict:
    """Load and cache a metadata descriptor sidecar from a corpus directory."""

    del modified_at_ns

    with open(path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


@lru_cache(maxsize=32)
def _load_metadata_descriptor_from_gzip(path: str, modified_at_ns: int) -> dict:
    """Build and cache a metadata descriptor from a file-backed ds_tokens artifact."""

    del modified_at_ns

    with gzip.open(path, "rb") as file_handle:
        df = pickle.load(file_handle)

    return build_corpus_metadata_descriptor(df)


def _get_core_metadata_descriptor(session_id: str, corpus_type: str) -> dict:
    """Resolve metadata without hydrating file-backed corpora into session."""

    start_time = perf_counter()
    manager = get_corpus_manager(session_id, corpus_type)
    core_data = manager.session_corpus_data.get("ds_tokens")
    if isinstance(core_data, pl.DataFrame):
        metadata = build_corpus_metadata_descriptor(core_data)
        _log_slow_metadata_descriptor(
            session_id, corpus_type, "session_core_data", start_time
        )
        return metadata

    refs = manager._get_artifact_refs()
    core_ref = refs.get("ds_tokens") if isinstance(refs, dict) else None
    if core_ref and core_ref.get("storage_type") == "gzip_pickle":
        path = core_ref.get("path")
        if path:
            sidecar_path = Path(path).with_name(METADATA_DESCRIPTOR_FILE)
            if sidecar_path.exists():
                modified_at_ns = sidecar_path.stat().st_mtime_ns
                metadata = _load_metadata_descriptor_from_json(
                    str(sidecar_path),
                    modified_at_ns,
                )
                _log_slow_metadata_descriptor(
                    session_id, corpus_type, "json_sidecar", start_time
                )
                return metadata
            modified_at_ns = Path(path).stat().st_mtime_ns
            metadata = _load_metadata_descriptor_from_gzip(path, modified_at_ns)
            _log_slow_metadata_descriptor(
                session_id, corpus_type, "gzip_pickle", start_time
            )
            return metadata

    core_data = manager.get_core_data()
    if core_data is None:
        raise ValueError(f"{corpus_type.title()} corpus tokens unavailable")
    metadata = build_corpus_metadata_descriptor(core_data)
    _log_slow_metadata_descriptor(
        session_id, corpus_type, "manager_core_data", start_time
    )
    return metadata


def _store_metadata(session_id: str, key: str, metadata: dict) -> None:
    """Store metadata in the same DataFrame-backed format used by session state."""

    st.session_state[session_id][key] = pl.from_dict(metadata, strict=False)
    mark_session_dirty(session_id)


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
        SessionKeys.CORPUS_PERSISTENCE_POLICY: (
            CorpusPersistencePolicy.SERVER_SAVED
        ),
        SessionKeys.TARGET_PERSISTENCE_POLICY: (
            CorpusPersistencePolicy.SERVER_SAVED
        ),
        SessionKeys.REFERENCE_PERSISTENCE_POLICY: (
            CorpusPersistencePolicy.SERVER_SAVED
        ),
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
    mark_session_dirty(session_id)

    # Do not persist the empty scaffold immediately.
    # It is cheap to recreate and avoiding this write reduces hot-path churn.


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
    update_start = perf_counter()
    session_raw = st.session_state[session_id]["session"]

    # Handle both DataFrame and dict cases (unified session management)
    if hasattr(session_raw, 'to_dict') and hasattr(session_raw, 'columns'):
        # It's a Polars DataFrame (has both to_dict and columns attributes)
        if session_raw.height == 1:
            st.session_state[session_id]["session"] = session_raw.with_columns(
                pl.lit(value).alias(key)
            )
            mark_session_dirty(session_id)

            persist_start = perf_counter()
            auto_persist_session(session_id)
            persist_ms = (perf_counter() - persist_start) * 1000
            _log_slow_session_update(session_id, key, update_start, persist_ms)
            return

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

    mark_session_dirty(session_id)

    # Persist the session changes
    persist_start = perf_counter()
    auto_persist_session(session_id)
    persist_ms = (perf_counter() - persist_start) * 1000
    _log_slow_session_update(session_id, key, update_start, persist_ms)


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


def init_metadata_target(session_id: str, metadata: dict | None = None) -> None:
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
    if metadata is None:
        metadata = _get_core_metadata_descriptor(session_id, "target")
    _store_metadata(session_id, SessionKeys.METADATA_TARGET, metadata)

    # Metadata is derived and can be reconstructed from corpus artifacts.
    # Avoid persisting it immediately on the hot path.


def init_metadata_reference(session_id: str, metadata: dict | None = None) -> None:
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
    if metadata is None:
        metadata = _get_core_metadata_descriptor(session_id, "reference")
    _store_metadata(session_id, SessionKeys.METADATA_REFERENCE, metadata)

    # Metadata is derived and can be reconstructed from corpus artifacts.
    # Avoid persisting it immediately on the hot path.
