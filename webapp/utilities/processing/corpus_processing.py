"""
Corpus processing functions for handling different types of corpus uploads and processing.

This module provides functions for processing internal corpora, external corpora,
newly uploaded text, and corpus finalization.
"""

import gzip
import hashlib
import io
import os
import pickle
from pathlib import Path
from time import perf_counter
import unidecode
import polars as pl
import streamlit as st
import docuscospacy as ds

# Module-specific imports
from webapp.utilities.processing.corpus_loading import load_corpus_internal
from webapp.utilities.session import (
    build_corpus_metadata_descriptor,
    init_metadata_target,
    init_metadata_reference,
    set_session_persistence_policy,
    write_metadata_descriptor_sidecar,
)
from webapp.utilities.session.session_persistence import (
    auto_persist_session,
    mark_session_dirty,
)
from webapp.utilities.analysis.data_validation import (
    check_corpus_external,
    check_corpus_new,
    check_language,
)
from webapp.utilities.state import (
    CorpusPersistencePolicy,
    LoadCorpusKeys,
    SessionKeys,
)
from webapp.utilities.corpus import get_corpus_manager
from webapp.utilities.core import app_core
from webapp.utilities.configuration.logging_config import get_logger
from webapp.config.unified import get_config
from webapp.persistence import registry_service
from webapp.persistence.registry import FREQUENCY_ARTIFACT_TYPE

# Warning constants for corpus processing
WARNING_CORRUPT_TARGET = 10
WARNING_CORRUPT_REFERENCE = 11
WARNING_DUPLICATE_REFERENCE = 21
WARNING_EXCLUDED_TARGET = 40
WARNING_EXCLUDED_REFERENCE = 41
TEST_MODE = get_config('test_mode', 'global', False)
PROCESS_TARGET_PROBE_ENV = "DOCUSCOPE_PROCESS_TARGET_PROBE"
PROCESS_TARGET_PROBE_NONE = "full"
PROCESS_TARGET_PROBE_NO_METADATA = "no_metadata"
PROCESS_TARGET_PROBE_METADATA_NO_PERSIST = "metadata_no_persist"
PROCESS_TARGET_PROBE_NO_RERUN = "no_rerun"
PROCESS_TARGET_PROBE_SPLIT_READY = "split_ready"
SLOW_INTERNAL_PROCESS_STAGE_MS = 50
SESSION_CORPUS_ARTIFACTS_DIRNAME = "corpora"
SESSION_CORE_DATA_FILENAME = "ds_tokens.gz"
PROCESS_TARGET_PROBE_STAGE_KEY = "load_test_process_target_probe_stage"
PROCESS_TARGET_PROBE_TIMINGS_KEY = "load_test_process_target_probe_stage_timings_ms"
PROCESS_TARGET_PROBE_FLAT_TIMINGS_KEY = "load_test_process_target_probe_stage_timings_flat"
PROCESS_TARGET_PROBE_SUMMARY_KEY = "load_test_process_target_probe_stage_summary"
logger = get_logger()


def _log_slow_process_internal_stage(
    user_session_id: str,
    corpus_type: str,
    stage: str,
    start_time: float,
    probe_mode: str,
) -> None:
    """Log slow internal corpus-load stages for benchmark profiling."""

    elapsed_ms = (perf_counter() - start_time) * 1000
    if elapsed_ms >= SLOW_INTERNAL_PROCESS_STAGE_MS:
        logger.warning(
            (
                "Slow internal corpus process session={} corpus_type={} stage={} "
                "probe_mode={} duration_ms={:.2f}"
            ),
            user_session_id,
            corpus_type,
            stage,
            probe_mode,
            elapsed_ms,
        )


def _get_process_target_probe_mode() -> str:
    """Return the active test-only probe mode for internal target loading."""

    if not TEST_MODE:
        return PROCESS_TARGET_PROBE_NONE

    mode = os.getenv(PROCESS_TARGET_PROBE_ENV, PROCESS_TARGET_PROBE_NONE).strip().lower()
    supported_modes = {
        PROCESS_TARGET_PROBE_NONE,
        PROCESS_TARGET_PROBE_NO_METADATA,
        PROCESS_TARGET_PROBE_METADATA_NO_PERSIST,
        PROCESS_TARGET_PROBE_NO_RERUN,
        PROCESS_TARGET_PROBE_SPLIT_READY,
    }
    return mode if mode in supported_modes else PROCESS_TARGET_PROBE_NONE


def _update_session_state_without_persistence(
    user_session_id: str, key: str, value: object
) -> None:
    """Update the in-memory session frame without triggering persistence."""

    session_raw = st.session_state[user_session_id]["session"]

    if hasattr(session_raw, 'to_dict') and hasattr(session_raw, 'columns'):
        session_data = session_raw.to_dict(as_series=False)
        session_data[key] = value
        st.session_state[user_session_id]["session"] = pl.from_dict(session_data)
        return

    if isinstance(session_raw, dict):
        session_data = session_raw.copy()
        session_data[key] = value
        st.session_state[user_session_id]["session"] = session_data
        return

    st.session_state[user_session_id]["session"] = {key: value}


def _mark_process_target_probe_ready(probe_mode: str, corpus_type: str) -> None:
    """Expose a small, test-only UI signal so load tests can stop at the probe boundary."""

    if TEST_MODE:
        st.caption(f"LOAD_TEST_PROCESS_TARGET_READY:{corpus_type}:{probe_mode}")


def _get_process_target_probe_ready_marker(
    probe_mode: str,
    corpus_type: str,
) -> str | None:
    """Return a test-only probe marker when the current run should expose one inline."""

    if not TEST_MODE:
        return None

    return f"LOAD_TEST_PROCESS_TARGET_READY:{corpus_type}:{probe_mode}"


def _record_process_target_probe_stage_duration(
    user_session_id: str,
    corpus_type: str,
    probe_mode: str,
    stage: str,
    elapsed_ms: float,
) -> None:
    """Persist split-ready callback-stage timings for the next rerun."""

    if probe_mode != PROCESS_TARGET_PROBE_SPLIT_READY:
        return

    probe_state = st.session_state.get(PROCESS_TARGET_PROBE_STAGE_KEY)
    if not isinstance(probe_state, dict):
        probe_state = {
            "corpus_type": corpus_type,
            "probe_mode": probe_mode,
            "session_id": user_session_id,
            "stage": "callback_running",
        }

    stage_timings = probe_state.get("stage_timings_ms")
    if not isinstance(stage_timings, dict):
        stage_timings = {}

    stage_timings[stage] = elapsed_ms
    probe_state["stage_timings_ms"] = stage_timings
    probe_state["stage_summary"] = ";".join(
        f"{stage_name}={stage_value:.2f}"
        for stage_name, stage_value in stage_timings.items()
    )
    probe_state["corpus_type"] = corpus_type
    probe_state["probe_mode"] = probe_mode
    probe_state["session_id"] = user_session_id
    st.session_state[PROCESS_TARGET_PROBE_STAGE_KEY] = probe_state
    st.session_state[PROCESS_TARGET_PROBE_FLAT_TIMINGS_KEY] = stage_timings.copy()
    st.session_state[PROCESS_TARGET_PROBE_SUMMARY_KEY] = probe_state["stage_summary"]

    user_bucket = st.session_state.setdefault(user_session_id, {})
    user_stage_timings = user_bucket.get(PROCESS_TARGET_PROBE_TIMINGS_KEY)
    if not isinstance(user_stage_timings, dict):
        user_stage_timings = {}
    user_stage_timings[stage] = elapsed_ms
    user_bucket[PROCESS_TARGET_PROBE_TIMINGS_KEY] = user_stage_timings


def _persist_session_updates(
    user_session_id: str,
    updates: dict[str, object],
    persist_immediately: bool = True,
) -> None:
    """Apply multiple session-flag updates and optionally persist them."""

    session_raw = st.session_state[user_session_id]["session"]

    if hasattr(session_raw, 'to_dict') and hasattr(session_raw, 'columns'):
        session_data = session_raw.to_dict(as_series=False)
        session_data.update(updates)
        st.session_state[user_session_id]["session"] = pl.from_dict(session_data)
    elif isinstance(session_raw, dict):
        session_data = session_raw.copy()
        session_data.update(updates)
        st.session_state[user_session_id]["session"] = session_data
    else:
        st.session_state[user_session_id]["session"] = dict(updates)

    mark_session_dirty(user_session_id)
    if persist_immediately:
        auto_persist_session(user_session_id)


def _build_session_corpus_artifact_path(
    user_session_id: str,
    corpus_type: str,
) -> Path:
    """Return the session-scoped on-disk artifact path for durable core data."""

    storage_root = Path(get_config('storage_path', 'session', 'webapp/_session'))
    session_slug = hashlib.sha256(user_session_id.encode('utf-8')).hexdigest()[:16]
    return (
        storage_root /
        SESSION_CORPUS_ARTIFACTS_DIRNAME /
        session_slug /
        corpus_type /
        SESSION_CORE_DATA_FILENAME
    )


def _store_session_corpus_artifact(
    manager,
    ds_tokens: pl.DataFrame,
    user_session_id: str,
    corpus_type: str,
    metadata_descriptor: dict,
) -> Path:
    """Persist ds_tokens for one session-backed corpus and register a file ref."""

    artifact_path = _build_session_corpus_artifact_path(user_session_id, corpus_type)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(f"{artifact_path}.tmp")

    with gzip.open(temp_path, 'wb') as file_handle:
        pickle.dump(ds_tokens, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp_path, artifact_path)

    write_metadata_descriptor_sidecar(
        artifact_path.parent,
        metadata=metadata_descriptor,
    )
    manager.set_file_refs({'ds_tokens': str(artifact_path)})
    manager.session_corpus_data['ds_tokens'] = ds_tokens
    return artifact_path


def finalize_corpus_load(
    ds_tokens,
    user_session_id: str,
    corpus_type: str,
    persistence_policy: str = CorpusPersistencePolicy.SERVER_SAVED,
) -> None:
    """
    Finalize corpus loading through the lightweight core-data path.

    Parameters
    ----------
    ds_tokens : pl.DataFrame
        The processed DocuScope tokens dataframe.
    user_session_id : str
        The user session identifier.
    corpus_type : str
        Type of corpus ('target' or 'reference').

    Returns
    -------
    None
    """
    finalize_corpus_load_optimized(
        ds_tokens,
        user_session_id,
        corpus_type,
        persistence_policy,
    )


def finalize_corpus_load_optimized(
    ds_tokens,
    user_session_id: str,
    corpus_type: str,
    persistence_policy: str = CorpusPersistencePolicy.SERVER_SAVED,
) -> None:
    """
    Finalize corpus loading using memory-efficient lazy loading approach.

    This optimized version only loads core data (ds_tokens) immediately and
    generates derived data on-demand, reducing initial memory usage by ~60-70%.

    Parameters
    ----------
    ds_tokens : pl.DataFrame
        The processed DocuScope tokens dataframe.
    user_session_id : str
        The user session identifier.
    corpus_type : str
        Type of corpus ('target' or 'reference').

    Returns
    -------
    None
    """
    # Use corpus manager for optimized loading
    manager = get_corpus_manager(user_session_id, corpus_type)
    set_session_persistence_policy(
        user_session_id,
        persistence_policy,
        corpus_type=corpus_type,
    )
    metadata_descriptor = build_corpus_metadata_descriptor(ds_tokens)

    # Only set core data - derived data will be generated on-demand.
    # Delay persistence until any durable file ref is registered.
    manager.set_core_data(ds_tokens, persist=False)

    if persistence_policy == CorpusPersistencePolicy.SERVER_SAVED:
        _store_session_corpus_artifact(
            manager,
            ds_tokens,
            user_session_id,
            corpus_type,
            metadata_descriptor,
        )

    # Initialize metadata and update session flags
    if corpus_type == 'target':
        init_metadata_target(user_session_id, metadata_descriptor)
        app_core.session_manager.update_session_state(
            user_session_id, SessionKeys.HAS_TARGET, True
        )
    else:
        init_metadata_reference(user_session_id, metadata_descriptor)
        app_core.session_manager.update_session_state(
            user_session_id, SessionKeys.HAS_REFERENCE, True
        )

    # Clean up the original corpus DataFrame to free memory
    cleanup_original_corpus_data(user_session_id, corpus_type)

    # Corpus loaded successfully - no console output needed for deployed app
    st.rerun()


def process_new(
    corp_df,
    nlp,
    user_session_id: str,
    corpus_type: str,
    exceptions=None,
    persistence_policy: str = CorpusPersistencePolicy.SERVER_SAVED,
) -> None:
    """
    Process a new corpus dataframe using DocuScope parsing.

    Parameters
    ----------
    corp_df : pl.DataFrame or None
        The corpus dataframe to process.
    nlp : spacy.Language
        The spaCy NLP model for processing.
    user_session_id : str
        The user session identifier.
    corpus_type : str
        Type of corpus ('target' or 'reference').
    exceptions : list, optional
        List of exceptions encountered during processing.

    Returns
    -------
    None
    """
    # Check if corpus dataframe is None (no files uploaded or validation failed)
    if corp_df is None:
        corpus_name = "reference" if corpus_type == "reference" else "target"
        st.warning(
            f"Please upload files for your {corpus_name} corpus before processing.",
            icon=":material/warning:"
        )
        return

    # Check if corpus dataframe is empty
    if corp_df.is_empty():
        corpus_name = "reference" if corpus_type == "reference" else "target"
        st.warning(
            f"No valid text files found for your {corpus_name} corpus. "
            "Please check your uploads.",
            icon=":material/warning:"
        )
        return

    try:
        # Process the corpus with DocuScope
        ds_tokens = ds.docuscope_parse(corp=corp_df, nlp_model=nlp)

        if exceptions and ds_tokens.is_empty():
            # Corpus is completely corrupt
            warning_msg = (
                WARNING_CORRUPT_TARGET if corpus_type == 'target'
                else WARNING_CORRUPT_REFERENCE
            )
            st.session_state[user_session_id]['warning'] = warning_msg
            st.rerun()
        elif exceptions:
            # Some files were excluded but processing succeeded
            st.session_state[user_session_id]['warning'] = (
                WARNING_EXCLUDED_TARGET
                if corpus_type == 'target'
                else WARNING_EXCLUDED_REFERENCE
            )
            st.session_state[user_session_id]['exceptions'] = exceptions
            finalize_corpus_load(
                ds_tokens,
                user_session_id,
                corpus_type,
                persistence_policy,
            )
        else:
            # Processing completed successfully
            st.success('Processing complete!')
            st.session_state[user_session_id]['warning'] = 0
            finalize_corpus_load(
                ds_tokens,
                user_session_id,
                corpus_type,
                persistence_policy,
            )

    except Exception as e:
        corpus_name = "reference" if corpus_type == "reference" else "target"
        st.error(f"Error processing {corpus_name} corpus: {str(e)}")
        # Don't call st.rerun() if there was an error


def process_external(
    df,
    user_session_id: str,
    corpus_type: str,
    persistence_policy: str = CorpusPersistencePolicy.SERVER_SAVED,
) -> None:
    """
    Process an external (preprocessed) corpus dataframe.

    Parameters
    ----------
    df : pl.DataFrame
        The preprocessed corpus dataframe.
    user_session_id : str
        The user session identifier.
    corpus_type : str
        Type of corpus ('target' or 'reference').

    Returns
    -------
    None
    """
    # For external (preprocessed) corpora, no parsing/model needed
    ds_tokens = df
    finalize_corpus_load(
        ds_tokens,
        user_session_id,
        corpus_type,
        persistence_policy,
    )


def process_internal(
    corp_path: str,
    user_session_id: str,
    corpus_type: str,
    persistence_policy: str = CorpusPersistencePolicy.SERVER_SAVED,
) -> None:
    """
    Process an internal corpus from a database path.

    Parameters
    ----------
    corp_path : str
        Path to the corpus database.
    user_session_id : str
        The user session identifier.
    corpus_type : str
        Type of corpus ('target' or 'reference').

    Returns
    -------
    None
    """
    try:
        probe_mode = _get_process_target_probe_mode()

        # Load the internal corpus
        load_start = perf_counter()
        load_corpus_internal(
            corp_path,
            user_session_id,
            corpus_type=corpus_type
        )
        load_elapsed_ms = (perf_counter() - load_start) * 1000
        _log_slow_process_internal_stage(
            user_session_id,
            corpus_type,
            "load_corpus_internal",
            load_start,
            probe_mode,
        )
        _record_process_target_probe_stage_duration(
            user_session_id,
            corpus_type,
            probe_mode,
            "load_corpus_internal",
            load_elapsed_ms,
        )

        # Verify the corpus was loaded successfully through the manager layer.
        manager_ready_start = perf_counter()
        manager = get_corpus_manager(user_session_id, corpus_type)
        if not manager.is_ready():
            st.error(f"Failed to load {corpus_type} corpus data.")
            return
        manager_ready_elapsed_ms = (perf_counter() - manager_ready_start) * 1000
        _log_slow_process_internal_stage(
            user_session_id,
            corpus_type,
            "manager_ready_check",
            manager_ready_start,
            probe_mode,
        )
        _record_process_target_probe_stage_duration(
            user_session_id,
            corpus_type,
            probe_mode,
            "manager_ready_check",
            manager_ready_elapsed_ms,
        )

        set_session_persistence_policy(
            user_session_id,
            persistence_policy,
            corpus_type=corpus_type,
        )

        session_key_db = (
            SessionKeys.TARGET_DB if corpus_type == "target"
            else SessionKeys.REFERENCE_DB
        )
        session_key_has_corpus = (
            SessionKeys.HAS_TARGET if corpus_type == "target"
            else SessionKeys.HAS_REFERENCE
        )

        if probe_mode == PROCESS_TARGET_PROBE_NO_METADATA:
            _update_session_state_without_persistence(
                user_session_id, session_key_db, str(corp_path)
            )
            _update_session_state_without_persistence(
                user_session_id, session_key_has_corpus, True
            )
            _mark_process_target_probe_ready(probe_mode, corpus_type)
            return

        # Update session state based on corpus type
        metadata_start = perf_counter()
        if corpus_type == "target":
            init_metadata_target(user_session_id)
        else:
            init_metadata_reference(user_session_id)
        metadata_elapsed_ms = (perf_counter() - metadata_start) * 1000
        _log_slow_process_internal_stage(
            user_session_id,
            corpus_type,
            "init_metadata",
            metadata_start,
            probe_mode,
        )
        _record_process_target_probe_stage_duration(
            user_session_id,
            corpus_type,
            probe_mode,
            "init_metadata",
            metadata_elapsed_ms,
        )

        if probe_mode == PROCESS_TARGET_PROBE_METADATA_NO_PERSIST:
            _update_session_state_without_persistence(
                user_session_id, session_key_db, str(corp_path)
            )
            _update_session_state_without_persistence(
                user_session_id, session_key_has_corpus, True
            )
            _mark_process_target_probe_ready(probe_mode, corpus_type)
            return

        if corpus_type == "target":
            persist_start = perf_counter()
            _persist_session_updates(
                user_session_id,
                {
                    SessionKeys.TARGET_DB: str(corp_path),
                    SessionKeys.HAS_TARGET: True,
                },
                persist_immediately=False,
            )
        else:
            persist_start = perf_counter()
            _persist_session_updates(
                user_session_id,
                {
                    SessionKeys.REFERENCE_DB: str(corp_path),
                    SessionKeys.HAS_REFERENCE: True,
                },
                persist_immediately=False,
            )
        persist_elapsed_ms = (perf_counter() - persist_start) * 1000
        _log_slow_process_internal_stage(
            user_session_id,
            corpus_type,
            "persist_session_updates",
            persist_start,
            probe_mode,
        )
        _record_process_target_probe_stage_duration(
            user_session_id,
            corpus_type,
            probe_mode,
            "persist_session_updates",
            persist_elapsed_ms,
        )

        if corpus_type == "target":
            warm_start = perf_counter()
            manager.warm_shared_frequency_data()
            warm_elapsed_ms = (perf_counter() - warm_start) * 1000
            _log_slow_process_internal_stage(
                user_session_id,
                corpus_type,
                "warm_shared_frequency_data",
                warm_start,
                probe_mode,
            )
            _record_process_target_probe_stage_duration(
                user_session_id,
                corpus_type,
                probe_mode,
                "warm_shared_frequency_data",
                warm_elapsed_ms,
            )

        if probe_mode == PROCESS_TARGET_PROBE_NO_RERUN:
            return _get_process_target_probe_ready_marker(probe_mode, corpus_type)

        if probe_mode == PROCESS_TARGET_PROBE_SPLIT_READY:
            probe_state = st.session_state.get(PROCESS_TARGET_PROBE_STAGE_KEY)
            stage_timings = {}
            if isinstance(probe_state, dict):
                existing_stage_timings = probe_state.get("stage_timings_ms")
                if isinstance(existing_stage_timings, dict):
                    stage_timings = existing_stage_timings.copy()
            if stage_timings:
                stage_summary = " ".join(
                    f"{stage_name}={stage_value:.2f}"
                    for stage_name, stage_value in stage_timings.items()
                )
                logger.warning(
                    (
                        "LOAD_TEST_PROCESS_TARGET_CALLBACK_STAGE_LOG session={} "
                        "corpus_type={} {}"
                    ),
                    user_session_id,
                    corpus_type,
                    stage_summary,
                )
            st.session_state[PROCESS_TARGET_PROBE_STAGE_KEY] = {
                "corpus_type": corpus_type,
                "stage": "callback_finished",
                "probe_mode": probe_mode,
                "session_id": user_session_id,
                "stage_timings_ms": stage_timings,
            }

        st.rerun()

    except Exception as e:
        st.error(f"Error processing {corpus_type} corpus: {str(e)}")
        # Don't call st.rerun() if there was an error


def attach_queued_internal_target(
    corp_path: str,
    user_session_id: str,
    persistence_policy: str = CorpusPersistencePolicy.SERVER_SAVED,
    queue_artifact_id: int | None = None,
) -> None:
    """Attach a queue-prepared built-in target corpus to the current session."""

    try:
        load_corpus_internal(
            corp_path,
            user_session_id,
            corpus_type="target",
        )
        manager = get_corpus_manager(user_session_id, "target")
        if not manager.is_ready():
            st.error("Failed to load target corpus data.")
            return

        set_session_persistence_policy(
            user_session_id,
            persistence_policy,
            corpus_type="target",
        )
        init_metadata_target(user_session_id)
        if queue_artifact_id is not None:
            queue_artifact = registry_service.get_artifact_by_id(queue_artifact_id)
            if queue_artifact is not None and queue_artifact.status == "ready":
                queue_payload = registry_service.load_json_artifact(queue_artifact)
                frequency_artifact_id = queue_payload.get("frequency_artifact_id")
                if isinstance(frequency_artifact_id, int):
                    manager.set_artifact_refs(
                        FREQUENCY_ARTIFACT_TYPE,
                        frequency_artifact_id,
                        ["ft_pos", "ft_ds"],
                    )
        _persist_session_updates(
            user_session_id,
            {
                SessionKeys.TARGET_DB: str(corp_path),
                SessionKeys.HAS_TARGET: True,
            },
            persist_immediately=False,
        )
        st.rerun()
    except Exception as exc:
        st.error(f"Error attaching queued target corpus: {str(exc)}")


def handle_uploaded_parquet(
        uploaded_file,
        check_size: bool,
        max_size: int,
        target_docs=None
        ) -> tuple[pl.DataFrame | None, bool]:
    """
    Handle processing of an uploaded Parquet file.
    Read a parquet file and check corpus validity, size,
    and (optionally) duplicates.

    Parameters
    ----------
    uploaded_file : UploadedFile or None
        The uploaded Parquet file.
    check_size : bool
        Whether to check corpus size.
    max_size : int
        Maximum allowed corpus size.
    target_docs : list, optional
        Target documents for duplicate checking.

    Returns
    -------
    tuple[pl.DataFrame | None, bool]
        Tuple of (dataframe, ready_to_process).
    """
    if uploaded_file is not None:
        try:
            df = pl.read_parquet(uploaded_file)
        except Exception as e:
            st.error(f"Error processing Parquet file: {e}")
            return None, False
    else:
        df = None

    check_kwargs = dict(tok_pl=df)
    if check_size:
        check_kwargs['check_size'] = True
    if target_docs is not None:
        check_kwargs['check_ref'] = True
        check_kwargs['target_docs'] = target_docs

    result = check_corpus_external(**check_kwargs)

    # Unpack result based on which checks are enabled
    if check_size and target_docs is not None:
        is_valid, dup_docs, corpus_size = result
    elif check_size:
        is_valid, corpus_size = result
        dup_docs = []
    elif target_docs is not None:
        is_valid, dup_docs = result
        corpus_size = 0
    else:
        is_valid = result
        dup_docs = []
        corpus_size = 0

    # Only show format error if a file was uploaded and is invalid
    if uploaded_file is not None and not is_valid:
        st.error(
            """
            Your pre-processed corpus is not in the correct format.
            You can try selecting a different file or processing your corpus
            from the original text files and saving it again.
            """,
            icon=":material/block:"
        )
    if check_size and corpus_size > max_size:
        st.error(
            """
            Your corpus is too large for online processing.
            The online version of DocuScope Corpus Analysis & Concordancer
            accepts data up to roughly 3 million words.
            If you'd like to process more data, try
            [the desktop version of the tool](https://github.com/browndw/docuscope-ca-desktop)
            which available for free.
            """,  # noqa: E501
            icon=":material/warning:"
            )
    if target_docs is not None and len(dup_docs) > 0:
        st.error(
            f"""
            The files you selected could not be processed.
            Files with these names were also submitted
            as part of your target corpus:
            ```
            {sorted(dup_docs)}
            ```
            Please remove files from your reference corpus before processing.
            To clear this warning click the **UPLOAD REFERENCE** button.
            """,
            icon=":material/block:"
        )

    ready = (
        is_valid and
        df is not None and
        df.is_empty() is False and
        (corpus_size <= max_size if check_size else True) and
        (len(dup_docs) == 0 if target_docs is not None else True)
    )

    if ready:
        st.success(
            """Success! Your corpus is ready to be processed.
            Use the **Process** button in the sidebar to continue.
            """,
            icon=":material/celebration:"
        )

    return df, ready


def handle_uploaded_text(
        uploaded_files: list,
        check_size: bool,
        max_size: int,
        check_language_flag=False,
        check_ref=False,
        target_docs=None
        ) -> tuple[pl.DataFrame | None, bool, list]:
    """
    Handle uploaded text files, run check_corpus_new,
    and return (DataFrame, ready, exceptions).

    Parameters
    ----------
    uploaded_files : list
        List of uploaded text files.
    check_size : bool
        Whether to check corpus size.
    max_size : int
        Maximum allowed corpus size.
    check_language_flag : bool, optional
        Whether to check language of documents.
    check_ref : bool, optional
        Whether to check for reference documents.
    target_docs : list, optional
        Target documents for duplicate checking.

    Returns
    -------
    tuple[pl.DataFrame | None, bool, list]
        Tuple of (dataframe, ready_to_process, exceptions).
    """
    if not uploaded_files or len(uploaded_files) == 0:
        # No files uploaded
        return None, False, []

    # Prepare kwargs for check_corpus_new
    check_kwargs = dict(docs=uploaded_files)
    if check_size:
        check_kwargs['check_size'] = True
    if check_language_flag:
        check_kwargs['check_language_flag'] = True
    if check_ref:
        check_kwargs['check_ref'] = True
        check_kwargs['target_docs'] = target_docs

    result = check_corpus_new(**check_kwargs)

    # Unpack result based on which options are enabled
    dup_ids, dup_docs, lang_fail, corpus_size = [], [], [], 0
    if check_ref and check_size and check_language_flag:
        dup_ids, dup_docs, lang_fail, corpus_size = result
    elif check_ref and check_size:
        dup_ids, dup_docs, corpus_size = result
        lang_fail = []
    elif check_ref and check_language_flag:
        dup_ids, dup_docs, lang_fail = result
        corpus_size = 0
    elif check_ref:
        dup_ids, dup_docs = result
        lang_fail = []
        corpus_size = 0
    elif check_size and check_language_flag:
        dup_ids, lang_fail, corpus_size = result
        dup_docs = []
    elif check_size:
        dup_ids, corpus_size = result
        dup_docs = []
        lang_fail = []
    elif check_language_flag:
        dup_ids, lang_fail = result
        dup_docs = []
        corpus_size = 0
    else:
        dup_ids = result
        dup_docs = []
        lang_fail = []
        corpus_size = 0

    # Streamlit error handling (for user feedback)
    if len(dup_ids) > 0:
        st.error(
            f"""
            The files you selected could not be processed.
            Your corpus contains these duplicate file names:
            ```
            {sorted(dup_ids)}
            ```
            Please remove duplicates before processing.
            To clear this warning click the **UPLOAD** button.
            """,
            icon=":material/block:"
        )
    if check_ref and len(dup_docs) > 0:
        st.error(
            f"""
            The files you selected could not be processed.
            Files with these names were also submitted
            as part of your target corpus:
            ```
            {sorted(dup_docs)}
            ```
            Please remove files from your reference corpus before processing.
            To clear this warning click the **UPLOAD REFERENCE** button.
            """,
            icon=":material/block:"
        )
    if check_language_flag and len(lang_fail) > 0:
        st.error(
            f"""
            The files you selected could not be processed.
            Files with these names are either not in English or
            are incompatible with the reqirement of the model:
            ```
            {sorted(lang_fail)}
            ```
            Please remove files from your corpus before processing.
            To clear this warning click the **UPLOAD TARGET** button.
            """,
            icon=":material/warning:"
        )
    if check_size and corpus_size > max_size:
        st.error(
            """
            Your corpus is too large for online processing.
            The online version of DocuScope Corpus Analysis & Concordancer
            accepts data up to roughly 3 million words.
            If you'd like to process more data, try
            [the desktop version of the tool](https://github.com/browndw/docuscope-ca-desktop)
            which available for free.
            """,  # noqa: E501
            icon=":material/warning:"
        )

    # Determine readiness
    ready = (
        len(uploaded_files) > 0 and
        len(dup_ids) == 0 and
        (len(dup_docs) == 0 if check_ref else True) and
        (corpus_size <= max_size if check_size else True) and
        (len(lang_fail) == 0 if check_language_flag else True)
    )

    # Only create DataFrame if ready
    if ready:
        st.success(
            f"""Success!
            **{len(uploaded_files)}** corpus files ready!
            Use the **Process** button in the sidebar to continue.
            """,
            icon=":material/celebration:"
        )
        df, exceptions = corpus_from_widget(uploaded_files)
    else:
        df, exceptions = None, []

    return df, ready, exceptions


def handle_uploaded_tabular(
        uploaded_file,
        check_size: bool,
        max_size: int,
        check_language_flag=False,
        check_ref=False,
        target_docs=None
        ) -> tuple[pl.DataFrame | None, bool, list]:
    """
    Handle a single tabular corpus upload with doc_id and text columns.

    Parameters
    ----------
    uploaded_file : UploadedFile or None
        Uploaded Parquet, CSV, or TSV file containing corpus rows.
    check_size : bool
        Whether to check corpus size.
    max_size : int
        Maximum allowed corpus size.
    check_language_flag : bool, optional
        Whether to check language of documents.
    check_ref : bool, optional
        Whether to check for reference documents.
    target_docs : list, optional
        Target documents for duplicate checking.

    Returns
    -------
    tuple[pl.DataFrame | None, bool, list]
        Tuple of (dataframe, ready_to_process, exceptions).
    """
    if uploaded_file is None:
        return None, False, []

    file_name = uploaded_file.name
    file_ext = os.path.splitext(file_name)[1].lower().lstrip(".")
    file_bytes = uploaded_file.getvalue()

    if file_ext not in {"parquet", "csv", "tsv"}:
        st.error(
            "Tabular corpora must be uploaded as Parquet, CSV, or TSV files.",
            icon=":material/block:"
        )
        return None, False, []

    try:
        file_buffer = io.BytesIO(file_bytes)
        if file_ext == "parquet":
            df = pl.read_parquet(file_buffer)
        elif file_ext == "tsv":
            df = pl.read_csv(file_buffer, separator="\t", infer_schema=False)
        else:
            df = pl.read_csv(file_buffer, infer_schema=False)
    except Exception as e:
        st.error(f"Error processing tabular corpus file: {e}")
        return None, False, []

    required_cols = ["doc_id", "text"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(
            f"Your table must include these columns: {', '.join(missing_cols)}.",
            icon=":material/block:"
        )
        return None, False, []

    if df.is_empty():
        st.error(
            "Your tabular corpus does not contain any rows.",
            icon=":material/block:"
        )
        return None, False, []

    df = df.select(required_cols).with_columns(
        pl.col("doc_id").cast(pl.String),
        pl.col("text").cast(pl.String)
    )

    null_doc_ids = df.select(pl.col("doc_id").is_null().sum()).item()
    null_texts = df.select(pl.col("text").is_null().sum()).item()
    if null_doc_ids > 0 or null_texts > 0:
        st.error(
            "Your table contains empty values in the doc_id or text columns.",
            icon=":material/block:"
        )
        return None, False, []

    df = df.with_columns(
        pl.col("doc_id").str.strip_chars().str.replace_all(" ", ""),
        pl.col("text").str.strip_chars()
    )

    empty_doc_ids = df.filter(pl.col("doc_id") == "").height
    empty_texts = df.filter(pl.col("text") == "").height
    if empty_doc_ids > 0 or empty_texts > 0:
        st.error(
            "Your table contains blank values in the doc_id or text columns.",
            icon=":material/block:"
        )
        return None, False, []

    duplicate_ids = (
        df.group_by("doc_id")
        .len()
        .filter(pl.col("len") > 1)
        .get_column("doc_id")
        .to_list()
    )
    if len(duplicate_ids) > 0:
        st.error(
            f"""
            Your table contains these duplicate doc_id values:
            ```
            {sorted(duplicate_ids)}
            ```
            Please remove duplicates before processing.
            """,
            icon=":material/block:"
        )
        return None, False, []

    if check_ref and target_docs is not None:
        dup_docs = list(set(target_docs).intersection(df.get_column("doc_id")))
    else:
        dup_docs = []
    if check_ref and len(dup_docs) > 0:
        st.error(
            f"""
            The table you selected could not be processed.
            Documents with these IDs were also submitted
            as part of your target corpus:
            ```
            {sorted(dup_docs)}
            ```
            Please remove documents from your reference corpus before processing.
            """,
            icon=":material/block:"
        )
        return None, False, []

    if check_language_flag:
        corpus_text = " ".join(df.get_column("text").to_list())
        if not check_language(corpus_text):
            st.error(
                """
                The table you selected could not be processed.
                The text column is either not in English or
                are incompatible with the requirement of the model:
                """,
                icon=":material/warning:"
            )
            return None, False, []

    corpus_size = sum(
        len(text.encode("utf-8"))
        for text in df.get_column("text").to_list()
    )
    if check_size and corpus_size > max_size:
        st.error(
            """
            Your corpus is too large for online processing.
            The online version of DocuScope Corpus Analysis & Concordancer
            accepts data up to roughly 3 million words.
            If you'd like to process more data, try
            [the desktop version of the tool](https://github.com/browndw/docuscope-ca-desktop)
            which available for free.
            """,  # noqa: E501
            icon=":material/warning:"
        )
        return None, False, []

    df = (
        df.with_columns(
            pl.col("text").map_elements(unidecode.unidecode, return_dtype=pl.String)
        )
        .sort("doc_id")
    )

    st.success(
        f"""Success!
        **{df.height}** corpus documents ready!
        Use the **Process** button in the sidebar to continue.
        """,
        icon=":material/celebration:"
    )

    return df, True, []


def sidebar_process_section(
    section_title: str,
    button_label: str,
    process_fn,
    button_icon: str = ":material/manufacturing:",
    spinner_text: str = "Processing corpus data..."
) -> None:
    """
    Helper to standardize sidebar processing UI.

    Parameters
    ----------
    section_title : str
        The sidebar section title.
    button_label : str
        The label for the action button.
    process_fn : callable
        Function to call when button is pressed.
    button_icon : str
        Icon for the button.
    spinner_text : str
        Text to show in the spinner.

    Returns
    -------
    None
    """
    st.sidebar.markdown(f"### {section_title}")
    st.sidebar.markdown(
        """
        Once you have selected your files,
        use the button to process your corpus.
        """)
    if st.sidebar.button(button_label, icon=button_icon):
        with st.sidebar.status(spinner_text, expanded=True):
            probe_marker = process_fn()
            if probe_marker:
                st.caption(probe_marker)
            st.success("Processing complete!",
                       icon=":material/celebration:")
    st.sidebar.markdown("---")


def corpus_from_widget(docs) -> tuple[pl.DataFrame, list]:
    """
    Process uploaded files from a widget and return
    a Polars DataFrame and a list of exceptions.

    Parameters
    ----------
    docs : iterable
        Iterable of file-like objects with .name and .getvalue() methods.

    Returns
    -------
    tuple
        (Polars DataFrame with columns 'doc_id' and 'text',
        list of filenames that failed to decode)
    """

    exceptions = []
    records = []
    for doc in docs:
        try:
            doc_txt = doc.getvalue().decode('utf-8')
            doc_txt = unidecode.unidecode(doc_txt)
            doc_id = str(os.path.splitext(doc.name.replace(" ", ""))[0])
            records.append({"doc_id": doc_id, "text": doc_txt})
        except Exception:
            exceptions.append(doc.name)

    if records:
        df = pl.DataFrame(records)
        df = (
            df.with_columns(
                pl.col("text").str.strip_chars()
            )
            .sort("doc_id")
        )
    else:
        df = pl.DataFrame({"doc_id": [], "text": []})

    return df, exceptions


def cleanup_original_corpus_data(user_session_id: str, corpus_type: str) -> None:
    """
    Clean up the original corpus DataFrame and related data from session state
    after successful processing to free memory.

    Parameters
    ----------
    user_session_id : str
        The user session identifier.
    corpus_type : str
        Type of corpus ('target' or 'reference').

    Returns
    -------
    None
    """
    if corpus_type == 'target':
        # Clear target corpus data
        if LoadCorpusKeys.CORPUS_DF in st.session_state[user_session_id]:
            original_df = st.session_state[user_session_id][LoadCorpusKeys.CORPUS_DF]
            if original_df is not None:
                # Log memory cleanup for debugging
                st.success(
                    "✅ Original corpus text data cleaned from memory",
                    icon=":material/cleaning_services:"
                )
            st.session_state[user_session_id][LoadCorpusKeys.CORPUS_DF] = None
        if LoadCorpusKeys.EXCEPTIONS in st.session_state[user_session_id]:
            st.session_state[user_session_id][LoadCorpusKeys.EXCEPTIONS] = None
        st.session_state[user_session_id][LoadCorpusKeys.READY_TO_PROCESS] = False
    else:
        # Clear reference corpus data
        if LoadCorpusKeys.REF_CORPUS_DF in st.session_state[user_session_id]:
            original_df = st.session_state[user_session_id][LoadCorpusKeys.REF_CORPUS_DF]
            if original_df is not None:
                # Log memory cleanup for debugging
                st.success(
                    "✅ Original reference corpus text data cleaned from memory",
                    icon=":material/cleaning_services:"
                )
            st.session_state[user_session_id][LoadCorpusKeys.REF_CORPUS_DF] = None
        if LoadCorpusKeys.REF_EXCEPTIONS in st.session_state[user_session_id]:
            st.session_state[user_session_id][LoadCorpusKeys.REF_EXCEPTIONS] = None
        st.session_state[user_session_id][LoadCorpusKeys.REF_READY_TO_PROCESS] = False
