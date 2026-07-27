"""
Corpus generation utilities for n-grams, collocations, and KWIC analysis.

This module provides functions for generating various types of corpus
analyses including n-grams, collocations, and keyword-in-context searches.
"""

import streamlit as st
import docuscospacy as ds

from webapp.persistence import (
    SharedArtifactWorkflow,
    build_shared_collocation_identity,
    build_shared_ngram_identity,
    registry_service,
)
from webapp.utilities.core import app_core
from webapp.utilities.configuration.logging_config import get_logger
from webapp.utilities.state import CorpusKeys, TargetKeys, WarningKeys, SessionKeys
from webapp.utilities.corpus import get_corpus_data_manager
from webapp.utilities.session import safe_session_get


logger = get_logger()
shared_artifact_workflow = SharedArtifactWorkflow(registry_service, logger)


def _get_shared_collocation_identity(
        user_session_id: str,
        node_word: str,
        node_tag: str,
        to_left: int,
        to_right: int,
        stat_mode: str,
        count_by: str
):
    """Return a shared collocation identity for built-in target corpora."""

    session = st.session_state.get(user_session_id, {})
    target_db = safe_session_get(session, SessionKeys.TARGET_DB, "")
    if not target_db:
        return None

    return build_shared_collocation_identity(
        target_source=target_db,
        node_word=node_word,
        node_tag=node_tag,
        to_left=to_left,
        to_right=to_right,
        stat_mode=stat_mode,
        count_by=count_by,
    )


def _get_shared_ngram_identity(
        user_session_id: str,
        analysis_type: str,
        ngram_span: int,
        count_by: str,
        from_anchor: str | None = None,
        node_word: str | None = None,
        tag: str | None = None,
        position: int | None = None,
        search_type: str | None = None,
):
    """Return a shared n-gram/cluster identity for built-in target corpora."""

    session = st.session_state.get(user_session_id, {})
    target_db = safe_session_get(session, SessionKeys.TARGET_DB, "")
    if not target_db:
        return None

    return build_shared_ngram_identity(
        target_source=target_db,
        analysis_type=analysis_type,
        ngram_span=ngram_span,
        count_by=count_by,
        from_anchor=from_anchor,
        node_word=node_word,
        tag=tag,
        position=position,
        search_type=search_type,
    )


def _ngram_parameters(
        analysis_type: str,
        ngram_span: int,
        count_by: str,
        from_anchor: str | None = None,
        node_word: str | None = None,
        tag: str | None = None,
        position: int | None = None,
        search_type: str | None = None,
) -> dict:
    """Return stable parameters for shared n-gram/cluster generation."""

    return {
        "analysis_type": analysis_type,
        "ngram_span": ngram_span,
        "count_by": count_by,
        "from_anchor": from_anchor,
        "node_word": node_word.strip() if isinstance(node_word, str) else node_word,
        "tag": tag,
        "position": position,
        "search_type": search_type,
    }


def attach_ngram_artifact(
        user_session_id: str,
        artifact_id: int,
        artifact_type: str | None = None,
) -> bool:
    """Attach a ready n-gram/cluster artifact to the target corpus session."""

    if artifact_type is None:
        artifact = registry_service.get_artifact_by_id(artifact_id)
        if artifact is None or artifact.status != "ready":
            return False
        artifact_type = artifact.artifact_type

    manager = get_corpus_data_manager(user_session_id, CorpusKeys.TARGET)
    manager.set_artifact_refs(
        artifact_type,
        artifact_id,
        [TargetKeys.NGRAMS],
    )
    app_core.session_manager.update_session_state(
        user_session_id,
        SessionKeys.NGRAMS,
        True,
    )
    st.session_state[user_session_id][WarningKeys.NGRAM] = None
    return True


def _load_cached_ngrams(user_session_id: str, identity) -> bool:
    """Attach built-in n-grams/clusters from the shared artifact registry if ready."""

    loaded = shared_artifact_workflow.load_ready(
        identity,
        registry_service.load_ngram_bundle,
        cache_name="ngram",
    )
    if loaded is None:
        return False

    artifact, _ = loaded
    try:
        attach_ngram_artifact(
            user_session_id,
            artifact.artifact_id,
            artifact_type=artifact.artifact_type,
        )
        st.success('N-grams loaded from shared cache!')
        st.rerun()
    except Exception as exc:
        logger.warning(f"Shared n-gram cache load failed: {exc}")
        return False

    return True


def _collocation_parameters(
        node_word: str,
        node_tag: str,
        to_left: int,
        to_right: int,
        stat_mode: str,
        count_by: str
) -> dict:
    """Return the session metadata payload for a collocation table."""

    return {
        'node_word': node_word,
        'node_tag': node_tag,
        'to_left': to_left,
        'to_right': to_right,
        'stat_mode': stat_mode,
        'count_by': count_by
    }


def attach_collocation_artifact(
        user_session_id: str,
        artifact_id: int,
        colloc_params: dict,
        artifact_type: str | None = None,
) -> bool:
    """Attach a ready collocation artifact to the current target corpus session."""

    if artifact_type is None:
        artifact = registry_service.get_artifact_by_id(artifact_id)
        if artifact is None or artifact.status != "ready":
            return False
        artifact_type = artifact.artifact_type

    manager = get_corpus_data_manager(user_session_id, CorpusKeys.TARGET)
    manager.set_artifact_refs(
        artifact_type,
        artifact_id,
        [TargetKeys.COLLOCATIONS],
    )
    app_core.session_manager.update_session_state(
        user_session_id,
        SessionKeys.COLLOCATIONS,
        True,
    )
    app_core.session_manager.update_metadata(
        user_session_id,
        CorpusKeys.TARGET,
        {SessionKeys.COLLOCATIONS: colloc_params}
    )
    st.session_state[user_session_id][WarningKeys.COLLOCATIONS] = None
    return True


def _load_cached_collocations(user_session_id: str, identity, colloc_params: dict) -> bool:
    """Attach built-in collocations from the shared artifact registry if available."""

    loaded = shared_artifact_workflow.load_ready(
        identity,
        registry_service.load_collocation_bundle,
        cache_name="collocation",
    )
    if loaded is None:
        return False

    artifact, _ = loaded
    try:
        attach_collocation_artifact(
            user_session_id,
            artifact.artifact_id,
            colloc_params,
            artifact_type=artifact.artifact_type,
        )
        st.success('Collocations loaded from shared cache!')
        st.rerun()
    except Exception as exc:
        logger.warning(f"Shared collocation cache load failed: {exc}")
        return False

    return True


def generate_ngrams(
        user_session_id: str,
        ngram_span: int,
        ts: str = 'doc_id'  # Default to 'doc_id' for ngram counting
) -> None:
    """
    Generate n-grams for the target corpus based on user input.

    Parameters
    ----------
    user_session_id : str
        The session ID for the user.
    ngram_span : int
        The span of the n-grams to generate (2–10).
    ts : str, optional
        The method to count n-grams, either 'doc_id' or 'token'.
        Defaults to 'doc_id'.

    Returns
    -------
    None
        The function updates the session state with the generated n-grams
        or an error message if the input is invalid.
    """
    # --- User input validation ---
    if not isinstance(ngram_span, int) or ngram_span < 2 or ngram_span > 10:
        st.session_state[user_session_id][WarningKeys.NGRAM] = (
            "Please select a valid n-gram span (2–10).",
            ":material/info:"
        )
        return

    identity = _get_shared_ngram_identity(
        user_session_id,
        analysis_type="ngrams",
        ngram_span=ngram_span,
        count_by=ts,
    )
    if _load_cached_ngrams(user_session_id, identity):
        return

    cache_decision = shared_artifact_workflow.reserve(
        identity,
        cache_name="ngram",
        ready_loader=lambda: _load_cached_ngrams(user_session_id, identity),
        poll_attempts=20,
        poll_interval_seconds=0.25,
    )
    if cache_decision.state == "ready":
        return
    if cache_decision.state == "pending":
        st.session_state[user_session_id][WarningKeys.NGRAM] = (
            "This shared n-grams table is already being prepared. Please try again in a moment.",
            ":material/hourglass_top:"
        )
        return

    job_id = cache_decision.job_id if cache_decision.state == "reserved" else None

    # --- Main logic ---
    manager = get_corpus_data_manager(user_session_id, CorpusKeys.TARGET)
    tok_pl = manager.get_data(TargetKeys.DS_TOKENS)

    if tok_pl is None or getattr(tok_pl, "height", 0) == 0:
        if job_id is not None:
            registry_service.mark_job_failed(job_id, "No tokens found for target corpus")
        st.session_state[user_session_id][WarningKeys.NGRAM] = (
            """
            No tokens found for the target corpus.
            Please load and process a corpus first.
            """,
            ":material/info:"
        )
        return

    try:
        ngram_df = ds.ngrams(
            tokens_table=tok_pl,
            span=ngram_span,
            count_by=ts
        )
    except Exception as exc:
        if job_id is not None:
            registry_service.mark_job_failed(job_id, str(exc))
        raise

    # --- Data-dependent warnings ---
    if ngram_df is None or getattr(ngram_df, "height", 0) < 2:
        if job_id is not None:
            registry_service.mark_job_failed(job_id, "N-gram search returned no results")
        st.session_state[user_session_id][WarningKeys.NGRAM] = (
            "Your search didn't return any results.",
            ":material/info:"
        )
        return

    # --- Success ---
    artifact = shared_artifact_workflow.store(
        identity,
        job_id,
        cache_name="ngram",
        store_func=lambda artifact_identity: registry_service.store_ngram_bundle(
            artifact_identity,
            ngram_df,
        ),
    )
    if artifact is not None:
        attach_ngram_artifact(
            user_session_id,
            artifact.artifact_id,
            artifact_type=artifact.artifact_type,
        )
    else:
        manager.set_data(TargetKeys.NGRAMS, ngram_df)
        app_core.session_manager.update_session_state(
            user_session_id,
            SessionKeys.NGRAMS,
            True
        )
    st.session_state[user_session_id][WarningKeys.NGRAM] = None
    st.rerun()


def generate_clusters(
    user_session_id: str,
    from_anchor: str,
    node_word: str,
    tag: str,
    position: int,
    ngram_span: int,
    search: str,
    ts: str = 'doc_id'
):
    # --- User input validation ---
    if from_anchor == 'Token':
        if not node_word or node_word == 'by_tag':
            st.session_state[user_session_id][WarningKeys.NGRAM] = (
                "Please enter a node word.",
                ":material/info:"
            )
            return
        if " " in node_word:
            st.session_state[user_session_id][WarningKeys.NGRAM] = (
                "Node word cannot contain spaces.",
                ":material/info:"
            )
            return
        if len(node_word) > 15:
            st.session_state[user_session_id][WarningKeys.NGRAM] = (
                "Node word is too long (max 15 characters).",
                ":material/info:"
            )
            return
    elif from_anchor == 'Tag':
        if not tag or tag == 'No tags currently loaded':
            st.session_state[user_session_id][WarningKeys.NGRAM] = (
                "Please select a valid tag.",
                ":material/info:"
            )
            return

    ngram_params = _ngram_parameters(
        analysis_type="clusters",
        ngram_span=ngram_span,
        count_by=ts,
        from_anchor=from_anchor,
        node_word=node_word,
        tag=tag,
        position=position,
        search_type=search,
    )
    identity = _get_shared_ngram_identity(user_session_id, **ngram_params)
    if _load_cached_ngrams(user_session_id, identity):
        return

    cache_decision = shared_artifact_workflow.reserve(
        identity,
        cache_name="ngram",
        ready_loader=lambda: _load_cached_ngrams(user_session_id, identity),
        poll_attempts=20,
        poll_interval_seconds=0.25,
    )
    if cache_decision.state == "ready":
        return
    if cache_decision.state == "pending":
        st.session_state[user_session_id][WarningKeys.NGRAM] = (
            "This shared cluster table is already being prepared. Please try again in a moment.",
            ":material/hourglass_top:"
        )
        return

    job_id = cache_decision.job_id if cache_decision.state == "reserved" else None

    # --- Main logic ---
    manager = get_corpus_data_manager(user_session_id, CorpusKeys.TARGET)
    tok_pl = manager.get_data(TargetKeys.DS_TOKENS)

    if tok_pl is None:
        if job_id is not None:
            registry_service.mark_job_failed(job_id, "No corpus data available")
        st.session_state[user_session_id][WarningKeys.NGRAM] = (
            "No corpus data available. Please load a corpus first.",
            ":material/error:"
        )
        return

    ngram_df = None
    try:
        if from_anchor == 'Token':
            ngram_df = ds.clusters_by_token(
                tokens_table=tok_pl,
                node_word=node_word,
                node_position=position,
                span=ngram_span,
                search_type=search,
                count_by=ts
            )
        elif from_anchor == 'Tag':
            ngram_df = ds.clusters_by_tag(
                tokens_table=tok_pl,
                tag=tag,
                tag_position=position,
                span=ngram_span,
                count_by=ts
            )
    except Exception as exc:
        if job_id is not None:
            registry_service.mark_job_failed(job_id, str(exc))
        raise

    # --- Data-dependent warnings ---
    if ngram_df is None or getattr(ngram_df, "height", 0) == 0:
        if job_id is not None:
            registry_service.mark_job_failed(job_id, "Cluster search returned no matches")
        st.session_state[user_session_id][WarningKeys.NGRAM] = (
            "Your search didn't return any matches. Try something else.",
            ":material/info:"
        )
        return
    elif getattr(ngram_df, "height", 0) > 100000:
        if job_id is not None:
            registry_service.mark_job_failed(job_id, "Cluster search returned too many matches")
        st.session_state[user_session_id][WarningKeys.NGRAM] = (
            "Your search returned too many matches! Try something more specific.",
            ":material/info:"
        )
        return

    # --- Success ---
    artifact = shared_artifact_workflow.store(
        identity,
        job_id,
        cache_name="ngram",
        store_func=lambda artifact_identity: registry_service.store_ngram_bundle(
            artifact_identity,
            ngram_df,
        ),
    )
    if artifact is not None:
        attach_ngram_artifact(
            user_session_id,
            artifact.artifact_id,
            artifact_type=artifact.artifact_type,
        )
    else:
        manager.set_data(TargetKeys.NGRAMS, ngram_df)
        app_core.session_manager.update_session_state(
            user_session_id,
            SessionKeys.NGRAMS,
            True,
        )
    st.session_state[user_session_id][WarningKeys.NGRAM] = None
    st.rerun()


def generate_kwic(
        user_session_id: str,
        node_word: str,
        search_type: str,
        ignore_case: bool
) -> None:
    """
    Generate a KWIC (Key Word in Context) table for the target corpus
    based on user input.
    Parameters
    ----------
    user_session_id : str
        The session ID for the user.
    node_word : str
        The word to search for in the KWIC table.
    search_type : str
        The type of search to perform, either 'fixed',
        'startswith', 'endswith', or 'contains'.
    ignore_case : bool
        Whether to ignore case when searching for the node word.
    Returns
    -------
    None
        The function updates the session state with the generated KWIC table
        or an error message if the input is invalid.
    """
    # --- User input validation ---
    if not node_word or len(node_word.strip()) == 0:
        st.session_state[user_session_id][WarningKeys.KWIC] = (
            "Please enter a search term.",
            ":material/info:"
        )
        return

    # --- Get tokens table ---
    manager = get_corpus_data_manager(user_session_id, CorpusKeys.TARGET)
    tok_pl = manager.get_data(TargetKeys.DS_TOKENS)

    if tok_pl is None:
        st.session_state[user_session_id][WarningKeys.KWIC] = (
            "KWIC table cannot be generated: no tokens found in the target corpus.",
            ":material/sentiment_stressed:"
        )
        return

    if tok_pl is None or getattr(tok_pl, "height", 0) == 0:
        st.session_state[user_session_id][WarningKeys.KWIC] = (
            "KWIC table cannot be generated: no tokens found in the target corpus.",
            ":material/sentiment_stressed:"
        )
        return

    # --- Generate KWIC table ---
    kwic_df = ds.kwic_center_node(
        tok_pl,
        node_word=node_word,
        search_type=search_type,
        ignore_case=ignore_case
    )

    if kwic_df is None or getattr(kwic_df, "height", 0) == 0:
        st.session_state[user_session_id][WarningKeys.KWIC] = (
            "No results found for the given search term.",
            ":material/info:"
        )
        return

    # Ensure target corpus dict exists before storing KWIC
    if CorpusKeys.TARGET not in st.session_state[user_session_id]:
        st.session_state[user_session_id][CorpusKeys.TARGET] = {}

    # Store KWIC result with defensive coding
    try:
        target_dict = st.session_state[user_session_id][CorpusKeys.TARGET]
        target_dict[TargetKeys.KWIC] = kwic_df
    except AttributeError as e:
        # Fallback in case of import issues
        st.session_state[user_session_id][CorpusKeys.TARGET]["kwic"] = kwic_df
        st.error(f"Session key access error: {e}. Using fallback key.")

    st.session_state[user_session_id][WarningKeys.KWIC] = None

    app_core.session_manager.update_session_state(
        user_session_id,
        SessionKeys.KWIC,
        True
        )
    st.success('KWIC table generated!')
    st.rerun()


def generate_collocations(
        user_session_id: str,
        node_word: str,
        node_tag: str,
        to_left: int,
        to_right: int,
        stat_mode: str,
        count_by: str
        ) -> None:
    """
    Generate collocations for the target corpus based on user input.

    Parameters
    ----------
    user_session_id : str
        The session ID for the user.
    node_word : str
        The word to search for in the collocations.
    node_tag : str
        The part-of-speech tag to filter the collocations.
    to_left : int
        The number of tokens to include to the left of the node word.
    to_right : int
        The number of tokens to include to the right of the node word.
    stat_mode : str
        The statistical mode to use for collocation analysis.
        Should be one of 'raw', 'log-likelihood', 't-score', or 'mi'.
    count_by : str
        The method to count collocations, either 'tokens' or 'documents'.

    Returns
    -------
    None
        The function updates the session state with the generated collocations
        or an error message if the input is invalid.
    """
    # --- User input validation ---
    if not node_word:
        st.session_state[user_session_id][WarningKeys.COLLOCATIONS] = (
            "Please enter a node word.",
            ":material/info:"
        )
        return
    if " " in node_word:
        st.session_state[user_session_id][WarningKeys.COLLOCATIONS] = (
            "Node word cannot contain spaces.",
            ":material/info:"
        )
        return
    if len(node_word) > 15:
        st.session_state[user_session_id][WarningKeys.COLLOCATIONS] = (
            "Node word is too long (max 15 characters).",
            ":material/info:"
        )
        return

    colloc_params = _collocation_parameters(
        node_word,
        node_tag,
        to_left,
        to_right,
        stat_mode,
        count_by,
    )
    identity = _get_shared_collocation_identity(
        user_session_id,
        node_word,
        node_tag,
        to_left,
        to_right,
        stat_mode,
        count_by,
    )

    if _load_cached_collocations(user_session_id, identity, colloc_params):
        return

    cache_decision = shared_artifact_workflow.reserve(
        identity,
        cache_name="collocation",
        ready_loader=lambda: _load_cached_collocations(
            user_session_id,
            identity,
            colloc_params,
        ),
        poll_attempts=20,
        poll_interval_seconds=0.25,
    )
    if cache_decision.state == "ready":
        return
    if cache_decision.state == "pending":
        st.session_state[user_session_id][WarningKeys.COLLOCATIONS] = (
            "This shared collocations table is already being prepared. Please try again in a moment.",
            ":material/hourglass_top:"
        )
        return

    job_id = cache_decision.job_id if cache_decision.state == "reserved" else None

    # --- Main logic ---
    manager = get_corpus_data_manager(user_session_id, "target")
    tok_pl = manager.get_data(TargetKeys.DS_TOKENS)

    if tok_pl is None or getattr(tok_pl, "height", 0) == 0:
        if job_id is not None:
            registry_service.mark_job_failed(job_id, "No tokens found for target corpus")
        st.session_state[user_session_id][WarningKeys.COLLOCATIONS] = (
            """
            No tokens found for the target corpus.
            Please load and process a corpus first.
            """,
            ":material/sentiment_stressed:"
        )
        return

    try:
        coll_df = ds.coll_table(
            tok_pl,
            node_word=node_word,
            node_tag=node_tag,
            preceding=to_left,
            following=to_right,
            statistic=stat_mode,
            count_by=count_by
        )
    except Exception as exc:
        if job_id is not None:
            registry_service.mark_job_failed(job_id, str(exc))
        raise

    # --- Data-dependent warnings ---
    if coll_df is None or coll_df.is_empty():
        if job_id is not None:
            registry_service.mark_job_failed(job_id, "Collocation search returned no matches")
        st.session_state[user_session_id][WarningKeys.COLLOCATIONS] = (
            "Your search didn't return any matches. Try something else.",
            ":material/info:"
        )
        return

    # --- Success ---
    artifact = shared_artifact_workflow.store(
        identity,
        job_id,
        cache_name="collocation",
        store_func=lambda artifact_identity: registry_service.store_collocation_bundle(
            artifact_identity,
            coll_df,
        ),
    )
    if artifact is not None:
        attach_collocation_artifact(
            user_session_id,
            artifact.artifact_id,
            colloc_params,
            artifact_type=artifact.artifact_type,
        )
    else:
        # Store collocations result using the new corpus data manager
        manager.set_data(TargetKeys.COLLOCATIONS, coll_df)

        app_core.session_manager.update_session_state(
            user_session_id,
            SessionKeys.COLLOCATIONS,
            True,
        )

        app_core.session_manager.update_metadata(
            user_session_id,
            CorpusKeys.TARGET,
            {SessionKeys.COLLOCATIONS: colloc_params}
        )

    st.session_state[user_session_id][WarningKeys.COLLOCATIONS] = None
    st.success('Collocations generated!')
    st.rerun()
