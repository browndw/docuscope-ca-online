"""
Corpus generation utilities for n-grams, collocations, and KWIC analysis.

This module provides functions for generating various types of corpus
analyses including n-grams, collocations, and keyword-in-context searches.
"""

import streamlit as st
import docuscospacy as ds

from webapp.utilities.core import app_core
from webapp.utilities.state import CorpusKeys, TargetKeys, WarningKeys, SessionKeys
from webapp.utilities.corpus import get_corpus_data_manager


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

    # --- Main logic ---
    manager = get_corpus_data_manager(user_session_id, CorpusKeys.TARGET)
    tok_pl = manager.get_data(TargetKeys.DS_TOKENS)

    if tok_pl is None or getattr(tok_pl, "height", 0) == 0:
        st.session_state[user_session_id][WarningKeys.NGRAM] = (
            """
            No tokens found for the target corpus.
            Please load and process a corpus first.
            """,
            ":material/info:"
        )
        return

    ngram_df = ds.ngrams(
        tokens_table=tok_pl,
        span=ngram_span,
        count_by=ts
    )

    # --- Data-dependent warnings ---
    if ngram_df is None or getattr(ngram_df, "height", 0) < 2:
        st.session_state[user_session_id][WarningKeys.NGRAM] = (
            "Your search didn't return any results.",
            ":material/info:"
        )
        return

    # --- Success ---
    st.session_state[user_session_id][CorpusKeys.TARGET][TargetKeys.NGRAMS] = ngram_df
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

    # --- Main logic ---
    manager = get_corpus_data_manager(user_session_id, CorpusKeys.TARGET)
    tok_pl = manager.get_data(TargetKeys.DS_TOKENS)

    if tok_pl is None:
        st.session_state[user_session_id][WarningKeys.NGRAM] = (
            "No corpus data available. Please load a corpus first.",
            ":material/error:"
        )
        return

    ngram_df = None
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

    # --- Data-dependent warnings ---
    if ngram_df is None or getattr(ngram_df, "height", 0) == 0:
        st.session_state[user_session_id][WarningKeys.NGRAM] = (
            "Your search didn't return any matches. Try something else.",
            ":material/info:"
        )
        return
    elif getattr(ngram_df, "height", 0) > 100000:
        st.session_state[user_session_id][WarningKeys.NGRAM] = (
            "Your search returned too many matches! Try something more specific.",
            ":material/info:"
        )
        return

    # --- Success ---
    st.session_state[user_session_id][CorpusKeys.TARGET][TargetKeys.NGRAMS] = ngram_df
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

    # --- Main logic ---
    manager = get_corpus_data_manager(user_session_id, "target")
    tok_pl = manager.get_data(TargetKeys.DS_TOKENS)

    if tok_pl is None or getattr(tok_pl, "height", 0) == 0:
        st.session_state[user_session_id][WarningKeys.COLLOCATIONS] = (
            """
            No tokens found for the target corpus.
            Please load and process a corpus first.
            """,
            ":material/sentiment_stressed:"
        )
        return

    coll_df = ds.coll_table(
        tok_pl,
        node_word=node_word,
        node_tag=node_tag,
        preceding=to_left,
        following=to_right,
        statistic=stat_mode,
        count_by=count_by
    )

    # --- Data-dependent warnings ---
    if coll_df is None or coll_df.is_empty():
        st.session_state[user_session_id][WarningKeys.COLLOCATIONS] = (
            "Your search didn't return any matches. Try something else.",
            ":material/info:"
        )
        return

    # --- Success ---
    # Store collocations result using the new corpus data manager
    manager.set_data("collocations", coll_df)

    app_core.session_manager.update_session_state(
        user_session_id,
        'collocations',
        True,
    )

    # Store collocation parameters as a dictionary in metadata
    colloc_params = {
        'node_word': node_word,
        'node_tag': node_tag,
        'to_left': to_left,
        'to_right': to_right,
        'stat_mode': stat_mode,
        'count_by': count_by
    }
    app_core.session_manager.update_metadata(
        user_session_id,
        CorpusKeys.TARGET,
        {'collocations': colloc_params}
    )

    st.session_state[user_session_id][WarningKeys.COLLOCATIONS] = None
    st.success('Collocations generated!')
    st.rerun()
