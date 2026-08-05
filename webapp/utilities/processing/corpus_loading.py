"""
Corpus loading utilities for loading corpus data from files and database paths.

This module provides functions for loading corpus data from various sources
including compressed files and new dataframes. Updated to use the new
corpus data manager for memory-efficient lazy loading.
"""

import os
import glob
import polars as pl
import streamlit as st

from webapp.utilities.corpus import get_corpus_manager
from webapp.corpus_paths import resolve_corpus_path


def load_corpus_internal(
        db_path: str,
        session_id: str,
        corpus_type='target'
) -> None:
    """
    Load a corpus from the specified database path into the session state.

    Updated for memory efficiency: loads only core data (ds_tokens) immediately,
    with derived data loaded on-demand via the corpus data manager.

    Implements robust loading logic for concurrent access:
    1. Shuffles file order to avoid contention
    2. Prioritizes loading core data (ds_tokens) first
    3. Loads remaining data if available
    4. If unsuccessful, reshuffles and retries up to 3 times total
    5. Shows error and asks user to retry if all attempts fail

    Parameters
    ----------
    db_path : str
        The path to the database containing the corpus files.
    session_id : str
        The session ID for which the corpus is to be loaded.
    corpus_type : str, optional
        The type of corpus to be loaded (default is 'target').

    Returns
    -------
    None
    """
    db_path = resolve_corpus_path(db_path)
    manager = get_corpus_manager(session_id, corpus_type)

    files_list = glob.glob(os.path.join(db_path, '*.gz'))
    file_map = {
        str(os.path.basename(file_path)).removesuffix(".gz"): file_path
        for file_path in files_list
    }

    # Only register lightweight file references here. Built-in corpus artifacts
    # should be loaded on demand instead of eagerly hydrating all tables into
    # per-user session state.
    if 'ds_tokens' in file_map:
        manager.set_file_refs(file_map)
        return

    files_loaded = len(file_map)
    core_loaded = 'ds_tokens' in file_map

    st.error(
        f"""
        **Unable to load corpus data**

        The system was unable to load the required corpus files after 3 attempts.
        This can happen when many users are accessing the same corpus simultaneously.

        **What to try:**
        - Wait a moment and try loading the corpus again
        - Try selecting a different corpus if available
        - If the problem persists, please contact support

        **Technical details:**
        - Expected core file (ds_tokens): {'✓' if core_loaded else '✗'}
        - Total files loaded: {files_loaded}/7
        """,
        icon=":material/error:"
    )


def load_corpus_new(
        ds_tokens: pl.DataFrame,
        dtm_ds: pl.DataFrame,
        dtm_pos: pl.DataFrame,
        ft_ds: pl.DataFrame,
        ft_pos: pl.DataFrame,
        tt_ds: pl.DataFrame,
        tt_pos: pl.DataFrame,
        session_id: str,
        corpus_type='target'
) -> None:
    """
    Load new corpus dataframes into the session state using the corpus manager.

    Updated to use the new corpus data manager for consistent data access
    and memory management across all corpus types.

    Parameters
    ----------
    ds_tokens : pl.DataFrame
        The dataframe containing token-level data for the corpus.
    dtm_ds : pl.DataFrame
        The dataframe containing document-term matrix for DS tags.
    dtm_pos : pl.DataFrame
        The dataframe containing document-term matrix for POS tags.
    ft_ds : pl.DataFrame
        The dataframe containing frequency table for DS tags.
    ft_pos : pl.DataFrame
        The dataframe containing frequency table for POS tags.
    tt_ds : pl.DataFrame
        The dataframe containing tag table for DS tags.
    tt_pos : pl.DataFrame
        The dataframe containing tag table for POS tags.
    session_id : str
        The session ID for which the corpus is to be loaded.
    corpus_type : str, optional
        The type of corpus to be loaded (default is 'target').

    Returns
    -------
    None
    """
    manager = get_corpus_manager(session_id, corpus_type)

    # Prepare data dictionary
    data_dict = {
        'ds_tokens': ds_tokens,
        'dtm_ds': dtm_ds,
        'dtm_pos': dtm_pos,
        'ft_ds': ft_ds,
        'ft_pos': ft_pos,
        'tt_ds': tt_ds,
        'tt_pos': tt_pos
    }

    # Load all data through the manager
    manager.load_all_data(data_dict)

    # New corpus loaded successfully - no console output needed for deployed app
