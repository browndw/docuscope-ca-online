"""
Corpus loading utilities for loading corpus data from files and database paths.

This module provides functions for loading corpus data from various sources
including compressed files and new dataframes. Updated to use the new
corpus data manager for memory-efficient lazy loading.
"""

import os
import gzip
import glob
import pickle
import random
import time
import polars as pl
import streamlit as st

from webapp.utilities.corpus import get_corpus_manager


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
    manager = get_corpus_manager(session_id, corpus_type)

    files_list = glob.glob(os.path.join(db_path, '*.gz'))

    # Prioritize core data file
    core_files = [f for f in files_list if 'ds_tokens' in f]
    other_files = [f for f in files_list if 'ds_tokens' not in f]

    # Try up to 3 times to load files
    for attempt in range(3):
        # Shuffle non-core files to prevent contention in concurrent access
        random.shuffle(other_files)
        # Always try core files first
        ordered_files = core_files + other_files

        data = {}

        # Attempt to load all files
        for file in ordered_files:
            try:
                with gzip.open(file, 'rb') as f:
                    file_key = str(os.path.basename(file)).removesuffix(".gz")
                    data[file_key] = pickle.load(f)
            except Exception:
                # Silently continue on individual file failures
                pass

        # Check if we successfully loaded core data at minimum
        if 'ds_tokens' in data:
            # Load all available data through the manager
            manager.load_all_data(data)

            # Data loaded successfully - no console output needed for deployed app
            # UI will show corpus loading status through session state

            return

        # If this wasn't the last attempt, we'll try again
        if attempt < 2:
            # Brief pause before retry to reduce contention
            time.sleep(0.1 * (attempt + 1))  # Increasing delay

    # All 3 attempts failed - show error to user
    files_loaded = len(data) if 'data' in locals() else 0
    core_loaded = 'ds_tokens' in data if 'data' in locals() else False

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
