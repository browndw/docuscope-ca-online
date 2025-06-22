"""
Corpus loading utilities for loading corpus data from files and database paths.

This module provides functions for loading corpus data from various sources
including compressed files and new dataframes.
"""

import os
import gzip
import glob
import pickle
import random
import time
import polars as pl
import streamlit as st


def load_corpus_internal(db_path: str,
                         session_id: str,
                         corpus_type='target'):
    """
    Load a corpus from the specified database path into the session state.

    Implements robust loading logic for concurrent access:
    1. Shuffles file order to avoid contention
    2. Attempts to load all 7 required files
    3. If unsuccessful, reshuffles and retries up to 3 times total
    4. Shows error and asks user to retry if all attempts fail

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
    if corpus_type not in st.session_state[session_id]:
        st.session_state[session_id][corpus_type] = {}

    files_list = glob.glob(os.path.join(db_path, '*.gz'))

    # Try up to 3 times to load all 7 files
    for attempt in range(3):
        # Shuffle files to prevent contention in concurrent access
        random.shuffle(files_list)
        data = {}

        # Attempt to load all files
        for file in files_list:
            try:
                with gzip.open(file, 'rb') as f:
                    data[
                        str(os.path.basename(file)).removesuffix(".gz")
                        ] = pickle.load(f)
            except Exception:
                # Silently continue on individual file failures
                pass

        # Check if we successfully loaded all 7 required files
        if len(data) == 7:
            # Success! Load data into session state
            for key, value in data.items():
                if key not in st.session_state[session_id][corpus_type]:
                    st.session_state[session_id][corpus_type][key] = {}
                st.session_state[session_id][corpus_type][key] = value
            return

        # If this wasn't the last attempt, we'll try again
        if attempt < 2:
            # Brief pause before retry to reduce contention
            time.sleep(0.1 * (attempt + 1))  # Increasing delay

    # All 3 attempts failed - show error to user
    files_loaded = len(data) if 'data' in locals() else 0
    st.error(
        f"""
        **Unable to load internal corpus data**

        The system was unable to load all required corpus files after 3 attempts.
        This can happen when many users are accessing the same corpus simultaneously.

        **What to try:**
        - Wait a moment and try loading the corpus again
        - Try selecting a different corpus if available
        - If the problem persists, please contact support

        **Technical details:** Expected 7 files, loaded {files_loaded}
        """,
        icon=":material/error:"
    )


def load_corpus_new(ds_tokens: pl.DataFrame,
                    dtm_ds: pl.DataFrame,
                    dtm_pos: pl.DataFrame,
                    ft_ds: pl.DataFrame,
                    ft_pos: pl.DataFrame,
                    tt_ds: pl.DataFrame,
                    tt_pos: pl.DataFrame,
                    session_id: str,
                    corpus_type='target') -> None:
    """
    Load new corpus dataframes into the session state
    for a given session ID and corpus type.

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
    if corpus_type not in st.session_state[session_id]:
        st.session_state[session_id][corpus_type] = {}

    # Store the dataframes in session state
    st.session_state[session_id][corpus_type]['ds_tokens'] = ds_tokens
    st.session_state[session_id][corpus_type]['dtm_ds'] = dtm_ds
    st.session_state[session_id][corpus_type]['dtm_pos'] = dtm_pos
    st.session_state[session_id][corpus_type]['ft_ds'] = ft_ds
    st.session_state[session_id][corpus_type]['ft_pos'] = ft_pos
    st.session_state[session_id][corpus_type]['tt_ds'] = tt_ds
    st.session_state[session_id][corpus_type]['tt_pos'] = tt_pos
