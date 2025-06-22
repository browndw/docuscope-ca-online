"""
Download handlers for corpus and tagged file exports.

This module provides standardized handlers for different types of download operations
used in the download pages.
"""

import streamlit as st
from webapp.utilities.exports import convert_corpus_to_zip, convert_to_zip
from webapp.utilities.state import CorpusKeys, TargetKeys, ReferenceKeys
from webapp.utilities.ui.download_interface import render_download_button


def handle_corpus_file_download(
    user_session_id: str,
    corpus_type: str
) -> None:
    """
    Handle single corpus file download (Parquet format).

    Parameters
    ----------
    user_session_id : str
        The user session identifier
    corpus_type : str
        Either "Target" or "Reference"
    """
    if corpus_type == "Target":
        corpus_session = st.session_state[user_session_id][CorpusKeys.TARGET]
        tokens_df = corpus_session[TargetKeys.DS_TOKENS]
    else:
        corpus_session = st.session_state[user_session_id][CorpusKeys.REFERENCE]
        tokens_df = corpus_session[ReferenceKeys.DS_TOKENS]

    download_file = tokens_df.to_pandas().to_parquet()

    render_download_button(
        label="Download Corpus File",
        data=download_file,
        filename="corpus.parquet",
        mime_type="parquet",
        message="Click the button to download your corpus file."
    )


def handle_all_data_download(
    user_session_id: str,
    corpus_type: str,
    file_format: str
) -> None:
    """
    Handle download of all processed data.

    Parameters
    ----------
    user_session_id : str
        The user session identifier
    corpus_type : str
        Either "Target" or "Reference"
    file_format : str
        Either "CSV" or "PARQUET"
    """
    corpus_key = CorpusKeys.TARGET if corpus_type == "Target" else CorpusKeys.REFERENCE

    with st.sidebar.status("Preparing files..."):
        if file_format == "CSV":
            download_file = convert_corpus_to_zip(
                user_session_id,
                corpus_key,
                file_type='csv'
            )
        else:
            download_file = convert_corpus_to_zip(
                user_session_id,
                corpus_key
            )

    render_download_button(
        label="Download Corpus Files",
        data=download_file,
        filename="corpus_files.zip",
        mime_type="application/zip",
        message="Click the button to download your corpus files."
    )


def handle_tagged_files_download(
    user_session_id: str,
    tagset: str
) -> None:
    """
    Handle tagged files download.

    Parameters
    ----------
    user_session_id : str
        The user session identifier
    tagset : str
        Either "pos" or "ds"
    """
    target_session = st.session_state[user_session_id][CorpusKeys.TARGET]
    tok_pl = target_session[TargetKeys.DS_TOKENS]

    download_file = convert_to_zip(tok_pl, tagset)

    st.sidebar.download_button(
        label="Download to Zip",
        data=download_file,
        file_name="tagged_files.zip",
        mime="application/zip",
    )
