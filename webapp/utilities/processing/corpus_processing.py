"""
Corpus processing functions for handling different types of corpus uploads and processing.

This module provides functions for processing internal corpora, external corpora,
newly uploaded text, and corpus finalization.
"""

import os
import unidecode
import polars as pl
import streamlit as st
import docuscospacy as ds

# Module-specific imports
from webapp.utilities.processing.corpus_loading import load_corpus_internal, load_corpus_new
from webapp.utilities.session import (
    init_metadata_target, init_metadata_reference
)
from webapp.utilities.analysis.data_validation import check_corpus_external, check_corpus_new  # noqa: E501
from webapp.utilities.state import LoadCorpusKeys, SessionKeys
from webapp.utilities.corpus import get_corpus_manager
from webapp.utilities.core import app_core

# Warning constants for corpus processing
WARNING_CORRUPT_TARGET = 10
WARNING_CORRUPT_REFERENCE = 11
WARNING_DUPLICATE_REFERENCE = 21
WARNING_EXCLUDED_TARGET = 40
WARNING_EXCLUDED_REFERENCE = 41


def finalize_corpus_load(ds_tokens, user_session_id: str, corpus_type: str) -> None:
    """
    Finalize corpus loading by generating frequency tables, tags tables, and DTMs.

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
    # Generate all the required tables and matrices
    ft_pos, ft_ds = ds.frequency_table(ds_tokens, count_by="both")
    tt_pos, tt_ds = ds.tags_table(ds_tokens, count_by="both")
    dtm_pos, dtm_ds = ds.tags_dtm(ds_tokens, count_by="both")

    # Load the processed corpus into session state
    load_corpus_new(
        ds_tokens,
        dtm_ds, dtm_pos,
        ft_ds, ft_pos,
        tt_ds, tt_pos,
        user_session_id, corpus_type
    )

    # Initialize metadata and update session flags
    if corpus_type == 'target':
        init_metadata_target(user_session_id)
        app_core.session_manager.update_session_state(
            user_session_id, SessionKeys.HAS_TARGET, True
        )
    else:
        init_metadata_reference(user_session_id)
        app_core.session_manager.update_session_state(
            user_session_id, SessionKeys.HAS_REFERENCE, True
        )

    # Clean up the original corpus DataFrame to free memory
    cleanup_original_corpus_data(user_session_id, corpus_type)

    st.rerun()


def finalize_corpus_load_optimized(
    ds_tokens, user_session_id: str, corpus_type: str
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

    # Only set core data - derived data will be generated on-demand
    manager.set_core_data(ds_tokens)

    # Initialize metadata and update session flags
    if corpus_type == 'target':
        init_metadata_target(user_session_id)
        app_core.session_manager.update_session_state(
            user_session_id, SessionKeys.HAS_TARGET, True
        )
    else:
        init_metadata_reference(user_session_id)
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
        exceptions=None
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
            finalize_corpus_load(ds_tokens, user_session_id, corpus_type)
        else:
            # Processing completed successfully
            st.success('Processing complete!')
            st.session_state[user_session_id]['warning'] = 0
            finalize_corpus_load(ds_tokens, user_session_id, corpus_type)

    except Exception as e:
        corpus_name = "reference" if corpus_type == "reference" else "target"
        st.error(f"Error processing {corpus_name} corpus: {str(e)}")
        # Don't call st.rerun() if there was an error


def process_external(
        df,
        user_session_id: str,
        corpus_type: str
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
    finalize_corpus_load(ds_tokens, user_session_id, corpus_type)


def process_internal(
        corp_path: str,
        user_session_id: str,
        corpus_type: str
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
        # Load the internal corpus
        load_corpus_internal(
            corp_path,
            user_session_id,
            corpus_type=corpus_type
        )

        # Verify the corpus was loaded successfully
        if (corpus_type not in st.session_state[user_session_id] or
                'ds_tokens' not in st.session_state[user_session_id][corpus_type]):
            st.error(f"Failed to load {corpus_type} corpus data.")
            return

        # Update session state based on corpus type
        if corpus_type == "target":
            init_metadata_target(user_session_id)
            app_core.session_manager.update_session_state(
                user_session_id, SessionKeys.TARGET_DB, str(corp_path)
            )
            app_core.session_manager.update_session_state(
                user_session_id, SessionKeys.HAS_TARGET, True
            )
        else:
            init_metadata_reference(user_session_id)
            app_core.session_manager.update_session_state(
                user_session_id, SessionKeys.REFERENCE_DB, str(corp_path)
            )
            app_core.session_manager.update_session_state(
                user_session_id, SessionKeys.HAS_REFERENCE, True
            )
        st.rerun()

    except Exception as e:
        st.error(f"Error processing {corpus_type} corpus: {str(e)}")
        # Don't call st.rerun() if there was an error


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
            process_fn()
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
