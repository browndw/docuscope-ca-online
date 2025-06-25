"""
Data validation utilities for corpus analysis.

This module provides functions for validating corpus data, checking schemas,
language detection, and ensuring data quality.
"""

import os
import random
import unidecode

import pandas as pd
import polars as pl
import streamlit as st

from collections import OrderedDict
from lingua import Language
from typing import Optional, List, Union

from webapp.utilities.analysis.corpus_loading import load_detector
from webapp.utilities.state import SessionKeys
from webapp.utilities.session.session_core import safe_session_get


def check_language(text_str: str) -> bool:
    """
    Check if the given text is in English based on language detection model.

    Parameters
    ----------
    text_str : str
        The text string to be checked.

    Returns
    -------
    bool
        True if the text is detected as English with high confidence.
    """
    detect_model = load_detector()
    detect_language = Language.ENGLISH

    # Clean up the text string
    doc_len = len(text_str)
    predictions = []

    if doc_len > 5000:
        # Sample multiple chunks for longer texts
        idx_a = random.randint(0, doc_len - 1500)
        idx_b = random.randint(0, doc_len - 1500)
        idx_c = random.randint(0, doc_len - 1500)

        sample_a = text_str[idx_a:idx_a + 1000]
        sample_a = " ".join(sample_a.split())
        sample_b = text_str[idx_b:idx_b + 1000]
        sample_b = " ".join(sample_b.split())
        sample_c = text_str[idx_c:idx_c + 1000]
        sample_c = " ".join(sample_c.split())

        text_sample = [sample_a, sample_b, sample_c]

        # Get prediction for each chunk
        for chunk in text_sample:
            value = detect_model.compute_language_confidence(
                chunk,
                detect_language
                )
            predictions.append(value)
    else:
        text_str = " ".join(text_str.split())
        value = detect_model.compute_language_confidence(
            text_str,
            detect_language
            )
        predictions.append(value)

    confidence = sum(predictions) / len(predictions)

    # Only want to know if this is English or not
    return confidence > 0.9


def check_schema(tok_pl: pl.DataFrame) -> bool:
    """
    Validate the schema of a Polars DataFrame.

    Parameters
    ----------
    tok_pl : pl.DataFrame
        A Polars DataFrame containing the corpus data.

    Returns
    -------
    bool
        True if the schema of the DataFrame matches the expected schema,
        False otherwise.
    """
    validation = OrderedDict(
        [
            ('doc_id', pl.String),
            ('token', pl.String),
            ('pos_tag', pl.String),
            ('ds_tag', pl.String),
            ('pos_id', pl.UInt32),
            ('ds_id', pl.UInt32)
            ])
    return tok_pl.schema == validation


def check_corpus_new(
        docs: list,
        check_size=False,
        check_language_flag=False,
        check_ref=False,
        target_docs=None
        ) -> Union[tuple, list]:
    """
    Check the corpus for duplicates, size, reference documents, and language.

    Parameters
    ----------
    docs : list
        A list of document objects to be checked.
    check_size : bool, optional
        If True, calculate the total size of the corpus (default is False).
    check_language_flag : bool, optional
        If True, check the language of the documents (default is False).
    check_ref : bool, optional
        If True, check for reference documents in the corpus (default is False).
    target_docs : list, optional
        A list of target document identifiers to check against (default is None).

    Returns
    -------
    tuple or list
        A tuple containing:
        - dup_ids (list): A list of duplicate document identifiers.
        - dup_docs (list, optional): A list of documents found in both the
          corpus and target_docs (only if check_ref is True).
        - lang_fail (list, optional): A list of doc_ids that fail the language check
          (only if check_language_flag is True).
        - corpus_size (int, optional): The total size of the corpus in bytes
          (only if check_size is True).
    """
    if len(docs) > 0:
        all_files = []
        if check_size:
            for file in docs:
                bytes_data = file.getvalue()
                file_size = len(bytes_data)
                all_files.append(file_size)
            corpus_size = sum(all_files)
        # check for duplicates
        doc_ids = [str(os.path.splitext(doc.name)[0]) for doc in docs]
        doc_ids = [doc.replace(" ", "") for doc in doc_ids]
        if len(doc_ids) > len(set(doc_ids)):
            dup_ids = [x for x in doc_ids if doc_ids.count(x) >= 2]
            dup_ids = list(set(dup_ids))
        else:
            dup_ids = []
        if check_ref and target_docs is not None:
            dup_docs = list(set(target_docs).intersection(doc_ids))
        else:
            dup_docs = []
        if check_language_flag:
            # Check language of each document
            lang_fail = []
            for doc in docs:
                try:
                    doc_txt = doc.getvalue().decode('utf-8')
                except Exception:
                    lang_fail.append(
                        str(os.path.splitext(doc.name.replace(" ", ""))[0])
                        )
                    continue
                if not check_language(doc_txt):
                    lang_fail.append(
                        str(os.path.splitext(doc.name.replace(" ", ""))[0])
                        )
        else:
            lang_fail = []
    else:
        corpus_size = 0
        dup_ids = []
        dup_docs = []
        lang_fail = []
    # Compose return tuple based on which checks are enabled
    result = [dup_ids]
    if check_ref:
        result.append(dup_docs)
    if check_language_flag:
        result.append(lang_fail)
    if check_size:
        result.append(corpus_size)
    if len(result) == 1:
        return result[0]
    return tuple(result)


def validate_dataframe_content(df: pl.DataFrame) -> list:
    """
    Validate the content of a corpus DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to validate.

    Returns
    -------
    list
        List of validation warnings/errors.
    """
    warnings = []

    # Check for empty DataFrame
    if df.height == 0:
        warnings.append("DataFrame is empty")
        return warnings

    # Check required columns
    required_columns = ['doc_id', 'token', 'pos_tag', 'ds_tag']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        warnings.append(f"Missing required columns: {missing_columns}")

    # Check for null values in critical columns
    for col in required_columns:
        if col in df.columns:
            null_count = df.select(pl.col(col).is_null().sum()).item()
            if null_count > 0:
                warnings.append(f"Column '{col}' has {null_count} null values")

    # Check for reasonable document count
    if 'doc_id' in df.columns:
        doc_count = df.select(pl.col('doc_id').n_unique()).item()
        if doc_count == 0:
            warnings.append("No documents found")
        elif doc_count < 5:
            warnings.append(
                f"Only {doc_count} documents found - "
                "may be insufficient for analysis"
            )

    return warnings


def normalize_text(text: str) -> str:
    """
    Normalize text for processing.

    Parameters
    ----------
    text : str
        Text to normalize.

    Returns
    -------
    str
        Normalized text.
    """
    # Remove accents and normalize unicode
    text = unidecode.unidecode(text)

    # Basic text cleaning
    text = text.strip()
    text = " ".join(text.split())  # Normalize whitespace

    return text


def is_valid_df(
        df: Union[pl.DataFrame, pd.DataFrame],
        required_cols: Optional[List[str]] = None
        ) -> bool:
    """
    Check if a DataFrame is valid for processing.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to check.
    required_cols : list, optional
        A list of required columns that must be present in the DataFrame.
        If None, no specific columns are checked.

    Returns
    -------
    bool
        True if the DataFrame is valid, False otherwise.
    """
    # Check if df is None or empty
    if df is None:
        return False
    if hasattr(df, "height"):
        if df.height == 0:
            return False
        cols = df.columns
    elif hasattr(df, "shape"):
        if df.shape[0] == 0:
            return False
        cols = df.columns
    else:
        return False
    if required_cols:
        return all(col in cols for col in required_cols)
    return True


# Corpus validation functions
def has_target_corpus(session: dict) -> bool:
    """Check if target corpus is loaded."""
    return safe_session_get(session, SessionKeys.HAS_TARGET, False)


def has_reference_corpus(session: dict) -> bool:
    """Check if reference corpus is loaded."""
    return safe_session_get(session, SessionKeys.HAS_REFERENCE, False)


def has_metadata(session: dict) -> bool:
    """Check if metadata has been processed."""
    return safe_session_get(session, SessionKeys.HAS_META, False)


def safe_get_categories(metadata_target: dict) -> list:
    """Safely extract categories from metadata."""
    try:
        return sorted(set(metadata_target.get('doccats', [{}])[0].get('cats', [])))
    except (KeyError, IndexError, TypeError):
        return []


def render_corpus_not_loaded_error(corpus_type: str = "target") -> None:
    """Display standardized error message when corpus is not loaded."""
    if corpus_type == "reference":
        message = (
            "It doesn't look like you've loaded the necessary reference corpus yet."
        )
    else:
        message = (
            f"It doesn't look like you've loaded the necessary {corpus_type} corpus yet."
        )

    st.warning(
        body=message,
        icon=":material/sentiment_stressed:"
    )


def render_metadata_not_processed_error() -> None:
    """Display standardized error message when metadata is not processed."""
    st.error(
        body=(
            "No metadata found for the target corpus. "
            "Please load or generate metadata first "
            "from **Manage Corpus Data**."
        ),
        icon=":material/sentiment_stressed:"
    )
    st.page_link(
        page="pages/1_load_corpus.py",
        label="Manage Corpus Data",
        icon=":material/database:",
    )


def check_corpus_external(
        tok_pl: pl.DataFrame,
        check_size=False,
        check_ref=False,
        target_docs=None
        ) -> Union[tuple, bool]:
    """
    Check the corpus for schema validation, duplicates, size,
    and reference documents.

    Parameters
    ----------
    tok_pl : pl.DataFrame or None
        A Polars DataFrame containing the corpus data, or None.
    check_size : bool, optional
        If True, calculate the total size of the corpus (default is False).
    check_ref : bool, optional
        If True, check for reference documents in the corpus
        (default is False).
    target_docs : list, optional
        A list of target document identifiers to check against
        (default is None).

    Returns
    -------
    tuple or bool
        Returns a tuple containing:
        - is_valid (bool): Whether the schema of the DataFrame is valid.
        - dup_docs (list, optional): Documents found in both the corpus and target_docs (if check_ref).
        - corpus_size (int, optional): Total size of the corpus in bytes (if check_size).
        If no options are enabled, returns just is_valid.
    """  # noqa: E501
    if tok_pl is None:
        # Return the correct number of outputs based on flags
        if check_ref and check_size:
            return False, [], 0
        elif check_ref:
            return False, []
        elif check_size:
            return False, 0
        else:
            return False

    is_valid = check_schema(tok_pl)
    result = [is_valid]

    if check_ref and target_docs is not None:
        doc_ids = tok_pl.get_column("doc_id").unique().to_list()
        dup_docs = list(set(target_docs).intersection(doc_ids))
        result.append(dup_docs)
    elif check_ref:
        result.append([])

    if check_size:
        corpus_size = tok_pl.estimated_size()
        result.append(corpus_size)

    if len(result) == 1:
        return result[0]
    return tuple(result)
