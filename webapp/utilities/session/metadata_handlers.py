"""
Metadata handling utilities for corpus analysis.

This module provides functions for initializing, loading, and updating
metadata for target and reference corpora.
"""

import streamlit as st
import polars as pl
from webapp.utilities.session.session_management import (
    get_corpus_categories, update_session
    )
from webapp.utilities.state import (
    SessionKeys, CorpusKeys,
    MetadataKeys, ReferenceKeys
    )


# Constants for metadata validation
MIN_CATEGORIES = 2
MAX_CATEGORIES = 20


def init_metadata_target(session_id: str) -> None:
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
    df = st.session_state[session_id]["target"]["ds_tokens"]
    tags_to_check = df.get_column("ds_tag").to_list()
    tags = [
        'Actors', 'Organization', 'Planning', 'Sentiment', 'Signposting', 'Stance'
    ]
    model = 'Common Dictionary' if any(tag in item for item in tags_to_check for tag in tags) else 'Large Dictionary'  # noqa: E501
    ds_tags = df.get_column("ds_tag").unique().to_list()
    tags_pos = df.get_column("pos_tag").unique().to_list()
    if "Untagged" in ds_tags:
        ds_tags.remove("Untagged")
    if "Y" in tags_pos:
        tags_pos.remove("Y")

    temp_metadata_target = {
        MetadataKeys.TOKENS_POS: df.group_by(
            ["doc_id", "pos_id", "pos_tag"]
        ).agg(pl.col("token").str.concat("")).filter(pl.col("pos_tag") != "Y").height,
        MetadataKeys.TOKENS_DS: df.group_by(
            ["doc_id", "ds_id", "ds_tag"]
        ).agg(pl.col("token").str.concat("")).filter(
            ~(pl.col("token").str.contains("^[[[:punct:]] ]+$") & pl.col("ds_tag").str.contains("Untagged"))  # noqa: E501
        ).height,
        MetadataKeys.NDOCS: len(df.get_column("doc_id").unique().to_list()),
        MetadataKeys.MODEL: model,
        MetadataKeys.DOCIDS: {'ids': sorted(df.get_column("doc_id").unique().to_list())},
        MetadataKeys.TAGS_DS: {'tags': sorted(ds_tags)},
        MetadataKeys.TAGS_POS: {'tags': sorted(tags_pos)},
        MetadataKeys.DOCCATS: {'cats': ''},
        MetadataKeys.COLLOCATIONS: {'temp': ''},
        MetadataKeys.KEYNESS_PARTS: {'temp': ''},
        MetadataKeys.VARIANCE: {'temp': ''},
    }
    df = pl.from_dict(temp_metadata_target, strict=False)
    st.session_state[session_id]["metadata_target"] = df


def init_metadata_reference(session_id: str) -> None:
    """
    Initialize the metadata for the reference corpus in the session state.

    Parameters
    ----------
    session_id : str
        The session ID for which the metadata is to be initialized.

    Returns
    -------
    None
    """
    df = st.session_state[session_id][CorpusKeys.REFERENCE][ReferenceKeys.DS_TOKENS]
    tags_to_check = df.get_column("ds_tag").to_list()
    tags = [
        'Actors',
        'Organization',
        'Planning',
        'Sentiment',
        'Signposting',
        'Stance'
    ]
    model = 'Common Dictionary' if any(tag in item for item in tags_to_check for tag in tags) else 'Large Dictionary'  # noqa: E501
    ds_tags = df.get_column("ds_tag").unique().to_list()
    tags_pos = df.get_column("pos_tag").unique().to_list()
    if "Untagged" in ds_tags:
        ds_tags.remove("Untagged")
    if "Y" in tags_pos:
        tags_pos.remove("Y")

    temp_metadata_reference = {
        MetadataKeys.TOKENS_POS: df.group_by(
            ["doc_id", "pos_id", "pos_tag"]
        ).agg(pl.col("token").str.concat("")).filter(pl.col("pos_tag") != "Y").height,
        MetadataKeys.TOKENS_DS: df.group_by(
            ["doc_id", "ds_id", "ds_tag"]
        ).agg(pl.col("token").str.concat("")).filter(
            ~(pl.col("token").str.contains("^[[[:punct:]] ]+$") & pl.col("ds_tag").str.contains("Untagged"))  # noqa: E501
        ).height,
        MetadataKeys.NDOCS: len(df.get_column("doc_id").unique().to_list()),
        MetadataKeys.MODEL: model,
        MetadataKeys.DOCIDS: {'ids': sorted(df.get_column("doc_id").unique().to_list())},
        MetadataKeys.TAGS_DS: {'tags': sorted(ds_tags)},
        MetadataKeys.TAGS_POS: {'tags': sorted(tags_pos)},
        MetadataKeys.DOCCATS: False,
        MetadataKeys.COLLOCATIONS: {'temp': ''},
        MetadataKeys.KEYNESS_PARTS: {'temp': ''},
        MetadataKeys.VARIANCE: {'temp': ''},
    }
    df = pl.from_dict(temp_metadata_reference, strict=False)
    st.session_state[session_id][SessionKeys.METADATA_REFERENCE] = df


def load_metadata(corpus_type: str, session_id: str) -> dict:
    """
    Load metadata for the specified corpus type from the session state.

    Parameters
    ----------
    corpus_type : str
        The type of corpus for which metadata is to be loaded.
        Should be either 'target' or 'reference'.
    session_id : str
        The session ID for which the metadata is to be loaded.

    Returns
    -------
    dict
        A dictionary containing the metadata for the specified corpus type.
    """
    if corpus_type == "target":
        table_name = SessionKeys.METADATA_TARGET
    elif corpus_type == "reference":
        table_name = SessionKeys.METADATA_REFERENCE
    else:
        raise ValueError("corpus_type must be 'target' or 'reference'")

    metadata = st.session_state[session_id][table_name]
    metadata = metadata.to_dict(as_series=False)
    return metadata


def update_metadata(
        corpus_type: str,
        key: str,
        value: any,
        session_id: str
        ) -> None:
    """
    Update metadata for the specified corpus type in the session state.

    Parameters
    ----------
    corpus_type : str
        The type of corpus for which metadata is to be updated.
        Should be either 'target' or 'reference'.
    key : str
        The key in the metadata dictionary to update.
    value : any
        The value to assign to the specified key in the metadata dictionary.
    session_id : str
        The session ID for which the metadata is to be updated.

    Returns
    -------
    None
        The function updates the metadata in the session state.
    """
    if corpus_type == "target":
        table_name = SessionKeys.METADATA_TARGET
    elif corpus_type == "reference":
        table_name = SessionKeys.METADATA_REFERENCE
    else:
        raise ValueError("corpus_type must be 'target' or 'reference'")

    metadata = st.session_state[session_id][table_name]
    metadata = metadata.to_dict(as_series=False)

    if key == "doccats":
        metadata['doccats'] = [{'cats': value}]
    elif key == "collocations":
        metadata['collocations'] = {'temp': [value]}
    elif key == "keyness_parts":
        metadata['keyness_parts'] = {'temp': [value]}
    elif key == "variance":
        metadata['variance'] = {'temp': [value]}
    else:
        metadata[key] = value

    df = pl.from_dict(metadata, strict=False)
    st.session_state[session_id][table_name] = df


def handle_target_metadata_processing(metadata_target: dict, user_session_id: str) -> None:
    """Handle target corpus metadata processing with validation."""
    st.sidebar.markdown('### Target corpus metadata:')
    load_cats = st.sidebar.radio(
        "Do you have categories in your file names to process?",
        ("No", "Yes"),
        horizontal=True,
        help=(
            "Metadata can be encoded into your file names, "
            "which can be used for further analysis. "
            "The tool can detect information that comes before "
            "the first underscore in the file name, and will "
            "use that information to assign categories to your "
            "documents. For example, if your file names are "
            "`cat1_doc1.txt`, `cat2_doc2.txt`, etc., "
            "the tool will assign `cat1` and `cat2` as categories. "
        )
    )

    if load_cats == 'Yes':
        if st.sidebar.button(
            label="Process Document Metadata",
            icon=":material/manufacturing:"
        ):
            with st.spinner('Processing metadata...'):
                try:
                    doc_ids = metadata_target.get(
                        MetadataKeys.DOCIDS, [{}]
                    )[0].get('ids', [])
                    if not doc_ids:
                        st.sidebar.error("No document IDs found to process.")
                        return

                    # Use cached function for efficiency
                    doc_cats, unique_count = get_corpus_categories(doc_ids, user_session_id)

                    if MIN_CATEGORIES <= unique_count <= MAX_CATEGORIES:
                        update_metadata(
                            CorpusKeys.TARGET,
                            MetadataKeys.DOCCATS,
                            doc_cats,
                            user_session_id)
                        update_session(
                            SessionKeys.HAS_META,
                            True,
                            user_session_id)
                        st.sidebar.success(
                            f"Successfully processed {unique_count} document categories!"
                        )
                        st.rerun()
                    else:
                        st.sidebar.error(
                            f"Found {unique_count} categories. "
                            f"Please ensure you have between {MIN_CATEGORIES} "
                            f"and {MAX_CATEGORIES} categories."
                        )
                except Exception as e:
                    st.sidebar.error(f"Error processing metadata: {str(e)}")
                    st.sidebar.exception(e)
