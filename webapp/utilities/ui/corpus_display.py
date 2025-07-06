"""
Corpus display utilities for the Streamlit interface.

This module provides functions for displaying corpus information and
generating formatted corpus information strings.
"""

from collections import Counter
import streamlit as st
from webapp.utilities.session import validate_session_state, safe_session_get
from webapp.utilities.session.metadata_handlers import load_metadata
from webapp.utilities.ui.shared_utils import add_category_description
from webapp.utilities.state import SessionKeys, MetadataKeys, CorpusKeys
from webapp.utilities.common import get_doc_cats


# Utility function to safely access metadata values in both formats
def safe_metadata_get(metadata: dict, key: str, default=None, nested_key: str = None):
    """
    Safely get a value from metadata dict, handling both list and scalar formats.

    Metadata can be in different formats:
    - DataFrame converted: {'docids': [{'ids': [...]}], 'doccats': [{'cats': [...]}]}
    - Direct dict: {'docids': {'ids': [...]}, 'doccats': {'cats': [...]}}

    Parameters
    ----------
    metadata : dict
        The metadata dictionary
    key : str
        The primary key to access
    default : any
        Default value if key not found
    nested_key : str, optional
        Optional nested key (e.g., 'ids' for docids, 'cats' for doccats)

    Returns
    -------
    any
        The value from the metadata
    """
    value = metadata.get(key, default)

    if value is None:
        return default

    # If it's a list (from DataFrame conversion) and we want nested access
    if isinstance(value, list) and len(value) > 0 and nested_key:
        if isinstance(value[0], dict):
            return value[0].get(nested_key, default)
        else:
            return value[0] if not nested_key else default

    # If it's a dict and we want nested access
    if isinstance(value, dict) and nested_key:
        return value.get(nested_key, default)

    # If it's a list but no nested key needed, return first element
    if isinstance(value, list) and len(value) > 0 and not nested_key:
        return value[0]

    # Return as-is
    return value


# Corpus information display functions
def target_info(target_metadata: dict) -> str:
    """
    Generate a string with information about the target corpus.
    This function extracts the number of part-of-speech tokens,
    DocuScope tokens, and documents from the target metadata.

    Parameters
    ----------
    target_metadata : dict
        Metadata dictionary containing information about the target corpus.
        Expected keys: 'tokens_pos', 'tokens_ds', 'ndocs'.

    Returns
    -------
    str
        A formatted string containing the target corpus information.
    """
    tokens_pos = safe_metadata_get(target_metadata, 'tokens_pos')
    tokens_ds = safe_metadata_get(target_metadata, 'tokens_ds')
    ndocs = safe_metadata_get(target_metadata, 'ndocs')
    target_info = f"""##### Target corpus information:

    Number of part-of-speech tokens in corpus: {tokens_pos:,}
    \n    Number of DocuScope tokens in corpus: {tokens_ds:,}
    \n    Number of documents in corpus: {ndocs:,}
    """
    return target_info


def reference_info(reference_metadata: dict) -> str:
    """
    Generate a string with information about the reference corpus.
    This function extracts the number of part-of-speech tokens,
    DocuScope tokens, and documents from the reference metadata.

    Parameters
    ----------
    reference_metadata : dict
        Metadata dictionary containing information about the reference corpus.
        Expected keys: 'tokens_pos', 'tokens_ds', 'ndocs'.

    Returns
    -------
    str
        A formatted string containing the reference corpus information.
    """
    tokens_pos = safe_metadata_get(reference_metadata, 'tokens_pos')
    tokens_ds = safe_metadata_get(reference_metadata, 'tokens_ds')
    ndocs = safe_metadata_get(reference_metadata, 'ndocs')
    reference_info = f"""##### Reference corpus information:

    Number of part-of-speech tokens in corpus: {tokens_pos:,}
    \n    Number of DocuScope tokens in corpus: {tokens_ds:,}
    \n    Number of documents in corpus: {ndocs:,}
    """
    return reference_info


def collocation_info(collocation_data):
    """Generate formatted collocation information string."""
    # Handle both dict and list/tuple formats
    if isinstance(collocation_data, dict):
        node_word = collocation_data.get('node_word', '')
        node_tag = collocation_data.get('node_tag', None)
        to_left = collocation_data.get('to_left', '')
        to_right = collocation_data.get('to_right', '')
        stat_mode = str(collocation_data.get('stat_mode', '')).upper()
        count_by = collocation_data.get('count_by', '')
    elif isinstance(collocation_data, (list, tuple)):
        # Old format: [node_word, stat_mode, to_left, to_right]
        node_word = collocation_data[0] if len(collocation_data) > 0 else ''
        stat_mode = str(collocation_data[1]).upper() if len(collocation_data) > 1 else ''
        to_left = collocation_data[2] if len(collocation_data) > 2 else ''
        to_right = collocation_data[3] if len(collocation_data) > 3 else ''
        node_tag = None
        count_by = ''
    else:
        return str(collocation_data)

    span = f"{to_left}L - {to_right}R"
    tag_str = f"\n    Node tag: {node_tag}" if node_tag else ""
    count_by_str = f"\n    Count by: {count_by}" if count_by else ""
    coll_info = f"""##### Collocate information:\n\n    Association measure: {stat_mode}\n    Span: {span}\n    Node word: {node_word}{tag_str}{count_by_str}\n    """  # noqa: E501
    return coll_info


def correlation_info(cc_dict):
    """
    Formats correlation info for display in a code block for easy copy-paste.
    r is always shown to 3 decimal places, p to 5 decimal places.
    """
    def fmt_r(val):
        try:
            return f"{float(val):.3f}"
        except Exception:
            return str(val)

    def fmt_p(val):
        try:
            return f"{float(val):.5f}"
        except Exception:
            return str(val)

    lines = []
    if 'all' in cc_dict and cc_dict['all']:
        lines.append(
            f"All points: r({cc_dict['all']['df']}) = {fmt_r(cc_dict['all']['r'])}, p = {fmt_p(cc_dict['all']['p'])}"  # noqa: E501
        )
    if 'highlight' in cc_dict and cc_dict['highlight']:
        lines.append(
            f"Highlighted group: r({cc_dict['highlight']['df']}) = {fmt_r(cc_dict['highlight']['r'])}, p = {fmt_p(cc_dict['highlight']['p'])}"  # noqa: E501
        )
        if 'non_highlight' in cc_dict and cc_dict['non_highlight']:
            lines.append(
                f"Non-highlighted group: r({cc_dict['non_highlight']['df']}) = {fmt_r(cc_dict['non_highlight']['r'])}, p = {fmt_p(cc_dict['non_highlight']['p'])}"  # noqa: E501
            )
    lines_str = "\n    ".join(lines)
    corr_info = f"""##### Pearson's correlation coefficient:

    {lines_str}
    """
    return corr_info


def variance_info(pca_x: str, pca_y: str, ve_1: str, ve_2: str) -> str:
    """Generate variance explained information string."""
    variance_info = f"""##### Variance explained:

    {pca_x}: {ve_1}\n    {pca_y}: {ve_2}
    """
    return variance_info


def contribution_info(pca_x: str, pca_y: str, contrib_x: str, contrib_y: str) -> str:
    """Generate contribution information string."""
    contrib_info = f"""##### Variables with contribution > mean:

    {pca_x}: {contrib_x}\n    {pca_y}: {contrib_y}
    """
    return contrib_info


def group_info(grp_a: list[str], grp_b: list[str]) -> str:
    """Generate group information string."""
    grp_a = [s.strip('_') for s in grp_a]
    grp_a = ", ".join(str(x) for x in grp_a)
    grp_b = [s.strip('_') for s in grp_b]
    grp_b = ", ".join(str(x) for x in grp_b)
    group_info = f"""##### Grouping variables:

    Group A: {grp_a}\n    Group B: {grp_b}
    """
    return group_info


def target_parts(keyness_parts: list[str]) -> str:
    """Generate target corpus parts information string."""
    t_cats = keyness_parts[0]
    tokens_pos = keyness_parts[2]
    tokens_ds = keyness_parts[4]
    ndocs = keyness_parts[6]
    target_info = f"""##### Target corpus information:

    Document categories: {t_cats}
    \n    Part-of-speech tokens: {tokens_pos}
    \n    DocuScope tokens: {tokens_ds}
    \n    Documents: {ndocs}
    """
    return target_info


def reference_parts(keyness_parts: list[str]) -> str:
    """Generate reference corpus parts information string."""
    r_cats = keyness_parts[1]
    tokens_pos = keyness_parts[3]
    tokens_ds = keyness_parts[5]
    ndocs = keyness_parts[7]
    reference_info = f"""##### Reference corpus information:

    Document categories: {r_cats}
    \n    Part-of-speech tokens: {tokens_pos}
    \n    DocuScope tokens: {tokens_ds}
    \n    Documents: {ndocs}
    """
    return reference_info


# Complex corpus loading and display functions
def load_and_display_target_corpus(session: dict, user_session_id: str) -> None:
    """Load and display target corpus information with error handling."""

    if not validate_session_state(user_session_id):
        st.error("Invalid session state. Please reset the corpus.")
        return

    try:
        # Load target corpus metadata using the unified metadata handler
        metadata_target = load_metadata(
            CorpusKeys.TARGET,
            user_session_id
        )

        # Check if reference is loaded
        has_reference = safe_session_get(session, SessionKeys.HAS_REFERENCE, False) is True
        if has_reference:
            metadata_reference = load_metadata(
                CorpusKeys.REFERENCE,
                user_session_id
            )

        # Create tabs for Target and Reference
        tab_labels = [":material/docs: Target corpus"]
        if has_reference:
            tab_labels.append(":material/text_compare: Reference corpus")
        tabs = st.tabs(tab_labels)

        # --- Target Tab ---
        with tabs[0]:
            st.info(target_info(metadata_target))
            with st.expander(
                label="Documents in target corpus:",
                icon=":material/home_storage:"
            ):
                doc_ids = safe_metadata_get(metadata_target, MetadataKeys.DOCIDS, [], 'ids')
                st.write(doc_ids)

            if bool(safe_session_get(session, SessionKeys.HAS_META, False)):
                st.markdown('##### Target corpus metadata:')
                # Handle both DataFrame and dict formats for DOCCATS
                doccats_raw = metadata_target.get(MetadataKeys.DOCCATS, [])

                if isinstance(doccats_raw, list) and len(doccats_raw) > 0:
                    # DataFrame format: [{'cats': [...]}]
                    if isinstance(doccats_raw[0], dict):
                        doc_cats = doccats_raw[0].get('cats', [])
                    else:
                        doc_cats = []
                elif isinstance(doccats_raw, dict):
                    # Dict format: {'cats': [...]}
                    doc_cats = doccats_raw.get('cats', [])
                else:
                    doc_cats = []

                if doc_cats:
                    cat_counts = Counter(doc_cats)
                    cat_df = add_category_description(
                        cat_counts,
                        session,
                        corpus_type="target")
                    st.data_editor(cat_df, hide_index=True, disabled=True)

        # --- Reference Tab (if loaded) ---
        if has_reference:
            display_reference_corpus_tab(tabs[1], metadata_reference, session)

    except Exception as e:
        st.error(f"Error loading corpus data: {str(e)}", icon=":material/error:")
        st.info("Try resetting the corpus if this error persists.")


def display_reference_corpus_tab(tab, metadata_reference: dict, session: dict) -> None:
    """Display reference corpus information in a tab with error handling."""

    with tab:
        try:
            st.info(reference_info(metadata_reference))
            with st.expander(
                label="Documents in reference corpus:",
                icon=":material/home_storage:"
            ):
                ref_doc_ids = safe_metadata_get(
                    metadata_reference, MetadataKeys.DOCIDS, [], 'ids')
                st.write(ref_doc_ids)

            # Try to process and display reference metadata if target has metadata
            if safe_session_get(session, SessionKeys.HAS_META, False):
                try:
                    st.markdown('##### Reference corpus metadata:')
                    # Extract categories from doc ids using get_doc_cats
                    doc_cats_ref = get_doc_cats(ref_doc_ids)
                    if doc_cats_ref:
                        cat_counts_ref = Counter(doc_cats_ref)
                        cat_df_ref = add_category_description(
                            cat_counts_ref,
                            session,
                            corpus_type="reference")
                        st.data_editor(cat_df_ref, hide_index=True, disabled=True)
                    else:
                        st.warning(
                            "No categories found in reference corpus file names.",
                            icon=":material/info:"
                        )
                except Exception as e:
                    st.warning(
                        f"Could not process metadata for the reference corpus: {str(e)}. "
                        "This may be due to missing or malformed category information.",
                        icon=":material/info:"
                    )
        except Exception as e:
            st.error(
                f"Error displaying reference corpus: {str(e)}",
                icon=":material/error:"
            )


def render_corpus_info_expanders() -> None:
    """Render information expanders for different corpus types."""

    st.markdown("##### :material/lightbulb: Learn more...")
    col_1, col_2, col_3, col_4 = st.columns(4)

    with col_1:
        with st.expander("About internal corpora", icon=":material/database:"):
            corpus_links = [
                ("MICUSP", "https://browndw.github.io/docuscope-docs/datasets/micusp.html"),
                ("BAWE", "https://browndw.github.io/docuscope-docs/datasets/bawe.html"),
                ("ELSEVIER",
                 "https://browndw.github.io/docuscope-docs/datasets/elsevier.html"),
                ("HAP-E", "https://browndw.github.io/docuscope-docs/datasets/hape.html")
            ]
            for label, url in corpus_links:
                st.link_button(
                    label=label,
                    url=url,
                    icon=":material/quick_reference:"
                )

    with col_2:
        with st.expander("About external corpora", icon=":material/upload:"):
            st.link_button(
                label="Preparing an external corpus",
                url="https://browndw.github.io/docuscope-docs/vignettes/"
                    "external-corpus.html",
                icon=":material/quick_reference:"
            )

    with col_3:
        with st.expander("About new corpora", icon=":material/library_books:"):
            st.link_button(
                label="Preparing a new corpus",
                url="https://browndw.github.io/docuscope-docs/vignettes/new-corpus.html",
                icon=":material/quick_reference:"
            )

    with col_4:
        with st.expander("About the models", icon=":material/modeling:"):
            st.link_button(
                label="Compare models",
                url="https://browndw.github.io/docuscope-docs/tagsets/"
                    "model-comparison.html",
                icon=":material/quick_reference:"
            )
