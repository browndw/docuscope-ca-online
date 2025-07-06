"""
This app allows users to generate and view collocations for the loaded target corpus.

Collocations are words that frequently occur together in a specific context.
Users can specify a node word, the span of words to consider,
the association measure to use, and optionally anchor the node word to a specific tag.
The results will be displayed in a table with the collocates, their frequencies, and the
association scores.
"""

import pandas as pd
import polars as pl
import streamlit as st

# Core application utilities
from webapp.utilities.core import app_core

from webapp.utilities.session import (
    get_or_init_user_session, load_metadata, safe_session_get
    )
from webapp.utilities.corpus import (
    get_corpus_data, clear_corpus_data
    )
from webapp.utilities.exports import (
    convert_to_excel
)
from webapp.utilities.ui import (
    collocation_info, render_dataframe,
    toggle_download, sidebar_action_button,
    sidebar_help_link, target_info,
    tag_filter_multiselect
)
from webapp.utilities.analysis import (
    has_target_corpus, render_corpus_not_loaded_error
)
from webapp.utilities.analysis import (
    generate_collocations
)
from webapp.utilities.state import (
    safe_clear_widget_state,
    SessionKeys, CorpusKeys,
    TargetKeys, WarningKeys
    )
from webapp.menu import (
    menu, require_login
    )


TITLE = "Collocates"
ICON = ":material/network_node:"

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )


def create_enhanced_dataframe_for_export(
        df: pl.DataFrame,
        metadata_target
) -> pd.DataFrame:
    """Create a DataFrame with context information for Excel export."""
    if df is None or df.height == 0:
        return None

    # Get the context information
    collocation_data = metadata_target.get(SessionKeys.COLLOCATIONS)
    if collocation_data:
        # If it's a list with one dict, use the dict
        if (
            isinstance(collocation_data, list) and
            len(collocation_data) == 1 and
            isinstance(collocation_data[0], dict)
        ):
            coll_info = collocation_info(collocation_data[0])
        elif isinstance(collocation_data, dict):
            coll_info = collocation_info(collocation_data)
        # If it's the old 'temp' structure, display the first item in the list
        elif (
            isinstance(collocation_data, dict) and
            'temp' in collocation_data and
            isinstance(collocation_data['temp'], list) and
            collocation_data['temp']
        ):
            coll_info = collocation_info(collocation_data['temp'][0])
        else:
            st.info("No collocation parameters available.")

    # Convert to pandas
    pandas_df = df.to_pandas()

    # Create a full-width context section
    context_data = {}

    # Get all column names (original + context columns)
    all_columns = list(pandas_df.columns) + ['Context_Details']

    # Initialize all columns
    for col in all_columns:
        if col in pandas_df.columns:
            # Original data + empty rows for context
            context_data[col] = pandas_df[col].tolist() + ['', '', '']
        elif col == 'Context_Details':
            # Empty for data rows + context information
            context_data[col] = [coll_info, '', ''] + [''] * len(pandas_df)

    enhanced_df = pd.DataFrame(context_data)

    return enhanced_df


def render_results_interface(user_session_id: str, session: dict) -> None:
    """Render the interface when collocations have been generated."""
    metadata_target = load_metadata(CorpusKeys.TARGET, user_session_id)

    # Get collocations data using the new system
    df = get_corpus_data(user_session_id, CorpusKeys.TARGET, TargetKeys.COLLOCATIONS)
    if df is None:
        st.error("Collocations data not found. Please regenerate the analysis.")
        return

    # Display corpus and collocation info
    col1, col2 = st.columns([1, 1])
    with col1:
        st.info(target_info(metadata_target))
    with col2:
        # Get collocation data from metadata
        collocation_data = metadata_target.get(SessionKeys.COLLOCATIONS)
        if collocation_data:
            # If it's a list with one dict, use the dict
            if (
                isinstance(collocation_data, list) and
                len(collocation_data) == 1 and
                isinstance(collocation_data[0], dict)
            ):
                st.info(collocation_info(collocation_data[0]))
            elif isinstance(collocation_data, dict):
                st.info(collocation_info(collocation_data))
            # If it's the old 'temp' structure, display the first item in the list
            elif (
                isinstance(collocation_data, dict) and
                'temp' in collocation_data and
                isinstance(collocation_data['temp'], list) and
                collocation_data['temp']
            ):
                st.info(collocation_info(collocation_data['temp'][0]))
            else:
                st.info("No collocation parameters available.")
        else:
            st.info("No collocation parameters available.")

    # Apply tag filtering and display table
    if df is not None and getattr(df, "height", 0) > 0:
        df = tag_filter_multiselect(df, user_session_id=user_session_id)
        render_dataframe(df)

        # Download option
        toggle_download(
            label="Excel",
            convert_func=convert_to_excel,
            convert_args=(
                (create_enhanced_dataframe_for_export(df, metadata_target),)
                if (df is not None and getattr(df, "height", 0) > 0)
                else (None,)
            ),
            file_name="collocations",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            location=st.sidebar
        )
    else:
        st.warning("No collocations data available.", icon=":material/info:")

    # Sidebar controls for generating new table
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        body=(
            "### Generate new table\n\n"
            "Use the button to reset the collocations table and start over."
        )
    )

    if st.sidebar.button(
        label="Create New Collocations Table",
        icon=":material/refresh:"
    ):
        # Clear existing data using the new system
        clear_corpus_data(user_session_id, CorpusKeys.TARGET, [TargetKeys.COLLOCATIONS])
        app_core.session_manager.update_session_state(
            user_session_id, SessionKeys.COLLOCATIONS, False
        )
        st.rerun()

    st.sidebar.markdown("---")


def render_setup_interface(user_session_id: str, session: dict) -> None:
    """Render the interface for setting up collocation analysis."""
    st.markdown(
        body=(
            ":material/manufacturing: Use the configuration below to "
            "**generate tables of collocations**.\n\n"
            ":material/priority: You have a number of options for statistics, "
            "and for specifying a node word and its context."
        )
    )

    # Load metadata if available
    metadata_target = None
    if has_target_corpus(session):
        metadata_target = load_metadata(CorpusKeys.TARGET, user_session_id)

    # Configuration in expanders
    with st.expander(
        label="Collocation Configuration",
        icon=":material/settings:",
        expanded=True
    ):
        node_word = render_node_word_config()
        to_left, to_right = render_span_config()
        stat_mode = render_association_measure_config()
        node_tag, count_by = render_anchor_tag_config(session, metadata_target)

    # Generation controls in sidebar
    render_generation_controls(
        user_session_id, session, node_word, node_tag,
        to_left, to_right, stat_mode, count_by
    )


def render_node_word_config() -> str:
    """Render node word configuration."""
    st.markdown("### Node word")
    st.markdown("Enter a node word without spaces.")
    return st.text_input(
        label="Node word:",
        placeholder="Enter a single word (e.g., 'data')",
        key="collocation_node_word",
        help="The central word around which to find collocations."
    )


def render_span_config() -> tuple[int, int]:
    """Render span configuration."""
    st.markdown(
        "### Span",
        help=(
            "You can choose the span of words to the left and right of the node word. "
            "This defines the context in which collocates are identified. "
            "A span of 4 means that 4 words to the left and 4 words to the right "
            "of the node word will be considered."
        )
    )

    col1, col2 = st.columns(2)
    with col1:
        to_left = st.slider(
            "Left span:",
            0, 9, 4,
            key="collocation_left_span",
            help="Number of words to the left of the node word to consider."
        )
    with col2:
        to_right = st.slider(
            "Right span:",
            0, 9, 4,
            key="collocation_right_span",
            help="Number of words to the right of the node word to consider."
        )

    return to_left, to_right


def render_association_measure_config() -> str:
    """Render association measure configuration."""
    st.markdown(
        "### Association measure",
        help=(
            "You can choose the association measure to use for collocations. "
            "NPMI is a normalized version of PMI that adjusts for the frequency "
            "of the node word. PMI 2 and PMI 3 are variations of PMI that consider "
            "different contextual spans. PMI is the standard pointwise mutual "
            "information measure. PMI was introduced by Church and Hanks (1990) "
            "and is widely used in computational linguistics for measuring the "
            "association between words."
        )
    )

    stat_mode = st.radio(
        "Select a statistic:",
        ["NPMI", "PMI 2", "PMI 3", "PMI"],
        horizontal=True,
        key="collocation_stat_mode",
        help="Different statistical measures for word association strength."
    )

    # Convert display names to internal names
    stat_mapping = {
        "PMI": "pmi",
        "PMI 2": "pmi2",
        "PMI 3": "pmi3",
        "NPMI": "npmi"
    }
    return stat_mapping[stat_mode]


def render_anchor_tag_config(session: dict, metadata_target: dict) -> tuple[str, str]:
    """Render anchor tag configuration."""
    st.markdown(
        "### Anchor tag",
        help=(
            "You can choose to **anchor** a token to a specific tag. "
            "For example, if you wanted to disambiguate *can* as a noun "
            "(e.g., *can of soda*) from *can* as a modal verb, you could "
            "**anchor** the node word to a part-of-speech tag (like **Noun**, "
            "**Verb** or more specifically **VM**)."
        )
    )

    tag_radio = st.radio(
        "Select tagset for node word:",
        ("No Tag", "Parts-of-Speech", "DocuScope"),
        horizontal=True,
        key="collocation_tag_radio",
        help="Choose whether to anchor the node word to a specific tag."
    )

    node_tag = None
    count_by = 'pos'

    if tag_radio == 'Parts-of-Speech':
        node_tag, count_by = render_pos_tag_selection(session, metadata_target)
    elif tag_radio == 'DocuScope':
        node_tag, count_by = render_docuscope_tag_selection(session, metadata_target)

    return node_tag, count_by


def render_pos_tag_selection(session: dict, metadata_target: dict) -> tuple[str, str]:
    """Render Parts-of-Speech tag selection."""
    tag_type = st.radio(
        "Select from general or specific tags:",
        ("General", "Specific"),
        horizontal=True,
        key="collocation_pos_tag_type",
        help=(
            "General tags are simplified categories, "
            "specific tags show detailed POS labels."
        )
    )

    if tag_type == 'General':
        node_tag = st.selectbox(
            "Select tag:",
            ("Noun Common", "Verb Lex", "Adjective", "Adverb"),
            key="collocation_pos_general_tag",
            help="Choose a general part-of-speech category."
        )
        # Map display names to internal tags
        tag_mapping = {
            "Noun Common": "NN",
            "Verb Lex": "VV",
            "Adjective": "JJ",
            "Adverb": "R"
        }
        node_tag = tag_mapping[node_tag]
    else:
        if not has_target_corpus(session):
            node_tag = st.selectbox(
                'Choose a tag:',
                ['No tags currently loaded'],
                key="collocation_pos_specific_tag_empty",
                help="Load a target corpus first to see available tags."
            )
        else:
            node_tag = st.selectbox(
                'Choose a tag:',
                metadata_target.get('tags_pos')[0]['tags'],
                key="collocation_pos_specific_tag",
                help="Choose a specific part-of-speech tag."
            )

    return node_tag, 'pos'


def render_docuscope_tag_selection(
    session: dict, metadata_target: dict
) -> tuple[str, str]:
    """Render DocuScope tag selection."""
    if not has_target_corpus(session):
        node_tag = st.selectbox(
            'Choose a tag:',
            ['No tags currently loaded'],
            key="collocation_ds_tag_empty",
            help="Load a target corpus first to see available tags."
        )
    else:
        node_tag = st.selectbox(
            'Choose a tag:',
            metadata_target.get('tags_ds')[0]['tags'],
            key="collocation_ds_tag",
            help="Choose a DocuScope rhetorical tag."
        )

    return node_tag, 'ds'


def render_generation_controls(
    user_session_id: str,
    session: dict,
    node_word: str,
    node_tag: str,
    to_left: int,
    to_right: int,
    stat_mode: str,
    count_by: str
) -> None:
    """Render the collocation generation controls."""
    st.sidebar.markdown(
        body=(
            "### Generate collocations\n\n"
            "Use the button to process collocations."
        ),
        help=(
            "Collocations are generated based on the node word and configuration above. "
            "The table will include collocates, their frequencies, "
            "and association scores.\n\n"
            "Click on the **Help** button for more information on how to use this app."
        )
    )

    # Create a custom action that handles validation and shows appropriate errors
    def collocations_action():
        if not has_target_corpus(session):
            render_corpus_not_loaded_error()
            return

        # Check if node word is provided
        if not node_word or node_word.strip() == "":
            st.warning(
                body=(
                    "Please enter a **node word**."
                ),
                icon=":material/edit:"
            )
            return

        # If all validation passes, generate the collocations
        generate_collocations(
            user_session_id, node_word, node_tag, to_left, to_right, stat_mode, count_by
        )

    sidebar_action_button(
        button_label="Collocations Table",
        button_icon=":material/manufacturing:",
        preconditions=[True],  # Always allow button click, handle validation in action
        action=collocations_action,
        spinner_message="Processing collocates..."
    )

    # Display any warnings
    if st.session_state[user_session_id].get(WarningKeys.COLLOCATIONS):
        msg, icon = st.session_state[user_session_id][WarningKeys.COLLOCATIONS]
        st.warning(msg, icon=icon)
        # Clear the warning after displaying it
        safe_clear_widget_state(f"{user_session_id}_{WarningKeys.COLLOCATIONS}")

    st.sidebar.markdown("---")


def main():
    """
    Main function to run the Streamlit app for collocations analysis.
    This function sets up the page configuration, checks user login status,
    initializes the user session, and renders the UI components for
    generating and viewing collocations from the loaded corpus.
    """
    # Set login requirements for navigation
    require_login()
    menu()
    st.markdown(
        body=f"## {TITLE}",
        help=(
            "This app allows you to generate and view collocations for the "
            "loaded target corpus. Collocations are words that frequently occur "
            "together in a specific context. You can specify a node word, the span "
            "of words to consider, the association measure to use, and optionally "
            "anchor the node word to a specific tag. The results will be displayed "
            "in a table with the collocates, their frequencies, and the association scores."
        )
    )

    # Get or initialize user session
    user_session_id, session = get_or_init_user_session()

    sidebar_help_link("collocations.html")

    # Check if collocations table has been generated
    if safe_session_get(session, SessionKeys.COLLOCATIONS, False):
        render_results_interface(user_session_id, session)
    else:
        render_setup_interface(user_session_id, session)


if __name__ == "__main__":
    main()
