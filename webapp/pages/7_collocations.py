# Copyright (C) 2025 David West Brown

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st

from webapp.utilities.session import (  # noqa: E402
    get_or_init_user_session, load_metadata,
    update_session
    )
# Temporary imports for unmigrated UI functions
from webapp.utilities.ui import (  # noqa: E402
    collocation_info, render_dataframe,
    render_excel_download_option, sidebar_action_button,
    sidebar_help_link, target_info,
    tag_filter_multiselect
)
from webapp.utilities.analysis import (  # noqa: E402
    has_target_corpus, render_corpus_not_loaded_error
)
from webapp.utilities.analysis import (  # noqa: E402
    generate_collocations
)
from webapp.menu import (   # noqa: E402
    menu, require_login
    )
from webapp.utilities.state import (  # noqa: E402
    SessionKeys, CorpusKeys,
    TargetKeys, WarningKeys
    )


TITLE = "Collocates"
ICON = ":material/network_node:"

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )


def render_results_interface(user_session_id: str, session: dict) -> None:
    """Render the interface when collocations have been generated."""
    metadata_target = load_metadata(CorpusKeys.TARGET, user_session_id)

    # Get collocations data with fallback
    try:
        df = st.session_state[user_session_id][CorpusKeys.TARGET][TargetKeys.COLLOCATIONS]
    except (KeyError, AttributeError):
        # Fallback to direct key access
        target_corpus = st.session_state[user_session_id][CorpusKeys.TARGET]
        df = target_corpus.get(TargetKeys.COLLOCATIONS)
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
        if (
            collocation_data
            and isinstance(collocation_data, dict)
            and 'temp' in collocation_data
        ):
            temp_data = collocation_data['temp']
            if temp_data and isinstance(temp_data, list) and len(temp_data) > 0:
                st.info(collocation_info(temp_data[0]))
            else:
                st.info("No collocation parameters set yet.")
        else:
            st.info("No collocation parameters available.")

    # Apply tag filtering and display table
    if df is not None and getattr(df, "height", 0) > 0:
        df = tag_filter_multiselect(df)
        render_dataframe(df)

        # Download option
        render_excel_download_option(df, "collocations")
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
        icon=":material/refresh:",
        help="Reset the analysis and configure new collocation parameters."
    ):
        # Clear existing data
        target_dict = st.session_state[user_session_id][CorpusKeys.TARGET]
        try:
            if TargetKeys.COLLOCATIONS in target_dict:
                target_dict[TargetKeys.COLLOCATIONS] = {}
        except AttributeError:
            # Fallback for attribute error
            if TargetKeys.COLLOCATIONS in target_dict:
                target_dict[TargetKeys.COLLOCATIONS] = {}
        update_session(SessionKeys.COLLOCATIONS, False, user_session_id)
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
    with st.expander("Collocation Configuration", expanded=True):
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
        "Node word:",
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
            help="Number of words to the left of the node word to consider."
        )
    with col2:
        to_right = st.slider(
            "Right span:",
            0, 9, 4,
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
        help=(
            "General tags are simplified categories, "
            "specific tags show detailed POS labels."
        )
    )

    if tag_type == 'General':
        node_tag = st.selectbox(
            "Select tag:",
            ("Noun Common", "Verb Lex", "Adjective", "Adverb"),
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
                help="Load a target corpus first to see available tags."
            )
        else:
            node_tag = st.selectbox(
                'Choose a tag:',
                metadata_target.get('tags_pos')[0]['tags'],
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
            help="Load a target corpus first to see available tags."
        )
    else:
        node_tag = st.selectbox(
            'Choose a tag:',
            metadata_target.get('tags_ds')[0]['tags'],
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
            st.error(
                body=(
                    "Please enter a **node word** in the configuration above. "
                    "The node word is the central word around which to find collocations."
                ),
                icon=":material/edit:"
            )
            return

        # If all validation passes, generate the collocations
        generate_collocations(
            user_session_id, node_word, node_tag, to_left, to_right, stat_mode, count_by
        )

    sidebar_action_button(
        button_label="Collocations",
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
        del st.session_state[user_session_id][WarningKeys.COLLOCATIONS]

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
    if session.get(SessionKeys.COLLOCATIONS, [False])[0]:
        render_results_interface(user_session_id, session)
    else:
        render_setup_interface(user_session_id, session)


if __name__ == "__main__":
    main()
