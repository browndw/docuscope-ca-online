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

import pathlib
import streamlit as st

from webapp.utilities.session import (
    get_or_init_user_session, load_metadata,
    update_session, validate_session_state
    )
from webapp.utilities.ui import (
    sidebar_action_button, sidebar_help_link,
    target_info, render_dataframe,
    toggle_download, multi_tag_filter_multiselect
)
from webapp.utilities.analysis import (
    has_target_corpus, render_corpus_not_loaded_error,
    generate_clusters, generate_ngrams
)
from webapp.menu import (
    menu, require_login
    )
from webapp.config.session_keys import (
    CorpusKeys, SessionKeys,
    WarningKeys
    )

TITLE = "N-gram and Cluster Frequency"
ICON = ":material/table_view:"

# Configuration constants
DEFAULT_WIDGET_STATE_PATH = pathlib.Path(__file__).stem
TAGSET_CONFIG = {
    "Parts-of-Speech": "pos",
    "DocuScope": "ds"
}
SEARCH_MODE_CONFIG = {
    "Fixed": "fixed",
    "Starts with": "starts_with",
    "Ends with": "ends_with",
    "Contains": "contains"
}
SPAN_OPTIONS = (2, 3, 4)

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )


def render_ngrams_display_interface(user_session_id: str, session: dict) -> None:
    """Render the interface for displaying existing n-grams/clusters data."""
    try:
        # Validate session state
        if not validate_session_state(user_session_id):
            st.error("Invalid session state. Please reload the page or reset your data.")
            return

        # Load metadata and data
        metadata_target = load_metadata(CorpusKeys.TARGET, user_session_id)
        if not metadata_target:
            st.error("Could not load target corpus metadata.")
            return

        # Load the session state for n-grams
        df = st.session_state[user_session_id][CorpusKeys.TARGET][SessionKeys.NGRAMS]

        # Display the target information first
        st.info(target_info(metadata_target))

        # Check if DataFrame has enumerated tag columns (Tag_1, Tag_2, etc.)
        tag_columns = []
        if df is not None and hasattr(df, "columns"):
            tag_columns = [col for col in df.columns if col.startswith("Tag_")]

        # Apply filtering for enumerated tag columns if they exist
        if tag_columns:
            df, _ = multi_tag_filter_multiselect(df, tag_columns)

        # Display the data table or warning
        if df is not None and hasattr(df, "height") and df.height > 0:
            render_dataframe(df)
            toggle_download(
                label="Excel",
                convert_args=(df.to_pandas(),) if (df is not None and getattr(df, "height", 0) > 0) else (None,),  # noqa: E501
                file_name="keywords_tags.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                location=st.sidebar
            )
        else:
            st.warning("No n-grams match the current filters.", icon=":material/info:")

        # Add new table generation option in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            body=(
                "### Generate new table\n\n"
                "Use the button to reset the n-grams or cluster table and start over."
                )
            )
        # Action button to create a new table
        if st.sidebar.button(
            label="Create a New Table",
            icon=":material/refresh:",
        ):
            st.session_state[user_session_id][CorpusKeys.TARGET][SessionKeys.NGRAMS] = {}
            update_session(SessionKeys.NGRAMS, False, user_session_id)
            st.rerun()

    except Exception as e:
        st.error(f"Error loading n-grams table: {str(e)}", icon=":material/error:")
        st.info("Try regenerating the n-grams table if this error persists.")


def render_ngrams_generation_interface(user_session_id: str, session: dict) -> None:
    """Render the interface for generating new n-grams/clusters."""
    st.markdown(
        body=(
            ":material/priority: Select either **N-grams** or **Clusters** "
            "from the options below.\n\n"
            ":material/manufacturing: Use the button in the sidebar to "
            "**generate the table**."
            )
        )

    st.markdown("---")

    ngram_type = st.radio(
        "What kind of table would you like to generate?",
        ["N-grams", "Clusters"],
        captions=[
            """:material/format_letter_spacing: Create a table of n-grams
            with a relative frequency > 10 (per million words)."
            """,
            """:material/match_word: Create counts of clusters
            that contain a specific word, part-of-a-word, or tag.
            """],
        horizontal=False,
        index=None,
        help=(
            "N-grams are sequences of words or tags that occur together "
            "in a corpus. Clusters are sequences of words or tags that "
            "contain a specific word, part-of-a-word, or tag. "
            "N-grams are useful for identifying common phrases, "
            "while clusters are useful for identifying patterns "
            "related to specific words or morphemes (like *-tion*)."
            )
        )

    # Render configuration based on selection
    if ngram_type == 'N-grams':
        render_ngrams_config(user_session_id, session)
    elif ngram_type == 'Clusters':
        render_clusters_config(user_session_id, session)


def render_ngrams_config(user_session_id: str, session: dict) -> None:
    """Render N-grams configuration using expander layout."""
    with st.expander("N-grams Configuration", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Span")
            ngram_span = st.radio(
                'Span of your n-grams:',
                SPAN_OPTIONS,
                horizontal=True,
                help=(
                    "The span of your n-grams determines how many words or "
                    "tags are included in each n-gram. For example, a span "
                    "of 2 will create bigrams (two-word sequences), a span "
                    "of 3 will create trigrams (three-word sequences), and "
                    "so on."
                    )
                )

        with col2:
            st.markdown("#### Tagset")
            tag_radio = st.radio(
                "Select a tagset:",
                list(TAGSET_CONFIG.keys()),
                horizontal=True,
                help=(
                    "Choose the tagset to use for generating n-grams. "
                    "Parts-of-Speech (POS) tags are used for grammatical "
                    "analysis, while DocuScope tags are used for "
                    "rhetorical analysis."
                    )
                )
            ts = TAGSET_CONFIG[tag_radio]

    # Keep generation button in sidebar for consistency
    st.sidebar.markdown("### Generate table")
    st.sidebar.markdown("Use the button to process a table.")

    # Create a custom action that handles validation and shows appropriate errors
    def ngrams_action():
        if not has_target_corpus(session):
            render_corpus_not_loaded_error()
            return

        # If validation passes, generate the n-grams
        generate_ngrams(user_session_id, ngram_span, ts)

    sidebar_action_button(
        button_label="N-grams Table",
        button_icon=":material/manufacturing:",
        preconditions=[True],  # Always allow button click, handle validation in action
        action=ngrams_action,
        spinner_message="Processing n-grams..."
    )

    # Display warning if there is an issue
    if st.session_state[user_session_id].get(WarningKeys.NGRAM):
        msg, icon = st.session_state[user_session_id][WarningKeys.NGRAM]
        st.error(msg, icon=icon)


def render_clusters_config(user_session_id: str, session: dict) -> None:
    """Render clusters configuration using expander layout."""
    # Initialize variables
    tag = None
    search = None
    node_word = None
    from_anchor = None
    position = 1
    ngram_span = 2
    ts = 'pos'

    # Load metadata if available
    metadata_target = None
    if has_target_corpus(session):
        metadata_target = load_metadata(CorpusKeys.TARGET, user_session_id)

    with st.expander("Clusters Configuration", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Search Mode")
            from_anchor = st.radio(
                "Create clusters from:",
                ("Token", "Tag"),
                horizontal=True,
                help=(
                    "Choose whether to create clusters based on a specific "
                    "word (token) or a tag. If you choose 'Token', you can "
                    "specify a word and how to search for it. If you choose "
                    "'Tag', you can select a tag from the available tagsets."
                    )
                )

            if from_anchor == 'Token':
                node_word = st.text_input("Node word:")
                search_mode = st.radio(
                    "Select search type:",
                    list(SEARCH_MODE_CONFIG.keys()),
                    horizontal=True,
                    help=(
                        "Choose how to search for the node word. "
                        "'Fixed' will match the exact word, 'Starts with' "
                        "will match words that begin with a character sequence, "
                        "'Ends with' will match words that end with the character "
                        "sequence, and 'Contains' will match words that contain "
                        "the character sequence anywhere."
                        )
                    )
                search = SEARCH_MODE_CONFIG[search_mode]

        with col2:
            st.markdown("#### Tagset & Parameters")
            tag_radio = st.radio(
                "Select a tagset:",
                list(TAGSET_CONFIG.keys()),
                horizontal=True,
                help=(
                    "Choose the tagset to use for clustering. "
                    "Parts-of-Speech (POS) tags are used for grammatical "
                    "analysis, while DocuScope tags are used for "
                    "rhetorical analysis."
                    )
                )
            ts = TAGSET_CONFIG[tag_radio]

            if from_anchor == 'Tag':
                if (not has_target_corpus(session) or
                        not metadata_target):
                    tag = st.selectbox(
                        'Choose a tag:',
                        ['No tags currently loaded']
                        )
                else:
                    tag_key = 'tags_pos' if tag_radio == 'Parts-of-Speech' else 'tags_ds'
                    available_tags = metadata_target.get(tag_key, [{}])[0].get('tags', [])
                    tag = st.selectbox('Choose a tag:', available_tags)
                    node_word = 'by_tag'

            st.markdown("#### Span & Position")
            ngram_span = st.radio(
                'Span of your clusters:',
                SPAN_OPTIONS,
                horizontal=True,
                help=(
                    "The span of your clusters determines how many words or "
                    "tags are included in each cluster."
                    )
                )
            position = st.selectbox(
                'Position of your node word or tag:',
                list(range(1, 1 + ngram_span))
                )

    # Keep generation button in sidebar for consistency
    st.sidebar.markdown("### Generate table")
    st.sidebar.markdown("Use the button to process a table.")

    # Validate configuration and show specific error messages only after button attempt
    corpus_loaded = has_target_corpus(session)
    is_valid_config = (
        (from_anchor == 'Token' and node_word and node_word.strip()) or
        (from_anchor == 'Tag' and tag and tag != 'No tags currently loaded')
    )

    # Create a custom action that handles validation and shows appropriate errors
    def clusters_action():
        if not corpus_loaded:
            st.error(
                body=(
                    "It doesn't look like you've loaded the necessary data yet. "
                    "Most apps require a target corpus to be loaded "
                    "before you can run them. "
                    "But other apps may require a reference corpus "
                    "or the processing of metadata. "
                    "You can review and update your corpus data "
                    "by navigating to **Manage Corpus Data**:"
                ),
                icon=":material/sentiment_stressed:"
            )
            st.page_link(
                page="pages/1_load_corpus.py",
                label="Manage Corpus Data",
                icon=":material/database:",
            )
            return
        elif not is_valid_config:
            if from_anchor == 'Token':
                st.error(
                    "Please enter a node word to search for.",
                    icon=":material/edit:"
                )
            elif from_anchor == 'Tag':
                st.error(
                    "Please select a valid tag from the dropdown.",
                    icon=":material/edit:"
                )
            return

        # If all validations pass, generate the clusters
        generate_clusters(
            user_session_id, from_anchor, node_word,
            tag, position, ngram_span, search, ts
        )

    sidebar_action_button(
        button_label="Clusters Table",
        button_icon=":material/manufacturing:",
        preconditions=[True],  # Always allow button click, handle validation in action
        action=clusters_action,
        spinner_message="Processing clusters..."
    )

    # Display warning if there is an issue
    if st.session_state[user_session_id].get(WarningKeys.NGRAM):
        msg, icon = st.session_state[user_session_id][WarningKeys.NGRAM]
        st.error(msg, icon=icon)


def main():
    """
    Main function to run the Streamlit app for n-grams and clusters.
    It initializes the user session, loads the necessary data,
    and displays the n-grams or clusters based on user selection.
    """
    # Set login requirements for navigaton
    require_login()
    menu()
    st.markdown(
        body=f"## {TITLE}",
        help=(
            "This page allows you to generate and view n-grams or clusters "
            "from your target corpus. N-grams are sequences of words or tags "
            "that occur together in a corpus, while clusters are sequences "
            "of words or tags that contain a specific word, part-of-a-word, "
            "or tag. You can filter the n-grams or clusters by tags and "
            "download the results in Excel format. Use the sidebar to "
            "generate new tables or access help documentation."
            )
        )
    # Get or initialize user session
    user_session_id, session = get_or_init_user_session()

    sidebar_help_link("ngrams.html")

    # Check if n-grams are already generated
    if session.get(SessionKeys.NGRAMS, [False])[0] is True:
        render_ngrams_display_interface(user_session_id, session)
    else:
        render_ngrams_generation_interface(user_session_id, session)


if __name__ == "__main__":
    main()
