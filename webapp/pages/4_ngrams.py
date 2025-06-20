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
import sys

import streamlit as st

# Ensure project root is in sys.path for both desktop and online
project_root = pathlib.Path(__file__).parent.parents[1].resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from webapp.utilities.handlers import (  # noqa: E402
    generate_clusters,
    generate_ngrams,
    get_or_init_user_session,
    load_metadata,
    update_session
    )
from webapp.utilities.ui import (   # noqa: E402
    multi_tag_filter_multiselect,
    render_dataframe,
    sidebar_action_button,
    sidebar_help_link,
    target_info,
    toggle_download
    )
from webapp.utilities.formatters import (  # noqa: E402
    convert_to_excel
    )
from webapp.menu import (   # noqa: E402
    menu,
    require_login
    )

TITLE = "N-gram and Cluster Frequency"
ICON = ":material/table_view:"

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )


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
    if session.get('ngrams', [False])[0] is True:
        metadata_target = load_metadata(
            'target',
            user_session_id
            )
        # Load the session state for n-grams
        df = st.session_state[user_session_id]["target"]["ngrams"]
        # Display target information
        st.info(target_info(metadata_target))
        # Display the n-grams table
        if df is not None and getattr(df, "height", 0) > 0:
            tag_cols = [col for col in df.columns if col.startswith("Tag_")]
            df, selections = multi_tag_filter_multiselect(df, tag_cols)
            if df is None or getattr(df, "height", 0) == 0:
                st.warning("No n-grams match the current filters.")
            render_dataframe(df)
        else:
            st.warning("No n-grams match the current filters.")

        st.sidebar.markdown("---")

        # Toggle download options for the n-grams table
        toggle_download(
            label="Excel",
            convert_func=convert_to_excel,
            convert_args=(df.to_pandas(),) if (df is not None and getattr(df, "height", 0) > 0) else (None,),  # noqa: E501
            file_name="ngrams.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            location=st.sidebar
        )

        st.sidebar.markdown("---")
        # Display the sidebar header for generating a new n-grams table
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
            st.session_state[user_session_id]["target"]["ngrams"] = {}
            update_session(
                'ngrams',
                False,
                user_session_id
            )
            st.rerun()

        st.sidebar.markdown("---")

    else:
        if session.get("has_target")[0] is True:
            metadata_target = load_metadata(
                'target',
                user_session_id
                )

        st.markdown(
            body=(
                ":material/priority: Select either **N-grams** or **Clusters** from the options below.\n\n"  # noqa: E501
                ":material/manufacturing: Use the button in the sidebar to **generate the table**.\n\n"  # noqa: E501
                ":material/priority: A **target corpus** must be loaded first.\n\n"
                ":material/priority: After the table has been generated, "
                "you will be able to **toggle between the tagsets**."
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
        # Set the tagset variable based on the ngram_type
        if ngram_type == 'N-grams':
            st.sidebar.markdown("### Span")
            ngram_span = st.sidebar.radio(
                'Span of your n-grams:',
                (2, 3, 4),
                horizontal=True,
                help=(
                    "The span of your n-grams determines how many words or "
                    "tags are included in each n-gram. For example, a span "
                    "of 2 will create bigrams (two-word sequences), a span "
                    "of 3 will create trigrams (three-word sequences), and "
                    "so on."
                    )
                )

            st.sidebar.markdown("---")
            # Select the tagset for n-grams
            tag_radio = st.sidebar.radio(
                "Select a tagset:",
                ("Parts-of-Speech", "DocuScope"),
                horizontal=True,
                help=(
                    "Choose the tagset to use for generating n-grams. "
                    "Parts-of-Speech (POS) tags are used for grammatical "
                    "analysis, while DocuScope tags are used for "
                    "rhetorical analysis."
                    )
                )
            if tag_radio == 'Parts-of-Speech':
                ts = 'pos'
            if tag_radio == 'DocuScope':
                ts = 'ds'

            st.sidebar.markdown("---")

            # Display the sidebar header for generating frequency table
            st.sidebar.markdown(
                body=(
                    "### Generate table\n\n"
                    "Use the button to process a table."
                    ),
                help=(
                    "Tables are generated based on the loaded target corpus. "
                    "You can filter the table after it has been generated. "
                    "The table will include ngrams for the selected tagsets.\n\n"
                    "Click on the **Help** button for more information on how to use this app."  # noqa: E501
                    )
                )
            # Action button to generate n-grams
            sidebar_action_button(
                button_label="N-grams Table",
                button_icon=":material/manufacturing:",
                preconditions=[
                    session.get('has_target')[0]
                ],
                action=lambda: generate_ngrams(
                    user_session_id, ngram_span, ts
                    ),
                spinner_message="Processing n-grams..."
            )
            st.sidebar.markdown("---")
            # Check if there is a warning message for the n-grams table
            if st.session_state[user_session_id].get("ngram_warning"):
                msg, icon = st.session_state[user_session_id]["ngram_warning"]
                st.error(msg, icon=icon)

        # If n-grams are not selected, proceed with clusters
        if ngram_type == 'Clusters':
            # Initialize all variables
            tag = None
            search = None
            node_word = None

            st.sidebar.markdown("### Search mode")
            st.sidebar.markdown("Create n-grams from a token or from a tag.")
            from_anchor = st.sidebar.radio(
                "Enter token or a tag:",
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
                node_word = st.sidebar.text_input("Node word:")

                search_mode = st.sidebar.radio(
                    "Select search type:",
                    ("Fixed", "Starts with", "Ends with", "Contains"),
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
                if search_mode == "Fixed":
                    search = "fixed"
                elif search_mode == "Starts with":
                    search = "starts_with"
                elif search_mode == "Ends with":
                    search = "ends_with"
                else:
                    search = "contains"

                tag_radio = st.sidebar.radio(
                    "Select a tagset:",
                    ("Parts-of-Speech", "DocuScope"),
                    horizontal=True,
                    help=(
                        "Choose the tagset to use for clustering. "
                        "Parts-of-Speech (POS) tags are used for grammatical "
                        "analysis, while DocuScope tags are used for "
                        "rhetorical analysis."
                        )
                    )

                if tag_radio == 'Parts-of-Speech':
                    ts = 'pos'
                if tag_radio == 'DocuScope':
                    ts = 'ds'

            if from_anchor == 'Tag':
                tag_radio = st.sidebar.radio(
                    "Select a tagset:",
                    ("Parts-of-Speech", "DocuScope"),
                    horizontal=True,
                    help=(
                        "In this mode, you can select a tag from the "
                        "available tagsets to create clusters based on that tag. "
                        "For example, if 'NN1' is selected, the clusters will "
                        "include all n-grams where the node word is tagged as "
                        "as 'NN1' (noun, singular)."
                        )
                    )

                if tag_radio == 'Parts-of-Speech':
                    if session.get('has_target')[0] is False:
                        tag = st.sidebar.selectbox(
                            'Choose a tag:',
                            ['No tags currently loaded']
                            )
                    else:
                        tag = st.sidebar.selectbox(
                            'Choose a tag:',
                            metadata_target.get('tags_pos')[0]['tags']
                            )
                        ts = 'pos'
                        node_word = 'by_tag'

                if tag_radio == 'DocuScope':
                    if session.get('has_target')[0] is False:
                        tag = st.sidebar.selectbox(
                            'Choose a tag:',
                            ['No tags currently loaded']
                            )
                    else:
                        tag = st.sidebar.selectbox(
                            'Choose a tag:',
                            metadata_target.get('tags_ds')[0]['tags']
                            )
                        ts = 'ds'
                        node_word = 'by_tag'

            st.sidebar.markdown("---")

            st.sidebar.markdown("### Span & position")
            # Set the span and position for clusters
            ngram_span = st.sidebar.radio(
                'Span of your clusters:',
                (2, 3, 4),
                horizontal=True,
                help=(
                    "The span of your clusters determines how many words or "
                    "tags are included in each cluster. For example, a span "
                    "of 2 will create two-word sequences, a span "
                    "of 3 will create three-word sequences, and "
                    "so on."
                    )
                )
            position = st.sidebar.selectbox(
                'Position of your node word or tag:',
                (list(range(1, 1+ngram_span)))
                )

            st.sidebar.markdown("---")

            # Display the sidebar header for generating frequency table
            st.sidebar.markdown(
                body=(
                    "### Generate table\n\n"
                    "Use the button to process a table."
                    ),
                help=(
                    "Tables are generated based on the loaded target corpus. "
                    "You can filter the table after it has been generated. "
                    "The table will include cluster frequencies for the selected tagsets.\n\n"  # noqa: E501
                    "Click on the **Help** button for more information on how to use this app."  # noqa: E501
                    )
                )
            # Action button to generate clusters
            sidebar_action_button(
                button_label="Clusters Table",
                button_icon=":material/manufacturing:",
                preconditions=[
                    session.get('has_target')[0],  # Only check for corpus presence here
                ],
                action=lambda: generate_clusters(
                    user_session_id, from_anchor, node_word,
                    tag, position, ngram_span, search, ts
                ),
                spinner_message="Processing clusters..."
            )
            # Display warning if there is an issue with n-grams
            if st.session_state[user_session_id].get("ngram_warning"):
                msg, icon = st.session_state[user_session_id]["ngram_warning"]
                st.error(msg, icon=icon)

            st.sidebar.markdown("---")


if __name__ == "__main__":
    main()
