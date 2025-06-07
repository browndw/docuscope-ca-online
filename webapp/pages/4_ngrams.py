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

import polars as pl
import streamlit as st

# Ensure project root is in sys.path for both desktop and online
project_root = pathlib.Path(__file__).parent.parents[1].resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import webapp.utilities as _utils   # noqa: E402
from webapp.menu import menu, require_login   # noqa: E402

TITLE = "N-gram and Cluster Frequency"
ICON = ":material/table_view:"

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )


def main():
    # Set login requirements for navigaton
    require_login()
    menu()
    st.markdown(f"## {TITLE}")
    # Get or initialize user session
    user_session_id, session = _utils.handlers.get_or_init_user_session()

    st.sidebar.link_button(
        label="Help",
        url="https://browndw.github.io/docuscope-docs/guide/ngrams.html",
        icon=":material/help:"
        )

    if session.get('ngrams')[0] is True:

        metadata_target = _utils.handlers.load_metadata(
            'target',
            user_session_id
            )

        df = st.session_state[user_session_id]["target"]["ngrams"]

        st.markdown(_utils.content.message_target_info(metadata_target))

        col1, col2 = st.columns(2)

        with col1:
            if df.height == 0 or df is None:
                cats_1 = []
            elif df.height > 0:
                cats_1 = sorted(
                    df.get_column("Tag_1").drop_nulls().unique().to_list()
                    )

            filter_tag_1 = st.multiselect(
                "Select tags to filter in position 1:",
                (cats_1)
                )
            if len(filter_tag_1) > 0:
                df = df.filter(pl.col("Tag_1").is_in(filter_tag_1))

            if "Tag_3" in df.columns:
                cats_3 = sorted(
                    df.get_column("Tag_3").drop_nulls().unique().to_list()
                    )
                filter_tag_3 = st.multiselect(
                    "Select tags to filter in position 3:",
                    (cats_3)
                    )
                if len(filter_tag_3) > 0:
                    df = df.filter(pl.col("Tag_3").is_in(filter_tag_3))

        with col2:
            if df.height == 0 or df is None:
                cats_2 = []
            elif df.height > 0:
                cats_2 = sorted(
                    df.get_column("Tag_2").drop_nulls().unique().to_list()
                    )

            filter_tag_2 = st.multiselect(
                "Select tags to filter in position 2:",
                (cats_2)
                )
            if len(filter_tag_2) > 0:
                df = df.filter(pl.col("Tag_2").is_in(filter_tag_2))

            if "Tag_4" in df.columns:
                cats_4 = sorted(
                    df.get_column("Tag_4").drop_nulls().unique().to_list()
                    )
                filter_tag_4 = st.multiselect(
                    "Select tags to filter in position 4:",
                    (cats_4)
                    )
                if len(filter_tag_4) > 0:
                    df = df.filter(pl.col("Tag_4").is_in(filter_tag_4))

        st.dataframe(
            df,
            hide_index=True,
            column_config=_utils.formatters.get_streamlit_column_config(df)
            )

        download_table = st.sidebar.toggle("Download to Excel?")
        if download_table is True:
            with st.sidebar:
                st.sidebar.markdown(_utils.content.message_download)
                download_file = _utils.formatters.convert_to_excel(
                    df.to_pandas()
                    )

                st.download_button(
                    label="Download to Excel",
                    data=download_file,
                    file_name="ngrams.xlsx",
                    mime="application/vnd.ms-excel",
                    )

        st.sidebar.markdown("---")

        st.sidebar.markdown("### Generate new table")
        st.sidebar.markdown("""
                            Click the button to reset the n-grams table.
                            """)

        if st.sidebar.button("Create a New Ngrams Table"):
            if "ngrams" not in st.session_state[user_session_id]["target"]:
                st.session_state[user_session_id]["target"]["ngrams"] = {}
            _utils.handlers.update_session(
                'ngrams',
                False,
                user_session_id
                )
            st.rerun()
        st.sidebar.markdown("---")

    else:
        if session.get("has_target")[0] is True:
            metadata_target = _utils.handlers.load_metadata(
                'target',
                user_session_id
                )

        st.markdown(_utils.content.message_ngrams)

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
            index=None
            )

        if ngram_type == 'N-grams':
            st.sidebar.markdown("### Span")
            ngram_span = st.sidebar.radio(
                'Span of your n-grams:',
                (2, 3, 4),
                horizontal=True
                )

            st.sidebar.markdown("---")

            tag_radio = st.sidebar.radio(
                "Select a tagset:",
                ("Parts-of-Speech", "DocuScope"),
                horizontal=True
                )
            if tag_radio == 'Parts-of-Speech':
                ts = 'pos'
            if tag_radio == 'DocuScope':
                ts = 'ds'

            st.sidebar.markdown("---")

            st.sidebar.markdown(_utils.content.message_generate_table)

            _utils.handlers.sidebar_action_button(
                button_label="N-grams Table",
                button_icon=":material/manufacturing:",
                preconditions=[
                    session.get('has_target')[0]
                ],
                action=lambda: _utils.handlers.generate_ngrams(
                    user_session_id, ngram_span, ts
                    ),
                spinner_message="Processing n-grams..."
            )

            if st.session_state[user_session_id].get("ngram_warning"):
                msg, icon = st.session_state[user_session_id]["ngram_warning"]
                st.error(msg, icon=icon)

        if ngram_type == 'Clusters':

            tag = None  # <-- Ensure tag is always defined
            search = None  # <-- If search is also conditionally set

            st.sidebar.markdown("### Search mode")
            st.sidebar.markdown("Create n-grams from a token or from a tag.")
            from_anchor = st.sidebar.radio(
                "Enter token or a tag:",
                ("Token", "Tag"),
                horizontal=True
                )

            if from_anchor == 'Token':
                node_word = st.sidebar.text_input("Node word:")

                search_mode = st.sidebar.radio(
                    "Select search type:",
                    ("Fixed", "Starts with", "Ends with", "Contains"),
                    horizontal=True
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
                    horizontal=True
                    )

                if tag_radio == 'Parts-of-Speech':
                    ts = 'pos'
                if tag_radio == 'DocuScope':
                    ts = 'ds'

            if from_anchor == 'Tag':
                tag_radio = st.sidebar.radio(
                    "Select a tagset:",
                    ("Parts-of-Speech", "DocuScope"),
                    horizontal=True
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
            ngram_span = st.sidebar.radio(
                'Span of your n-grams:',
                (2, 3, 4),
                horizontal=True
                )
            position = st.sidebar.selectbox(
                'Position of your node word or tag:',
                (list(range(1, 1+ngram_span)))
                )

            st.sidebar.markdown("---")

            st.sidebar.markdown(_utils.content.message_generate_table)

            _utils.handlers.sidebar_action_button(
                button_label="Clusters Table",
                button_icon=":material/manufacturing:",
                preconditions=[
                    session.get('has_target')[0],  # Only check for corpus presence here
                ],
                action=lambda: _utils.handlers.generate_clusters(
                    user_session_id, from_anchor, node_word,
                    tag, position, ngram_span, search, ts
                ),
                spinner_message="Processing clusters..."
            )
            # Display warning in main container
            if st.session_state[user_session_id].get("ngram_warning"):
                msg, icon = st.session_state[user_session_id]["ngram_warning"]
                st.error(msg, icon=icon)

            st.sidebar.markdown("---")


if __name__ == "__main__":
    main()
