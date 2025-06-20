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

TITLE = "Collocates"
ICON = ":material/network_node:"

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
        url="https://browndw.github.io/docuscope-docs/guide/collocations.html",
        icon=":material/help:"
        )

    if session.get('collocations')[0] is True:

        metadata_target = _utils.handlers.load_metadata(
            'target',
            user_session_id
            )

        df = st.session_state[user_session_id]["target"]["collocations"]

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(
                _utils.content.message_target_info(metadata_target)
                )
        with col2:
            st.markdown(_utils.content.message_collocation_info(
                metadata_target.get('collocations')[0]['temp'])
                )

        if df.height == 0 or df is None:
            cats = []
        elif df.height > 0:
            cats = sorted(df.get_column("Tag").unique().to_list())

        filter_vals = st.multiselect("Select tags to filter:", (cats))
        if len(filter_vals) > 0:
            df = df.filter(pl.col("Tag").is_in(filter_vals))

        st.dataframe(df,
                     hide_index=True,
                     column_config=_utils.formatters.get_streamlit_column_config(df)
                     )

        download_table = st.sidebar.toggle("Download to Excel?")
        if download_table is True:
            with st.sidebar:
                st.markdown(_utils.content.message_download)
                download_file = _utils.formatters.convert_to_excel(
                    df.to_pandas()
                    )

                st.download_button(
                    label="Download to Excel",
                    data=download_file,
                    file_name="collocations.xlsx",
                    mime="application/vnd.ms-excel",
                    )
        st.sidebar.markdown("---")

        st.sidebar.markdown(_utils.content.message_reset_table)

        if st.sidebar.button("Create New Collocations Table"):
            if "collocations" not in st.session_state[user_session_id]["target"]:  # noqa: E501
                st.session_state[
                    user_session_id
                    ]["target"]["collocations"] = {}
            st.session_state[user_session_id]["target"]["collocations"] = {}

            _utils.handlers.update_session(
                'collocations',
                False,
                user_session_id
                )
            st.rerun()

        st.sidebar.markdown("---")

    else:

        st.markdown(_utils.content.message_collocations)

        if session.get("has_target")[0] is True:
            metadata_target = _utils.handlers.load_metadata(
                'target',
                user_session_id
                )

        st.sidebar.markdown("### Node word")
        st.sidebar.markdown("""Enter a node word without spaces.
                            """)
        node_word = st.sidebar.text_input("Node word:")

        st.sidebar.markdown("---")

        with st.sidebar.expander("Span explanation",
                                 icon=":material/fit_width:"):
            st.markdown(_utils.content.message_span)

        st.sidebar.markdown("### Span")
        to_left = st.sidebar.slider(
            "Choose a span to the left of the node word:",
            0,
            9,
            (4)
            )
        to_right = st.sidebar.slider(
            "Choose a span to the right of the node word:",
            0,
            9,
            (4)
            )

        st.sidebar.markdown("---")
        with st.sidebar.expander("Statistics explanation",
                                 icon=":material/functions:"):
            st.markdown(_utils.content.message_association_measures)

        st.sidebar.markdown("### Association measure")
        stat_mode = st.sidebar.radio(
            "Select a statistic:",
            ["NPMI", "PMI 2", "PMI 3", "PMI"],
            horizontal=True)

        if stat_mode == "PMI":
            stat_mode = "pmi"
        elif stat_mode == "PMI 2":
            stat_mode = "pmi2"
        elif stat_mode == "PMI 3":
            stat_mode = "pmi3"
        elif stat_mode == "NPMI":
            stat_mode = "npmi"

        st.sidebar.markdown("---")
        with st.sidebar.expander("Anchor tag for node word explanation",
                                 icon=":material/anchor:"):
            st.markdown(_utils.content.message_anchor_tags)

        st.sidebar.markdown("### Anchor tag")
        tag_radio = st.sidebar.radio(
            "Select tagset for node word:",
            ("No Tag", "Parts-of-Speech", "DocuScope"),
            horizontal=True
            )
        if tag_radio == 'Parts-of-Speech':
            tag_type = st.sidebar.radio(
                "Select from general or specific tags",
                ("General", "Specific"),
                horizontal=True
                )
            if tag_type == 'General':
                node_tag = st.sidebar.selectbox(
                    "Select tag:",
                    ("Noun Common",
                     "Verb Lex",
                     "Adjective",
                     "Adverb")
                     )
                if node_tag == "Noun Common":
                    node_tag = "NN"
                elif node_tag == "Verb Lex":
                    node_tag = "VV"
                elif node_tag == "Adjective":
                    node_tag = "JJ"
                elif node_tag == "Adverb":
                    node_tag = "R"
            else:
                if session.get('has_target')[0] is False:
                    node_tag = st.sidebar.selectbox(
                        'Choose a tag:',
                        ['No tags currently loaded']
                        )
                else:
                    node_tag = st.sidebar.selectbox(
                        'Choose a tag:',
                        metadata_target.get('tags_pos')[0]['tags']
                        )
            count_by = 'pos'

        elif tag_radio == 'DocuScope':
            if session.get('has_target')[0] is False:
                node_tag = st.sidebar.selectbox(
                    'Choose a tag:',
                    ['No tags currently loaded']
                    )
            else:
                node_tag = st.sidebar.selectbox(
                    'Choose a tag:', metadata_target.get('tags_ds')[0]['tags']
                    )
                count_by = 'ds'
        else:
            node_tag = None
            count_by = 'pos'

        st.sidebar.markdown("---")
        st.sidebar.markdown(_utils.content.message_generate_table)

        _utils.handlers.sidebar_action_button(
            button_label="Collocations",
            button_icon=":material/manufacturing:",
            preconditions=[
                session.get('has_target')[0],
            ],
            action=lambda: _utils.handlers.generate_collocations(
                user_session_id, node_word, node_tag, to_left, to_right, stat_mode, count_by
            ),
            spinner_message="Processing collocates..."
        )

        if st.session_state[user_session_id].get("collocations_warning"):
            msg, icon = st.session_state[user_session_id]["collocations_warning"]
            st.warning(msg, icon=icon)

        st.sidebar.markdown("---")


if __name__ == "__main__":
    main()
