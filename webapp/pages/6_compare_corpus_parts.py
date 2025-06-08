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


TITLE = "Compare Corpus Parts"
ICON = ":material/compare_arrows:"

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
        url="https://browndw.github.io/docuscope-docs/guide/compare-corpus-parts.html",
        icon=":material/help:"
        )

    if session.get('keyness_parts')[0] is True:

        _utils.handlers.load_widget_state(
            pathlib.Path(__file__).stem,
            user_session_id
            )
        metadata_target = _utils.handlers.load_metadata(
            'target',
            user_session_id
            )

        col1, col2 = st.columns([1, 1])
        with col1:
            st.info(
                _utils.content.message_target_parts(
                    metadata_target.get('keyness_parts')[0]['temp'])
                )
        with col2:
            st.info(
                _utils.content.message_reference_parts(
                    metadata_target.get('keyness_parts')[0]['temp'])
                    )

        st.markdown("Showing keywords that reach significance at *p* < 0.01")

        st.sidebar.markdown("### Comparison")
        table_radio = st.sidebar.radio(
            "Select the keyness table to display:",
            ("Tokens", "Tags Only"),
            key=_utils.handlers.persist(
                "cp_radio1",
                pathlib.Path(__file__).stem,
                user_session_id),
            horizontal=True
                )

        st.sidebar.markdown("---")

        if table_radio == 'Tokens':
            tag_radio_tokens = st.sidebar.radio(
                "Select tags to display:",
                ("Parts-of-Speech",
                 "DocuScope"),
                key=_utils.handlers.persist(
                     "cp_radio2",
                     pathlib.Path(__file__).stem,
                     user_session_id),
                horizontal=True
                )

            if tag_radio_tokens == 'Parts-of-Speech':
                tag_type = st.sidebar.radio(
                    "Select from general or specific tags",
                    ("General", "Specific"),
                    horizontal=True
                    )
                if tag_type == 'General':
                    df = st.session_state[user_session_id]["target"]["kw_pos_cp"]
                    df = _utils.analysis.freq_simplify_pl(df)
                else:
                    df = st.session_state[user_session_id]["target"]["kw_pos_cp"]
            else:
                df = st.session_state[user_session_id]["target"]["kw_ds_cp"]

            if df.height == 0 or df is None:
                cats = []
            elif df.height > 0:
                cats = sorted(df.get_column("Tag").unique().to_list())

            filter_vals = st.multiselect("Select tags to filter:", (cats))
            if len(filter_vals) > 0:
                df = df.filter(pl.col("Tag").is_in(filter_vals))

            st.dataframe(
                df,
                hide_index=True,
                column_config=_utils.formatters.get_streamlit_column_config(df)
                )

            with st.expander("Column explanation"):
                st.markdown(_utils.content.message_columns_keyness)

            st.sidebar.markdown("---")

            _utils.formatters.toggle_download(
                label="Excel",
                convert_func=_utils.formatters.convert_to_excel,
                convert_args=(df.to_pandas(),),
                file_name="keywords_tokens.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                message=_utils.content.message_download,
                location=st.sidebar
            )

            st.sidebar.markdown("---")

            st.sidebar.markdown("### Generate new table")
            st.sidebar.markdown("""
                            Click the button to reset the keyness table.
                            """)

            if st.sidebar.button(
                label="Compare New Categories",
                icon=":material/refresh:",
                ):
                if "kw_pos_cp" not in st.session_state[user_session_id]["target"]:  # noqa: E501
                    st.session_state[user_session_id]["target"]["kw_pos_cp"] = {}
                st.session_state[user_session_id]["target"]["kw_pos_cp"] = {}

                if "kw_ds_cp" not in st.session_state[user_session_id]["target"]:  # noqa: E501
                    st.session_state[user_session_id]["target"]["kw_ds_cp"] = {}
                st.session_state[user_session_id]["target"]["kw_ds_cp"] = {}

                if "kt_pos_cp" not in st.session_state[user_session_id]["target"]:  # noqa: E501
                    st.session_state[user_session_id]["target"]["kt_pos_cp"] = {}
                st.session_state[user_session_id]["target"]["kt_pos_cp"] = {}

                if "kt_ds_cp" not in st.session_state[user_session_id]["target"]:  # noqa: E501
                    st.session_state[user_session_id]["target"]["kt_ds_cp"] = {}
                st.session_state[user_session_id]["target"]["kt_ds_cp"] = {}

                _utils.handlers.update_session(
                    'keyness_parts',
                    False,
                    user_session_id
                    )

                st.rerun()

            st.sidebar.markdown("---")

        else:

            st.sidebar.markdown("### Tagset")
            tag_radio_tags = st.sidebar.radio(
                "Select tags to display:",
                ("Parts-of-Speech",
                 "DocuScope"),
                key=_utils.handlers.persist(
                    "cp_radio3", pathlib.Path(__file__).stem, user_session_id
                    ),
                horizontal=True
                )

            if tag_radio_tags == 'Parts-of-Speech':
                df = (
                    st.session_state[user_session_id]["target"]["kt_pos_cp"]
                    .filter(pl.col("Tag") != "FU")
                    )

            else:
                df = (
                    st.session_state[user_session_id]["target"]["kt_ds_cp"]
                    .filter(pl.col("Tag") != "Untagged")
                    )

            tab1, tab2 = st.tabs(["Keyness Table", "Keyness Plot"])
            with tab1:
                if df.height == 0 or df is None:
                    cats = []
                elif df.height > 0:
                    cats = sorted(df.get_column("Tag").unique().to_list())

                filter_vals = st.multiselect("Select tags to filter:", (cats))
                if len(filter_vals) > 0:
                    df = df.filter(pl.col("Tag").is_in(filter_vals))

                st.dataframe(
                    df,
                    hide_index=True,
                    column_config=_utils.formatters.get_streamlit_column_config(df)
                    )

            with tab2:
                if df.height > 0:
                    fig = _utils.formatters.plot_compare_corpus_bar(df)
                    st.plotly_chart(fig, use_container_width=True)

            with st.expander("Column explanation"):
                st.markdown(_utils.content.message_columns_keyness)

            st.sidebar.markdown("---")

            _utils.formatters.toggle_download(
                label="Excel",
                convert_func=_utils.formatters.convert_to_excel,
                convert_args=(df.to_pandas(),),
                file_name="keywords_tags.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                message=_utils.content.message_download,
                location=st.sidebar
            )

            st.sidebar.markdown("---")

            st.sidebar.markdown(_utils.content.message_reset_table)

            if st.sidebar.button("Compare New Categories", icon=":material/refresh:"):
                for key in ["kw_pos_cp", "kw_ds_cp", "kt_pos_cp", "kt_ds_cp"]:
                    st.session_state[user_session_id]["target"][key] = {}
                _utils.handlers.update_session('keyness_parts', False, user_session_id)
                st.rerun()

            st.sidebar.markdown("---")

    else:

        st.markdown(
            """
            ###### :material/manufacturing: \
            Use the options in the sidebar to generate a table from subcorpora.\n

            * To use this tool, you must first process **metadata**
            from your file names. This can be done from **Manage Corpus Data**.

            * Categories of interest can be placed at the beginning
            of file names before an underscore:
            ```
            BIO_G0_02_1.txt, BIO_G0_03_1.txt, ENG_G0_16_1.txt, ENG_G0_21_1.txt, HIS_G0_02_1.txt, HIS_G0_03_1.txt
            ```
            * Processing these names would yield the categories:
            ```
            BIO, ENG, HIS
            ```
            * Those categories could then be compared in any combination.\n

            :material/block: Selecting of the same category as target
            and reference is prevented.
            """  # noqa: E501
            )

        st.sidebar.markdown("### Select categories to compare")
        st.sidebar.markdown("""After **target** and **reference** categories
                            have been selected,
                            click the button to generate a keyness table.
                            """)

        if session.get('has_meta')[0] is True:
            metadata_target = _utils.handlers.load_metadata(
                'target',
                user_session_id
            )
            st.sidebar.markdown('#### Target corpus categories:')
            st.session_state[user_session_id]['tar'] = st.sidebar.multiselect(
                "Select target categories:",
                (sorted(set(metadata_target.get('doccats')[0]['cats']))),
                _utils.handlers.update_tar(user_session_id),
                key=f"tar_{user_session_id}"
            )

        else:
            st.sidebar.multiselect(
                "Select reference categories:",
                (['No categories to select']),
                key='empty_tar'
                )

        if session.get('has_meta')[0] is True:
            metadata_target = _utils.handlers.load_metadata(
                'target',
                user_session_id
                )

            st.sidebar.markdown('#### Reference corpus categories:')
            st.session_state[user_session_id]['ref'] = st.sidebar.multiselect(
                "Select reference categories:",
                (sorted(set(metadata_target.get('doccats')[0]['cats']))),
                _utils.handlers.update_ref(user_session_id),
                key=f"ref_{user_session_id}"
                )

        else:
            st.sidebar.multiselect(
                "Select reference categories:",
                (['No categories to select']),
                key='empty_ref'
                )

        st.sidebar.markdown("---")

        st.sidebar.markdown(_utils.content.message_generate_table)

        _utils.handlers.sidebar_action_button(
            button_label="Keyness Table of Corpus Parts",
            button_icon=":material/manufacturing:",
            preconditions=[
                session.get('has_target')[0],
            ],
            action=lambda: _utils.handlers.generate_keyness_parts(user_session_id),
            spinner_message="Generating keywords..."
        )

        if st.session_state[user_session_id].get("keyness_parts_warning"):
            msg, icon = st.session_state[user_session_id]["keyness_parts_warning"]
            st.warning(msg, icon=icon)

        st.sidebar.markdown("---")


if __name__ == "__main__":
    main()
