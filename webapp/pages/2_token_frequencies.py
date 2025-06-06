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
import docuscospacy as ds
import polars as pl
import streamlit as st
import sys

# Ensure project root is in sys.path for both desktop and online
project_root = pathlib.Path(__file__).parent.parents[1].resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import webapp.utilities as _utils   # noqa: E402
from webapp.menu import menu, require_login   # noqa: E402

TITLE = "Token Frequencies"
ICON = ":material/table_view:"

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )


def main() -> None:
    # Set login requirements for navigaton
    require_login()
    menu()
    st.markdown(f"## {TITLE}")
    # Get or initialize user session
    user_session_id, session = _utils.handlers.get_or_init_user_session()

    st.sidebar.link_button(
        label="Help",
        url="https://browndw.github.io/docuscope-docs/guide/token-frequencies.html",
        icon=":material/help:"
        )

    if session.get('freq_table')[0] is True:

        _utils.handlers.load_widget_state(
            pathlib.Path(__file__).stem,
            user_session_id
            )
        metadata_target = _utils.handlers.load_metadata(
            'target',
            user_session_id
            )

        st.sidebar.markdown("### Tagset")

        tag_radio = st.sidebar.radio(
            "Select tags to display:", ("Parts-of-Speech", "DocuScope"),
            key=_utils.handlers.persist(
                "ft_radio",
                pathlib.Path(__file__).stem, user_session_id
                ), horizontal=True
            )

        if tag_radio == 'Parts-of-Speech':
            tag_type = st.sidebar.radio(
                "Select from general or specific tags",
                ("General", "Specific"),
                horizontal=True
                )
            if tag_type == 'General':
                df = st.session_state[user_session_id]["target"]["ft_pos"]
                df = ds.freq_simplify(df)
            else:
                df = st.session_state[user_session_id]["target"]["ft_pos"]
        else:
            df = st.session_state[user_session_id]["target"]["ft_ds"]

        st.markdown(_utils.content.message_target_info(metadata_target))

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

        st.sidebar.markdown("---")

        with st.sidebar.expander("Column explanation",
                                 icon=":material/view_column:"):
            st.markdown(_utils.content.message_columns_tokens)

        st.sidebar.markdown("---")

        _utils.formatters.toggle_download(
            label="Excel",
            convert_func=_utils.formatters.convert_to_excel,
            convert_args=(df.to_pandas(),),
            file_name="token_frequencies.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            message=_utils.content.message_download,
            location=st.sidebar
        )

    else:
        st.markdown(_utils.content.message_tables)
        st.sidebar.markdown(_utils.content.message_generate_table)

        _utils.handlers.sidebar_action_button(
            button_label="Frequency Table",
            button_icon=":material/manufacturing:",
            preconditions=[
                session.get('has_target')[0],
            ],
            action=lambda: _utils.handlers.generate_frequency_table(
                user_session_id
            ),
            spinner_message="Processing frequencies..."
        )
        if st.session_state[user_session_id].get("frequency_warning"):
            msg, icon = st.session_state[user_session_id]["frequency_warning"]
            st.warning(msg, icon=icon)

        st.sidebar.markdown("---")


if __name__ == "__main__":
    main()
