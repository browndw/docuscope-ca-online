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

import webapp.utilities as _utils   # noqa: E402
from webapp.menu import menu, require_login   # noqa: E402

TITLE = "KWIC Tables"
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

    if session.get('kwic')[0] is True:

        df = st.session_state[user_session_id]["target"]["kwic"]

        st.dataframe(df, hide_index=True)

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
                    file_name="kwic.xlsx",
                    mime="application/vnd.ms-excel",
                    )

        st.sidebar.markdown("---")

        st.sidebar.markdown(_utils.content.message_reset_table)

        if st.sidebar.button("Create New KWIC Table"):
            if "kwic" not in st.session_state[user_session_id]["target"]:
                st.session_state[user_session_id]["target"]["kwic"] = {}
            _utils.handlers.update_session(
                'kwic',
                False,
                user_session_id
                )

            st.rerun()

        st.sidebar.markdown("---")

    else:

        st.markdown(_utils.content.message_kwic)

        st.sidebar.markdown("### Node word")
        st.sidebar.markdown("""Enter a node word without spaces.
                            """)
        node_word = st.sidebar.text_input("Node word")

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Search mode")
        search_mode = st.sidebar.radio(
            "Select search type:",
            ("Fixed", "Starts with", "Ends with", "Contains"),
            horizontal=True
            )

        if search_mode == "Fixed":
            search_type = "fixed"
        elif search_mode == "Starts with":
            search_type = "starts_with"
        elif search_mode == "Ends with":
            search_type = "ends_with"
        else:
            search_type = "contains"

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Case")
        case_sensitive = st.sidebar.checkbox("Make search case sensitive")

        if bool(case_sensitive) is True:
            ignore_case = False
        else:
            ignore_case = True

        st.sidebar.markdown("---")
        st.sidebar.markdown(_utils.content.message_generate_table)

        _utils.handlers.sidebar_action_button(
            button_label="KWIC",
            button_icon=":material/manufacturing:",
            preconditions=[
                session.get('has_target')[0],
            ],
            action=lambda: _utils.handlers.generate_kwic(
                user_session_id, node_word, search_type, ignore_case
            ),
            spinner_message="Processing KWIC..."
        )

        if st.session_state[user_session_id].get("kwic_warning"):
            msg, icon = st.session_state[user_session_id]["kwic_warning"]
            st.warning(msg, icon=icon)

        st.sidebar.markdown("---")


if __name__ == "__main__":
    main()
