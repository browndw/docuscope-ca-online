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

from webapp.utilities.handlers import get_or_init_user_session, sidebar_action_button, generate_tags_table  # noqa: E402, E501
from webapp.utilities.formatters import convert_to_zip  # noqa: E402
from webapp.menu import menu, require_login   # noqa: E402

TITLE = "Download Tagged Files"
ICON = ":material/download:"

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
    user_session_id, session = get_or_init_user_session()

    st.sidebar.link_button(
        label="Help",
        url="https://browndw.github.io/docuscope-docs/guide/download-tagged-files.html",
        icon=":material/help:"
        )

    # Display a markdown message for downloading tagged files
    st.markdown(
        """
        ##### :material/manufacturing: \
        Generate a zipped folder of tagged text files.
        :material/help:
        Use the Help link in the sidebar
        to learn more about how the embbed tags are formatted.
        """
        )

    if session.get('tags_table')[0] is True:

        # Sidebar for selecting the tagset to embed
        st.sidebar.markdown("### Tagset to embed")
        download_radio = st.sidebar.radio(
            "Select tagset:",
            ("Parts-of-Speech", "DocuScope"),
            horizontal=True
            )

        # Determine the tagset based on user selection
        if download_radio == 'Parts-of-Speech':
            tagset = 'pos'
        else:
            tagset = 'ds'

        # Check if the session has a target and proceed with file download
        if session.get('has_target')[0] is True:
            tok_pl = st.session_state[user_session_id]["target"]["ds_tokens"]

            with st.sidebar:
                # Convert the tokenized data to a zip file
                download_file = convert_to_zip(tok_pl, tagset)

                # Provide a download button for the zip file
                st.download_button(
                    label="Download to Zip",
                    data=download_file,
                    file_name="tagged_files.zip",
                    mime="application/zip",
                    )

        # Add a horizontal rule in the sidebar
        st.sidebar.markdown("---")

    else:

        st.sidebar.markdown(
            """
            ### Load tables
            Use the button to load corpus tables.
            """,
            help="For tables to be loaded, you must first process a target corpus using: **:material/database: Manage Corpus Data**"  # noqa: E501
            )

        sidebar_action_button(
            button_label="Load Data",
            button_icon=":material/manufacturing:",
            preconditions=[
                session.get('has_target')[0],
            ],
            action=lambda: generate_tags_table(
                user_session_id
            ),
            spinner_message="Loading data..."
        )

        if st.session_state[user_session_id].get("tags_warning"):
            msg, icon = st.session_state[user_session_id]["tags_warning"]
            st.warning(msg, icon=icon)

        st.sidebar.markdown("---")


if __name__ == "__main__":
    main()
