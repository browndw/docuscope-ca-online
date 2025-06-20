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

from webapp.utilities.handlers import get_or_init_user_session, sidebar_action_button, generate_tags_table   # noqa: E402, E501
from webapp.utilities.formatters import convert_corpus_to_zip   # noqa: E402
from webapp.menu import menu, require_login   # noqa: E402

TITLE = "Download Corpus Files"
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
            Generate files to use locally on your computer.
        :material/help:
        Use the Help link in the sidebar
        to learn more about the download options and
        the files you can save.
        """
        )

    if session.get('tags_table')[0] is True:

        corpus_select = st.radio(
            "Choose a corpus",
            ["Target", "Reference"],
            captions=["",
                      """You can only download reference
                      corpus data if you've processed one.
                      """])

        if corpus_select == "Target":
            data_select = st.radio(
                "Choose the data to download",
                ["Corpus file only", "All of the processed data"],
                captions=[
                    """This is the option you want
                    if you're planning to save your corpus
                    for future analysis using this tool.
                    """,
                    """This is the option you want
                    if you're planning to explore your data
                    outside of the tool, in coding enviroments
                    like R or Python.
                    The data include the token file,
                    frequency tables, and document-term-matrices.
                    """
                    ])

            if data_select == "Corpus file only":
                download_file = st.session_state[
                    user_session_id
                    ]["target"]["ds_tokens"].to_pandas().to_parquet()

                st.sidebar.markdown(
                    """#### Click the button \
                    to download your corpus file."""
                    )
                st.sidebar.download_button(
                    label="Download Corpus File",
                    icon=":material/download:",
                    data=download_file,
                    file_name="corpus.parquet",
                    mime="parquet",
                        )
                st.sidebar.markdown("---")

            if data_select == "All of the processed data":

                format_select = st.radio("Select a file format",
                                         ["CSV", "PARQUET"],
                                         horizontal=True)

                if format_select == "CSV":
                    with st.sidebar.status("Preparing files..."):
                        download_file = convert_corpus_to_zip(
                            user_session_id,
                            'target',
                            file_type='csv'
                            )

                        st.sidebar.markdown(
                            """#### Click the button to \
                            download your corpus files.
                            """
                            )
                        st.sidebar.download_button(
                            label="Download Corpus Files",
                            icon=":material/download:",
                            data=download_file,
                            file_name="corpus_files.zip",
                            mime="application/zip",
                            )
                        st.sidebar.markdown("---")

                if format_select == "PARQUET":
                    with st.sidebar.status("Preparing files..."):
                        download_file = convert_corpus_to_zip(
                            user_session_id,
                            'target'
                            )

                        st.sidebar.markdown(
                            """#### Click the button \
                            to download your corpus files."""
                            )
                        st.sidebar.download_button(
                            label="Download Corpus Files",
                            icon=":material/download:",
                            data=download_file,
                            file_name="corpus_files.zip",
                            mime="application/zip",
                            )
                        st.sidebar.markdown("---")

        if corpus_select == "Reference":
            if session.get('has_reference')[0] is False:
                st.error(
                    """
                    It doesn't look like you've loaded a reference corpus yet.
                    You can do this by clicking on the **Manage Corpus Data** button above.
                    """,
                    icon=":material/sentiment_stressed:"
                    )

            if session.get('has_reference')[0] is True:
                data_select = st.radio(
                    "Choose the data to download",
                    ["Corpus file only", "All of the processed data"],
                    captions=[
                        """This is the option you want
                        if you're planning to save your corpus
                        for future analysis using this tool.
                        """,
                        """This is the option you want
                        if you're planning to explore your data
                        outside of the tool, in coding enviroments
                        like R or Python.
                        The data include the token file,
                        frequency tables, and document-term-matrices.
                        """
                        ])

                if data_select == "Corpus file only":
                    download_file = st.session_state[
                        user_session_id
                        ]["reference"
                          ]["ds_tokens"].to_pandas().to_parquet()

                    st.sidebar.markdown(
                        """#### Click the button \
                        to download your corpus file."""
                        )
                    st.sidebar.download_button(
                        label="Download Corpus File",
                        data=download_file,
                        file_name="corpus.parquet",
                        mime="parquet",
                            )
                    st.sidebar.markdown("---")

                if data_select == "All of the processed data":

                    format_select = st.radio(
                        "Select a file format",
                        ["CSV",
                            "PARQUET"],
                        horizontal=True)

                    if format_select == "CSV":
                        with st.sidebar.status("Preparing files..."):
                            download_file = convert_corpus_to_zip(  # noqa: E501
                                user_session_id,
                                'reference',
                                file_type='csv')

                            st.sidebar.markdown("""### Click the button \
                                        to download your corpus files.
                                        """)
                            st.sidebar.download_button(
                                label="Download Corpus Files",
                                data=download_file,
                                file_name="corpus_files.zip",
                                mime="application/zip",
                                )
                            st.sidebar.markdown("---")

                    if format_select == "PARQUET":
                        with st.sidebar.status("Preparing files..."):
                            download_file = convert_corpus_to_zip(  # noqa: E501
                                user_session_id,
                                'reference'
                                )

                            st.sidebar.markdown("""### Click the button \
                                        to download your corpus files.
                                        """)
                            st.sidebar.download_button(
                                label="Download Corpus Files",
                                data=download_file,
                                file_name="corpus_files.zip",
                                mime="application/zip",
                                )
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
