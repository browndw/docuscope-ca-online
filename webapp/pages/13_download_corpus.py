"""
This app provides an interface for downloading corpus files
for offline use. Users can download the target corpus and, if available,
the reference corpus in various formats (CSV or Parquet).
Users can choose to download the entire corpus data or just the corpus file itself.
It requires the user to be logged in and displays options for downloading
the target and reference corpus files.
"""

import streamlit as st

from webapp.utilities.session import (
    get_or_init_user_session, safe_session_get
    )
from webapp.utilities.exports import (
    handle_corpus_file_download,
    handle_all_data_download
)
from webapp.utilities.ui import (
    render_download_page_header,
    render_data_loading_interface,
    render_corpus_selection,
    render_data_type_selection,
    render_format_selection,
    check_reference_corpus_availability
)
from webapp.utilities.state import (
    SessionKeys
)
from webapp.menu import (
    menu,
    require_login
)

TITLE = "Download Corpus Files"
ICON = ":material/download:"

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )


def render_download_interface(user_session_id: str, session: dict) -> None:
    """
    Render the main download interface when tables are loaded.

    Parameters
    ----------
    user_session_id : str
        The user session identifier
    session : dict
        The session state dictionary
    """
    corpus_select = render_corpus_selection()

    if corpus_select == "Target":
        render_target_corpus_options(user_session_id)

    elif corpus_select == "Reference":
        if check_reference_corpus_availability(session):
            render_reference_corpus_options(user_session_id)


def render_target_corpus_options(user_session_id: str) -> None:
    """
    Render download options for target corpus.

    Parameters
    ----------
    user_session_id : str
        The user session identifier
    """
    data_select = render_data_type_selection()

    if data_select == "Corpus file only":
        handle_corpus_file_download(user_session_id, "Target")
    elif data_select == "All of the processed data":
        format_select = render_format_selection()
        handle_all_data_download(user_session_id, "Target", format_select)


def render_reference_corpus_options(user_session_id: str) -> None:
    """
    Render download options for reference corpus.

    Parameters
    ----------
    user_session_id : str
        The user session identifier
    """
    data_select = render_data_type_selection()

    if data_select == "Corpus file only":
        handle_corpus_file_download(user_session_id, "Reference")
    elif data_select == "All of the processed data":
        format_select = render_format_selection()
        handle_all_data_download(user_session_id, "Reference", format_select)


def main() -> None:
    """
    Main function to render the download corpus page.
    This function sets up the page layout, handles user sessions,
    and provides options for downloading corpus files.
    It requires the user to be logged in and displays options
    for downloading target and reference corpus files in various formats.
    """
    # Set login requirements for navigation
    require_login()
    menu()

    # Render page header with help link
    render_download_page_header(
        title=TITLE,
        help_url=(
            "https://browndw.github.io/docuscope-docs/guide/"
            "download-tagged-files.html"
        ),
        description=(
            "This page allows you to download your corpus files "
            "for offline use. You can download the target corpus "
            "and, if available, the reference corpus. "
            "The files can be downloaded in CSV or Parquet format, "
            "and you can choose to download the entire corpus data "
            "or just the corpus file itself."
        )
    )

    # Get or initialize user session
    user_session_id, session = get_or_init_user_session()

    # Display processing message
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

    # Check if tables are loaded
    if safe_session_get(session, SessionKeys.TAGS_TABLE, None) is True:
        render_download_interface(user_session_id, session)
    else:
        render_data_loading_interface(user_session_id, session)


if __name__ == "__main__":
    main()
