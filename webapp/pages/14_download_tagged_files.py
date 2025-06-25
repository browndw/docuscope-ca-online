"""
This app provides an interface for downloading tagged text files
from a loaded target corpus.
"""

import streamlit as st

from webapp.utilities.session import (
    get_or_init_user_session, safe_session_get
    )
from webapp.utilities.exports import (
    handle_tagged_files_download
)
from webapp.utilities.ui import (
    render_download_page_header, render_data_loading_interface,
    render_tagset_selection
)
from webapp.utilities.state import (
    SessionKeys
)
from webapp.menu import (
    menu, require_login
)


TITLE = "Download Tagged Files"
ICON = ":material/download:"

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )


def render_tagged_files_interface(user_session_id: str, session: dict) -> None:
    """
    Render the tagged files download interface when tables are loaded.

    Parameters
    ----------
    user_session_id : str
        The user session identifier
    session : dict
        The session state dictionary
    """
    # Get tagset selection
    tagset = render_tagset_selection()

    # Check if target corpus is available and handle download
    if safe_session_get(session, SessionKeys.HAS_TARGET, None) is True:
        handle_tagged_files_download(user_session_id, tagset)


def main() -> None:
    """
    Main function to render the download tagged files page.
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
        )
    )

    # Get or initialize user session
    user_session_id, session = get_or_init_user_session()

    # Display processing message
    st.markdown(
        """
        ##### :material/manufacturing: \
        Generate a zipped folder of tagged text files.
        :material/help:
        Use the Help link in the sidebar
        to learn more about how the embedded tags are formatted.
        """
    )

    # Check if tables are loaded
    if safe_session_get(session, SessionKeys.TAGS_TABLE, None) is True:
        render_tagged_files_interface(user_session_id, session)
    else:
        render_data_loading_interface(user_session_id, session)


if __name__ == "__main__":
    main()
