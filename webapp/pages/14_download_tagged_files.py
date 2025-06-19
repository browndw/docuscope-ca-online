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

import streamlit as st

from webapp.utilities.session import (  # noqa: E402
    get_or_init_user_session
)
from webapp.utilities.exports import (  # noqa: E402
    handle_tagged_files_download
)
from webapp.utilities.ui import (  # noqa: E402
    render_download_page_header,
    render_data_loading_interface,
    render_tagset_selection
)
from webapp.config.session_keys import (  # noqa: E402
    SessionKeys
)
from webapp.menu import (  # noqa: E402
    menu,
    require_login
)

TITLE = "Download Tagged Files"
ICON = ":material/download:"

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )

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
    if session.get(SessionKeys.TAGS_TABLE)[0] is True:
        render_tagged_files_interface(user_session_id, session)
    else:
        render_data_loading_interface(user_session_id, session)

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
    if session.get(SessionKeys.HAS_TARGET)[0] is True:
        handle_tagged_files_download(user_session_id, tagset)

if __name__ == "__main__":
    main()
