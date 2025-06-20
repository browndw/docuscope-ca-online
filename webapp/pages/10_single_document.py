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

import polars as pl
import streamlit as st

from webapp.utilities.session import (
    get_or_init_user_session
)
from webapp.utilities.ui import (
    tagset_selection,
    render_document_interface,
    render_document_selection_interface
)
from webapp.utilities.state import (
    load_widget_state, persist
)
from webapp.menu import (   # noqa: E402
    menu, require_login
)
from webapp.config.session_keys import (
    TargetKeys
)


TITLE = "Single Documents"
ICON = ":material/find_in_page:"

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )


def main() -> None:
    """
    Main function to render the Single Document page.
    This function sets up the page, handles user sessions,
    and manages the selection and display of individual documents
    with their associated tags and statistics.
    """
    # Set login requirements for navigaton
    require_login()
    menu()
    st.markdown(
        body=f"## {TITLE}",
        help=(
            "This page allows you to explore individual documents "
            "from your target corpus. You can select tags to highlight "
            "in the text, visualize their distribution, and download "
            "the results in a Word document."
            )
        )
    # Get or initialize user session
    user_session_id, session = get_or_init_user_session()

    st.sidebar.link_button(
        label="Help",
        url="https://browndw.github.io/docuscope-docs/guide/single-document.html",
        icon=":material/help:"
        )

    # Route to appropriate interface based on whether document is loaded
    if session.get('doc')[0] is True:
        load_widget_state(user_session_id)

        st.sidebar.markdown(
            body="### Tagset"
            )

        # Use the reusable tagset selection function
        tag_loc, tag_options, tag_radio, tag_type = tagset_selection(
            user_session_id=user_session_id,
            session_state=st.session_state,
            persist_func=persist,
            tagset_keys={
                "Parts-of-Speech": {
                    "General": TargetKeys.DOC_SIMPLE,
                    "Specific": TargetKeys.DOC_POS
                    },
                "DocuScope": TargetKeys.DOC_DS
                },
            tag_filters={
                "Parts-of-Speech": {
                    "Specific": lambda df: df.filter(pl.col("Tag") != "Y"),
                    "General": lambda df: df.filter(pl.col("Tag") != "Other")
                },
                "DocuScope": lambda df: df.filter(pl.col("Tag") != "Untagged")
                },
            tag_radio_key="sd_radio",
            tag_type_key="sd_tag_type"
        )

        # Get document key
        if tag_loc is not None:
            doc_key = tag_loc.get_column("doc_id").unique().to_list()
        else:
            doc_key = []

        # Render the document interface using the modular function
        render_document_interface(user_session_id, tag_loc, tag_options, doc_key)

    else:
        # Render document selection interface using the modular function
        render_document_selection_interface(user_session_id, session)

    st.sidebar.markdown("---")


if __name__ == "__main__":
    main()
