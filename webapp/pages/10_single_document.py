"""
This app provides an interface for exploring individual documents
from a target corpus. Users can select tags to highlight in the text,
visualize their distribution, and download the results in a Word document.
"""

import streamlit as st

from webapp.utilities.core import app_core
from webapp.utilities.session import (
    get_or_init_user_session
)
from webapp.utilities.ui import (
    tagset_selection, render_document_interface,
    render_document_selection_interface
)
from webapp.utilities.state import (
    TargetKeys, SessionKeys
)
from webapp.utilities.state.widget_key_manager import (
    create_persist_function
)
from webapp.menu import (
    menu, require_login
)
from webapp.utilities.session import (
    safe_session_get
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
    if safe_session_get(session, SessionKeys.DOC, None) is True:
        # Initialize widget state management
        app_core.widget_manager.register_persistent_keys([
            'doc_tagset_select', 'doc_tag_radio', 'doc_display_options'
        ])

        st.sidebar.markdown(
            body="### Tagset"
            )

        # Use the reusable tagset selection function
        tag_loc, tag_options, tag_radio, tag_type = tagset_selection(
            user_session_id=user_session_id,
            session_state=st.session_state,
            persist_func=create_persist_function(user_session_id),
            tagset_keys={
                "Parts-of-Speech": {
                    "General": TargetKeys.DOC_SIMPLE,
                    "Specific": TargetKeys.DOC_POS
                    },
                "DocuScope": TargetKeys.DOC_DS
                },
            tag_filters=None,
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
