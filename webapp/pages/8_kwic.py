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
    get_or_init_user_session,
    load_metadata,
    update_session
    )
# Temporary imports for unmigrated UI functions
from webapp.utilities.ui import (  # noqa: E402
    render_data_table_interface,
    sidebar_action_button,
    sidebar_help_link
)
from webapp.utilities.analysis import (  # noqa: E402
    has_target_corpus,
    render_corpus_not_loaded_error,
    generate_kwic
)
from webapp.config.session_keys import (  # noqa: E402
    SessionKeys,
    CorpusKeys,
    TargetKeys,
    WarningKeys
    )
from webapp.menu import (   # noqa: E402
    menu,
    require_login
    )

TITLE = "KWIC Tables"
ICON = ":material/network_node:"

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )

def render_results_interface(user_session_id: str, session: dict) -> None:
    """Render the interface when KWIC table has been generated."""
    metadata_target = load_metadata(CorpusKeys.TARGET, user_session_id)

    # Get KWIC data with defensive access
    try:
        df = st.session_state[user_session_id][CorpusKeys.TARGET][TargetKeys.KWIC]
    except (KeyError, AttributeError):
        # Fallback to direct key access
        df = st.session_state[user_session_id][CorpusKeys.TARGET].get("kwic")
        if df is None:
            st.error("KWIC data not found. Please regenerate the analysis.")
            return

    # Render the data table interface
    render_data_table_interface(
        df=df,
        metadata_target=metadata_target,
        base_filename="kwic",
        no_data_message="No KWIC data available to display.",
        apply_tag_filter=False  # KWIC tables typically don't need tag filtering
    )

    # Reset table option in sidebar
    st.sidebar.markdown("---")
    st.sidebar.info(
        body=(
            "### Generate new table\n\n"
            "Use the button to reset the KWIC table and start over."
        )
    )

    if st.sidebar.button(
        label="Create New KWIC Table",
        icon=":material/refresh:",
        help="Reset the analysis and configure new KWIC parameters."
    ):
        # Clear existing data
        target_dict = st.session_state[user_session_id][CorpusKeys.TARGET]
        try:
            if TargetKeys.KWIC in target_dict:
                target_dict[TargetKeys.KWIC] = {}
        except AttributeError:
            # Fallback for attribute error
            if "kwic" in target_dict:
                target_dict["kwic"] = {}
        update_session(SessionKeys.KWIC, False, user_session_id)
        st.rerun()

    st.sidebar.markdown("---")

def render_setup_interface(user_session_id: str, session: dict) -> None:
    """Render the interface for setting up KWIC parameters."""
    st.markdown(
        body=(
            ":material/manufacturing: Use the button in the sidebar to "
            "**generate a KWIC table**.\n\n"
            ":material/priority: A **target corpus** must be loaded first."
        )
    )

    # Configuration in main area expander
    with st.expander("🔧 **KWIC Configuration**", expanded=True):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Node word")
            st.markdown("Enter a node word without spaces.")
            node_word = st.text_input(
                "Node word",
                help="The word to search for in the KWIC table."
            )

            st.markdown("### Case sensitivity")
            case_sensitive = st.checkbox("Make search case sensitive")
            ignore_case = not bool(case_sensitive)

        with col2:
            st.markdown("### Search mode")
            search_mode = st.radio(
                "Select search type:",
                ("Fixed", "Starts with", "Ends with", "Contains"),
                horizontal=False,
                help=(
                    "- **Fixed**: Exact match\n"
                    "- **Starts with**: Words beginning with the search term\n"
                    "- **Ends with**: Words ending with the search term\n"
                    "- **Contains**: Words containing the search term"
                )
            )

            # Convert search mode to internal format
            search_type_mapping = {
                "Fixed": "fixed",
                "Starts with": "starts_with",
                "Ends with": "ends_with",
                "Contains": "contains"
            }
            search_type = search_type_mapping[search_mode]

    # Generation controls (button stays in sidebar)
    render_generation_controls(
        user_session_id=user_session_id,
        session=session,
        node_word=node_word,
        search_type=search_type,
        ignore_case=ignore_case
    )

def render_generation_controls(
    user_session_id: str,
    session: dict,
    node_word: str,
    search_type: str,
    ignore_case: bool
) -> None:
    """Render generation controls and handle KWIC generation."""

    def kwic_action():
        if not has_target_corpus(session):
            render_corpus_not_loaded_error()
            return

        # Check if node word is provided
        if not node_word or node_word.strip() == "":
            st.error(
                body=(
                    "Please enter a **node word** in the configuration above. "
                    "The node word is the central word to search for."
                ),
                icon=":material/edit:"
            )
            return

        # If all validation passes, generate the KWIC table
        generate_kwic(user_session_id, node_word, search_type, ignore_case)

    sidebar_action_button(
        button_label="KWIC",
        button_icon=":material/manufacturing:",
        preconditions=[True],  # Always allow button click, handle validation in action
        action=kwic_action,
        spinner_message="Processing KWIC..."
    )

    # Display warnings if any
    if st.session_state[user_session_id].get(WarningKeys.KWIC):
        msg, icon = st.session_state[user_session_id][WarningKeys.KWIC]
        st.warning(msg, icon=icon)

def main():
    # Set login requirements for navigation
    require_login()
    menu()

    st.markdown(
        body=f"## {TITLE}",
        help=(
            "Generate a KWIC (Key Word in Context) table for a node word "
            "in the target corpus. The KWIC table displays occurrences of "
            "the node word along with its surrounding context."
        )
    )

    # Get or initialize user session
    user_session_id, session = get_or_init_user_session()

    sidebar_help_link("kwic.html")

    # Check if KWIC table has been generated
    if session.get(SessionKeys.KWIC, [False])[0]:
        render_results_interface(user_session_id, session)
    else:
        render_setup_interface(user_session_id, session)

if __name__ == "__main__":
    main()
