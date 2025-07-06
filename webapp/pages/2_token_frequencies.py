"""
App for generating and displaying token frequency tables
for the loaded target corpus.

This app allows users to:
- Generate frequency tables for tokens in the target corpus
- Filter tokens by tagset and tag type
- Download frequency tables in Excel format
- Handle errors and session state validation
- Provide a user-friendly interface for corpus analysis
"""

import docuscospacy as ds
import streamlit as st

# Core application utilities with standardized patterns
from webapp.utilities.core import app_core
from webapp.utilities.state.widget_key_manager import create_persist_function

from webapp.utilities.session import (
    get_or_init_user_session, load_metadata, safe_session_get
    )
from webapp.utilities.analysis import (
    generate_frequency_table
    )
from webapp.utilities.ui import (
    render_data_table_interface, render_table_generation_interface,
    sidebar_help_link, tagset_selection,
    )
from webapp.utilities.state import (
    CorpusKeys, SessionKeys,
    TargetKeys, WarningKeys
)
from webapp.utilities.corpus import (
    get_corpus_data_manager
    )
from webapp.menu import (
    menu, require_login
    )

TITLE = "Token Frequencies"
ICON = ":material/table_view:"

# Register persistent widgets for this page
TOKEN_FREQUENCIES_PERSISTENT_WIDGETS = [
    "ft_radio",       # Radio button for frequency table type selection
    "ft_type_radio",  # Radio button for tag type selection
]
app_core.register_page_widgets(TOKEN_FREQUENCIES_PERSISTENT_WIDGETS)

# Configuration constants
TAGSET_CONFIG = {
    "Parts-of-Speech": {
        "General": TargetKeys.FT_POS,
        "Specific": TargetKeys.FT_POS
    },
    "DocuScope": TargetKeys.FT_DS
}
SIMPLIFY_CONFIG = {
    "Parts-of-Speech": {
        "General": ds.freq_simplify,
        "Specific": None
    }
}

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )


def render_frequency_table_interface(
        user_session_id: str, session: dict
) -> None:
    """Render the frequency table interface with error handling."""
    try:
        # Validate corpus data using the new manager
        manager = get_corpus_data_manager(user_session_id, CorpusKeys.TARGET)
        if not manager.is_ready():
            st.warning(
                "No target corpus loaded. Please load a corpus first.",
                icon=":material/warning:"
            )
            st.markdown(
                body=("Go to **Load Corpus** page to load your data "
                      "before generating frequency tables."
                      )
                    )
            return

        # Initialize widget state management
        app_core.widget_manager.register_persistent_keys([
            'token_freq_sort', 'token_freq_ascending', 'token_freq_display_limit'
        ])

        metadata_target = load_metadata(CorpusKeys.TARGET, user_session_id)
        if not metadata_target:
            st.warning(
                "Could not load target corpus metadata. Please reload your corpus.",
                icon=":material/warning:"
            )
            st.markdown("Go to **Load Corpus** page to reload your data.")
            return

        # Load the tags table for the target using the new system
        df, tag_options, tag_radio, tag_type = tagset_selection(
            user_session_id=user_session_id,
            session_state=st.session_state,
            persist_func=create_persist_function(user_session_id),
            tagset_keys=TAGSET_CONFIG,
            simplify_funcs=SIMPLIFY_CONFIG,
            tag_filters={
                # Add filters here to exclude tags for specific tagsets/subtypes
            },
            tag_radio_key="ft_radio",
            tag_type_key="ft_type_radio"
        )

        # Use generalized data table interface (filtering applied inside)
        render_data_table_interface(
            df=df,
            metadata_target=metadata_target,
            base_filename="token_frequencies",
            no_data_message="No frequency data available to display.",
            apply_tag_filter=True,
            user_session_id=user_session_id
        )

    except Exception as e:
        st.error(f"Error loading frequency table: {str(e)}", icon=":material/error:")
        st.info("Try regenerating the frequency table if this error persists.")


def main() -> None:
    """
    Main function to run the Streamlit app for token frequencies.

    Displays token frequency tables for the loaded target corpus with:
    - Interactive tagset selection and filtering
    - Excel download functionality
    - Comprehensive error handling and validation
    """
    # Set login requirements for navigation
    require_login()
    menu()
    st.markdown(
        body=f"## {TITLE}",
        help=(
            "This app allows you to generate and view token frequency tables "
            "for the loaded target corpus. You can filter by tags and download "
            "the table in Excel format."
        )
    )

    # Get or initialize user session
    user_session_id, session = get_or_init_user_session()

    sidebar_help_link("token-frequencies.html")

    # Route to appropriate interface based on whether frequency table exists
    if safe_session_get(session, SessionKeys.FREQ_TABLE, False):
        render_frequency_table_interface(user_session_id, session)
    else:
        render_table_generation_interface(
            user_session_id=user_session_id,
            session=session,
            table_type="frequency table",
            button_label="Frequency Table",
            generation_func=generate_frequency_table,
            session_key=SessionKeys.FREQ_TABLE,
            warning_key=WarningKeys.FREQUENCY
        )

    st.sidebar.markdown("---")


if __name__ == "__main__":
    main()
