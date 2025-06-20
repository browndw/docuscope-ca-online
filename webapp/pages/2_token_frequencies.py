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

import docuscospacy as ds
import streamlit as st

from webapp.utilities.session import (  # noqa: E402
    get_or_init_user_session,
    load_metadata,
    validate_session_state
    )
from webapp.utilities.analysis import (  # noqa: E402
    generate_frequency_table
    )
from webapp.utilities.ui import (   # noqa: E402
    get_page_base_filename,
    render_data_table_interface,
    render_table_generation_interface,
    sidebar_help_link,
    tagset_selection,
    )
from webapp.utilities.state import (   # noqa: E402
    load_widget_state,
    persist
    )
from webapp.config.session_keys import (  # noqa: E402
    CorpusKeys,
    SessionKeys,
    TargetKeys,
    WarningKeys
    )
from webapp.menu import (   # noqa: E402
    menu,
    require_login
    )

TITLE = "Token Frequencies"
ICON = ":material/table_view:"

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


def render_frequency_table_interface(user_session_id: str, session: dict) -> None:
    """Render the frequency table interface with error handling."""
    try:
        # Validate session state first
        if not validate_session_state(user_session_id):
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

        load_widget_state(user_session_id)

        metadata_target = load_metadata(CorpusKeys.TARGET, user_session_id)
        if not metadata_target:
            st.warning(
                "Could not load target corpus metadata. Please reload your corpus.",
                icon=":material/warning:"
            )
            st.markdown("Go to **Load Corpus** page to reload your data.")
            return

        # Load the tags table for the target
        df, tag_options, tag_radio, tag_type = tagset_selection(
            user_session_id=user_session_id,
            session_state=st.session_state,
            persist_func=persist,
            tagset_keys=TAGSET_CONFIG,
            simplify_funcs=SIMPLIFY_CONFIG,
            tag_filters={
                # Add filters here to exclude tags for specific tagsets/subtypes
            },
            tag_radio_key="ft_radio",
            tag_type_key="ft_type_radio"
        )

        # Use generalized data table interface (filtering applied inside)
        base_filename = get_page_base_filename(__file__)
        render_data_table_interface(
            df=df,
            metadata_target=metadata_target,
            base_filename=base_filename,
            no_data_message="No frequency data available to display.",
            apply_tag_filter=True
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
    if session.get(SessionKeys.FREQ_TABLE, [False])[0]:
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
