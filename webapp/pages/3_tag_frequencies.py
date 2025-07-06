"""
This app provides an interface for generating and viewing tag frequency tables
for a loaded target corpus.

Users can:
- Select different tagsets (Parts-of-Speech, DocuScope)
- Filter tags displayed in the table
- View tag frequencies in a bar plot
"""

import docuscospacy as ds
import polars as pl
import streamlit as st

# Core application utilities with standardized patterns
from webapp.utilities.core import app_core

from webapp.utilities.analysis import (
    generate_tags_table, load_metadata
)
from webapp.utilities.session import (
    get_or_init_user_session, safe_session_get
    )
from webapp.utilities.corpus import (
    get_corpus_data_manager
    )
from webapp.utilities.ui import (
    render_table_generation_interface, sidebar_help_link,
    tagset_selection, color_picker_controls,
    tag_filter_multiselect, target_info,
    render_data_table_interface
    )
from webapp.utilities.ui.error_boundaries import SafeComponentRenderer
from webapp.utilities.plotting import (
    plot_download_link, plot_tag_frequencies_bar
    )
from webapp.utilities.state import (
    CorpusKeys, SessionKeys,
    TargetKeys, WarningKeys
)
from webapp.utilities.state.widget_key_manager import create_persist_function
from webapp.menu import (
    menu, require_login
    )

TITLE = "Tag Frequencies"
ICON = ":material/table_view:"

# Register persistent widgets for this page
TAG_FREQUENCIES_PERSISTENT_WIDGETS = [
    "tt_radio",       # Radio button for tagset selection
    "tt_type_radio",  # Radio button for tag type selection
]
app_core.register_page_widgets(TAG_FREQUENCIES_PERSISTENT_WIDGETS)

# Configuration constants
TAGSET_CONFIG = {
    "Parts-of-Speech": {
        "General": TargetKeys.DTM_POS,
        "Specific": TargetKeys.TT_POS
    },
    "DocuScope": TargetKeys.TT_DS
}
SIMPLIFY_CONFIG = {
    "Parts-of-Speech": {
        "General": ds.tags_simplify,
        "Specific": None
    }
}
TAG_FILTERS_CONFIG = {
    "Parts-of-Speech": {
        "Specific": lambda df: df.filter(pl.col("Tag") != "FU")
    },
    "DocuScope": lambda df: df.filter(pl.col("Tag") != "Untagged")
}

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )


def render_tag_frequency_interface(
        user_session_id: str, session: dict
) -> None:
    """Render the tag frequency interface with tabs for table and plot."""
    try:
        # Validate corpus data using the new manager
        manager = get_corpus_data_manager(user_session_id, CorpusKeys.TARGET)
        if not manager.is_ready():
            st.error("Invalid session state. Please reload the page or reset your data.")
            return

        # Initialize widget state management
        app_core.widget_manager.register_persistent_keys([
            'tag_freq_sort', 'tag_freq_ascending', 'tag_freq_display_limit',
            'tag_freq_filter_zero'
        ])

        metadata_target = load_metadata(CorpusKeys.TARGET, user_session_id)

        if not metadata_target:
            st.error("Could not load target corpus metadata.")
            return

        # Generate the tags table using the new system
        df, tag_options, tag_radio, tag_type = tagset_selection(
            user_session_id=user_session_id,
            session_state=st.session_state,
            persist_func=create_persist_function(user_session_id),
            tagset_keys=TAGSET_CONFIG,
            simplify_funcs=SIMPLIFY_CONFIG,
            tag_filters=TAG_FILTERS_CONFIG,
            tag_radio_key="tt_radio",
            tag_type_key="tt_type_radio"
        )

        # Create tabs for table and plot display
        tab1, tab2 = st.tabs([":material/table_view: Table", ":material/bar_chart: Plot"])

        # Render the table in the first tab with custom key handling
        with tab1:
            # Use generalized data table interface (filtering applied inside)
            render_data_table_interface(
                df=df,
                metadata_target=metadata_target,
                base_filename="tag_frequencies",
                no_data_message="No frequency data available to display.",
                apply_tag_filter=True,
                user_session_id=user_session_id
            )

        # Plot the tag frequencies in the second tab
        with tab2:
            render_tag_frequency_plot(df, metadata_target)

    except Exception as e:
        st.error(f"Error loading tag frequency table: {str(e)}", icon=":material/error:")
        st.info("Try regenerating the tag frequency table if this error persists.")


def render_tag_frequency_plot(
        df, metadata_target: dict
) -> None:
    """Render the tag frequency plot with color controls."""
    # Display the target information
    st.info(target_info(metadata_target))

    # Apply tag filtering with unique key for plot tab
    filtered_df = tag_filter_multiselect(df, key="plot_tag_filter")

    if filtered_df is None or getattr(filtered_df, "height", 0) == 0:
        st.warning("No tags to plot.", icon=":material/info:")
        return

    # Color picker for bar color
    color_dict = color_picker_controls(
        ["Bar Color"],
        key_prefix="tag_freq_bar_"
    )
    bar_color = color_dict.get("Bar Color", "#133955")

    # Plot the tag frequencies bar chart
    fig = plot_tag_frequencies_bar(filtered_df, color=bar_color)
    SafeComponentRenderer.safe_plotly_chart(fig, use_container_width=True)
    plot_download_link(fig, filename="tag_frequency_plot.png")


def main():
    """
    Main function to run the Streamlit app for tag frequencies.
    It initializes the user session, loads the necessary data,
    and displays the tag frequencies in a table and plot.
    """
    # Set login requirements for navigation
    require_login()
    menu()
    st.markdown(
        body=f"## {TITLE}",
        help=(
            "This app allows you to generate and view tag frequency tables "
            "for the loaded target corpus. You can toggle between different "
            "tagsets and filter the tags displayed in the table."
            )
        )
    # Get or initialize user session
    user_session_id, session = get_or_init_user_session()

    sidebar_help_link("tag-frequencies.html")

    # Check if the tags table is available in the session
    if safe_session_get(session, SessionKeys.TAGS_TABLE, False) is True:
        render_tag_frequency_interface(user_session_id, session)
    else:
        render_table_generation_interface(
            user_session_id=user_session_id,
            session=session,
            table_type="tags table",
            button_label="Tags Table",
            generation_func=generate_tags_table,
            session_key=SessionKeys.TAGS_TABLE,
            warning_key=WarningKeys.TAGS
        )

    st.sidebar.markdown("---")


if __name__ == "__main__":
    main()
