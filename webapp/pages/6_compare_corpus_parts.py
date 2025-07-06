"""
This app allows users to compare different parts of their corpus
by generating a keyness table based on selected categories.

Users can:
- Generate keyness tables for target and reference using selected categories
- Filter results by tags
- Toggle between different tagsets (Parts-of-Speech, DocuScope)
- Download the keyness table in Excel format
"""

import pandas as pd
import polars as pl
import streamlit as st

# Core application utilities with standardized patterns
from webapp.utilities.core import app_core

from webapp.utilities.session import (
    get_or_init_user_session, load_metadata,
    safe_session_get
    )
from webapp.utilities.corpus import (
    get_corpus_data_manager, clear_corpus_data
)
from webapp.utilities.ui import (
    keyness_settings_info, keyness_sort_controls,
    reference_parts, render_dataframe,
    sidebar_action_button, sidebar_help_link,
    sidebar_keyness_options, tag_filter_multiselect,
    tagset_selection, target_parts,
    toggle_download, color_picker_controls
)
from webapp.utilities.plotting import (
    update_ref, update_tar,
    plot_compare_corpus_bar, plot_download_link
)
from webapp.utilities.ui.error_boundaries import SafeComponentRenderer
from webapp.utilities.exports import (
    convert_to_excel
)
from webapp.utilities.state import (
    safe_clear_widget_state,
    SessionKeys, CorpusKeys,
    TargetKeys, WarningKeys
)
from webapp.utilities.state.widget_key_manager import create_persist_function
from webapp.utilities.analysis import (
    has_target_corpus, render_corpus_not_loaded_error,
    freq_simplify_pl, generate_keyness_parts
)
from webapp.menu import (   # noqa: E402
    menu,
    require_login
    )


TITLE = "Compare Corpus Parts"
ICON = ":material/compare_arrows:"

# Register persistent widgets for this page
COMPARE_CORPUS_PARTS_PERSISTENT_WIDGETS = [
    "cp_radio1",  # Radio button for tagset selection
    "cp_radio3",  # Radio button for keyness table type
    "tar",        # Target categories segmented control
    "ref",        # Reference categories segmented control
]
app_core.register_page_widgets(COMPARE_CORPUS_PARTS_PERSISTENT_WIDGETS)

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )


def render_results_interface(user_session_id: str, session: dict) -> None:
    """Render the interface when keyness table has been generated."""
    # Initialize widget state management
    app_core.widget_manager.register_persistent_keys([
        'corpus_parts_sort', 'corpus_parts_ascending', 'corpus_parts_display_limit',
        'corpus_parts_metric', 'corpus_parts_filter_zero'
    ])

    # Use the new corpus data manager
    target_manager = get_corpus_data_manager(user_session_id, CorpusKeys.TARGET)
    metadata_target = load_metadata(CorpusKeys.TARGET, user_session_id)

    # Display target and reference parts information
    col1, col2 = st.columns([1, 1])
    with col1:
        st.info(target_parts(metadata_target.get(SessionKeys.KEYNESS_PARTS)[0]['temp']))
    with col2:
        st.info(reference_parts(metadata_target.get(SessionKeys.KEYNESS_PARTS)[0]['temp']))

    # Show user selections
    st.info(keyness_settings_info(user_session_id))

    # Table type selection in sidebar
    st.sidebar.markdown("### Comparison")
    table_radio = st.sidebar.radio(
        "Select the keyness table to display:",
        ("Tokens", "Tags Only"),
        key="cp_radio1",
        horizontal=True,
        help="Choose between tokens with tags or tags-only analysis."
    )

    st.sidebar.markdown("---")

    if table_radio == 'Tokens':
        render_tokens_interface(user_session_id, target_manager)
    else:
        render_tags_interface(user_session_id, target_manager)


def render_tokens_interface(user_session_id: str, target_manager) -> None:
    """Render the tokens analysis interface."""
    # Use the tagset selection utility for sidebar controls
    df, _, tag_radio, tag_type = tagset_selection(
        user_session_id=user_session_id,
        session_state=st.session_state,
        persist_func=create_persist_function(user_session_id),
        tagset_keys={
            "Parts-of-Speech": {
                "General": TargetKeys.KW_POS_CP,
                "Specific": TargetKeys.KW_POS_CP
            },
            "DocuScope": TargetKeys.KW_DS_CP
        },
        simplify_funcs={
            "Parts-of-Speech": {"General": freq_simplify_pl, "Specific": None}
        },
        tag_radio_key="cp_radio2",
        tag_type_key="cp_tag_type"
    )

    # Sorting and filtering controls
    sort_by, reverse = keyness_sort_controls(
        sort_options=["Keyness (LL)", "Effect Size (LR)"],
        default="Keyness (LL)",
        reverse_default=True,
        key_prefix="kt_"
    )

    df = tag_filter_multiselect(df, user_session_id=user_session_id)

    # Map UI label to actual DataFrame column
    sort_col_map = {
        "Keyness (LL)": "LL",
        "Effect Size (LR)": "LR"
    }
    sort_col = sort_col_map[sort_by]

    if df is not None and getattr(df, "height", 0) > 0:
        df = df.sort(sort_col, descending=reverse)

    render_dataframe(df)

    # Sidebar controls
    st.sidebar.markdown("---")
    render_sidebar_controls(df, user_session_id)


def render_tags_interface(
        user_session_id: str,
        target_manager
) -> None:
    """Render the tags-only analysis interface."""
    # Use sidebar tagset selection
    st.sidebar.markdown("### Tagset")
    tag_radio_tags = st.sidebar.radio(
        "Select tags to display:",
        ("Parts-of-Speech", "DocuScope"),
        key="cp_radio3",
        horizontal=True,
        help="Choose the tagset for tag frequency analysis."
    )

    if tag_radio_tags == 'Parts-of-Speech':
        df = target_manager.get_data(TargetKeys.KT_POS_CP)
        if df is not None:
            df = df.filter(pl.col("Tag") != "FU")
    else:
        df = target_manager.get_data(TargetKeys.KT_DS_CP)
        if df is not None:
            df = df.filter(pl.col("Tag") != "Untagged")

    tab1, tab2 = st.tabs(["Keyness Table", "Keyness Plot"])
    with tab1:
        # Sorting and filtering controls
        sort_by, reverse = keyness_sort_controls(
            sort_options=["Keyness (LL)", "Effect Size (LR)"],
            default="Keyness (LL)",
            reverse_default=True,
            key_prefix="kt_"
        )

        df = tag_filter_multiselect(df, user_session_id=user_session_id)

        # Map UI label to actual DataFrame column
        sort_col_map = {
            "Keyness (LL)": "LL",
            "Effect Size (LR)": "LR"
        }
        sort_col = sort_col_map[sort_by]

        if df is not None and getattr(df, "height", 0) > 0:
            df = df.sort(sort_col, descending=reverse)

        render_dataframe(df)

    with tab2:
        if df.height > 0:
            # Color picker for bar colors
            color_dict = color_picker_controls(
                ["Target Color", "Reference Corpus"],
                key_prefix="compare_corpus_parts_bar_",
                default_hex="#133955"
            )

            # Plot with color customization
            fig = plot_compare_corpus_bar(df, color_dict=color_dict)
            SafeComponentRenderer.safe_plotly_chart(fig, use_container_width=True)
            plot_download_link(fig, filename="compare_corpus_parts_bar.png")
        else:
            st.info("No data available for plotting. Please adjust your filters.")

    # Sidebar controls
    st.sidebar.markdown("---")
    render_sidebar_controls(df, user_session_id)


def create_enhanced_dataframe_for_export(
        df: pl.DataFrame,
        metadata_target
) -> pd.DataFrame:
    """Create a DataFrame with context information for Excel export."""
    if df is None or df.height == 0:
        return None

    # Get the context information
    keyness_parts_data = metadata_target.get(SessionKeys.KEYNESS_PARTS)[0]['temp']
    target_info = target_parts(keyness_parts_data)
    reference_info = reference_parts(keyness_parts_data)

    # Convert to pandas
    pandas_df = df.to_pandas()

    # Create a full-width context section
    context_data = {}

    # Get all column names (original + context columns)
    all_columns = list(pandas_df.columns) + ['Context_Details']

    # Initialize all columns
    for col in all_columns:
        if col in pandas_df.columns:
            # Original data + empty rows for context
            context_data[col] = pandas_df[col].tolist() + ['', '', '']
        elif col == 'Context_Details':
            # Empty for data rows + context information
            context_data[col] = [target_info, reference_info, ''] + [''] * len(pandas_df)

    enhanced_df = pd.DataFrame(context_data)

    return enhanced_df


def render_sidebar_controls(
        df,
        user_session_id: str
) -> None:
    """Render common sidebar controls for downloads and table regeneration."""
    # Get metadata needed for enhanced export
    metadata_target = load_metadata(CorpusKeys.TARGET, user_session_id)

    toggle_download(
        label="Excel",
        convert_func=convert_to_excel,
        convert_args=(
            (create_enhanced_dataframe_for_export(df, metadata_target),)
            if (df is not None and getattr(df, "height", 0) > 0)
            else (None,)
        ),
        file_name="keyness_corpus_parts",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        location=st.sidebar
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        body=(
            "### Generate new table\n\n"
            "Use the button to reset the keyness table and start over."
        )
    )

    if st.sidebar.button(
        label="Compare New Categories",
        icon=":material/refresh:"
    ):
        # Clear only corpus parts comparison data using the corpus data manager
        clear_corpus_data(user_session_id, CorpusKeys.TARGET, [
            TargetKeys.KW_POS_CP, TargetKeys.KW_DS_CP,
            TargetKeys.KT_POS_CP, TargetKeys.KT_DS_CP
        ])

        # Reset corpus parts state using session key
        app_core.session_manager.update_session_state(
            user_session_id, SessionKeys.KEYNESS_PARTS, False
        )

        # Clear warnings using session key
        st.session_state[user_session_id][WarningKeys.KEYNESS_PARTS] = None
        st.rerun()

    st.sidebar.markdown("---")


def render_setup_interface(
        user_session_id: str,
        session: dict
) -> None:
    """Render the interface for setting up corpus part comparison."""
    st.markdown(
        body=(
            ":material/manufacturing: Use the configuration below to "
            "**generate keywords** from subcorpora.\n\n"
            ":material/priority: To use this tool, you must first process "
            "**metadata** from **Manage Corpus Data**.\n\n"
            ":material/priority: After the table has been generated, "
            "you will be able to **toggle between the tagsets**."
        )
    )

    if safe_session_get(session, SessionKeys.HAS_META, False):
        render_category_selection_interface(user_session_id)
    else:
        render_no_metadata_message()

    # Keyness options
    st.sidebar.markdown("---")
    pval_selected, swap_selected = sidebar_keyness_options(
        user_session_id,
        load_metadata_func=load_metadata,
        require_reference=False
    )

    # Generate button
    render_generation_controls(user_session_id, session, pval_selected, swap_selected)


def render_category_selection_interface(
        user_session_id: str
) -> None:
    """Render the category selection interface using segmented controls."""
    metadata_target = load_metadata(CorpusKeys.TARGET, user_session_id)
    all_cats = sorted(set(metadata_target.get('doccats')[0]['cats']))

    with st.expander("Category Selection", expanded=True):
        st.markdown(
            body="### Select categories to compare",
            help=(
                "Categories can be generated from file names in the target corpus.\n\n"
                "For example, if your file names are formatted like `BIO_G0_02_1.txt`, "
                "`ENG_G0_16_1.txt`, etc., you can extract the categories `BIO` and `ENG`. "
                "These categories can then be selected for comparison "
                "in the keyness table.\n\n"
                "You can select multiple categories for both target and reference corpora, "
                "but you cannot select the same category for both target and reference."
            )
        )

        st.markdown(
            "Select the categories you want to compare. The target categories will be "
            "compared against the reference categories in the keyness analysis."
        )

        # Target categories
        st.markdown("#### Target corpus categories:")
        st.session_state[user_session_id]['tar'] = st.segmented_control(
            "Select target categories:",
            all_cats,
            selection_mode="multi",
            key="tar",
            on_change=update_tar,
            args=(user_session_id,),
            help=(
                "These categories will be treated as the target group "
                "in the keyness analysis."
            )
        )

        # Reference categories
        st.markdown("#### Reference corpus categories:")
        st.session_state[user_session_id]['ref'] = st.segmented_control(
            "Select reference categories:",
            all_cats,
            selection_mode="multi",
            key="ref",
            on_change=update_ref,
            args=(user_session_id,),
            help=(
                "These categories will be treated as the reference group "
                "in the keyness analysis."
            )
        )

        # Validation message
        tar_selected = st.session_state[user_session_id].get('tar', [])
        ref_selected = st.session_state[user_session_id].get('ref', [])

        if tar_selected and ref_selected:
            overlap = set(tar_selected) & set(ref_selected)
            if overlap:
                st.error(
                    f"Categories cannot be in both target and reference groups: "
                    f"{', '.join(overlap)}",
                    icon=":material/error:"
                )
            else:
                st.success(
                    f"Ready to compare {len(tar_selected)} target vs "
                    f"{len(ref_selected)} reference categories.",
                    icon=":material/check_circle:"
                )
        elif tar_selected or ref_selected:
            st.warning(
                "Please select categories for both target and reference groups.",
                icon=":material/warning:"
            )


def render_no_metadata_message() -> None:
    """Render message when no metadata is available."""
    st.warning(
        "No metadata has been processed yet. Please use the **Manage Corpus Data** "
        "page to process metadata first.",
        icon=":material/new_label:"
    )


def render_generation_controls(
    user_session_id: str,
    session: dict,
    pval_selected: float,
    swap_selected: bool
) -> None:
    """Render the table generation controls."""
    st.sidebar.markdown(
        body=(
            "### Generate table\n\n"
            "Use the button to process a table."
        ),
        help=(
            "Tables are generated based on the target and reference corpora. "
            "You can filter the table after it has been generated. "
            "The table will include frequencies and hypothesis testing "
            "for the selected tagsets.\n\n"
            "Click on the **Help** button for more information on how to use this app."
        )
    )

    # Check if we have both target and reference categories selected
    tar_selected = st.session_state[user_session_id].get('tar', [])
    ref_selected = st.session_state[user_session_id].get('ref', [])

    # Create a custom action that handles validation and shows appropriate errors
    def keyness_action():
        if not has_target_corpus(session):
            render_corpus_not_loaded_error()
            return

        # Check category selection
        if not tar_selected:
            st.warning(
                body=(
                    "Please select at least one **target category** from the "
                    "category selection interface above."
                ),
                icon=":material/warning:"
            )
            return

        if not ref_selected:
            st.error(
                body=(
                    "Please select at least one **reference category** from the "
                    "category selection interface above. Reference categories "
                    "represent the comparison group."
                ),
                icon=":material/category:"
            )
            return

        # Check for overlap
        overlap = set(tar_selected) & set(ref_selected)
        if overlap:
            st.error(
                body=(
                    f"Categories cannot be in both target and reference groups. "
                    f"Please remove these overlapping categories: {', '.join(overlap)}"
                ),
                icon=":material/error:"
            )
            return

        # If all validation passes, generate the keyness table
        generate_keyness_parts(
            user_session_id,
            threshold=pval_selected,
            swap_target=swap_selected
        )

    sidebar_action_button(
        button_label="Keyness Table of Corpus Parts",
        button_icon=":material/manufacturing:",
        preconditions=[True],  # Always allow button click, handle validation in action
        action=keyness_action,
        spinner_message="Generating keywords..."
    )

    # Display any warnings
    if st.session_state[user_session_id].get(WarningKeys.KEYNESS_PARTS):
        msg, icon = st.session_state[user_session_id][WarningKeys.KEYNESS_PARTS]
        st.error(msg, icon=icon)
        # Clear the warning after displaying it
        safe_clear_widget_state(f"{user_session_id}_{WarningKeys.KEYNESS_PARTS}")

    st.sidebar.markdown("---")


def main():
    """
    Main function to run the Streamlit app for comparing corpus parts.
    This function sets up the page configuration, checks user login status,
    initializes the user session, and renders the UI components for
    comparing corpus parts based on selected categories.
    It allows users to generate keyness tables, filter results by tags,
    and download the data in Excel format.
    """
    # Set login requirements for navigation
    require_login()
    menu()
    st.markdown(
        body=f"## {TITLE}",
        help=(
            "This page allows you to compare different parts of your corpus "
            "by generating a keyness table based on selected categories. "
            "You can filter the results by tags and download the data in Excel format."
            )
        )

    # Get or initialize user session
    user_session_id, session = get_or_init_user_session()

    sidebar_help_link("compare-corpus-parts.html")

    # Check if keyness table has been generated
    if safe_session_get(session, SessionKeys.KEYNESS_PARTS, False):
        render_results_interface(user_session_id, session)
    else:
        render_setup_interface(user_session_id, session)


if __name__ == "__main__":
    main()
