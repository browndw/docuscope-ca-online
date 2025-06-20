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

import pathlib
import sys

import polars as pl
import streamlit as st

# Ensure project root is in sys.path for both desktop and online
project_root = pathlib.Path(__file__).parent.parents[1].resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from webapp.utilities.handlers import (  # noqa: E402
    generate_keyness_tables,
    get_or_init_user_session,
    load_metadata,
    update_session
    )
from webapp.utilities.ui import (   # noqa: E402
    keyness_sort_controls,
    load_widget_state,
    persist,
    reference_info,
    render_dataframe,
    sidebar_action_button,
    sidebar_help_link,
    sidebar_keyness_options,
    tag_filter_multiselect,
    tagset_selection,
    target_info,
    toggle_download
    )
from webapp.utilities.formatters import (  # noqa: E402
    convert_to_excel,
    plot_compare_corpus_bar,
    plot_download_link
    )
from webapp.utilities.analysis import (   # noqa: E402
    freq_simplify_pl
    )
from webapp.menu import (   # noqa: E402
    menu,
    require_login
    )

TITLE = "Compare Corpora"
ICON = ":material/compare_arrows:"

TOKEN_LIMIT = 1_500_000

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )


def main():
    """
    Main function to run the Streamlit app for comparing corpora.
    It initializes the user session, loads the necessary data,
    and provides the UI for generating and displaying keyness tables.
    """
    # Set login requirements for navigaton
    require_login()
    menu()
    st.markdown(
        body=f"## {TITLE}",
        help=(
            "This app allows you to generate and view keyness tables "
            "for the loaded target and reference corpora. You can filter by tags, "
            "select p-value thresholds, and download the results."
            )
        )
    # Get or initialize user session
    user_session_id, session = get_or_init_user_session()

    sidebar_help_link("compare-corpora.html")
    # --- Check if keyness table is already generated ---
    if session.get('keyness_table')[0] is True:
        # Load the widget state for the current user session
        load_widget_state(
            pathlib.Path(__file__).stem,
            user_session_id
            )
        # Load target and reference metadata
        metadata_target = load_metadata('target', user_session_id)
        metadata_reference = load_metadata('reference', user_session_id)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.info(target_info(metadata_target))
        with col2:
            st.info(reference_info(metadata_reference))

        # --- Show user selections ---
        st.info(
            f"**p-value threshold:** {st.session_state[user_session_id]['pval_threshold']} &nbsp;&nbsp; "  # noqa: E501
            f"**Swapped:** {'Yes' if st.session_state[user_session_id]['swap_target'] else 'No'}"  # noqa: E501
        )

        st.sidebar.markdown("### Comparison")
        table_radio = st.sidebar.radio(
            "Select the keyness table to display:",
            ("Tokens", "Tags Only"),
            key=persist(
                "kt_radio1", pathlib.Path(__file__).stem,
                user_session_id),
            horizontal=True)

        st.sidebar.markdown("---")
        # Set up the tagset selection based on the radio button choice
        if table_radio == 'Tokens':
            df, tag_options, tag_radio, tag_type = tagset_selection(
                user_session_id=user_session_id,
                session_state=st.session_state,
                persist_func=persist,
                page_stem=pathlib.Path(__file__).stem,
                tagset_keys={
                    "Parts-of-Speech": {"General": "kw_pos", "Specific": "kw_pos"},
                    "DocuScope": "kw_ds"
                },
                simplify_funcs={
                    "Parts-of-Speech": {"General": freq_simplify_pl, "Specific": None}
                },
                tag_filters={
                    # Add filters here if needed
                },
                tag_radio_key="kt_radio2",
                tag_type_key="kt_type_radio2"
            )

            sort_by, reverse = keyness_sort_controls(
                sort_options=["Keyness (LL)", "Effect Size (LR)"],
                default="Keyness (LL)",
                reverse_default=True,
                key_prefix="kt_"  # or something unique per page/tab
            )

            df = tag_filter_multiselect(df)

            # Map UI label to actual DataFrame column
            sort_col_map = {
                "Keyness (LL)": "LL",
                "Effect Size (LR)": "LR"
            }
            sort_col = sort_col_map[sort_by]

            if df is not None and getattr(df, "height", 0) > 0:
                df = df.sort(sort_col, descending=reverse)
            render_dataframe(df)

            st.sidebar.markdown("---")
            # Add download button for the DataFrame
            toggle_download(
                label="Excel",
                convert_func=convert_to_excel,
                convert_args=(df.to_pandas(),) if (df is not None and getattr(df, "height", 0) > 0) else (None,),  # noqa: E501
                file_name="keywords_tokens.xlsx",
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
            if st.sidebar.button("Generate New Keyness Table", icon=":material/refresh:"):
                # Clear keyness tables for this session
                for key in ["kw_pos", "kw_ds", "kt_pos", "kt_ds"]:
                    if key not in st.session_state[user_session_id]["target"]:
                        st.session_state[user_session_id]["target"][key] = {}
                    st.session_state[user_session_id]["target"][key] = {}
                # Reset keyness_table state
                update_session('keyness_table', False, user_session_id)
                # Optionally clear warnings
                st.session_state[user_session_id]["keyness_warning"] = None
                st.rerun()

            st.sidebar.markdown("---")

        else:
            df, tag_options, tag_radio, tag_type = tagset_selection(
                user_session_id=user_session_id,
                session_state=st.session_state,
                persist_func=persist,
                page_stem=pathlib.Path(__file__).stem,
                tagset_keys={
                    "Parts-of-Speech": "kt_pos",
                    "DocuScope": "kt_ds"
                },
                tag_filters={
                    "Parts-of-Speech": lambda df: df.filter(pl.col("Tag") != "FU"),
                    "DocuScope": lambda df: df.filter(pl.col("Tag") != "Untagged")
                },
                tag_radio_key="kt_radio3"
            )
            # Tabs for displaying keyness table and plot
            tab1, tab2 = st.tabs(["Keyness Table", "Keyness Plot"])
            with tab1:
                filter_vals = st.multiselect("Select tags to filter:", tag_options)
                if filter_vals and df is not None:
                    df = df.filter(pl.col("Tag").is_in(filter_vals))

                render_dataframe(df)

            with tab2:
                if df.height > 0 and df is not None:
                    fig = plot_compare_corpus_bar(df)
                    st.plotly_chart(fig, use_container_width=True)
                    plot_download_link(fig, filename="compare_corpus_bar.png")

            st.sidebar.markdown("---")
            # Add download button for the DataFrame
            toggle_download(
                label="Excel",
                convert_func=convert_to_excel,
                convert_args=(df.to_pandas(),) if (df is not None and getattr(df, "height", 0) > 0) else (None,),  # noqa: E501
                file_name="keywords_tags.xlsx",
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
            if st.sidebar.button("Generate New Keyness Table", icon=":material/refresh:"):
                # Clear keyness tables for this session
                for key in ["kw_pos", "kw_ds", "kt_pos", "kt_ds"]:
                    if key not in st.session_state[user_session_id]["target"]:
                        st.session_state[user_session_id]["target"][key] = {}
                    st.session_state[user_session_id]["target"][key] = {}
                # Reset keyness_table state
                update_session('keyness_table', False, user_session_id)
                # Optionally clear warnings
                st.session_state[user_session_id]["keyness_warning"] = None
                st.rerun()

            st.sidebar.markdown("---")

    else:

        st.markdown(
            body=(
                ":material/manufacturing: Use the button in the sidebar to **generate keywords**.\n\n"  # noqa: E501
                ":material/priority: A **target corpus** and a **reference corpus** must be loaded first.\n\n"  # noqa: E501
                ":material/priority: After the table has been generated, "
                "you will be able to **toggle between the tagsets**."
                )
        )

        pval_selected, swap_selected = sidebar_keyness_options(
            user_session_id,
            load_metadata_func=load_metadata
        )

        # Display the sidebar header for generating frequency table
        st.sidebar.markdown(
            body=(
                "### Generate table\n\n"
                "Use the button to process a table."
                ),
            help=(
                "Tables are generated based on the loaded target and reference corpora. "
                "You can filter the table after it has been generated. "
                "The table will include frequencies and hypothesis testing for the selected tagsets.\n\n"  # noqa: E501
                "Click on the **Help** button for more information on how to use this app."  # noqa: E501
                )
            )
        # Action button to generate keyness tables
        sidebar_action_button(
            button_label="Keyness Table",
            button_icon=":material/manufacturing:",
            preconditions=[
                session.get('has_target')[0],
                session.get('has_reference')[0]
            ],
            action=lambda: generate_keyness_tables(
                user_session_id,
                threshold=pval_selected,
                swap_target=swap_selected
            ),
            spinner_message="Generating keywords..."
        )

        if st.session_state[user_session_id].get("keyness_warning"):
            msg, icon = st.session_state[user_session_id]["keyness_warning"]
            st.error(msg, icon=icon)

        st.sidebar.markdown("---")


if __name__ == "__main__":
    main()
