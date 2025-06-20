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
    generate_keyness_parts,
    get_or_init_user_session,
    load_metadata,
    update_session
    )
from webapp.utilities.ui import (   # noqa: E402
    keyness_sort_controls,
    load_widget_state,
    persist,
    reference_parts,
    render_dataframe,
    sidebar_action_button,
    sidebar_help_link,
    sidebar_keyness_options,
    tag_filter_multiselect,
    target_parts,
    toggle_download,
    update_ref,
    update_tar
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


TITLE = "Compare Corpus Parts"
ICON = ":material/compare_arrows:"

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )


def main():
    """
    Main function to run the Streamlit app for comparing corpus parts.
    This function sets up the page configuration, checks user login status,
    initializes the user session, and renders the UI components for
    comparing corpus parts based on selected categories.
    It allows users to generate keyness tables, filter results by tags,
    and download the data in Excel format.
    """
    # Set login requirements for navigaton
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

    if session.get('keyness_parts')[0] is True:

        load_widget_state(
            pathlib.Path(__file__).stem,
            user_session_id
            )
        metadata_target = load_metadata(
            'target',
            user_session_id
            )

        col1, col2 = st.columns([1, 1])
        with col1:
            st.info(target_parts(metadata_target.get('keyness_parts')[0]['temp']))
        with col2:
            st.info(reference_parts(metadata_target.get('keyness_parts')[0]['temp']))

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
                "cp_radio1",
                pathlib.Path(__file__).stem,
                user_session_id),
            horizontal=True
                )

        st.sidebar.markdown("---")

        if table_radio == 'Tokens':
            tag_radio_tokens = st.sidebar.radio(
                "Select tags to display:",
                ("Parts-of-Speech",
                 "DocuScope"),
                key=persist(
                     "cp_radio2",
                     pathlib.Path(__file__).stem,
                     user_session_id),
                horizontal=True
                )

            if tag_radio_tokens == 'Parts-of-Speech':
                tag_type = st.sidebar.radio(
                    "Select from general or specific tags",
                    ("General", "Specific"),
                    horizontal=True
                    )
                if tag_type == 'General':
                    df = st.session_state[user_session_id]["target"]["kw_pos_cp"]
                    df = freq_simplify_pl(df)
                else:
                    df = st.session_state[user_session_id]["target"]["kw_pos_cp"]
            else:
                df = st.session_state[user_session_id]["target"]["kw_ds_cp"]

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

            if st.sidebar.button(
                    label="Compare New Categories",
                    icon=":material/refresh:",
            ):
                for key in ["kw_pos_cp", "kw_ds_cp", "kt_pos_cp", "kt_ds_cp"]:
                    st.session_state[user_session_id]["target"][key] = {}
                update_session('keyness_parts', False, user_session_id)
                st.rerun()

            st.sidebar.markdown("---")

        else:

            st.sidebar.markdown("### Tagset")
            tag_radio_tags = st.sidebar.radio(
                "Select tags to display:",
                ("Parts-of-Speech",
                 "DocuScope"),
                key=persist(
                    "cp_radio3", pathlib.Path(__file__).stem, user_session_id
                    ),
                horizontal=True
                )

            if tag_radio_tags == 'Parts-of-Speech':
                df = (
                    st.session_state[user_session_id]["target"]["kt_pos_cp"]
                    .filter(pl.col("Tag") != "FU")
                    )

            else:
                df = (
                    st.session_state[user_session_id]["target"]["kt_ds_cp"]
                    .filter(pl.col("Tag") != "Untagged")
                    )

            tab1, tab2 = st.tabs(["Keyness Table", "Keyness Plot"])
            with tab1:
                df = tag_filter_multiselect(df)
                render_dataframe(df)

            with tab2:
                if df.height > 0:
                    fig = plot_compare_corpus_bar(df)
                    st.plotly_chart(fig, use_container_width=True)
                    plot_download_link(fig, filename="compare_corpus_bar.png")

            st.sidebar.markdown("---")

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

            if st.sidebar.button("Compare New Categories", icon=":material/refresh:"):
                for key in ["kw_pos_cp", "kw_ds_cp", "kt_pos_cp", "kt_ds_cp"]:
                    st.session_state[user_session_id]["target"][key] = {}
                update_session('keyness_parts', False, user_session_id)
                st.rerun()

            st.sidebar.markdown("---")

    else:
        st.markdown(
            body=(
                ":material/manufacturing: Use the button in the sidebar to **generate keywords** from subcorpora.\n\n"  # noqa: E501
                ":material/priority: To use this tool, you must first process **metadata** from **Manage Corpus Data**.\n\n"  # noqa: E501
                ":material/priority: After the table has been generated, "
                "you will be able to **toggle between the tagsets**."
                )
        )

        st.sidebar.markdown(
            body="### Select categories to compare",
            help=(
                "Categories can be generated from file names in the target corpus.\n\n"
                "For example, if your file names are formatted like `BIO_G0_02_1.txt`, "
                "`ENG_G0_16_1.txt`, etc., you can extract the categories `BIO` and `ENG`. "  # noqa: E501
                "These categories can then be selected for comparison in the keyness table.\n\n"  # noqa: E501
                "You can select multiple categories for both target and reference corpora, "
                "but you cannot select the same category for both target and reference."
                "If you have not yet processed metadata, please do so in the **Manage Corpus Data** app."  # noqa: E501
            )
            )
        st.sidebar.markdown(
            body=(
                "After **target** and **reference** categories have been selected, "
                "click the button to generate a keyness table."
                )
            )

        if session.get('has_meta')[0] is True:
            metadata_target = load_metadata(
                'target',
                user_session_id
            )
            st.sidebar.markdown('#### Target corpus categories:')
            st.session_state[user_session_id]['tar'] = st.sidebar.multiselect(
                "Select target categories:",
                (sorted(set(metadata_target.get('doccats')[0]['cats']))),
                update_tar(user_session_id),
                key=f"tar_{user_session_id}"
            )

        else:
            st.sidebar.multiselect(
                "Select reference categories:",
                (['No categories to select']),
                key='empty_tar'
                )

        if session.get('has_meta')[0] is True:
            metadata_target = load_metadata(
                'target',
                user_session_id
                )

            st.sidebar.markdown('#### Reference corpus categories:')
            st.session_state[user_session_id]['ref'] = st.sidebar.multiselect(
                "Select reference categories:",
                (sorted(set(metadata_target.get('doccats')[0]['cats']))),
                update_ref(user_session_id),
                key=f"ref_{user_session_id}"
                )

        else:
            st.sidebar.multiselect(
                "Select reference categories:",
                (['No categories to select']),
                key='empty_ref'
                )

        st.sidebar.markdown("---")

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
                "Tables are generated based on the target and reference corpora. "
                "You can filter the table after it has been generated. "
                "The table will include frequencies and hypothesis testing for the selected tagsets.\n\n"  # noqa: E501
                "Click on the **Help** button for more information on how to use this app."  # noqa: E501
                )
            )

        sidebar_action_button(
            button_label="Keyness Table of Corpus Parts",
            button_icon=":material/manufacturing:",
            preconditions=[
                session.get('has_target')[0],
            ],
            action=lambda: generate_keyness_parts(
                user_session_id,
                threshold=pval_selected,
                swap_target=swap_selected
                ),
            spinner_message="Generating keywords..."
        )

        if st.session_state[user_session_id].get("keyness_parts_warning"):
            msg, icon = st.session_state[user_session_id]["keyness_parts_warning"]
            st.error(msg, icon=icon)
            # Clear the warning after displaying it
            del st.session_state[user_session_id]["keyness_parts_warning"]
        st.sidebar.markdown("---")


if __name__ == "__main__":
    main()
