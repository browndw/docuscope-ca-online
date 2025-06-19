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

import pandas as pd
import polars as pl
import streamlit as st
import streamlit.components.v1 as components

from webapp.utilities.session import (  # noqa: E402
    load_metadata,
    get_or_init_user_session,
    update_session,
    )
from webapp.utilities.exports import (  # noqa: E402
    convert_to_word,
    )
from webapp.utilities.ui import (  # noqa: E402
    generate_tag_html_legend,
    plot_tag_density,
    sidebar_action_button,
    tagset_selection,
    toggle_download,
    update_tags
)
from webapp.utilities.state import (  # noqa: E402
    load_widget_state,
    persist
)
# Temporary import for functions not yet migrated
from webapp.utilities.processing import generate_document_html  # noqa: E402
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

HEX_HIGHLIGHTS = ['#5fb7ca', '#e35be5', '#ffc701', '#fe5b05', '#cb7d60']

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

    if session.get('doc')[0] is True:

        load_widget_state(
            pathlib.Path(__file__).stem,
            user_session_id
            )
        metadata_target = load_metadata(
            CorpusKeys.TARGET,
            user_session_id
            )

        st.sidebar.markdown("### Tagset")

        st.sidebar.markdown("""Use the menus to select
                            up to **5 tags** you would like to highlight.
                            """)

        # Use the reusable tagset selection function
        tag_loc, tag_options, tag_radio, tag_type = tagset_selection(
            user_session_id=user_session_id,
            session_state=st.session_state,
            persist_func=persist,
            page_stem=pathlib.Path(__file__).stem,
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

        # Get document data
        if tag_loc is not None:
            html_content = ''.join(tag_loc.get_column("Text").to_list())
            doc_key = tag_loc.get_column("doc_id").unique().to_list()
        else:
            html_content = ""
            doc_key = []

        # Tag selection using segmented control in main area (modern pattern)
        # Move tag selection from sidebar to main area with segmented control
        with st.expander(
            label="Select tags to highlight",
            icon=":material/filter_alt:",
            expanded=True
        ):
            st.markdown(
                "Use the controls to select up to **5 tags** "
                "you would like to highlight."
            )

            # Deselect all button
            if st.button(
                label="Deselect All",
                key=f"sd_deselect_{user_session_id}",
                type="tertiary"
            ):
                st.session_state[f"sd_tags_{user_session_id}"] = []

            # Use tag_options from tagset_selection
            tag_list = st.segmented_control(
                "Select tags:",
                options=tag_options,
                selection_mode="multi",
                key=f"sd_tags_{user_session_id}",
                help=(
                    "Click to select tags for highlighting. "
                    "Click again to deselect. Maximum 5 tags."
                )
            )

            # Convert None to empty list and limit to 5 tags
            if tag_list is None:
                tag_list = []
            elif len(tag_list) > 5:
                tag_list = tag_list[:5]
                st.warning("Only the first 5 selected tags will be used for highlighting.")

        # Generate colors and HTML legend
        tag_colors = HEX_HIGHLIGHTS[:len(tag_list)]
        tag_html = generate_tag_html_legend(tag_list, tag_colors)

        # Update HTML content with selected tags
        if html_content:
            # Store tags in the session state key that update_tags expects
            st.session_state[f"tags_{user_session_id}"] = tag_list if tag_list else []
            update_tags(html_content, user_session_id)

        # Generate DataFrame for statistics (now handled by tag_filters)
        if tag_loc is not None:
            df = (tag_loc
                  .group_by("Tag").len("AF")
                  .with_columns(
                      pl.col("AF")
                      .truediv(pl.sum("AF")).mul(100).alias("RF")
                      )
                  .sort(["AF", "Tag"], descending=[True, False])
                  ).to_pandas()
        else:
            df = pd.DataFrame()

        st.sidebar.markdown("---")
        st.sidebar.markdown("### Plot tag locations")
        with st.sidebar.expander("Plot explanation"):
            st.write("""The plot(s) shows lines segment
                    where tags occur in what might be called
                    'normalized text time.'
                    For example, if you had a text 100 tokens long
                    and a tag occurred at the 10th, 25th, and 60th token,
                    the plot would show lines at
                    10%, 25%, and 60% along the x-axis.
                    """)

        st.markdown(f"""
                    ###  {doc_key[0]}
                    """)

        if st.sidebar.button("Tag Density Plot"):
            if len(tag_list) > 5:
                st.write(""":no_entry_sign: You can only plot
                         a maximum of 5 tags.
                         """)
            elif len(tag_list) == 0:
                st.write('There are no tags to plot.')
            else:
                # Prepare data for plotting
                df_plot = tag_loc.to_pandas()
                df_plot['X'] = (df_plot.index + 1)/(len(df_plot.index))
                df_plot = df_plot[df_plot['Tag'].isin(tag_list)]

                # Create and display the plotly chart
                fig = plot_tag_density(df_plot, tag_list, tag_colors)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
                    ##### Tags:  {tag_html}
                    """,
                    unsafe_allow_html=True
                    )

        if 'html_str' not in st.session_state[user_session_id]:
            st.session_state[user_session_id]['html_str'] = ''

        components.html(
            st.session_state[user_session_id]['html_str'],
            height=500,
            scrolling=True
            )

        st.dataframe(df, hide_index=True)

        st.sidebar.markdown("---")

        # Use the reusable toggle_download function
        toggle_download(
            label="Word",
            convert_func=convert_to_word,
            convert_args=(
                st.session_state[user_session_id]['html_str'],
                tag_html,
                doc_key,
                df
            ),
            file_name="document_tags.docx",
            mime="docx"
        )

        st.sidebar.markdown("---")

        st.sidebar.markdown("### Reset document")
        st.sidebar.markdown("""
                            Click the button to explore a new document.
                            """)
        if st.sidebar.button("Select a new document"):
            _TAGS = f"tags_{user_session_id}"
            target_session = st.session_state[user_session_id][CorpusKeys.TARGET]

            if TargetKeys.DOC_POS not in target_session:
                target_session[TargetKeys.DOC_POS] = {}
            target_session[TargetKeys.DOC_POS] = {}

            if TargetKeys.DOC_SIMPLE not in target_session:
                target_session[TargetKeys.DOC_SIMPLE] = {}
            target_session[TargetKeys.DOC_SIMPLE] = {}

            if TargetKeys.DOC_DS not in target_session:
                target_session[TargetKeys.DOC_DS] = {}
            target_session[TargetKeys.DOC_DS] = {}

            update_session(SessionKeys.DOC, False, user_session_id)

            if _TAGS in st.session_state:
                del st.session_state[_TAGS]
            st.rerun()

        st.sidebar.markdown("---")

    else:

        st.markdown("_utils.content.message_single_document")

        try:
            metadata_target = load_metadata(
                CorpusKeys.TARGET,
                user_session_id
                )
        except Exception:
            pass

        st.sidebar.markdown("### Choose document")
        st.sidebar.write("""Use the menus to select
            the tags you would like to highlight.
            """)

        if session.get('has_target')[0] is True:
            doc_key = st.sidebar.selectbox(
                "Select document to view:",
                (sorted(metadata_target.get('docids')[0]['ids']))
                )
        else:
            doc_key = st.sidebar.selectbox(
                "Select document to view:",
                (['No documents to view'])
                )

        sidebar_action_button(
            button_label="Process Document",
            button_icon=":material/manufacturing:",
            preconditions=[
                session.get('has_target')[0],
            ],
            action=lambda: generate_document_html(
                user_session_id, doc_key
            ),
            spinner_message="Processing document..."
        )

        if st.session_state[user_session_id].get(WarningKeys.DOC):
            msg, icon = st.session_state[user_session_id][WarningKeys.DOC]
            st.warning(msg, icon=icon)

        st.sidebar.markdown("---")

if __name__ == "__main__":
    main()
