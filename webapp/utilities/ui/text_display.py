"""
Text and document processing utilities for the corpus analysis application.
"""

import pandas as pd
import polars as pl
import random
import streamlit as st
import streamlit.components.v1 as components

from webapp.utilities.core import app_core
from webapp.utilities.session import load_metadata, safe_session_get
from webapp.utilities.exports import convert_to_word
from webapp.utilities.ui.text_visualization import generate_tag_html_legend
from webapp.utilities.ui.corpus_display import safe_metadata_get
from webapp.utilities.ui.data_tables import get_streamlit_column_config
from webapp.utilities.plotting import plot_tag_density
from webapp.utilities.ui.helpers import toggle_download, sidebar_action_button
from webapp.utilities.plotting import plot_download_link
from webapp.utilities.processing import generate_document_html
from webapp.utilities.state import CorpusKeys, SessionKeys, TargetKeys, WarningKeys, MetadataKeys  # noqa: E501
from webapp.utilities.state.widget_state import safe_clear_widget_state

# Document highlighting colors
HEX_HIGHLIGHTS = ['#5fb7ca', '#e35be5', '#ffc701', '#fe5b05', '#cb7d60']


def update_tags(html_state: str, session_id: str) -> None:
    """
    Update the HTML style string for tag highlights in the session state.

    Parameters
    ----------
    html_state : str
        The HTML string representing the current tag highlights.
    session_id : str
        The session ID for which the tag highlights are to be updated.

    Returns
    -------
    None
    """
    _TAGS = f"tags_{session_id}"
    html_highlights = [
        ' { background-color:#5fb7ca; }',
        ' { background-color:#e35be5; }',
        ' { background-color:#ffc701; }',
        ' { background-color:#fe5b05; }',
        ' { background-color:#cb7d60; }'
        ]
    if 'html_str' not in st.session_state[session_id]:
        st.session_state[session_id]['html_str'] = ''
    if _TAGS in st.session_state:
        tags = st.session_state[_TAGS]
        if len(tags) > 5:
            tags = tags[:5]
            st.session_state[_TAGS] = tags
    else:
        tags = []
    tags = ['.' + x for x in tags]
    highlights = html_highlights[:len(tags)]
    style_str = [''.join(x) for x in zip(tags, highlights)]
    style_str = ''.join(style_str)
    style_sheet_str = '<style>' + style_str + '</style>'
    st.session_state[session_id]['html_str'] = style_sheet_str + html_state


def render_document_selection_interface(user_session_id: str, session: dict) -> None:
    """
    Render the document selection interface when no document is loaded.

    Parameters
    ----------
    user_session_id : str
        The user session identifier
    session : dict
        The session state dictionary
    """
    st.markdown(
        body=(
            ":material/manufacturing: Select a document to view "
            "from the sidebar.\n\n"
            ":material/priority: Then select tags to highlight "
            "in the text."
        )
    )

    try:
        metadata_target = load_metadata(CorpusKeys.TARGET, user_session_id)
    except Exception:
        metadata_target = None

    st.sidebar.markdown("### Choose document")
    st.sidebar.write("""Use the menus to select
        the tags you would like to highlight.
        """)
    if (
        safe_session_get(session, SessionKeys.HAS_TARGET, None) is True
    ):
        doc_ids = safe_metadata_get(metadata_target, MetadataKeys.DOCIDS, [], 'ids')

        # Random selection checkbox
        random_selection = st.sidebar.checkbox(
            "Select random document",
            key=f"sd_random_{user_session_id}",
            help="Check this box to randomly select a document from your corpus"
        )

        # Handle random selection
        if random_selection:
            # Generate a random selection if not already set or if checkbox was just checked
            random_key = f"sd_random_doc_{user_session_id}"
            random_changed_key = f"sd_random_changed_{user_session_id}"
            if (random_key not in st.session_state or
                    st.session_state.get(random_changed_key, False)):
                st.session_state[random_key] = random.choice(doc_ids)
                st.session_state[random_changed_key] = False

            # Use the randomly selected document
            doc_key = st.session_state[random_key]

            # Display the selected document (read-only when random is enabled)
            st.sidebar.info(f":material/check: Random selection: **{doc_key}**")

            # Track state changes for re-randomization
            if st.sidebar.button(
                ":material/shuffle: Pick different random document",
                key=f"sd_reroll_{user_session_id}"
            ):
                st.session_state[random_key] = random.choice(doc_ids)
                st.rerun()
        else:
            # Regular selectbox when random is not enabled
            doc_key = st.sidebar.selectbox(
                "Select document to view:",
                doc_ids
            )
            # Clear random selection state when switching back to manual
            random_key = f"sd_random_doc_{user_session_id}"
            safe_clear_widget_state(random_key)
    else:
        doc_key = st.sidebar.selectbox(
            "Select document to view:",
            (['No documents to view'])
        )

    sidebar_action_button(
        button_label="Process Document",
        button_icon=":material/manufacturing:",
        preconditions=[
            safe_session_get(session, SessionKeys.HAS_TARGET, None) is True,
        ],
        action=lambda: generate_document_html(user_session_id, doc_key),
        spinner_message="Processing document..."
    )

    if st.session_state[user_session_id].get(WarningKeys.DOC):
        msg, icon = st.session_state[user_session_id][WarningKeys.DOC]
        st.warning(msg, icon=icon)


def render_tag_selection_interface(tag_options: list, user_session_id: str) -> list:
    """
    Render the tag selection interface with segmented control.

    Parameters
    ----------
    tag_options : list
        Available tag options for selection
    user_session_id : str
        The user session identifier

    Returns
    -------
    list
        Selected tags for highlighting
    """
    with st.expander(
        label="Select tags to highlight",
        icon=":material/colors:",
        expanded=True
    ):
        # Deselect all button
        if st.button(
            label="Deselect All",
            key=f"sd_deselect_{user_session_id}",
            type="tertiary"
        ):
            st.session_state[f"sd_tags_{user_session_id}"] = []

        # Tag selection with segmented control
        tags_to_remove = ["Untagged", "Other", "Y", "FU"]
        tag_options = [item for item in tag_options if item not in tags_to_remove]
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

    return tag_list


def render_document_display(
    html_content: str, tag_list: list, user_session_id: str,
    tag_colors: list = None, tag_html: str = None
) -> None:
    """
    Render the document display with highlighted tags.

    Parameters
    ----------
    html_content : str
        The HTML content of the document
    tag_list : list
        Selected tags for highlighting
    user_session_id : str
        The user session identifier
    tag_colors : list, optional
        Pre-calculated tag colors
    tag_html : str, optional
        Pre-calculated HTML legend
    """
    # Generate colors and HTML legend if not provided
    if tag_colors is None:
        tag_colors = HEX_HIGHLIGHTS[:len(tag_list)]
    if tag_html is None:
        tag_html = generate_tag_html_legend(tag_list, tag_colors)

    # Update HTML content with selected tags
    if html_content:
        # Store tags in the session state key that update_tags expects
        st.session_state[f"tags_{user_session_id}"] = tag_list if tag_list else []
        update_tags(html_content, user_session_id)

    # Display the tag legend
    st.markdown(f"""
                ##### Tags:  {tag_html}
                """,
                unsafe_allow_html=True
                )

    # Display the HTML document
    if 'html_str' not in st.session_state[user_session_id]:
        st.session_state[user_session_id]['html_str'] = ''

    components.html(
        st.session_state[user_session_id]['html_str'],
        height=500,
        scrolling=True
        )


def render_tag_density_plot_interface(
    tag_loc, tag_list: list, tag_colors: list
) -> tuple:
    """
    Render the tag density plot interface in the sidebar.

    Parameters
    ----------
    tag_loc : polars.DataFrame
        Tag location data
    tag_list : list
        Selected tags for plotting
    tag_colors : list
        Colors for the tags

    Returns
    -------
    tuple
        (should_show_plot, error_message, fig) - Whether to show plot,
        error message, and figure
    """

    if len(tag_list) > 5:
        return False, ":no_entry_sign: You can only plot a maximum of 5 tags.", None
    elif len(tag_list) == 0:
        return False, 'There are no tags to plot.', None
    else:
        # Prepare data for plotting
        df_plot = tag_loc.to_pandas()
        df_plot['X'] = (df_plot.index + 1)/(len(df_plot.index))
        df_plot = df_plot[df_plot['Tag'].isin(tag_list)]

        # Create the plotly chart
        fig = plot_tag_density(df_plot, tag_list, tag_colors)
        return True, None, fig

    return False, None, None


def render_document_reset_interface(user_session_id: str) -> None:
    """
    Render the document reset interface in the sidebar.

    Parameters
    ----------
    user_session_id : str
        The user session identifier
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Reset document")
    st.sidebar.markdown("""
                        Click the button to explore a new document.
                        """)
    if st.sidebar.button(
        label="Select a new document",
        icon=":material/refresh:"
    ):
        _TAGS = f"tags_{user_session_id}"
        target_session = st.session_state[user_session_id][CorpusKeys.TARGET]

        # Reset document session data
        for key in [TargetKeys.DOC_POS, TargetKeys.DOC_SIMPLE, TargetKeys.DOC_DS]:
            if key not in target_session:
                target_session[key] = {}
            target_session[key] = {}

        # Update session and clear tags
        app_core.session_manager.update_session_state(
            user_session_id, SessionKeys.DOC, False
        )

        safe_clear_widget_state(_TAGS)

        # Clear document selection widgets (main issue)
        document_widget_keys = [
            f"sd_random_{user_session_id}",
            f"sd_random_doc_{user_session_id}",
            f"sd_random_changed_{user_session_id}",
            f"sd_reroll_{user_session_id}",
        ]

        keys_to_remove = [
            k for k in st.session_state.keys()
            if any(k.startswith(prefix) for prefix in document_widget_keys)
            ]

        safe_clear_widget_state(keys_to_remove)
        st.rerun()


def render_document_interface(
    user_session_id: str, tag_loc, tag_options: list, doc_key: list
) -> None:
    """
    Render the complete document interface when a document is loaded.

    Parameters
    ----------
    user_session_id : str
        The user session identifier
    tag_loc : polars.DataFrame
        Tag location data from tagset selection
    tag_options : list
        Available tag options
    doc_key : list
        Document key/identifier
    """
    # Get document data
    if tag_loc is not None:
        html_content = ''.join(tag_loc.get_column("Text").to_list())
    else:
        html_content = ""

    # Render tag selection interface
    tag_list = render_tag_selection_interface(tag_options, user_session_id)

    # Display document title
    if doc_key:
        st.markdown(
            body=f"### {doc_key[0]}",
            help=(
                "Text highlighting and tag density plots "
                "together to visualize the selected tags in the document. "
                "A tag density plot shows where the selected tags occur "
                "in the document, normalized to text length. "
                "In this way, you can see both what words and phrases "
                "are tagged, and how they are distributed in the text."
            )
            )

    # Generate tag colors for all components
    tag_colors = HEX_HIGHLIGHTS[:len(tag_list)]
    tag_html = generate_tag_html_legend(tag_list, tag_colors)

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(
        [":material/docs: Document",
         ":material/bar_chart: Tag Density Plot",
         ":material/table_view: Statistics"]
         )

    with tab1:
        # Render document display with tags
        render_document_display(
            html_content, tag_list, user_session_id, tag_colors, tag_html
        )
        toggle_download(
            label="Word",
            convert_func=convert_to_word,
            convert_args=(
                st.session_state[user_session_id]['html_str'],
                tag_html,
                doc_key
            ),
            file_name="document_tags.docx",
            mime="docx",
            location=st
            )

    with tab2:
        # Render tag density plot
        should_show_plot, error_message, fig = render_tag_density_plot_interface(
            tag_loc, tag_list, tag_colors
        )

        if error_message:
            st.error(error_message)
        elif should_show_plot and fig:
            st.plotly_chart(fig, use_container_width=True)
            plot_download_link(fig, filename="tag_density_plot.png")
        else:
            st.info("Click 'Tag Density Plot' in the sidebar to generate the plot.")

    with tab3:
        # Generate and display statistics
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
        if df is not None and len(df) > 0:
            column_config = get_streamlit_column_config(df)
            st.data_editor(df,
                           hide_index=True,
                           column_config=column_config,
                           disabled=True)
        toggle_download(
            label="Excel",
            convert_args=(df,) if (df is not None) else (None,),
            file_name="document_tags.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            location=st
            )

    # Render remaining sidebar interfaces
    render_document_reset_interface(user_session_id)
