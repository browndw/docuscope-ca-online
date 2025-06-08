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

import docuscospacy as ds
import streamlit as st

# Ensure project root is in sys.path for both desktop and online
project_root = pathlib.Path(__file__).parent.parents[1].resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import webapp.utilities as _utils   # noqa: E402
from webapp.menu import menu, require_login   # noqa: E402

TITLE = "Advanced Plotting"
ICON = ":material/line_axis:"


def main() -> None:
    # Set login requirements for navigaton
    require_login()
    menu()
    st.markdown(f"## {TITLE}")
    # Get or initialize user session
    user_session_id, session = _utils.handlers.get_or_init_user_session()

    st.sidebar.link_button(
        label="Help",
        url="https://browndw.github.io/docuscope-docs/guide/advanced-plotting.html",
        icon=":material/help:"
        )

    try:
        # Load metadata for the target
        metadata_target = _utils.handlers.load_metadata(
            'target',
            user_session_id
            )
    except Exception:
        pass

    # Display a markdown message for plotting
    st.markdown(_utils.content.message_plotting)

    # Radio button to select the type of plot
    plot_type = st.radio(
        "What kind of plot would you like to make?",
        ["Boxplot", "Scatterplot", "PCA"],
        captions=[
            """:material/box: Boxplots of normalized tag frequencies
            with grouping variables (if you've processed corpus metadata).
            """,
            """:material/scatter_plot: Scatterplots of
            normalized tag frequencies with grouping variables
            (if you've processed corpus metadata).
            """,
            """:material/linear_scale: Principal component analysis
            from scaled tag frequences with highlighting
            for groups (if you've processed corpus metadata).
            """
            ],
        horizontal=False,
        index=None
        )

    # Handle Boxplot selection
    if plot_type == "Boxplot" and session.get('has_target')[0] is True:

        _utils.handlers.update_session('pca', False, user_session_id)
        st.sidebar.markdown("### Tagset")

        with st.sidebar.expander("About general tags"):
            st.markdown(_utils.content.message_general_tags)

        # Radio button to select tag type
        tag_radio_tokens = st.sidebar.radio(
            "Select tags to display:",
            ("Parts-of-Speech", "DocuScope"),
            on_change=_utils.handlers.clear_plots, args=(user_session_id,),
            horizontal=True
            )

        if tag_radio_tokens == 'Parts-of-Speech':
            tag_type = st.sidebar.radio(
                "Select from general or specific tags",
                ("General", "Specific"),
                on_change=_utils.handlers.clear_plots, args=(user_session_id,),
                horizontal=True)
            if tag_type == 'General':
                df = st.session_state[user_session_id]["target"]["dtm_pos"]
                df = ds.dtm_simplify(df)

            elif tag_type == 'Specific':
                df = st.session_state[user_session_id]["target"]["dtm_pos"]

        # Handle DocuScope tag selection
        elif tag_radio_tokens == 'DocuScope':
            df = st.session_state[user_session_id]["target"]["dtm_ds"]
            tag_type = None

        st.sidebar.markdown("---")

        st.markdown("""---""")

        # Toggle to plot using grouping variables
        by_group = st.toggle("Plot using grouping variables.")

        # Determine categories for plotting
        if df.height == 0 or df is None:
            cats = []
        elif df.height > 0:
            to_drop = ['doc_id', 'Other', 'FU', 'Untagged']
            cats = sorted(
                list(df.drop([x for x in to_drop if x in df.columns]).columns)
                )

        # Handle plotting with grouping variables
        if by_group:
            if session['has_meta'][0] is False:
                st.warning(
                    """
                    It doesn't look like you've processed any metadata yet.
                    You can do this by clicking on the **Manage Corpus Data**
                    button above.
                    """,
                    icon=":material/new_label:"
                    )
            else:
                with st.sidebar.form(key="boxplot_group_form"):
                    st.markdown("### Variables")
                    box_vals = st.multiselect(
                        "Select variables for plotting:",
                        (cats),
                        key=f"boxplot_vars_{user_session_id}",
                        help="Choose one or more tags to plot as boxplots."
                        # REMOVE on_change and args here!
                    )

                    st.markdown('### Grouping variables')
                    st.markdown(
                        """Select grouping variables from your metadata
                        and click the button to generate boxplots of frequencies.
                        """
                    )
                    st.markdown('#### Group A')
                    grpa = st.multiselect(
                        "Select categories for group A:",
                        (sorted(set(metadata_target.get('doccats')[0]['cats']))),
                        key=f"grpa_{user_session_id}",
                        on_change=_utils.handlers.update_grpa(user_session_id),
                        help="Group A will be shown in one color.",
                        disabled=not cats
                    )

                    st.markdown('#### Group B')
                    grpb = st.multiselect(
                        "Select categories for group B:",
                        (sorted(set(metadata_target.get('doccats')[0]['cats']))),
                        on_change=_utils.handlers.update_grpb(user_session_id),
                        key=f"grpb_{user_session_id}",
                        help="Group B will be shown in another color.",
                        disabled=not cats
                    )

                    submitted = st.form_submit_button(
                        label="Boxplots of Frequencies by Group",
                        icon=":material/manufacturing:"
                        )

                    if submitted:
                        # If you need to clear state, do it here:
                        _utils.handlers.clear_boxplot_multiselect(
                            user_session_id
                            )
                        _utils.handlers.generate_boxplot_by_group(
                            user_session_id, df, box_vals, grpa, grpb
                        )

                if st.session_state[user_session_id].get("boxplot_group_warning"):
                    msg, icon = st.session_state[user_session_id]["boxplot_group_warning"]
                    st.warning(msg, icon=icon)

                if (
                    "boxplot_group_df" in st.session_state[user_session_id] and
                    st.session_state[user_session_id]["boxplot_group_df"] is not None and
                    st.session_state[user_session_id]["boxplot_group_df"].shape[0] > 0
                ):
                    df_plot = st.session_state[user_session_id]["boxplot_group_df"]
                    if _utils.handlers.is_valid_df(df_plot, ['Group', 'Tag']):
                        hex_color, palette = _utils.formatters.color_picker_controls()
                        fig = _utils.formatters.plot_grouped_boxplot(df_plot, color=hex_color, palette=palette)
                        st.plotly_chart(fig, use_container_width=True)

                        stats = st.session_state[user_session_id]["boxplot_group_stats"]
                        st.markdown("##### Descriptive statistics:")
                        st.dataframe(stats, hide_index=True)
                    else:
                        st.info(
                            """
                            Please select valid variables and ensure data is available.
                            """
                            )

        # Handle plotting without grouping variables
        else:
            with st.sidebar.form(key="boxplot_form"):
                st.markdown("### Variables")
                box_vals = st.multiselect(
                    "Select variables for plotting:",
                    (cats),
                    key=f"boxplot_vars_{user_session_id}",
                    help="Choose one or more tags to plot as boxplots."
                    # REMOVE on_change and args here!
                )
                submitted = st.form_submit_button(
                    label="Boxplots of Frequencies",
                    icon=":material/manufacturing:"
                    )

                if submitted:
                    _utils.handlers.clear_boxplot_multiselect(user_session_id)
                    _utils.handlers.generate_boxplot(
                        user_session_id, df, box_vals
                    )

            if st.session_state[user_session_id].get("boxplot_warning"):
                msg, icon = st.session_state[user_session_id]["boxplot_warning"]
                st.warning(msg, icon=icon)

            # Plot if available
            if (
                "boxplot_df" in st.session_state[user_session_id] and
                st.session_state[user_session_id]["boxplot_df"] is not None and
                st.session_state[user_session_id]["boxplot_df"].shape[0] > 0
            ):
                df_plot = st.session_state[user_session_id]["boxplot_df"]
                if _utils.handlers.is_valid_df(df_plot, ['Tag', 'RF']):
                    # --- color controls ---
                    hex_color, palette = _utils.formatters.color_picker_controls()
                    fig = _utils.formatters.plot_general_boxplot(df_plot, color=hex_color, palette=palette)
                    st.plotly_chart(fig, use_container_width=True)

                    stats = st.session_state[user_session_id]["boxplot_stats"]
                    st.markdown("##### Descriptive statistics:")
                    st.dataframe(stats, hide_index=True)
                else:
                    st.info("Please select valid variables and ensure data is available.")

    # Handle Scatterplot selection
    elif plot_type == "Scatterplot" and session.get('has_target')[0] is True:

        _utils.handlers.update_session('pca', False, user_session_id)
        st.sidebar.markdown("### Tagset")

        with st.sidebar.expander("About general tags"):
            st.markdown(_utils.content.message_general_tags)

        # Radio button to select tag type
        tag_radio_tokens = st.sidebar.radio(
            "Select tags to display:",
            ("Parts-of-Speech", "DocuScope"),
            on_change=_utils.handlers.clear_plots,
            args=(user_session_id,), horizontal=True
            )

        # Handle Parts-of-Speech tag selection
        if tag_radio_tokens == 'Parts-of-Speech':
            tag_type = st.sidebar.radio(
                "Select from general or specific tags",
                ("General", "Specific"),
                on_change=_utils.handlers.clear_plots, args=(user_session_id,),
                horizontal=True
                )
            if tag_type == 'General':
                df = st.session_state[user_session_id]["target"]["dtm_pos"]
                df = ds.dtm_simplify(df)

            elif tag_type == 'Specific':
                df = st.session_state[user_session_id]["target"]["dtm_pos"]

        # Handle DocuScope tag selection
        elif tag_radio_tokens == 'DocuScope':
            df = st.session_state[user_session_id]["target"]["dtm_ds"]
            tag_type = None

        st.sidebar.markdown("---")

        st.markdown("---")

        by_group_highlight = st.toggle("Hightlight groups in scatterplots.")

        # Determine categories for plotting
        if df.height == 0 or df is None:
            cats = []
        elif df.height > 0:
            to_drop = ['doc_id', 'Other', 'FU', 'Untagged']
            cats = sorted(
                list(df.drop([x for x in to_drop if x in df.columns]).columns)
                )

        # Handle scatterplot with group highlighting
        if by_group_highlight:
            if session['has_meta'][0] is False:
                st.warning(
                    """
                    It doesn't look like you've processed any metadata yet.
                    You can do this by clicking on the **Manage Corpus Data** button above.
                    """,
                    icon=":material/new_label:"
                )
            else:
                with st.sidebar.form(key="scatterplot_group_form"):
                    st.markdown("### Variables")
                    xaxis = st.selectbox(
                        "Select variable for the x-axis",
                        (cats),
                        key=f"scatter_x_{user_session_id}",
                        help="Choose a tag for the x-axis.",
                        disabled=not cats
                    )
                    yaxis = st.selectbox(
                        "Select variable for the y-axis",
                        (cats),
                        key=f"scatter_y_{user_session_id}",
                        help="Choose a tag for the y-axis.",
                        disabled=not cats
                    )
                    submitted = st.form_submit_button("Scatterplot of Frequencies by Group")

                    if submitted:
                        _utils.handlers.generate_scatterplot_with_groups(
                            user_session_id, df, xaxis, yaxis, metadata_target
                        )

            if st.session_state[user_session_id].get("scatter_group_warning"):
                msg, icon = st.session_state[user_session_id]["scatter_group_warning"]
                st.warning(msg, icon=icon)

            # --- Multiselect for categories to highlight ---
            all_groups = sorted(set(metadata_target.get('doccats')[0]['cats']))
            selected_groups = st.multiselect(
                "Highlight categories in plot:",
                all_groups,
                default=[],  # Start with no groups selected
                key=f"highlight_scatter_groups_{user_session_id}"
            )

            # Plot if available
            if (
                "scatterplot_group_df" in st.session_state[user_session_id] and
                st.session_state[user_session_id]["scatterplot_group_df"] is not None
            ):
                df_plot = st.session_state[user_session_id]["scatterplot_group_df"]
                if _utils.handlers.is_valid_df(df_plot, [xaxis, yaxis]):
                    fig = _utils.formatters.plot_scatter_highlight(
                        df_plot,
                        xaxis,
                        yaxis,
                        'Group',
                        selected_groups
                        )
                    st.plotly_chart(fig, use_container_width=True)

                    cc_df, cc_r, cc_p = st.session_state[user_session_id]["scatter_group_correlation"]  # noqa: E501
                    st.markdown(_utils.content.message_correlation_info(
                        cc_df,
                        cc_r,
                        cc_p
                    ))
                else:
                    st.info("Please select valid variables and ensure data is available.")

        # Handle scatterplot without group highlighting
        else:
            with st.sidebar.form(key="scatterplot_form"):
                st.markdown("### Variables")
                xaxis = st.selectbox(
                    "Select variable for the x-axis",
                    (cats),
                    key=f"scatter_x_{user_session_id}",
                    help="Choose a tag for the x-axis.",
                    disabled=not cats
                )
                yaxis = st.selectbox(
                    "Select variable for the y-axis",
                    (cats),
                    key=f"scatter_y_{user_session_id}",
                    help="Choose a tag for the y-axis.",
                    disabled=not cats
                )
                submitted = st.form_submit_button("Scatterplot of Frequencies")

                if submitted:
                    _utils.handlers.generate_scatterplot(
                        user_session_id, df, xaxis, yaxis
                    )

            if st.session_state[user_session_id].get("scatter_warning"):
                msg, icon = st.session_state[user_session_id]["scatter_warning"]
                st.warning(msg, icon=icon)

            # Plot if available
            if (
                "scatterplot_df" in st.session_state[user_session_id] and
                st.session_state[user_session_id]["scatterplot_df"] is not None
            ):
                df_plot = st.session_state[user_session_id]["scatterplot_df"]
                if _utils.handlers.is_valid_df(df_plot, [xaxis, yaxis]):
                    fig = _utils.formatters.plot_scatter(df_plot, xaxis, yaxis)
                    st.plotly_chart(fig, use_container_width=True)

                    cc_df, cc_r, cc_p = st.session_state[user_session_id]["scatter_correlation"]  # noqa: E501
                    st.markdown(_utils.content.message_correlation_info(
                        cc_df,
                        cc_r,
                        cc_p)
                    )
                else:
                    st.info("Please select valid variables and ensure data is available.")

    # Handle PCA selection
    elif plot_type == "PCA" and session.get('has_target')[0] is True:

        st.sidebar.markdown("### Tagset")

        with st.sidebar.expander("About general tags"):
            st.markdown(_utils.content.message_general_tags)

        # Radio button to select tag type
        tag_radio_tokens = st.sidebar.radio(
            "Select tags to display:",
            ("Parts-of-Speech", "DocuScope"),
            on_change=_utils.handlers.clear_plots, args=(user_session_id,),
            horizontal=True
            )

        # Handle Parts-of-Speech tag selection
        if tag_radio_tokens == 'Parts-of-Speech':
            tag_type = st.sidebar.radio(
                "Select from general or specific tags",
                ("General", "Specific"),
                on_change=_utils.handlers.clear_plots, args=(user_session_id,),
                horizontal=True
                )
            if tag_type == 'General':
                df = st.session_state[user_session_id]["target"]["dtm_pos"]
                df = ds.dtm_simplify(df)
                df = ds.dtm_weight(df, scheme="prop")
                df = ds.dtm_weight(df, scheme="scale").to_pandas()

            elif tag_type == 'Specific':
                df = st.session_state[user_session_id]["target"]["dtm_pos"]
                df = ds.dtm_weight(df, scheme="prop")
                df = ds.dtm_weight(df, scheme="scale").to_pandas()

        # Handle DocuScope tag selection
        elif tag_radio_tokens == 'DocuScope':
            df = st.session_state[user_session_id]["target"]["dtm_ds"]
            df = ds.dtm_weight(df, scheme="prop")
            df = ds.dtm_weight(df, scheme="scale").to_pandas()
            tag_type = None

            st.sidebar.markdown("---")
            st.sidebar.markdown("### Principal Component Analysis")
            st.sidebar.markdown("""
                                Click the button to plot principal compenents.
                                """)

        st.markdown("---")

        # Handle PCA button click
        with st.sidebar.form(key="pca_form"):
            st.markdown("### PCA")
            submitted = st.form_submit_button("Principal Component Analysis")

            if submitted:
                _utils.handlers.generate_pca(
                    user_session_id, df, metadata_target, session
                )

        if st.session_state[user_session_id].get("pca_warning"):
            msg, icon = st.session_state[user_session_id]["pca_warning"]
            st.warning(msg, icon=icon)

        # Plot PCA results if PCA has been performed
        if (
            session.get('pca')[0] is True and
            "pca_df" in st.session_state[user_session_id]["target"] and
            st.session_state[user_session_id]["target"]["pca_df"] is not None and
            st.session_state[user_session_id]["target"]["pca_df"].shape[0] > 0
        ):
            pca_df = st.session_state[user_session_id]["target"]["pca_df"]
            if _utils.handlers.is_valid_df(pca_df, ['PC1', 'PC2']):
                contrib_df = st.session_state[user_session_id]["target"]["contrib_df"]
                ve = metadata_target.get("variance")[0]['temp']

                st.session_state[user_session_id]['pca_idx'] = st.sidebar.selectbox(
                    "Select principal component to plot ",
                    (list(range(1, len(df.columns))))
                    )

                pca_x, pca_y, contrib_x, contrib_y, ve_1, ve_2, contrib_1_plot, contrib_2_plot = _utils.analysis.update_pca_plot(  # noqa: E501
                    pca_df,
                    contrib_df,
                    ve,
                    int(st.session_state[user_session_id]['pca_idx'])
                    )

                tab1, tab2 = st.tabs(["PCA Plot", "Variable Contribution"])
                with tab1:
                    # --- NEW: Multiselect for categories to highlight ---
                    if session.get('has_meta')[0] is True:
                        groups = sorted(set(metadata_target.get('doccats')[0]['cats']))
                        selected_groups = st.multiselect(
                            "Highlight categories in PCA plot:",
                            groups,
                            default=[],  # Start with no groups selected
                            key=f"highlight_pca_groups_{user_session_id}"
                        )
                    else:
                        selected_groups = []

                    # Add a Highlight column for coloring
                    fig = _utils.formatters.plot_pca_scatter_highlight(
                        pca_df,
                        pca_x,
                        pca_y,
                        'Group',
                        selected_groups,
                        x_label=pca_x,
                        y_label=pca_y
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display variance explained
                    st.markdown(_utils.content.message_variance_info(
                        pca_x,
                        pca_y,
                        ve_1,
                        ve_2)
                    )

                with tab2:
                    # Display contribution information
                    st.markdown(_utils.content.message_contribution_info(
                        pca_x,
                        pca_y,
                        contrib_x,
                        contrib_y)
                    )
                    st.markdown(
                        "##### Variable contribution (by %) to principal component:"
                    )

                    with st.expander("About variable contribution"):
                        st.markdown(_utils.content.message_variable_contrib)

                    # --- Add sort_by radio ---
                    sort_by = st.radio(
                        "Sort variables by:",
                        (pca_x, pca_y),
                        index=0,
                        horizontal=True
                    )

                    fig = _utils.formatters.plot_pca_variable_contrib_bar(
                        contrib_1_plot, contrib_2_plot,
                        pc1_label=pca_x, pc2_label=pca_y,
                        sort_by=sort_by
                        )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(
                    """
                    PCA data is not available or required components are missing.
                    """
                    )

            st.sidebar.markdown("---")


if __name__ == "__main__":
    main()
