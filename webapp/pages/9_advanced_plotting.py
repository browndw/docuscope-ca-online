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

from webapp.utilities.handlers import (  # noqa: E402
    generate_boxplot,
    generate_boxplot_by_group,
    generate_pca,
    generate_scatterplot,
    generate_scatterplot_with_groups,
    get_or_init_user_session,
    is_valid_df,
    load_metadata,
    update_session
    )
from webapp.utilities.ui import (   # noqa: E402
    clear_boxplot_multiselect,
    clear_plots,
    clear_scatterplot_multiselect,
    color_picker_controls,
    contribution_info,
    correlation_info,
    tagset_selection,
    update_grpa,
    update_grpb,
    update_pca_idx_tab1,
    update_pca_idx_tab2,
    variance_info
    )
from webapp.utilities.formatters import (  # noqa: E402
    plot_download_link,
    plot_general_boxplot,
    plot_grouped_boxplot,
    plot_pca_scatter_highlight,
    plot_pca_variable_contrib_bar,
    plot_scatter,
    plot_scatter_highlight
    )
from webapp.utilities.analysis import (  # noqa: E402
    correlation_update,
    update_pca_plot
    )
from webapp.menu import (   # noqa: E402
    menu,
    require_login
    )

TITLE = "Advanced Plotting"
ICON = ":material/line_axis:"

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )


def main() -> None:
    # Set login requirements for navigaton
    require_login()
    menu()
    st.markdown(f"## {TITLE}")
    # Get or initialize user session
    user_session_id, session = get_or_init_user_session()

    st.sidebar.link_button(
        label="Help",
        url="https://browndw.github.io/docuscope-docs/guide/advanced-plotting.html",
        icon=":material/help:"
        )

    try:
        # Load metadata for the target
        metadata_target = load_metadata(
            'target',
            user_session_id
            )
    except Exception:
        pass

    # Display a markdown message for plotting
    st.markdown(
        body="_utils.content.message_plotting"
        )

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
        on_change=clear_plots, args=(user_session_id,),
        horizontal=False,
        index=None
        )

    # Handle Boxplot selection
    if plot_type == "Boxplot" and session.get('has_target')[0] is True:

        update_session('pca', False, user_session_id)
        st.sidebar.markdown("### Tagset")

        # Radio button to select tag type
        df, cats, tag_radio, tag_type = tagset_selection(
            user_session_id=user_session_id,
            session_state=st.session_state,
            persist_func=lambda key, page, user: f"{key}_{page}_{user}",
            page_stem="advanced_plotting",
            tagset_keys={
                "Parts-of-Speech": {"General": "dtm_pos", "Specific": "dtm_pos"},
                "DocuScope": "dtm_ds"
            },
            simplify_funcs={
                "Parts-of-Speech": {"General": ds.dtm_simplify, "Specific": None}
            },
            tag_filters={
                "Parts-of-Speech": {
                    "Specific": lambda df: df.drop([col for col in ["FU"] if col in df.columns]),  # noqa: E501
                    "General": lambda df: df.drop([col for col in ["Other"] if col in df.columns])  # noqa: E501
                },
                "DocuScope": lambda df: df.drop([col for col in ["Untagged"] if col in df.columns])  # noqa: E501
            },
            tag_radio_key="tag_radio",
            tag_type_key="tag_type_radio",
            on_change=clear_plots,
            on_change_args=(user_session_id,)
        )

        st.markdown("""---""")

        # Toggle to plot using grouping variables
        by_group = st.toggle(
            label="Plot using grouping variables.",
            on_change=clear_plots, args=(user_session_id,)
            )

        # Determine categories for plotting
        if df is None or getattr(df, "height", 0) == 0:
            cats = []
        else:
            cats = sorted([col for col in df.columns if col != "doc_id"])

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
                with st.expander("Boxplot Variables", expanded=True):
                    # Create a form for boxplot grouping variables
                    # Before the form
                    st.markdown('### Grouping variables')
                    st.markdown(
                        """Select grouping variables from your metadata
                        and click the button to generate boxplots of frequencies."""
                    )
                    all_cats = sorted(set(metadata_target.get('doccats')[0]['cats']))

                    grpa = st.segmented_control(
                        "Select categories for group A:",
                        all_cats,
                        selection_mode="multi",
                        key=f"grpa_{user_session_id}",
                        on_change=update_grpa,
                        args=(user_session_id,),
                        help="Group A will be shown in one boxplot.",
                        disabled=not cats
                    )

                    grpb = st.segmented_control(
                        "Select categories for group B:",
                        all_cats,
                        selection_mode="multi",
                        key=f"grpb_{user_session_id}",
                        on_change=update_grpb,
                        args=(user_session_id,),
                        help="Group B will be shown in another boxplot.",
                        disabled=not cats
                    )

                    st.markdown("### Variables")
                    box_val1 = st.segmented_control(
                        "Select variables for plotting:",
                        cats,
                        selection_mode="multi",
                        key=f"boxplot_vars_grouped_{user_session_id}",
                        help="Choose one or more tags to plot as boxplots."
                    )

                # Sidebar action button
                boxplot_group_btn = st.sidebar.button(
                    label="Generate Boxplots",
                    key=f"boxplot_group_btn_{user_session_id}",
                    help="Generate grouped boxplots for selected variables.",
                    type="secondary",
                    use_container_width=False,
                    icon=":material/manufacturing:"
                )

                if boxplot_group_btn:
                    clear_boxplot_multiselect(user_session_id)
                    generate_boxplot_by_group(user_session_id, df, box_val1, grpa, grpb)

                if st.session_state[user_session_id].get("boxplot_group_warning"):
                    msg, icon = st.session_state[user_session_id]["boxplot_group_warning"]
                    st.warning(msg, icon=icon)

                if (
                    "boxplot_group_df" in st.session_state[user_session_id] and
                    st.session_state[user_session_id]["boxplot_group_df"] is not None and
                    st.session_state[user_session_id]["boxplot_group_df"].shape[0] > 0
                ):
                    df_plot = st.session_state[user_session_id]["boxplot_group_df"]
                    if is_valid_df(df_plot, ['Group', 'Tag']):
                        # Place color controls and plotting here, outside the form
                        color_dict = color_picker_controls(
                            [", ".join(grpa), ", ".join(grpb)],
                            key_prefix=f"color_picker_boxplot_{user_session_id}"
                        )
                        fig = plot_grouped_boxplot(df_plot, color=color_dict)
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
            with st.expander("Boxplot Variables", expanded=True):
                st.markdown("### Variables")
                box_val2 = st.segmented_control(
                    "Select variables for plotting:",
                    cats,
                    selection_mode="multi",
                    key=f"boxplot_vars_nongrouped_{user_session_id}",
                    help="Choose one or more tags to plot as boxplots."
                )

            # Sidebar action button
            boxplot_btn = st.sidebar.button(
                label="Boxplots of Frequencies",
                key=f"boxplot_btn_{user_session_id}",
                help="Generate boxplots for selected variables.",
                type="secondary",
                use_container_width=False,
                icon=":material/manufacturing:"
            )

            if boxplot_btn:
                clear_boxplot_multiselect(user_session_id)
                generate_boxplot(
                    user_session_id, df, box_val2
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
                if is_valid_df(df_plot, ['Tag', 'RF']):
                    # --- color controls ---
                    color_dict = color_picker_controls(box_val2)
                    fig = plot_general_boxplot(df_plot, color=color_dict)
                    st.plotly_chart(fig, use_container_width=True)
                    plot_download_link(fig, filename="boxplots.png")

                    stats = st.session_state[user_session_id]["boxplot_stats"]
                    st.markdown("##### Descriptive statistics:")
                    st.dataframe(stats, hide_index=True)
                else:
                    st.info("Please select valid variables and ensure data is available.")

    # Handle Scatterplot selection
    elif plot_type == "Scatterplot" and session.get('has_target')[0] is True:

        update_session('pca', False, user_session_id)
        st.sidebar.markdown("### Tagset")

        df, cats, tag_radio, tag_type = tagset_selection(
            user_session_id=user_session_id,
            session_state=st.session_state,
            persist_func=lambda key, page, user: f"{key}_{page}_{user}",
            page_stem="advanced_plotting",
            tagset_keys={
                "Parts-of-Speech": {"General": "dtm_pos", "Specific": "dtm_pos"},
                "DocuScope": "dtm_ds"
            },
            simplify_funcs={
                "Parts-of-Speech": {"General": ds.dtm_simplify, "Specific": None}
            },
            tag_filters={
                "Parts-of-Speech": {
                    "Specific": lambda df: df.drop([col for col in ["FU"] if col in df.columns]),
                    "General": lambda df: df.drop([col for col in ["Other"] if col in df.columns])
                },
                "DocuScope": lambda df: df.drop([col for col in ["Untagged"] if col in df.columns])
            },
            tag_radio_key="tag_radio",
            tag_type_key="tag_type_radio",
            on_change=clear_plots,
            on_change_args=(user_session_id,)
        )

        # Determine categories for plotting
        if df is None or getattr(df, "height", 0) == 0:
            cats = []
        else:
            cats = sorted([col for col in df.columns if col != "doc_id"])

        by_group_highlight = st.toggle("Highlight groups in scatterplots.")

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
                with st.expander("Scatterplot Variables", expanded=True):

                    st.markdown("### Highlight Groups")
                    all_groups = sorted(set(metadata_target.get('doccats')[0]['cats']))
                    selected_groups = st.segmented_control(
                        "Highlight categories in plot:",
                        options=all_groups,
                        selection_mode="multi",
                        key=f"highlight_scatter_groups_{user_session_id}"
                    )
                    st.markdown("### Variables")
                    xaxis1 = st.segmented_control(
                        "Select variable for the x-axis:",
                        cats,
                        selection_mode="single",
                        key=f"scatter_x_grouped_{user_session_id}",
                        help="Choose a tag for the x-axis.",
                        disabled=not cats
                    )
                    yaxis1 = st.segmented_control(
                        "Select variable for the y-axis:",
                        cats,
                        selection_mode="single",
                        key=f"scatter_y_grouped_{user_session_id}",
                        help="Choose a tag for the y-axis.",
                        disabled=not cats
                    )

                    # Sidebar action button
                    scatterplot_group_btn = st.sidebar.button(
                        label="Scatterplot of Frequencies by Group",
                        key=f"scatterplot_group_btn_{user_session_id}",
                        help="Generate grouped scatterplot for selected variables.",
                        type="secondary",
                        use_container_width=False,
                        icon=":material/manufacturing:"
                    )

                    if scatterplot_group_btn:
                        clear_scatterplot_multiselect(user_session_id)
                        clear_scatterplot_multiselect(user_session_id)
                        # Store the selected variables in session state
                        st.session_state[user_session_id]["scatterplot_group_x"] = xaxis1
                        st.session_state[user_session_id]["scatterplot_group_y"] = yaxis1
                        st.session_state[user_session_id]["scatterplot_group_selected_groups"] = selected_groups  # noqa: E501
                        generate_scatterplot_with_groups(
                            user_session_id, df, xaxis1, yaxis1
                        )

            if st.session_state[user_session_id].get("scatter_group_warning"):
                msg, icon = st.session_state[user_session_id]["scatter_group_warning"]
                st.warning(msg, icon=icon)

            # Plot if available
            if (
                "scatterplot_group_df" in st.session_state[user_session_id] and
                st.session_state[user_session_id]["scatterplot_group_df"] is not None
            ):
                df_plot = st.session_state[user_session_id]["scatterplot_group_df"]
                x_col = st.session_state[user_session_id].get("scatterplot_group_x")
                y_col = st.session_state[user_session_id].get("scatterplot_group_y")
                plot_groups = st.session_state[user_session_id].get("scatterplot_group_selected_groups", [])  # noqa: E501
                if is_valid_df(df_plot, [x_col, y_col]):
                    color_dict = color_picker_controls(
                        ["Highlight", "Non-Highlight"],
                        key_prefix=f"color_picker_scatter_{user_session_id}"
                    )
                    show_trend = st.checkbox(
                        label="Show linear fit (regression line)",
                        value=False
                        )
                    fig = plot_scatter_highlight(
                        df=df_plot,
                        x_col=x_col,
                        y_col=y_col,
                        group_col="Group",
                        selected_groups=plot_groups,
                        color=color_dict,
                        trendline=show_trend)

                    st.plotly_chart(fig, use_container_width=False)
                    plot_download_link(fig, filename="scatterplot_highlight.png")
                    cc_dict = st.session_state[user_session_id]["scatter_group_correlation"]
                    cc_dict = correlation_update(
                        cc_dict,
                        df_plot,
                        x_col,
                        y_col,
                        group_col="Group",
                        highlight_groups=selected_groups
                    )
                    st.info(correlation_info(cc_dict))
                else:
                    st.info("Please select valid variables and ensure data is available.")

        else:
            with st.expander("Scatterplot Variables", expanded=True):
                st.markdown("### Variables")
                xaxis2 = st.segmented_control(
                    "Select variable for the x-axis:",
                    cats,
                    selection_mode="single",
                    key=f"scatter_x_nongrouped_{user_session_id}",
                    help="Choose a tag for the x-axis.",
                    disabled=not cats
                )
                yaxis2 = st.segmented_control(
                    "Select variable for the y-axis:",
                    cats,
                    selection_mode="single",
                    key=f"scatter_y_nongrouped_{user_session_id}",
                    help="Choose a tag for the y-axis.",
                    disabled=not cats
                )

                # Sidebar action button
                scatterplot_btn = st.sidebar.button(
                    label="Scatterplot of Frequencies",
                    key=f"scatterplot_btn_{user_session_id}",
                    help="Generate scatterplot for selected variables.",
                    type="secondary",
                    use_container_width=False,
                    icon=":material/manufacturing:"
                )

                if scatterplot_btn:
                    # Optionally clear previous scatterplot state here
                    clear_scatterplot_multiselect(user_session_id)
                    generate_scatterplot(user_session_id, df, xaxis2, yaxis2)
                    # Store the selected variables in session state
                    st.session_state[user_session_id]["scatterplot_nongrouped_x"] = xaxis2
                    st.session_state[user_session_id]["scatterplot_nongrouped_y"] = yaxis2

            # Only display the plot if it has been generated
            if (
                "scatterplot_df" in st.session_state[user_session_id] and
                st.session_state[user_session_id]["scatterplot_df"] is not None
            ):
                df_plot = st.session_state[user_session_id]["scatterplot_df"]
                x_col = st.session_state[user_session_id].get("scatterplot_nongrouped_x")
                y_col = st.session_state[user_session_id].get("scatterplot_nongrouped_y")
                if is_valid_df(df_plot, [x_col, y_col]):
                    color_dict = color_picker_controls(["All Points"])
                    show_trend = st.checkbox(
                        label="Show linear fit (regression line)",
                        value=False
                        )
                    fig = plot_scatter(
                        df_plot,
                        x_col,
                        y_col,
                        color=color_dict,
                        trendline=show_trend
                        )
                    st.plotly_chart(fig, use_container_width=False)
                    cc_dict = st.session_state[user_session_id]["scatter_correlation"]
                    st.info(correlation_info(cc_dict))
                else:
                    st.info("Please select valid variables and ensure data is available.")

    # Handle PCA selection
    elif plot_type == "PCA" and session.get('has_target')[0] is True:

        st.sidebar.markdown("### Tagset")

        # Radio button to select tag type
        tag_radio_tokens = st.sidebar.radio(
            "Select tags to display:",
            ("Parts-of-Speech", "DocuScope"),
            on_change=clear_plots, args=(user_session_id,),
            horizontal=True
            )

        # Handle Parts-of-Speech tag selection
        if tag_radio_tokens == 'Parts-of-Speech':
            tag_type = st.sidebar.radio(
                "Select from general or specific tags",
                ("General", "Specific"),
                on_change=clear_plots, args=(user_session_id,),
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
        st.sidebar.markdown("### PCA")
        pca_btn = st.sidebar.button(
            label="Principal Component Analysis",
            key=f"pca_btn_{user_session_id}",
            icon=":material/manufacturing:"
        )

        if pca_btn:
            generate_pca(user_session_id, df, metadata_target, session)

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
            if is_valid_df(pca_df, ['PC1', 'PC2']):
                contrib_df = st.session_state[user_session_id]["target"]["contrib_df"]
                ve = metadata_target.get("variance")[0]['temp']

                # Get the current PC index from session state, default to 1
                current_idx = st.session_state[user_session_id].get('pca_idx', 1)
                # --- SHARED PC INDEX LOGIC ---
                tab1, tab2 = st.tabs(["PCA Plot", "Variable Contribution"])

                # --- TAB 1 ---
                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.selectbox(
                            "Select principal component to plot",
                            list(range(1, len(df.columns))),
                            key=f"pca_idx_tab1_{user_session_id}",
                            index=current_idx - 1,
                            on_change=update_pca_idx_tab1,
                            args=(user_session_id,)
                        )
                    with col2:
                        if session.get('has_meta')[0] is True:
                            groups = sorted(set(metadata_target.get('doccats')[0]['cats']))
                            selected_groups = st.multiselect(
                                "Highlight categories in PCA plot:",
                                groups,
                                default=[],
                                key=f"highlight_pca_groups_{user_session_id}"
                            )
                        else:
                            selected_groups = []

                    # Always use the value from session state for plotting
                    if 'pca_idx' not in st.session_state[user_session_id]:
                        st.session_state[user_session_id]['pca_idx'] = 1
                    idx = st.session_state[user_session_id].get('pca_idx', 1)
                    pca_x, pca_y, contrib_x, contrib_y, ve_1, ve_2, contrib_1_plot, contrib_2_plot = update_pca_plot(
                        pca_df,
                        contrib_df,
                        ve,
                        idx
                    )
                    fig = plot_pca_scatter_highlight(
                        pca_df,
                        pca_x,
                        pca_y,
                        'Group',
                        selected_groups,
                        x_label=pca_x,
                        y_label=pca_y
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.info(variance_info(pca_x, pca_y, ve_1, ve_2))

                # --- TAB 2 ---
                with tab2:
                    st.markdown(
                        body="##### Variable contribution (by %) to principal component:",
                        help=(
                            "The plots are a Python implementation of [fviz_contrib()](http://www.sthda.com/english/wiki/fviz-contrib-quick-visualization-of-row-column-contributions-r-software-and-data-mining), "
                            "an **R** function that is part of the **factoextra** package."
                        )
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        st.selectbox(
                            "Select principal component to plot",
                            list(range(1, len(df.columns))),
                            key=f"pca_idx_tab2_{user_session_id}",
                            index=st.session_state[user_session_id].get('pca_idx', 1) - 1,
                            on_change=update_pca_idx_tab2,
                            args=(user_session_id,)
                        )

                    # Always use the value from session state for plotting
                    idx = st.session_state[user_session_id].get('pca_idx', 1)
                    pca_x2, pca_y2, contrib_x2, contrib_y2, ve_1_2, ve_2_2, contrib_1_plot2, contrib_2_plot2 = update_pca_plot(
                        pca_df,
                        contrib_df,
                        ve,
                        idx
                    )
                    with col2:
                        sort_by = st.radio(
                            "Sort variables by:",
                            (pca_x2, pca_y2),
                            index=0,
                            horizontal=True,
                            key=f"sort_by_{user_session_id}"
                        )

                    st.info(contribution_info(pca_x2, pca_y2, contrib_x2, contrib_y2))

                    fig = plot_pca_variable_contrib_bar(
                        contrib_1_plot2, contrib_2_plot2,
                        pc1_label=pca_x2, pc2_label=pca_y2,
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
