"""
This app provides an interface for advanced plotting of tag frequencies,
scatterplots, and PCA (Principal Component Analysis) for a loaded target corpus.

Users can:
- Generate boxplots of tag frequencies, either grouped by metadata variables or not.
- Create scatterplots of tag frequencies, with options to highlight groups.
- Perform PCA on tag frequencies and visualize the results.
"""

import re
import docuscospacy as ds
import streamlit as st

# Core application utilities with standardized patterns
from webapp.utilities.core import app_core

# UI error boundaries (imported directly to avoid None fallback)
from webapp.utilities.ui.error_boundaries import SafeComponentRenderer

# Module-specific imports
from webapp.utilities.session import (
    get_or_init_user_session, load_metadata, safe_session_get
    )
from webapp.utilities.plotting import (
    generate_boxplot, generate_boxplot_by_group,
    clear_plots, clear_scatterplot_multiselect,
    update_grpa, update_grpb,
    update_pca_idx_tab1, update_pca_idx_tab2,
    plot_download_link, plot_general_boxplot,
    plot_grouped_boxplot, plot_pca_scatter_highlight,
    plot_pca_variable_contrib_bar, plot_scatter,
    plot_scatter_highlight, clear_plot_toggle
)
from webapp.utilities.analysis import (
    generate_pca, generate_scatterplot,
    generate_scatterplot_with_groups, is_valid_df,
    correlation_update, update_pca_plot
)
from webapp.utilities.ui import (
    color_picker_controls, contribution_info,
    correlation_info, plot_action_button,
    show_plot_warning, tagset_selection,
    variance_info
)
from webapp.menu import (
    menu, require_login
    )
from webapp.utilities.state import (
    BoxplotKeys, ScatterplotKeys,
    PCAKeys, SessionKeys,
    WarningKeys, CorpusKeys, TargetKeys,
    MetadataKeys
    )
from webapp.utilities.state.widget_key_manager import create_persist_function

# Register persistent widgets for this page
ADVANCED_PLOTTING_PERSISTENT_WIDGETS = [
    "tag_radio",                      # Radio button for tag selection
    "tag_type_radio",                 # Radio button for tag type selection
    "by_group_boxplot",              # Toggle for grouped boxplots
    "by_group_scatter",              # Toggle for grouped scatterplots
    "highlight_scatter_groups",       # Multiselect for scatter groups
]
app_core.register_page_widgets(ADVANCED_PLOTTING_PERSISTENT_WIDGETS)

TITLE = "Advanced Plotting"
ICON = ":material/line_axis:"

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )


def render_boxplot_interface(
    user_session_id: str,
    session: dict,
    metadata_target: dict
) -> None:
    """Render the boxplot interface and handle boxplot generation."""
    # Validate inputs
    if not user_session_id or not isinstance(session, dict):
        st.error("Invalid session parameters.")
        return

    app_core.session_manager.update_session_state(user_session_id, SessionKeys.PCA, False)
    st.sidebar.markdown("### Tagset")

    # Radio button to select tag type
    df, cats, tag_radio, tag_type = tagset_selection(
        user_session_id=user_session_id,
        session_state=st.session_state,
        persist_func=create_persist_function(user_session_id),
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

    st.sidebar.markdown("""---""")

    # Toggle to plot using grouping variables
    by_group = st.toggle(
        label="Plot using grouping variables.",
        key=f"by_group_boxplot_{user_session_id}",
        help=(
            "If you have processed metadata for your corpus, "
            "you can select grouping variables to plot tag frequencies by group."
        ),
        on_change=clear_plot_toggle, args=(user_session_id,)
        )

    # Determine categories for plotting
    if df is None or getattr(df, "height", 0) == 0:
        cats = []
    else:
        cats = sorted([col for col in df.columns if col != "doc_id"])

    # Handle plotting with grouping variables
    if by_group:
        if not safe_session_get(session, SessionKeys.HAS_META, False):
            st.warning(
                """
                It doesn't look like you've processed any metadata yet.
                You can do this from **Manage Corpus Data**.
                """,
                icon=":material/warning:"
            )
        else:
            with st.expander(
                label="Boxplot Variables",
                icon=":material/settings:",
                expanded=True
            ):
                st.markdown(
                    body="### Grouping variables",
                    help=(
                        "Select one or more variables to plot as boxplots. "
                        "You can select multiple variables for comparison. "
                        "Note that plots will only update after "
                        "you click the **Generate Boxplots** button."
                    ))
                st.markdown(
                    body=(
                        "Select grouping variables from your metadata "
                        "and click the button to generate boxplots of frequencies."
                    )
                )
                doccats_data = metadata_target.get(MetadataKeys.DOCCATS, [{}])[0]
                all_cats = sorted(set(doccats_data.get('cats', [])))

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

                st.markdown(
                    body="### Variables"
                )
                box_val1 = st.segmented_control(
                    "Select variables for plotting:",
                    cats,
                    selection_mode="multi",
                    key=f"boxplot_vars_grouped_{user_session_id}",
                    help="Choose one or more tags to plot as boxplots."
                )

            st.sidebar.markdown(
                body="### Boxplots of frequencies by group",
                help=(
                    "Click the button to generate boxplots of tag frequencies "
                    "for the selected variables and groups."
                    "Be sure to select at least one variable "
                    "and one category from group A and one from group B."
                )
            )

            st.sidebar.markdown(
                body="Use the button to generate  grouped boxplots "
                "for the selected variables.",
                )

            boxplot_group_btn = plot_action_button(
                label="Generate Boxplots",
                key=f"boxplot_group_btn_{user_session_id}",
                help_text="Generate grouped boxplots for selected variables.",
                user_session_id=user_session_id,
                attempted_flag=BoxplotKeys.GROUP_ATTEMPTED
            )
            st.sidebar.markdown("---")

            # Only update the confirmed selection when the button is pressed
            if boxplot_group_btn:
                try:
                    st.session_state[user_session_id][BoxplotKeys.CONFIRMED_VAL1] = box_val1
                    st.session_state[user_session_id][BoxplotKeys.CONFIRMED_GRPA] = grpa
                    st.session_state[user_session_id][BoxplotKeys.CONFIRMED_GRPB] = grpb
                    generate_boxplot_by_group(user_session_id, df, box_val1, grpa, grpb)
                except Exception:
                    st.error(
                        body=(":material/error: Error generating grouped boxplot. "
                              "Please check your selections.")
                            )
            if show_plot_warning(
                st.session_state,
                user_session_id,
                WarningKeys.BOX_GROUP,
                BoxplotKeys.GROUP_ATTEMPTED,
                [BoxplotKeys.GROUP_DF, BoxplotKeys.GROUP_STATS]
            ):
                return

            if (
                BoxplotKeys.GROUP_DF in st.session_state[user_session_id] and
                is_valid_df(st.session_state[user_session_id][BoxplotKeys.GROUP_DF], ['Group', 'Tag'])  # noqa: E501
            ):
                df_plot = st.session_state[user_session_id][BoxplotKeys.GROUP_DF]
                # Use the confirmed selection for color controls and plotting
                confirmed_box_val1 = st.session_state[user_session_id].get(BoxplotKeys.CONFIRMED_VAL1, [])  # noqa: E501, F841
                confirmed_grpa = st.session_state[user_session_id].get(BoxplotKeys.CONFIRMED_GRPA, [])  # noqa: E501
                confirmed_grpb = st.session_state[user_session_id].get(BoxplotKeys.CONFIRMED_GRPB, [])  # noqa: E501
                color_dict = color_picker_controls(
                    [", ".join(confirmed_grpa), ", ".join(confirmed_grpb)],
                    key_prefix=f"color_picker_boxplot_{user_session_id}"
                )
                fig = plot_grouped_boxplot(df_plot, color=color_dict)
                SafeComponentRenderer.safe_plotly_chart(fig, use_container_width=True)
                plot_download_link(fig, filename="grouped_boxplots.png")

                stats = st.session_state[user_session_id][BoxplotKeys.GROUP_STATS]
                st.markdown("##### Descriptive statistics:")
                st.dataframe(stats, hide_index=True)
            else:
                if st.session_state[user_session_id].get(BoxplotKeys.GROUP_ATTEMPTED):
                    st.warning(
                        body=(
                            ":material/error: No valid data available for plotting. "
                            "Please ensure you have selected valid variables."
                        )
                    )

    # Handle plotting without grouping variables
    else:
        with st.expander(
            label="Boxplot Variables",
            icon=":material/settings:",
            expanded=True
        ):
            st.markdown("### Variables")
            box_val2 = st.segmented_control(
                "Select variables for plotting:",
                cats,
                selection_mode="multi",
                key=f"boxplot_vars_nongrouped_{user_session_id}",
                help="Choose one or more tags to plot as boxplots."
            )

        # Sidebar action button
        st.sidebar.markdown(
            body="### Boxplots of frequencies",
            help=(
                "Click the button to generate boxplots of tag frequencies "
                "for the selected variables."
                )
            )
        st.sidebar.markdown(
            body="Use the button to generate boxplots for the selected variables. ",
            )
        boxplot_btn = plot_action_button(
            label="Generate Boxplots",
            key=f"boxplot_btn_{user_session_id}",
            help_text="Generate boxplots for selected variables.",
            user_session_id=user_session_id,
            attempted_flag=BoxplotKeys.ATTEMPTED
        )
        st.sidebar.markdown("---")

        # Only update the confirmed selection when the button is pressed
        if boxplot_btn:
            st.session_state[user_session_id][BoxplotKeys.CONFIRMED_VAL2] = box_val2
            try:
                generate_boxplot(user_session_id, df, box_val2)
            except Exception:
                st.error(
                    body=(":material/error: Error generating boxplot. "
                          "Please check your selections.")
                    )

        if show_plot_warning(
            st.session_state,
            user_session_id,
            WarningKeys.BOX,
            BoxplotKeys.ATTEMPTED,
            [BoxplotKeys.DF, BoxplotKeys.STATS]
        ):
            return

        # Plot if available
        if (
            BoxplotKeys.DF in st.session_state[user_session_id] and
            is_valid_df(st.session_state[user_session_id][BoxplotKeys.DF], ['Tag', 'RF'])
        ):
            df_plot = st.session_state[user_session_id][BoxplotKeys.DF]
            # Use the confirmed selection for color controls and plotting
            confirmed_box_val2 = st.session_state[user_session_id].get(BoxplotKeys.CONFIRMED_VAL2, [])  # noqa: E501
            color_dict = color_picker_controls(
                confirmed_box_val2,
                key_prefix=f"color_picker_boxplot_general_{user_session_id}"
            )
            fig = plot_general_boxplot(df_plot, color=color_dict)
            SafeComponentRenderer.safe_plotly_chart(fig, use_container_width=True)
            plot_download_link(fig, filename="boxplots.png")

            stats = st.session_state[user_session_id][BoxplotKeys.STATS]
            st.markdown("##### Descriptive statistics:")
            st.dataframe(stats, hide_index=True)
        else:
            if st.session_state[user_session_id].get(BoxplotKeys.ATTEMPTED):
                st.warning(
                    body=(
                        ":material/error: No valid data available for plotting. "
                        "Please ensure you have selected valid variables."
                    )
                )


def render_scatterplot_interface(
    user_session_id: str,
    session: dict,
    metadata_target: dict
) -> None:
    """Render the scatterplot interface and handle scatterplot generation."""
    # Validate inputs
    if not user_session_id or not isinstance(session, dict):
        st.error("Invalid session parameters.")
        return

    app_core.session_manager.update_session_state(user_session_id, SessionKeys.PCA, False)
    st.sidebar.markdown("### Tagset")

    df, cats, tag_radio, tag_type = tagset_selection(
        user_session_id=user_session_id,
        session_state=st.session_state,
        persist_func=create_persist_function(user_session_id),
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
    st.sidebar.markdown("""---""")

    # Determine categories for plotting
    if df is None or getattr(df, "height", 0) == 0:
        cats = []
    else:
        cats = sorted([col for col in df.columns if col != "doc_id"])

    by_group_highlight = st.toggle(
        label="Highlight groups in scatterplots.",
        key=f"by_group_scatter_{user_session_id}",
        help=(
            "If you have processed metadata for your corpus, "
            "you can select groups to highlight in scatterplots."
        ),
        on_change=clear_plot_toggle, args=(user_session_id,)
        )

    if by_group_highlight:
        if not safe_session_get(session, SessionKeys.HAS_META, False):
            st.warning(
                """
                It doesn't look like you've processed any metadata yet.
                You can do this from **Manage Corpus Data**.
                """,
                icon=":material/warning:"
            )
        else:
            with st.expander(
                label="Scatterplot Variables",
                icon=":material/settings:",
                expanded=True
            ):

                st.markdown("### Highlight Groups")
                doccats_list = metadata_target.get(MetadataKeys.DOCCATS, [{}])
                doccats_data = doccats_list[0].get('cats', [])
                all_groups = sorted(set(doccats_data))
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
                st.sidebar.markdown(
                    body="### Scatterplot of frequencies by group",
                    help=(
                        "Click the button to generate scatterplots of tag frequencies "
                        "for the selected variables and groups."
                        "Be sure to select at least one variable and one group."
                    )
                )

                st.sidebar.markdown(
                    body="Use the button to generate grouped scatterplots "
                    "for the selected variables.",
                    )

                scatterplot_group_btn = plot_action_button(
                    label="Generate Scatterplot",
                    key=f"scatterplot_group_btn_{user_session_id}",
                    help_text="Generate grouped scatterplot for selected variables.",
                    user_session_id=user_session_id,
                    attempted_flag=ScatterplotKeys.GROUP_ATTEMPTED
                )
                st.sidebar.markdown("---")

                if scatterplot_group_btn:
                    clear_scatterplot_multiselect(user_session_id)
                    try:
                        generate_scatterplot_with_groups(
                            user_session_id, df, xaxis1, yaxis1, metadata_target, session
                            )
                        # Store the selected variables in session state
                        st.session_state[user_session_id][ScatterplotKeys.GROUP_X] = xaxis1  # noqa: E501
                        st.session_state[user_session_id][ScatterplotKeys.GROUP_Y] = yaxis1  # noqa: E501
                        st.session_state[user_session_id][ScatterplotKeys.GROUP_SELECTED_GROUPS] = selected_groups  # noqa: E501
                    except Exception:
                        st.error(
                            body=(":material/error: Error generating scatterplot. "
                                  "Please check your selections.")
                            )

        if show_plot_warning(
            st.session_state,
            user_session_id,
            ScatterplotKeys.GROUP_WARNING,
            ScatterplotKeys.GROUP_ATTEMPTED,
            [ScatterplotKeys.GROUP_DF, ScatterplotKeys.GROUP_CORRELATION]
        ):
            return

        # Plot if available
        x_col = st.session_state[user_session_id].get(ScatterplotKeys.GROUP_X)
        y_col = st.session_state[user_session_id].get(ScatterplotKeys.GROUP_Y)
        plot_groups = st.session_state[user_session_id].get(ScatterplotKeys.GROUP_SELECTED_GROUPS, [])  # noqa: E501
        if (
            ScatterplotKeys.GROUP_DF in st.session_state[user_session_id] and
            is_valid_df(st.session_state[user_session_id][ScatterplotKeys.GROUP_DF], ["Group", x_col, y_col])  # noqa: E501
        ):
            df_plot = st.session_state[user_session_id][ScatterplotKeys.GROUP_DF]
            color_dict = color_picker_controls(
                ["Highlight", "Non-Highlight"],
                key_prefix=f"color_picker_scatter_{user_session_id}"
            )
            show_trend = st.checkbox(
                label="Show linear fit (regression line)",
                key=f"trend_scatter_groups_{user_session_id}",
                value=False
            )
            fig = plot_scatter_highlight(
                df=df_plot,
                x_col=x_col,
                y_col=y_col,
                group_col="Group",
                selected_groups=plot_groups,
                color=color_dict,
                trendline=show_trend
            )
            SafeComponentRenderer.safe_plotly_chart(fig, use_container_width=False)
            plot_download_link(fig, filename="scatterplot_highlight.png")
            cc_dict = st.session_state[user_session_id][ScatterplotKeys.GROUP_CORRELATION]
            cc_dict = correlation_update(
                cc_dict,
                df_plot,
                x_col,
                y_col,
                group_col="Group",
                highlight_groups=plot_groups
            )
            st.info(correlation_info(cc_dict))
        else:
            if st.session_state[user_session_id].get(ScatterplotKeys.GROUP_ATTEMPTED):
                st.warning(
                    body=(
                        ":material/error: No valid data available for plotting. "
                        "Please ensure you have selected valid variables."
                    )
                )

    else:
        with st.expander(
            label="Scatterplot Variables",
            icon=":material/settings:",
            expanded=True
        ):
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
            st.sidebar.markdown(
                body="### Scatterplot of frequencies",
                help=(
                    "Click the button to generate scatterplots of tag frequencies "
                    "for the selected variables."
                    )
                )

            st.sidebar.markdown(
                body="Use the button to generate scatterplots for the selected variables.",
                )

            scatterplot_btn = plot_action_button(
                label="Generate Scatterplot",
                key=f"scatterplot_btn_{user_session_id}",
                help_text="Generate scatterplot for selected variables.",
                user_session_id=user_session_id,
                attempted_flag=ScatterplotKeys.ATTEMPTED
            )
            st.sidebar.markdown("---")

            if scatterplot_btn:
                try:
                    clear_scatterplot_multiselect(user_session_id)
                    generate_scatterplot(user_session_id, df, xaxis2, yaxis2)

                    # Store the selected variables in session state
                    st.session_state[user_session_id]["scatterplot_nongrouped_x"] = xaxis2  # noqa: E501
                    st.session_state[user_session_id]["scatterplot_nongrouped_y"] = yaxis2  # noqa: E501
                except Exception:
                    st.error(
                        body=(":material/error: Error generating scatterplot. "
                              "Please check your selections.")
                        )

        if show_plot_warning(
            st.session_state,
            user_session_id,
            ScatterplotKeys.WARNING,
            ScatterplotKeys.ATTEMPTED,
            [ScatterplotKeys.DF, ScatterplotKeys.CORRELATION]
        ):
            return

        # Only display the plot if it has been generated
        x_col = st.session_state[user_session_id].get("scatterplot_nongrouped_x")
        y_col = st.session_state[user_session_id].get("scatterplot_nongrouped_y")

        if (
            ScatterplotKeys.DF in st.session_state[user_session_id] and
            is_valid_df(st.session_state[user_session_id][ScatterplotKeys.DF], [x_col, y_col])  # noqa: E501
        ):
            df_plot = st.session_state[user_session_id][ScatterplotKeys.DF]
            color_dict = color_picker_controls(
                ["All Points"],
                key_prefix=f"color_picker_scatter_all_{user_session_id}"
            )
            show_trend = st.checkbox(
                label="Show linear fit (regression line)",
                key=f"trend_scatter_{user_session_id}",
                value=False
                )
            fig = plot_scatter(
                df_plot,
                x_col,
                y_col,
                color=color_dict,
                trendline=show_trend
                )
            SafeComponentRenderer.safe_plotly_chart(fig, use_container_width=False)
            plot_download_link(fig, filename="scatterplot.png")
            cc_dict = st.session_state[user_session_id][ScatterplotKeys.CORRELATION]
            st.info(correlation_info(cc_dict))
        else:
            if st.session_state[user_session_id].get(ScatterplotKeys.ATTEMPTED):
                st.warning(
                    body=(
                        ":material/error: No valid data available for plotting. "
                        "Please ensure you have selected valid variables."
                    )
                )


def render_pca_interface(
    user_session_id: str,
    session: dict,
    metadata_target: dict
) -> None:
    """Render the PCA interface and handle PCA generation."""
    # Validate inputs
    if not user_session_id or not isinstance(session, dict):
        st.error("Invalid session parameters.")
        return

    app_core.session_manager.update_session_state(user_session_id, SessionKeys.PCA, True)

    df, cats, tag_radio, tag_type = tagset_selection(
        user_session_id=user_session_id,
        session_state=st.session_state,
        persist_func=create_persist_function(user_session_id),
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

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        body="### Principal Component Analysis",
        help=(
            "Click the button to generate a PCA plot of scaled tag frequencies. "
            "Once generated, you can select principal components "
            "to visualize the PCA results and variable contributions."
        )
    )

    st.sidebar.markdown(
        body=(
            "Use the button to generate scatterplots of principal components "
            "for the selected tagset."
        )
    )
    pca_btn = plot_action_button(
        label="Generate PCA",
        key=f"pca_btn_{user_session_id}",
        help_text="Generate PCA plot for the selected variables.",
        user_session_id=user_session_id,
        attempted_flag=PCAKeys.ATTEMPTED
    )
    st.sidebar.markdown("---")

    if pca_btn:
        try:
            generate_pca(user_session_id, df, metadata_target, session)
        except Exception:
            st.error(
                body=(":material/error: Error generating PCA. "
                      "Please check your selections.")
                )

    if show_plot_warning(
        st.session_state,
        user_session_id,
        PCAKeys.WARNING,
        PCAKeys.ATTEMPTED,
        [PCAKeys.TARGET_PCA_DF, PCAKeys.TARGET_CONTRIB_DF]
    ):
        return

    # Plot PCA results if PCA has been performed
    if (
        safe_session_get(session, SessionKeys.PCA, None) is True and
        "pca_df" in st.session_state[user_session_id][CorpusKeys.TARGET] and
        is_valid_df(st.session_state[user_session_id][CorpusKeys.TARGET][TargetKeys.PCA_DF], ['PC1', 'PC2'])  # noqa: E501
    ):
        pca_df = st.session_state[user_session_id][CorpusKeys.TARGET][TargetKeys.PCA_DF]
        pc_cols = [col for col in pca_df.columns if re.match(r"PC\d+$", col)]
        # Limit to mathematically valid number of PCs
        n_variables = len(pc_cols)
        max_valid_pcs = n_variables - 1 if n_variables > 1 else 1
        pc_cols = pc_cols[:max_valid_pcs]

        target_data = st.session_state[user_session_id][CorpusKeys.TARGET]
        contrib_df = target_data[TargetKeys.CONTRIB_DF]
        variance_data = metadata_target.get(MetadataKeys.VARIANCE, [{}])
        ve = (variance_data[0].get('temp', {})
              if variance_data and isinstance(variance_data[0], dict)
              else {})

        # Get the current PC index from session state, default to 1 (1-based)
        current_idx = st.session_state[user_session_id].get('pca_idx', 1)
        if not (1 <= current_idx <= len(pc_cols)):
            current_idx = 1

        tab1, tab2 = st.tabs([
            ":material/scatter_plot: PCA Plot",
            ":material/bar_chart: Variable Contribution"
            ])

        # --- TAB 1 ---
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                selected_idx = st.selectbox(
                    "Select principal component to plot",
                    list(range(1, len(pc_cols) + 1)),
                    key=f"pca_idx_tab1_{user_session_id}",
                    index=current_idx - 1,
                    on_change=update_pca_idx_tab1,
                    args=(user_session_id,)
                )
                st.session_state[user_session_id]['pca_idx'] = selected_idx
            with col2:
                if safe_session_get(session, SessionKeys.HAS_META, None) is True:
                    doccats_list = metadata_target.get(MetadataKeys.DOCCATS, [{}])
                    doccats_data = doccats_list[0].get('cats', [])
                    groups = sorted(set(doccats_data))
                    selected_groups = st.multiselect(
                        "Highlight categories in PCA plot:",
                        groups,
                        default=[],
                        key=f"highlight_pca_groups_{user_session_id}"
                    )
                else:
                    selected_groups = []

            idx = st.session_state[user_session_id].get('pca_idx', 1)
            # Use 0-based index for pc_cols
            pca_x = pc_cols[idx - 1]
            pca_y = pc_cols[1] if len(pc_cols) > 1 else pc_cols[0]
            pca_x, pca_y, contrib_x, contrib_y, ve_1, ve_2, contrib_1_plot, contrib_2_plot = update_pca_plot(  # noqa: E501
                pca_df,
                contrib_df,
                ve,
                idx  # update_pca_plot expects 1-based index
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
            SafeComponentRenderer.safe_plotly_chart(fig, use_container_width=False)
            plot_download_link(fig, filename="pca_scatter.png")
            st.info(variance_info(pca_x, pca_y, ve_1, ve_2))
        # --- TAB 2 ---
        with tab2:
            st.markdown(
                body="##### Variable contribution (by %) to principal component:",
                help=(
                    "The plots are a Python implementation of [fviz_contrib()](http://www.sthda.com/english/wiki/fviz-contrib-quick-visualization-of-row-column-contributions-r-software-and-data-mining), "  # noqa: E501
                    "an **R** function that is part of the **factoextra** package."
                )
            )
            col1, col2 = st.columns(2)
            with col1:
                selected_idx2 = st.selectbox(
                    "Select principal component to plot",
                    list(range(1, len(pc_cols) + 1)),
                    key=f"pca_idx_tab2_{user_session_id}",
                    index=current_idx - 1,
                    on_change=update_pca_idx_tab2,
                    args=(user_session_id,)
                )
                st.session_state[user_session_id]['pca_idx'] = selected_idx2

            idx2 = st.session_state[user_session_id].get('pca_idx', 1)
            pca_x2 = pc_cols[idx2 - 1]
            pca_y2 = pc_cols[1] if len(pc_cols) > 1 else pc_cols[0]
            pca_x2, pca_y2, contrib_x2, contrib_y2, ve_1_2, ve_2_2, contrib_1_plot2, contrib_2_plot2 = update_pca_plot(  # noqa: E501
                pca_df,
                contrib_df,
                ve,
                idx2
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
            SafeComponentRenderer.safe_plotly_chart(fig, use_container_width=True)
            plot_download_link(fig, filename="pca_variable_contrib_bar.png")

    else:
        if st.session_state[user_session_id].get(PCAKeys.ATTEMPTED):
            st.warning(
                body=(
                    ":material/error: No valid data available for plotting. "
                    "Please ensure you have selected valid variables."
                )
            )


def main() -> None:
    """
    Main function to run the advanced plotting page.
    This function sets up the Streamlit page, handles user sessions,
    and provides options for generating various plots based on user input.
    """
    # Set login requirements for navigaton
    require_login()
    menu()
    st.markdown(
        body=f"## {TITLE}",
        help=(
            "This page allows you to create advanced plots of tag frequencies from your corpus. "  # noqa: E501
            "You can generate boxplots, scatterplots, and perform principal component analysis (PCA) on the tag frequencies. "   # noqa: E501
            "Use the **Manage Corpus Data** button from the **Navigation** menu "
            "to process metadata for your corpus, "
            "which will enable grouping variables in your plots."
        ))
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
            CorpusKeys.TARGET,
            user_session_id
            )
    except Exception:
        pass

    # Display a markdown message for plotting
    st.markdown(
        body=(
            ":material/manufacturing: This page allows you to create advanced plots of tag frequencies "  # noqa: E501
            "from your corpus.\n\n"
            ":material/priority: You can also highlight groups in scatterplots and PCA plots if you "  # noqa: E501
            "have processed metadata for your corpus."
        )
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

    st.markdown("---")

    # Handle Boxplot selection
    if plot_type == "Boxplot" and safe_session_get(session, SessionKeys.HAS_TARGET, None) is True:  # noqa: E501

        app_core.session_manager.update_session_state(user_session_id, SessionKeys.PCA, False)  # noqa: E501
        render_boxplot_interface(user_session_id, session, metadata_target)

    # Handle Scatterplot selection
    elif plot_type == "Scatterplot" and safe_session_get(session, SessionKeys.HAS_TARGET, None) is True:  # noqa: E501

        app_core.session_manager.update_session_state(user_session_id, SessionKeys.PCA, False)  # noqa: E501
        render_scatterplot_interface(user_session_id, session, metadata_target)

    # Handle PCA selection
    elif plot_type == "PCA" and safe_session_get(session, SessionKeys.HAS_TARGET, None) is True:  # noqa: E501

        render_pca_interface(user_session_id, session, metadata_target)

    elif not safe_session_get(session, SessionKeys.HAS_TARGET, False):
        st.sidebar.warning(
            body=(
                "Please load a target corpus first."
            ),
            icon=":material/warning:"
        )


if __name__ == "__main__":
    main()
