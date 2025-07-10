"""
Boxplot utilities for corpus analysis visualization.

This module provides functions for preparing data and generating
boxplots for corpus analysis.
"""
import streamlit as st

from webapp.utilities.core import app_core
from webapp.utilities.state import ScatterplotKeys, BoxplotKeys, PCAKeys
from webapp.utilities.state.widget_state import (
    safe_clear_widget_states,
    safe_clear_widget_state
)


def clear_plot_toggle(session_id: str) -> None:
    """
    Clear the plots on group toggle change for the given user session.
    """
    if session_id not in st.session_state:
        return

    app_core.session_manager.update_session_state(session_id, 'pca', False)
    _GRPA = f"grpa_{session_id}"
    _GRPB = f"grpb_{session_id}"
    _BOXPLOT_VARS = f"boxplot_vars_{session_id}"

    # Clear group selections
    for key in [_GRPA, _GRPB, _BOXPLOT_VARS]:
        if key in st.session_state:
            st.session_state[key] = []

    # Clear highlight multiselects
    highlight_keys = [
        f"highlight_pca_groups_{session_id}",
        f"highlight_scatter_groups_{session_id}",
        # add other highlight keys as needed
    ]
    for key in highlight_keys:
        if key in st.session_state:
            st.session_state[key] = []

    # Clear regression checkbox
    regression_keys = [
        f"trend_scatter_groups_{session_id}",
        f"trend_scatter_{session_id}"
    ]
    for key in regression_keys:
        if key in st.session_state:
            st.session_state[key] = False

    # Clear plot results and warnings, and remove 'Highlight' column
    if session_id in st.session_state:
        for key in [
            BoxplotKeys.DF, BoxplotKeys.GROUP_DF,
            ScatterplotKeys.DF, ScatterplotKeys.GROUP_DF
        ]:
            df = st.session_state[session_id].get(key)
            if df is not None and hasattr(df, "columns") and "Highlight" in df.columns:
                st.session_state[session_id][key] = df.drop(columns=["Highlight"])

        keys_to_clear = [
            BoxplotKeys.DF, BoxplotKeys.STATS, BoxplotKeys.WARNING,
            BoxplotKeys.GROUP_DF, BoxplotKeys.GROUP_STATS, BoxplotKeys.GROUP_WARNING,
            BoxplotKeys.CONFIRMED_VAL2, BoxplotKeys.CONFIRMED_VAL1,
            BoxplotKeys.CONFIRMED_GRPA, BoxplotKeys.CONFIRMED_GRPB,
            ScatterplotKeys.DF, ScatterplotKeys.CORRELATION, ScatterplotKeys.WARNING,
            ScatterplotKeys.GROUP_DF, ScatterplotKeys.GROUP_CORRELATION,
            ScatterplotKeys.GROUP_WARNING,
            ScatterplotKeys.GROUP_X, ScatterplotKeys.GROUP_Y,
            ScatterplotKeys.GROUP_SELECTED_GROUPS,
        ]
        for key in keys_to_clear:
            st.session_state[session_id][key] = None

    # --- Clear color picker and segmented control widget states ---
    widget_prefixes = [
        f"color_picker_form_{session_id}", f"seg_{session_id}",
        f"filter_{session_id}", f"highlight_{session_id}",
        f"toggle_{session_id}", f"download_{session_id}",
        f"boxplot_vars_{session_id}", f"color_picker_boxplot_{session_id}",
        f"color_picker_boxplot_general_{session_id}"
    ]
    keys_to_remove = [k for k in st.session_state.keys()
                      if any(k.startswith(prefix) for prefix in widget_prefixes)]

    # Safe deletion to prevent KeyErrors
    safe_clear_widget_states(keys_to_remove)

    # --- Clear attempted flags ---
    for flag in [
        BoxplotKeys.ATTEMPTED,
        BoxplotKeys.GROUP_ATTEMPTED,
        ScatterplotKeys.ATTEMPTED,
        ScatterplotKeys.GROUP_ATTEMPTED,
        PCAKeys.ATTEMPTED
    ]:
        st.session_state[session_id][flag] = False


def clear_plots(session_id: str) -> None:
    """
    Clear all plot-related session state for the given user session.

    This includes group selections, boxplot and scatterplot variables,
    highlight selections, DataFrames, statistics, warnings, PCA data,
    and color picker/segmented control widget states.
    """
    if session_id not in st.session_state:
        return

    app_core.session_manager.update_session_state(session_id, 'pca', False)
    _GRPA = f"grpa_{session_id}"
    _GRPB = f"grpb_{session_id}"
    _BOXPLOT_VARS = f"boxplot_vars_{session_id}"

    # Clear group selections
    for key in [_GRPA, _GRPB, _BOXPLOT_VARS]:
        if key in st.session_state:
            st.session_state[key] = []

    # Clear highlight multiselects
    highlight_keys = [
        f"highlight_pca_groups_{session_id}",
        f"highlight_scatter_groups_{session_id}",
        # add other highlight keys as needed
    ]
    for key in highlight_keys:
        if key in st.session_state:
            st.session_state[key] = []

    # Clear regression checkbox
    regression_keys = [
        f"trend_scatter_groups_{session_id}",
        f"trend_scatter_{session_id}"
    ]
    for key in regression_keys:
        if key in st.session_state:
            st.session_state[key] = False

    # Clear plot results and warnings, and remove 'Highlight' column
    if session_id in st.session_state:
        for key in [
            BoxplotKeys.DF, BoxplotKeys.GROUP_DF,
            ScatterplotKeys.DF, ScatterplotKeys.GROUP_DF
        ]:
            df = st.session_state[session_id].get(key)
            if df is not None and hasattr(df, "columns") and "Highlight" in df.columns:
                st.session_state[session_id][key] = df.drop(columns=["Highlight"])

        keys_to_clear = [
            BoxplotKeys.DF, BoxplotKeys.STATS, BoxplotKeys.WARNING,
            BoxplotKeys.GROUP_DF, BoxplotKeys.GROUP_STATS, BoxplotKeys.GROUP_WARNING,
            BoxplotKeys.CONFIRMED_VAL2, BoxplotKeys.CONFIRMED_VAL1,
            BoxplotKeys.CONFIRMED_GRPA, BoxplotKeys.CONFIRMED_GRPB,
            ScatterplotKeys.DF, ScatterplotKeys.CORRELATION, ScatterplotKeys.WARNING,
            ScatterplotKeys.GROUP_DF, ScatterplotKeys.GROUP_CORRELATION,
            ScatterplotKeys.GROUP_WARNING,
            ScatterplotKeys.GROUP_X, ScatterplotKeys.GROUP_Y,
            ScatterplotKeys.GROUP_SELECTED_GROUPS,
        ]
        for key in keys_to_clear:
            st.session_state[session_id][key] = None

        # --- Clear PCA data and warnings ---
        if "target" in st.session_state[session_id]:
            parent, child = PCAKeys.TARGET_PCA_DF
            st.session_state[session_id][parent][child] = None
            parent, child = PCAKeys.TARGET_CONTRIB_DF
            st.session_state[session_id][parent][child] = None
        st.session_state[session_id][PCAKeys.WARNING] = None
        if "pca_idx" in st.session_state[session_id]:
            st.session_state[session_id]["pca_idx"] = 1

    # --- Clear color picker and segmented control widget states ---
    widget_prefixes = [
        f"color_picker_form_{session_id}", f"seg_{session_id}",
        f"filter_{session_id}", f"highlight_{session_id}",
        f"toggle_{session_id}", f"download_{session_id}",
        f"boxplot_vars_{session_id}", f"color_picker_boxplot_{session_id}",
        f"color_picker_boxplot_general_{session_id}"
    ]
    keys_to_remove = [k for k in st.session_state.keys()
                      if any(k.startswith(prefix) for prefix in widget_prefixes)]
    # Safe deletion to prevent KeyErrors
    safe_clear_widget_states(keys_to_remove)

    # --- Clear persistent color map for boxplots if present ---
    color_map_key = f"color_picker_boxplot_{session_id}"
    safe_clear_widget_state(color_map_key)

    # --- Clear attempted flags ---
    for flag in [
        BoxplotKeys.ATTEMPTED,
        BoxplotKeys.GROUP_ATTEMPTED,
        ScatterplotKeys.ATTEMPTED,
        ScatterplotKeys.GROUP_ATTEMPTED,
        PCAKeys.ATTEMPTED
    ]:
        st.session_state[session_id][flag] = False

    # --- Clear boxplot and scatterplot multiselects ---
    clear_boxplot_multiselect(session_id)
    clear_scatterplot_multiselect(session_id)

    # --- Set toggle states to False ---
    group_boxplot_toggle = f"by_group_boxplot_{session_id}"
    if group_boxplot_toggle in st.session_state:
        st.session_state[group_boxplot_toggle] = False

    group_scatter_toggle = f"by_group_scatter_{session_id}"
    if group_scatter_toggle in st.session_state:
        st.session_state[group_scatter_toggle] = False


def clear_boxplot_multiselect(user_session_id: str) -> None:
    """
    Clear the boxplot multiselects and reset related session state.
    """
    if user_session_id not in st.session_state:
        return

    # Clear boxplot-related session state
    keys = [
        BoxplotKeys.DF,
        BoxplotKeys.STATS,
        BoxplotKeys.WARNING,
        BoxplotKeys.GROUP_DF,
        BoxplotKeys.GROUP_STATS,
        BoxplotKeys.GROUP_WARNING,
        BoxplotKeys.CONFIRMED_VAL1,
        BoxplotKeys.CONFIRMED_VAL2,
        BoxplotKeys.CONFIRMED_GRPA,
        BoxplotKeys.CONFIRMED_GRPB,
        "boxplot_nongrouped_vars",
        "boxplot_grouped_vars"
    ]
    for key in keys:
        st.session_state[user_session_id][key] = None

    # Clear widget keys
    widget_keys = [
        f"boxplot_vars_{user_session_id}",
        f"boxplot_btn_{user_session_id}",
        f"boxplot_group_btn_{user_session_id}",
        f"highlight_boxplot_groups_{user_session_id}",
    ]
    # Safe deletion to prevent KeyErrors
    safe_clear_widget_states(widget_keys)


def clear_scatterplot_multiselect(user_session_id: str) -> None:
    """
    Clear the scatterplot multiselects and reset related session state.
    This function resets the scatterplot variable selections and clears
    any associated DataFrames, statistics, and widget state in the session state.
    """
    if user_session_id not in st.session_state:
        return

    # Clear DataFrames, stats, warnings, and selected variables/groups
    keys = [
        ScatterplotKeys.DF,
        ScatterplotKeys.CORRELATION,
        ScatterplotKeys.WARNING,
        ScatterplotKeys.GROUP_DF,
        ScatterplotKeys.GROUP_CORRELATION,
        ScatterplotKeys.GROUP_WARNING,
        ScatterplotKeys.GROUP_X,
        ScatterplotKeys.GROUP_Y,
        ScatterplotKeys.GROUP_SELECTED_GROUPS,
        "scatterplot_nongrouped_x",
        "scatterplot_nongrouped_y"
    ]
    for key in keys:
        st.session_state[user_session_id][key] = None

    # Also clear widget keys related to scatterplot UI,
    # including all segmented controls and buttons
    widget_keys = [
        f"scatterplot_btn_{user_session_id}",
        f"scatterplot_group_btn_{user_session_id}",
        f"scatter_x_grouped_{user_session_id}",
        f"scatter_y_grouped_{user_session_id}",
        f"scatter_x_nongrouped_{user_session_id}",
        f"scatter_y_nongrouped_{user_session_id}",
        f"highlight_scatter_groups_{user_session_id}",
        f"color_picker_scatter_{user_session_id}_Highlight_0",
        f"color_picker_scatter_{user_session_id}_Non-Highlight_1",
        f"color_picker_scatter_{user_session_id}_All_Points_0",
    ]
    # Safe deletion to prevent KeyErrors
    safe_clear_widget_states(widget_keys)


def update_pca_idx_tab1(session_id: str) -> None:
    """
    Update the PCA index for tab 1.
    This function initializes the selectbox state for PCA index in tab 1
    if it doesn't exist, and updates the shared PCA index in the session state.
    """
    # Initialize the selectbox state if it doesn't exist
    if f"pca_idx_tab1_{session_id}" not in st.session_state:
        st.session_state[f"pca_idx_tab1_{session_id}"] = (
            st.session_state[session_id].get('pca_idx', 1)
        )
    # Now update the shared PC index
    st.session_state[session_id]['pca_idx'] = st.session_state[f"pca_idx_tab1_{session_id}"]


def update_pca_idx_tab2(session_id: str) -> None:
    """
    Update the PCA index for tab 2.
    """
    # Initialize the selectbox state if it doesn't exist
    if f"pca_idx_tab2_{session_id}" not in st.session_state:
        st.session_state[f"pca_idx_tab2_{session_id}"] = (
            st.session_state[session_id].get('pca_idx', 1)
        )
    # Now update the shared PC index
    st.session_state[session_id]['pca_idx'] = st.session_state[f"pca_idx_tab2_{session_id}"]


def update_grpa(session_id: str) -> None:
    """
    Prevent categories from being chosen in both multiselects for group A.
    This function checks if the selected items in group A and group B
    overlap, and if so, removes the overlapping items from group A.
    """
    _GRPA = f"grpa_{session_id}"
    _GRPB = f"grpb_{session_id}"
    if _GRPA not in st.session_state.keys():
        st.session_state[_GRPA] = []
    if _GRPB not in st.session_state.keys():
        st.session_state[_GRPB] = []
    if len(
        list(set(st.session_state[_GRPA]) &
             set(st.session_state[_GRPB]))
    ) > 0:
        item = list(
            set(st.session_state[_GRPA]) &
            set(st.session_state[_GRPB])
            )
        st.session_state[_GRPA] = list(
            set(list(st.session_state[_GRPA])) ^ set(item)
            )


def update_grpb(session_id: str) -> None:
    """
    Prevent categories from being chosen in both multiselects for group B.
    This function checks if the selected items in group A and group B
    overlap, and if so, removes the overlapping items from group B.
    """
    _GRPA = f"grpa_{session_id}"
    _GRPB = f"grpb_{session_id}"
    if _GRPA not in st.session_state.keys():
        st.session_state[_GRPA] = []
    if _GRPB not in st.session_state.keys():
        st.session_state[_GRPB] = []
    if len(
        list(set(st.session_state[_GRPA]) &
             set(st.session_state[_GRPB]))
    ) > 0:
        item = list(
            set(st.session_state[_GRPA]) &
            set(st.session_state[_GRPB])
            )
        st.session_state[_GRPB] = list(
            set(list(st.session_state[_GRPB])) ^ set(item)
            )


def update_tar(session_id: str) -> None:
    """
    Prevent categories from being chosen in both target and reference multiselects.
    This function checks if the selected items in target and reference
    overlap, and if so, removes the overlapping items from target.
    """
    _TAR = f"tar_{session_id}"
    _REF = f"ref_{session_id}"
    if _TAR not in st.session_state.keys():
        st.session_state[_TAR] = []
    if _REF not in st.session_state.keys():
        st.session_state[_REF] = []
    if len(
        list(set(st.session_state[_TAR]) &
             set(st.session_state[_REF]))
    ) > 0:
        item = list(
            set(st.session_state[_TAR]) &
            set(st.session_state[_REF])
            )
        st.session_state[_TAR] = list(
            set(list(st.session_state[_TAR])) ^ set(item)
            )


def update_ref(session_id: str) -> None:
    """
    Prevent categories from being chosen in both target and reference multiselects.
    This function checks if the selected items in target and reference
    overlap, and if so, removes the overlapping items from reference.
    """
    _REF = f"ref_{session_id}"
    _TAR = f"tar_{session_id}"
    if _TAR not in st.session_state.keys():
        st.session_state[_TAR] = []
    if _REF not in st.session_state.keys():
        st.session_state[_REF] = []
    if len(
        list(set(st.session_state[_TAR]) &
             set(st.session_state[_REF]))
    ) > 0:
        item = list(
            set(st.session_state[_TAR]) &
            set(st.session_state[_REF])
            )
        st.session_state[_REF] = list(
            set(list(st.session_state[_REF])) ^ set(item)
            )
