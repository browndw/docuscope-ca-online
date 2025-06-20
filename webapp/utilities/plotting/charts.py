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

# Basic chart plotting functions: bar charts, scatter plots, boxplots

import base64
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from webapp.config.session_keys import ScatterplotKeys, BoxplotKeys, PCAKeys
from webapp.utilities.session import update_session


def plot_download_link(
        fig: go.Figure,
        filename="plot.png",
        scale=2,
        button_text="Download high-res PNG"
        ) -> None:
    """
    Display a download link for a high-resolution PNG of a Plotly figure.

    Parameters
    ----------
    fig : plotly.graph_objs.Figure
        The Plotly figure to export.
    filename : str
        The filename for the downloaded PNG.
    scale : int or float
        The scale factor for the image resolution (default 2).
    button_text : str
        The text to display for the download link/button.

    Returns
    -------
    None
        Renders a download link in the Streamlit app.
    """
    # Export the figure to a PNG byte stream
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    img_bytes = fig.to_image(format="png", scale=scale)
    b64 = base64.b64encode(img_bytes).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{button_text}</a>'
    st.markdown(href, unsafe_allow_html=True)


def plot_tag_frequencies_bar(
        df: pl.DataFrame | pd.DataFrame,
        color: str = "#133955"
        ) -> go.Figure:
    """
    Plot a horizontal bar chart of tag frequencies.
    Expects columns: 'Tag' and 'RF' (relative frequency, as percent).
    Optionally specify a color for the bars.
    """
    # Sort tags by frequency descending
    if hasattr(df, 'sort'):
        df_sorted = df.sort('RF', descending=True)
    else:
        df_sorted = df.sort_values('RF', ascending=False)

    # If using polars, convert to pandas for Plotly
    if hasattr(df_sorted, 'to_pandas'):
        df_sorted = df_sorted.to_pandas()

    min_height = 200  # Minimum plot height in pixels
    height = max(24 * len(df_sorted) + 40, min_height)

    fig = px.bar(
        df_sorted,
        x='RF',
        y='Tag',
        orientation='h',
        color_discrete_sequence=[color],
        hover_data={'Tag': True, 'RF': ':.2f'},
        height=height,
    )

    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=40),
        xaxis_title='Frequency (% of tokens)',
        yaxis_title=None,
        yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
    )
    fig.update_traces(
        marker_line_width=0,
        hovertemplate="<b>Tag:</b> %{y}<br><b>RF:</b> %{x:.2f}%<extra></extra>"
    )
    return fig


def plot_compare_corpus_bar(
        df: pl.DataFrame | pd.DataFrame,
        target_color: str = "#133955",
        reference_color: str = "#e67e22",
        color_dict: dict = None
        ) -> go.Figure:
    """
    Plot a horizontal bar chart comparing tag frequencies in two corpus parts.
    Expects columns: 'Tag', 'RF', 'RF_Ref'.
    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        DataFrame containing tag frequencies with columns 'Tag', 'RF', and 'RF_Ref'.
        'RF' is the target corpus frequency, 'RF_Ref' is the reference corpus frequency.
    target_color : str, optional
        Hex color code for the target corpus bars (default "#133955").
    reference_color : str, optional
        Hex color code for the reference corpus bars (default "#e67e22").
    color_dict : dict, optional
        Dictionary from color_picker_controls containing color selections.
        If provided, will override target_color and reference_color parameters.
        Expected keys: "Target Color", "Reference Corpus" (or "Reference Color")
    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    # Parse color_dict if provided
    if color_dict:
        target_color = color_dict.get("Target Color", target_color)
        reference_color = color_dict.get(
            "Reference Corpus",
            color_dict.get("Reference Color", reference_color)
        )
    # Prepare DataFrame
    df_plot = df.to_pandas() if hasattr(df, "to_pandas") else df.copy()
    df_plot = df_plot[["Tag", "RF", "RF_Ref"]].copy()
    df_plot["Mean"] = df_plot[["RF", "RF_Ref"]].mean(axis=1)
    df_plot.rename(
        columns={"RF": "Target", "RF_Ref": "Reference"},
        inplace=True
    )
    df_plot = pd.melt(
        df_plot,
        id_vars=['Tag', 'Mean'],
        value_vars=['Target', 'Reference'],
        var_name='Corpus',
        value_name='RF'
    )
    # Do not sort after this point!

    # Set tag order by descending mean
    tag_order = df_plot.groupby("Tag")["Mean"].mean().sort_values(ascending=False).index.tolist()  # noqa: E501
    corpus_order = ['Reference', 'Target']  # Target will be on top

    height = max(24 * len(tag_order) + 100, 400)

    fig = px.bar(
        df_plot,
        x="RF",
        y="Tag",
        color="Corpus",
        color_discrete_sequence=[reference_color, target_color],
        orientation="h",
        category_orders={"Tag": tag_order, "Corpus": corpus_order},
        hover_data={"Tag": True, "RF": ':.2f', "Corpus": True},
        height=height,
        custom_data=["Corpus"],  # <-- This ensures correct mapping
    )

    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        legend_title_text='',
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title='Frequency (% of tokens)',
        yaxis_title=None,
        bargap=0.1,
        bargroupgap=0.05,
        barmode='group',
    )
    fig.update_traces(
        marker_line_width=0,
        hovertemplate="<b>Tag:</b> %{y}<br><b>Corpus:</b> %{customdata[0]}<br><b>RF:</b> %{x:.2f}%<extra></extra>",  # noqa: E501
    )
    return fig


def plot_general_boxplot(
        df: pl.DataFrame | pd.DataFrame,
        tag_col='Tag',
        value_col='RF',
        color=None,
        palette=None
        ) -> go.Figure:
    """
    General boxplot for the corpus, colored by tag, with legend at bottom left,
    and boxes sorted by median (highest to lowest).
    Allows user to specify a custom HEX color, a dict, or a Plotly palette.
    """
    # Sort tags by median value (descending)
    medians = df.groupby(tag_col)[value_col].median().sort_values(ascending=False)
    tag_order = medians.index.tolist()

    # Color logic
    if isinstance(color, dict):
        color_map = color
    elif isinstance(color, str) and color.lower().startswith("#"):
        color_map = {cat: color for cat in tag_order}
    elif palette:
        palette_colors = palette if isinstance(palette, list) else px.colors.qualitative.Set1  # noqa: E501
        color_map = {cat: palette_colors[i % len(palette_colors)] for i, cat in enumerate(tag_order)}  # noqa: E501
    else:
        palette_colors = px.colors.qualitative.Set1
        color_map = {cat: palette_colors[i % len(palette_colors)] for i, cat in enumerate(tag_order)}  # noqa: E501

    # Compute summary stats for hover
    stats = (
        df.groupby(tag_col)[value_col]
        .agg(['mean', 'median', lambda s: s.quantile(0.75) - s.quantile(0.25), 'min', 'max'])  # noqa: E501
        .rename(columns={'mean': 'Mean', 'median': 'Median', '<lambda_0>': 'IQR', 'min': 'Min', 'max': 'Max'})  # noqa: E501
        .reset_index()
    )

    # Create boxplot
    fig = px.box(
        df,
        x=value_col,
        y=tag_col,
        color=tag_col,
        color_discrete_map=color_map,
        points=False,
        orientation='h',
        category_orders={tag_col: tag_order}
    )

    # Turn off default boxplot hover for all traces
    for trace in fig.data:
        if trace.type == "box":
            trace.hoverinfo = "skip"
            trace.hoveron = "boxes"

    # Overlay transparent bar for custom hover
    bar_df = stats.copy()
    bar_df['bar'] = bar_df['Max'] - bar_df['Min']
    bar_df['base'] = bar_df['Min']
    fig2 = px.bar(
        bar_df,
        y=tag_col,
        x='bar',
        base='base',
        orientation='h',
        color=tag_col,
        color_discrete_map=color_map,
        hover_data={
            'Mean': ':.2f',
            'Median': ':.2f',
            'IQR': ':.2f',
            'Min': ':.2f',
            'Max': ':.2f',
            'base': False,
            'bar': False,
        },
    ).update_traces(opacity=0.01,  # nearly invisible, but hoverable
                    hovertemplate="<b>%{y}</b><br>Min: %{customdata[3]:.2f}%<br>IQR: %{customdata[2]:.2f}%<br>Median: %{customdata[1]:.2f}%<br>Mean: %{customdata[0]:.2f}%<br>Max: %{customdata[4]:.2f}%<extra></extra>")  # noqa: E501

    # Add bar traces to boxplot
    for trace in fig2.data:
        fig.add_trace(trace)

    fig.update_layout(
        hovermode="closest",
        showlegend=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.5,
            xanchor="left",
            x=0
        ),
        legend_title_text='',
        margin=dict(l=0, r=0, t=30, b=0),
        height=100 * len(tag_order) + 120,
        xaxis_title='Frequency (per 100 tokens)',
        yaxis_title="Tag"
    )
    fig.update_yaxes(showticklabels=True, title=None, tickangle=0)
    fig.update_xaxes(title_text='Frequency (per 100 tokens)', range=[0, None])
    # Add vertical line at x=0, colored gray
    fig.add_vline(
        x=0,
        line_width=2,
        line_color="lightgray",
        layer="below"
    )
    return fig


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
    for wkey in widget_keys:
        if wkey in st.session_state:
            del st.session_state[wkey]


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
    for wkey in widget_keys:
        if wkey in st.session_state:
            del st.session_state[wkey]


def clear_plots(session_id: str) -> None:
    """
    Clear all plot-related session state for the given user session.

    This includes group selections, boxplot and scatterplot variables,
    highlight selections, DataFrames, statistics, warnings, PCA data,
    and color picker/segmented control widget states.
    """
    if session_id not in st.session_state:
        return

    update_session('pca', False, session_id)
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
        "color_picker_form_", "seg_", "filter_", "highlight_",
        "toggle_", "download_", "boxplot_vars_"
    ]
    keys_to_remove = [k for k in st.session_state.keys()
                      if any(k.startswith(prefix) for prefix in widget_prefixes)]
    for k in keys_to_remove:
        del st.session_state[k]

    # --- Clear persistent color map for boxplots if present ---
    color_map_key = f"boxplot_color_map_{session_id}"
    if color_map_key in st.session_state:
        del st.session_state[color_map_key]

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
