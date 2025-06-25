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

import math
import docuscospacy as ds
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from scipy.stats import pearsonr

# Module-specific imports
from webapp.utilities.core import app_core
from webapp.utilities.state import ScatterplotKeys, SessionKeys
from webapp.utilities.memory import optimize_dataframe_memory


def plot_scatter(
        df: pl.DataFrame | pd.DataFrame,
        x_col: str,
        y_col: str,
        color=None,
        trendline: bool = False
        ) -> go.Figure:
    """
    Simple scatterplot for two variables, with optional color support and trendline.
    """
    # Convert to pandas only once if needed for plotly
    if hasattr(df, "to_pandas"):
        df_pd = df.to_pandas()
    else:
        df_pd = df

    x_label = x_col + ' (per 100 tokens)'
    y_label = y_col + ' (per 100 tokens)'

    # Determine color
    if isinstance(color, dict):
        color_val = list(color.values())[0] if color else '#133955'
    elif isinstance(color, str) and color.lower().startswith("#"):
        color_val = color
    else:
        color_val = '#133955'

    # Axis scaling and ticks
    x_max = df_pd[x_col].max() if not df_pd.empty else 1
    y_max = df_pd[y_col].max() if not df_pd.empty else 1
    axis_max = max(x_max, y_max)
    axis_max = axis_max * 1.05 if axis_max > 0 else 1

    # --- Tick calculation: min 4, max 8, multiples of 2.5 ---
    def get_tick_interval(axis_max):
        candidates = [0.5, 1, 2.5, 5, 10, 25, 50, 100]
        for interval in candidates:
            n_ticks = int(axis_max // interval) + 1
            if 4 <= n_ticks <= 8:
                return interval
        for interval in reversed(candidates):
            n_ticks = int(axis_max // interval) + 1
            if n_ticks >= 4:
                return interval
        return None  # fallback: let Plotly decide

    tick_interval = get_tick_interval(axis_max)
    if tick_interval:
        axis_max = math.ceil(axis_max / tick_interval) * tick_interval
        n_ticks = int(axis_max // tick_interval) + 1
        tickvals = [round(i * tick_interval, 2) for i in range(n_ticks)]
        ticktext = [str(v) for v in tickvals]
        xaxis_kwargs = dict(
            range=[0, axis_max],
            showgrid=False,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black',
            tickvals=tickvals,
            ticktext=ticktext
        )
        yaxis_kwargs = dict(
            range=[0, axis_max],
            showgrid=False,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black',
            tickvals=tickvals,
            ticktext=ticktext
        )
    else:
        # Force both axes to have the same range, but let Plotly pick ticks
        axis_max = math.ceil(axis_max)  # round up to nearest integer for safety
        xaxis_kwargs = dict(
            range=[0, axis_max],
            showgrid=False,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black'
        )
        yaxis_kwargs = dict(
            range=[0, axis_max],
            showgrid=False,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black'
        )

    fig = go.Figure()

    # All points
    fig.add_trace(go.Scatter(
        x=df_pd[x_col],
        y=df_pd[y_col],
        mode='markers',
        marker=dict(
            color=color_val,
            size=8,
            opacity=0.75,
            line=dict(width=0)
        ),
        name="All Points",
        text=df_pd['doc_id'] if 'doc_id' in df_pd.columns else None,
        hovertemplate=(
            "<b>doc_id:</b> %{text}<br>"
            f"<b>{x_col}:</b> %{{x:.2f}}%<br>"
            f"<b>{y_col}:</b> %{{y:.2f}}%<extra></extra>"
        ) if 'doc_id' in df_pd.columns else (
            f"<b>{x_col}:</b> %{{x:.2f}}%<br>"
            f"<b>{y_col}:</b> %{{y:.2f}}%<extra></extra>"
        ),
        showlegend=False
    ))

    # Optional: Add trendline for all points
    if trendline and not df_pd.empty and len(df_pd) > 1:
        x = df_pd[x_col]
        y = df_pd[y_col]
        coeffs = np.polyfit(x, y, 1)
        x_fit = np.array([0, axis_max])
        y_fit = coeffs[0] * x_fit + coeffs[1]
        fig.add_trace(go.Scatter(
            x=x_fit,
            y=y_fit,
            mode='lines',
            line=dict(color='tomato', width=2, dash='dash'),
            name="Linear fit",
            showlegend=False
        ))

    fig.update_xaxes(**xaxis_kwargs)
    fig.update_yaxes(**yaxis_kwargs)
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=40),
        xaxis_title=x_label or x_col,
        yaxis_title=y_label or y_col,
        height=500,
        width=500
    )
    fig.update_yaxes(**yaxis_kwargs, title_standoff=20)
    return fig


def plot_scatter_highlight(
        df: pl.DataFrame | pd.DataFrame,
        x_col: str,
        y_col: str,
        group_col: str,
        selected_groups: list = None,
        color=None,
        trendline: bool = False
        ) -> go.Figure:
    """
    Scatterplot with optional group highlighting and user-defined colors.
    Highlighted points are plotted on top of non-highlighted points.
    """

    # Convert to pandas only once if needed for plotly
    if hasattr(df, "to_pandas"):
        df_pd = df.to_pandas().copy()
    else:
        df_pd = df.copy()

    x_label = x_col + ' (per 100 tokens)'
    y_label = y_col + ' (per 100 tokens)'

    if 'Highlight' in df_pd.columns:
        df_pd = df_pd.drop(columns=['Highlight'])
    df_pd['Highlight'] = True
    if selected_groups:
        df_pd['Highlight'] = df_pd[group_col].apply(lambda g: g in selected_groups)
    else:
        df_pd['Highlight'] = True

    # Color logic
    if isinstance(color, dict):
        highlight_color = color.get('Highlight', '#133955')
        non_highlight_color = color.get('Non-Highlight', 'lightgray')
    elif isinstance(color, str) and color.lower().startswith("#"):
        highlight_color = color
        non_highlight_color = 'lightgray'
    else:
        highlight_color = '#133955'
        non_highlight_color = 'lightgray'

    # Axis scaling and ticks
    x_max = df_pd[x_col].max() if not df_pd.empty else 1
    y_max = df_pd[y_col].max() if not df_pd.empty else 1
    axis_max = max(x_max, y_max)
    axis_max = axis_max * 1.05 if axis_max > 0 else 1

    # --- Tick calculation: min 4, max 8, multiples of 2.5 ---
    def get_tick_interval(axis_max):
        candidates = [0.5, 1, 2.5, 5, 10, 25, 50, 100]
        for interval in candidates:
            n_ticks = int(axis_max // interval) + 1
            if 4 <= n_ticks <= 8:
                return interval
        for interval in reversed(candidates):
            n_ticks = int(axis_max // interval) + 1
            if n_ticks >= 4:
                return interval
        return None  # fallback: let Plotly decide

    # Get tick interval
    tick_interval = get_tick_interval(axis_max)
    if tick_interval:
        axis_max = math.ceil(axis_max / tick_interval) * tick_interval
        n_ticks = int(axis_max // tick_interval) + 1
        tickvals = [round(i * tick_interval, 2) for i in range(n_ticks)]
        ticktext = [str(v) for v in tickvals]
        xaxis_kwargs = dict(
            range=[0, axis_max],
            showgrid=False,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black',
            tickvals=tickvals,
            ticktext=ticktext
        )
        yaxis_kwargs = dict(
            range=[0, axis_max],
            showgrid=False,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black',
            tickvals=tickvals,
            ticktext=ticktext
        )
    else:
        axis_max = math.ceil(axis_max)  # round up for safety
        xaxis_kwargs = dict(
            range=[0, axis_max],
            showgrid=False,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black'
        )
        yaxis_kwargs = dict(
            range=[0, axis_max],
            showgrid=False,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black'
        )

    # Split data
    df_non_highlight = df_pd[df_pd['Highlight'] == False]  # noqa: E712
    df_highlight = df_pd[df_pd['Highlight'] == True]  # noqa: E712

    fig = go.Figure()

    # Non-highlighted points (bottom layer)
    if not df_non_highlight.empty:
        fig.add_trace(go.Scatter(
            x=df_non_highlight[x_col],
            y=df_non_highlight[y_col],
            mode='markers',
            marker=dict(
                color=non_highlight_color,
                size=8,
                opacity=0.5,
                line=dict(width=0)
            ),
            name="Non-Highlight",
            text=df_non_highlight[group_col] if group_col in df_non_highlight.columns else None,  # noqa: E501
            hovertemplate=(
                f"<b>{group_col}:</b> %{{text}}<br>"
                f"<b>{x_col}:</b> %{{x:.2f}}%<br>"
                f"<b>{y_col}:</b> %{{y:.2f}}%<extra></extra>"
            ) if group_col in df_non_highlight.columns else None,
            showlegend=False
        ))

    # Highlighted points (top layer)
    if not df_highlight.empty:
        fig.add_trace(go.Scatter(
            x=df_highlight[x_col],
            y=df_highlight[y_col],
            mode='markers',
            marker=dict(
                color=highlight_color,
                size=8,
                opacity=0.85,
                line=dict(width=1, color='black')
            ),
            name="Highlight",
            text=df_highlight[group_col] if group_col in df_highlight.columns else None,
            hovertemplate=(
                f"<b>{group_col}:</b> %{{text}}<br>"
                f"<b>{x_col}:</b> %{{x:.2f}}%<br>"
                f"<b>{y_col}:</b> %{{y:.2f}}%<extra></extra>"
            ) if group_col in df_highlight.columns else None,
            showlegend=False
        ))

    # Optional: Add trendline for highlighted points only
    if trendline and not df_highlight.empty and len(df_highlight) > 1:
        x = df_highlight[x_col]
        y = df_highlight[y_col]
        coeffs = np.polyfit(x, y, 1)
        x_fit = np.array([0, axis_max])
        y_fit = coeffs[0] * x_fit + coeffs[1]
        fig.add_trace(go.Scatter(
            x=x_fit,
            y=y_fit,
            mode='lines',
            line=dict(color='tomato', width=2, dash='dash'),
            name="Linear fit",
            showlegend=False
        ))

    fig.update_xaxes(**xaxis_kwargs)
    fig.update_yaxes(**yaxis_kwargs)
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=40),
        xaxis_title=x_label or x_col,
        yaxis_title=y_label or y_col,
        height=500,
        width=500
    )
    fig.update_yaxes(**yaxis_kwargs, title_standoff=20)
    return fig


def generate_scatterplot(
        user_session_id: str,
        df: pl.DataFrame,
        xaxis: str,
        yaxis: str
        ) -> None:
    """
    Generate a scatterplot from the given dataframe.

    Parameters
    ----------
    user_session_id : str
        User session identifier.
    df : pl.DataFrame
        Dataframe containing the data to plot.
    xaxis : str
        Column name for x-axis.
    yaxis : str
        Column name for y-axis.

    Returns
    -------
    None
        Updates session state with scatterplot results.
    """
    # --- User input validation ---
    if df is None or df.is_empty():
        st.session_state[user_session_id][ScatterplotKeys.WARNING] = (
            "No data available for plotting. Please process your corpus and select valid tags.",  # noqa: E501
            ":material/info:"
        )
        return

    if xaxis not in df.columns or yaxis not in df.columns:
        st.session_state[user_session_id][ScatterplotKeys.WARNING] = (
            "Selected axes are not present in the data.",
            ":material/info:"
        )
        return

    # --- Prepare data ---
    try:
        df_plot = ds.dtm_weight(df).with_columns(
            pl.selectors.numeric().mul(100)
        )

        # Optimize memory usage for large datasets
        df_plot = optimize_dataframe_memory(df_plot)

        # Calculate correlation for the selected variables - convert only once
        if hasattr(df_plot, "to_pandas"):
            df_pd = df_plot.to_pandas()
        else:
            df_pd = df_plot
        cc = pearsonr(df_pd[xaxis], df_pd[yaxis])
        correlation_dict = {
            'all': {
                'df': len(df_pd.index) - 2,
                'r': round(cc.statistic, 3),
                'p': round(cc.pvalue, 5)
            }
        }

        # Store the processed dataframe and correlation for plotting
        st.session_state[user_session_id][ScatterplotKeys.DF] = df_plot
        st.session_state[user_session_id][ScatterplotKeys.CORRELATION] = correlation_dict
        st.session_state[user_session_id][ScatterplotKeys.WARNING] = None

        app_core.session_manager.update_session_state(user_session_id, 'scatterplot', True)

    except Exception as e:
        st.session_state[user_session_id][ScatterplotKeys.WARNING] = (
            f"Scatterplot generation failed: {e}",
            ":material/sentiment_stressed:"
        )
        return


def generate_scatterplot_with_groups(
        user_session_id: str,
        df: pl.DataFrame,
        xaxis: str,
        yaxis: str,
        metadata_target: dict,
        session: dict
        ) -> None:
    """
    Generate a scatterplot with grouping from metadata.

    Parameters
    ----------
    user_session_id : str
        User session identifier.
    df : pl.DataFrame
        Dataframe containing the data to plot.
    xaxis : str
        Column name for x-axis.
    yaxis : str
        Column name for y-axis.
    metadata_target : dict
        Target corpus metadata.
    session : dict
        Session state dictionary.

    Returns
    -------
    None
        Updates session state with grouped scatterplot results.
    """
    # --- User input validation ---
    if df is None or df.is_empty():
        st.session_state[user_session_id][ScatterplotKeys.WARNING] = (
            "No data available for plotting. Please process your corpus and select valid tags.",  # noqa: E501
            ":material/info:"
        )
        return

    if xaxis not in df.columns or yaxis not in df.columns:
        st.session_state[user_session_id][ScatterplotKeys.WARNING] = (
            "Selected axes are not present in the data.",
            ":material/info:"
        )
        return

    # --- Prepare grouped data ---
    try:
        df_plot = ds.dtm_weight(df).with_columns(
            pl.selectors.numeric().mul(100)
        )

        # Add grouping information if available
        from webapp.utilities.session import safe_session_get
        if safe_session_get(session, SessionKeys.HAS_META, False):
            grouping = metadata_target.get('doccats', [{}])[0].get('cats', [])
            if grouping:
                df_plot = df_plot.with_columns(
                    pl.Series("Group", grouping)
                )

        # Optimize memory usage for large datasets
        df_plot = optimize_dataframe_memory(df_plot)

        # Calculate correlation for the selected variables
        if hasattr(df_plot, "to_pandas"):
            df_pd = df_plot.to_pandas()
        else:
            df_pd = df_plot
        cc = pearsonr(df_pd[xaxis], df_pd[yaxis])
        correlation_dict = {
            'all': {
                'df': len(df_pd.index) - 2,
                'r': round(cc.statistic, 3),
                'p': round(cc.pvalue, 5)
            }
        }

        # Store the processed dataframe and correlation for plotting
        st.session_state[user_session_id][ScatterplotKeys.GROUP_DF] = df_plot
        st.session_state[user_session_id][ScatterplotKeys.GROUP_CORRELATION] = (
            correlation_dict
        )
        st.session_state[user_session_id][ScatterplotKeys.GROUP_WARNING] = None

        app_core.session_manager.update_session_state(
            user_session_id, 'scatterplot_grouped', True
        )

    except Exception as e:
        st.session_state[user_session_id][ScatterplotKeys.WARNING] = (
            f"Grouped scatterplot generation failed: {e}",
            ":material/sentiment_stressed:"
        )
        return
