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

# PCA and specialized visualization functions

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import streamlit as st
import docuscospacy.corpus_analysis as ds
from plotly.subplots import make_subplots

from webapp.utilities.plotting.boxplot_utils import boxplots_pl
from webapp.config.session_keys import BoxplotKeys


def plot_tag_density(
    df_plot: pd.DataFrame,
    tag_list: list,
    tag_colors: list
) -> go.Figure:
    """
    Create a plotly tag density plot showing where tags occur in normalized text time.
    Each tag gets its own faceted subplot with vertical lines at occurrence positions.

    Args:
        df_plot: DataFrame with columns ['Tag', 'X'] where X is normalized position (0-1)
        tag_list: List of tags to plot (used for ordering)
        tag_colors: List of hex color codes for each tag (must match text highlighting)

    Returns:
        Plotly figure object with faceted subplots
    """
    if df_plot.empty or not tag_list:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No data to plot",
            xaxis_title="Normalized Text Position (%)",
            height=200
        )
        return fig

    # Ensure we have the right number of colors
    tag_colors = tag_colors[:len(tag_list)]

    # Filter to only selected tags and ensure ordering matches tag_list
    df_filtered = df_plot[df_plot['Tag'].isin(tag_list)].copy()

    if df_filtered.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No selected tags found in document",
            xaxis_title="Normalized Text Position (%)",
            height=200
        )
        return fig

    # Create subplots - one row for each tag
    fig = make_subplots(
        rows=len(tag_list),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[1] * len(tag_list)
    )

    # Add traces for each tag in its own subplot
    for i, tag in enumerate(tag_list, 1):
        tag_data = df_filtered[df_filtered['Tag'] == tag]
        color = tag_colors[i-1] if i-1 < len(tag_colors) else '#1f77b4'

        if not tag_data.empty:
            # Create vertical lines for each occurrence
            for _, row in tag_data.iterrows():
                fig.add_trace(
                    go.Scatter(
                        x=[row['X'], row['X']],
                        y=[0, 1],
                        mode='lines',
                        line=dict(color=color, width=2),
                        showlegend=False,
                        hovertemplate=(
                            f'<b>{tag}</b><br>Position: {row["X"]:.1%}<extra></extra>'
                        )
                    ),
                    row=i, col=1
                )

    # Update layout
    fig.update_layout(
        height=max(150, len(tag_list) * 30),
        margin=dict(l=100, r=50, t=30, b=50),
        plot_bgcolor='white',
        showlegend=False
    )

    # Update x-axes (only the bottom one needs labels)
    for i in range(1, len(tag_list) + 1):
        fig.update_xaxes(
            tickformat=".0%",
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            range=[0, 1],
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            row=i, col=1
        )

        # Only show x-axis title on bottom subplot
        if i == len(tag_list):
            fig.update_xaxes(title_text="Normalized Text Position (%)", row=i, col=1)

    # Update y-axes (add tag labels on the left)
    for i, tag in enumerate(tag_list, 1):
        fig.update_yaxes(
            showticklabels=True,
            tickvals=[0.5],  # Middle of the subplot
            ticktext=[tag],  # Tag name as label
            tickfont=dict(size=12),
            showgrid=False,
            range=[0, 1],
            row=i, col=1
        )

    return fig


def generate_boxplot(
        user_session_id: str,
        df: pl.DataFrame,
        box_vals: list
        ) -> None:
    """Generate a boxplot for the given data and save to session state."""
    # --- User input validation ---
    if df is None or df.is_empty():
        st.session_state[user_session_id][BoxplotKeys.WARNING] = (
            """
            No data available for plotting.
            Please process your corpus and select valid tags.
            """,
            ":material/info:"
        )
        return

    if not box_vals or any(val not in df.columns for val in box_vals):
        st.session_state[user_session_id][BoxplotKeys.WARNING] = (
            "Please select at least one valid variable for plotting.",
            ":material/info:"
        )
        return

    # --- Prepare data ---
    try:
        df_plot = ds.dtm_weight(df)
        df_plot = boxplots_pl(
            df_plot,
            box_vals,
            grp_a=None,
            grp_b=None
        )
    except Exception as e:
        st.session_state[user_session_id][BoxplotKeys.WARNING] = (
            f"Failed to prepare data for plotting: {e}",
            ":material/sentiment_stressed:"
        )
        return

    # --- Check if enough data for plotting ---
    try:
        df_pandas = df_plot.to_pandas()
    except Exception as e:
        st.session_state[user_session_id][BoxplotKeys.WARNING] = (
            f"Failed to convert data for plotting: {e}",
            ":material/sentiment_stressed:"
        )
        return

    if df_pandas.empty:
        st.session_state[user_session_id][BoxplotKeys.WARNING] = (
            "No data available after weighting for plotting.",
            ":material/info:"
        )
        return

    # --- Compute descriptive statistics ---
    try:
        stats = (
            df_plot
            .group_by(["Tag"])
            .agg(
                pl.len().alias("count"),
                pl.col("RF").mean().alias("mean"),
                pl.col("RF").median().alias("median"),
                pl.col("RF").std().alias("std"),
                pl.col("RF").min().alias("min"),
                pl.col("RF").quantile(0.25).alias("25%"),
                pl.col("RF").quantile(0.5).alias("50%"),
                pl.col("RF").quantile(0.75).alias("75%"),
                pl.col("RF").max().alias("max")
            )
            .sort("Tag")
        )
    except Exception as e:
        st.session_state[user_session_id][BoxplotKeys.WARNING] = (
            f"Failed to compute descriptive statistics: {e}",
            ":material/sentiment_stressed:"
        )
        return

    # --- Save results and clear warning ---
    st.session_state[user_session_id][BoxplotKeys.DF] = df_pandas
    st.session_state[user_session_id][BoxplotKeys.STATS] = stats
    st.session_state[user_session_id][BoxplotKeys.WARNING] = None


def generate_boxplot_by_group(
        user_session_id: str,
        df: pl.DataFrame,
        box_vals: list,
        grpa_list: list,
        grpb_list: list
        ) -> None:
    """Generate a grouped boxplot for the given data and save to session state."""
    # --- User input validation ---
    if df is None or df.is_empty():
        st.session_state[user_session_id][BoxplotKeys.GROUP_WARNING] = (
            """
            No data available for plotting.
            Please process your corpus and select valid tags.
            """,
            ":material/info:"
        )
        return

    if not box_vals or any(val not in df.columns for val in box_vals):
        st.session_state[user_session_id][BoxplotKeys.GROUP_WARNING] = (
            "Please select at least one valid variable for plotting.",
            ":material/info:"
        )
        return

    if len(grpa_list) == 0 or len(grpb_list) == 0:
        st.session_state[user_session_id][BoxplotKeys.GROUP_WARNING] = (
            "You must select at least one category for both Group A and Group B.",
            ":material/info:"
        )
        return

    # --- Prepare data ---
    try:
        df_plot = ds.dtm_weight(df)
        df_plot = boxplots_pl(
            df_plot,
            box_vals,
            grp_a=grpa_list,
            grp_b=grpb_list
        )
        df_pandas = df_plot.to_pandas()
    except Exception as e:
        st.session_state[user_session_id][BoxplotKeys.GROUP_WARNING] = (
            f"Failed to prepare data for plotting: {e}",
            ":material/sentiment_stressed:"
        )
        return

    if df_pandas.empty:
        st.session_state[user_session_id][BoxplotKeys.GROUP_WARNING] = (
            "No data available after weighting for plotting.",
            ":material/info:"
        )
        return

    # --- Compute descriptive statistics ---
    try:
        stats = (
            df_plot
            .group_by(["Group", "Tag"])
            .agg(
                pl.len().alias("count"),
                pl.col("RF").mean().alias("mean"),
                pl.col("RF").median().alias("median"),
                pl.col("RF").std().alias("std"),
                pl.col("RF").min().alias("min"),
                pl.col("RF").quantile(0.25).alias("25%"),
                pl.col("RF").quantile(0.5).alias("50%"),
                pl.col("RF").quantile(0.75).alias("75%"),
                pl.col("RF").max().alias("max")
            )
            .sort(["Tag", "Group"])
        )
    except Exception as e:
        st.session_state[user_session_id][BoxplotKeys.GROUP_WARNING] = (
            f"Failed to compute descriptive statistics: {e}",
            ":material/sentiment_stressed:"
        )
        return

    # --- Save results and clear warning ---
    st.session_state[user_session_id][BoxplotKeys.GROUP_DF] = df_pandas
    st.session_state[user_session_id][BoxplotKeys.GROUP_STATS] = stats
    st.session_state[user_session_id][BoxplotKeys.GROUP_WARNING] = None
