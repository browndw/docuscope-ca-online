"""
Text visualization and tagging utilities.

This module provides functions for generating HTML legends and tag density plots
for text analysis visualization.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_tag_html_legend(tag_list: list, tag_colors: list) -> str:
    """
    Generate HTML legend string for tag highlighting.

    Args:
        tag_list: List of tag names to display in legend
        tag_colors: List of hex color codes corresponding to tags

    Returns:
        HTML string with colored spans for tag legend
    """
    if not tag_list or not tag_colors:
        return ""

    # Ensure we don't have more tags than colors
    tag_colors = tag_colors[:len(tag_list)]

    # Create HTML spans for each tag
    tag_html = []
    for color, tag in zip(tag_colors, tag_list):
        tag_html.append(f'<span style="background-color: {color}">{tag}</span>')

    return '; '.join(tag_html)


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
