"""
Legacy plotting functions migrated from formatters.py.

This module contains plotting functions that were previously in the legacy
formatters module, now properly organized and imported.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots


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
        palette_colors = palette if isinstance(palette, list) else px.colors.qualitative.Set1
        color_map = {
            cat: palette_colors[i % len(palette_colors)]
            for i, cat in enumerate(tag_order)
        }
    else:
        color_map = {
            cat: px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
            for i, cat in enumerate(tag_order)
        }

    # Create the boxplot
    fig = go.Figure()
    for tag in tag_order:
        tag_data = df[df[tag_col] == tag][value_col]
        fig.add_trace(go.Box(
            y=tag_data,
            name=tag,
            marker_color=color_map.get(tag, 'blue'),
            boxmean=True
        ))

    fig.update_layout(
        title="Tag Distribution",
        xaxis_title="Tags",
        yaxis_title="Relative Frequency (%)",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="left",
            x=0
        ),
        height=600
    )

    return fig


def plot_grouped_boxplot(
        df: pl.DataFrame | pd.DataFrame,
        tag_col='Tag',
        value_col='RF',
        group_col='Group',
        color=None,
        palette=None
        ) -> go.Figure:
    """
    Grouped boxplot for comparing tags across different groups.
    """
    # Convert to pandas if polars
    if isinstance(df, pl.DataFrame):
        df_pd = df.to_pandas()
    else:
        df_pd = df

    # Get unique groups and tags
    groups = df_pd[group_col].unique()
    tags = df_pd[tag_col].unique()

    # Color logic
    if isinstance(color, dict):
        color_map = color
    elif isinstance(color, str) and color.lower().startswith("#"):
        color_map = {group: color for group in groups}
    elif palette:
        palette_colors = palette if isinstance(palette, list) else px.colors.qualitative.Set1
        color_map = {
            group: palette_colors[i % len(palette_colors)]
            for i, group in enumerate(groups)
        }
    else:
        color_map = {
            group: px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
            for i, group in enumerate(groups)
        }

    # Create grouped boxplot
    fig = go.Figure()
    
    for i, group in enumerate(groups):
        group_data = df_pd[df_pd[group_col] == group]
        for tag in tags:
            tag_data = group_data[group_data[tag_col] == tag][value_col]
            if not tag_data.empty:
                fig.add_trace(go.Box(
                    y=tag_data,
                    name=f"{group} - {tag}",
                    marker_color=color_map.get(group, 'blue'),
                    legendgroup=group,
                    offsetgroup=i,
                    boxmean=True
                ))

    fig.update_layout(
        title="Grouped Tag Distribution",
        xaxis_title="Tags",
        yaxis_title="Relative Frequency (%)",
        boxmode='group',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="left",
            x=0
        ),
        height=600
    )

    return fig


def plot_scatter(
        df: pl.DataFrame | pd.DataFrame,
        x_col: str,
        y_col: str,
        hover_col: str = None,
        color=None,
        palette=None
        ) -> go.Figure:
    """
    Create a scatter plot from the given dataframe.
    """
    # Convert to pandas if polars
    if isinstance(df, pl.DataFrame):
        df_pd = df.to_pandas()
    else:
        df_pd = df

    # Create scatter plot
    if color and isinstance(color, str) and color in df_pd.columns:
        fig = px.scatter(
            df_pd,
            x=x_col,
            y=y_col,
            color=color,
            hover_data=[hover_col] if hover_col else None,
            color_discrete_sequence=palette if palette else px.colors.qualitative.Set1
        )
    else:
        fig = px.scatter(
            df_pd,
            x=x_col,
            y=y_col,
            hover_data=[hover_col] if hover_col else None
        )

    fig.update_layout(
        title=f"{y_col} vs {x_col}",
        height=600
    )

    return fig


def plot_scatter_highlight(
        df: pl.DataFrame | pd.DataFrame,
        x_col: str,
        y_col: str,
        highlight_col: str = None,
        hover_col: str = None,
        color=None,
        palette=None
        ) -> go.Figure:
    """
    Create a scatter plot with highlighted points.
    """
    # Convert to pandas if polars
    if isinstance(df, pl.DataFrame):
        df_pd = df.to_pandas()
    else:
        df_pd = df

    fig = go.Figure()

    if highlight_col and highlight_col in df_pd.columns:
        # Plot highlighted and non-highlighted points separately
        highlighted = df_pd[df_pd[highlight_col] == True]
        non_highlighted = df_pd[df_pd[highlight_col] == False]

        # Non-highlighted points
        fig.add_trace(go.Scatter(
            x=non_highlighted[x_col],
            y=non_highlighted[y_col],
            mode='markers',
            name='Other',
            marker=dict(color='lightgray', size=8),
            hovertemplate=f"<b>{x_col}:</b> %{{x}}<br><b>{y_col}:</b> %{{y}}<extra></extra>"
        ))

        # Highlighted points
        fig.add_trace(go.Scatter(
            x=highlighted[x_col],
            y=highlighted[y_col],
            mode='markers',
            name='Highlighted',
            marker=dict(color='red', size=10),
            hovertemplate=f"<b>{x_col}:</b> %{{x}}<br><b>{y_col}:</b> %{{y}}<extra></extra>"
        ))
    else:
        # Regular scatter plot
        fig.add_trace(go.Scatter(
            x=df_pd[x_col],
            y=df_pd[y_col],
            mode='markers',
            marker=dict(size=8),
            hovertemplate=f"<b>{x_col}:</b> %{{x}}<br><b>{y_col}:</b> %{{y}}<extra></extra>"
        ))

    fig.update_layout(
        title=f"{y_col} vs {x_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=600
    )

    return fig


def plot_pca_scatter_highlight(
        df: pl.DataFrame | pd.DataFrame,
        x_col: str,
        y_col: str,
        group_col: str = None,
        highlight_col: str = None,
        color=None,
        palette=None
        ) -> go.Figure:
    """
    Create a PCA scatter plot with optional grouping and highlighting.
    """
    # Convert to pandas if polars
    if isinstance(df, pl.DataFrame):
        df_pd = df.to_pandas()
    else:
        df_pd = df

    fig = go.Figure()

    if group_col and group_col in df_pd.columns:
        # Group-based coloring
        groups = df_pd[group_col].unique()
        colors = palette if palette else px.colors.qualitative.Set1
        
        for i, group in enumerate(groups):
            group_data = df_pd[df_pd[group_col] == group]
            fig.add_trace(go.Scatter(
                x=group_data[x_col],
                y=group_data[y_col],
                mode='markers',
                name=str(group),
                marker=dict(
                    color=colors[i % len(colors)],
                    size=8
                ),
                hovertemplate=f"<b>Group:</b> {group}<br><b>{x_col}:</b> %{{x}}<br><b>{y_col}:</b> %{{y}}<extra></extra>"
            ))
    else:
        # Regular PCA scatter
        fig.add_trace(go.Scatter(
            x=df_pd[x_col],
            y=df_pd[y_col],
            mode='markers',
            marker=dict(size=8),
            hovertemplate=f"<b>{x_col}:</b> %{{x}}<br><b>{y_col}:</b> %{{y}}<extra></extra>"
        ))

    fig.update_layout(
        title=f"PCA: {y_col} vs {x_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=600
    )

    return fig


def plot_pca_variable_contrib_bar(
        contrib_data: pd.DataFrame,
        x_component: str,
        y_component: str,
        n_vars: int = 10
        ) -> go.Figure:
    """
    Create bar plots showing variable contributions to PCA components.
    """
    # Get top contributing variables for each component
    x_contrib = contrib_data.nlargest(n_vars, x_component)
    y_contrib = contrib_data.nlargest(n_vars, y_component)

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"Top {n_vars} Contributors to {x_component}",
                       f"Top {n_vars} Contributors to {y_component}"]
    )

    # X component contributions
    fig.add_trace(
        go.Bar(
            x=x_contrib[x_component],
            y=x_contrib['Tag'],
            orientation='h',
            name=x_component,
            marker_color='blue'
        ),
        row=1, col=1
    )

    # Y component contributions
    fig.add_trace(
        go.Bar(
            x=y_contrib[y_component],
            y=y_contrib['Tag'],
            orientation='h',
            name=y_component,
            marker_color='red'
        ),
        row=1, col=2
    )

    fig.update_layout(
        title="PCA Variable Contributions",
        height=600,
        showlegend=False
    )

    return fig
