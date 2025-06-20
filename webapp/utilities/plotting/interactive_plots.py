"""
Legacy plotting functions migrated from formatters.py.

This module contains plotting functions that were previously in the legacy
formatters module, now properly organized and imported.
"""
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl


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
        default_palette = px.colors.qualitative.Set1
        palette_colors = palette if isinstance(palette, list) else default_palette
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
        default_palette = px.colors.qualitative.Set1
        palette_colors = palette if isinstance(palette, list) else default_palette
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
        color=None,
        trendline: bool = False
        ) -> go.Figure:
    """
    Simple scatterplot for two variables, with optional color support and trendline.
    """
    # Convert to pandas if polars
    if isinstance(df, pl.DataFrame):
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

    # Convert to pandas if polars
    if isinstance(df, pl.DataFrame):
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


def plot_pca_scatter_highlight(
        df: pl.DataFrame,
        x_col: str,
        y_col: str,
        group_col: str,
        selected_groups: list = None,
        x_label: str = None,
        y_label: str = None
        ) -> go.Figure:
    """
    Create a scatter plot for PCA results with optional highlighting of groups.
    Highlighted points are plotted on top of non-highlighted points.
    Ensures both axes have the same range and tick marks.
    """
    # Convert to pandas if needed
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()
    else:
        df = df.copy()

    # Drop 'Highlight' if present, then copy to avoid SettingWithCopyWarning
    if 'Highlight' in df.columns:
        df = df.drop(columns=['Highlight']).copy()
    else:
        df = df.copy()

    df['Highlight'] = True
    if selected_groups:
        df['Highlight'] = df[group_col].apply(lambda g: g in selected_groups)
    else:
        df['Highlight'] = True

    # Color logic
    highlight_color = '#133955'
    non_highlight_color = 'lightgray'

    # Find max absolute value for axis normalization
    max_abs = max(
        abs(df[x_col].min()), abs(df[x_col].max()),
        abs(df[y_col].min()), abs(df[y_col].max())
    )

    # Use only "nice" intervals for PCA
    candidates = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    axis_max = max_abs
    for interval in candidates:
        if axis_max <= interval * 4:
            axis_max = math.ceil(axis_max / interval) * interval
            tick_interval = interval
            break
    else:
        tick_interval = candidates[-1]
        axis_max = math.ceil(axis_max / tick_interval) * tick_interval

    n_ticks = int((2 * axis_max) // tick_interval) + 1
    tickvals = [round(-axis_max + i * tick_interval, 2) for i in range(n_ticks)]
    ticktext = [str(v) for v in tickvals]

    # Split data
    df_non_highlight = df[df['Highlight'] == False]  # noqa: E712
    df_highlight = df[df['Highlight'] == True]  # noqa: E712

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
                f"<b>{x_col}:</b> %{{x:.2f}}<br>"
                f"<b>{y_col}:</b> %{{y:.2f}}<extra></extra>"
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
                f"<b>{x_col}:</b> %{{x:.2f}}<br>"
                f"<b>{y_col}:</b> %{{y:.2f}}<extra></extra>"
            ) if group_col in df_highlight.columns else None,
            showlegend=False
        ))

    # Add zero axes
    fig.add_shape(type="line",
                  x0=0, x1=0,
                  y0=-axis_max, y1=axis_max,
                  line=dict(color="black", width=1, dash="dash"),
                  layer="below")
    fig.add_shape(type="line",
                  x0=-axis_max, x1=axis_max,
                  y0=0, y1=0,
                  line=dict(color="black", width=1, dash="dash"),
                  layer="below")

    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=40),
        xaxis_title=x_label or x_col,
        yaxis_title=y_label or y_col,
        height=500,
        width=500
    )
    fig.update_xaxes(
        showgrid=False,
        range=[-axis_max, axis_max],
        zeroline=False,
        tickvals=tickvals,
        ticktext=ticktext
    )
    fig.update_yaxes(
        showgrid=False,
        range=[-axis_max, axis_max],
        zeroline=False,
        tickvals=tickvals,
        ticktext=ticktext,
        title_standoff=20
    )
    return fig


def plot_pca_variable_contrib_bar(
        contrib_1_plot,
        contrib_2_plot,
        pc1_label="PC1",
        pc2_label="PC2",
        sort_by=None
        ) -> go.Figure:
    """
    Create a horizontal bar plot comparing variable contributions
    to two principal components (PC1 and PC2).
    Parameters
    ----------
    contrib_1_plot : pd.DataFrame
        DataFrame containing contributions for PC1.
        Must have columns: 'Tag', 'Contribution'.
    contrib_2_plot : pd.DataFrame
        DataFrame containing contributions for PC2.
        Must have columns: 'Tag', 'Contribution'.
    pc1_label : str
        Label for PC1, default is "PC1".
    pc2_label : str
        Label for PC2, default is "PC2".
    sort_by : str, optional
        If provided, sort the bars by this PC label.
        If None, sort by PC1.
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The resulting bar plot figure.
    """
    # Merge on Tag for alignment
    merged = contrib_1_plot.merge(
        contrib_2_plot, on="Tag", how="outer", suffixes=(f"_{pc1_label}", f"_{pc2_label}")
    ).fillna(0)

    # Get column names for contributions
    col_pc1 = merged.columns[1]
    col_pc2 = merged.columns[2]

    # Calculate mean absolute contributions
    mean_pc1 = merged[col_pc1].abs().mean()
    mean_pc2 = merged[col_pc2].abs().mean()

    # Decide which PC to sort by
    if sort_by == pc2_label:
        sort_col = col_pc2
        main_col = col_pc2
        mean_main = mean_pc2
        other_col = col_pc1
        mean_other = mean_pc1
    else:
        sort_col = col_pc1
        main_col = col_pc1
        mean_main = mean_pc1
        other_col = col_pc2
        mean_other = mean_pc2

    merged = merged.sort_values(by=sort_col, ascending=True)

    # Assign color and opacity for each bar
    colors_main = []
    opacities_main = []
    colors_other = []
    opacities_other = []

    for _, row in merged.iterrows():
        # Main (sorted-by) PC
        if abs(row[main_col]) > mean_main:
            colors_main.append("#133955")  # dark blue
            opacities_main.append(1.0)
        else:
            colors_main.append("#216495")  # light blue
            opacities_main.append(0.6)
        # Other PC always gray
        colors_other.append("#FFFFFF")  # white
        opacities_other.append(0.4)

    # Plot bars: main PC first, then other PC
    fig = go.Figure()
    # Main PC bars
    fig.add_trace(go.Bar(
        y=merged["Tag"],
        x=merged[main_col],
        name=sort_by if sort_by else pc1_label,
        orientation='h',
        marker_color=colors_main,
        opacity=1.0,
        hovertemplate=(
            f"<b>{sort_by if sort_by else pc1_label}</b><br>"
            "Variable: %{y}<br>"
            "Contribution: %{x:.2%}<extra></extra>"
        ),
        marker=dict(opacity=opacities_main)
    ))
    # Other PC bars
    fig.add_trace(go.Bar(
        y=merged["Tag"],
        x=merged[other_col],
        name=pc2_label if main_col == col_pc1 else pc1_label,
        orientation='h',
        marker_color=colors_other,
        opacity=1.0,
        hovertemplate=(
            f"<b>{pc2_label if main_col == col_pc1 else pc1_label}</b><br>"
            "Variable: %{y}<br>"
            "Contribution: %{x:.2%}<extra></extra>"
        ),
        marker=dict(opacity=opacities_other)
    ))

    # Add vertical lines for mean absolute contributions (main and other PC)
    for mean_val in [mean_main, -mean_main, mean_other, -mean_other]:
        fig.add_vline(
            x=mean_val,
            line=dict(color="tomato", width=2, dash="dot"),
            annotation_text="|mean|",
            annotation_position="top",
            opacity=0.7
        )

    # Set tick labels every 5% (0.05), covering the full range
    min_val = min(merged[col_pc1].min(), merged[col_pc2].min())
    max_val = max(merged[col_pc1].max(), merged[col_pc2].max())
    tick_start = (int(min_val * 20) - 1) / 20  # round down to nearest 0.05
    tick_end = (int(max_val * 20) + 1) / 20    # round up to nearest 0.05
    tickvals = [x / 100 for x in range(int(tick_start * 100), int(tick_end * 100) + 1, 5)]
    ticktext = [f"{abs(x)*100:.0f}%" for x in tickvals]

    fig.update_layout(
        barmode='group',
        height=30 * len(merged) + 100,
        margin=dict(l=0, r=0, t=30, b=40),
        xaxis_title="Contribution",
        yaxis_title="",
        xaxis=dict(
            tickvals=tickvals,
            ticktext=ticktext,
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1
        ),
        showlegend=False,
    )
    return fig
