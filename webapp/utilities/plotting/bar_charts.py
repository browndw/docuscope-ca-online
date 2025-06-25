"""
Module for plotting horizontal bar charts using Plotly.
Includes functions for plotting tag frequencies and comparing corpus parts.
"""
import pandas as pd
import plotly.express as px
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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

    # Convert to pandas only once if needed for Plotly
    if hasattr(df_sorted, 'to_pandas'):
        df_plot = df_sorted.to_pandas()
    else:
        df_plot = df_sorted

    min_height = 200  # Minimum plot height in pixels
    height = max(24 * len(df_plot) + 40, min_height)

    fig = px.bar(
        df_plot,
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
        hovertemplate="<b>Tag:</b> %{y}<br>" +
        "<b>RF:</b> %{x:.2f}%" +
        "<extra></extra>"
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
    # Prepare DataFrame - convert once and reuse
    if hasattr(df, "to_pandas"):
        df_plot = df.to_pandas()
    else:
        df_plot = df.copy()
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
        hovertemplate="<b>Tag:</b> %{y}<br>" +
        "<b>Corpus:</b> %{customdata[0]}<br>" +
        "<b>RF:</b> %{x:.2f}%" +
        "<extra></extra>",
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
                        line=dict(color=color, width=.75),
                        showlegend=False,
                        hovertemplate=(
                            f'<b>{tag}</b><br>Position: {row["X"]:.1%}<extra></extra>'
                        )
                    ),
                    row=i, col=1
                )

    # Update layout with dynamic height based on number of tags
    # Base height + proportional height per tag to maintain visibility
    base_height = 200
    height_per_tag = 80 if len(tag_list) <= 2 else 60 if len(tag_list) <= 3 else 50
    total_height = base_height + (len(tag_list) * height_per_tag)

    fig.update_layout(
        height=total_height,
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
            showgrid=False,
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
