"""
Module for plotting boxplots in the web application.
"""
import docuscospacy as ds
import pandas as pd
import polars as pl
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Module-specific imports
from webapp.utilities.plotting.utils import boxplots_pl
from webapp.utilities.state import BoxplotKeys


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
                    hovertemplate="<b>%{y}</b>" +
                    "<br>Min: %{customdata[3]:.2f}%" +
                    "<br>IQR: %{customdata[2]:.2f}%" +
                    "<br>Median: %{customdata[1]:.2f}%" +
                    "<br>Mean: %{customdata[0]:.2f}%" +
                    "<br>Max: %{customdata[4]:.2f}%" +
                    "<extra></extra>"
                    )  # noqa: E501

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


def plot_grouped_boxplot(
        df,
        tag_col='Tag',
        value_col='RF',
        group_col='Group',
        color=None,
        palette=None
        ) -> go.Figure:
    """
    Boxplot comparing categories in subcorpora, grouped by tag and colored by group.
    Allows user to specify a custom HEX color or a Plotly palette.
    """
    tag_order = (
        df.groupby(tag_col)[value_col].median().sort_values(ascending=False).index.tolist()
    )
    group_order = sorted(df[group_col].unique())

    if isinstance(color, dict):
        color_discrete_map = color
        color_arg = group_col
    elif isinstance(color, str) and color.lower().startswith("#"):
        color_discrete_map = {cat: color for cat in df[group_col].unique()}
        color_arg = group_col
    else:
        color_discrete_map = None
        color_arg = group_col

    # Compute summary stats for hover (per tag+group)
    stats = (
        df.groupby([tag_col, group_col])[value_col]
        .agg(['mean', 'median', lambda s: s.quantile(0.75) - s.quantile(0.25), 'min', 'max'])  # noqa: E501
        .rename(columns={'mean': 'Mean', 'median': 'Median', '<lambda_0>': 'IQR', 'min': 'Min', 'max': 'Max'})  # noqa: E501
        .reset_index()
    )

    # Create boxplot
    fig = px.box(
        df,
        y=tag_col,
        x=value_col,
        color=color_arg,
        color_discrete_map=color_discrete_map,
        color_discrete_sequence=palette if palette else None,
        points=False,
        orientation='h',
        category_orders={
            tag_col: tag_order,
            group_col: group_order
        },
        boxmode="group"
    )

    # Turn off default boxplot hover for all traces
    for trace in fig.data:
        if trace.type == "box":
            trace.hoverinfo = "skip"
            trace.hoveron = "boxes"

    # Overlay transparent bar for custom hover (per tag+group)
    bar_df = stats.copy()
    bar_df['bar'] = bar_df['Max'] - bar_df['Min']
    bar_df['base'] = bar_df['Min']
    fig2 = px.bar(
        bar_df,
        y=tag_col,
        x='bar',
        base='base',
        color=group_col,
        orientation='h',
        color_discrete_map=color_discrete_map,
        category_orders={group_col: group_order, tag_col: tag_order},
        hover_data={
            'Mean': ':.2f',
            'Median': ':.2f',
            'IQR': ':.2f',
            'Min': ':.2f',
            'Max': ':.2f',
            'base': False,
            'bar': False,
            tag_col: False,
            group_col: False,
        },
    ).update_traces(
        opacity=0.01,  # nearly invisible, but hoverable
        hovertemplate="<b>%{y} | %{customdata[5]}</b>" +
        "<br>Min: %{customdata[3]:.2f}%" +
        "<br>IQR: %{customdata[2]:.2f}%" +
        "<br>Median: %{customdata[1]:.2f}%" +
        "<br>Mean: %{customdata[0]:.2f}%" +
        "<br>Max: %{customdata[4]:.2f}%" +
        "<extra></extra>"
    )

    # Add bar traces to boxplot
    for trace in fig2.data:
        trace.showlegend = False  # Hide bar overlay from legend
        fig.add_trace(trace)

    fig.update_layout(
        hovermode="closest",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.5,  # Move legend well below the plot
            xanchor="left",
            x=0
        ),
        legend_title_text='',
        margin=dict(l=0, r=0, t=30, b=0),
        height=100 * len(tag_order) + 120,
        xaxis_title='Frequency (per 100 tokens)',
        yaxis_title=None
    )
    fig.update_yaxes(showticklabels=True, title=None)
    fig.update_xaxes(title_text='Frequency (per 100 tokens)', range=[0, None])

    # Add vertical line at x=0, colored gray
    fig.add_vline(
        x=0,
        line_width=2,
        line_color="gray",
        layer="below"
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
        if hasattr(df_plot, "to_pandas"):
            df_pandas = df_plot.to_pandas()
        else:
            df_pandas = df_plot
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
        if hasattr(df_plot, "to_pandas"):
            df_pandas = df_plot.to_pandas()
        else:
            df_pandas = df_plot
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
