"""
Advanced plotting functions including PCA and scatterplot generation.

This module provides functions for generating PCA plots, scatterplots with grouping,
and other advanced statistical visualizations.
"""

import math
import polars as pl
import pandas as pd
import numpy as np
import streamlit as st
import docuscospacy as ds
import plotly.graph_objects as go
from sklearn import decomposition

# Import session utilities
from webapp.utilities.core import app_core
from webapp.utilities.state import PCAKeys, SessionKeys
from webapp.utilities.memory import lazy_computation


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


def update_pca_plot(
        coord_data,
        contrib_data,
        variance,
        pca_idx
        ) -> tuple:
    """
    Update PCA plot data for specific components.

    Parameters
    ----------
    coord_data : pd.DataFrame
        PCA coordinate data.
    contrib_data : pd.DataFrame
        PCA contribution data.
    variance : list
        Explained variance ratios.
    pca_idx : int
        Index of PCA component (1-based).

    Returns
    -------
    tuple
        (contrib_x, contrib_y) where each is a list of (tag, value) tuples.
    """
    pca_x = coord_data.columns[pca_idx - 1]
    pca_y = coord_data.columns[pca_idx]

    mean_x = contrib_data[pca_x].abs().mean()
    mean_y = contrib_data[pca_y].abs().mean()

    # Always use .copy() after filtering
    contrib_x = contrib_data[contrib_data[pca_x].abs() > mean_x].copy()
    contrib_x.sort_values(by=pca_x, ascending=False, inplace=True)
    contrib_x_values = contrib_x.loc[:, pca_x].tolist()
    contrib_x_values = ['%.2f' % x for x in contrib_x_values]
    contrib_x_values = [x + "%" for x in contrib_x_values]
    contrib_x_tags = contrib_x.loc[:, "Tag"].tolist()
    contrib_x = list(zip(contrib_x_tags, contrib_x_values))

    contrib_y = contrib_data[contrib_data[pca_y].abs() > mean_y].copy()
    contrib_y.sort_values(by=pca_y, ascending=False, inplace=True)
    contrib_y_values = contrib_y.loc[:, pca_y].tolist()
    contrib_y_values = ['%.2f' % x for x in contrib_y_values]
    contrib_y_values = [x + "%" for x in contrib_y_values]
    contrib_y_tags = contrib_y.loc[:, "Tag"].tolist()
    contrib_y = list(zip(contrib_y_tags, contrib_y_values))

    return contrib_x, contrib_y


def generate_pca(
        user_session_id: str,
        df: pl.DataFrame,
        metadata_target: dict,
        session: dict
        ) -> None:
    """
    Generate PCA analysis for the given dataframe.

    Parameters
    ----------
    user_session_id : str
        User session identifier.
    df : pl.DataFrame
        Document-term matrix dataframe.
    metadata_target : dict
        Target corpus metadata.
    session : dict
        Session state dictionary.

    Returns
    -------
    None
        Updates session state with PCA results.
    """
    # --- User input validation ---
    if df is None or df.is_empty():
        st.session_state[user_session_id][PCAKeys.WARNING] = (
            "No data available for PCA. Please process your corpus and select valid tags.",
            ":material/info:"
        )
        return

    # --- Always scale the data before PCA ---
    df = ds.dtm_weight(df, scheme="prop")
    df = ds.dtm_weight(df, scheme="scale")

    # --- Check for metadata grouping ---
    from webapp.utilities.session import safe_session_get
    if safe_session_get(session, SessionKeys.HAS_META, False):
        grouping = metadata_target.get('doccats', [{}])[0].get('cats', [])
    else:
        grouping = []

    # --- Drop unwanted columns ---
    to_drop = ['Other', 'FU', 'Untagged']
    df = df.drop([col for col in to_drop if col in df.columns])

    # --- Check if enough columns remain for PCA ---
    if df.width < 2:
        st.session_state[user_session_id][PCAKeys.WARNING] = (
            "Not enough variables for PCA after dropping excluded columns.",
            ":material/info:"
        )
        return

    # --- Convert to pandas only if needed for scikit-learn and use caching ---
    try:
        pca_df, contrib_df, ve = _compute_pca_with_caching(df, grouping, user_session_id)
    except Exception as e:
        st.session_state[user_session_id][PCAKeys.WARNING] = (
            f"PCA computation failed: {e}",
            ":material/sentiment_stressed:"
        )
        return

    # --- Check for empty PCA results ---
    if pca_df is None or pca_df.empty or contrib_df is None or contrib_df.empty:
        st.session_state[user_session_id][PCAKeys.WARNING] = (
            "PCA computation returned no results. Try different data.",
            ":material/info:"
        )
        return

    # --- Save results and clear warning ---
    pca_key = PCAKeys.TARGET_PCA_DF
    contrib_key = PCAKeys.TARGET_CONTRIB_DF
    st.session_state[user_session_id][pca_key[0]][pca_key[1]] = pca_df
    st.session_state[user_session_id][contrib_key[0]][contrib_key[1]] = contrib_df

    app_core.session_manager.update_metadata(
        user_session_id,
        'target',
        {'variance': ve}
    )
    app_core.session_manager.update_session_state(user_session_id, 'pca', True)
    st.session_state[user_session_id][PCAKeys.WARNING] = None
    st.rerun()


def pca_contributions(
        dtm: pd.DataFrame,
        doccats: list
        ) -> tuple:
    """
    Calculate PCA contributions for a document-term matrix.

    Parameters
    ----------
    dtm : pd.DataFrame
        Document-term matrix with 'doc_id' column.
    doccats : list
        List of document categories for grouping.

    Returns
    -------
    tuple
        (pca_df, contrib_df, variance_explained) where:
        - pca_df: DataFrame with PCA coordinates
        - contrib_df: DataFrame with variable contributions
        - variance_explained: List of explained variance ratios
    """
    df = dtm.set_index('doc_id')
    n = min(len(df.index), len(df.columns))
    pca = decomposition.PCA(n_components=n)
    pca_result = pca.fit_transform(df.values)
    pca_df = pd.DataFrame(pca_result)
    pca_df.columns = ['PC' + str(col + 1) for col in pca_df.columns]

    sdev = pca_df.std(ddof=0)
    contrib = []

    for i in range(0, len(sdev)):
        coord = pca.components_[i] * sdev.iloc[i]
        polarity = np.divide(coord, abs(coord))
        coord = np.square(coord)
        coord = np.divide(coord, sum(coord))*100
        coord = np.multiply(coord, polarity)
        contrib.append(coord)
    contrib_df = pd.DataFrame(contrib).transpose()
    contrib_df.columns = ['PC' + str(col + 1) for col in contrib_df.columns]
    contrib_df['Tag'] = df.columns

    if len(doccats) > 0:
        pca_df['Group'] = doccats
    pca_df['doc_id'] = list(df.index)
    ve = np.array(pca.explained_variance_ratio_).tolist()

    return pca_df, contrib_df, ve


def _compute_pca_with_caching(df: pl.DataFrame, grouping: list,
                              user_session_id: str) -> tuple:
    """
    Compute PCA with lazy caching for better performance.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame for PCA
    grouping : list
        Document categories for grouping
    user_session_id : str
        User session identifier for caching

    Returns
    -------
    tuple
        (pca_df, contrib_df, variance_explained)
    """
    # Create cache key based on dataframe characteristics
    cache_key = f"pca_{hash((tuple(df.columns), df.shape))}"

    def compute_pca():
        if hasattr(df, "to_pandas"):
            df_pd = df.to_pandas()
        else:
            df_pd = df
        return pca_contributions(df_pd, grouping)

    return lazy_computation(
        cache_key=cache_key,
        computation_func=compute_pca,
        user_session_id=user_session_id
    )
