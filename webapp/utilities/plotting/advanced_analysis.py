"""
Advanced plotting functions including PCA and scatterplot generation.

This module provides functions for generating PCA plots, scatterplots with grouping,
and other advanced statistical visualizations.
"""

import polars as pl
import pandas as pd
import numpy as np
import streamlit as st
import docuscospacy as ds
from scipy.stats import pearsonr
from sklearn import decomposition

# Import session utilities
from webapp.utilities.session import update_session
from webapp.utilities.analysis import update_metadata
from webapp.utilities.state import PCAKeys, ScatterplotKeys


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
    if session.get('has_meta', [False])[0]:
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

    # --- Convert to pandas only if needed ---
    try:
        df_pd = df.to_pandas()
        pca_df, contrib_df, ve = pca_contributions(df_pd, grouping)
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

    update_metadata(
        'target',
        'variance',
        ve,
        user_session_id
    )
    update_session(
        'pca',
        True,
        user_session_id
    )
    st.session_state[user_session_id][PCAKeys.WARNING] = None
    st.rerun()


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

        # Calculate correlation for the selected variables
        df_pd = df_plot.to_pandas()
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

        update_session(
            'scatterplot',
            True,
            user_session_id
        )

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
        if session.get('has_meta', [False])[0]:
            grouping = metadata_target.get('doccats', [{}])[0].get('cats', [])
            if grouping:
                df_plot = df_plot.with_columns(
                    pl.Series("Group", grouping)
                )

        # Calculate correlation for the selected variables
        df_pd = df_plot.to_pandas()
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

        update_session(
            'scatterplot_grouped',
            True,
            user_session_id
        )

    except Exception as e:
        st.session_state[user_session_id][ScatterplotKeys.WARNING] = (
            f"Grouped scatterplot generation failed: {e}",
            ":material/sentiment_stressed:"
        )
        return
