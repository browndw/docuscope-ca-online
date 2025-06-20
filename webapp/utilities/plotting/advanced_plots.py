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

import polars as pl
import streamlit as st
import docuscospacy as ds

from webapp.utilities.plotting.boxplot_utils import boxplots_pl
from webapp.config.session_keys import BoxplotKeys


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
