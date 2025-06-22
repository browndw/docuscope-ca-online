"""
Boxplot utilities for corpus analysis visualization.

This module provides functions for preparing data and generating
boxplots for corpus analysis.
"""

import polars as pl


def boxplots_pl(
        dtm_pl: pl.DataFrame,
        box_vals: list,
        grp_a=None,
        grp_b=None
        ) -> pl.DataFrame:
    """
    Prepare data for boxplot visualization from a document-term matrix.

    Parameters
    ----------
    dtm_pl : pl.DataFrame
        Document-term matrix as a Polars DataFrame.
    box_vals : list
        List of column names (tags) to include in the boxplot.
    grp_a : list, optional
        List of category IDs for Group A (for grouped boxplots).
    grp_b : list, optional
        List of category IDs for Group B (for grouped boxplots).

    Returns
    -------
    pl.DataFrame
        Prepared data for boxplot visualization.
    """
    df_plot = (
        dtm_pl
        .unpivot(
            pl.selectors.numeric(),
            index="doc_id",
            variable_name="Tag",
            value_name="RF")
        .with_columns(pl.col("RF").mul(100))
        .filter(pl.col("Tag").is_in(box_vals))
        .with_columns(
            pl.col("doc_id").str.split_exact("_", 0)
            .struct.rename_fields(["cat_id"])
            .alias("id")
            )
        .unnest("id")
    )

    if grp_a is None and grp_b is None:
        df_plot = (df_plot
                   .drop("cat_id")
                   .with_columns(
                       pl.median("RF").over("Tag").alias("Median")
                       )
                   .sort(
                       ["Median", "Tag"],
                       descending=[True, False]
                       )
                   )

        return df_plot

    if grp_a is not None and grp_b is not None:
        grp_a_str = ", ".join(str(x) for x in grp_a)
        grp_b_str = ", ".join(str(x) for x in grp_b)

        df_plot = (df_plot
                   .with_columns(
                       pl.when(pl.col("cat_id").is_in(grp_a))
                       .then(pl.lit(grp_a_str))
                       .when(pl.col("cat_id").is_in(grp_b))
                       .then(pl.lit(grp_b_str))
                       .otherwise(pl.lit("Other"))
                       .alias("Group")
                       )
                   .drop("cat_id")
                   .filter(pl.col("Group") != "Other")
                   .with_columns(
                       pl.median("RF").over("Group", "Tag").alias("Median")
                       )
                   .sort(
                       ["Median", "Tag"],
                       descending=[True, False]
                       )
                   )

        return df_plot
