"""
Grouped analysis utilities for corpus analysis.

This module provides functions for analyzing corpus data with grouping,
including grouped DTM operations and grouped tag frequency tables.
"""

import polars as pl


def dtm_simplify_grouped(dtm_pl: pl.DataFrame) -> pl.DataFrame:
    """
    Simplify a grouped document-term matrix by collapsing POS tags into broader categories.

    Parameters
    ----------
    dtm_pl : pl.DataFrame
        A grouped document-term matrix with 'doc_id' and 'Group' columns.

    Returns
    -------
    pl.DataFrame
        A simplified grouped DTM with broader POS categories.
    """
    simple_df = (
        dtm_pl
        .unpivot(pl.selectors.numeric(), index=["doc_id", "Group"])
        .with_columns(
            pl.col("variable")
            .str.replace(r'^NN\S*$', '#NounCommon')
            .str.replace(r'^VV\S*$', '#VerbLex')
            .str.replace(r'^J\S*$', '#Adjective')
            .str.replace(r'^R\S*$', '#Adverb')
            .str.replace(r'^P\S*$', '#Pronoun')
            .str.replace(r'^I\S*$', '#Preposition')
            .str.replace(r'^C\S*$', '#Conjunction')
            .str.replace(r'^N\S*$', '#NounOther')
            .str.replace(r'^VB\S*$', '#VerbBe')
            .str.replace(r'^V\S*$', '#VerbOther')
        )
        .with_columns(
            pl.when(pl.col("variable").str.starts_with("#"))
            .then(pl.col("variable"))
            .otherwise(pl.col("variable").str.replace(r'^\S+$', '#Other'))
            )
        .with_columns(
            pl.col("variable").str.replace("#", "")
        )
        .group_by(["doc_id", "Group", "variable"], maintain_order=True).sum()
        .pivot(index=["doc_id", "Group"], on="variable", values="value")
        )

    return simple_df


def tags_table_grouped(df: pl.DataFrame) -> pl.DataFrame:
    """
    Process a document-feature-matrix to compute absolute frequency (AF),
    relative frequency (RF), and range grouped by the 'Group' column.

    Parameters
    ----------
    df : pl.DataFrame
        A Polars DataFrame where the first column is 'doc_id',
        the second column is 'Group',
        and the remaining columns are raw counts of features per 'doc_id'.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame with columns:
        - 'Tag': Feature names (numeric column names from the input DataFrame).
        - 'Group': Group names.
        - 'AF': Absolute frequency of the tag by group.
        - 'RF': Relative frequency of the tag by group.
        - 'Range': Percentage of documents a feature occurs in by group.
    """
    # Ensure the first two columns are 'doc_id' and 'Group'
    if df.columns[:2] != ['doc_id', 'Group']:
        raise ValueError("""
                         The first column must be 'doc_id' \
                         and the second column must be 'Group'.
                         """)

    # Unpivot the DataFrame to long format for easier aggregation
    unpivoted = df.unpivot(
        pl.selectors.numeric(),
        index=['doc_id', 'Group'],
        variable_name="Tag",
        value_name="Count"
    )

    # Compute absolute frequency (AF) by summing counts for each Tag and Group
    af = (
        unpivoted.group_by(["Group", "Tag"])
        .agg(pl.sum("Count").alias("AF"))
    )

    # Compute total counts per group for relative frequency (RF)
    group_totals = (
        unpivoted.group_by("Group")
        .agg(pl.sum("Count").alias("Group_Total"))
    )

    # Join group totals to compute RF
    af = af.join(group_totals, on="Group")
    af = af.with_columns(
        (pl.col("AF") / pl.col("Group_Total")).alias("RF")
        )

    # Compute range (percentage of documents a feature occurs in by group)
    doc_counts = (
        unpivoted.filter(pl.col("Count") > 0)
        .group_by(["Group", "Tag"])
        .agg(pl.n_unique("doc_id").alias("Doc_Count"))
    )

    total_docs_per_group = (
        unpivoted.group_by("Group")
        .agg(pl.n_unique("doc_id").alias("Total_Docs"))
    )

    range_df = doc_counts.join(total_docs_per_group, on="Group")
    range_df = range_df.with_columns(
        ((pl.col("Doc_Count") / pl.col("Total_Docs")) * 100).alias("Range")
    )

    # Combine AF, RF, and Range into the final DataFrame
    result = af.join(
        range_df.select(["Group", "Tag", "Range"]),
        on=["Group", "Tag"], how="full"
        ).with_columns(pl.col("Range").fill_null(0))

    # Select and reorder columns for the final output
    result = result.select(["Tag", "Group", "AF", "RF", "Range"])
    result = result.sort(["RF", "Group"], descending=[True, True])

    return result
