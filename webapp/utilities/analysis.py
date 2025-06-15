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

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import pearsonr
from sklearn import decomposition


def subset_pl(
        tok_pl,
        select_ids: list
        ) -> pl.DataFrame:
    token_subset = (
        tok_pl
        .with_columns(
            pl.col("doc_id").str.split_exact("_", 0)
            .struct.rename_fields(["cat_id"])
            .alias("id")
        )
        .unnest("id")
        .filter(pl.col("cat_id").is_in(select_ids))
        .drop("cat_id")
        )
    return token_subset


def split_corpus(
        tok: dict,
        tar_list: list,
        ref_list: list
        ) -> tuple:

    tar_docs = {
        key: value for key, value in tok.items() if key.startswith
        (tuple(tar_list)
         )}
    ref_docs = {
        key: value for key, value in tok.items() if key.startswith(
            tuple(ref_list))}
    tar_ndocs = len(tar_docs)
    ref_ndocs = len(ref_docs)
    # get target counts
    tar_tok = list(tar_docs.values())
    tar_tags = []
    for i in range(0, len(tar_tok)):
        tags = [x[1] for x in tar_tok[i]]
        tar_tags.append(tags)
    tar_tags = [x for xs in tar_tags for x in xs]
    tar_tokens = len(tar_tags)
    tar_words = len([x for x in tar_tags if not x.startswith('Y')])

    # get reference counts
    ref_tok = list(ref_docs.values())
    ref_tags = []
    for i in range(0, len(ref_tok)):
        tags = [x[1] for x in ref_tok[i]]
        ref_tags.append(tags)
    ref_tags = [x for xs in ref_tags for x in xs]
    ref_tokens = len(ref_tags)
    ref_words = len([x for x in ref_tags if not x.startswith('Y')])
    return tar_docs, ref_docs, tar_words, ref_words, tar_tokens, ref_tokens, tar_ndocs, ref_ndocs  # noqa: E501


def freq_simplify_pl(
        frequency_table
        ) -> pl.DataFrame:
    """
    A function for aggregating part-of-speech tags \
        into more general lexical categories \
            returning the equivalent of the frequency_table function.

    :param frequency_table: A frequency table.
    :return: A polars DataFrame of token counts.
    """
    required_columns = {'Token', 'Tag', 'AF', 'RF', 'Range'}
    table_columns = set(frequency_table.columns)
    if not required_columns.issubset(table_columns):
        raise ValueError("""
                         Invalid DataFrame.
                         Expected a DataFrame produced by frequency_table
                         with columns: Token, Tag, AF, RF, Range.
                         """)
    tag_prefix = ["NN", "VV", "II"]
    if (not any(
        x.startswith(tuple(tag_prefix)) for x in
        frequency_table.get_column("Tag").to_list()
                )):
        raise ValueError("""
                         Invalid DataFrame.
                         Expected a frequency table with part-of-speech tags.
                         """)

    simple_df = (
        frequency_table
        .with_columns(
            pl.selectors.starts_with("Tag")
            .str.replace('^NN\\S*$', '#NounCommon')
            .str.replace('^VV\\S*$', '#VerbLex')
            .str.replace('^J\\S*$', '#Adjective')
            .str.replace('^R\\S*$', '#Adverb')
            .str.replace('^P\\S*$', '#Pronoun')
            .str.replace('^I\\S*$', '#Preposition')
            .str.replace('^C\\S*$', '#Conjunction')
            .str.replace('^N\\S*$', '#NounOther')
            .str.replace('^VB\\S*$', '#VerbBe')
            .str.replace('^V\\S*$', '#VerbOther')
        )
        .with_columns(
            pl.when(pl.selectors.starts_with("Tag").str.starts_with("#"))
            .then(pl.selectors.starts_with("Tag"))
            .otherwise(
                pl.selectors.starts_with("Tag").str.replace('^\\S+$', '#Other')
                ))
        .with_columns(
            pl.selectors.starts_with("Tag").str.replace("#", "")
        ))

    return simple_df


def dtm_simplify_grouped(
        dtm_pl
        ) -> pl.DataFrame:
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


def tags_table_grouped(
        df: pl.DataFrame
        ) -> pl.DataFrame:
    """
    Processes a document-feature-matrix to compute absolute frequency (AF),
    relative frequency (RF), and range
    (percentage of documents a feature occurs in)
    grouped by the 'Group' column.

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


def pca_contributions(
        dtm: pd.DataFrame,
        doccats: list
        ) -> tuple:

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
    contrib_x = list(map(', '.join, contrib_x))
    contrib_x = '; '.join(contrib_x)

    contrib_y = contrib_data[contrib_data[pca_y].abs() > mean_y].copy()
    contrib_y.sort_values(by=pca_y, ascending=False, inplace=True)
    contrib_y_values = contrib_y.loc[:, pca_y].tolist()
    contrib_y_values = ['%.2f' % y for y in contrib_y_values]
    contrib_y_values = [y + "%" for y in contrib_y_values]
    contrib_y_tags = contrib_y.loc[:, "Tag"].tolist()
    contrib_y = list(zip(contrib_y_tags, contrib_y_values))
    contrib_y = list(map(', '.join, contrib_y))
    contrib_y = '; '.join(contrib_y)

    contrib_1 = contrib_data[contrib_data[pca_x].abs() > 0].copy()
    contrib_1[pca_x] = contrib_1[pca_x].div(100)
    contrib_1.sort_values(by=pca_x, ascending=True, inplace=True)

    contrib_2 = contrib_data[contrib_data[pca_y].abs() > 0].copy()
    contrib_2[pca_y] = contrib_2[pca_y].div(100)
    contrib_2.sort_values(by=pca_y, ascending=True, inplace=True)

    ve_1 = "{:.2%}".format(variance[pca_idx - 1])
    ve_2 = "{:.2%}".format(variance[pca_idx])

    # For plotting: keep the filtered and sorted DataFrames
    contrib_1_plot = contrib_data[contrib_data[pca_x].abs() > 0][["Tag", pca_x]].copy()
    contrib_1_plot[pca_x] = contrib_1_plot[pca_x] / 100
    contrib_1_plot.sort_values(by=pca_x, ascending=True, inplace=True)

    contrib_2_plot = contrib_data[contrib_data[pca_y].abs() > 0][["Tag", pca_y]].copy()
    contrib_2_plot[pca_y] = contrib_2_plot[pca_y] / 100
    contrib_2_plot.sort_values(by=pca_y, ascending=True, inplace=True)

    return (
        pca_x, pca_y, contrib_x, contrib_y, ve_1, ve_2,
        contrib_1_plot, contrib_2_plot
    )


def correlation(
        df: pd.DataFrame,
        x: str,
        y: str
        ) -> dict:
    """
    Returns a dict with Pearson correlation for all points only.
    """
    cc = pearsonr(df[x], df[y])
    return {
        'all': {
            'df': len(df.index) - 2,
            'r': round(cc.statistic, 3),
            'p': round(cc.pvalue, 5)
        }
    }


def correlation_update(
        cc_dict, df: pd.DataFrame,
        x: str,
        y: str,
        group_col: str,
        highlight_groups: list
        ) -> dict:
    """
    Updates cc_dict with highlight and non-highlight group correlations.
    """
    # Highlight group
    df_high = df[df[group_col].isin(highlight_groups)]
    if len(df_high) > 2:
        cc_high = pearsonr(df_high[x], df_high[y])
        cc_dict['highlight'] = {
            'df': len(df_high.index) - 2,
            'r': round(cc_high.statistic, 3),
            'p': round(cc_high.pvalue, 5)
        }
    else:
        cc_dict['highlight'] = None

    # Non-highlight group
    df_non = df[~df[group_col].isin(highlight_groups)]
    if len(df_non) > 2:
        cc_non = pearsonr(df_non[x], df_non[y])
        cc_dict['non_highlight'] = {
            'df': len(df_non.index) - 2,
            'r': round(cc_non.statistic, 3),
            'p': round(cc_non.pvalue, 5)
        }
    else:
        cc_dict['non_highlight'] = None

    return cc_dict


def boxplots_pl(
        dtm_pl: pl.DataFrame,
        box_vals: list,
        grp_a=None,
        grp_b=None
        ) -> pl.DataFrame:

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


def scatterplots_pl(
        dtm_pl: pl.DataFrame,
        axis_vals: list
        ) -> pl.DataFrame:

    df_plot = (
        dtm_pl
        .unpivot(
            pl.selectors.numeric(),
            index="doc_id",
            variable_name="Tag",
            value_name="AF"
            )
        .with_columns(
            pl.when(pl.col("Tag").is_in(axis_vals))
            .then(pl.col("Tag"))
            .otherwise(pl.lit("Other"))
            .alias("Tag_Sort")
        )
        .with_columns(
            pl.col("doc_id").str.split_exact("_", 0)
            .struct.rename_fields(["cat_id"])
            .alias("id")
            )
        .unnest("id")
        .drop("cat_id", "Tag")
        .group_by(["doc_id", "Group", "Tag_Sort"]).sum()
        .with_columns(
            pl.col("AF")
            .truediv(pl.sum("AF")
                     .over("doc_id")
                     )
            .mul(100).alias("RF")
            )
        .filter(pl.col("Tag_Sort") != "Other")
        .rename({"Tag_Sort": "Tag"})
        .sort("doc_id", "Tag")
        .pivot("Tag", index=["doc_id", "Group"], values="RF")
    )

    return df_plot


def html_build_pl(tok_pl: pl.DataFrame,
                  doc_key: str):
    html_pos = (
        tok_pl
        .filter(pl.col("doc_id") == doc_key)
        .group_by(
            ["pos_id", "pos_tag"],
            maintain_order=True
            )
        .agg(pl.col("token").str.concat(""))
        .with_columns(
            pl.col("token").str.extract(r"(\s)$")
            .alias("ws"))
        .with_columns(pl.col("token").str.strip_chars())
        .with_columns(pl.col("token").str.len_chars()
                        .alias("tag_end"))
        .with_columns(
            pl.col("tag_end")
            .shift(1, fill_value=0)
            .alias("tag_start")
            )
        .with_columns(pl.col("tag_end").cum_sum())
        .with_columns(pl.col("tag_start").cum_sum())
        .with_columns(pl.col("ws").fill_null(""))
        .with_columns(
            pl.when(pl.col("pos_tag") != "Y")
            .then(pl.concat_str(pl.col("token"), pl.lit("</span>")))
            .otherwise(pl.col("token"))
            .alias("token_html"))
        .with_columns(
            pl.when(pl.col("pos_tag") != "Y")
            .then(
                pl.concat_str(pl.lit('<span class="'),
                              pl.col("pos_tag"), pl.lit('">'))
                )
            .otherwise(pl.lit(""))
            .alias("tag_html"))
        .with_columns(
            pl.concat_str(pl.col("tag_html"),
                          pl.col("token_html"),
                          pl.col("ws")).alias("Text")
            )
        .with_columns(pl.lit(doc_key).alias("doc_id"))
        .rename({"pos_tag": "Tag"})
        .select("doc_id", "token", "Tag", "tag_start", "tag_end", "Text")
    )

    html_simple = (
        tok_pl
        .filter(pl.col("doc_id") == doc_key)
        .group_by(
            ["pos_id", "pos_tag"],
            maintain_order=True
            )
        .agg(pl.col("token").str.concat(""))
        .with_columns(pl.col("pos_tag")
                      .str.replace(r'^NN\S*$', '#NounCommon')
                      .str.replace(r'^VV\S*$', '#VerbLex')
                      .str.replace(r'^J\S*$', '#Adjective')
                      .str.replace(r'^R\S*$', '#Adverb')
                      .str.replace(r'^P\S*$', '#Pronoun')
                      .str.replace(r'^I\S*$', '#Preposition')
                      .str.replace(r'^C\S*$', '#Conjunction')
                      .str.replace(r'^N\S*$', '#NounOther')
                      .str.replace(r'^VB\S*$', '#VerbBe')
                      .str.replace(r'^V\S*$', '#VerbOther'))
        .with_columns(
            pl.when(pl.col("pos_tag").str.starts_with("#"))
            .then(pl.col("pos_tag"))
            .otherwise(pl.col("pos_tag").str.replace(r'^\S+$', '#Other')))
        .with_columns(
            pl.col("pos_tag").str.replace("#", ""))
        .with_columns(
            pl.col("token").str.extract(r"(\s)$")
            .alias("ws"))
        .with_columns(
            pl.col("token").str.strip_chars())
        .with_columns(
            pl.col("token").str.len_chars()
            .alias("tag_end"))
        .with_columns(
            pl.col("tag_end").shift(1, fill_value=0)
            .alias("tag_start"))
        .with_columns(
            pl.col("tag_end").cum_sum())
        .with_columns(
            pl.col("tag_start").cum_sum())
        .with_columns(
            pl.col("ws").fill_null(""))
        .with_columns(
            pl.when(pl.col("pos_tag") != "Other")
            .then(pl.concat_str(pl.col("token"), pl.lit("</span>")))
            .otherwise(pl.col("token"))
            .alias("token_html"))
        .with_columns(
            pl.when(pl.col("pos_tag") != "Other")
            .then(
                pl.concat_str(pl.lit('<span class="'),
                              pl.col("pos_tag"),
                              pl.lit('">')))
            .otherwise(pl.lit(""))
            .alias("tag_html"))
        .with_columns(
            pl.concat_str(pl.col("tag_html"),
                          pl.col("token_html"),
                          pl.col("ws")).alias("Text"))
        .with_columns(pl.lit(doc_key).alias("doc_id"))
        .rename({"pos_tag": "Tag"})
        .select("doc_id", "token", "Tag", "tag_start", "tag_end", "Text")
    )

    html_ds = (
        tok_pl
        .filter(pl.col("doc_id") == doc_key)
        .group_by(
            ["ds_id", "ds_tag"],
            maintain_order=True
            )
        .agg(pl.col("token").str.concat(""))
        .with_columns(
            pl.col("token").str.extract(r"(\s)$")
            .alias("ws"))
        .with_columns(
            pl.col("token").str.strip_chars()
            )
        .with_columns(
            pl.col("token").str.len_chars()
            .alias("tag_end"))
        .with_columns(
            pl.col("tag_end").shift(1, fill_value=0)
            .alias("tag_start"))
        .with_columns(
            pl.col("tag_end").cum_sum()
            )
        .with_columns(
            pl.col("tag_start").cum_sum()
            )
        .with_columns(
            pl.col("ws").fill_null("")
            )
        .with_columns(
            pl.when(pl.col("ds_tag") != "Untagged")
            .then(pl.concat_str(pl.col("token"), pl.lit("</span>")))
            .otherwise(pl.col("token"))
            .alias("token_html"))
        .with_columns(
            pl.when(pl.col("ds_tag") != "Untagged")
            .then(pl.concat_str(pl.lit('<span class="'),
                                pl.col("ds_tag"),
                                pl.lit('">')))
            .otherwise(pl.lit(""))
            .alias("tag_html"))
        .with_columns(
            pl.concat_str(pl.col("tag_html"),
                          pl.col("token_html"),
                          pl.col("ws")).alias("Text"))
        .with_columns(pl.lit(doc_key).alias("doc_id"))
        .rename({"ds_tag": "Tag"})
        .select("doc_id", "token", "Tag", "tag_start", "tag_end", "Text")
    )

    return html_pos, html_simple, html_ds
