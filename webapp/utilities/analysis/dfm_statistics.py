"""Deterministic statistics for document-feature matrices."""

from dataclasses import dataclass
from typing import Literal

import docuscospacy as ds
import polars as pl


DfmStatistic = Literal[
    "mean_rf",
    "median_rf",
    "std_rf",
    "cv_rf",
    "range_pct",
    "dp_dispersion",
    "total_af",
]
RankAxis = Literal["features", "groups"]

STATISTIC_LABELS: dict[DfmStatistic, str] = {
    "mean_rf": "Mean RF",
    "median_rf": "Median RF",
    "std_rf": "Standard deviation of RF",
    "cv_rf": "Coefficient of variation of RF",
    "range_pct": "Document range",
    "dp_dispersion": "DP dispersion",
    "total_af": "Total AF",
}

STATISTIC_NOTES: dict[DfmStatistic, str] = {
    "mean_rf": "Mean RF is the average normalized feature frequency across documents.",
    "median_rf": (
        "Median RF is less affected by unusually high-frequency documents than the mean."
    ),
    "std_rf": (
        "Standard deviation of RF shows how much normalized frequency varies "
        "across documents."
    ),
    "cv_rf": (
        "Coefficient of variation divides RF standard deviation by mean RF; "
        "it helps compare variability for features with different average frequencies."
    ),
    "range_pct": (
        "Document range is the percent of documents where the feature appears "
        "at least once."
    ),
    "dp_dispersion": (
        "DP dispersion estimates how unevenly a feature is distributed across "
        "documents. Higher values suggest stronger concentration in fewer documents."
    ),
    "total_af": (
        "Total AF is the raw count across documents and is sensitive to corpus "
        "or group size."
    ),
}


@dataclass(frozen=True)
class DfmStatisticsResult:
    """Computed DFM statistics and a short deterministic explanation."""

    table: pl.DataFrame
    statistic: DfmStatistic
    rank_axis: RankAxis
    note: str


def get_dfm_statistic_options() -> dict[DfmStatistic, str]:
    """Return available DFM statistics for UI controls."""
    return dict(STATISTIC_LABELS)


def get_dfm_statistic_note(statistic: DfmStatistic) -> str:
    """Return a short explanation for a DFM statistic."""
    return STATISTIC_NOTES.get(statistic, STATISTIC_NOTES["mean_rf"])


def _feature_columns(dfm: pl.DataFrame) -> list[str]:
    return [column for column in dfm.columns if column not in {"doc_id", "Group"}]


def _with_group_column(dfm: pl.DataFrame, groups: list[str] | None) -> pl.DataFrame:
    if groups is None:
        return dfm
    if len(groups) != dfm.height:
        raise ValueError("Group labels must align with DFM rows.")
    if "Group" in dfm.columns:
        return dfm.with_columns(pl.Series("Group", groups))
    return dfm.insert_column(1, pl.Series("Group", groups))


def _filter_output_features(dfm: pl.DataFrame, tagset: str) -> pl.DataFrame:
    if tagset == "DocuScope" and "Untagged" in dfm.columns:
        return dfm.drop("Untagged")
    if tagset == "Parts-of-Speech" and "FU" in dfm.columns:
        return dfm.drop("FU")
    return dfm


def available_dfm_features(dfm: pl.DataFrame, tagset: str) -> list[str]:
    """Return selectable feature columns after tool-specific exclusions."""
    return _feature_columns(_filter_output_features(dfm, tagset))


def _select_features(
    dfm: pl.DataFrame,
    selected_features: list[str] | None,
) -> pl.DataFrame:
    if not selected_features:
        return dfm
    id_columns = [column for column in ["doc_id", "Group"] if column in dfm.columns]
    available = set(_feature_columns(dfm))
    missing = [feature for feature in selected_features if feature not in available]
    if missing:
        raise ValueError(f"Selected features are not available: {', '.join(missing)}")
    return dfm.select(id_columns + selected_features)


def _filter_groups(dfm: pl.DataFrame, selected_groups: list[str] | None) -> pl.DataFrame:
    if not selected_groups:
        return dfm
    if "Group" not in dfm.columns:
        raise ValueError("Group filtering requires metadata categories.")
    return dfm.filter(pl.col("Group").is_in(selected_groups))


def _long_weighted_dfm(
    dfm: pl.DataFrame,
    tagset: str,
    selected_features: list[str] | None = None,
) -> pl.DataFrame:
    group_values = dfm.get_column("Group") if "Group" in dfm.columns else None
    weight_input = dfm.drop("Group") if "Group" in dfm.columns else dfm
    weighted = ds.dtm_weight(weight_input, scheme="prop")
    if group_values is not None:
        weighted = weighted.insert_column(1, group_values)
    weighted = _filter_output_features(weighted, tagset)
    weighted = _select_features(weighted, selected_features)
    return weighted.unpivot(
        _feature_columns(weighted),
        index=[column for column in ["doc_id", "Group"] if column in weighted.columns],
        variable_name="Feature",
        value_name="RF",
    )


def _long_count_dfm(
    dfm: pl.DataFrame,
    tagset: str,
    selected_features: list[str] | None = None,
) -> pl.DataFrame:
    filtered = _filter_output_features(dfm, tagset)
    filtered = _select_features(filtered, selected_features)
    return filtered.unpivot(
        _feature_columns(filtered),
        index=[column for column in ["doc_id", "Group"] if column in filtered.columns],
        variable_name="Feature",
        value_name="AF",
    )


def _dp_expr() -> pl.Expr:
    observed = pl.when(pl.col("Feature_AF") > 0).then(
        pl.col("AF") / pl.col("Feature_AF")
    ).otherwise(0)
    expected = pl.when(pl.col("Corpus_AF") > 0).then(
        pl.col("Doc_AF") / pl.col("Corpus_AF")
    ).otherwise(0)
    return ((observed - expected).abs().sum() / 2).alias("dp_dispersion")


def _feature_dp(counts_long: pl.DataFrame, group_column: str | None = None) -> pl.DataFrame:
    keys = [group_column, "Feature"] if group_column else ["Feature"]
    doc_keys = [group_column, "doc_id"] if group_column else ["doc_id"]
    corpus_keys = [group_column] if group_column else None

    doc_totals = counts_long.group_by(doc_keys).agg(pl.col("AF").sum().alias("Doc_AF"))
    feature_totals = counts_long.group_by(keys).agg(pl.col("AF").sum().alias("Feature_AF"))
    if corpus_keys:
        corpus_totals = counts_long.group_by(corpus_keys).agg(
            pl.col("AF").sum().alias("Corpus_AF")
        )
    else:
        corpus_totals = counts_long.select(pl.col("AF").sum().alias("Corpus_AF"))

    joined = counts_long.join(doc_totals, on=doc_keys).join(feature_totals, on=keys)
    if corpus_keys:
        joined = joined.join(corpus_totals, on=corpus_keys)
    else:
        joined = joined.with_columns(pl.lit(corpus_totals.item()).alias("Corpus_AF"))

    return joined.group_by(keys).agg(_dp_expr())


def _feature_statistics(
    dfm: pl.DataFrame,
    tagset: str,
    group_column: str | None = None,
    selected_features: list[str] | None = None,
) -> pl.DataFrame:
    weighted_long = _long_weighted_dfm(dfm, tagset, selected_features=selected_features)
    counts_long = _long_count_dfm(dfm, tagset, selected_features=selected_features)
    dispersion_counts_long = _long_count_dfm(dfm, tagset)
    keys = [group_column, "Feature"] if group_column else ["Feature"]

    stats = weighted_long.group_by(keys).agg(
        pl.len().alias("n_docs"),
        pl.col("RF").mean().alias("mean_rf"),
        pl.col("RF").median().alias("median_rf"),
        pl.col("RF").std().fill_null(0).alias("std_rf"),
        (
            pl.col("RF").std().fill_null(0) / pl.col("RF").mean()
        ).fill_nan(None).alias("cv_rf"),
        ((pl.col("RF") > 0).sum() / pl.len() * 100).alias("range_pct"),
    )
    af = counts_long.group_by(keys).agg(pl.col("AF").sum().alias("total_af"))
    dp = _feature_dp(dispersion_counts_long, group_column=group_column)
    if selected_features:
        dp = dp.filter(pl.col("Feature").is_in(selected_features))
    return stats.join(af, on=keys).join(dp, on=keys)


def _group_statistics(
    dfm: pl.DataFrame,
    tagset: str,
    selected_features: list[str] | None = None,
) -> pl.DataFrame:
    feature_stats = _feature_statistics(
        dfm,
        tagset,
        group_column="Group",
        selected_features=selected_features,
    )
    return feature_stats.group_by("Group").agg(
        pl.len().alias("n_features"),
        pl.col("n_docs").max().alias("n_docs"),
        pl.col("mean_rf").mean().alias("mean_rf"),
        pl.col("median_rf").median().alias("median_rf"),
        pl.col("std_rf").mean().alias("std_rf"),
        pl.col("cv_rf").mean().alias("cv_rf"),
        pl.col("range_pct").mean().alias("range_pct"),
        pl.col("dp_dispersion").mean().alias("dp_dispersion"),
        pl.col("total_af").sum().alias("total_af"),
    )


def compute_dfm_statistics(
    dfm: pl.DataFrame,
    statistic: DfmStatistic,
    tagset: str,
    rank_axis: RankAxis = "features",
    groups: list[str] | None = None,
    selected_features: list[str] | None = None,
    selected_groups: list[str] | None = None,
    descending: bool = True,
    limit: int = 20,
) -> DfmStatisticsResult:
    """Compute deterministic statistics from a document-feature matrix."""
    if dfm is None or dfm.is_empty():
        raise ValueError("No DFM data is available.")
    if "doc_id" not in dfm.columns:
        raise ValueError("DFM must include a doc_id column.")
    if statistic not in STATISTIC_LABELS:
        raise ValueError(f"Unsupported statistic: {statistic}")
    if rank_axis not in {"features", "groups"}:
        raise ValueError(f"Unsupported rank axis: {rank_axis}")

    grouped_dfm = _with_group_column(dfm, groups)
    grouped_dfm = _filter_groups(grouped_dfm, selected_groups)
    if grouped_dfm.is_empty():
        raise ValueError("No DFM rows remain after applying group filters.")
    if rank_axis == "groups":
        if "Group" not in grouped_dfm.columns:
            raise ValueError("Group statistics require metadata categories.")
        table = _group_statistics(
            grouped_dfm,
            tagset,
            selected_features=selected_features,
        )
        sort_column = statistic
    else:
        table = _feature_statistics(
            grouped_dfm,
            tagset,
            group_column="Group" if "Group" in grouped_dfm.columns else None,
            selected_features=selected_features,
        )
        sort_column = statistic

    table = table.sort(sort_column, descending=descending).head(limit)
    return DfmStatisticsResult(
        table=table,
        statistic=statistic,
        rank_axis=rank_axis,
        note=get_dfm_statistic_note(statistic),
    )
