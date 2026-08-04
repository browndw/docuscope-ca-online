"""Tests for deterministic DFM statistics."""

import polars as pl
import pytest

from webapp.utilities.analysis.dfm_statistics import (
    available_dfm_features,
    compute_dfm_statistics,
    get_dfm_statistic_options,
)


def test_compute_dfm_statistics_excludes_docuscope_untagged_after_weighting():
    dfm = pl.DataFrame({
        "doc_id": ["BIO_1", "ENG_1", "ENG_2"],
        "AcademicTerms": [10, 2, 0],
        "Narrative": [0, 3, 1],
        "Untagged": [90, 5, 9],
    })

    result = compute_dfm_statistics(
        dfm=dfm,
        statistic="mean_rf",
        tagset="DocuScope",
        limit=10,
    )

    assert "mean_rf" in get_dfm_statistic_options()
    assert result.statistic == "mean_rf"
    assert "Untagged" not in result.table.get_column("Feature").to_list()
    academic = result.table.filter(pl.col("Feature") == "AcademicTerms").row(0, named=True)
    assert academic["total_af"] == 12
    assert academic["range_pct"] == pytest.approx(100 * 2 / 3)
    assert academic["mean_rf"] == pytest.approx(((10 / 100) + (2 / 10) + 0) / 3)


def test_compute_dfm_statistics_ranks_features_by_dp_dispersion():
    dfm = pl.DataFrame({
        "doc_id": ["D1", "D2", "D3"],
        "Concentrated": [9, 0, 0],
        "Even": [3, 3, 3],
        "Filler": [0, 9, 9],
    })

    result = compute_dfm_statistics(
        dfm=dfm,
        statistic="dp_dispersion",
        tagset="DocuScope",
        limit=3,
    )

    assert result.table.get_column("Feature").to_list()[0] == "Concentrated"
    concentrated = result.table.filter(pl.col("Feature") == "Concentrated").row(
        0,
        named=True,
    )
    even = result.table.filter(pl.col("Feature") == "Even").row(0, named=True)
    assert concentrated["dp_dispersion"] > even["dp_dispersion"]
    assert "unevenly" in result.note


def test_compute_dfm_statistics_can_rank_groups_by_rf_variability():
    dfm = pl.DataFrame({
        "doc_id": ["BIO_1", "BIO_2", "ENG_1", "ENG_2"],
        "AcademicTerms": [8, 0, 2, 2],
        "Narrative": [0, 8, 2, 2],
    })

    result = compute_dfm_statistics(
        dfm=dfm,
        statistic="std_rf",
        tagset="DocuScope",
        rank_axis="groups",
        groups=["BIO", "BIO", "ENG", "ENG"],
        limit=1,
    )

    assert result.rank_axis == "groups"
    assert result.table.get_column("Group").to_list() == ["BIO"]
    assert result.table.get_column("std_rf").item() > 0


def test_compute_dfm_statistics_requires_groups_for_group_ranking():
    dfm = pl.DataFrame({"doc_id": ["D1"], "AcademicTerms": [1]})

    with pytest.raises(ValueError, match="metadata categories"):
        compute_dfm_statistics(
            dfm=dfm,
            statistic="std_rf",
            tagset="DocuScope",
            rank_axis="groups",
        )


def test_compute_dfm_statistics_filters_selected_features_after_weighting():
    dfm = pl.DataFrame({
        "doc_id": ["D1", "D2"],
        "Noun": [5, 0],
        "Pronoun": [0, 5],
        "Untagged": [5, 5],
    })

    result = compute_dfm_statistics(
        dfm=dfm,
        statistic="mean_rf",
        tagset="DocuScope",
        selected_features=["Noun"],
    )

    assert available_dfm_features(dfm, "DocuScope") == ["Noun", "Pronoun"]
    assert result.table.get_column("Feature").to_list() == ["Noun"]
    noun = result.table.row(0, named=True)
    assert noun["mean_rf"] == pytest.approx(((5 / 10) + 0) / 2)


def test_selected_feature_dp_uses_full_dfm_distribution():
    dfm = pl.DataFrame({
        "doc_id": ["D1", "D2", "D3"],
        "Concentrated": [9, 0, 0],
        "Even": [3, 3, 3],
        "Filler": [0, 9, 9],
    })

    result = compute_dfm_statistics(
        dfm=dfm,
        statistic="dp_dispersion",
        tagset="DocuScope",
        selected_features=["Concentrated"],
        limit=1,
    )

    row = result.table.row(0, named=True)
    assert row["Feature"] == "Concentrated"
    assert row["dp_dispersion"] > 0


def test_compute_dfm_statistics_filters_groups_and_can_sort_lowest():
    dfm = pl.DataFrame({
        "doc_id": ["BIO_1", "BIO_2", "ENG_1", "ENG_2", "HIS_1", "HIS_2"],
        "Pronoun": [5, 0, 2, 2, 4, 1],
        "Noun": [0, 5, 2, 2, 1, 4],
    })

    result = compute_dfm_statistics(
        dfm=dfm,
        statistic="std_rf",
        tagset="Parts-of-Speech",
        rank_axis="groups",
        groups=["STEM", "STEM", "Humanities", "Humanities", "Humanities", "Humanities"],
        selected_features=["Pronoun"],
        selected_groups=["Humanities", "STEM"],
        descending=False,
        limit=1,
    )

    assert result.table.get_column("Group").to_list() == ["Humanities"]
    assert result.table.get_column("n_features").to_list() == [1]
