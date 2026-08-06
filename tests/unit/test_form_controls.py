"""Tests for shared Streamlit form controls."""

import polars as pl

from webapp.utilities.ui import form_controls


def test_simplified_frame_cache_isolates_table_and_transform() -> None:
    source = pl.DataFrame({"value": [1]})
    calls = {"tags": 0, "dtm": 0}

    def simplify_tags(df: pl.DataFrame) -> pl.DataFrame:
        calls["tags"] += 1
        return df.rename({"value": "Tag"})

    def simplify_dtm(df: pl.DataFrame) -> pl.DataFrame:
        calls["dtm"] += 1
        return df.rename({"value": "Noun"})

    form_controls._simplified_frame_cache.clear()

    tags = form_controls._apply_cached_simplify(
        source,
        "tt_pos",
        "Parts-of-Speech:General",
        simplify_tags,
    )
    dtm = form_controls._apply_cached_simplify(
        source,
        "dtm_pos",
        "Parts-of-Speech:General",
        simplify_dtm,
    )
    cached_tags = form_controls._apply_cached_simplify(
        source,
        "tt_pos",
        "Parts-of-Speech:General",
        simplify_tags,
    )

    assert tags.columns == ["Tag"]
    assert dtm.columns == ["Noun"]
    assert cached_tags is tags
    assert calls == {"tags": 1, "dtm": 1}
