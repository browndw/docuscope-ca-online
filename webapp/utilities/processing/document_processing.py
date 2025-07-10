"""
Document processing utilities for single document analysis.

This module provides functions for processing individual documents
and generating HTML representations with various tag highlighting.
"""

import streamlit as st
import polars as pl
from webapp.utilities.core import app_core
from webapp.utilities.session import get_or_init_user_session, safe_session_get
from webapp.utilities.state import SessionKeys
from webapp.utilities.corpus import get_corpus_data_manager


def generate_document_html(
        user_session_id: str,
        doc_key: str
        ) -> None:
    """
    Process a single document and generate HTML representations.

    Parameters
    ----------
    user_session_id : str
        The session ID for the user.
    doc_key : str
        The document key or identifier.

    Returns
    -------
    None
    """
    # --- Check if target corpus is loaded ---
    user_session_id, session = get_or_init_user_session()
    if safe_session_get(session, SessionKeys.HAS_TARGET, None) is False:
        st.session_state[user_session_id]["doc_warning"] = (
            "No target corpus loaded. Please load a document first.",
            ":material/info:"
        )
        return

    # --- Try to get the target tokens table using corpus data manager ---
    try:
        manager = get_corpus_data_manager(user_session_id, "target")
        tok_pl = manager.get_data("ds_tokens")

        if tok_pl is None:
            raise KeyError("No tokens data available")

    except (KeyError, Exception):
        st.session_state[user_session_id]["doc_warning"] = (
            "No tokens found in the target corpus.",
            ":material/info:"
        )
        return

    if tok_pl is None or getattr(tok_pl, "height", 0) == 0:
        st.session_state[user_session_id]["doc_warning"] = (
            "No tokens found in the target corpus.",
            ":material/info:"
        )
        return

    # --- Generate HTML representations ---
    try:
        doc_pos, doc_simple, doc_ds = html_build_pl(tok_pl, doc_key)
    except Exception as e:
        st.session_state[user_session_id]["doc_warning"] = (
            f"Failed to process document: {e}",
            ":material/sentiment_stressed:"
        )
        return

    # --- Save results ---
    st.session_state[user_session_id]["target"]["doc_pos"] = doc_pos
    st.session_state[user_session_id]["target"]["doc_simple"] = doc_simple
    st.session_state[user_session_id]["target"]["doc_ds"] = doc_ds

    app_core.session_manager.update_session_state(user_session_id, 'doc', True)
    st.session_state[user_session_id]["doc_warning"] = None
    st.success('Document processed!')
    st.rerun()


# --- HTML generation function ---
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
        .agg(pl.col("token").str.join(""))
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
