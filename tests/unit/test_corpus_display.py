import pytest
import streamlit as st

from webapp.utilities.state import SessionKeys
from webapp.utilities.ui.corpus_display import (
    extract_keyness_parts,
    extract_keyness_parts_settings,
    reference_parts,
    target_parts,
)
from webapp.utilities.ui.form_controls import keyness_settings_info


KEYNESS_PARTS = [
    ["BIO"],
    ["ENG"],
    "100",
    "200",
    "80",
    "160",
    "10",
    "20",
]


@pytest.mark.parametrize(
    "stored_value",
    [
        KEYNESS_PARTS,
        {"temp": [KEYNESS_PARTS]},
        [{"temp": KEYNESS_PARTS}],
    ],
)
def test_extract_keyness_parts_supports_current_and_legacy_shapes(stored_value):
    metadata = {SessionKeys.KEYNESS_PARTS: stored_value}

    assert extract_keyness_parts(metadata) == KEYNESS_PARTS


def test_extract_keyness_parts_rejects_incomplete_metadata():
    with pytest.raises(ValueError, match="missing or incomplete"):
        extract_keyness_parts({SessionKeys.KEYNESS_PARTS: {"temp": []}})


def test_current_keyness_parts_metadata_renders_bio_and_eng_summaries():
    metadata = {SessionKeys.KEYNESS_PARTS: {"temp": [KEYNESS_PARTS]}}

    keyness_parts = extract_keyness_parts(metadata)

    assert "Document categories: ['BIO']" in target_parts(keyness_parts)
    assert "Document categories: ['ENG']" in reference_parts(keyness_parts)


def test_legacy_keyness_parts_settings_default_to_not_swapped():
    assert extract_keyness_parts_settings(KEYNESS_PARTS) == (0.01, False)


def test_keyness_parts_settings_do_not_inherit_whole_corpus_swap():
    keyness_parts = [*KEYNESS_PARTS, 0.05, False]
    st.session_state["settings-test"] = {"pval_threshold": 0.01, "swap_target": True}

    settings = extract_keyness_parts_settings(keyness_parts)

    assert settings == (0.05, False)
    assert keyness_settings_info("settings-test", *settings).endswith("**Swapped:** No")
