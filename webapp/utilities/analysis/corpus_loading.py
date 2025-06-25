"""
Corpus loading utilities for data processing and validation.

This module provides functions for loading, validating, and processing
corpus data from various sources including internal databases and user uploads.
"""

import streamlit as st
from lingua import LanguageDetectorBuilder
from webapp.utilities.state import SessionKeys

# Warning constants
WARNING_CORRUPT_TARGET = 10
WARNING_CORRUPT_REFERENCE = 11
WARNING_DUPLICATE_REFERENCE = 21
WARNING_EXCLUDED_TARGET = 40
WARNING_EXCLUDED_REFERENCE = 41


@st.cache_resource(show_spinner=False)
def load_detector():
    """
    Load and cache the language detector.

    Returns
    -------
    LanguageDetector
        Configured language detector instance.
    """
    detector = LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()  # noqa: E501
    return detector

# Note: load_metadata and update_metadata functions moved to 
# webapp.utilities.session.metadata_handlers for centralization
