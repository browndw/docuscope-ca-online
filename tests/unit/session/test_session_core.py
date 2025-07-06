"""
Tests for webapp.utilities.session.session_core module.

Tests session initialization, updates, and management functionality.
"""

import polars as pl
import sys
from unittest.mock import patch, MagicMock

# Mock Streamlit to avoid import issues
try:
    import streamlit as st
except ImportError:
    # Fallback to mocking if Streamlit not available
    sys.modules['streamlit'] = MagicMock()
    st = MagicMock()

from webapp.utilities.session.session_core import (
    init_session,
    update_session,
    get_corpus_categories
)
from webapp.utilities.state import SessionKeys


class TestInitSession:
    """Test session initialization functionality."""

    @patch('streamlit.session_state', {})
    def test_init_session_creates_default_session(self):
        """Test that init_session creates a session with default values."""
        session_id = "test_session"
        st.session_state[session_id] = {}

        init_session(session_id)

        assert session_id in st.session_state
        assert "session" in st.session_state[session_id]

        session_df = st.session_state[session_id]["session"]
        assert isinstance(session_df, pl.DataFrame)

        # Check all expected keys are present with default values
        session_dict = session_df.to_dict(as_series=False)

        expected_defaults = {
            SessionKeys.HAS_TARGET: [False],
            SessionKeys.TARGET_DB: [''],
            SessionKeys.HAS_META: [False],
            SessionKeys.HAS_REFERENCE: [False],
            SessionKeys.REFERENCE_DB: [''],
            SessionKeys.FREQ_TABLE: [False],
            SessionKeys.TAGS_TABLE: [False],
            SessionKeys.KEYNESS_TABLE: [False],
            SessionKeys.NGRAMS: [False],
            SessionKeys.KWIC: [False],
            SessionKeys.KEYNESS_PARTS: [False],
            SessionKeys.DTM: [False],
            SessionKeys.PCA: [False],
            SessionKeys.COLLOCATIONS: [False],
            SessionKeys.DOC: [False],
        }

        for key, expected_value in expected_defaults.items():
            assert session_dict[key] == expected_value

    @patch('streamlit.session_state', {})
    def test_init_session_overwrites_existing_session(self):
        """Test that init_session overwrites an existing session."""
        session_id = "test_session"
        st.session_state[session_id] = {
            "session": {"existing": "data"}
        }

        init_session(session_id)

        session_df = st.session_state[session_id]["session"]
        session_dict = session_df.to_dict(as_series=False)

        # Should not contain old data
        assert "existing" not in session_dict
        # Should contain new default data
        assert SessionKeys.HAS_TARGET in session_dict


class TestUpdateSession:
    """Test session update functionality."""

    def setup_method(self):
        """Set up test session before each test."""
        self.session_id = "test_session"
        st.session_state.clear()
        st.session_state[self.session_id] = {}
        init_session(self.session_id)

    @patch('streamlit.session_state')
    def test_update_session_dataframe_format(self, mock_session_state):
        """Test updating session when stored as DataFrame."""
        # Create a mock DataFrame session
        session_data = {SessionKeys.HAS_TARGET: [False], SessionKeys.TARGET_DB: ['']}
        mock_df = pl.from_dict(session_data)
        mock_df.to_dict = MagicMock(return_value={
            SessionKeys.HAS_TARGET: False,
            SessionKeys.TARGET_DB: ''
        })
        mock_df.columns = ['has_target', 'target_db']

        mock_session_state.__getitem__.return_value = {"session": mock_df}
        mock_session_state.__setitem__ = MagicMock()

        with patch('polars.from_dict') as mock_from_dict:
            mock_new_df = MagicMock()
            mock_from_dict.return_value = mock_new_df

            update_session(SessionKeys.HAS_TARGET, True, self.session_id)

            # Verify the session was updated and converted back to DataFrame
            mock_from_dict.assert_called_once()
            call_args = mock_from_dict.call_args[0][0]
            assert call_args[SessionKeys.HAS_TARGET] is True

    @patch('streamlit.session_state')
    def test_update_session_dict_format(self, mock_session_state):
        """Test updating session when stored as dictionary."""
        session_dict = {SessionKeys.HAS_TARGET: False, SessionKeys.TARGET_DB: ''}

        # Mock the session structure
        mock_session_data = {"session": session_dict}
        mock_session_state.__getitem__.return_value = mock_session_data

        update_session(SessionKeys.HAS_TARGET, True, self.session_id)

        # Verify the session dict was updated in place
        expected_dict = {SessionKeys.HAS_TARGET: True, SessionKeys.TARGET_DB: ''}
        assert mock_session_data["session"] == expected_dict

    def test_update_session_creates_key_if_not_exists(self):
        """Test that update_session creates new keys if they don't exist."""
        new_key = "new_test_key"
        new_value = "test_value"

        update_session(new_key, new_value, self.session_id)

        session_df = st.session_state[self.session_id]["session"]
        session_dict = session_df.to_dict(as_series=False)

        assert new_key in session_dict
        assert session_dict[new_key] == [new_value]

    def test_update_session_boolean_values(self):
        """Test updating session with boolean values."""
        update_session(SessionKeys.HAS_TARGET, True, self.session_id)

        session_df = st.session_state[self.session_id]["session"]
        session_dict = session_df.to_dict(as_series=False)

        assert session_dict[SessionKeys.HAS_TARGET] == [True]

    def test_update_session_string_values(self):
        """Test updating session with string values."""
        test_db_path = "/path/to/test/database"
        update_session(SessionKeys.TARGET_DB, test_db_path, self.session_id)

        session_df = st.session_state[self.session_id]["session"]
        session_dict = session_df.to_dict(as_series=False)

        assert session_dict[SessionKeys.TARGET_DB] == [test_db_path]


class TestGetCorpusCategories:
    """Test corpus categories functionality."""

    def setup_method(self):
        """Set up test session before each test."""
        self.session_id = "test_session"
        st.session_state.clear()
        st.session_state[self.session_id] = {}

    @patch('webapp.utilities.session.session_core.get_doc_cats')
    def test_get_corpus_categories_calculates_and_caches(self, mock_get_doc_cats):
        """Test that corpus categories are calculated and cached."""
        doc_ids = ["doc1", "doc2", "doc3"]
        mock_categories = ["fiction", "academic", "fiction"]
        mock_get_doc_cats.return_value = mock_categories

        result = get_corpus_categories(doc_ids, self.session_id)

        expected_categories = mock_categories
        expected_unique_count = 2  # "fiction" and "academic"
        expected_result = (expected_categories, expected_unique_count)

        assert result == expected_result
        mock_get_doc_cats.assert_called_once_with(doc_ids)

        # Check that result is cached
        cache_key = f"corpus_categories_{self.session_id}"
        assert cache_key in st.session_state[self.session_id]
        assert st.session_state[self.session_id][cache_key] == expected_result

    @patch('webapp.utilities.session.session_core.get_doc_cats')
    def test_get_corpus_categories_returns_cached_result(self, mock_get_doc_cats):
        """Test that cached results are returned without recalculation."""
        doc_ids = ["doc1", "doc2"]
        cached_result = (["fiction", "academic"], 2)
        cache_key = f"corpus_categories_{self.session_id}"

        # Pre-populate cache
        st.session_state[self.session_id][cache_key] = cached_result

        result = get_corpus_categories(doc_ids, self.session_id)

        assert result == cached_result
        # get_doc_cats should not be called when cache hit
        mock_get_doc_cats.assert_not_called()

    @patch('webapp.utilities.session.session_core.get_doc_cats')
    def test_get_corpus_categories_empty_doc_ids(self, mock_get_doc_cats):
        """Test corpus categories with empty document list."""
        doc_ids = []
        mock_get_doc_cats.return_value = []

        result = get_corpus_categories(doc_ids, self.session_id)

        expected_result = ([], 0)
        assert result == expected_result

    @patch('webapp.utilities.session.session_core.get_doc_cats')
    def test_get_corpus_categories_none_doc_cats(self, mock_get_doc_cats):
        """Test corpus categories when get_doc_cats returns None."""
        doc_ids = ["doc1"]
        mock_get_doc_cats.return_value = None

        result = get_corpus_categories(doc_ids, self.session_id)

        expected_result = (None, 0)
        assert result == expected_result

    @patch('webapp.utilities.session.session_core.get_doc_cats')
    def test_get_corpus_categories_single_category(self, mock_get_doc_cats):
        """Test corpus categories with documents of same category."""
        doc_ids = ["doc1", "doc2", "doc3"]
        mock_categories = ["fiction", "fiction", "fiction"]
        mock_get_doc_cats.return_value = mock_categories

        result = get_corpus_categories(doc_ids, self.session_id)

        expected_result = (mock_categories, 1)  # Only one unique category
        assert result == expected_result


class TestSessionIntegration:
    """Integration tests for session management."""

    def setup_method(self):
        """Set up test session before each test."""
        self.session_id = "unit_test_session"
        st.session_state.clear()
        st.session_state[self.session_id] = {}

    def test_init_and_update_workflow(self):
        """Test complete workflow of initializing and updating session."""
        # Initialize session
        init_session(self.session_id)

        # Update multiple values
        update_session(SessionKeys.HAS_TARGET, True, self.session_id)
        update_session(SessionKeys.TARGET_DB, "/path/to/db", self.session_id)
        update_session(SessionKeys.HAS_META, True, self.session_id)

        # Verify all updates are present
        session_df = st.session_state[self.session_id]["session"]
        session_dict = session_df.to_dict(as_series=False)

        assert session_dict[SessionKeys.HAS_TARGET] == [True]
        assert session_dict[SessionKeys.TARGET_DB] == ["/path/to/db"]
        assert session_dict[SessionKeys.HAS_META] == [True]
        # Unchanged values should remain default
        assert session_dict[SessionKeys.HAS_REFERENCE] == [False]

    @patch('webapp.utilities.session.session_core.get_doc_cats')
    def test_multiple_session_isolation(self, mock_get_doc_cats):
        """Test that multiple sessions are properly isolated."""
        session_id_1 = "session_1"
        session_id_2 = "session_2"

        st.session_state[session_id_1] = {}
        st.session_state[session_id_2] = {}

        # Initialize both sessions
        init_session(session_id_1)
        init_session(session_id_2)

        # Update session 1
        update_session(SessionKeys.HAS_TARGET, True, session_id_1)

        # Verify session 2 is unchanged
        session_2_df = st.session_state[session_id_2]["session"]
        session_2_dict = session_2_df.to_dict(as_series=False)
        assert session_2_dict[SessionKeys.HAS_TARGET] == [False]

        # Test corpus categories isolation
        mock_get_doc_cats.return_value = ["fiction"]

        # Call get_corpus_categories for both sessions to test isolation
        get_corpus_categories(["doc1"], session_id_1)
        get_corpus_categories(["doc2"], session_id_2)

        # Both should have been calculated independently
        assert mock_get_doc_cats.call_count == 2

        # Verify caching is session-specific
        cache_key_1 = f"corpus_categories_{session_id_1}"
        cache_key_2 = f"corpus_categories_{session_id_2}"

        assert cache_key_1 in st.session_state[session_id_1]
        assert cache_key_2 in st.session_state[session_id_2]
        assert cache_key_1 not in st.session_state[session_id_2]
        assert cache_key_2 not in st.session_state[session_id_1]
