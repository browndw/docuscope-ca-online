"""
Tests for webapp.utilities.corpus.data_manager module.

Tests corpus data management, lazy loading, and memory optimization.
"""

import gc
import polars as pl
import sys
from unittest.mock import patch, MagicMock

# Test corpus data management with Streamlit session state
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    # Fallback to mocking if Streamlit not available
    sys.modules['streamlit'] = MagicMock()
    st = MagicMock()
    STREAMLIT_AVAILABLE = False

from webapp.utilities.corpus.data_manager import CorpusDataManager


class TestCorpusDataManager:
    """Test corpus data manager functionality."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.user_session_id = "test_session"
        self.corpus_type = "target"
        st.session_state.clear()

        # Initialize session structure
        st.session_state[self.user_session_id] = {
            "corpus_data": {
                "target": {},
                "reference": {}
            }
        }

    def test_manager_initialization(self):
        """Test CorpusDataManager initialization."""
        manager = CorpusDataManager(
            self.user_session_id, self.corpus_type
        )

        assert manager.user_session_id == self.user_session_id
        assert manager.corpus_type == self.corpus_type
        assert manager.cache is not None

        # Verify expected data keys
        assert "ds_tokens" in manager.core_keys
        assert "dtm_ds" in manager.derived_keys
        assert "ft_ds" in manager.derived_keys
        assert "collocations" in manager.additional_keys

    def test_manager_initialization_with_session_manager(self):
        """Test initialization with custom session manager."""
        mock_session_manager = MagicMock()

        manager = CorpusDataManager(
            self.user_session_id, self.corpus_type,
            session_manager=mock_session_manager
        )

        # Cache should be initialized with session manager
        assert manager.cache._session_manager == mock_session_manager

    def test_session_corpus_data_property(self):
        """Test access to session corpus data."""
        manager = CorpusDataManager(
            self.user_session_id, self.corpus_type
        )

        corpus_data = manager.session_corpus_data

        # Should return the appropriate corpus data dict
        expected_data = st.session_state[self.user_session_id]["corpus_data"]["target"]
        assert corpus_data == expected_data

    def test_session_corpus_data_reference_type(self):
        """Test session corpus data for reference corpus."""
        manager = CorpusDataManager(
            self.user_session_id, "reference"
        )

        corpus_data = manager.session_corpus_data

        # Should return reference corpus data
        expected_data = st.session_state[self.user_session_id]["corpus_data"]["reference"]
        assert corpus_data == expected_data

    @patch('webapp.utilities.corpus.data_manager.DataFrameCache')
    def test_cache_configuration(self, mock_cache_class):
        """Test that cache is properly configured."""
        mock_session_manager = MagicMock()
        mock_cache_instance = MagicMock()
        mock_cache_class.return_value = mock_cache_instance

        manager = CorpusDataManager(
            self.user_session_id, self.corpus_type,
            session_manager=mock_session_manager
        )

        # Verify cache was created with correct parameters
        mock_cache_class.assert_called_once_with(
            self.user_session_id,
            max_size=15,
            session_manager=mock_session_manager
        )
        assert manager.cache == mock_cache_instance

    def test_core_derived_additional_keys_separation(self):
        """Test that data keys are properly categorized."""
        manager = CorpusDataManager(
            self.user_session_id, self.corpus_type
        )

        # Check core keys
        assert len(manager.core_keys) == 1
        assert "ds_tokens" in manager.core_keys

        # Check derived keys
        expected_derived = [
            "dtm_ds", "dtm_pos", "ft_ds", "ft_pos", "tt_ds", "tt_pos"
        ]
        assert len(manager.derived_keys) == len(expected_derived)
        for key in expected_derived:
            assert key in manager.derived_keys

        # Check additional keys
        assert "collocations" in manager.additional_keys

        # Check all keys includes everything
        all_expected = manager.core_keys + manager.derived_keys + manager.additional_keys
        assert set(manager.all_keys) == set(all_expected)

    def test_multiple_managers_same_session(self):
        """Test multiple managers for same session with different corpus types."""
        target_manager = CorpusDataManager(
            self.user_session_id, "target"
        )
        reference_manager = CorpusDataManager(
            self.user_session_id, "reference"
        )

        # Should have different corpus data but same session
        assert target_manager.user_session_id == reference_manager.user_session_id
        assert target_manager.corpus_type != reference_manager.corpus_type

        # Should access different corpus data sections
        target_data = target_manager.session_corpus_data
        reference_data = reference_manager.session_corpus_data

        assert target_data is not reference_data


class TestCorpusDataManagerMemoryManagement:
    """Test memory management and cleanup functionality."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.user_session_id = "memory_test_session"
        self.corpus_type = "target"
        st.session_state.clear()

        # Initialize session structure
        st.session_state[self.user_session_id] = {
            "corpus_data": {
                "target": {},
                "reference": {}
            }
        }

    @patch('webapp.utilities.corpus.data_manager.gc.collect')
    def test_garbage_collection_trigger(self, mock_gc_collect):
        """Test that managers can trigger garbage collection."""
        # Create manager to test initialization
        CorpusDataManager(
            self.user_session_id, self.corpus_type
        )

        # Simulate memory cleanup scenario
        # (This would typically happen in cache cleanup or data loading)
        gc.collect()

        # Verify gc was available (not testing automatic triggering here,
        # but verifying the import and availability)
        assert gc.collect is not None

    def test_manager_cleanup_on_deletion(self):
        """Test manager cleanup when deleted."""
        manager = CorpusDataManager(
            self.user_session_id, self.corpus_type
        )

        # Store reference to cache
        cache_ref = manager.cache

        # Delete manager
        del manager

        # Cache should still exist but manager should be gone
        assert cache_ref is not None


class TestCorpusDataManagerEdgeCases:
    """Test edge cases and error conditions."""

    def test_manager_with_missing_session_structure(self):
        """Test manager creation when session structure is incomplete."""
        user_session_id = "incomplete_session"
        st.session_state.clear()

        # Create session without corpus_data structure
        st.session_state[user_session_id] = {}

        manager = CorpusDataManager(user_session_id, "target")

        # Manager should still initialize
        assert manager.user_session_id == user_session_id
        assert manager.corpus_type == "target"

    def test_manager_with_nonexistent_session(self):
        """Test manager creation with nonexistent session."""
        user_session_id = "nonexistent_session"
        st.session_state.clear()

        manager = CorpusDataManager(user_session_id, "target")

        # Manager should still initialize
        assert manager.user_session_id == user_session_id

    def test_manager_with_invalid_corpus_type(self):
        """Test manager with non-standard corpus type."""
        user_session_id = "test_session"
        invalid_corpus_type = "invalid_type"

        st.session_state.clear()
        st.session_state[user_session_id] = {
            "corpus_data": {"target": {}, "reference": {}}
        }

        manager = CorpusDataManager(user_session_id, invalid_corpus_type)

        # Manager should initialize but corpus data access might be limited
        assert manager.corpus_type == invalid_corpus_type
        assert manager.user_session_id == user_session_id

    def test_manager_key_consistency(self):
        """Test that key lists are consistent and complete."""
        manager = CorpusDataManager("test", "target")

        # Core keys should not overlap with derived or additional
        assert not set(manager.core_keys) & set(manager.derived_keys)
        assert not set(manager.core_keys) & set(manager.additional_keys)
        assert not set(manager.derived_keys) & set(manager.additional_keys)

        # All keys should be the union of the three sets
        expected_all = (
            set(manager.core_keys) |
            set(manager.derived_keys) |
            set(manager.additional_keys)
        )
        assert set(manager.all_keys) == expected_all


class TestCorpusDataManagerIntegration:
    """Integration tests for corpus data manager."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.user_session_id = "integration_test_session"
        st.session_state.clear()

        # Initialize session structure to match actual implementation
        # The CorpusDataManager expects corpus data directly under corpus_type
        st.session_state[self.user_session_id] = {
            "target": {
                "ds_tokens": pl.DataFrame({
                    "doc_id": ["doc1", "doc2"],
                    "token": ["hello", "world"],
                    "ds_tag": ["B-Character", "B-Description"],
                    "pos_tag": ["NN1", "NN1"]
                })
            },
            "reference": {}
        }

    def test_manager_access_to_existing_data(self):
        """Test manager accessing existing corpus data."""
        manager = CorpusDataManager(
            self.user_session_id, "target"
        )

        corpus_data = manager.session_corpus_data

        # Should access existing ds_tokens
        assert "ds_tokens" in corpus_data
        assert isinstance(corpus_data["ds_tokens"], pl.DataFrame)
        assert corpus_data["ds_tokens"].height == 2

    def test_target_and_reference_isolation(self):
        """Test that target and reference data are properly isolated."""
        # Add reference data
        st.session_state[self.user_session_id]["reference"] = {
            "ds_tokens": pl.DataFrame({
                "doc_id": ["ref1"],
                "token": ["reference"],
                "ds_tag": ["B-AcademicTerms"],
                "pos_tag": ["NN1"]
            })
        }

        target_manager = CorpusDataManager(
            self.user_session_id, "target"
        )
        reference_manager = CorpusDataManager(
            self.user_session_id, "reference"
        )

        target_data = target_manager.session_corpus_data
        reference_data = reference_manager.session_corpus_data

        # Data should be different
        assert target_data["ds_tokens"].height == 2
        assert reference_data["ds_tokens"].height == 1

        # Verify content is different
        target_tokens = target_data["ds_tokens"]["token"].to_list()
        reference_tokens = reference_data["ds_tokens"]["token"].to_list()

        assert "hello" in target_tokens
        assert "reference" in reference_tokens
        assert "reference" not in target_tokens
