"""
Tests for webapp.utilities.processing.corpus_processing module.

Tests corpus processing workflows, memory management, and data handling.
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

from webapp.utilities.processing.corpus_processing import (
    finalize_corpus_load,
    finalize_corpus_load_optimized,
    process_new,
)
from webapp.utilities.state import SessionKeys


class TestFinalizeCorpusLoad:
    """Test corpus finalization functionality."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.user_session_id = "test_session"
        self.test_tokens = pl.DataFrame({
            "doc_id": ["doc1", "doc2"],
            "token": ["hello", "world"],
            "tag": ["Character", "Description"],
            "pos": ["UH", "NN1"]
        })

    @patch('webapp.utilities.processing.corpus_processing.ds.frequency_table')
    @patch('webapp.utilities.processing.corpus_processing.ds.tags_table')
    @patch('webapp.utilities.processing.corpus_processing.ds.tags_dtm')
    @patch('webapp.utilities.processing.corpus_processing.load_corpus_new')
    @patch('webapp.utilities.processing.corpus_processing.cleanup_original_corpus_data')
    @patch('webapp.utilities.processing.corpus_processing.init_metadata_target')
    @patch('webapp.utilities.processing.corpus_processing.app_core')
    @patch('streamlit.rerun')
    def test_finalize_corpus_load_target(
        self, mock_rerun, mock_app_core, mock_init_metadata, mock_cleanup,
        mock_load_corpus, mock_dtm, mock_tags_table, mock_freq_table
    ):
        """Test finalizing target corpus load."""
        # Mock return values - each function returns a tuple of 2 DataFrames
        mock_freq_table.return_value = (
            pl.DataFrame({"token": ["hello"], "frequency": [1]}),  # ft_pos
            pl.DataFrame({"tag": ["Character"], "frequency": [1]})  # ft_ds
        )
        mock_tags_table.return_value = (
            pl.DataFrame({"doc_id": ["doc1"], "token": ["hello"], "count": [1]}),  # tt_pos
            pl.DataFrame({"doc_id": ["doc1"], "tag": ["Character"], "count": [1]})  # tt_ds
        )
        mock_dtm.return_value = (
            pl.DataFrame({"doc_id": ["doc1"], "hello": [1]}),  # dtm_pos
            pl.DataFrame({"doc_id": ["doc1"], "Character": [1]})  # dtm_ds
        )

        with patch('streamlit.session_state', {}):
            st.session_state[self.user_session_id] = {}

            # Mock session manager
            mock_session_manager = MagicMock()
            mock_app_core.session_manager = mock_session_manager

            finalize_corpus_load(
                self.test_tokens, self.user_session_id, 'target'
            )

            # Verify all processing functions were called
            mock_freq_table.assert_called_once_with(
                self.test_tokens, count_by="both"
            )
            mock_tags_table.assert_called_once_with(
                self.test_tokens, count_by="both"
            )
            mock_dtm.assert_called_once_with(
                self.test_tokens, count_by="both"
            )

            # Verify load_corpus_new was called with all the data
            mock_load_corpus.assert_called_once()

            # Verify metadata initialization for target
            mock_init_metadata.assert_called_once_with(self.user_session_id)

            # Verify session manager update
            mock_session_manager.update_session_state.assert_called_once()

            # Verify cleanup and rerun
            mock_cleanup.assert_called_once_with(self.user_session_id, 'target')
            mock_rerun.assert_called_once()

    @patch('webapp.utilities.processing.corpus_processing.ds.frequency_table')
    @patch('webapp.utilities.processing.corpus_processing.ds.tags_table')
    @patch('webapp.utilities.processing.corpus_processing.ds.tags_dtm')
    @patch('webapp.utilities.processing.corpus_processing.load_corpus_new')
    @patch('webapp.utilities.processing.corpus_processing.cleanup_original_corpus_data')
    @patch('webapp.utilities.processing.corpus_processing.init_metadata_reference')
    @patch('webapp.utilities.processing.corpus_processing.app_core')
    @patch('streamlit.rerun')
    def test_finalize_corpus_load_reference(
        self, mock_rerun, mock_app_core, mock_init_metadata, mock_cleanup,
        mock_load_corpus, mock_dtm, mock_tags_table, mock_freq_table
    ):
        """Test finalizing reference corpus load."""
        mock_freq_table.return_value = (MagicMock(), MagicMock())
        mock_tags_table.return_value = (MagicMock(), MagicMock())
        mock_dtm.return_value = (MagicMock(), MagicMock())

        with patch('streamlit.session_state', {}):
            st.session_state[self.user_session_id] = {}

            finalize_corpus_load(
                self.test_tokens, self.user_session_id, 'reference'
            )

            # Verify metadata initialization for reference
            mock_init_metadata.assert_called_once_with(self.user_session_id)


class TestFinalizeCorpusLoadOptimized:
    """Test optimized corpus finalization functionality."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.user_session_id = "test_session"
        self.test_tokens = pl.DataFrame({
            "doc_id": ["doc1", "doc2"],
            "token": ["hello", "world"],
            "tag": ["Character", "Description"]
        })

    @patch('webapp.utilities.processing.corpus_processing.get_corpus_manager')
    @patch('webapp.utilities.processing.corpus_processing.init_metadata_target')
    @patch('webapp.utilities.processing.corpus_processing.app_core')
    @patch('webapp.utilities.processing.corpus_processing.cleanup_original_corpus_data')
    @patch('streamlit.rerun')
    def test_finalize_corpus_load_optimized_target(
        self, mock_rerun, mock_cleanup, mock_app_core,
        mock_init_metadata, mock_get_manager
    ):
        """Test optimized finalization for target corpus."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        mock_session_manager = MagicMock()
        mock_app_core.session_manager = mock_session_manager

        finalize_corpus_load_optimized(
            self.test_tokens, self.user_session_id, 'target'
        )

        # Verify corpus manager setup
        mock_get_manager.assert_called_once_with(
            self.user_session_id, 'target'
        )
        mock_manager.set_core_data.assert_called_once_with(self.test_tokens)

        # Verify session state updates
        mock_init_metadata.assert_called_once_with(self.user_session_id)
        mock_session_manager.update_session_state.assert_called_once_with(
            self.user_session_id, SessionKeys.HAS_TARGET, True
        )

        # Verify cleanup and rerun
        mock_cleanup.assert_called_once_with(self.user_session_id, 'target')
        mock_rerun.assert_called_once()

    @patch('webapp.utilities.processing.corpus_processing.get_corpus_manager')
    @patch('webapp.utilities.processing.corpus_processing.init_metadata_reference')
    @patch('webapp.utilities.processing.corpus_processing.app_core')
    @patch('webapp.utilities.processing.corpus_processing.cleanup_original_corpus_data')
    @patch('streamlit.rerun')
    def test_finalize_corpus_load_optimized_reference(
        self, mock_rerun, mock_cleanup, mock_app_core,
        mock_init_metadata, mock_get_manager
    ):
        """Test optimized finalization for reference corpus."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        mock_session_manager = MagicMock()
        mock_app_core.session_manager = mock_session_manager

        finalize_corpus_load_optimized(
            self.test_tokens, self.user_session_id, 'reference'
        )

        # Verify reference-specific behavior
        mock_init_metadata.assert_called_once_with(self.user_session_id)
        mock_session_manager.update_session_state.assert_called_once_with(
            self.user_session_id, SessionKeys.HAS_REFERENCE, True
        )


class TestProcessNew:
    """Test new corpus processing functionality."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.user_session_id = "test_session"
        self.corpus_type = "target"
        self.mock_nlp = MagicMock()
        self.test_df = pl.DataFrame({
            "doc_id": ["doc1", "doc2"],
            "text": ["Hello world", "Test document"]
        })

    @patch('streamlit.success')
    @patch('webapp.utilities.processing.corpus_processing.ds.docuscope_parse')
    @patch('webapp.utilities.processing.corpus_processing.finalize_corpus_load')
    def test_process_new_valid_corpus(
        self, mock_finalize, mock_docuscope_parse, mock_success
    ):
        """Test processing a valid new corpus."""
        # Mock successful parsing
        mock_processed_tokens = pl.DataFrame({
            "doc_id": ["doc1"],
            "token": ["hello"],
            "pos_tag": ["NN1"],
            "ds_tag": ["Character"],
            "pos_id": [0],
            "ds_id": [0]
        })
        mock_docuscope_parse.return_value = mock_processed_tokens

        # Mock session state
        with patch('streamlit.session_state', {self.user_session_id: {}}):
            process_new(
                self.test_df, self.mock_nlp, self.user_session_id,
                self.corpus_type, exceptions=None
            )

        # Verify parsing was called
        mock_docuscope_parse.assert_called_once_with(
            corp=self.test_df, nlp_model=self.mock_nlp
        )

        # Verify finalization was called
        mock_finalize.assert_called_once_with(
            mock_processed_tokens, self.user_session_id, self.corpus_type
        )

    @patch('webapp.utilities.processing.corpus_processing.ds.docuscope_parse')
    @patch('webapp.utilities.processing.corpus_processing.finalize_corpus_load')
    def test_process_new_with_exceptions(
        self, mock_finalize, mock_docuscope_parse
    ):
        """Test processing new corpus with exception handling."""
        exceptions = ["doc2_error", "doc3_error"]

        # Mock parsing with successful result
        mock_processed_tokens = pl.DataFrame({
            "doc_id": ["doc1"],
            "token": ["hello"],
            "pos_tag": ["NN1"],
            "ds_tag": ["Character"],
            "pos_id": [0],
            "ds_id": [0]
        })
        mock_docuscope_parse.return_value = mock_processed_tokens

        # Mock session state
        with patch('streamlit.session_state', {self.user_session_id: {}}):
            process_new(
                self.test_df, self.mock_nlp, self.user_session_id,
                self.corpus_type, exceptions=exceptions
            )

        # Verify parsing was called
        mock_docuscope_parse.assert_called_once_with(
            corp=self.test_df, nlp_model=self.mock_nlp
        )

        # Verify finalization was called since processing succeeded
        mock_finalize.assert_called_once_with(
            mock_processed_tokens, self.user_session_id, self.corpus_type
        )

    @patch('webapp.utilities.processing.corpus_processing.ds.docuscope_parse')
    def test_process_new_invalid_corpus(self, mock_docuscope_parse):
        """Test processing with DataFrame that would cause parsing error."""
        # Mock parsing that raises an exception
        mock_docuscope_parse.side_effect = Exception("Parsing failed")

        # This should not raise an exception - the function handles it
        process_new(
            self.test_df, self.mock_nlp, self.user_session_id,
            self.corpus_type, exceptions=None
        )

        # Verify parsing was attempted
        mock_docuscope_parse.assert_called_once_with(
            corp=self.test_df, nlp_model=self.mock_nlp
        )

    def test_process_new_none_corpus(self):
        """Test processing with None corpus DataFrame."""
        # No mocking needed - function should handle None gracefully
        process_new(
            None, self.mock_nlp, self.user_session_id,
            self.corpus_type, exceptions=None
        )
        # Test passes if no exception is raised

    def test_process_new_empty_corpus(self):
        """Test processing with empty corpus DataFrame."""
        empty_df = pl.DataFrame()

        # No mocking needed - function should handle empty DataFrame gracefully
        process_new(
            empty_df, self.mock_nlp, self.user_session_id,
            self.corpus_type, exceptions=None
        )
        # Test passes if no exception is raised


class TestCorpusProcessingIntegration:
    """Integration tests for corpus processing workflows."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.user_session_id = "integration_test_session"
        self.mock_nlp = MagicMock()

    @patch('webapp.utilities.processing.corpus_processing.finalize_corpus_load')
    @patch('webapp.utilities.processing.corpus_processing.ds.docuscope_parse')
    def test_full_new_corpus_workflow(
        self, mock_docuscope_parse, mock_finalize
    ):
        """Test complete workflow from new corpus to finalization."""
        # Setup test data
        test_df = pl.DataFrame({
            "doc_id": ["doc1", "doc2"],
            "text": ["Hello world", "Test document"]
        })

        # Mock successful processing
        mock_processed_tokens = pl.DataFrame({
            "doc_id": ["doc1", "doc1", "doc2", "doc2"],
            "token": ["hello", "world", "test", "document"],
            "pos_tag": ["NN1", "NN1", "NN1", "NN1"],
            "ds_tag": ["Character", "Description", "AcademicTerms", "AcademicTerms"],
            "pos_id": [0, 1, 0, 1],
            "ds_id": [0, 1, 2, 3]
        })
        mock_docuscope_parse.return_value = mock_processed_tokens

        # Execute workflow with mocked session state
        with patch('streamlit.session_state', {self.user_session_id: {}}):
            with patch('streamlit.success'):
                process_new(
                    test_df, self.mock_nlp, self.user_session_id, 'target'
                )

        # Verify complete workflow
        mock_docuscope_parse.assert_called_once_with(
            corp=test_df, nlp_model=self.mock_nlp
        )
        mock_finalize.assert_called_once_with(
            mock_processed_tokens, self.user_session_id, 'target'
        )

    @patch('webapp.utilities.processing.corpus_processing.ds.docuscope_parse')
    def test_workflow_stops_on_parsing_failure(self, mock_docuscope_parse):
        """Test that workflow handles parsing failure gracefully."""
        test_df = pl.DataFrame({"doc_id": ["doc1"], "text": ["test"]})
        mock_docuscope_parse.side_effect = Exception("Parsing failed")

        # Should not raise exception
        process_new(
            test_df, self.mock_nlp, self.user_session_id, 'target'
        )

        # Verify parsing was attempted
        mock_docuscope_parse.assert_called_once_with(
            corp=test_df, nlp_model=self.mock_nlp
        )
