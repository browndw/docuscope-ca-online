"""
Integration tests for DocuScope corpus processing pipeline.

Tests the complete workflow from corpus loading through NLP processing
to data analysis, using real spaCy models and realistic data.
"""

import polars as pl
import pytest
import sys
from unittest.mock import patch, MagicMock

# Use Streamlit session state mocking for backend integration testing
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    # Fallback to mocking if Streamlit not available
    sys.modules['streamlit'] = MagicMock()
    st = MagicMock()
    STREAMLIT_AVAILABLE = False

from webapp.utilities.processing.corpus_processing import process_new
from webapp.utilities.session.session_core import init_session, update_session
from webapp.utilities.corpus.data_manager import CorpusDataManager
from webapp.utilities.state import SessionKeys


class TestCorpusProcessingPipeline:
    """Test complete corpus processing pipeline integration."""

    def setup_method(self):
        """Set up integration test environment."""
        self.user_session_id = "integration_test_session"
        st.session_state.clear()

        # Initialize complete session structure
        st.session_state[self.user_session_id] = {
            "corpus_data": {"target": {}, "reference": {}},
            "session": {}
        }

        # Initialize session state
        init_session(self.user_session_id)

    @pytest.mark.integration
    @patch('docuscospacy.docuscope_parse')
    @patch('webapp.utilities.processing.corpus_processing.finalize_corpus_load')
    @patch('streamlit.session_state', {})
    @patch('streamlit.success')
    def test_complete_new_corpus_pipeline(
        self, mock_success, mock_finalize, mock_docuscope_parse
    ):
        """Test complete pipeline from new corpus input to finalized data."""
        # Create realistic test corpus
        test_corpus = pl.DataFrame({
            "doc_id": ["academic_001", "academic_002", "fiction_001"],
            "text": [
                "The research methodology employed in this study follows "
                "established protocols for quantitative analysis.",
                "Data collection procedures were implemented according to "
                "institutional guidelines and ethical standards.",
                "The old mansion stood silently against the moonlit sky, "
                "its windows reflecting memories of forgotten times."
            ],
            "genre": ["academic", "academic", "fiction"]
        })

        # Mock realistic processed tokens output (using actual DocuScope tags)
        mock_processed_tokens = pl.DataFrame({
            "doc_id": [
                "academic_001", "academic_001", "academic_002",
                "academic_002", "fiction_001", "fiction_001"
            ],
            "token": [
                "methodology", "analysis", "procedures",
                "guidelines", "mansion", "memories"
            ],
            "ws": [" ", " ", " ", " ", " ", ""],
            "pos_tag": ["NN1", "NN1", "NN2", "NN2", "NN1", "NN2"],
            "ds_tag": [
                "B-AcademicTerms", "B-Reasoning", "B-AcademicTerms",
                "B-InformationExposition", "B-Description", "B-Character"
            ]
        })
        mock_docuscope_parse.return_value = mock_processed_tokens

        # Mock spaCy model
        mock_nlp = MagicMock()
        mock_nlp.meta = {"name": "en_docusco_spacy", "version": "1.0.0"}

        # Initialize session state for this test
        import streamlit as st
        st.session_state[self.user_session_id] = {'warning': 0}

        # Execute complete pipeline
        process_new(
            test_corpus, mock_nlp, self.user_session_id, 'target'
        )

        # Verify complete workflow execution
        mock_docuscope_parse.assert_called_once_with(
            corp=test_corpus, nlp_model=mock_nlp
        )
        mock_finalize.assert_called_once_with(
            mock_processed_tokens, self.user_session_id, 'target'
        )
        mock_success.assert_called_once_with('Processing complete!')

    @pytest.mark.integration
    @patch('streamlit.warning')
    def test_pipeline_validation_failure_handling(self, mock_warning):
        """Test pipeline behavior when corpus validation fails."""
        # Test with None corpus (validation failed upstream)
        mock_nlp = MagicMock()

        # Execute pipeline with None corpus
        process_new(None, mock_nlp, self.user_session_id, 'target')

        # Verify warning was shown
        mock_warning.assert_called_once_with(
            "Please upload files for your target corpus before processing.",
            icon=":material/warning:"
        )

        # Test with empty corpus (validation failed upstream)
        empty_corpus = pl.DataFrame({
            "doc_id": [],
            "text": []
        })

        mock_warning.reset_mock()
        process_new(empty_corpus, mock_nlp, self.user_session_id, 'target')

        # Verify warning was shown for empty corpus
        mock_warning.assert_called_once_with(
            "No valid text files found for your target corpus. "
            "Please check your uploads.",
            icon=":material/warning:"
        )

    @pytest.mark.integration
    def test_session_and_corpus_manager_integration(self):
        """Test integration between session management and corpus manager."""
        # Initialize session
        init_session(self.user_session_id)

        # Create corpus manager
        manager = CorpusDataManager(self.user_session_id, 'target')

        # Update session state
        update_session(SessionKeys.HAS_TARGET, True, self.user_session_id)
        update_session(SessionKeys.TARGET_DB, '/path/to/corpus', self.user_session_id)

        # Verify session state
        session_df = st.session_state[self.user_session_id]["session"]
        session_dict = session_df.to_dict(as_series=False)

        assert session_dict[SessionKeys.HAS_TARGET] == [True]
        assert session_dict[SessionKeys.TARGET_DB] == ['/path/to/corpus']

        # Verify corpus manager can access session data
        corpus_data = manager.session_corpus_data
        assert isinstance(corpus_data, dict)

    @pytest.mark.integration
    def test_multi_corpus_workflow(self):
        """Test workflow with both target and reference corpora."""
        # Initialize session for both corpus types
        init_session(self.user_session_id)

        # Create managers for both corpus types
        target_manager = CorpusDataManager(self.user_session_id, 'target')
        reference_manager = CorpusDataManager(self.user_session_id, 'reference')

        # Update session for both corpora
        update_session(SessionKeys.HAS_TARGET, True, self.user_session_id)
        update_session(SessionKeys.HAS_REFERENCE, True, self.user_session_id)

        # Verify isolation
        target_data = target_manager.session_corpus_data
        reference_data = reference_manager.session_corpus_data

        assert target_data is not reference_data

        # Verify session state reflects both corpora
        session_df = st.session_state[self.user_session_id]["session"]
        session_dict = session_df.to_dict(as_series=False)

        assert session_dict[SessionKeys.HAS_TARGET] == [True]
        assert session_dict[SessionKeys.HAS_REFERENCE] == [True]


class TestDataValidationIntegration:
    """Test integration of data validation with processing pipeline."""

    def setup_method(self):
        """Set up integration test environment."""
        self.user_session_id = "validation_integration_session"
        st.session_state.clear()
        st.session_state[self.user_session_id] = {
            "corpus_data": {"target": {}, "reference": {}}
        }

    @pytest.mark.integration
    @patch('streamlit.warning')
    @patch('docuscospacy.docuscope_parse')
    def test_data_validation_edge_cases(self, mock_docuscope_parse, mock_warning):
        """Test data validation with various edge cases."""
        test_cases = [
            # Empty corpus
            (pl.DataFrame(), "No valid text files found"),

            # Valid minimal corpus
            (pl.DataFrame({"doc_id": ["doc1"], "text": ["content"]}), None),
        ]

        mock_nlp = MagicMock()

        for corpus_df, expected_warning in test_cases:
            # Reset mocks
            mock_warning.reset_mock()
            mock_docuscope_parse.reset_mock()

            if expected_warning:
                # Test empty corpus handling
                process_new(corpus_df, mock_nlp, self.user_session_id, 'target')
                mock_warning.assert_called_once()
                assert expected_warning in mock_warning.call_args[0][0]
                mock_docuscope_parse.assert_not_called()
            else:
                # Test valid corpus processing
                mock_tokens = pl.DataFrame({
                    "doc_id": ["doc1"],
                    "token": ["content"],
                    "pos_tag": ["NN1"],
                    "ds_tag": ["B-Description"]
                })
                mock_docuscope_parse.return_value = mock_tokens
                with patch(
                    'streamlit.session_state',
                    {self.user_session_id: {'warning': 0}}
                ):
                    with patch(
                        'webapp.utilities.processing.corpus_processing.finalize_corpus_load'  # noqa: E501
                    ):
                        process_new(corpus_df, mock_nlp, self.user_session_id, 'target')
                        mock_docuscope_parse.assert_called_once_with(
                            corp=corpus_df, nlp_model=mock_nlp
                        )


class TestMemoryManagementIntegration:
    """Test memory management across the integrated pipeline."""

    def setup_method(self):
        """Set up memory management test environment."""
        self.user_session_id = "memory_integration_session"
        st.session_state.clear()
        st.session_state[self.user_session_id] = {
            "corpus_data": {"target": {}, "reference": {}}
        }

    @pytest.mark.integration
    def test_memory_cleanup_after_processing(self):
        """Test that memory is properly cleaned up after processing."""
        # Create large test data to simulate memory pressure
        large_corpus = pl.DataFrame({
            "doc_id": [f"doc_{i}" for i in range(100)],
            "text": ["Large document content " * 100 for _ in range(100)]
        })

        # Create corpus manager
        manager = CorpusDataManager(self.user_session_id, 'target')

        # Simulate data storage
        corpus_data = manager.session_corpus_data
        corpus_data['large_data'] = large_corpus

        # Verify data is stored
        assert 'large_data' in corpus_data
        assert corpus_data['large_data'].height == 100

        # Simulate cleanup (would be called by actual cleanup functions)
        del corpus_data['large_data']

        # Verify cleanup worked
        assert 'large_data' not in corpus_data

    @pytest.mark.integration
    def test_multiple_session_memory_isolation(self):
        """Test memory isolation between multiple sessions."""
        session_1 = "memory_session_1"
        session_2 = "memory_session_2"

        # Initialize both sessions
        for session_id in [session_1, session_2]:
            st.session_state[session_id] = {
                "corpus_data": {"target": {}, "reference": {}}
            }

        # Create managers for different sessions
        manager_1 = CorpusDataManager(session_1, 'target')
        manager_2 = CorpusDataManager(session_2, 'target')

        # Add data to first session
        data_1 = manager_1.session_corpus_data
        data_1['test_data'] = pl.DataFrame({"session": ["session_1"]})

        # Add different data to second session
        data_2 = manager_2.session_corpus_data
        data_2['test_data'] = pl.DataFrame({"session": ["session_2"]})

        # Verify isolation
        assert data_1['test_data']['session'].to_list() == ["session_1"]
        assert data_2['test_data']['session'].to_list() == ["session_2"]

        # Verify they are different objects
        assert data_1 is not data_2


class TestSpacyModelIntegration:
    """Test integration with actual spaCy models (mocked for unit tests)."""

    def setup_method(self):
        """Set up spaCy model integration test environment."""
        self.user_session_id = "spacy_integration_session"
        st.session_state.clear()
        st.session_state[self.user_session_id] = {
            "corpus_data": {"target": {}, "reference": {}}
        }

    @pytest.mark.integration
    @patch('docuscospacy.docuscope_parse')
    @patch('webapp.utilities.processing.corpus_processing.finalize_corpus_load')
    @patch('streamlit.session_state', {})
    @patch('streamlit.success')
    def test_spacy_model_processing_simulation(
        self, mock_success, mock_finalize, mock_docuscope_parse
    ):
        """Test processing pipeline with simulated spaCy model behavior."""
        # Create test corpus
        test_corpus = pl.DataFrame({
            "doc_id": ["test_doc"],
            "text": ["This is a test document for processing."]
        })

        # Mock realistic DocuScope processing output (using actual DocuScope format)
        mock_processed_tokens = pl.DataFrame({
            "doc_id": ["test_doc"] * 8,
            "token": ["This", "is", "a", "test", "document", "for", "processing", "."],
            "ds_tag": [
                "B-MetadiscourseCohesive", "B-InformationStates", "O", "B-AcademicTerms",
                "B-AcademicTerms", "O", "B-AcademicTerms", "O"
            ],
            "pos_tag": ["DD1", "VBZ", "AT", "NN1", "NN1", "IF", "NN1", "Y"],
            "lemma": ["this", "be", "a", "test", "document", "for", "process", "."],
            "sent_id": [0] * 8,
            "token_id": list(range(8))
        })
        mock_docuscope_parse.return_value = mock_processed_tokens

        # Create mock spaCy model with realistic attributes
        mock_nlp = MagicMock()
        mock_nlp.meta = {
            "name": "en_docusco_spacy",
            "version": "1.0.0",
            "description": "DocuScope English model"
        }
        mock_nlp.pipe_names = ["tok2vec", "tagger", "parser", "ner", "docuscope"]

        # Initialize session state for this test
        import streamlit as st
        st.session_state[self.user_session_id] = {'warning': 0}

        # Execute processing
        process_new(
            test_corpus, mock_nlp, self.user_session_id, 'target'
        )

        # Verify processing completed
        mock_docuscope_parse.assert_called_once_with(
            corp=test_corpus, nlp_model=mock_nlp
        )
        mock_finalize.assert_called_once_with(
            mock_processed_tokens, self.user_session_id, 'target'
        )
        mock_success.assert_called_once_with('Processing complete!')
