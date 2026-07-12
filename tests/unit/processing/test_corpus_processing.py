"""
Tests for webapp.utilities.processing.corpus_processing module.

Tests corpus processing workflows, memory management, and data handling.
"""

import polars as pl
from pathlib import Path
import sys
import tempfile
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
    process_internal,
    attach_queued_internal_target,
    PROCESS_TARGET_PROBE_METADATA_NO_PERSIST,
    PROCESS_TARGET_PROBE_NO_METADATA,
)
from webapp.utilities.state import CorpusPersistencePolicy, SessionKeys


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

    @patch('webapp.utilities.processing.corpus_processing.get_corpus_manager')
    @patch('webapp.utilities.processing.corpus_processing.set_session_persistence_policy')
    @patch('webapp.utilities.processing.corpus_processing.build_corpus_metadata_descriptor')
    @patch('webapp.utilities.processing.corpus_processing.cleanup_original_corpus_data')
    @patch('webapp.utilities.processing.corpus_processing.init_metadata_target')
    @patch('webapp.utilities.processing.corpus_processing.app_core')
    @patch('streamlit.rerun')
    def test_finalize_corpus_load_target(
        self,
        mock_rerun,
        mock_app_core,
        mock_init_metadata,
        mock_cleanup,
        mock_build_metadata,
        mock_set_policy,
        mock_get_manager,
    ):
        """Test finalizing target corpus load."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        mock_build_metadata.return_value = {"ndocs": 2}

        with patch('streamlit.session_state', {}):
            st.session_state[self.user_session_id] = {}

            # Mock session manager
            mock_session_manager = MagicMock()
            mock_app_core.session_manager = mock_session_manager

            finalize_corpus_load(
                self.test_tokens, self.user_session_id, 'target'
            )

            mock_set_policy.assert_called_once_with(
                self.user_session_id,
                CorpusPersistencePolicy.SERVER_SAVED,
                corpus_type='target',
            )

            mock_get_manager.assert_called_once_with(
                self.user_session_id, 'target'
            )
            mock_manager.set_core_data.assert_called_once_with(
                self.test_tokens,
                persist=False,
            )

            # Verify metadata initialization for target
            mock_init_metadata.assert_called_once_with(
                self.user_session_id,
                {"ndocs": 2},
            )

            # Verify session manager update
            mock_session_manager.update_session_state.assert_called_once()

            # Verify cleanup and rerun
            mock_cleanup.assert_called_once_with(self.user_session_id, 'target')
            mock_rerun.assert_called_once()

    @patch('webapp.utilities.processing.corpus_processing.get_corpus_manager')
    @patch('webapp.utilities.processing.corpus_processing.set_session_persistence_policy')
    @patch('webapp.utilities.processing.corpus_processing.build_corpus_metadata_descriptor')
    @patch('webapp.utilities.processing.corpus_processing.cleanup_original_corpus_data')
    @patch('webapp.utilities.processing.corpus_processing.init_metadata_reference')
    @patch('webapp.utilities.processing.corpus_processing.app_core')
    @patch('streamlit.rerun')
    def test_finalize_corpus_load_reference(
        self, mock_rerun, mock_app_core, mock_init_metadata, mock_cleanup,
        mock_build_metadata, mock_set_policy, mock_get_manager
    ):
        """Test finalizing reference corpus load."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        mock_build_metadata.return_value = {"ndocs": 2}

        with patch('streamlit.session_state', {}):
            st.session_state[self.user_session_id] = {}

            finalize_corpus_load(
                self.test_tokens, self.user_session_id, 'reference'
            )

            mock_set_policy.assert_called_once_with(
                self.user_session_id,
                CorpusPersistencePolicy.SERVER_SAVED,
                corpus_type='reference',
            )

            # Verify metadata initialization for reference
            mock_init_metadata.assert_called_once_with(
                self.user_session_id,
                {"ndocs": 2},
            )
            mock_get_manager.assert_called_once_with(
                self.user_session_id, 'reference'
            )
            mock_manager.set_core_data.assert_called_once_with(
                self.test_tokens,
                persist=False,
            )
            mock_cleanup.assert_called_once_with(self.user_session_id, 'reference')
            mock_rerun.assert_called_once()


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
    @patch('webapp.utilities.processing.corpus_processing.set_session_persistence_policy')
    @patch('webapp.utilities.processing.corpus_processing.build_corpus_metadata_descriptor')
    @patch('webapp.utilities.processing.corpus_processing.init_metadata_target')
    @patch('webapp.utilities.processing.corpus_processing.app_core')
    @patch('webapp.utilities.processing.corpus_processing.cleanup_original_corpus_data')
    @patch('streamlit.rerun')
    def test_finalize_corpus_load_optimized_target(
        self, mock_rerun, mock_cleanup, mock_app_core,
        mock_init_metadata, mock_build_metadata, mock_set_policy,
        mock_get_manager
    ):
        """Test optimized finalization for target corpus."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        mock_build_metadata.return_value = {"ndocs": 2}
        mock_session_manager = MagicMock()
        mock_app_core.session_manager = mock_session_manager

        finalize_corpus_load_optimized(
            self.test_tokens, self.user_session_id, 'target'
        )

        mock_set_policy.assert_called_once_with(
            self.user_session_id,
            CorpusPersistencePolicy.SERVER_SAVED,
            corpus_type='target',
        )

        # Verify corpus manager setup
        mock_get_manager.assert_called_once_with(
            self.user_session_id, 'target'
        )
        mock_manager.set_core_data.assert_called_once_with(
            self.test_tokens,
            persist=False,
        )
        mock_manager.set_file_refs.assert_called_once()

        # Verify session state updates
        mock_init_metadata.assert_called_once_with(
            self.user_session_id,
            {"ndocs": 2},
        )
        mock_session_manager.update_session_state.assert_called_once_with(
            self.user_session_id, SessionKeys.HAS_TARGET, True
        )

        # Verify cleanup and rerun
        mock_cleanup.assert_called_once_with(self.user_session_id, 'target')
        mock_rerun.assert_called_once()

    @patch('webapp.utilities.processing.corpus_processing.get_corpus_manager')
    @patch('webapp.utilities.processing.corpus_processing.set_session_persistence_policy')
    @patch('webapp.utilities.processing.corpus_processing.build_corpus_metadata_descriptor')
    @patch('webapp.utilities.processing.corpus_processing.init_metadata_reference')
    @patch('webapp.utilities.processing.corpus_processing.app_core')
    @patch('webapp.utilities.processing.corpus_processing.cleanup_original_corpus_data')
    @patch('streamlit.rerun')
    def test_finalize_corpus_load_optimized_reference(
        self, mock_rerun, mock_cleanup, mock_app_core,
        mock_init_metadata, mock_build_metadata, mock_set_policy,
        mock_get_manager
    ):
        """Test optimized finalization for reference corpus."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        mock_build_metadata.return_value = {"ndocs": 2}
        mock_session_manager = MagicMock()
        mock_app_core.session_manager = mock_session_manager

        finalize_corpus_load_optimized(
            self.test_tokens, self.user_session_id, 'reference'
        )

        mock_set_policy.assert_called_once_with(
            self.user_session_id,
            CorpusPersistencePolicy.SERVER_SAVED,
            corpus_type='reference',
        )

        mock_manager.set_core_data.assert_called_once_with(
            self.test_tokens,
            persist=False,
        )
        mock_manager.set_file_refs.assert_called_once()

        # Verify reference-specific behavior
        mock_init_metadata.assert_called_once_with(
            self.user_session_id,
            {"ndocs": 2},
        )
        mock_session_manager.update_session_state.assert_called_once_with(
            self.user_session_id, SessionKeys.HAS_REFERENCE, True
        )

    @patch('webapp.utilities.processing.corpus_processing.set_session_persistence_policy')
    @patch('webapp.utilities.processing.corpus_processing.build_corpus_metadata_descriptor')
    @patch('webapp.utilities.processing.corpus_processing.init_metadata_target')
    @patch('webapp.utilities.processing.corpus_processing.app_core')
    @patch('webapp.utilities.processing.corpus_processing.cleanup_original_corpus_data')
    @patch('streamlit.rerun')
    def test_finalize_corpus_load_optimized_writes_session_artifact_for_server_saved(
        self,
        mock_rerun,
        mock_cleanup,
        mock_app_core,
        mock_init_metadata,
        mock_build_metadata,
        mock_set_policy,
    ):
        class StubManager:
            def __init__(self):
                self.session_corpus_data = {}
                self.file_refs = None

            def set_core_data(self, ds_tokens, persist=True):
                self.session_corpus_data['ds_tokens'] = ds_tokens

            def set_file_refs(self, file_map):
                self.file_refs = file_map
                self.session_corpus_data.setdefault('_artifact_refs', {})
                for key, path in file_map.items():
                    self.session_corpus_data['_artifact_refs'][key] = {
                        'storage_type': 'gzip_pickle',
                        'path': path,
                    }
                    self.session_corpus_data.pop(key, None)

        mock_manager = StubManager()
        mock_build_metadata.return_value = {'ndocs': 2}
        mock_session_manager = MagicMock()
        mock_app_core.session_manager = mock_session_manager

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                'webapp.utilities.processing.corpus_processing.get_corpus_manager',
                return_value=mock_manager,
            ):
                with patch(
                    'webapp.utilities.processing.corpus_processing.get_config',
                    return_value=tmpdir,
                ):
                    finalize_corpus_load_optimized(
                        self.test_tokens,
                        self.user_session_id,
                        'target',
                        CorpusPersistencePolicy.SERVER_SAVED,
                    )

            artifact_path = Path(mock_manager.file_refs['ds_tokens'])
            assert artifact_path.exists()
            assert artifact_path.name == 'ds_tokens.gz'
            assert artifact_path.parent.joinpath('metadata_descriptor.json').exists()
            assert mock_manager.session_corpus_data['ds_tokens'].equals(self.test_tokens)


class TestAttachQueuedInternalTarget:
    """Test queue-prepared target attachment behavior."""

    @patch('webapp.utilities.processing.corpus_processing.load_corpus_internal')
    @patch('webapp.utilities.processing.corpus_processing.get_corpus_manager')
    @patch('webapp.utilities.processing.corpus_processing.set_session_persistence_policy')
    @patch('webapp.utilities.processing.corpus_processing.init_metadata_target')
    @patch('webapp.utilities.processing.corpus_processing._persist_session_updates')
    @patch('webapp.utilities.processing.corpus_processing.registry_service')
    @patch('streamlit.rerun')
    def test_attach_queued_internal_target_sets_frequency_artifact_refs(
        self,
        mock_rerun,
        mock_registry,
        mock_persist_updates,
        mock_init_metadata,
        mock_set_policy,
        mock_get_manager,
        mock_load_internal,
    ):
        """Queue completion payload should attach shared frequency refs eagerly."""

        mock_manager = MagicMock()
        mock_manager.is_ready.return_value = True
        mock_get_manager.return_value = mock_manager

        queue_artifact = MagicMock(status="ready")
        mock_registry.get_artifact_by_id.return_value = queue_artifact
        mock_registry.load_json_artifact.return_value = {"frequency_artifact_id": 42}

        attach_queued_internal_target(
            "webapp/_corpora/ld/A_MICUSP_mini",
            "queued-session",
            queue_artifact_id=17,
        )

        mock_load_internal.assert_called_once_with(
            "webapp/_corpora/ld/A_MICUSP_mini",
            "queued-session",
            corpus_type="target",
        )
        mock_registry.get_artifact_by_id.assert_called_once_with(17)
        mock_registry.load_json_artifact.assert_called_once_with(queue_artifact)
        mock_manager.set_artifact_refs.assert_called_once_with(
            "frequency_bundle",
            42,
            ["ft_pos", "ft_ds"],
        )
        mock_set_policy.assert_called_once()
        mock_init_metadata.assert_called_once_with("queued-session")
        mock_persist_updates.assert_called_once()
        mock_rerun.assert_called_once()


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
            mock_processed_tokens,
            self.user_session_id,
            self.corpus_type,
            CorpusPersistencePolicy.SERVER_SAVED,
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
            mock_processed_tokens,
            self.user_session_id,
            self.corpus_type,
            CorpusPersistencePolicy.SERVER_SAVED,
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
            mock_processed_tokens,
            self.user_session_id,
            'target',
            CorpusPersistencePolicy.SERVER_SAVED,
        )

    @patch('streamlit.success')
    @patch('webapp.utilities.processing.corpus_processing.ds.docuscope_parse')
    @patch('webapp.utilities.processing.corpus_processing.finalize_corpus_load')
    def test_process_new_temporary_policy_passes_through(
        self, mock_finalize, mock_docuscope_parse, mock_success
    ):
        test_df = pl.DataFrame({
            "doc_id": ["doc1", "doc2"],
            "text": ["Hello world", "Test document"],
        })
        mock_processed_tokens = pl.DataFrame({
            "doc_id": ["doc1"],
            "token": ["hello"],
            "pos_tag": ["NN1"],
            "ds_tag": ["Character"],
            "pos_id": [0],
            "ds_id": [0],
        })
        mock_docuscope_parse.return_value = mock_processed_tokens

        with patch('streamlit.session_state', {self.user_session_id: {}}):
            process_new(
                test_df,
                self.mock_nlp,
                self.user_session_id,
                'target',
                persistence_policy=CorpusPersistencePolicy.TEMPORARY_SESSION_ONLY,
            )

        mock_finalize.assert_called_once_with(
            mock_processed_tokens,
            self.user_session_id,
            'target',
            CorpusPersistencePolicy.TEMPORARY_SESSION_ONLY,
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


class TestProcessInternalProbeModes:
    """Test load-test probe branches for internal corpus processing."""

    def setup_method(self):
        self.user_session_id = "test_session"
        self.corpus_path = "webapp/_corpora/ld/A_MICUSP_mini"

    @patch('webapp.utilities.processing.corpus_processing.st.caption')
    @patch('webapp.utilities.processing.corpus_processing.st.rerun')
    @patch('webapp.utilities.processing.corpus_processing.init_metadata_target')
    @patch('webapp.utilities.processing.corpus_processing.get_corpus_manager')
    @patch('webapp.utilities.processing.corpus_processing.load_corpus_internal')
    @patch('webapp.utilities.processing.corpus_processing.os.getenv')
    def test_process_internal_no_metadata_probe(
        self,
        mock_getenv,
        mock_load_internal,
        mock_get_manager,
        mock_init_metadata,
        mock_rerun,
        mock_caption,
    ):
        mock_getenv.return_value = PROCESS_TARGET_PROBE_NO_METADATA
        mock_manager = MagicMock()
        mock_manager.is_ready.return_value = True
        mock_get_manager.return_value = mock_manager
        session_state = {
            self.user_session_id: {
                'session': pl.from_dict({
                    SessionKeys.HAS_TARGET: [False],
                    SessionKeys.TARGET_DB: [''],
                })
            }
        }

        with patch('streamlit.session_state', session_state):
            process_internal(self.corpus_path, self.user_session_id, 'target')

        mock_load_internal.assert_called_once_with(
            self.corpus_path,
            self.user_session_id,
            corpus_type='target',
        )
        mock_init_metadata.assert_not_called()
        mock_rerun.assert_not_called()
        mock_caption.assert_called_once_with(
            f'LOAD_TEST_PROCESS_TARGET_READY:target:{PROCESS_TARGET_PROBE_NO_METADATA}'
        )

        session = session_state[self.user_session_id]['session'].to_dict(as_series=False)
        assert session[SessionKeys.TARGET_DB] == [self.corpus_path]
        assert session[SessionKeys.HAS_TARGET] == [True]
        mock_manager.warm_shared_frequency_data.assert_not_called()

    @patch('webapp.utilities.processing.corpus_processing.st.caption')
    @patch('webapp.utilities.processing.corpus_processing.st.rerun')
    @patch('webapp.utilities.processing.corpus_processing.init_metadata_target')
    @patch('webapp.utilities.processing.corpus_processing.get_corpus_manager')
    @patch('webapp.utilities.processing.corpus_processing.load_corpus_internal')
    @patch('webapp.utilities.processing.corpus_processing.os.getenv')
    def test_process_internal_metadata_no_persist_probe(
        self,
        mock_getenv,
        mock_load_internal,
        mock_get_manager,
        mock_init_metadata,
        mock_rerun,
        mock_caption,
    ):
        mock_getenv.return_value = PROCESS_TARGET_PROBE_METADATA_NO_PERSIST
        mock_manager = MagicMock()
        mock_manager.is_ready.return_value = True
        mock_get_manager.return_value = mock_manager
        session_state = {
            self.user_session_id: {
                'session': pl.from_dict({
                    SessionKeys.HAS_TARGET: [False],
                    SessionKeys.TARGET_DB: [''],
                })
            }
        }

        with patch('streamlit.session_state', session_state):
            process_internal(self.corpus_path, self.user_session_id, 'target')

        mock_load_internal.assert_called_once_with(
            self.corpus_path,
            self.user_session_id,
            corpus_type='target',
        )
        mock_init_metadata.assert_called_once_with(self.user_session_id)
        mock_rerun.assert_not_called()
        mock_caption.assert_called_once_with(
            'LOAD_TEST_PROCESS_TARGET_READY:'
            f'target:{PROCESS_TARGET_PROBE_METADATA_NO_PERSIST}'
        )

        session = session_state[self.user_session_id]['session'].to_dict(as_series=False)
        assert session[SessionKeys.TARGET_DB] == [self.corpus_path]
        assert session[SessionKeys.HAS_TARGET] == [True]
        mock_manager.warm_shared_frequency_data.assert_not_called()

    @patch('webapp.utilities.processing.corpus_processing.st.rerun')
    @patch('webapp.utilities.processing.corpus_processing.auto_persist_session')
    @patch('webapp.utilities.processing.corpus_processing.mark_session_dirty')
    @patch('webapp.utilities.processing.corpus_processing.init_metadata_target')
    @patch('webapp.utilities.processing.corpus_processing.get_corpus_manager')
    @patch('webapp.utilities.processing.corpus_processing.load_corpus_internal')
    @patch('webapp.utilities.processing.corpus_processing.os.getenv')
    def test_process_internal_full_probe(
        self,
        mock_getenv,
        mock_load_internal,
        mock_get_manager,
        mock_init_metadata,
        mock_mark_dirty,
        mock_auto_persist,
        mock_rerun,
    ):
        mock_getenv.return_value = 'full'
        mock_manager = MagicMock()
        mock_manager.is_ready.return_value = True
        mock_get_manager.return_value = mock_manager
        session_state = {
            self.user_session_id: {
                'session': pl.from_dict({
                    SessionKeys.HAS_TARGET: [False],
                    SessionKeys.TARGET_DB: [''],
                })
            }
        }

        with patch('streamlit.session_state', session_state):
            process_internal(self.corpus_path, self.user_session_id, 'target')

        mock_load_internal.assert_called_once_with(
            self.corpus_path,
            self.user_session_id,
            corpus_type='target',
        )
        mock_init_metadata.assert_called_once_with(self.user_session_id)
        mock_mark_dirty.assert_called_once_with(self.user_session_id)
        mock_auto_persist.assert_not_called()
        mock_manager.warm_shared_frequency_data.assert_called_once_with()

        session = session_state[self.user_session_id]['session'].to_dict(as_series=False)
        assert session[SessionKeys.TARGET_DB] == [self.corpus_path]
        assert session[SessionKeys.HAS_TARGET] == [True]
        mock_rerun.assert_called_once()
