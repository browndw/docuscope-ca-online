"""Tests for session persistence policy behavior."""

import sys
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import polars as pl

sys.modules.setdefault('google', MagicMock())
sys.modules.setdefault('google.cloud', MagicMock())
sys.modules.setdefault('google.cloud.firestore', MagicMock())
sys.modules.setdefault('google.oauth2', MagicMock())
sys.modules.setdefault('google.oauth2.service_account', MagicMock())
sys.modules.setdefault('streamlit', MagicMock())
sys.modules.setdefault('streamlit.components', MagicMock())
sys.modules.setdefault('streamlit.components.v1', MagicMock())

try:
    import streamlit as st
except ImportError:
    sys.modules['streamlit'] = MagicMock()
    st = MagicMock()

from webapp.utilities.session.session_persistence import (  # noqa: E402
    SessionPersistenceManager,
    build_persistable_session_data,
    get_session_persistence_policy,
    session_allows_persistence,
    set_session_persistence_policy,
    update_session_value_without_persistence,
)
from webapp.utilities.corpus import get_corpus_manager  # noqa: E402
from webapp.utilities.storage.cache_management import persistent_hash  # noqa: E402
from webapp.utilities.storage.sqlite_session_backend import (  # noqa: E402
    SQLiteSessionBackend,
)
from webapp.utilities.state import (  # noqa: E402
    CorpusPersistencePolicy,
    MetadataKeys,
    SessionKeys,
)


class TestSessionPersistencePolicy:
    @patch('streamlit.session_state', {})
    def test_update_session_value_without_persistence_updates_single_row_dataframe(self):
        session_id = 'test_session'
        st.session_state[session_id] = {
            'session': pl.from_dict({
                SessionKeys.CORPUS_PERSISTENCE_POLICY: [
                    CorpusPersistencePolicy.SERVER_SAVED
                ],
                SessionKeys.HAS_TARGET: [False],
            })
        }

        update_session_value_without_persistence(
            session_id,
            SessionKeys.FREQ_TABLE,
            True,
        )

        session_df = st.session_state[session_id]['session']
        assert session_df.height == 1
        assert session_df.get_column(SessionKeys.HAS_TARGET).to_list() == [False]
        assert session_df.get_column(SessionKeys.FREQ_TABLE).to_list() == [True]

    @patch('streamlit.session_state', {})
    def test_set_session_persistence_policy_updates_dataframe_session(self):
        session_id = 'test_session'
        st.session_state[session_id] = {
            'session': pl.from_dict({
                SessionKeys.CORPUS_PERSISTENCE_POLICY: [
                    CorpusPersistencePolicy.SERVER_SAVED
                ],
                SessionKeys.HAS_TARGET: [False],
            })
        }

        set_session_persistence_policy(
            session_id,
            CorpusPersistencePolicy.LOCAL_EXPORT_ONLY,
            corpus_type='target',
        )

        session_df = st.session_state[session_id]['session']
        assert session_df.get_column(
            SessionKeys.TARGET_PERSISTENCE_POLICY
        ).to_list() == [CorpusPersistencePolicy.LOCAL_EXPORT_ONLY]
        assert session_df.get_column(
            SessionKeys.REFERENCE_PERSISTENCE_POLICY
        ).to_list() == [CorpusPersistencePolicy.SERVER_SAVED]
        assert session_df.get_column(
            SessionKeys.CORPUS_PERSISTENCE_POLICY
        ).to_list() == [CorpusPersistencePolicy.SERVER_SAVED]
        assert get_session_persistence_policy(session_id, corpus_type='target') == (
            CorpusPersistencePolicy.LOCAL_EXPORT_ONLY
        )
        assert get_session_persistence_policy(session_id, corpus_type='reference') == (
            CorpusPersistencePolicy.SERVER_SAVED
        )

    @patch('streamlit.session_state', {})
    def test_auto_save_session_skips_non_durable_policy(self):
        session_id = 'test_session'
        st.session_state[session_id] = {
            'session': pl.from_dict({
                SessionKeys.TARGET_PERSISTENCE_POLICY: [
                    CorpusPersistencePolicy.TEMPORARY_SESSION_ONLY
                ],
                SessionKeys.REFERENCE_PERSISTENCE_POLICY: [
                    CorpusPersistencePolicy.TEMPORARY_SESSION_ONLY
                ],
                SessionKeys.HAS_TARGET: [True],
            })
        }

        manager = SessionPersistenceManager()
        manager.save_session = MagicMock(return_value=True)

        assert manager.auto_save_session(session_id) is True
        manager.save_session.assert_not_called()

    @patch('streamlit.session_state', {})
    def test_session_allows_persistence_with_mixed_corpus_policies(self):
        session_id = 'test_session'
        st.session_state[session_id] = {
            'session': pl.from_dict({
                SessionKeys.TARGET_PERSISTENCE_POLICY: [
                    CorpusPersistencePolicy.TEMPORARY_SESSION_ONLY
                ],
                SessionKeys.REFERENCE_PERSISTENCE_POLICY: [
                    CorpusPersistencePolicy.SERVER_SAVED
                ],
                SessionKeys.HAS_TARGET: [True],
                SessionKeys.HAS_REFERENCE: [True],
            })
        }

        assert session_allows_persistence(session_id) is True

    @patch('streamlit.session_state', {})
    def test_build_persistable_session_data_drops_heavy_tables(self):
        session_id = 'test_session'
        st.session_state[session_id] = {
            'session': pl.from_dict({
                SessionKeys.CORPUS_PERSISTENCE_POLICY: [
                    CorpusPersistencePolicy.SERVER_SAVED
                ],
                SessionKeys.TARGET_PERSISTENCE_POLICY: [
                    CorpusPersistencePolicy.SERVER_SAVED
                ],
                SessionKeys.REFERENCE_PERSISTENCE_POLICY: [
                    CorpusPersistencePolicy.SERVER_SAVED
                ],
                SessionKeys.HAS_TARGET: [True],
                SessionKeys.TARGET_DB: ['webapp/_corpora/demo'],
                SessionKeys.FREQ_TABLE: [True],
            }),
            SessionKeys.METADATA_TARGET: pl.from_dict({
                MetadataKeys.NDOCS: [3],
                MetadataKeys.DOCIDS: [{'ids': ['a', 'b', 'c']}],
                MetadataKeys.COLLOCATIONS: [{'temp': ['heavy']}],
            }, strict=False),
            'target': {
                '_artifact_refs': {
                    'ds_tokens': {
                        'storage_type': 'gzip_pickle',
                        'path': 'webapp/_corpora/demo/ds_tokens.gz',
                    }
                },
                'ds_tokens': pl.DataFrame({'token': ['x']}),
                'ft_pos': pl.DataFrame({'token': ['y']}),
            },
            'messages': [{'role': 'user', 'content': 'drop me'}],
        }

        projected = build_persistable_session_data(st.session_state[session_id])

        assert projected['session'][SessionKeys.HAS_TARGET] is True
        assert SessionKeys.FREQ_TABLE not in projected['session']
        assert projected['target']['_artifact_refs']['ds_tokens']['path'].endswith(
            'ds_tokens.gz'
        )
        assert 'ds_tokens' not in projected['target']
        assert MetadataKeys.NDOCS in projected[SessionKeys.METADATA_TARGET]
        assert MetadataKeys.COLLOCATIONS not in projected[SessionKeys.METADATA_TARGET]
        assert 'messages' not in projected

    @patch('streamlit.session_state', {})
    def test_build_persistable_session_data_omits_non_durable_target_state(self):
        session_id = 'test_session'
        st.session_state[session_id] = {
            'session': pl.from_dict({
                SessionKeys.TARGET_PERSISTENCE_POLICY: [
                    CorpusPersistencePolicy.TEMPORARY_SESSION_ONLY
                ],
                SessionKeys.REFERENCE_PERSISTENCE_POLICY: [
                    CorpusPersistencePolicy.SERVER_SAVED
                ],
                SessionKeys.HAS_TARGET: [True],
                SessionKeys.TARGET_DB: ['webapp/_corpora/uploaded'],
                SessionKeys.HAS_META: [True],
                SessionKeys.HAS_REFERENCE: [True],
                SessionKeys.REFERENCE_DB: ['webapp/_corpora/reference'],
            }),
            SessionKeys.METADATA_TARGET: pl.from_dict({
                MetadataKeys.NDOCS: [4],
            }),
            SessionKeys.METADATA_REFERENCE: pl.from_dict({
                MetadataKeys.NDOCS: [2],
            }),
            'target': {
                '_artifact_refs': {
                    'ds_tokens': {
                        'storage_type': 'gzip_pickle',
                        'path': 'webapp/_corpora/uploaded/ds_tokens.gz',
                    }
                },
            },
            'reference': {
                '_artifact_refs': {
                    'ds_tokens': {
                        'storage_type': 'gzip_pickle',
                        'path': 'webapp/_corpora/reference/ds_tokens.gz',
                    }
                },
            },
        }

        projected = build_persistable_session_data(st.session_state[session_id])

        assert SessionKeys.HAS_TARGET not in projected['session']
        assert SessionKeys.TARGET_DB not in projected['session']
        assert SessionKeys.METADATA_TARGET not in projected
        assert 'target' not in projected
        assert projected['session'][SessionKeys.HAS_REFERENCE] is True
        assert projected[SessionKeys.METADATA_REFERENCE][MetadataKeys.NDOCS] == [2]
        assert projected['reference']['_artifact_refs']['ds_tokens']['path'].endswith(
            'reference/ds_tokens.gz'
        )

    @patch('streamlit.session_state', {})
    def test_save_session_skips_backend_for_transient_only_changes(self):
        session_id = 'test_session'
        st.session_state[session_id] = {
            'session': pl.from_dict({
                SessionKeys.CORPUS_PERSISTENCE_POLICY: [
                    CorpusPersistencePolicy.SERVER_SAVED
                ],
                SessionKeys.TARGET_PERSISTENCE_POLICY: [
                    CorpusPersistencePolicy.SERVER_SAVED
                ],
                SessionKeys.REFERENCE_PERSISTENCE_POLICY: [
                    CorpusPersistencePolicy.SERVER_SAVED
                ],
                SessionKeys.HAS_TARGET: [True],
                SessionKeys.TARGET_DB: ['webapp/_corpora/demo'],
            }),
            'target': {
                '_artifact_refs': {
                    'ds_tokens': {
                        'storage_type': 'gzip_pickle',
                        'path': 'webapp/_corpora/demo/ds_tokens.gz',
                    }
                },
            },
        }

        manager = SessionPersistenceManager()
        manager._backend = MagicMock()
        manager._session_cache[session_id] = manager._hash_session_data(
            st.session_state[session_id]
        )
        manager._dirty_sessions.add(session_id)

        st.session_state[session_id]['target']['ft_pos'] = pl.DataFrame({'token': ['x']})

        assert manager.save_session(session_id) is True
        manager._backend.save_session.assert_not_called()
        assert session_id not in manager._dirty_sessions


class TestSQLiteSessionBackendUserHashing:
    def _create_backend(self, tmpdir: str) -> SQLiteSessionBackend:
        with patch.object(SQLiteSessionBackend, '_start_cleanup_thread', return_value=None):
            return SQLiteSessionBackend(storage_path=tmpdir)

    def _create_session_artifact(self, tmpdir: str, session_slug: str) -> tuple[Path, Path]:
        artifact_dir = Path(tmpdir) / 'corpora' / session_slug / 'target'
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / 'ds_tokens.gz'
        artifact_path.write_bytes(b'payload')
        sidecar_path = artifact_dir / 'metadata_descriptor.json'
        sidecar_path.write_text('{}', encoding='utf-8')
        return artifact_path, sidecar_path

    def _artifact_backed_payload(self, artifact_path: Path) -> dict:
        return {
            'session': {'has_target': True},
            'target': {
                '_artifact_refs': {
                    'ds_tokens': {
                        'storage_type': 'gzip_pickle',
                        'path': str(artifact_path),
                    }
                }
            },
        }

    def test_save_session_hashes_user_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = self._create_backend(tmpdir)

            assert backend.save_session(
                'session-1',
                {'session': {'has_target': True}},
                'person@example.com',
            ) is True

            with backend.pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT user_id FROM sessions WHERE session_id = ?',
                    ('session-1',),
                )
                stored_user_id = cursor.fetchone()[0]

            assert stored_user_id == persistent_hash('person@example.com')

    def test_delete_session_removes_session_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = self._create_backend(tmpdir)
            artifact_path, sidecar_path = self._create_session_artifact(tmpdir, 'abc123')

            assert backend.save_session(
                'session-1',
                self._artifact_backed_payload(artifact_path),
                'person@example.com',
            ) is True

            assert backend.delete_session('session-1') is True
            assert not artifact_path.exists()
            assert not sidecar_path.exists()

    def test_delete_session_uses_configured_session_artifact_root(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            configured_parent = Path(tmpdir) / 'configured'
            configured_root = configured_parent / 'corpora'
            monkeypatch.setenv(
                'DOCUSCOPE_SESSION_ARTIFACT_ROOT',
                str(configured_root),
            )
            backend = self._create_backend(tmpdir)
            artifact_path, sidecar_path = self._create_session_artifact(
                str(configured_parent),
                'configured123',
            )
            default_artifact_path, _ = self._create_session_artifact(
                tmpdir,
                'default123',
            )

            assert backend.save_session(
                'session-1',
                self._artifact_backed_payload(artifact_path),
                'person@example.com',
            ) is True
            assert backend.delete_session('session-1') is True

            assert not artifact_path.exists()
            assert not sidecar_path.exists()
            assert default_artifact_path.exists()

    def test_cleanup_expired_sessions_removes_session_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = self._create_backend(tmpdir)
            artifact_path, sidecar_path = self._create_session_artifact(
                tmpdir,
                'expired123',
            )

            assert backend.save_session(
                'session-expired',
                self._artifact_backed_payload(artifact_path),
                'person@example.com',
            ) is True

            with backend.pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE sessions "
                    "SET expires_at = datetime('now', '-1 second') "
                    "WHERE session_id = ?",
                    ('session-expired',),
                )
                conn.commit()

            assert backend.cleanup_expired_sessions() == 1
            assert not artifact_path.exists()
            assert not sidecar_path.exists()

    @patch('streamlit.session_state', {})
    def test_load_session_restores_file_backed_corpus_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = self._create_backend(tmpdir)
            artifact_dir = Path(tmpdir) / 'corpora' / 'restore123' / 'target'
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / 'ds_tokens.gz'

            ds_tokens = pl.DataFrame({
                'doc_id': ['doc1'],
                'token': ['hello'],
                'pos_tag': ['NN1'],
                'ds_tag': ['Character'],
                'pos_id': [0],
                'ds_id': [0],
            })

            import gzip
            import pickle

            with gzip.open(artifact_path, 'wb') as file_handle:
                pickle.dump(ds_tokens, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

            manager = SessionPersistenceManager()
            manager._backend = backend

            assert backend.save_session(
                'session-restore',
                {
                    'session': {
                        SessionKeys.HAS_TARGET: True,
                        SessionKeys.TARGET_PERSISTENCE_POLICY: (
                            CorpusPersistencePolicy.SERVER_SAVED
                        ),
                        SessionKeys.REFERENCE_PERSISTENCE_POLICY: (
                            CorpusPersistencePolicy.SERVER_SAVED
                        ),
                    },
                    'target': {
                        '_artifact_refs': {
                            'ds_tokens': {
                                'storage_type': 'gzip_pickle',
                                'path': str(artifact_path),
                            }
                        }
                    },
                },
                'person@example.com',
            ) is True

            assert manager.load_session('session-restore') is True

            corpus_manager = get_corpus_manager('session-restore', 'target')
            restored_tokens = corpus_manager.get_core_data()

            assert restored_tokens is not None
            assert restored_tokens.equals(ds_tokens)
