"""Tests for shared keyness artifact caching."""

from unittest.mock import MagicMock, patch

import polars as pl
import streamlit as st

from webapp.utilities.analysis import statistical_analysis as stats
from webapp.utilities.state import WarningKeys


def _make_freq_df() -> pl.DataFrame:
    return pl.DataFrame({"Tag": ["NN1"], "AF": [1], "RF": [1.0], "Range": [1]})


def _make_keyness_df() -> pl.DataFrame:
    return pl.DataFrame({"Tag": ["NN1"], "LL": [1.0], "LR": [0.5]})


class TestSharedKeynessCache:
    """Test shared cache integration for built-in keyness workflows."""

    @patch("streamlit.rerun")
    @patch("streamlit.success")
    @patch.object(stats, "get_corpus_data_manager")
    @patch.object(stats, "_get_shared_keyness_identity")
    def test_load_cached_keyness_tables_uses_registry_hit(
        self,
        mock_identity,
        mock_get_manager,
        mock_success,
        mock_rerun,
    ):
        user_session_id = "user-session"
        st.session_state.clear()
        st.session_state[user_session_id] = {WarningKeys.KEYNESS: ("old", "icon")}

        identity = MagicMock(selector_hash="selector", parameter_hash="params")
        artifact = MagicMock(artifact_id=123)
        keyness_frames = {
            "kw_pos": _make_keyness_df(),
            "kw_ds": _make_keyness_df(),
            "kt_pos": _make_keyness_df(),
            "kt_ds": _make_keyness_df(),
        }

        mock_identity.return_value = identity
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager

        with patch.object(
            stats.registry_service,
            "find_ready_artifact",
            return_value=artifact,
        ):
            with patch.object(
                stats.registry_service,
                "load_keyness_bundle",
                return_value=keyness_frames,
            ):
                with patch.object(stats.app_core, "session_manager", MagicMock()):
                    loaded = stats._load_cached_keyness_tables(user_session_id, 0.01, False)

        assert loaded is True
        mock_manager.set_artifact_refs.assert_called_once_with(
            artifact.artifact_type,
            artifact.artifact_id,
            [
                stats.TargetKeys.KW_POS,
                stats.TargetKeys.KW_DS,
                stats.TargetKeys.KT_POS,
                stats.TargetKeys.KT_DS,
            ],
        )
        mock_success.assert_called_once_with("Keywords loaded from shared cache!")
        mock_rerun.assert_called_once()
        assert st.session_state[user_session_id][WarningKeys.KEYNESS] is None

    @patch.object(stats.ds, "keyness_table")
    @patch.object(stats, "_load_cached_keyness_tables", return_value=True)
    @patch.object(stats, "get_corpus_data")
    def test_generate_keyness_tables_skips_recompute_on_cache_hit(
        self,
        mock_get_corpus_data,
        mock_cached_load,
        mock_keyness_table,
    ):
        user_session_id = "user-session"
        st.session_state.clear()
        st.session_state[user_session_id] = {}

        mock_get_corpus_data.side_effect = [_make_freq_df()] * 8

        stats.generate_keyness_tables(user_session_id, threshold=0.01, swap_target=False)

        mock_cached_load.assert_called_once_with(user_session_id, 0.01, False)
        mock_keyness_table.assert_not_called()

    @patch("streamlit.rerun")
    @patch("streamlit.success")
    @patch.object(stats, "_store_cached_keyness_tables")
    @patch.object(stats, "_reserve_shared_keyness_artifact", return_value=77)
    @patch.object(stats, "_get_shared_keyness_identity", return_value=MagicMock())
    @patch.object(stats, "set_corpus_data")
    @patch.object(stats.ds, "keyness_table")
    @patch.object(stats, "_load_cached_keyness_tables", return_value=False)
    @patch.object(stats, "get_corpus_data")
    def test_generate_keyness_tables_stores_cache_after_compute(
        self,
        mock_get_corpus_data,
        mock_cached_load,
        mock_keyness_table,
        mock_set_corpus_data,
        mock_identity,
        mock_reserve,
        mock_store_cache,
        mock_success,
        mock_rerun,
    ):
        user_session_id = "user-session"
        st.session_state.clear()
        st.session_state[user_session_id] = {}

        mock_get_corpus_data.side_effect = [_make_freq_df()] * 8
        mock_keyness_table.side_effect = [_make_keyness_df()] * 4

        with patch.object(stats.app_core, "session_manager", MagicMock()):
            stats.generate_keyness_tables(user_session_id, threshold=0.05, swap_target=True)

        assert mock_keyness_table.call_count == 4
        mock_set_corpus_data.assert_not_called()
        mock_identity.assert_called()
        mock_reserve.assert_called_once_with(user_session_id, 0.05, True)
        mock_store_cache.assert_called_once()
        args = mock_store_cache.call_args.args
        assert args[0] == user_session_id
        assert args[1] == 0.05
        assert args[2] is True
        assert args[3] == 77
        mock_success.assert_called_once_with("Keywords generated!")
        mock_rerun.assert_called_once()

    @patch.object(stats, "_get_shared_keyness_identity")
    def test_reserve_shared_keyness_artifact_returns_true_when_reserved(
        self,
        mock_identity,
    ):
        user_session_id = "user-session"
        mock_identity.return_value = MagicMock(selector_hash="selector")

        reservation = MagicMock(state="reserved", job=MagicMock(job_id=99))
        with patch.object(
            stats.registry_service,
            "reserve_artifact",
            return_value=reservation,
        ):
            with patch.object(stats.registry_service, "mark_job_running") as mock_running:
                job_id = stats._reserve_shared_keyness_artifact(
                    user_session_id,
                    0.01,
                    False,
                )

        assert job_id == 99
        mock_running.assert_called_once_with(99)

    @patch.object(stats, "_get_shared_keyness_identity")
    def test_reserve_shared_keyness_artifact_blocks_duplicate_pending_work(
        self,
        mock_identity,
    ):
        user_session_id = "user-session"
        st.session_state.clear()
        st.session_state[user_session_id] = {}
        mock_identity.return_value = MagicMock(selector_hash="selector")

        reservation = MagicMock(state="pending")
        with patch.object(
            stats.registry_service,
            "reserve_artifact",
            return_value=reservation,
        ):
            proceed = stats._reserve_shared_keyness_artifact(user_session_id, 0.01, False)

        assert proceed is None
        warning = st.session_state[user_session_id][WarningKeys.KEYNESS]
        assert "already being generated" in warning[0]

    @patch.object(stats, "_get_shared_keyness_identity", return_value=MagicMock())
    @patch.object(stats, "_reserve_shared_keyness_artifact", return_value=None)
    @patch.object(stats, "_load_cached_keyness_tables", return_value=False)
    @patch.object(stats.ds, "keyness_table")
    @patch.object(stats, "get_corpus_data")
    def test_generate_keyness_tables_skips_compute_when_reservation_blocks(
        self,
        mock_get_corpus_data,
        mock_keyness_table,
        mock_cached_load,
        mock_reserve,
        mock_identity,
    ):
        user_session_id = "user-session"
        st.session_state.clear()
        st.session_state[user_session_id] = {}
        mock_get_corpus_data.side_effect = [_make_freq_df()] * 8

        stats.generate_keyness_tables(user_session_id, threshold=0.01, swap_target=False)

        mock_cached_load.assert_called_once_with(user_session_id, 0.01, False)
        mock_reserve.assert_called_once_with(user_session_id, 0.01, False)
        mock_keyness_table.assert_not_called()

    def test_reserve_artifact_creates_pending_job_and_marks_complete(self):
        identity = stats.build_shared_keyness_identity(
            target_source='webapp/_corpora/ld/A_MICUSP_mini',
            reference_source='webapp/_corpora/ld/C_BAWE_mini',
            threshold=0.01,
            swap_target=False,
        )
        reservation = stats.registry_service.reserve_artifact(identity)
        assert reservation.state in {"reserved", "ready", "pending"}
