"""Tests for shared frequency artifact caching in CorpusDataManager."""

import gzip
import pickle
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
from unittest.mock import patch

import polars as pl
import streamlit as st

from webapp.utilities.corpus.data_manager import CorpusDataManager
from webapp.persistence import SharedArtifactDecision


data_manager_module = sys.modules[CorpusDataManager.__module__]


def _make_tokens() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "doc_id": ["doc1"],
            "token": ["hello"],
            "pos_tag": ["NN1"],
            "ds_tag": ["AcademicTerms"],
        }
    )


def _make_frequency_table(tag: str) -> pl.DataFrame:
    return pl.DataFrame({"Tag": [tag], "AF": [1], "RF": [1.0], "Range": [1]})


class TestSharedFrequencyCache:
    """Test shared artifact caching for token-frequency generation."""

    def setup_method(self):
        self.user_session_id = "freq-cache-session"
        st.session_state.clear()
        with data_manager_module._artifact_frame_cache_state_lock:
            data_manager_module._artifact_frame_cache.clear()
            data_manager_module._artifact_frame_cache_locks.clear()
        st.session_state[self.user_session_id] = {
            "target": {"ds_tokens": _make_tokens()},
            "session": {"target_db": "webapp/_corpora/ld/A_MICUSP_mini"},
        }

    def test_generate_frequency_tables_uses_cache_hit(self):
        manager = CorpusDataManager(self.user_session_id, "target")
        cached = (_make_frequency_table("NN1"), _make_frequency_table("AcademicTerms"))

        with patch.object(manager, "_load_cached_frequency_tables", return_value=cached):
            with patch(
                "webapp.utilities.corpus.data_manager.ds.frequency_table"
            ) as mock_frequency:
                result = manager._generate_frequency_tables()

        assert result == cached
        mock_frequency.assert_not_called()

    def test_concurrent_lru_operations_remain_bounded(self):
        worker_count = 12
        iterations = 200
        start = Barrier(worker_count)

        def exercise_cache(worker_id: int) -> None:
            start.wait()
            for iteration in range(iterations):
                owner = f"worker-{worker_id}-{iteration}"
                key = f"frame-{iteration % 5}"
                frame = _make_frequency_table(str(iteration))
                data_manager_module._set_cached_artifact_frame(owner, key, frame)
                cached = data_manager_module._get_cached_artifact_frame(
                    owner,
                    key,
                )
                if cached is not None:
                    assert cached.equals(frame)
                if iteration % 3 == 0:
                    data_manager_module._clear_cached_artifact_frame(owner, key)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(executor.map(exercise_cache, range(worker_count)))

        with data_manager_module._artifact_frame_cache_state_lock:
            assert len(data_manager_module._artifact_frame_cache) <= (
                data_manager_module.ARTIFACT_FRAME_CACHE_MAX_ITEMS
            )
            assert data_manager_module._artifact_frame_cache_locks == {}

    def test_same_key_load_leases_serialize_and_are_released(self):
        worker_count = 12
        start = Barrier(worker_count)
        state_lock = Lock()
        overlap_window = Event()
        active_users = 0
        max_active_users = 0

        def exercise_load_lock(_worker_id: int) -> None:
            nonlocal active_users, max_active_users
            start.wait()
            with data_manager_module._artifact_frame_load_lock(17, "ft_pos"):
                with state_lock:
                    active_users += 1
                    max_active_users = max(max_active_users, active_users)
                overlap_window.wait(0.002)
                with state_lock:
                    active_users -= 1

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(executor.map(exercise_load_lock, range(worker_count)))

        assert max_active_users == 1
        with data_manager_module._artifact_frame_cache_state_lock:
            assert data_manager_module._artifact_frame_cache_locks == {}

    def test_get_data_reuses_ephemeral_artifact_frame_cache(self):
        manager = CorpusDataManager(self.user_session_id, "target")
        ft_pos = _make_frequency_table("NN1")
        ft_ds = _make_frequency_table("AcademicTerms")

        manager.set_artifact_refs("frequency", 17, ["ft_pos", "ft_ds"])

        artifact = type("Artifact", (), {"status": "ready"})()
        with patch.object(
            data_manager_module.registry_service,
            "get_public_artifact_by_id",
            return_value=artifact,
        ):
            with patch.object(
                data_manager_module.registry_service,
                "load_artifact_payload",
                return_value={"ft_pos": ft_pos, "ft_ds": ft_ds},
            ) as mock_load_payload:
                first = manager.get_data("ft_pos")
                second = manager.get_data("ft_pos")

        assert first.equals(ft_pos)
        assert second.equals(ft_pos)
        mock_load_payload.assert_called_once_with(artifact)

    def test_get_data_reuses_artifact_cached_general_pos_across_sessions(self):
        first_manager = CorpusDataManager(self.user_session_id, "target")
        second_session_id = "freq-cache-session-2"
        st.session_state[second_session_id] = {
            "target": {},
            "session": {"target_db": "webapp/_corpora/ld/A_MICUSP_mini"},
        }
        second_manager = CorpusDataManager(second_session_id, "target")

        ft_pos = _make_frequency_table("NN1")
        ft_ds = _make_frequency_table("AcademicTerms")
        ft_pos_general = _make_frequency_table("NOUN")

        first_manager.set_artifact_refs("frequency", 17, ["ft_pos", "ft_ds"])
        second_manager.set_artifact_refs("frequency", 17, ["ft_pos", "ft_ds"])

        artifact = type("Artifact", (), {"status": "ready"})()
        with patch.object(
            data_manager_module.registry_service,
            "get_public_artifact_by_id",
            return_value=artifact,
        ):
            with patch.object(
                data_manager_module.registry_service,
                "load_artifact_payload",
                return_value={"ft_pos": ft_pos, "ft_ds": ft_ds},
            ) as mock_load_payload:
                with patch(
                    "webapp.utilities.analysis.freq_simplify_pl",
                    return_value=ft_pos_general,
                ) as mock_simplify:
                    first = first_manager.get_data("ft_pos_general")
                    second = second_manager.get_data("ft_pos_general")

        assert first.equals(ft_pos_general)
        assert second.equals(ft_pos_general)
        mock_load_payload.assert_called_once_with(artifact)
        mock_simplify.assert_called_once_with(ft_pos)

    def test_get_data_reuses_shared_identity_cached_general_pos_without_refs(self):
        first_manager = CorpusDataManager(self.user_session_id, "target")
        second_session_id = "freq-cache-session-3"
        st.session_state[second_session_id] = {
            "target": {"ds_tokens": _make_tokens()},
            "session": {"target_db": "webapp/_corpora/ld/A_MICUSP_mini"},
        }
        second_manager = CorpusDataManager(second_session_id, "target")

        ft_pos = _make_frequency_table("NN1")
        ft_ds = _make_frequency_table("AcademicTerms")
        ft_pos_general = _make_frequency_table("NOUN")

        with patch.object(
            CorpusDataManager,
            "_generate_frequency_tables",
            return_value=(ft_pos, ft_ds),
        ) as mock_generate_frequency_tables:
            with patch(
                "webapp.utilities.analysis.freq_simplify_pl",
                return_value=ft_pos_general,
            ) as mock_simplify:
                first = first_manager.get_data("ft_pos_general")
                second = second_manager.get_data("ft_pos_general")

        assert first.equals(ft_pos_general)
        assert second.equals(ft_pos_general)
        mock_generate_frequency_tables.assert_called_once()
        mock_simplify.assert_called_once_with(ft_pos)
        assert (
            f"_dataframe_cache_{self.user_session_id}"
            not in st.session_state[self.user_session_id]
        )
        assert (
            f"_dataframe_cache_{second_session_id}"
            not in st.session_state[second_session_id]
        )

    def test_get_data_reuses_session_ephemeral_general_pos_alias(self):
        manager = CorpusDataManager(self.user_session_id, "target")
        ft_pos = _make_frequency_table("NN1")
        ft_ds = _make_frequency_table("AcademicTerms")
        ft_pos_general = _make_frequency_table("NOUN")

        with patch.object(
            CorpusDataManager,
            "_generate_frequency_tables",
            return_value=(ft_pos, ft_ds),
        ):
            with patch(
                "webapp.utilities.analysis.freq_simplify_pl",
                return_value=ft_pos_general,
            ) as mock_simplify:
                first = manager.get_data("ft_pos_general")

            session_cache_owner = data_manager_module._build_session_frame_cache_owner(
                self.user_session_id,
                "target",
            )
            with patch.object(
                data_manager_module,
                "_get_cached_artifact_frame",
                wraps=data_manager_module._get_cached_artifact_frame,
            ) as mock_get_cached_frame:
                second = manager.get_data("ft_pos_general")

        assert first.equals(ft_pos_general)
        assert second.equals(ft_pos_general)
        mock_simplify.assert_called_once_with(ft_pos)
        assert (
            f"_dataframe_cache_{self.user_session_id}"
            not in st.session_state[self.user_session_id]
        )
        assert mock_get_cached_frame.call_args_list == [
            ((session_cache_owner, "ft_pos_general"),)
        ]

    def test_get_data_reuses_file_backed_frame_cache_across_sessions(self):
        first_manager = CorpusDataManager(self.user_session_id, "target")
        second_session_id = "freq-cache-session-4"
        st.session_state[second_session_id] = {
            "target": {},
            "session": {"target_db": "webapp/_corpora/ld/A_MICUSP_mini"},
        }
        second_manager = CorpusDataManager(second_session_id, "target")

        ft_pos = _make_frequency_table("NN1")
        with tempfile.NamedTemporaryFile(suffix=".gz") as temp_file:
            with gzip.open(temp_file.name, "wb") as file_handle:
                pickle.dump(ft_pos, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

            first_manager.set_file_refs({"ft_pos": temp_file.name})
            second_manager.set_file_refs({"ft_pos": temp_file.name})

            with patch.object(
                data_manager_module.pickle,
                "load",
                wraps=data_manager_module.pickle.load,
            ) as mock_pickle_load:
                first = first_manager.get_data("ft_pos")
                second = second_manager.get_data("ft_pos")

        assert first.equals(ft_pos)
        assert second.equals(ft_pos)
        mock_pickle_load.assert_called_once()

    def test_warm_shared_frequency_data_skips_precomputed_file_refs(self):
        manager = CorpusDataManager(self.user_session_id, "target")
        manager.set_file_refs({
            "ft_pos": "/tmp/ft_pos.gz",
            "ft_ds": "/tmp/ft_ds.gz",
        })

        with patch.object(manager, "_load_cached_frequency_tables") as mock_load_cached:
            with patch.object(
                manager,
                "_reserve_shared_frequency_artifact",
            ) as mock_reserve:
                with patch(
                    "webapp.utilities.corpus.data_manager.ds.frequency_table"
                ) as mock_frequency:
                    result = manager.warm_shared_frequency_data()

        assert result == "precomputed_file_refs"
        mock_load_cached.assert_not_called()
        mock_reserve.assert_not_called()
        mock_frequency.assert_not_called()

    def test_generate_frequency_tables_stores_after_compute(self):
        manager = CorpusDataManager(self.user_session_id, "target")
        generated = (_make_frequency_table("NN1"), _make_frequency_table("AcademicTerms"))

        with patch.object(manager, "_load_cached_frequency_tables", return_value=None):
            with patch.object(
                manager,
                "_reserve_shared_frequency_artifact",
                return_value=SharedArtifactDecision("reserved", job_id=22),
            ):
                with patch.object(manager, "_store_cached_frequency_tables") as mock_store:
                    with patch(
                        "webapp.utilities.corpus.data_manager.ds.frequency_table",
                        return_value=generated,
                    ):
                        result = manager._generate_frequency_tables()

        assert result == generated
        mock_store.assert_called_once_with(22, generated[0], generated[1])

    def test_generate_frequency_tables_skips_duplicate_when_pending_ready(self):
        manager = CorpusDataManager(self.user_session_id, "target")
        cached = (_make_frequency_table("NN1"), _make_frequency_table("AcademicTerms"))

        with patch.object(manager, "_load_cached_frequency_tables", return_value=None):
            with patch.object(
                manager,
                "_reserve_shared_frequency_artifact",
                return_value=SharedArtifactDecision("ready", payload=cached),
            ):
                with patch(
                    "webapp.utilities.corpus.data_manager.ds.frequency_table"
                ) as mock_frequency:
                    result = manager._generate_frequency_tables()

        assert result == cached
        mock_frequency.assert_not_called()
