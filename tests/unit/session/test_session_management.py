"""Tests for webapp.utilities.session.session_management module."""

from types import SimpleNamespace
import sys
from unittest.mock import MagicMock, patch

try:
    import streamlit as st
except ImportError:
    sys.modules['streamlit'] = MagicMock()
    st = MagicMock()

from webapp.utilities.session import session_management
from webapp.utilities.session.session_management import (
    generate_temp,
    get_or_init_user_session,
)
from webapp.utilities.state import LoadCorpusKeys, SessionKeys


@patch('streamlit.session_state', {})
@patch.object(session_management, 'persist_session_changes')
@patch.object(session_management, 'ensure_session_loaded')
def test_generate_temp_backfills_reference_processing_state(
    mock_ensure_loaded,
    mock_persist_changes,
):
    session_id = 'test_session'
    st.session_state[session_id] = {
        LoadCorpusKeys.READY_TO_PROCESS: True,
    }

    generate_temp(
        {
            LoadCorpusKeys.READY_TO_PROCESS: False,
            LoadCorpusKeys.REF_READY_TO_PROCESS: False,
            LoadCorpusKeys.REF_CORPUS_DF: None,
            LoadCorpusKeys.REF_EXCEPTIONS: None,
        }.items(),
        session_id,
    )

    assert st.session_state[session_id][LoadCorpusKeys.READY_TO_PROCESS] is True
    assert st.session_state[session_id][LoadCorpusKeys.REF_READY_TO_PROCESS] is False
    assert st.session_state[session_id][LoadCorpusKeys.REF_CORPUS_DF] is None
    assert st.session_state[session_id][LoadCorpusKeys.REF_EXCEPTIONS] is None
    mock_ensure_loaded.assert_called_once_with(session_id)
    mock_persist_changes.assert_called_once_with(session_id)


class TestGetOrInitUserSession:
    """Test session bootstrap normalization behavior."""

    def setup_method(self):
        session_management._SESSION_DICT_CACHE.clear()

    @patch('streamlit.session_state', {})
    @patch(
        'streamlit.runtime.scriptrunner_utils.script_run_context.get_script_run_ctx'
    )
    def test_get_or_init_user_session_caches_dataframe_normalization(
        self,
        mock_get_ctx,
    ):
        """Repeated reads should reuse the normalized session payload."""

        session_id = 'test_session'
        mock_get_ctx.return_value = SimpleNamespace(session_id=session_id)

        session_frame = MagicMock()
        session_frame.columns = [SessionKeys.HAS_TARGET]
        session_frame.to_dict.return_value = {SessionKeys.HAS_TARGET: [True]}
        st.session_state[session_id] = {'session': session_frame}

        _, first_session = get_or_init_user_session()
        _, second_session = get_or_init_user_session()

        assert first_session == {SessionKeys.HAS_TARGET: [True]}
        assert second_session == {SessionKeys.HAS_TARGET: [True]}
        assert first_session is not second_session
        session_frame.to_dict.assert_called_once_with(as_series=False)

    @patch('streamlit.session_state', {})
    @patch(
        'streamlit.runtime.scriptrunner_utils.script_run_context.get_script_run_ctx'
    )
    def test_get_or_init_user_session_invalidates_cache_for_new_frame(
        self,
        mock_get_ctx,
    ):
        """Replacing the session frame should refresh the normalized cache."""

        session_id = 'test_session'
        mock_get_ctx.return_value = SimpleNamespace(session_id=session_id)

        first_frame = MagicMock()
        first_frame.columns = [SessionKeys.HAS_TARGET]
        first_frame.to_dict.return_value = {SessionKeys.HAS_TARGET: [False]}

        second_frame = MagicMock()
        second_frame.columns = [SessionKeys.HAS_TARGET]
        second_frame.to_dict.return_value = {SessionKeys.HAS_TARGET: [True]}

        st.session_state[session_id] = {'session': first_frame}
        _, first_session = get_or_init_user_session()

        st.session_state[session_id]['session'] = second_frame
        _, second_session = get_or_init_user_session()

        assert first_session == {SessionKeys.HAS_TARGET: [False]}
        assert second_session == {SessionKeys.HAS_TARGET: [True]}
        first_frame.to_dict.assert_called_once_with(as_series=False)
        second_frame.to_dict.assert_called_once_with(as_series=False)
