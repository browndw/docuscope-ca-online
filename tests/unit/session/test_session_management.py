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
from webapp.utilities.session.session_management import get_or_init_user_session
from webapp.utilities.state import SessionKeys


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