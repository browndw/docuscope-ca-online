"""
Menu system and session management for the DocuScope Corpus Analysis application.

This module provides navigation menu functions and implements a dual-timeout session
management system for online users:

1. Inactivity Timeout: Logs out users after 90 minutes of inactivity
2. Absolute Timeout: Logs out users after 24 hours regardless of activity

Both timeouts are configurable via options.toml and include warning systems.
"""

import base64
import os
import time

import streamlit as st

from webapp.utilities.storage import add_login
from webapp.config.unified import get_config
from webapp.utilities.core import safe_config_value
from webapp.utilities.configuration.logging_config import get_logger, setup_debug_logging
from webapp.utilities.session import get_or_init_user_session
from webapp.utilities.auth import is_user_authorized
from webapp.config.config_utils import get_runtime_setting

GOOGLE_LOGO = get_config('google_logo_path', 'global', 'webapp/_static/web_light_rd_na.svg')
DESKTOP = get_config('desktop_mode', 'global')
TEST_MODE = get_config('test_mode', 'global', False)
SLOW_MENU_STEP_MS = 25
setup_debug_logging("navigation_probe")
logger = get_logger()

CORE_PAGE_LINKS = [
    ("index.py", "Main Page", ":material/home:"),
    ("pages/1_load_corpus.py", "Manage Corpus Data", ":material/database:"),
    ("pages/2_token_frequencies.py", "Token Frequencies", ":material/table_view:"),
    ("pages/3_tag_frequencies.py", "Tag Frequencies", ":material/table_view:"),
    ("pages/4_ngrams.py", "Ngrams & Clusters", ":material/table_view:"),
    ("pages/5_compare_corpora.py", "Compare Corpora", ":material/compare_arrows:"),
    ("pages/6_compare_corpus_parts.py", "Compare Corpus Parts", ":material/compare_arrows:"),  # noqa: E501
    ("pages/7_collocations.py", "Collocations", ":material/network_node:"),
    ("pages/8_kwic.py", "Key Words in Context", ":material/network_node:"),
    ("pages/9_advanced_plotting.py", "Advanced Plotting", ":material/line_axis:"),
    ("pages/12_assisted_analysis.py", "Matrix Explorer", ":material/line_axis:"),
    ("pages/10_single_document.py", "Single Document", ":material/find_in_page:"),
    ("pages/11_assisted_plotting.py", "AI-Asissted Plotting", ":material/smart_toy:"),
    ("pages/13_download_corpus.py", "Download Corpus Data", ":material/download:"),
    ("pages/14_download_tagged_files.py", "Download Tagged Files", ":material/download:"),
]

ADMIN_PAGE_LINKS = [
    ("pages/98_user_management.py", "User Management", ":material/admin_panel_settings:"),
    ("pages/99_health_monitor.py", "Health Monitor", ":material/cardiology:"),
]

MINIMAL_PAGE_LINKS = CORE_PAGE_LINKS[:3]


def _load_test_minimal_menu_enabled() -> bool:
    """Return whether load-test-only minimal navigation should be used."""

    return os.environ.get("DOCUSCOPE_LOAD_TEST_MINIMAL_MENU", "").strip() == "1"


def navigation_experiment_enabled() -> bool:
    """Legacy compatibility shim for removed navigation experiment."""

    return False


def _render_page_links(page_links: list[tuple[str, str, str]]) -> None:
    """Render page links from a static link definition."""

    for page_path, label, icon in page_links:
        st.page_link(page_path, label=label, icon=icon)


def _admin_features_enabled(user_session_id: str | None) -> bool:
    """Return whether admin-only navigation should be shown for this session."""

    if TEST_MODE:
        return True

    if DESKTOP or user_session_id is None:
        return False

    if not hasattr(st, "user") or not getattr(st.user, "is_logged_in", False):
        return False

    user_email = getattr(st.user, "email", "")
    if not user_email:
        return False

    session_state = st.session_state.setdefault(user_session_id, {})
    cache_key = "menu_is_admin_user"
    cached_value = session_state.get(cache_key)
    if isinstance(cached_value, dict):
        if cached_value.get("email") == user_email:
            return bool(cached_value.get("authorized", False))
    elif cached_value is not None:
        return bool(cached_value)

    is_admin_user = is_user_authorized(user_email, 'admin')
    session_state[cache_key] = {
        "email": user_email,
        "authorized": is_admin_user,
    }
    return is_admin_user


def _is_authenticated() -> bool:
    """Return whether the current request should be treated as authenticated."""
    return TEST_MODE or (hasattr(st, "user") and getattr(st.user, "is_logged_in", False))


def update_last_activity(session_id) -> None:
    """Update the last activity timestamp for the current user session."""
    if not DESKTOP and not TEST_MODE and hasattr(st, "user") and getattr(st.user, "is_logged_in", False):  # noqa: E501
        st.session_state[session_id]["last_activity_time"] = time.time()


def check_session_timeouts(session_id) -> bool:
    """
    Check both inactivity and absolute session timeouts.

    Returns
    -------
    bool
        True if session is valid, False if should logout
    """
    if DESKTOP or TEST_MODE or not _is_authenticated():
        return True

    current_time = time.time()

    # Get timeout settings from config
    inactivity_timeout = (
        safe_config_value('inactivity_timeout_minutes', config_type='session') * 60
    )
    inactivity_warning = (
        safe_config_value('inactivity_warning_minutes', config_type='session') * 60
    )
    absolute_timeout = (
        safe_config_value('absolute_timeout_hours', config_type='session') * 3600
    )
    absolute_warning = (
        safe_config_value('absolute_warning_hours', config_type='session') * 3600
    )

    # Check absolute timeout (based on login time)
    if hasattr(st.user, 'iat'):
        login_time = st.user.iat
        session_duration = current_time - login_time

        if session_duration >= absolute_timeout:
            timeout_hours = safe_config_value(
                'absolute_timeout_hours', config_type='global'
            )
            st.error(
                f"Your session has expired after {timeout_hours} hours. "
                "Please log in again.",
                icon=":material/schedule:"
            )
            st.logout()
            return False
        elif session_duration >= absolute_warning:
            remaining_seconds = absolute_timeout - session_duration
            remaining_minutes = remaining_seconds / 60

            # Critical warning in final 30 seconds
            if remaining_seconds <= 30:
                st.error(
                    f"⚠️ SESSION EXPIRING IN {remaining_seconds:.0f} SECONDS! "
                    "Click anywhere to stay logged in!",
                    icon=":material/schedule:"
                )
                st.rerun()
            else:
                # Regular warning - no rerun to avoid disrupting work
                st.warning(
                    f"Your session will expire in {remaining_minutes:.0f} minutes. "
                    "Please save your work.",
                    icon=":material/schedule:"
                )

    # Check inactivity timeout
    last_activity = st.session_state[session_id].get("last_activity_time")
    if last_activity is None:
        # First time - set activity time
        update_last_activity(session_id)
        return True

    inactive_duration = current_time - last_activity

    if inactive_duration >= inactivity_timeout:
        timeout_minutes = safe_config_value(
            'inactivity_timeout_minutes', config_type='global'
        )
        st.error(
            f"You have been logged out due to inactivity ({timeout_minutes} minutes). "
            "Please log in again.",
            icon=":material/schedule:"
        )
        st.logout()
        return False
    elif inactive_duration >= inactivity_warning:
        remaining_seconds = inactivity_timeout - inactive_duration
        remaining_minutes = remaining_seconds / 60

        # Critical warning in final 30 seconds
        if remaining_seconds <= 30:
            st.error(
                f"⚠️ LOGGING OUT IN {remaining_seconds:.0f} SECONDS DUE TO INACTIVITY! "
                "Click anywhere to stay active!",
                icon=":material/schedule:"
            )
            st.rerun()
        else:
            # Regular warning - no rerun to avoid disrupting work
            st.warning(
                f"You've been inactive for {inactive_duration/60:.0f} minutes. "
                f"You'll be logged out in {remaining_minutes:.0f} minutes "
                "if no activity is detected.",
                icon=":material/schedule:"
            )

    return True


def unauthenticated_menu() -> None:
    with st.sidebar:
        with open(GOOGLE_LOGO, encoding='utf-8', errors='ignore') as f:
            google_logo_text = f.read()
        b64 = base64.b64encode(google_logo_text.encode('utf-8')).decode("utf-8")
        google_html = f"""
        <div class="image-txt-container">
            <img src="data:image/svg+xml;base64,{b64}" style="height:40px; margin-right:12px;"/>
            <span>To access the application, please log in with your Google account</span>
        </div>
        """  # noqa: E501
        st.markdown("# Please log in")
        st.markdown("---")
        st.markdown(google_html, unsafe_allow_html=True)
        st.markdown("---")
        st.button("Log in with Google", icon=":material/login:", on_click=st.login)


def authenticated_menu(user_session_id: str | None = None):
    minimal_menu = _load_test_minimal_menu_enabled()

    # Show log out button only if not DESKTOP and user is logged in
    if not DESKTOP and not TEST_MODE and hasattr(st, "user") and getattr(st.user, "is_logged_in", False):  # noqa: E501
        st.sidebar.button("Log out of Google", on_click=st.logout, icon=":material/logout:")

    if minimal_menu:
        with st.sidebar:
            _render_page_links(MINIMAL_PAGE_LINKS)
        return

    with st.sidebar.expander("**Navigation**",
                             icon=":material/explore:",
                             expanded=False):
        _render_page_links(CORE_PAGE_LINKS)

        # Admin-only features (only show in online mode with authorization)
        if _admin_features_enabled(user_session_id):

            st.markdown("---")
            st.markdown("**Admin Features**")
            _render_page_links(ADMIN_PAGE_LINKS)


def get_navigation_page_sections(
        user_session_id: str | None = None
) -> dict[str, list[tuple[str, str, str]]]:
    """Return page-link definitions for the navigation experiment."""

    page_sections = {"Navigation": CORE_PAGE_LINKS}
    if _admin_features_enabled(user_session_id):
        page_sections["Admin Features"] = ADMIN_PAGE_LINKS
    return page_sections


def render_navigation_sidebar_shell(is_authenticated: bool) -> None:
    """Render sidebar controls that should remain outside the navigation widget."""

    if is_authenticated:
        if not DESKTOP and not TEST_MODE and hasattr(st, "user") and getattr(st.user, "is_logged_in", False):  # noqa: E501
            st.sidebar.button(
                "Log out of Google",
                on_click=st.logout,
                icon=":material/logout:",
            )
        return

    if not DESKTOP:
        unauthenticated_menu()


def prepare_navigation_context() -> tuple[str | None, bool]:
    """Apply session timeout and login bookkeeping before st.navigation renders."""

    if DESKTOP or TEST_MODE:
        return None, True

    user_session_id, _ = get_or_init_user_session()

    current_login_state = _is_authenticated()
    if not current_login_state:
        st.session_state[user_session_id]["previous_login_state"] = False
        return user_session_id, False

    if not check_session_timeouts(user_session_id):
        st.session_state[user_session_id]["previous_login_state"] = False
        return user_session_id, False

    update_last_activity(user_session_id)

    previous_login_state = st.session_state[user_session_id].get(
        "previous_login_state",
        False,
    )

    cache_enabled = get_runtime_setting('cache_mode', False, 'cache')
    if cache_enabled and not previous_login_state:
        try:
            add_login(
                user_id=st.user.email,
                session_id=user_session_id,
            )
        except Exception:
            pass

    st.session_state[user_session_id]["previous_login_state"] = True
    return user_session_id, True


def require_login():
    """
    Redirect unauthenticated users to the login page and
    show the unauthenticated menu.
    """
    if not DESKTOP and not _is_authenticated():
        unauthenticated_menu()
        st.switch_page("index.py")
        st.stop()


def menu():

    if DESKTOP or TEST_MODE:
        authenticated_menu()
        st.sidebar.markdown("---")
        return

    user_session_id, current_login_state = prepare_navigation_context()

    if current_login_state:
        authenticated_menu(user_session_id)
        st.sidebar.markdown("---")
        return
    else:
        # User is not logged in
        st.session_state[user_session_id]["previous_login_state"] = False
        unauthenticated_menu()
