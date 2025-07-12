"""
Menu system and session management for the DocuScope Corpus Analysis application.

This module provides navigation menu functions and implements a dual-timeout session
management system for online users:

1. Inactivity Timeout: Logs out users after 90 minutes of inactivity
2. Absolute Timeout: Logs out users after 24 hours regardless of activity

Both timeouts are configurable via options.toml and include warning systems.
"""

import base64
import time

import streamlit as st

from webapp.utilities.storage import add_login
from webapp.config.unified import get_config
from webapp.utilities.core import safe_config_value
from webapp.utilities.session import get_or_init_user_session
from webapp.utilities.auth import is_user_authorized
from webapp.config.config_utils import get_runtime_setting

GOOGLE_LOGO = get_config('google_logo_path', 'global', 'webapp/_static/web_light_rd_na.svg')
DESKTOP = get_config('desktop_mode', 'global')


def update_last_activity(session_id) -> None:
    """Update the last activity timestamp for the current user session."""
    if not DESKTOP and hasattr(st, "user") and getattr(st.user, "is_logged_in", False):
        st.session_state[session_id]["last_activity_time"] = time.time()


def check_session_timeouts(session_id) -> bool:
    """
    Check both inactivity and absolute session timeouts.

    Returns
    -------
    bool
        True if session is valid, False if should logout
    """
    if DESKTOP or not (hasattr(st, "user") and getattr(st.user, "is_logged_in", False)):
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


def authenticated_menu():
    # Show log out button only if not DESKTOP and user is logged in
    if not DESKTOP and hasattr(st, "user") and getattr(st.user, "is_logged_in", False):
        st.sidebar.button("Log out of Google", on_click=st.logout, icon=":material/logout:")

    with st.sidebar.expander("**Navigation**",
                             icon=":material/explore:",
                             expanded=False):
        st.page_link("index.py",
                     label="Main Page",
                     icon=":material/home:")
        st.page_link("pages/1_load_corpus.py",
                     label="Manage Corpus Data",
                     icon=":material/database:")
        st.page_link("pages/2_token_frequencies.py",
                     label="Token Frequencies",
                     icon=":material/table_view:")
        st.page_link("pages/3_tag_frequencies.py",
                     label="Tag Frequencies",
                     icon=":material/table_view:")
        st.page_link("pages/4_ngrams.py",
                     label="Ngrams & Clusters",
                     icon=":material/table_view:")
        st.page_link("pages/5_compare_corpora.py",
                     label="Compare Corpora",
                     icon=":material/compare_arrows:")
        st.page_link("pages/6_compare_corpus_parts.py",
                     label="Compare Corpus Parts",
                     icon=":material/compare_arrows:")
        st.page_link("pages/7_collocations.py",
                     label="Collocations",
                     icon=":material/network_node:")
        st.page_link("pages/8_kwic.py",
                     label="Key Words in Context",
                     icon=":material/network_node:")
        st.page_link("pages/9_advanced_plotting.py",
                     label="Advanced Plotting",
                     icon=":material/line_axis:")
        st.page_link("pages/10_single_document.py",
                     label="Single Document",
                     icon=":material/find_in_page:")
        st.page_link("pages/11_assisted_plotting.py",
                     label="AI-Asissted Plotting",
                     icon=":material/smart_toy:")
        st.page_link("pages/12_assisted_analysis.py",
                     label="AI-Assisted Analysis",
                     icon=":material/smart_toy:")
        st.page_link("pages/13_download_corpus.py",
                     label="Download Corpus Data",
                     icon=":material/download:")
        st.page_link("pages/14_download_tagged_files.py",
                     label="Download Tagged Files",
                     icon=":material/download:")

        # Admin-only features (only show in online mode with authorization)
        if (not DESKTOP and hasattr(st, "user") and
                getattr(st.user, "is_logged_in", False) and
                is_user_authorized(st.user.email, 'admin')):

            st.markdown("---")
            st.markdown("**Admin Features**")
            st.page_link("pages/98_user_management.py",
                         label="User Management",
                         icon=":material/admin_panel_settings:")
            st.page_link("pages/99_health_monitor.py",
                         label="Health Monitor",
                         icon=":material/cardiology:")


def require_login():
    """
    Redirect unauthenticated users to the login page and
    show the unauthenticated menu.
    """
    if not DESKTOP and not (hasattr(st, "user") and getattr(st.user, "is_logged_in", False)):  # noqa: E501
        unauthenticated_menu()
        # If you have a dedicated login page, use st.switch_page:
        st.switch_page("index.py")  # <-- adjust path if needed
        st.stop()


def menu():
    # Hide default Streamlit navigation to prevent duplication with custom menu
    hide_nav_css = """
        <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
        </style>
    """
    st.markdown(hide_nav_css, unsafe_allow_html=True)

    if DESKTOP:
        authenticated_menu()
        st.sidebar.markdown("---")
        return

    user_session_id, _ = get_or_init_user_session()
    # Check current login state
    current_login_state = hasattr(st, "user") and getattr(st.user, "is_logged_in", False)

    if current_login_state:
        # Check session timeouts first - this may log the user out
        if not check_session_timeouts(user_session_id):
            # User was logged out due to timeout
            st.session_state[user_session_id]["previous_login_state"] = False
            unauthenticated_menu()
            return

        # Update activity timestamp for valid interaction
        update_last_activity(user_session_id)

        # Check if this is a new login (state changed from False to True)
        previous_login_state = st.session_state[user_session_id].get(
            "previous_login_state", False
            )

        CACHE = get_runtime_setting('cache_mode', False, 'cache')
        if CACHE and not previous_login_state:
            # User just logged in - record the login
            try:
                add_login(
                    user_id=st.user.email,
                    session_id=user_session_id
                )
            except Exception:
                # Silently handle any errors
                pass

        # Update the login state
        st.session_state[user_session_id]["previous_login_state"] = True

        authenticated_menu()
        st.sidebar.markdown("---")
        return
    else:
        # User is not logged in
        st.session_state[user_session_id]["previous_login_state"] = False
        unauthenticated_menu()
