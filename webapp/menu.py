import base64
import pathlib

import streamlit as st

# Ensure project root is in sys.path for both desktop and online
project_root = pathlib.Path(__file__).parent.parent.resolve()

from webapp.utilities.configuration import import_options_general  # noqa: E402

OPTIONS = str(project_root.joinpath("webapp/config/options.toml"))
GOOGLE_LOGO = str(project_root.joinpath("webapp/_static/web_light_rd_na.svg"))

_options = import_options_general(OPTIONS)
DESKTOP = _options['global']['desktop_mode']


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
    if DESKTOP:
        authenticated_menu()
        st.sidebar.markdown("---")
        return
    
    # Check current login state
    current_login_state = hasattr(st, "user") and getattr(st.user, "is_logged_in", False)
    
    if current_login_state:
        # Ensure session_id is properly stored
        if "session_id" not in st.session_state:
            # Get the actual session ID from Streamlit's script run context
            try:
                user_session = (
                    st.runtime.scriptrunner_utils.script_run_context
                    .get_script_run_ctx()
                )
                st.session_state["session_id"] = user_session.session_id
            except Exception:
                # Fallback to a generated session ID
                import uuid
                st.session_state["session_id"] = str(uuid.uuid4())
        
        # Check if this is a new login (state changed from False to True)
        previous_login_state = st.session_state.get("previous_login_state", False)
        
        if not previous_login_state:
            # User just logged in - record the login
            from webapp.utilities.storage import add_login
            try:
                add_login(
                    user_id=st.user.email,
                    session_id=st.session_state["session_id"]
                )
            except Exception:
                # Silently handle any errors
                pass
        
        # Update the login state
        st.session_state["previous_login_state"] = True
        
        authenticated_menu()
        st.sidebar.markdown("---")
        return
    else:
        # User is not logged in
        st.session_state["previous_login_state"] = False
        unauthenticated_menu()
