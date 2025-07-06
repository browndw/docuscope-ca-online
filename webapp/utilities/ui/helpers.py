"""
Helper utilities for UI components.

This module provides miscellaneous helper functions for the user interface,
including file handling, page utilities, and other support functions.
"""

import pathlib
import streamlit as st

from webapp.utilities.exports import convert_to_excel
from webapp.utilities.state import SessionKeys
from webapp.utilities.session.session_core import safe_session_get

# Documentation base URL
DOCS_BASE_URL = "https://browndw.github.io/docuscope-docs/guide/"


def get_page_base_filename(file_path: str) -> str:
    """
    Extract the base filename from a file path for use in downloads and persistence.

    Parameters
    ----------
    file_path : str
        The full file path.

    Returns
    -------
    str
        The base filename without extension.
    """
    return pathlib.Path(file_path).stem


def sidebar_help_link(
        doc_page: str,
        label: str = "Help",
        icon: str = ":material/help:"
        ) -> None:
    """
    Render a styled help link at the top of the sidebar.

    Parameters
    ----------
    doc_page : str
        The page-specific part of the documentation URL (e.g., "token-frequencies.html").
    label : str
        The label for the help link.
    icon : str
        The icon to display with the help link.

    Returns
    -------
    None
        This function does not return anything.
        It renders a link in the sidebar that navigates to the documentation page.
    """
    st.sidebar.link_button(
        label=label,
        url=f"{DOCS_BASE_URL}{doc_page}",
        icon=icon
    )
    st.sidebar.markdown("<div style='margin-bottom: 0.5em'></div>", unsafe_allow_html=True)


def sidebar_action_button(
        button_label: str,
        button_icon: str,
        preconditions: list,  # Now just a list of bools
        action: callable,
        spinner_message: str = "Processing...",
        sidebar=True
        ) -> None:
    """
    Render a sidebar button that checks preconditions and runs an action.

    Parameters
    ----------
    button_label : str
        The label for the sidebar button.
    preconditions : list
        Lis of conditions.
        If any condition is False, show the error and do not run action.
    action : Callable
        Function to run if all preconditions are met.
    spinner_message : str
        Message to show in spinner.
    sidebar : bool
        If True, use st.sidebar, else use main area.
    error_in_sidebar : bool
        If True, show error messages in the sidebar,
        otherwise show in the main area.
    """
    container = st.sidebar if sidebar else st
    if container.button(button_label, icon=button_icon, type="secondary"):
        if not all(preconditions):
            st.warning(
                    body=(
                        "It doesn't look like you've loaded the necessary data yet."
                        ),
                    icon=":material/sentiment_stressed:"
                )
            return
        with container:
            with st.spinner(spinner_message):
                action()


def render_table_generation_interface(
    user_session_id: str,
    session: dict,
    table_type: str,
    button_label: str,
    generation_func: callable,
    session_key: str,
    warning_key: str = None
) -> None:
    """
    Render standardized table generation interface.

    Args:
        user_session_id: User session identifier
        session: Session state dictionary
        table_type: Type of table (e.g., "frequency table", "ngram table")
        button_label: Label for the generation button
        generation_func: Function to call for table generation
        session_key: Session key to check if table exists
        warning_key: Optional warning key to display warnings
    """
    st.markdown(
        body=(
            f":material/manufacturing: Use the button in the sidebar to "
            f"**generate a {table_type}**.\n\n"
            ":material/priority: A **target corpus** must be loaded first.\n\n"
            f":material/priority: After the {table_type} has been generated, "
            "you will be able to **toggle between the tagsets**."
        )
    )

    # Display warnings if warning_key is provided
    if warning_key and warning_key in st.session_state:
        st.warning(st.session_state[warning_key], icon=":material/warning:")

    # Display the sidebar header
    st.sidebar.markdown(
        body=(
            f"### Generate {table_type}\n\n"
            "Use the button to process a table."
        ),
        help=(
            f"{table_type.title()}s are generated based on the loaded target corpus. "
            "You can filter the table by tags after it has been generated. "
            f"The table will include data for the selected tagsets.\n\n"
            "Click on the **Help** button for more information on how to use this app."
        )
    )

    # Validate preconditions with better error messaging
    has_target = safe_session_get(session, SessionKeys.HAS_TARGET, False)
    if not has_target:
        st.sidebar.warning("Please load a target corpus first.", icon=":material/warning:")

    # Sidebar button for generation
    if st.sidebar.button(
        label=button_label,
        icon=":material/manufacturing:",
        type="secondary",
        use_container_width=False
    ):
        with st.spinner(f"Generating {table_type}..."):
            generation_func(user_session_id)


def toggle_download(
        label: str,
        user_session_id: str = None,
        convert_func: callable = convert_to_excel,
        convert_args: tuple = (),
        convert_kwargs: dict = None,
        file_name: str = "download.bin",
        mime: str = "application/octet-stream",
        message: str = None,
        location=None
        ) -> None:
    """
    Generalized toggle-based download for Streamlit with session scoping.

    Parameters
    ----------
    label : str
        The label for the toggle and download button.
    user_session_id : str, optional
        User session ID for proper widget scoping. If not provided,
        uses a simple non-scoped key (for backward compatibility).
    convert_func : callable
        The function to convert data to bytes.
    convert_args : tuple
        Positional arguments for the conversion function.
    convert_kwargs : dict
        Keyword arguments for the conversion function.
    file_name : str
        The name of the file to download.
    mime : str
        The MIME type of the file.
    message : str
        Optional markdown message to display above the button.
    location : Streamlit container
        Where to place the toggle and download button (default: st.sidebar).
    """
    if location is None:
        location = st.sidebar

    convert_kwargs = convert_kwargs or {}

    # Create session-scoped key if user_session_id is provided
    if user_session_id:
        toggle_key = f"toggle_{label.replace(' ', '_')}_{user_session_id}"
    else:
        # Backward compatibility - use simple key
        toggle_key = f"toggle_{label.replace(' ', '_')}"

    location.markdown("### Download Options")
    # Render the toggle button
    download = location.toggle(f"Download to {label}?", key=toggle_key)
    if download:
        with location.status(
            f"Generating {label} file...",
            expanded=True
        ):
            data = convert_func(*convert_args, **convert_kwargs)
            if message:
                st.success(message,
                           icon=":material/celebration:")
            location.download_button(
                label=f"Download {label} file",
                data=data,
                file_name=file_name,
                mime=mime,
                icon=":material/download:"
            )
