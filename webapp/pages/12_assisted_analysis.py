# Copyright (C) 2025 David West Brown

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib
import streamlit as st

from webapp.utilities.session import (  # noqa: E402
    get_or_init_user_session
)
from webapp.utilities.configuration import (  # noqa: E402
    get_ai_configuration
)
from webapp.utilities.ai import (   # noqa: E402
    clear_pandasai, pandabot_user_query,
    setup_ai_session_state, get_api_key,
    render_api_key_input, render_data_selection_interface,
    render_data_preview_controls
)
from webapp.utilities.analysis import (   # noqa: E402
    generate_tags_table
)
from webapp.utilities.ui import (  # noqa: E402
    sidebar_help_link, render_table_generation_interface
)
from webapp.utilities.state import (  # noqa: E402
    load_widget_state
)
from webapp.config.session_keys import (  # noqa: E402
    SessionKeys, WarningKeys
)
from webapp.menu import (   # noqa: E402
    menu, require_login
)

TITLE = "AI-Assisted Analysis"
ICON = ":material/smart_toy:"

# Configuration constants
DEFAULT_WIDGET_STATE_PATH = pathlib.Path(__file__).stem

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
)

# Get AI configuration
_options, DESKTOP, CACHE, LLM_MODEL, LLM_PARAMS, QUOTA = get_ai_configuration()


def render_pandabot_chat_interface(
    user_session_id: str,
    api_key: str,
    df,
    selected_query: str
) -> None:
    """Render the chat interface for Pandabot."""
    # Display chat history
    for message in st.session_state[user_session_id]["pandasai"]:
        with st.chat_message(message["role"]):
            if message["type"] == "string":
                st.markdown(message["value"])
            elif message["type"] == "code":
                st.code(message["value"], language="python")
            elif message["type"] == "error":
                st.error(message["value"], icon=":material/error:")
            elif message["type"] == "plot":
                # Display plot image
                st.image(message["value"])
            elif message["type"] == "dataframe":
                st.dataframe(
                    message["value"], use_container_width=True
                )

    # Chat input
    user_prompt = st.chat_input(
        "Ask a question about your data or request an analysis."
    )

    if user_prompt:
        with st.spinner(":sparkles: Analyzing data..."):
            st.session_state[user_session_id]["pandasai"].append(
                {"role": "user", "type": "string", "value": user_prompt}
            )
            # Increment user prompt count
            prompt_count_key = "pandabot_user_prompt_count"
            if prompt_count_key not in st.session_state[user_session_id]:
                st.session_state[user_session_id][prompt_count_key] = 1
            else:
                st.session_state[user_session_id][prompt_count_key] += 1

            # Generate response
            pandabot_user_query(
                df=df.to_pandas() if hasattr(df, 'to_pandas') else df,
                api_key=api_key,
                prompt=user_prompt,
                session_id=user_session_id,
                prompt_position=st.session_state[user_session_id][prompt_count_key],
                cache_mode=CACHE
            )
            st.rerun()


def render_pandabot_interface(user_session_id: str, session: dict) -> None:
    """Render the main Pandabot interface with data selection and analysis."""
    try:
        # Initialize session state
        setup_ai_session_state(user_session_id, "pandasai")

        # Get API key
        # Get API key first
        api_key = get_api_key(user_session_id, DESKTOP, CACHE, QUOTA)

        # Introduction
        st.markdown(
            body=(
                ":panda_face: Pandabot is a chat assistant designed to work "
                "with tabular data (or data frames).\n\n"
                ":material/priority: I can help you analyze, filter, and "
                "summarize your data using natural language.\n\n"
                ":material/priority: Ask me questions about patterns, "
                "statistics, or trends in your data."
            )
        )

        # Only show data interfaces if user has valid API key
        if api_key:
            # Add clear chat button to sidebar
            st.sidebar.markdown(
                body="### Chat Controls",
                help=(
                    "You can clear the chat history to start a new conversation. "
                    "This will remove all previous messages and plots."
                ))
            if st.sidebar.button(
                "Clear Chat History",
                icon=":material/delete:"
            ):
                clear_pandasai(user_session_id)
                st.rerun()

            # Get metadata if available
            metadata_target = None
            if session.get(SessionKeys.HAS_TARGET, [False])[0]:
                metadata_target = (
                    st.session_state[user_session_id]['metadata_target'].to_dict()
                )

            # Load widget state
            load_widget_state(DEFAULT_WIDGET_STATE_PATH, user_session_id)

            # Data selection interface
            selected_corpus, selected_query, df = render_data_selection_interface(
                user_session_id, session, "pandasai", DEFAULT_WIDGET_STATE_PATH,
                clear_pandasai, metadata_target
            )

            # Data preview with controls
            if df is not None:
                df = render_data_preview_controls(
                    df, selected_query, DEFAULT_WIDGET_STATE_PATH, user_session_id
                )

            # Chat interface
            render_pandabot_chat_interface(
                user_session_id, api_key, df, selected_query
            )
        else:
            # Show API key input if no valid key available
            render_api_key_input(user_session_id)

    except Exception as e:
        st.error(f"Error loading Pandabot interface: {str(e)}", icon=":material/error:")


def main():
    """Main function to run the Streamlit app for AI-assisted analysis."""
    # Set login requirements for navigation
    require_login()
    menu()
    st.markdown(
        body=f"## {TITLE}",
        help=(
            "To use Pandabot, you need to select a table from the sidebar. "
            "Once you have selected a table, you can enter your prompt "
            "in the chat input box. "
            "Pandabot will then generate a response based on the table you selected.\n\n"
            "If you are using the online version, you can use the API key "
            "provided by CMU, though there is a daily quota limit. "
            "If you're using the desktop version or you reach your quota, "
            "you can enter your own OpenAI API key to use Pandabot "
            "without any quota limits."
        )
    )

    # Get or initialize user session
    user_session_id, session = get_or_init_user_session()

    # Add help link
    sidebar_help_link("assisted-analysis.html")

    # Check if tags table is available
    if session.get(SessionKeys.TAGS_TABLE, [False])[0]:
        render_pandabot_interface(user_session_id, session)
    else:
        # Show generation interface for tags table
        render_table_generation_interface(
            user_session_id=user_session_id,
            session=session,
            table_type="tags table",
            button_label="Tags Table",
            generation_func=generate_tags_table,
            session_key=SessionKeys.TAGS_TABLE,
            warning_key=WarningKeys.TAGS
        )

    st.sidebar.markdown("---")


if __name__ == "__main__":
    main()
