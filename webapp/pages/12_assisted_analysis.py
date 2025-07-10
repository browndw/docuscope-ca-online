"""
This app provides an interface for AI-assisted analysis of tabular data
using Pandabot, a chat assistant that can answer questions and perform
data analysis tasks.
"""

import json
import streamlit as st
from datetime import datetime, timezone

# Core application utilities with standardized patterns
from webapp.config.unified import get_ai_config
from webapp.config.config_utils import get_runtime_setting

# UI error boundaries (imported directly to avoid None fallback)
from webapp.utilities.ui.error_boundaries import SafeComponentRenderer

# Module-specific imports
from webapp.utilities.session import (
    get_or_init_user_session, safe_session_get
)
from webapp.utilities.ai import (
    clear_pandasai, pandabot_user_query,
    setup_ai_session_state, get_api_key,
    render_api_key_input, render_data_selection_interface,
    render_data_preview_controls, render_quota_tracker,
    should_show_api_key_input, render_work_preservation_interface,
    should_show_work_preservation_interface, export_conversation_history,
    clear_pandasai_table
)
from webapp.utilities.analysis import (
    generate_tags_table
)
from webapp.utilities.core import app_core
from webapp.utilities.ui import (
    sidebar_help_link, render_table_generation_interface
)
from webapp.utilities.state import (
    SessionKeys, WarningKeys
)
from webapp.menu import (
    menu, require_login
)

TITLE = "AI-Assisted Analysis"
ICON = ":material/smart_toy:"

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
)

# Get AI configuration using standardized access
AI_CONFIG = get_ai_config()
DESKTOP = AI_CONFIG['desktop_mode']
CACHE = AI_CONFIG['cache_enabled']
LLM_MODEL = AI_CONFIG['model']
LLM_PARAMS = AI_CONFIG['parameters']
QUOTA = AI_CONFIG['quota']


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
                # Display plot image with error boundary
                SafeComponentRenderer.safe_image(
                    message["value"], "Generated plot unavailable"
                )
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
            prompt_count_key = SessionKeys.AI_PANDABOT_PROMPT_COUNT
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
                cache_mode=get_runtime_setting('cache_mode', False, 'cache')
            )
            st.rerun()


def render_pandabot_interface(user_session_id: str, session: dict) -> None:
    """Render the main Pandabot interface with data selection and analysis."""
    try:
        # Initialize session state
        setup_ai_session_state(user_session_id, "pandasai")

        # Get user info for quota tracking
        try:
            user_email = (st.user.email if hasattr(st, 'user') and st.user.email
                          else 'anonymous')
        except Exception:
            user_email = session.get('user_email', 'anonymous')

        # Get API key first
        api_key = get_api_key(user_session_id, DESKTOP, CACHE, QUOTA)

        # Check if we should show API key input based on quota and current key status
        has_user_key = (
            api_key is not None and
            st.session_state[user_session_id].get(SessionKeys.AI_USER_KEY) is not None
        )

        # Render quota tracker in sidebar (for online mode)
        if not has_user_key:
            render_quota_tracker(user_email)

        # Check if we should show work preservation interface first
        show_work_preservation = should_show_work_preservation_interface(
            user_email, user_session_id, has_user_key, "pandabot"
        )

        # Only show API key input if work preservation is not needed
        show_api_input = (
            should_show_api_key_input(user_email, has_user_key) and
            not show_work_preservation
        )

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

        # Show appropriate interface based on state
        if show_work_preservation:
            # Show work preservation interface when quota is exhausted but user has work
            render_work_preservation_interface(user_session_id, user_email, "pandabot")
        elif show_api_input:
            # Show API key input when no work preservation needed
            render_api_key_input(user_session_id)
        elif api_key:
            # Add chat controls to sidebar
            st.sidebar.markdown(
                body="### Chat Controls",
                help=(
                    "You can clear the chat history to start a new conversation. "
                    "This will remove all previous messages and plots."
                ))
            if st.sidebar.button(
                "Clear Chat History",
                icon=":material/refresh:"
            ):
                clear_pandasai(user_session_id)
                st.rerun()

            st.sidebar.markdown("---")

            # Add workflow export to sidebar
            st.sidebar.markdown("### Export Workflow")

            # Get workflow data
            workflow_json = export_conversation_history(user_session_id, "pandabot")

            if workflow_json:
                # Parse to show summary
                try:
                    data = json.loads(workflow_json)
                    step_count = len(data.get("workflow_steps", []))
                    plot_count = data.get("summary", {}).get("plots_generated", 0)

                    st.sidebar.write(f"**{step_count}** conversation steps")
                    if plot_count > 0:
                        st.sidebar.write(f"**{plot_count}** plots included")
                except Exception:
                    st.sidebar.write("Workflow available")

                # Download button
                timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                filename = f"pandabot_workflow_{timestamp}.json"

                st.sidebar.download_button(
                    label="Download Workflow",
                    data=workflow_json,
                    file_name=filename,
                    mime="application/json",
                    icon=":material/file_download:",
                    help="Download your complete analysis workflow with embedded plots"
                )
            else:
                st.sidebar.info("Start a conversation to create a workflow")

            # Get metadata if available
            metadata_target = None
            if safe_session_get(session, SessionKeys.HAS_TARGET, False):
                from webapp.utilities.session import load_metadata
                from webapp.utilities.state import CorpusKeys
                metadata_target = load_metadata(CorpusKeys.TARGET, user_session_id)

            # Initialize widget state management
            app_core.widget_manager.register_persistent_keys([
                'analysis_corpus_select', 'analysis_query_select', 'analysis_prompt',
                'analysis_model_select'
            ])

            # Data selection interface
            selected_corpus, selected_query, df = render_data_selection_interface(
                user_session_id=user_session_id,
                session=session,
                bot_prefix="pandasai",
                clear_function=clear_pandasai_table,
                metadata_target=metadata_target
            )

            # Data preview with controls
            if df is not None:
                df = render_data_preview_controls(
                    df=df,
                    query=selected_query,
                    user_session_id=user_session_id
                )

            # Chat interface
            render_pandabot_chat_interface(
                user_session_id, api_key, df, selected_query
            )

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
    if safe_session_get(session, SessionKeys.TAGS_TABLE, False):
        render_pandabot_interface(user_session_id, session)
    else:
        # Show generation interface for tags table
        render_table_generation_interface(
            user_session_id=user_session_id,
            session=session,
            table_type="tags table",
            button_label="Load Tables",
            generation_func=generate_tags_table,
            session_key=SessionKeys.TAGS_TABLE,
            warning_key=WarningKeys.TAGS
        )

    st.sidebar.markdown("---")


if __name__ == "__main__":
    main()
