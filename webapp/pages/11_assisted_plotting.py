"""
This app provides an interface for AI-assisted plotting
from a loaded target corpus. Users can interact with Plotbot to create
and refine plots based on their data.

Users can:
- Select a plotting library (Plotly, Matplotlib, Seaborn)
- Interact with Plotbot to generate and refine plots
"""

import base64
import io
import json
import streamlit as st
from datetime import datetime, timezone

# Core application utilities with standardized patterns
from webapp.utilities.core import app_core
from webapp.config.unified import get_ai_config
from webapp.config.config_utils import get_runtime_setting

# Module-specific imports
from webapp.utilities.session import (
    get_or_init_user_session, safe_session_get
)
from webapp.utilities.ai import (
    clear_plotbot, previous_code_chunk,
    plotbot_user_query, setup_ai_session_state,
    get_api_key, render_api_key_input,
    render_data_selection_interface, render_data_preview_controls,
    render_quota_tracker, should_show_api_key_input,
    render_work_preservation_interface, should_show_work_preservation_interface,
    export_conversation_history, clear_plotbot_table
)
from webapp.utilities.analysis import (
    generate_tags_table
)
from webapp.utilities.ui import (
    sidebar_help_link, render_table_generation_interface,
    graceful_component
)
from webapp.utilities.state import (
    SessionKeys, WarningKeys
)
from webapp.menu import (
    menu, require_login
)

TITLE = "AI-Assisted Plotting"
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

# Register persistent widgets for this page
app_core.register_page_widgets(["plot_radio"])


def render_plotting_library_selection(user_session_id: str) -> str:
    """Render the plotting library selection interface."""
    st.markdown(
        body="### Plotting Library",
        help=(
            "To create plots, I can use the following libraries:\n\n"
            "* [Plotly express](https://plotly.com/python/plotly-express/)\n"
            "* [Matplotlib](https://matplotlib.org/)\n"
            "* [Seaborn](https://seaborn.pydata.org/)\n\n"
            "Each library has its own aesthetics and features. "
            "If you're unfamiliar with them, you should check out their "
            "documentation, as well as examples of their use."
        )
    )

    plot_lib = st.radio(
        "Select the plotting library:",
        ("plotly.express", "matplotlib", "seaborn"),
        key="plot_radio",
        on_change=clear_plotbot,
        args=(user_session_id, False,),
        horizontal=True
    )

    return plot_lib


def render_plotbot_chat_interface(
    user_session_id: str,
    api_key: str,
    df,
    selected_query: str,
    plot_lib: str
) -> None:
    """Render the chat interface for Plotbot."""
    # Convert DataFrame once for reuse in API calls
    if hasattr(df, 'to_pandas'):
        df_pandas = df.to_pandas()
    else:
        df_pandas = df
    # Display chat history
    for message in st.session_state[user_session_id]["plotbot"]:
        with st.chat_message(message["role"]):
            if message["type"] == "string":
                st.markdown(message["value"])
            elif message["type"] == "code":
                st.code(message["value"], language="python")
            elif message["type"] == "error":
                st.error(message["value"], icon=":material/error:")
            elif message["type"] == "plot":
                # Handle different plot types with safe rendering
                if plot_lib in ["matplotlib", "seaborn"]:
                    try:
                        fig = message['value']
                        # Convert matplotlib figure to image
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                        buf.seek(0)
                        img_bytes = buf.getvalue()
                        st.image(img_bytes)

                        # Add download link
                        b64 = base64.b64encode(img_bytes).decode()
                        href = (f'<a href="data:image/png;base64,{b64}" '
                                'download="plot.png">Download PNG</a>')
                        st.markdown(href, unsafe_allow_html=True)
                        buf.close()
                    except Exception as e:
                        st.error(
                            f"Failed to render plot: {str(e)}",
                            icon=":material/error:"
                        )

                elif plot_lib == "plotly.express":
                    try:
                        fig = message['value']
                        # Only call plotly methods on plotly figures
                        if hasattr(fig, 'update_xaxes'):
                            fig.update_xaxes(automargin=True)
                            fig.update_yaxes(automargin=True)
                        img_bytes = fig.to_image(format="png", scale=2)
                        st.image(img_bytes)
                        # Add download link
                        b64 = base64.b64encode(img_bytes).decode()
                        href = (f'<a href="data:image/png;base64,{b64}" '
                                'download="plot.png">Download PNG</a>')
                        st.markdown(href, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(
                            f"Failed to render plot: {str(e)}",
                            icon=":material/error:"
                        )

    # Get last code chunk
    last_code = previous_code_chunk(st.session_state[user_session_id]["plotbot"])

    # Chat input
    if last_code is None or len(last_code) == 0:
        input_prompt = st.chat_input(
            "Please describe what kind of plot you'd like to create."
        )

        if input_prompt:
            with st.spinner(":sparkles: Generating response..."):
                st.session_state[user_session_id]["plotbot"].append(
                    {"role": "user", "type": "string", "value": input_prompt}
                )
                # Increment user prompt count
                prompt_count_key = SessionKeys.AI_PLOTBOT_PROMPT_COUNT
                if prompt_count_key not in st.session_state[user_session_id]:
                    st.session_state[user_session_id][prompt_count_key] = 1
                else:
                    st.session_state[user_session_id][prompt_count_key] += 1

                # Generate response
                plotbot_user_query(
                    session_id=user_session_id,
                    df=df_pandas,
                    plot_lib=plot_lib,
                    user_input=input_prompt,
                    api_key=api_key,
                    llm_params=LLM_PARAMS,
                    code_chunk=last_code,
                    prompt_position=st.session_state[user_session_id][prompt_count_key],
                    cache_mode=get_runtime_setting('cache_mode', False, 'cache')
                )
                st.rerun()
    else:
        # Show refinement input
        input_refine = st.chat_input("How would you like me to refine this plot?")

        if input_refine:
            with st.spinner(":sparkles: Refining plot..."):
                st.session_state[user_session_id]["plotbot"].append(
                    {"role": "user", "type": "string", "value": input_refine}
                )
                st.session_state[user_session_id][SessionKeys.AI_PLOTBOT_PROMPT_COUNT] += 1

                # Generate refined response
                plotbot_user_query(
                    session_id=user_session_id,
                    df=df_pandas,
                    plot_lib=plot_lib,
                    user_input=input_refine,
                    api_key=api_key,
                    llm_params=LLM_PARAMS,
                    code_chunk=last_code,
                    prompt_position=st.session_state[user_session_id][
                        SessionKeys.AI_PLOTBOT_PROMPT_COUNT
                    ],
                    cache_mode=get_runtime_setting('cache_mode', False, 'cache')
                )
                st.rerun()


@graceful_component("Plotbot Interface", show_errors=True)
def render_plotbot_interface(user_session_id: str, session: dict) -> None:
    """Render the main Plotbot interface with data selection and plotting."""
    try:
        # Initialize session state
        setup_ai_session_state(user_session_id, "plotbot")

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
            user_email, user_session_id, has_user_key, "plotbot"
        )

        # Only show API key input if work preservation is not needed
        show_api_input = (
            should_show_api_key_input(user_email, has_user_key) and
            not show_work_preservation
        )

        # Introduction
        st.markdown(
            body=(
                ":robot_face: Plotbot is an **interactive** chat assistant "
                "designed to help you create and refine plots from your data.\n\n"
                ":material/priority: I remember your previous messages "
                "and use them to generate new responses.\n\n"
                ":material/priority: I am not a general-purpose chatbot, "
                "so I can not answer questions that are not related to plotting."
            )
        )

        # Show appropriate interface based on state
        if show_work_preservation:
            # Show work preservation interface when quota is exhausted but user has work
            render_work_preservation_interface(user_session_id, user_email, "plotbot")
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
                clear_plotbot(user_session_id)
                st.rerun()

            st.sidebar.markdown("---")

            # Add workflow export to sidebar
            st.sidebar.markdown(
                body="### Export Workflow",
                help=(
                    "You can export the conversation history as a JSON file. "
                    "This file contains all the steps and plots generated during your session."  # noqa: E501
                )
            )

            # Get workflow data
            workflow_json = export_conversation_history(user_session_id, "plotbot")

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
                filename = f"plotbot_workflow_{timestamp}.json"

                st.sidebar.download_button(
                    label="Download Workflow",
                    data=workflow_json,
                    file_name=filename,
                    mime="application/json",
                    icon=":material/file_download:"
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
                'plot_corpus_select', 'plot_query_select', 'plot_type_select',
                'plot_x_axis', 'plot_y_axis', 'plot_color_by'
            ])

            # Data selection interface
            selected_corpus, selected_query, df = render_data_selection_interface(
                user_session_id=user_session_id,
                session=session,
                bot_prefix="plotbot",
                clear_function=clear_plotbot_table,
                metadata_target=metadata_target
            )

            # Data preview with controls
            if df is not None:
                df = render_data_preview_controls(
                    df=df,
                    query=selected_query,
                    user_session_id=user_session_id
                )

            # Plotting library selection
            plot_lib = render_plotting_library_selection(user_session_id)

            # Chat interface
            render_plotbot_chat_interface(
                user_session_id, api_key, df, selected_query, plot_lib
            )

    except Exception as e:
        st.error(f"Error loading Plotbot interface: {str(e)}", icon=":material/error:")


def main():
    """Main function to run the Streamlit app for AI-assisted plotting."""
    # Set login requirements for navigation
    require_login()
    menu()

    st.markdown(
        body=f"## {TITLE}",
        help=(
            "To use Plotbot, you need to load tables from the sidebar, "
            "then select one from the interface. "
            "Once you have selected a table, you can enter your prompt "
            "in the chat input box. "
            "Plotbot will then generate a response based on the table you selected.\n\n"
            "If you are using the online version, you can use the API key "
            "provided by CMU, though there is a daily quota limit. "
            "If you're using the desktop version or you reach your quota, "
            "you can enter your own OpenAI API key to use Plotbot "
            "without any quota limits."
        )
    )

    # Get or initialize user session
    user_session_id, session = get_or_init_user_session()

    # Register persistent widgets for this page
    app_core.register_page_widgets(["plot_radio"])

    # Add help link
    sidebar_help_link("assisted-plotting.html")

    # Check if tags table is available
    if safe_session_get(session, SessionKeys.TAGS_TABLE, False):
        render_plotbot_interface(user_session_id, session)
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
