"""
This app provides an interface for AI-assisted plotting
from a loaded target corpus. Users can interact with Plotbot to create
and refine plots based on their data.

Users can interact with Plotbot to generate and refine Plotly Express plots.
"""

import base64
import json
import streamlit as st
from datetime import datetime, timezone

# Core application utilities with standardized patterns
from webapp.utilities.core import app_core
from webapp.config.unified import get_ai_config
from webapp.config.config_utils import get_runtime_setting

# Module-specific imports
from webapp.utilities.session import (
    get_or_init_user_session, safe_session_get, load_metadata
)
from webapp.utilities.ai import (
    clear_plotbot, previous_code_chunk,
    plotbot_user_query, setup_ai_session_state,
    get_api_key, render_api_key_input,
    render_data_selection_interface, render_data_preview_controls,
    render_quota_tracker, should_show_api_key_input,
    render_work_preservation_interface, should_show_work_preservation_interface,
    export_conversation_history, clear_plotbot_table,
    prune_message_thread
)
from webapp.utilities.ai.providers import get_openai_compatible_provider_config
from webapp.utilities.configuration.logging_config import get_logger
from webapp.persistence import registry_service
from webapp.queue import (
    enqueue_plotbot_generation,
    get_queue,
    get_redis_queue_config,
)
from webapp.utilities.analysis import (
    generate_tags_table
)
from webapp.utilities.ui import (
    sidebar_help_link, render_table_generation_interface,
    graceful_component
)
from webapp.utilities.state import (
    SessionKeys, WarningKeys, CorpusKeys
)
from webapp.menu import (
    menu, require_login
)

TITLE = "AI-Assisted Plotting"
ICON = ":material/smart_toy:"

logger = get_logger()

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

PLOTBOT_QUEUE_STATE_KEY = "plotbot_queue_state"
PLOTBOT_LIBRARY = "plotly.express"


def _get_plotbot_queue_state(user_session_id: str) -> dict | None:
    """Return pending Plotbot queue state for this Streamlit session."""
    queue_state = st.session_state.get(PLOTBOT_QUEUE_STATE_KEY)
    if not isinstance(queue_state, dict):
        return None
    if queue_state.get("session_id") != user_session_id:
        return None
    return queue_state


def _clear_plotbot_queue_state(user_session_id: str) -> None:
    """Clear pending Plotbot queue state for this Streamlit session."""
    if _get_plotbot_queue_state(user_session_id) is not None:
        st.session_state.pop(PLOTBOT_QUEUE_STATE_KEY, None)


def _has_local_plotbot_provider() -> bool:
    """Return True when Plotbot can use a configured local model provider."""
    return get_openai_compatible_provider_config() is not None


def _dataframe_to_plotbot_records(df) -> list[dict[str, object]]:
    """Convert a selected Plotbot dataframe to JSON-friendly row records."""
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _append_queued_plotbot_result(user_session_id: str, artifact_id: int) -> bool:
    """Attach a completed queued Plotbot JSON result to the chat state."""
    artifact = registry_service.get_artifact_by_id(artifact_id)
    if artifact is None:
        return False

    try:
        payload = registry_service.load_json_artifact(artifact)
    except Exception as exc:
        logger.warning(f"Queued Plotbot artifact load failed for {artifact_id}: {exc}")
        return False

    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, dict):
        return False

    if result.get("result_type") != "plot":
        error_message = result.get("error") or "Plotbot did not generate a plot."
        st.session_state[user_session_id][SessionKeys.AI_PLOTBOT_CHAT].append(
            {"role": "assistant", "type": "error", "value": error_message}
        )
        st.session_state[user_session_id]["plotbot"].append(
            {"role": "assistant", "type": "error", "value": error_message}
        )
        prune_message_thread(user_session_id, SessionKeys.AI_PLOTBOT_CHAT)
        prune_message_thread(user_session_id, "plotbot")
        return True

    plot_code = result.get("code")
    if isinstance(plot_code, str) and plot_code.strip():
        code_message = {"role": "assistant", "type": "code", "value": plot_code}
        st.session_state[user_session_id][SessionKeys.AI_PLOTBOT_CHAT].append(code_message)
        st.session_state[user_session_id]["plotbot"].append(code_message)

    plot_svg = result.get("plot_svg")
    if isinstance(plot_svg, str) and plot_svg.strip():
        st.session_state[user_session_id]["plotbot_plot_svg"] = plot_svg
        st.session_state[user_session_id]["plotbot"].append(
            {"role": "assistant", "type": "plot_svg", "value": plot_svg}
        )

    prune_message_thread(user_session_id, SessionKeys.AI_PLOTBOT_CHAT)
    prune_message_thread(user_session_id, "plotbot")
    return True


def _submit_plotbot_job(
    user_session_id: str,
    df,
    plot_lib: str,
    user_input: str,
    api_key: str,
    llm_params: dict,
    code_chunk: str | None,
    user_email: str,
) -> None:
    """Submit Plotbot generation to Redis/RQ and store pending state."""
    if df is None:
        error_message = "No plot was generated. Please select a table first."
        st.session_state[user_session_id][SessionKeys.AI_PLOTBOT_CHAT].append(
            {"role": "assistant", "type": "error", "value": error_message}
        )
        st.session_state[user_session_id]["plotbot"].append(
            {"role": "assistant", "type": "error", "value": error_message}
        )
        prune_message_thread(user_session_id, SessionKeys.AI_PLOTBOT_CHAT)
        prune_message_thread(user_session_id, "plotbot")
        return

    dataframe_records = _dataframe_to_plotbot_records(df)
    schema = df.dtypes.to_string() if hasattr(df, "dtypes") else str(type(df))
    result = enqueue_plotbot_generation(
        dataframe_records=dataframe_records,
        plot_lib=plot_lib,
        user_input=user_input,
        llm_params=llm_params,
        schema=schema,
        code_chunk=code_chunk,
        api_key=api_key,
        requester_principal_id=user_email or "anonymous",
    )

    if result.state == "ready" and result.artifact_id is not None:
        _clear_plotbot_queue_state(user_session_id)
        if _append_queued_plotbot_result(user_session_id, result.artifact_id):
            st.rerun()
        st.error("Ready Plotbot result could not be attached. Please retry.")
        return

    st.session_state[PLOTBOT_QUEUE_STATE_KEY] = {
        "session_id": user_session_id,
        "control_plane_job_id": result.control_plane_job_id,
        "rq_job_id": result.rq_job_id,
        "artifact_id": result.artifact_id,
    }
    st.rerun()


@st.fragment(run_every="2s")
def _render_plotbot_queue_status(user_session_id: str) -> None:
    """Poll queued Plotbot status and attach the serialized result."""
    queue_state = _get_plotbot_queue_state(user_session_id)
    if queue_state is None:
        return

    control_plane_job_id = queue_state.get("control_plane_job_id")
    rq_job_id = queue_state.get("rq_job_id")
    if not isinstance(rq_job_id, str) or not rq_job_id:
        if isinstance(control_plane_job_id, int):
            rq_job_id = f"plotbot-{control_plane_job_id}"
            queue_state["rq_job_id"] = rq_job_id
            st.session_state[PLOTBOT_QUEUE_STATE_KEY] = queue_state
        else:
            rq_job_id = None

    if control_plane_job_id is None:
        _clear_plotbot_queue_state(user_session_id)
        st.warning("Queued Plotbot job state was incomplete. Please try again.")
        return

    job_row = registry_service.get_job_by_id(control_plane_job_id)
    if job_row is None:
        _clear_plotbot_queue_state(user_session_id)
        st.warning("Queued Plotbot job could not be found. Please try again.")
        return

    if job_row.status == "completed" and job_row.artifact_id is not None:
        if _append_queued_plotbot_result(user_session_id, job_row.artifact_id):
            _clear_plotbot_queue_state(user_session_id)
            st.rerun()
        _clear_plotbot_queue_state(user_session_id)
        st.error("Completed Plotbot result could not be attached. Please retry.")
        return

    if job_row.status == "failed":
        _clear_plotbot_queue_state(user_session_id)
        st.error(f"Plotbot generation failed: {job_row.failure_reason}")
        return

    rq_status = "unknown"
    if rq_job_id:
        try:
            rq_job = get_queue().fetch_job(rq_job_id)
            if rq_job is not None:
                rq_status_raw = rq_job.get_status(refresh=True)
                rq_status = str(
                    getattr(rq_status_raw, "value", rq_status_raw)
                ).strip().lower()
        except Exception:
            rq_status = "unavailable"

    st.status(
        f"Generating plot in the background... ({job_row.status}, queue: {rq_status})",
        state="running",
    )


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
            elif message["type"] == "plot_svg":
                try:
                    svg_value = message["value"]
                    st.image(svg_value)
                    b64 = base64.b64encode(svg_value.encode("utf-8")).decode()
                    href = (f'<a href="data:image/svg+xml;base64,{b64}" '
                            'download="plot.svg">Download SVG</a>')
                    st.markdown(href, unsafe_allow_html=True)
                except Exception as e:
                    st.error(
                        f"Failed to render plot: {str(e)}",
                        icon=":material/error:"
                    )
            elif message["type"] == "plot":
                if plot_lib == "plotly.express":
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

    if _get_plotbot_queue_state(user_session_id) is not None:
        _render_plotbot_queue_status(user_session_id)
        return

    # Get last code chunk
    last_code = previous_code_chunk(st.session_state[user_session_id]["plotbot"])
    queue_enabled = get_redis_queue_config().enabled
    try:
        user_email = (st.user.email if hasattr(st, 'user') and st.user and
                      hasattr(st.user, 'email') else 'anonymous')
    except Exception:
        user_email = 'anonymous'

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

                if queue_enabled:
                    _submit_plotbot_job(
                        user_session_id=user_session_id,
                        df=df_pandas,
                        plot_lib=plot_lib,
                        user_input=input_prompt,
                        api_key=api_key,
                        llm_params=LLM_PARAMS,
                        code_chunk=last_code,
                        user_email=user_email,
                    )
                else:
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

                if queue_enabled:
                    _submit_plotbot_job(
                        user_session_id=user_session_id,
                        df=df_pandas,
                        plot_lib=plot_lib,
                        user_input=input_refine,
                        api_key=api_key,
                        llm_params=LLM_PARAMS,
                        code_chunk=last_code,
                        user_email=user_email,
                    )
                else:
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

        local_provider_available = _has_local_plotbot_provider()
        api_key = "" if local_provider_available else get_api_key(
            user_session_id, DESKTOP, CACHE, QUOTA
        )
        model_available = local_provider_available or bool(api_key)

        # Check if we should show API key input based on quota and current key status
        has_user_key = (
            not local_provider_available and
            api_key is not None and
            st.session_state[user_session_id].get(SessionKeys.AI_USER_KEY) is not None
        )

        # Render quota tracker only for hosted/community API-key mode.
        if not local_provider_available and not has_user_key:
            render_quota_tracker(user_email)

        # Check if we should show work preservation interface first
        show_work_preservation = (
            not local_provider_available and
            should_show_work_preservation_interface(
                user_email, user_session_id, has_user_key, "plotbot"
            )
        )

        # Only show API key input if work preservation is not needed
        show_api_input = (
            not local_provider_available and
            should_show_api_key_input(user_email, has_user_key) and
            not model_available and
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
        elif show_api_input and not safe_session_get(
            session, SessionKeys.TAGS_TABLE, False
        ):
            # Show API key input when no work preservation needed
            render_api_key_input(user_session_id)
        else:
            if show_api_input:
                render_api_key_input(user_session_id)

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
                metadata_target = load_metadata(CorpusKeys.TARGET, user_session_id)

            metadata_reference = None
            if safe_session_get(session, SessionKeys.HAS_REFERENCE, False):
                metadata_reference = load_metadata(CorpusKeys.REFERENCE, user_session_id)

            # Initialize widget state management
            app_core.widget_manager.register_persistent_keys([
                'plot_corpus_select', 'plot_query_select', 'plot_type_select',
                'plot_x_axis', 'plot_y_axis', 'plot_color_by'
            ])
            # Get last code chunk
            last_code = previous_code_chunk(st.session_state[user_session_id]["plotbot"])

            # Chat input
            if last_code is None or len(last_code) == 0:

                # Data selection interface
                selected_query, df = render_data_selection_interface(
                    user_session_id=user_session_id,
                    session=session,
                    bot_prefix="plotbot",
                    clear_function=clear_plotbot_table,
                    metadata_target=metadata_target,
                    metadata_reference=metadata_reference
                )

                # Data preview with controls
                if df is not None:
                    df = render_data_preview_controls(
                        df=df,
                        query=selected_query,
                        user_session_id=user_session_id
                    )

                    st.session_state[user_session_id]['plotbot_df'] = df
                    st.session_state[user_session_id]['plotbot_query'] = selected_query
                else:
                    st.session_state[user_session_id].pop('plotbot_df', None)
                    st.session_state[user_session_id].pop('plotbot_query', None)

                st.session_state[user_session_id]['plotbot_library'] = PLOTBOT_LIBRARY

            # Chat interface
            df = st.session_state[user_session_id].get('plotbot_df', None)
            selected_query = st.session_state[user_session_id].get('plotbot_query', None)
            plot_lib = PLOTBOT_LIBRARY

            if last_code is not None and len(last_code) > 0:
                st.markdown(
                    body="### Data Preview",
                    help=(
                        "Here is a preview of the data used for the current plot. "
                        "To modidify the data, please Clear Chat History "
                        "and start a new conversation."
                    )
                )
                df = render_data_preview_controls(
                    df=df,
                    query=selected_query,
                    user_session_id=user_session_id
                )
                st.info(
                    "Use the chat input below to refine the previous plot "
                    "in the message thread. For example, you can instuct the Plotbot to "
                    "change colors, add titles, or modify axes.",
                    icon=":material/info:"
                )

            if model_available:
                render_plotbot_chat_interface(
                    user_session_id, api_key or "", df, selected_query, plot_lib
                )
            elif selected_query:
                st.info(
                    "Enter an OpenAI API key or configure a local model provider to generate plots.",  # noqa: E501
                    icon=":material/info:"
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
