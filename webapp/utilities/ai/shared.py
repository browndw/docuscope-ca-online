"""
Shared AI utilities for corpus analysis bots.

This module contains common functionality used by both plotbot and pandabot,
avoiding circular imports by separating shared utilities from bot-specific implementations.
"""

import io
import re
import json
from typing import List, Dict, Any
from datetime import datetime, timezone, timedelta

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st

from webapp.config.unified import get_config
from webapp.utilities.storage.cache_management import get_query_count

# Import centralized logging configuration and logger
from webapp.utilities.configuration.logging_config import get_logger

from webapp.utilities.state import SessionKeys

logger = get_logger()

# LLM Configuration
LLM_MODEL = "gpt-4o-mini"
LLM_PARAMS = {
    "model": LLM_MODEL,
    "temperature": 0.1,
    "max_tokens": 3000
}

PLOT_INTENT_PATTERN = re.compile(
    r"\b("
    r"plot(s)?|chart(s)?|graph(s)?|draw|visualize|sketch|illustrate|render|depict|map|trace|diagram(s)?|"  # noqa: E501
    r"scatter(plot)?s?|bar(plot)?s?|hist(ogram)?s?|hist(s)?|pie(chart)?s?|pie(s)?|line(plot)?s?|line(s)?|"  # noqa: E501
    r"area(s)?|heatmap(s)?|box(plot)?s?|box(es)?|violin(plot)?s?|violin(s)?|bubble(chart)?s?|bubble(s)?|"  # noqa: E501
    r"density(plot)?s?|density(s)?|hexbin(s)?|error(bar)?s?|error(s)?|stacked|polar|donut(chart)?s?|donut(s)?|"  # noqa: E501
    r"funnel(s)?|distribution(s)?|dist(plot)?s?|point(s)?|joint(plot)?s?|pair(plot)?s?|categorical|swarm(plot)?s?|"  # noqa: E501
    r"fit|reg(plot)?s?|lm(plot)?s?|kde(plot)?s?|boxen(plot)?s?|strip(plot)?s?|count(plot)?s?|"  # noqa: E501
    r"treemap(s)?|sunburst(s)?|waterfall(s)?|step(plot)?s?|ribbon(s)?|contour(f)?s?|contour(s)?|"  # noqa: E501
    r"mosaic(s)?|matrix|matrices|ridge(s)?|ridgeline(s)?|par(coord)?s?|parallel(s)?|dendrogram(s)?|"  # noqa: E501
    r"network(s)?|chord(s)?|sankey(s)?|facet(s)?|subplot(s)?|axes|axis|x-?axis|y-?axis|z-?axis|"   # noqa: E501
    r"color|hue|size|shape|label(s)?|legend(s)?|title(s)?|grid(s)?|background|foreground|font(s)?|"  # noqa: E501
    r"scale(s)?|range(s)?|tick(s)?|mark(s)?|spine(s)?|border(s)?|strip(s)?|dot(plot)?s?|dot(s)?"  # noqa: E501
    r")\b",
    re.IGNORECASE
)


def export_conversation_history(
    user_session_id: str,
    bot_type: str = "plotbot"
) -> str:
    """
    Export comprehensive workflow history including conversation and plots.

    Parameters
    ----------
    user_session_id : str
        User session identifier
    bot_type : str, default "plotbot"
        Type of bot ("plotbot" or "pandabot")

    Returns
    -------
    str
        JSON formatted workflow export with embedded plots
    """
    try:
        # Get messages from session state based on bot type
        if bot_type == "plotbot":
            messages_key = SessionKeys.AI_PLOTBOT_CHAT
            # Also get plotbot-specific messages from the plotbot list
            plotbot_messages = st.session_state[user_session_id].get(
                SessionKeys.AI_PLOTBOT_CHAT, []
            )
        elif bot_type == "pandabot":
            messages_key = SessionKeys.AI_PANDABOT_CHAT
            plotbot_messages = []
        else:
            messages_key = f"{bot_type}_messages"
            plotbot_messages = []

        chat_messages = st.session_state[user_session_id].get(messages_key, [])

        # Combine and organize all workflow data
        workflow_data = {
            "workflow_type": f"{bot_type}_analysis",
            "export_date": datetime.now(timezone.utc).isoformat(),
            "user_session_id": user_session_id,
            "summary": {
                "total_interactions": 0,
                "plots_generated": 0,
                "analysis_steps": 0
            },
            "workflow_steps": []
        }

        # Process chat messages to create a user-friendly workflow
        current_step = 1

        # Extract user queries and assistant responses
        for i, msg in enumerate(chat_messages):
            if msg.get("role") == "user":
                step_data = {
                    "step": current_step,
                    "type": "user_query",
                    "timestamp": msg.get("timestamp", ""),
                    "user_prompt": msg.get("value", ""),  # Fixed: use "value" not "content"
                    "assistant_response": None,
                    "code_generated": None,
                    "plot_svg": None
                }

                # Look for assistant response and any plots
                if i + 1 < len(chat_messages):
                    next_msg = chat_messages[i + 1]
                    if next_msg.get("role") == "assistant":
                        if next_msg.get("type") == "text":
                            step_data["assistant_response"] = next_msg.get("value", "")
                        elif next_msg.get("type") == "code":
                            step_data["code_generated"] = next_msg.get("value", "")

                # Check for associated plots in plotbot messages
                for plot_msg in plotbot_messages:
                    if plot_msg.get("type") == "plot":
                        # Convert plot to SVG if available
                        try:
                            plot_obj = plot_msg.get("value")
                            if plot_obj:
                                svg_str = fig_to_svg(plot_obj, bot_type)
                                if svg_str:
                                    step_data["plot_svg"] = svg_str
                                    workflow_data["summary"]["plots_generated"] += 1
                                    break
                        except Exception:
                            pass

                workflow_data["workflow_steps"].append(step_data)
                current_step += 1
                workflow_data["summary"]["total_interactions"] += 1

                if step_data["code_generated"] or step_data["assistant_response"]:
                    workflow_data["summary"]["analysis_steps"] += 1

        # If no meaningful workflow found, return a helpful message
        if not workflow_data["workflow_steps"]:
            return json.dumps({
                "workflow_type": f"{bot_type}_analysis",
                "export_date": datetime.now(timezone.utc).isoformat(),
                "message": ("No conversation history found. Start a conversation "
                            "with the AI assistant to generate a workflow to export."),
                "suggestion": ("Try asking the assistant to create a plot or "
                               "analyze your data, then export the workflow.")
            }, indent=2)

        # Add current plot if available
        current_plot_svg = get_current_plot_as_svg(user_session_id, bot_type)
        has_embedded_plots = any(
            step.get("plot_svg") for step in workflow_data["workflow_steps"]
        )
        if current_plot_svg and not has_embedded_plots:
            workflow_data["current_plot"] = {
                "description": "Most recent plot generated",
                "svg_data": current_plot_svg
            }
            workflow_data["summary"]["plots_generated"] += 1

        return json.dumps(workflow_data, indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "error": f"Failed to export workflow: {str(e)}",
            "export_date": datetime.now(timezone.utc).isoformat(),
            "suggestion": "Please try again."
        }, indent=2)


def get_current_plot_as_png(user_session_id: str, bot_type: str = "plotbot") -> bytes:
    """
    Get the current plot as PNG bytes if available.

    Parameters
    ----------
    user_session_id : str
        User session identifier
    bot_type : str, default "plotbot"
        Type of bot ("plotbot" or "pandabot")

    Returns
    -------
    bytes
        PNG bytes of current plot or empty bytes if none
    """
    try:
        # Import here to avoid circular imports
        from webapp.utilities.state import SessionKeys

        if bot_type == "plotbot":
            # Get the current plot from plotbot session
            plotbot_messages = st.session_state[user_session_id].get(
                SessionKeys.AI_PLOTBOT_CHAT, []
            )
            for msg in reversed(plotbot_messages):  # Get most recent plot
                if msg.get("type") == "plot":
                    plot_obj = msg.get("value")
                    if plot_obj:
                        # Handle matplotlib/seaborn figures
                        if hasattr(plot_obj, 'savefig'):
                            buffer = io.BytesIO()
                            plot_obj.savefig(
                                buffer, format='png', bbox_inches='tight', dpi=150
                            )
                            buffer.seek(0)
                            png_bytes = buffer.getvalue()
                            buffer.close()
                            return png_bytes
                        # Handle plotly figures
                        elif hasattr(plot_obj, 'to_image'):
                            png_bytes = plot_obj.to_image(format="png", scale=2)
                            return png_bytes

            # Fallback: if plot objects aren't available but SVG is stored
            svg_str = st.session_state[user_session_id].get("plotbot_plot_svg", "")
            if svg_str:
                # Create a simple PNG with text indicating plot is available
                try:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.text(0.5, 0.6, 'Plot Available', ha='center', va='center',
                            fontsize=16, transform=ax.transAxes)
                    ax.text(0.5, 0.4, 'View in workflow export', ha='center', va='center',
                            fontsize=12, transform=ax.transAxes, style='italic')

                    # Add a simple border
                    rect = patches.Rectangle((0.1, 0.1), 0.8, 0.8, linewidth=2,
                                             edgecolor='gray', facecolor='lightgray',
                                             alpha=0.3, transform=ax.transAxes)
                    ax.add_patch(rect)

                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')

                    buffer = io.BytesIO()
                    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    buffer.seek(0)
                    png_bytes = buffer.getvalue()
                    buffer.close()
                    return png_bytes
                except Exception:
                    pass
        elif bot_type == "pandabot":
            # Check for pandabot plot storage
            img_key = f"pandabot_img_bytes_{user_session_id}"
            if img_key in st.session_state:
                plots = st.session_state[img_key]
                if plots:
                    # Get the most recent plot
                    latest_key = max(plots.keys()) if plots else None
                    if latest_key:
                        img_bytes = plots[latest_key]
                        if isinstance(img_bytes, bytes):
                            return img_bytes

        return b""

    except Exception:
        return b""


def get_current_plot_as_svg(user_session_id: str, bot_type: str = "plotbot") -> str:
    """
    Get the current plot as SVG string if available.

    Parameters
    ----------
    user_session_id : str
        User session identifier
    bot_type : str, default "plotbot"
        Type of bot ("plotbot" or "pandabot")

    Returns
    -------
    str
        SVG string of current plot or empty string if none
    """
    try:
        if bot_type == "plotbot":
            # Check for plotbot plot
            plot_key = "plotbot_plot_svg"
            return st.session_state[user_session_id].get(plot_key, "")
        elif bot_type == "pandabot":
            # Check for pandabot plot storage
            session_id = user_session_id
            img_key = f"pandabot_img_bytes_{session_id}"
            if img_key in st.session_state:
                plots = st.session_state[img_key]
                if plots:
                    # Get the most recent plot
                    latest_key = max(plots.keys()) if plots else None
                    if latest_key:
                        img_bytes = plots[latest_key]
                        if isinstance(img_bytes, bytes):
                            # Convert bytes to base64 for embedding in SVG
                            import base64
                            b64_img = base64.b64encode(img_bytes).decode()
                            return f'<image href="data:image/png;base64,{b64_img}"/>'

        return ""

    except Exception:
        return ""


def render_work_preservation_interface(
    user_session_id: str,
    user_email: str,
    bot_type: str = "plotbot"
) -> bool:
    """
    Render interface for users to save their work before quota exhaustion.

    Parameters
    ----------
    user_session_id : str
        User session identifier
    user_email : str
        User email for quota tracking
    bot_type : str, default "plotbot"
        Type of bot ("plotbot" or "pandabot")

    Returns
    -------
    bool
        True if user has acknowledged and wants to proceed to API key input
    """
    # Check if user has already acknowledged
    ack_key = f"quota_exhausted_acknowledged_{bot_type}"
    if st.session_state[user_session_id].get(ack_key, False):
        return True

    # Get quota info for display
    quota_info = get_quota_info(user_email)

    st.error(
        ":material/warning: **Community API Quota Exhausted**",
        icon=":material/error:"
    )

    st.markdown(
        f"""
        You've used **{quota_info['used']}/{quota_info['total']}** of your daily
        community API queries.

        **Before continuing with your own API key, you can save your work:**
        """
    )

    # Create columns for save options
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### :material/file_download: Export Analysis Workflow")

        # Get workflow history
        workflow_json = export_conversation_history(user_session_id, bot_type)

        if workflow_json:
            # Parse to check if meaningful content exists
            try:
                data = json.loads(workflow_json)
                if "workflow_steps" in data and data["workflow_steps"]:
                    step_count = len(data["workflow_steps"])
                    plot_count = data.get("summary", {}).get("plots_generated", 0)
                    st.write(f"**{step_count}** analysis steps")
                    if plot_count > 0:
                        st.write(f"**{plot_count}** plots included")
                else:
                    st.write("**Complete workflow** with embedded plots")
            except Exception:
                st.write("**Analysis workflow** available")

            # Download button
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            filename = f"{bot_type}_workflow_{timestamp}.json"
            st.download_button(
                label="Download Analysis Workflow",
                data=workflow_json,
                file_name=filename,
                mime="application/json",
                help="Download your complete analysis workflow with embedded plots as JSON"
            )
        else:
            st.info("No workflow to export")

    with col2:
        st.markdown("### :material/image: Export Current Plot")

        # Get current plot as PNG
        plot_png = get_current_plot_as_png(user_session_id, bot_type)

        if plot_png:
            st.write("**Current plot available**")

            # Show plot preview (small)
            with st.expander("Preview Current Plot"):
                st.image(plot_png, width=300)

            # Download button for plot
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            filename = f"{bot_type}_plot_{timestamp}.png"
            st.download_button(
                label="Download Current Plot (PNG)",
                data=plot_png,
                file_name=filename,
                mime="image/png",
                help="Download your current plot as a high-quality PNG file"
            )
        else:
            st.info("No plot to export")

    st.markdown("---")

    # Acknowledgment and proceed
    st.markdown("### :material/key: Continue with Your API Key")
    st.markdown(
        """
        To continue using the AI assistant, you'll need to provide your own OpenAI API key.

        :material/info: **Your API key will be:**
        - Used only for your current session
        - Not stored permanently
        - Cleared when you close the app
        """
    )

    # Checkbox for acknowledgment
    acknowledged = st.checkbox(
        "I understand that I need to provide my own API key to continue, "
        "and I have saved any work I want to keep.",
        key=f"ack_checkbox_{bot_type}"
    )

    # Proceed button
    if acknowledged:
        if st.button(
            "Proceed to API Key Input",
            type="primary",
            icon=":material/arrow_forward:"
        ):
            # Mark as acknowledged
            st.session_state[user_session_id][ack_key] = True
            st.rerun()
    else:
        st.info("Please check the box above to proceed")

    return False


def should_show_work_preservation_interface(
    user_id: str,
    user_session_id: str,
    has_user_key: bool = False,
    bot_type: str = "plotbot"
) -> bool:
    """
    Determine if the work preservation interface should be shown.

    This should be shown when:
    1. Quota is exhausted
    2. User doesn't have their own API key
    3. User hasn't already acknowledged the quota exhaustion
    4. There's work to potentially save (conversation or plots)

    Parameters
    ----------
    user_id : str
        The user ID to check quota for
    user_session_id : str
        User session identifier
    has_user_key : bool
        Whether the user has already provided their own API key
    bot_type : str, default "plotbot"
        Type of bot ("plotbot" or "pandabot")

    Returns
    -------
    bool
        True if work preservation interface should be shown
    """
    # If user already has their own key, no need for preservation interface
    if has_user_key:
        return False

    # In desktop mode, don't show preservation interface
    if get_config('desktop_mode', 'global'):
        return False

    # Check if user has already acknowledged quota exhaustion
    ack_key = f"quota_exhausted_acknowledged_{bot_type}"
    if st.session_state[user_session_id].get(ack_key, False):
        return False

    # Check if quota is exhausted
    try:
        if "openai" not in st.secrets or "api_key" not in st.secrets["openai"]:
            return False  # No community key available anyway
    except Exception:
        return False

    quota_info = get_quota_info(user_id)
    if quota_info['remaining'] > 0:
        return False  # Quota not exhausted

    # Check if there's work to save
    try:
        # Import here to avoid circular imports
        from webapp.utilities.state import SessionKeys

        if bot_type == "plotbot":
            messages_key = SessionKeys.AI_PLOTBOT_CHAT
        elif bot_type == "pandabot":
            messages_key = SessionKeys.AI_PANDABOT_CHAT
        else:
            messages_key = f"{bot_type}_messages"

        messages = st.session_state[user_session_id].get(messages_key, [])
        has_conversation = len(messages) > 0
    except Exception:
        has_conversation = False

    # Check for plots
    has_plot = False
    if bot_type == "plotbot":
        plot_key = "plotbot_plot_svg"
        has_plot = bool(st.session_state[user_session_id].get(plot_key, ""))
    elif bot_type == "pandabot":
        img_key = f"pandabot_img_bytes_{user_session_id}"
        has_plot = bool(st.session_state.get(img_key, {}))

    # Show preservation interface if there's work to save
    return has_conversation or has_plot


def detect_intent(user_input: str) -> str:
    """
    Detects if the user's input is a plotting request.

    **Note**: This function is designed for PRE-FILTERING requests to protect API usage,
    particularly in plotbot where non-plotting requests should be rejected. It should
    NOT be used to determine how to handle responses from AI systems that can return
    structured output indicating their response type.

    Returns
    -------
    str
        "plot" if plotting intent is detected,
        "chat" if not plotting-related,
        "none" if input is empty or invalid.
    """
    if not isinstance(user_input, str) or not user_input.strip():
        return "none"

    if PLOT_INTENT_PATTERN.search(user_input):
        return "plot"
    return "chat"


def fig_to_svg(
        figure, plot_lib: str = "matplotlib",
        width: int = 6,
        height: int = 4
) -> str:
    """
    Convert a matplotlib or plotly figure to SVG string.

    Parameters
    ----------
    figure : matplotlib.figure.Figure or plotly figure
        The figure to convert
    plot_lib : str, default "matplotlib"
        The plotting library used ("matplotlib", "seaborn", "plotly.express")
    width : int, default 6
        Figure width in inches (for matplotlib)
    height : int, default 4
        Figure height in inches (for matplotlib)

    Returns
    -------
    str
        SVG representation of the figure
    """
    try:
        # Auto-detect figure type for better reliability
        figure_type = str(type(figure))

        if "matplotlib" in figure_type.lower() or hasattr(figure, 'savefig'):
            # This is a matplotlib figure
            figure.set_size_inches(width, height)
            figure.patch.set_facecolor('white')

            buffer = io.StringIO()
            figure.savefig(buffer, format='svg', bbox_inches='tight')
            buffer.seek(0)
            svg_string = buffer.getvalue()
            buffer.close()

            # Close the Matplotlib figure to free memory
            plt.close(figure)
            return svg_string

        elif "plotly" in figure_type.lower() or hasattr(figure, 'update_layout'):
            # This is a plotly figure
            figure.update_layout(template="plotly_white")
            img_bytes = figure.to_image(format="svg")
            svg_string = img_bytes.decode('utf-8')
            return svg_string

        else:
            # Fallback: try matplotlib approach first
            logger.warning(f"Unknown figure type: {figure_type}, trying matplotlib")
            if hasattr(figure, 'savefig'):
                figure.set_size_inches(width, height)
                figure.patch.set_facecolor('white')

                buffer = io.StringIO()
                figure.savefig(buffer, format='svg', bbox_inches='tight')
                buffer.seek(0)
                svg_string = buffer.getvalue()
                buffer.close()

                plt.close(figure)
                return svg_string
            else:
                return ""

    except Exception:
        # Try to close matplotlib figures if possible
        try:
            if hasattr(figure, 'savefig'):
                plt.close(figure)
        except Exception:
            pass
        return ""


def prune_message_thread(
        session_id: str = None,
        thread_key: str = None,
        max_length: int = 20,
        messages: List[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Prune message thread to stay within limits.

    Can work with either Streamlit session state or a direct message list.

    Parameters
    ----------
    session_id : str, optional
        Session identifier for Streamlit session state
    thread_key : str, optional
        Key for the thread in session state
    max_length : int, default 20
        Maximum number of messages to keep
    messages : List[Dict[str, Any]], optional
        Direct message list to prune

    Returns
    -------
    List[Dict[str, Any]]
        Pruned message list
    """
    # Handle direct message list (for compatibility with original interface)
    if messages is not None:
        if not messages or len(messages) <= max_length:
            return messages

        # Simple pruning strategy: keep system messages + recent messages
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        other_messages = [msg for msg in messages if msg.get("role") != "system"]

        if len(other_messages) > max_length - len(system_messages):
            other_messages = other_messages[-(max_length - len(system_messages)):]

        return system_messages + other_messages

    # Handle Streamlit session state (original interface)
    if session_id is None or thread_key is None:
        raise ValueError(
            "Either provide messages directly or both session_id and thread_key"
        )

    try:
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        # Initialize session state if it doesn't exist
        if session_id not in st.session_state:
            st.session_state[session_id] = {}

        if thread_key not in st.session_state[session_id]:
            st.session_state[session_id][thread_key] = []
            return []

        thread = st.session_state[session_id][thread_key]
        if len(thread) <= max_length:
            return thread

        # Keep the first user message and the last (max_length-1) messages
        first_user_idx = next(
            (i for i, m in enumerate(thread) if m.get("role") == "user"), 0
        )
        pruned = (
            [thread[first_user_idx]] + thread[-(max_length-1):]
            if first_user_idx < len(thread) else thread[-max_length:]
        )

        # Update session state
        st.session_state[session_id][thread_key] = pruned
        return pruned

    except ImportError:
        # Fallback if streamlit not available
        logger.warning("Streamlit not available for session state pruning")
        return []


def validate_api_key(api_key: str) -> bool:
    """
    Validate OpenAI API key.

    Parameters
    ----------
    api_key : str
        The API key to validate

    Returns
    -------
    bool
        True if valid, False otherwise
    """
    if not api_key or not api_key.strip():
        return False

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        # Simple test call
        client.models.list()
        return True
    except Exception:
        return False


def get_quota_info(user_id: str, force_refresh: bool = False) -> dict:
    """
    Get quota information for a user with session-based counting optimization.

    Parameters
    ----------
    user_id : str
        The user ID to check quota for
    force_refresh : bool, default False
        Force a fresh Firestore query instead of using session-based counting

    Returns
    -------
    dict
        Dictionary containing quota information with keys:
        - 'total': Total quota allowed
        - 'used': Number of queries used in last 24 hours
        - 'remaining': Number of queries remaining
        - 'percentage_used': Percentage of quota used (0-100)
    """
    try:
        # Import here to avoid circular imports
        try:
            from webapp.config.config_utils import get_runtime_setting
            total_quota = get_runtime_setting('quota', 100, 'llm')
        except ImportError:
            # Fallback to static config if runtime config not available
            total_quota = get_config('quota', 'llm', 100)

        # Only check usage in online mode
        if get_config('desktop_mode', 'global'):
            # In desktop mode, no quota limits
            return {
                'total': total_quota,
                'used': 0,
                'remaining': total_quota,
                'percentage_used': 0
            }

        # Use session-based quota counting to minimize Firestore calls
        session_quota_key = f"quota_base_{user_id}"
        session_count_key = f"quota_session_count_{user_id}"
        session_time_key = f"quota_base_time_{user_id}"

        current_time = datetime.now(timezone.utc)

        # Check if we need to refresh base count from Firestore
        need_refresh = (
            force_refresh or
            session_quota_key not in st.session_state or
            session_time_key not in st.session_state or
            (current_time - st.session_state[session_time_key]) > timedelta(hours=1)
        )

        if need_refresh:
            # Get fresh count from Firestore
            base_count = get_query_count(user_id)
            st.session_state[session_quota_key] = base_count
            st.session_state[session_count_key] = 0  # Reset session counter
            st.session_state[session_time_key] = current_time
            used_queries = base_count
        else:
            # Use cached base count + session increments
            base_count = st.session_state.get(session_quota_key, 0)
            session_increments = st.session_state.get(session_count_key, 0)
            used_queries = base_count + session_increments

        remaining_queries = max(0, total_quota - used_queries)
        percentage_used = (used_queries / total_quota * 100) if total_quota > 0 else 100

        return {
            'total': total_quota,
            'used': used_queries,
            'remaining': remaining_queries,
            'percentage_used': min(100, percentage_used)  # Cap at 100%
        }
    except Exception:
        # Return safe defaults
        try:
            from webapp.config.config_utils import get_runtime_setting
            quota = get_runtime_setting('quota', 100, 'llm')
        except ImportError:
            quota = get_config('quota', 'llm', 100)
        return {
            'total': quota,
            'used': 0,
            'remaining': quota,
            'percentage_used': 0
        }


def increment_session_quota(user_id: str) -> None:
    """
    Increment the session-based quota counter when a user makes an API call.

    This should be called whenever a user makes an actual API query to the LLM,
    allowing us to track quota usage without repeatedly querying Firestore.

    Parameters
    ----------
    user_id : str
        The user ID to increment quota for
    """
    try:
        # Only track in online mode
        if get_config('desktop_mode', 'global'):
            return

        session_count_key = f"quota_session_count_{user_id}"

        # Initialize if not exists
        if session_count_key not in st.session_state:
            st.session_state[session_count_key] = 0

        # Increment the session counter
        st.session_state[session_count_key] += 1

        # Clear display cache to force refresh on next render
        quota_cache_key = f"quota_display_{user_id}"
        if quota_cache_key in st.session_state:
            del st.session_state[quota_cache_key]
        if f"{quota_cache_key}_time" in st.session_state:
            del st.session_state[f"{quota_cache_key}_time"]

    except Exception as e:
        logger.warning(f"Failed to increment session quota: {e}")


def render_quota_tracker(user_id: str) -> dict:
    """
    Render a quota tracking component in the sidebar with optimized caching.

    Parameters
    ----------
    user_id : str
        The user ID to track quota for

    Returns
    -------
    dict
        Quota information dictionary
    """
    # Use extended caching to reduce display refresh frequency
    try:
        # Check if we have recent quota info cached for display
        quota_cache_key = f"quota_display_{user_id}"
        current_time = datetime.now(timezone.utc)

        # Use 5-minute cache for display (quota only changes when user makes calls)
        cache_duration = timedelta(minutes=5)

        if (quota_cache_key in st.session_state and
                hasattr(st.session_state, quota_cache_key) and
                current_time - st.session_state.get(f"{quota_cache_key}_time",
                                                    datetime.min) < cache_duration):
            # Use cached quota info for display
            quota_info = st.session_state[quota_cache_key]
        else:
            # Get quota info (which uses its own session-based optimization)
            quota_info = get_quota_info(user_id)
            st.session_state[quota_cache_key] = quota_info
            st.session_state[f"{quota_cache_key}_time"] = current_time
    except Exception:
        # Fallback to fresh call if caching fails
        quota_info = get_quota_info(user_id)

    # Only show quota tracker in online mode when secrets are available
    if not get_config('desktop_mode', 'global'):
        try:
            import streamlit as st
            # Check if we have access to community key (secrets)
            if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
                with st.sidebar:
                    st.markdown("### Community API Usage")

                    # Progress bar
                    st.progress(
                        quota_info['percentage_used'] / 100,
                        text=f"{quota_info['used']}/{quota_info['total']} queries used"
                    )

                    # Remaining queries info
                    if quota_info['remaining'] > 0:
                        st.info(
                            f":material/info: {quota_info['remaining']} queries remaining "
                            "in 24h window",
                            icon=":material/query_stats:"
                        )
                    else:
                        st.error(
                            ":material/warning: Community quota exhausted",
                            icon=":material/error:"
                        )

                    # Warnings based on usage
                    if quota_info['percentage_used'] >= 95 and quota_info['remaining'] > 0:
                        if quota_info['remaining'] == 1:
                            st.warning(
                                ":material/warning: **Last query available!** "
                                "After this query, you'll need to provide your own "
                                "OpenAI API key to continue.",
                                icon=":material/priority_high:"
                            )
                        else:
                            st.warning(
                                f":material/warning: **Critical:** Only "
                                f"{quota_info['remaining']} queries left! Consider "
                                "using your own OpenAI API key.",
                                icon=":material/priority_high:"
                            )
                    elif quota_info['percentage_used'] >= 90:
                        st.warning(
                            f":material/info: **Notice:** {quota_info['remaining']} "
                            "queries remaining. You may want to provide your own "
                            "OpenAI API key soon.",
                            icon=":material/info:"
                        )
        except Exception as e:
            # If we can't access secrets or any other error, don't show quota tracker
            logger.error(f"Cannot render quota tracker: {e}")

    return quota_info


def should_show_api_key_input(user_id: str, has_user_key: bool = False) -> bool:
    """
    Determine if the API key input should be shown.

    Parameters
    ----------
    user_id : str
        The user ID to check quota for
    has_user_key : bool
        Whether the user has already provided their own API key

    Returns
    -------
    bool
        True if API key input should be shown
    """
    # In desktop mode, always require user to provide their own API key
    if get_config('desktop_mode', 'global'):
        return not has_user_key

    # In online mode, check if secrets are available for community key access
    try:
        if "openai" not in st.secrets or "api_key" not in st.secrets["openai"]:
            # No community key available, user must provide their own
            return not has_user_key
    except Exception:
        # Can't access secrets, user must provide their own key
        return not has_user_key

    # Online mode with community key available - check quota
    quota_info = get_quota_info(user_id)
    return quota_info['remaining'] <= 0 and not has_user_key
