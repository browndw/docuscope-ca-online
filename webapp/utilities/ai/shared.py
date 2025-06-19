"""
Shared AI utilities for corpus analysis bots.

This module contains common functionality used by both plotbot and pandabot,
avoiding circular imports by separating shared utilities from bot-specific implementations.
"""

import io
import pathlib
from typing import List, Dict, Any

import matplotlib.pyplot as plt
from loguru import logger


# LLM Configuration
LLM_MODEL = "gpt-4o-mini"
LLM_PARAMS = {
    "model": LLM_MODEL,
    "temperature": 0.1,
    "max_tokens": 3000
}


def detect_intent(user_input: str) -> str:
    """
    Detect if user input is asking for a plot or analysis.

    Parameters
    ----------
    user_input : str
        User's input text

    Returns
    -------
    str
        "plot" if plotting intent detected, "analysis" otherwise
    """
    plot_keywords = [
        "plot", "chart", "graph", "visualize", "show", "display",
        "histogram", "scatter", "bar", "line", "pie", "boxplot",
        "heatmap", "distribution", "correlation"
    ]

    user_lower = user_input.lower()
    for keyword in plot_keywords:
        if keyword in user_lower:
            return "plot"
    
    return "analysis"


def fig_to_svg(figure, plot_lib: str = "matplotlib", width: int = 6, height: int = 4) -> str:
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
        if plot_lib in ["matplotlib", "seaborn"]:
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
        elif plot_lib == "plotly.express":
            figure.update_layout(template="plotly_white")
            img_bytes = figure.to_image(format="svg")
            svg_string = img_bytes.decode('utf-8')
            return svg_string
        else:
            logger.error(f"Unsupported plot library: {plot_lib}")
            return ""
    except Exception as e:
        logger.error(f"Error converting figure to SVG: {e}")
        plt.close(figure) if plot_lib in ["matplotlib", "seaborn"] else None
        return ""


def prune_message_thread(session_id: str = None, thread_key: str = None, max_length: int = 20, messages: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
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
        raise ValueError("Either provide messages directly or both session_id and thread_key")
    
    try:
        import streamlit as st
        
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        if thread_key not in st.session_state[session_id]:
            return []

        thread = st.session_state[session_id][thread_key]
        if len(thread) <= max_length:
            return thread

        # Keep the first user message and the last (max_length-1) messages
        first_user_idx = next((i for i, m in enumerate(thread) if m.get("role") == "user"), 0)
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
    except Exception as e:
        logger.error(f"OpenAI API key validation failed: {e}")
        return False


def setup_ai_logging() -> None:
    """Set up logging for AI modules."""
    try:
        from webapp.utilities.configuration import setup_ai_logging
        setup_ai_logging()
    except ImportError:
        # Fallback to manual setup if centralized logging not available
        log_dir = pathlib.Path(__file__).parent.parents[1] / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Set up basic logging for AI modules
        logger.add(
            log_dir / "ai_error.log",
            rotation="10 MB",
            retention="10 days",
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
        )
