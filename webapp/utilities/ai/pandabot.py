"""
Pandabot AI assistant using PandasAI for analysis and plotting.

This module provides pandabot-specific functionality for AI-assisted data analysis
and plotting using PandasAI. Pandabot is a conversational assistant that can handle
both analytical queries and plot generation with thread-safe plot capture.
"""
import os
import builtins
import io
import threading
import time
import weakref
from contextlib import contextmanager

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pandasai.exceptions import MaliciousQueryError, NoResultFoundError
from pandasai_openai import OpenAI
import pandasai as pai

# Import shared AI utilities
from webapp.utilities.ai.shared import prune_message_thread
# Add async storage import for non-blocking Firestore operations
from webapp.utilities.storage import conditional_async_add_message
from webapp.utilities.ai.shared import increment_session_quota
from webapp.utilities.ai.enterprise_integration import (
    enterprise_ai_call,
    determine_api_key_type
)
from webapp.utilities.state import SessionKeys
from webapp.utilities.state.widget_state import safe_clear_widget_state
from webapp.utilities.storage.backend_factory import get_session_backend
from webapp.config.unified import get_ai_config
from webapp.utilities.core import app_core

# Get AI configuration using standardized access
AI_CONFIG = get_ai_config()
DESKTOP = AI_CONFIG['desktop_mode']

# Thread-safe global lock for monkeypatching
_monkeypatch_lock = threading.RLock()

# Weak reference set to track active sessions for cleanup
_active_sessions = weakref.WeakSet()


class SessionPlotStorage:
    """Thread-safe session-specific plot storage with automatic cleanup."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.session_img_key = SessionKeys.get_pandabot_img_key(session_id)
        self.last_access = time.time()

        # Initialize session storage if needed
        if self.session_img_key not in st.session_state:
            st.session_state[self.session_img_key] = {}

        # Track this session for cleanup
        _active_sessions.add(self)

    @property
    def storage(self):
        """Get the session-specific storage dict."""
        self.last_access = time.time()
        return st.session_state[self.session_img_key]

    def cleanup_old_sessions(self, max_age_seconds=3600):
        """Clean up old session data to prevent memory leaks."""
        current_time = time.time()
        keys_to_remove = []

        for key in st.session_state:
            if key.startswith("pandabot_img_bytes_"):
                # Extract session from key and check if it's old
                try:
                    # Simple heuristic: if no recent access, clean up
                    if current_time - self.last_access > max_age_seconds:
                        keys_to_remove.append(key)
                except Exception:
                    # If we can't determine age, err on side of keeping it
                    pass

        for key in keys_to_remove:
            safe_clear_widget_state(key)


@contextmanager
def thread_safe_monkeypatch(session_storage: SessionPlotStorage):
    """
    Thread-safe context manager for monkeypatching matplotlib and file operations.

    Uses a global lock to ensure only one thread can modify the global state at a time,
    while using session-specific storage for captured images.
    """
    with _monkeypatch_lock:
        # Store original functions
        _original_savefig = plt.Figure.savefig
        _original_plt_savefig = plt.savefig
        _original_exists = os.path.exists
        _original_isfile = os.path.isfile
        _original_open = open

        def fake_open(file, mode='r', *args, **kwargs):
            if (isinstance(file, str) and "temp_chart" in file and
                    ('w' in mode or 'a' in mode)):
                class FakeFile:
                    def __init__(self):
                        self.data = io.BytesIO()

                    def write(self, data):
                        if isinstance(data, bytes):
                            self.data.write(data)
                        else:
                            self.data.write(data.encode())

                    def close(self):
                        self.data.seek(0)
                        storage = session_storage.storage
                        storage["img"] = self.data.getvalue()
                        storage["path"] = file
                        storage["session_id"] = session_storage.session_id

                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        self.close()

                return FakeFile()
            else:
                return _original_open(file, mode, *args, **kwargs)

        def fake_exists(path):
            if isinstance(path, str) and "temp_chart" in path:
                return True
            return _original_exists(path)

        def fake_isfile(path):
            if isinstance(path, str) and "temp_chart" in path:
                return True
            return _original_isfile(path)

        def plt_savefig_to_buffer(fname, *args, **kwargs):
            if isinstance(fname, str) and "temp_chart" in fname:
                buf = io.BytesIO()
                _original_plt_savefig(buf, format="png", *args, **kwargs)
                buf.seek(0)
                storage = session_storage.storage
                storage["img"] = buf.getvalue()
                storage["path"] = fname
                storage["session_id"] = session_storage.session_id
                buf.close()
            else:
                _original_plt_savefig(fname, *args, **kwargs)

        def savefig_to_buffer(self, fname, *args, **kwargs):
            if isinstance(fname, str) and "temp_chart" in fname:
                buf = io.BytesIO()
                _original_savefig(self, buf, format="png", *args, **kwargs)
                buf.seek(0)
                storage = session_storage.storage
                storage["img"] = buf.getvalue()
                storage["path"] = fname
                storage["session_id"] = session_storage.session_id
                buf.close()
            else:
                _original_savefig(self, fname, *args, **kwargs)

        # Apply monkeypatches
        plt.Figure.savefig = savefig_to_buffer
        plt.savefig = plt_savefig_to_buffer
        os.path.exists = fake_exists
        os.path.isfile = fake_isfile
        builtins.open = fake_open

        try:
            yield session_storage
        finally:
            # Always restore original functions, even if an exception occurs
            plt.Figure.savefig = _original_savefig
            plt.savefig = _original_plt_savefig
            os.path.exists = _original_exists
            os.path.isfile = _original_isfile
            builtins.open = _original_open


def clear_pandasai_table():
    """
    Clear the plotbot table state in the session.

    Parameters
    ----------
    session_id : str
        The session identifier.
    """
    # Clear the query selectbox when corpus changes
    query_key = SessionKeys.get_bot_query_key("pandasai")
    scoped_query_key = app_core.widget_manager.get_scoped_key(query_key)
    if scoped_query_key in st.session_state:
        st.session_state[scoped_query_key] = None

    # Clear data preview control widgets
    widget_keys_to_clear = ["pivot_table", "make_percent"]
    for widget_key in widget_keys_to_clear:
        scoped_key = app_core.widget_manager.get_scoped_key(widget_key)
        if scoped_key in st.session_state:
            # Reset to default values
            if widget_key == "pivot_table":
                st.session_state[scoped_key] = False
            elif widget_key == "make_percent":
                st.session_state[scoped_key] = False


def clear_pandasai(session_id: str, clear_all=True):
    """
    Clear pandasai conversation history and reset analysis state.

    Parameters
    ----------
    session_id : str
        The session identifier.
    clear_all : bool
        Whether to clear all related state including widget persistence.
    """
    # Clear pandabot chat history
    if SessionKeys.AI_PANDABOT_CHAT not in st.session_state[session_id]:
        st.session_state[session_id][SessionKeys.AI_PANDABOT_CHAT] = []
    else:
        st.session_state[session_id][SessionKeys.AI_PANDABOT_CHAT] = []

    # Reset the user prompt counter for accurate message indexing
    st.session_state[session_id][SessionKeys.AI_PANDABOT_PROMPT_COUNT] = 0

    # Clear pandabot conversation history (fallback key used by prune_message_thread)
    if "pandasai" in st.session_state[session_id]:
        st.session_state[session_id]["pandasai"] = []

    # Clear pandabot plot storage
    pandabot_img_key = SessionKeys.get_pandabot_img_key(session_id)
    if pandabot_img_key in st.session_state:
        del st.session_state[pandabot_img_key]

    if clear_all:
        # Clear widget manager state for AI-related widgets
        try:
            # Clear data preview control widgets (shared between plotbot and pandabot)
            widget_keys_to_clear = ["pivot_table", "make_percent"]
            for widget_key in widget_keys_to_clear:
                scoped_key = app_core.widget_manager.get_scoped_key(widget_key)
                if scoped_key in st.session_state:
                    # Reset to default values
                    if widget_key == "pivot_table":
                        st.session_state[scoped_key] = False
                    elif widget_key == "make_percent":
                        st.session_state[scoped_key] = False

            # Clear pandabot-specific corpus and query selection widgets
            pandabot_widget_keys = [
                SessionKeys.get_bot_corpus_key("pandasai"),
                SessionKeys.get_bot_query_key("pandasai")
            ]
            # First delete all the keys to clear them completely
            for widget_key in pandabot_widget_keys:
                scoped_key = app_core.widget_manager.get_scoped_key(widget_key)
                if scoped_key in st.session_state[session_id]:
                    del st.session_state[session_id][scoped_key]
                elif widget_key in st.session_state[session_id]:
                    del st.session_state[session_id][widget_key]  # Fallback for direct keys

            # Then set the corpus back to the first option (index 0 = "Target")
            query_key = SessionKeys.get_bot_corpus_key("pandasai")
            scoped_query_key = app_core.widget_manager.get_scoped_key(query_key)
            if scoped_query_key in st.session_state:
                st.session_state[scoped_query_key] = None
        except Exception:
            # Don't fail if widget clearing encounters issues
            pass


@enterprise_ai_call("pandabot_analysis")
def pandabot_user_query(
    df: pd.DataFrame,
    api_key: str,
    prompt: str,
    session_id: str,
    prompt_position: int = 1,
    cache_mode: bool = False
) -> None:
    """
    Handles natural language queries for dataframe analysis using pandasai.

    Primary focus on data analysis and exploration, with secondary support for plotting.
    Uses thread-safe plot capture for visualization requests while maintaining the
    core analytical capabilities that make PandasAI powerful.
    """
    # Get user email with proper fallback for desktop mode
    try:
        user_email = (st.user.email if hasattr(st, 'user') and st.user and
                      hasattr(st.user, 'email') else 'anonymous')
    except Exception:
        user_email = 'anonymous'

    # Only store to Firestore if using community key
    try:
        community_key_available = (
            "openai" in st.secrets and "api_key" in st.secrets["openai"]
        )
    except Exception:
        community_key_available = False

    key_type = determine_api_key_type(DESKTOP, api_key, community_key_available)
    should_store_firestore = cache_mode and key_type == "community"

    conditional_async_add_message(
        enable_firestore=should_store_firestore,
        user_id=user_email,
        session_id=session_id,
        assistant_id=1,
        role="user",
        message_idx=prompt_position,
        message=prompt
    )

    model = OpenAI(api_token=api_key)
    pai.config.set({
        "llm": model,
        "save_logs": False,
        "verbose": False,
        "max_retries": 3,
        "enable_cache": False,
        "use_error_correction_framework": True
    })

    dfs = pai.DataFrame(df)

    # Check if the session state exists
    if SessionKeys.AI_PANDABOT_CHAT not in st.session_state[session_id]:
        st.session_state[session_id][SessionKeys.AI_PANDABOT_CHAT] = []

    response = st.session_state[session_id][SessionKeys.AI_PANDABOT_CHAT]

    # Always use thread-safe plot capture - lightweight when no plots generated
    # Create thread-safe session storage for potential plotting
    session_storage = SessionPlotStorage(session_id)
    session_storage.cleanup_old_sessions()

    with thread_safe_monkeypatch(session_storage) as storage:
        try:
            result = dfs.chat(prompt)

            # Increment quota tracker after successful API call
            try:
                # Only track quota when NOT in desktop mode AND using community API key
                if not DESKTOP:
                    try:
                        user_email = (st.user.email if hasattr(st, 'user') and st.user and
                                      hasattr(st.user, 'email') else 'anonymous')
                    except Exception:
                        user_email = 'anonymous'

                    # Determine if we're using community or individual API key
                    key_type = determine_api_key_type(DESKTOP, api_key)

                    # Only track quota if using community key (not user's personal key)
                    if user_email != 'anonymous' and key_type == "community":
                        # Update session quota (for current session)
                        increment_session_quota(user_email)

                        # Log to database for persistent quota tracking
                        try:
                            backend = get_session_backend()
                            backend.log_user_query(
                                user_id=user_email,
                                session_id=None,  # Use NULL to avoid FK constraints
                                assistant_type="pandabot",
                                message_content=prompt[:500] if prompt else None
                            )
                        except Exception as log_error:
                            # Log the error but don't fail the main request
                            st.error(f"Warning: Failed to log query for quota tracking: "
                                     f"{log_error}")
            except Exception:
                pass  # Don't fail if quota tracking fails

            # Handle different result types based on actual PandasAI output
            # First check for PandasAI 3.0 response objects
            if hasattr(result, '__class__') and 'Response' in result.__class__.__name__:
                # Handle different response types
                if 'NumberResponse' in result.__class__.__name__:
                    # Numeric response (mean, sum, count, etc.)
                    response.append({
                        "role": "assistant",
                        "type": "string",
                        "value": str(result)
                    })
                elif 'StringResponse' in result.__class__.__name__:
                    # String response
                    response.append({
                        "role": "assistant",
                        "type": "string",
                        "value": str(result)
                    })
                elif 'DataFrameResponse' in result.__class__.__name__:
                    # DataFrame response
                    response.append({
                        "role": "assistant",
                        "type": "dataframe",
                        "value": result.value if hasattr(result, 'value') else result
                    })
                elif ('PlotResponse' in result.__class__.__name__ or
                      'ChartResponse' in result.__class__.__name__):
                    # Plot/Chart response - check if we actually captured a plot
                    if "img" in storage.storage and len(storage.storage["img"]) > 0:
                        response.append({
                            "role": "assistant",
                            "type": "plot",
                            "value": storage.storage["img"]
                        })
                    elif plt.get_fignums():
                        fig = plt.gcf()
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches='tight')
                        buf.seek(0)
                        img_bytes = buf.getvalue()
                        buf.close()
                        plt.close(fig)
                        response.append({
                            "role": "assistant",
                            "type": "plot",
                            "value": img_bytes
                        })
                    else:
                        # Plot response but no actual plot - return as string
                        response.append({
                            "role": "assistant",
                            "type": "string",
                            "value": str(result)
                        })
                else:
                    # Other response types - default to string
                    response.append({
                        "role": "assistant",
                        "type": "string",
                        "value": str(result)
                    })
            elif isinstance(result, dict) and "type" in result and "value" in result:
                # Legacy PandasAI structured response (dict format)
                if result["type"] == "chart":
                    # Handle chart/plot results - check if we actually captured a plot
                    # Check captured image first
                    if "img" in storage.storage and len(storage.storage["img"]) > 0:
                        response.append({
                            "role": "assistant",
                            "type": "plot",
                            "value": storage.storage["img"]
                        })
                    # Fallback to checking matplotlib figures in memory
                    elif plt.get_fignums():
                        fig = plt.gcf()
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches='tight')
                        buf.seek(0)
                        img_bytes = buf.getvalue()
                        buf.close()
                        plt.close(fig)
                        response.append({
                            "role": "assistant",
                            "type": "plot",
                            "value": img_bytes
                        })
                    else:
                        # Chart type but no plot captured - likely PandasAI
                        # misclassified a simple statistical answer as a chart.
                        # Return the actual value.
                        response.append({
                            "role": "assistant",
                            "type": "string",
                            "value": str(result["value"])
                        })
                elif result["type"] == "dataframe":
                    # Handle DataFrame results - convert to format expected by UI
                    if isinstance(result["value"], pd.DataFrame):
                        response.append({
                            "role": "assistant",
                            "type": "dataframe",
                            "value": result["value"]
                        })
                    else:
                        # Try to convert to DataFrame if possible
                        try:
                            df_result = pd.DataFrame(result["value"])
                            response.append({
                                "role": "assistant",
                                "type": "dataframe",
                                "value": df_result
                            })
                        except Exception:
                            # Fallback to string if conversion fails
                            response.append({
                                "role": "assistant",
                                "type": "string",
                                "value": str(result["value"])
                            })
                elif result["type"] == "string":
                    # Handle string responses
                    response.append({
                        "role": "assistant",
                        "type": "string",
                        "value": str(result["value"])
                    })
                elif result["type"] == "number":
                    # Handle numeric responses (like mean, sum, etc.)
                    response.append({
                        "role": "assistant",
                        "type": "string",
                        "value": str(result["value"])
                    })
                else:
                    # Other structured response types - default to string
                    response.append({
                        "role": "assistant",
                        "type": "string",
                        "value": str(result["value"])
                    })
            elif isinstance(result, pd.DataFrame):
                # Handle DataFrame results (non-structured response)
                response.append({
                    "role": "assistant",
                    "type": "dataframe",
                    "value": result
                })
            elif isinstance(result, (int, float, str)):
                # Handle simple scalar results
                response.append({
                    "role": "assistant",
                    "type": "string",
                    "value": str(result)
                })
            else:
                # Check if a plot was generated even though result isn't structured
                # (fallback for backwards compatibility)
                if "img" in storage.storage:
                    response.append({
                        "role": "assistant",
                        "type": "plot",
                        "value": storage.storage["img"]
                    })
                elif plt.get_fignums():
                    fig = plt.gcf()
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches='tight')
                    buf.seek(0)
                    img_bytes = buf.getvalue()
                    buf.close()
                    plt.close(fig)
                    response.append({
                        "role": "assistant",
                        "type": "plot",
                        "value": img_bytes
                    })
                else:
                    # Default: treat as string response
                    response.append({
                        "role": "assistant",
                        "type": "string",
                        "value": str(result)
                    })

        except MaliciousQueryError:
            error = (
                ":confused: Sorry, your request could not be processed. "
                "It may be too complex or reference restricted operations."
            )
            response.append({"role": "assistant", "type": "error", "value": error})
        except NoResultFoundError:
            error = (
                ":confused: Sorry, I couldn't find a result for your request. "
                "Try rephrasing or checking your column names."
            )
            response.append({"role": "assistant", "type": "error", "value": error})
        except Exception as e:
            # Handle enterprise circuit breaker and rate limiting errors
            error_msg = str(e).lower()
            if "circuit breaker" in error_msg:
                error = (
                    ":warning: **AI Analysis Service Temporarily Unavailable**\n\n"
                    "The AI analysis assistant is experiencing high demand. "
                    "Please try again in a few moments."
                )
            elif "rate limit" in error_msg:
                error = (
                    ":hourglass_flowing_sand: **Rate Limit Reached**\n\n"
                    "Please wait a moment before making another analysis request."
                )
            elif "quota" in error_msg or "usage" in error_msg:
                error = (
                    ":information_source: **Usage Limit Reached**\n\n"
                    "You've reached your AI analysis quota. "
                    "Consider using manual analysis tools or try again later."
                )
            else:
                error = (
                    ":confused: I couldn't process your request. "
                    "Try rephrasing it or using a different approach."
                )
            response.append({"role": "assistant", "type": "error", "value": error})

    # Prune conversation history
    prune_message_thread(session_id, "pandasai")
