import os
import builtins

"""
Pandabot AI assistant using PandasAI for analysis and plotting.

This module provides pandabot-specific functionality for AI-assisted data analysis
and plotting using PandasAI. Pandabot is a conversational assistant that can handle
both analytical queries and plot generation with thread-safe plot capture.
"""

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

from loguru import logger

# Import shared utilities from llm_core
# Import shared AI utilities
from webapp.utilities.ai.shared import detect_intent, prune_message_thread  # noqa: E402
from webapp.utilities.storage import add_message  # noqa: E402
from webapp.config.session_keys import SessionKeys  # noqa: E402

# Thread-safe global lock for monkeypatching
_monkeypatch_lock = threading.RLock()

# Weak reference set to track active sessions for cleanup
_active_sessions = weakref.WeakSet()


class SessionPlotStorage:
    """Thread-safe session-specific plot storage with automatic cleanup."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.session_img_key = f"pandabot_img_bytes_{session_id}"
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
            try:
                del st.session_state[key]
                logger.debug(f"Cleaned up old session data: {key}")
            except KeyError:
                pass


@contextmanager
def thread_safe_monkeypatch(session_storage: SessionPlotStorage):
    """
    Thread-safe context manager for monkeypatching matplotlib and file operations.

    Uses a global lock to ensure only one thread can modify the global state at a time,
    while using session-specific storage for captured images.
    """
    # import os  # Moved to module level
    # import builtins  # Moved to module level

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
                logger.debug(f"Intercepting file open for writing: {file}")
                logger.debug(f"Session ID: {session_storage.session_id}")

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
                        captured_size = len(storage['img'])
                        logger.debug(f"Captured {captured_size} bytes via file open")
                        logger.debug(f"Session ID: {session_storage.session_id}")

                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        self.close()

                return FakeFile()
            else:
                return _original_open(file, mode, *args, **kwargs)

        def fake_exists(path):
            if isinstance(path, str) and "temp_chart" in path:
                logger.debug(f"fake_exists called for temp_chart: {path} - returning True")
                return True
            return _original_exists(path)

        def fake_isfile(path):
            if isinstance(path, str) and "temp_chart" in path:
                logger.debug(f"fake_isfile called for temp_chart: {path} - returning True")
                return True
            return _original_isfile(path)

        def plt_savefig_to_buffer(fname, *args, **kwargs):
            logger.debug(f"plt.savefig called with fname: {fname}")
            logger.debug(f"Session ID: {session_storage.session_id}")
            if isinstance(fname, str) and "temp_chart" in fname:
                logger.debug("Intercepting plt.savefig temp_chart save, "
                             "redirecting to buffer")
                buf = io.BytesIO()
                _original_plt_savefig(buf, format="png", *args, **kwargs)
                buf.seek(0)
                storage = session_storage.storage
                storage["img"] = buf.getvalue()
                storage["path"] = fname
                storage["session_id"] = session_storage.session_id
                buf.close()
                captured_size = len(storage['img'])
                logger.debug(f"Captured {captured_size} bytes via plt.savefig")
            else:
                logger.debug("Normal plt.savefig, not intercepting")
                _original_plt_savefig(fname, *args, **kwargs)

        def savefig_to_buffer(self, fname, *args, **kwargs):
            logger.debug(f"savefig_to_buffer called with fname: {fname}")
            logger.debug(f"Session ID: {session_storage.session_id}")
            if isinstance(fname, str) and "temp_chart" in fname:
                logger.debug("Intercepting temp_chart save, redirecting to buffer")
                buf = io.BytesIO()
                _original_savefig(self, buf, format="png", *args, **kwargs)
                buf.seek(0)
                storage = session_storage.storage
                storage["img"] = buf.getvalue()
                storage["path"] = fname
                storage["session_id"] = session_storage.session_id
                buf.close()
                captured_size = len(storage['img'])
                logger.debug(f"Captured {captured_size} bytes to buffer")
            else:
                logger.debug("Normal savefig, not intercepting")
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


def clear_pandasai(session_id):
    """
    Clear pandasai conversation history.

    Parameters
    ----------
    session_id : str
        The session identifier.
    """
    if SessionKeys.AI_PANDABOT_CHAT not in st.session_state[session_id]:
        st.session_state[session_id][SessionKeys.AI_PANDABOT_CHAT] = []
    else:
        st.session_state[session_id][SessionKeys.AI_PANDABOT_CHAT] = []


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
    if cache_mode:
        add_message(
            user_id=st.user.email,
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
        "max_retries": 3
    })

    dfs = pai.DataFrame(df)

    # Check if the session state exists
    if SessionKeys.AI_PANDABOT_CHAT not in st.session_state[session_id]:
        st.session_state[session_id][SessionKeys.AI_PANDABOT_CHAT] = []

    response = st.session_state[session_id][SessionKeys.AI_PANDABOT_CHAT]

    # Only use thread-safe plot capture for detected plot requests
    intent = detect_intent(prompt)

    if intent == "plot":
        # Create thread-safe session storage for plotting
        session_storage = SessionPlotStorage(session_id)
        session_storage.cleanup_old_sessions()

        with thread_safe_monkeypatch(session_storage) as storage:
            try:
                result = dfs.chat(prompt)

                # Handle plot results - check captured image first
                if "img" in storage.storage:
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
                    # Plot request but no plot generated - return as string
                    response.append({
                        "role": "assistant",
                        "type": "string",
                        "value": str(result)
                    })

            except MaliciousQueryError:
                logger.error("MaliciousQueryError in pandabot plot generation")
                error = (
                    ":confused: Sorry, your plot request could not be processed. "
                    "It may be too complex or reference restricted operations."
                )
                response.append({"role": "assistant", "type": "error", "value": error})
            except NoResultFoundError:
                logger.error("NoResultFoundError in pandabot plot generation")
                error = (
                    ":confused: Sorry, I couldn't generate a plot for your request. "
                    "Try rephrasing or checking your column names."
                )
                response.append({"role": "assistant", "type": "error", "value": error})
            except Exception as e:
                logger.error(f"Error in pandabot plot generation: {e}")
                error = (
                    ":confused: I couldn't create that plot. "
                    "Try rephrasing your request or check your data structure."
                )
                response.append({"role": "assistant", "type": "error", "value": error})
    else:
        # Handle non-plot requests (primary use case)
        try:
            result = dfs.chat(prompt)

            # Handle different result types from PandasAI
            if isinstance(result, pd.DataFrame):
                # Convert DataFrame to dict for display
                response.append({
                    "role": "assistant",
                    "type": "table",
                    "value": result.to_dict()
                })
            elif isinstance(result, dict) and "type" in result and "value" in result:
                # PandasAI structured response
                if result["type"] != "chart":
                    response.append({
                        "role": "assistant",
                        "type": result["type"],
                        "value": result["value"]
                    })
                else:
                    # Unexpected chart in non-plot request
                    response.append({
                        "role": "assistant",
                        "type": "string",
                        "value": ("A chart was generated but not captured. "
                                  "Try asking for a specific plot.")
                    })
            else:
                # Default: treat as string response
                response.append({
                    "role": "assistant",
                    "type": "string",
                    "value": str(result)
                })

        except MaliciousQueryError:
            logger.error("MaliciousQueryError in pandabot analysis")
            error = (
                ":confused: Sorry, your request could not be processed. "
                "It may be too complex or reference restricted operations."
            )
            response.append({"role": "assistant", "type": "error", "value": error})
        except NoResultFoundError:
            logger.error("NoResultFoundError in pandabot analysis")
            error = (
                ":confused: Sorry, I couldn't find a result for your request. "
                "Try rephrasing or checking your column names."
            )
            response.append({"role": "assistant", "type": "error", "value": error})
        except Exception as e:
            logger.error(f"Error in pandabot analysis: {e}")
            error = (
                ":confused: I couldn't process your request. "
                "Try rephrasing it or using a different approach."
            )
            response.append({"role": "assistant", "type": "error", "value": error})

    # Prune conversation history
    prune_message_thread(session_id, "pandasai")
