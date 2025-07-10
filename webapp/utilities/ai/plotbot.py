"""
Plotbot AI assistant for iterative plotting.

This module provides plotbot-specific functionality for AI-assisted code generation
and execution for data visualization. Plotbot is an iterative assistant that generates
executable plotting code and can refine it based on user feedback.
"""

import hashlib

import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import plotly.express as px
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins, guarded_unpack_sequence
from RestrictedPython.Eval import default_guarded_getitem as guarded_getitem
from RestrictedPython.Eval import default_guarded_getiter as guarded_getiter

from webapp.utilities.state import SessionKeys
# Add async storage import for non-blocking Firestore operations
from webapp.utilities.storage import (
    conditional_async_add_message, conditional_async_add_plot
)
from webapp.utilities.ai.shared import (
    LLM_MODEL, detect_intent,
    prune_message_thread, fig_to_svg, increment_session_quota
)
from webapp.utilities.ai.code_execution import is_code_safe, strip_imports
from webapp.utilities.ai.enterprise_integration import (
    make_protected_openai_call, determine_api_key_type
)
from webapp.utilities.storage.backend_factory import get_session_backend
from webapp.config.unified import get_ai_config
from webapp.utilities.core import app_core


# Get AI configuration using standardized access
AI_CONFIG = get_ai_config()
DESKTOP = AI_CONFIG['desktop_mode']

# Plotbot-specific constants
FORBIDDEN_PATTERNS = [
    r'^\s*import\s',         # import statement at line start
    r'\bexec\s*\(',          # exec(
    r'\beval\s*\(',          # eval(
    r'\bopen\s*\(',          # open(
    r'^\s*os\.',             # os. usage at line start
    r'^\s*sys\.',            # sys. usage at line start
    r'^\s*subprocess\.',     # subprocess. usage at line start
]

# AI logging is automatically configured by importing the centralized logging system


def previous_code_chunk(session_id: str) -> str:
    """
    Extract the most recent code chunk from plotbot conversation history.

    Parameters
    ----------
    session_id : str
        The session identifier.

    Returns
    -------
    str or None
        The most recent code chunk, or None if no code found.
    """
    chat_history = st.session_state[session_id].get(SessionKeys.AI_PLOTBOT_CHAT, [])

    # Look for the most recent code chunk in conversation history
    for message in reversed(chat_history):
        if (message.get("role") == "assistant" and
                message.get("type") == "code" and
                isinstance(message.get("value"), str) and
                message.get("value").strip()):
            return message.get("value")

    return None


def clear_plotbot_table():
    """
    Clear the plotbot table state in the session.

    Parameters
    ----------
    session_id : str
        The session identifier.
    """
    # Clear the query selectbox when corpus changes
    query_key = SessionKeys.get_bot_query_key("plotbot")
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


def clear_plotbot(session_id: str, clear_all=True):
    """
    Clear plotbot conversation history and reset plotting state.

    Parameters
    ----------
    session_id : str
        The session identifier.
    clear_all : bool
        Whether to clear all related state including widget persistence.
    """
    if SessionKeys.AI_PLOTBOT_CHAT not in st.session_state[session_id]:
        st.session_state[session_id][SessionKeys.AI_PLOTBOT_CHAT] = []
    else:
        st.session_state[session_id][SessionKeys.AI_PLOTBOT_CHAT] = []

    st.session_state[session_id][SessionKeys.AI_PLOT_INTENT] = False

    # Reset the user prompt counter for accurate message indexing
    st.session_state[session_id][SessionKeys.AI_PLOTBOT_PROMPT_COUNT] = 0

    # Clear plotbot cache
    if SessionKeys.AI_PLOTBOT_CACHE in st.session_state[session_id]:
        st.session_state[session_id][SessionKeys.AI_PLOTBOT_CACHE] = {}

    # Clear plotbot conversation history (fallback key)
    if "plotbot" in st.session_state[session_id]:
        st.session_state[session_id]["plotbot"] = []

    # Clear plotbot SVG export data
    if "plotbot_plot_svg" in st.session_state[session_id]:
        del st.session_state[session_id]["plotbot_plot_svg"]

    if clear_all:
        # Clear persistent plotbot-specific session keys
        if SessionKeys.AI_PLOTBOT_PERSIST not in st.session_state[session_id]:
            st.session_state[session_id][SessionKeys.AI_PLOTBOT_PERSIST] = {}
        else:
            try:
                persist = st.session_state[session_id][SessionKeys.AI_PLOTBOT_PERSIST]
                persist[SessionKeys.AI_PLOTBOT_QUERY] = None
                persist[SessionKeys.AI_PLOTBOT_CORPUS] = 0
                persist[SessionKeys.AI_PLOTBOT_PIVOT_TABLE] = False
                persist[SessionKeys.AI_PLOTBOT_MAKE_PERCENT] = False
            except KeyError:
                pass

        # Clear widget manager state for AI-related widgets
        try:
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

            # Clear plotbot-specific corpus and query selection widgets
            plotbot_widget_keys = [
                SessionKeys.get_bot_corpus_key("plotbot"),
                SessionKeys.get_bot_query_key("plotbot")
            ]

            # First delete all the keys to clear them completely
            for widget_key in plotbot_widget_keys:
                scoped_key = app_core.widget_manager.get_scoped_key(widget_key)
                if scoped_key in st.session_state[session_id]:
                    del st.session_state[session_id][scoped_key]
                elif widget_key in st.session_state[session_id]:
                    del st.session_state[session_id][widget_key]  # Fallback for direct keys

            # Then set the corpus back to the first option (index 0 = "Target")
            query_key = SessionKeys.get_bot_corpus_key("plotbot")
            scoped_query_key = app_core.widget_manager.get_scoped_key(query_key)
            if scoped_query_key in st.session_state:
                st.session_state[scoped_query_key] = None

        except Exception:
            # Don't fail if widget clearing encounters issues
            pass


def make_plotbot_cache_key(user_input, df, plot_lib, code_chunk=None):
    """
    Generate a cache key for plotbot requests.

    Parameters
    ----------
    user_input : str
        The user's plotting request.
    df : pd.DataFrame
        The dataframe being plotted.
    plot_lib : str
        The plotting library.
    code_chunk : str, optional
        Existing code chunk for updates.

    Returns
    -------
    str
        A unique cache key for this request.
    """
    # Create a hash of the user input, dataframe shape/columns, and plot_lib
    df_hash = hashlib.md5(
        str(df.shape).encode() + str(df.columns.tolist()).encode()
    ).hexdigest()[:8]

    input_hash = hashlib.md5(user_input.encode()).hexdigest()[:8]
    lib_hash = hashlib.md5(plot_lib.encode()).hexdigest()[:4]

    code_hash = ""
    if code_chunk:
        code_hash = hashlib.md5(code_chunk.encode()).hexdigest()[:8]

    return f"plotbot_{input_hash}_{df_hash}_{lib_hash}_{code_hash}"


def plotbot_code_generate_or_update(
    df: pd.DataFrame,
    user_request: str,
    plot_lib: str,
    schema: str,
    api_key: str,
    llm_params: dict,
    code_chunk: str = None
) -> str:
    """
    Generate or update plotting code using the LLM.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to plot.
    user_request : str
        User's plotting request.
    plot_lib : str
        The plotting library to use ('matplotlib', 'seaborn', 'plotly.express').
    schema : str
        String representation of the dataframe schema.
    api_key : str
        OpenAI API key.
    llm_params : dict
        LLM parameters.
    code_chunk : str, optional
        Existing code to update/modify.

    Returns
    -------
    str or dict
        Generated code string, or error dict if generation failed.
    """
    valid_columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns.tolist()

    if code_chunk is None:
        # Library-specific instructions and examples
        if plot_lib == "matplotlib":
            lib_instructions = """
    - Use the format 'fig, ax = plt.subplots()' to create the figure.
    - Do not call 'fig.show()' or 'plt.show()'.
    - Use matplotlib functions like ax.plot(), ax.bar(), ax.scatter(), etc.

    Example:
    fig, ax = plt.subplots()
    ax.plot(df['col1'], df['col2'])
    ax.set_xlabel('Column 1')
    ax.set_ylabel('Column 2')"""
        elif plot_lib == "seaborn":
            lib_instructions = """
    - Use seaborn functions like sns.barplot(), sns.scatterplot(), sns.lineplot(), etc.
    - Always create a figure first with 'fig, ax = plt.subplots()'.
    - Pass the 'ax' parameter to seaborn functions.
    - Do not call 'plt.show()'.

    Example:
    fig, ax = plt.subplots()
    sns.barplot(data=df, x='col1', y='col2', ax=ax)
    ax.set_title('My Plot')"""
        elif plot_lib == "plotly.express":
            lib_instructions = """
    - Use plotly.express functions like px.bar(), px.scatter(), px.line(), etc.
    - Assign the result to a variable called 'fig'.
    - Do not call 'fig.show()'.

    Example:
    fig = px.bar(df, x='col1', y='col2')
    fig.update_layout(title='My Plot')"""
        else:
            # Default to matplotlib
            lib_instructions = """
    - Use the format 'fig, ax = plt.subplots()' to create the figure.
    - Do not call 'fig.show()' or 'plt.show()'.
    - Use matplotlib functions like ax.plot(), ax.bar(), ax.scatter(), etc."""

        prompt = f"""
    You are a Python plotting assistant.

    The user has requested to create a plot.
    Here is the data schema:
    {schema}

    The available columns in the DataFrame are:
    {', '.join(valid_columns)}.
    Numeric columns are: {', '.join(numeric_columns)}.
    Non-numeric columns are: {', '.join(non_numeric_columns)}.

    Based on the user request: '{user_request}', generate Python code for plotting using {plot_lib}.

    Instructions:
    - Only output valid Python code, with no explanations or markdown formatting.
    - Do not include any import statements.
    - The DataFrame is called 'df'.
    - Use only columns that exist in the DataFrame. If the user mentions a column that does not exist, ignore it and use available columns instead.
    - If the request involves numeric data (like line charts, bar charts, histograms), use only numeric columns.
    - If the request involves categorical or non-numeric data (like pie charts or scatter plots with labels), you can use non-numeric columns.
    - Ensure the code is error-free and matches the DataFrame schema.
    - If you need to set axis labels or titles, use generic names if the user does not specify.
    - Include concise comments in the code to explain non-obvious steps or terminology (e.g., what a spine is or how to remove it).
    - Do not include explanations or markdown outside the code.
    {lib_instructions}

    Now, generate the code:
    """  # noqa: E501

    else:
        # Library-specific instructions for updates
        if plot_lib == "matplotlib":
            lib_instructions = """
    - Use the format 'fig, ax = plt.subplots()' to create the figure.
    - Do not call 'fig.show()' or 'plt.show()'.
    - Use matplotlib functions like ax.plot(), ax.bar(), ax.scatter(), etc.

    Example:
    fig, ax = plt.subplots()
    ax.plot(df['col1'], df['col2'])
    ax.set_xlabel('Column 1')
    ax.set_ylabel('Column 2')"""
        elif plot_lib == "seaborn":
            lib_instructions = """
    - Use seaborn functions like sns.barplot(), sns.scatterplot(), sns.lineplot(), etc.
    - Always create a figure first with 'fig, ax = plt.subplots()'.
    - Pass the 'ax' parameter to seaborn functions.
    - Do not call 'plt.show()'.

    Example:
    fig, ax = plt.subplots()
    sns.barplot(data=df, x='col1', y='col2', ax=ax)
    ax.set_title('My Plot')"""
        elif plot_lib == "plotly.express":
            lib_instructions = """
    - Use plotly.express functions like px.bar(), px.scatter(), px.line(), etc.
    - Assign the result to a variable called 'fig'.
    - Do not call 'fig.show()'.

    Example:
    fig = px.bar(df, x='col1', y='col2')
    fig.update_layout(title='My Plot')"""
        else:
            # Default to matplotlib
            lib_instructions = """
    - Use the format 'fig, ax = plt.subplots()' to create the figure.
    - Do not call 'fig.show()' or 'plt.show()'.
    - Use matplotlib functions like ax.plot(), ax.bar(), ax.scatter(), etc."""

        prompt = f"""
    You are a Python plotting assistant.

    The user has requested to update code that generates a plot.
    Here is the data schema:
    {schema}

    The available columns in the DataFrame are:
    {', '.join(valid_columns)}.
    Numeric columns are: {', '.join(numeric_columns)}.
    Non-numeric columns are: {', '.join(non_numeric_columns)}.

    Based on the user request: '{user_request}',
    and the current code:
    {code_chunk}

    Update the code to generate the plot using the following instructions:
    - Only output valid Python code, with no explanations or markdown formatting.
    - Do not include any import statements.
    - The DataFrame is called 'df'.
    - Use only columns that exist in the DataFrame. If the user mentions a column that does not exist, ignore it and use available columns instead.
    - If the request involves numeric data (like line charts, bar charts, histograms), use only numeric columns.
    - If the request involves categorical or non-numeric data (like pie charts or scatter plots with labels), you can use non-numeric columns.
    - Ensure the code is error-free and matches the DataFrame schema.
    - If you need to set axis labels or titles, use generic names if the user does not specify.
    - Include concise comments in the code to explain non-obvious steps or terminology (e.g., what a spine is or how to remove it).
    - Do not include explanations or markdown outside the code.
    {lib_instructions}

    Now, update and output the code:
    """  # noqa: E501

    try:
        # Use enterprise-protected OpenAI call with circuit breaker and fallback
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        request_params = {
            "model": LLM_MODEL,
            "messages": messages,
            "stream": True,
            "temperature": llm_params["temperature"],
            "max_tokens": llm_params["max_tokens"],
            "top_p": llm_params["top_p"],
            "frequency_penalty": llm_params["frequency_penalty"],
            "presence_penalty": llm_params["presence_penalty"]
        }

        # Make the protected call using enterprise infrastructure
        response = make_protected_openai_call(
            api_key=api_key,
            request_params=request_params,
            request_type="chat_completion",
            cache_key=f"plotbot_code_{user_request}_{plot_lib}_{len(str(df.shape))}"
        )

        # Handle error responses from circuit breaker
        if isinstance(response, dict) and response.get("type") == "error":
            return response

        full_response = ""
        for chunk in response:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content:
                full_response += chunk_content

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
                            assistant_type="plotbot",
                            message_content=user_request[:500] if user_request else None
                        )
                    except Exception as log_error:
                        # Log the error but don't fail the main request
                        st.error(f"Warning: Failed to log query for quota tracking: "
                                 f"{log_error}")
        except Exception:
            pass  # Don't fail if quota tracking fails

        if "```python" in full_response:
            full_response = full_response.replace("```python", "")
        if "```" in full_response:
            full_response = full_response.replace("```", "")
        if "fig.show()" in full_response:
            full_response = full_response.replace("fig.show()", "")

        valid_columns = df.columns
        for col in valid_columns:
            if "labels_column_name" in full_response:
                full_response = full_response.replace(
                    "labels_column_name",
                    col
                )

        return full_response

    except Exception:
        return {
            "type": "error",
            "value": "Sorry, I couldn't generate your plot. Please try rephrasing your request."  # noqa: E501
        }


def plotbot_code_execute(plot_code: str,
                         df: pd.DataFrame,
                         plot_lib: str) -> dict:
    """
    Execute plotting code in a safe, restricted environment.

    Parameters
    ----------
    plot_code : str
        The plotting code to execute.
    df : pd.DataFrame
        The dataframe to plot.
    plot_lib : str
        The plotting library being used.

    Returns
    -------
    dict
        Result dictionary with 'type' and 'value' keys.
        Type can be 'plot' (success) or 'error' (failure).
        For plots, value contains the matplotlib figure.
    """
    if not isinstance(plot_code, str) or not plot_code.strip():
        return {
            "type": "error",
            "value": "Sorry, I couldn't generate your plot. Please try rephrasing your request."  # noqa: E501
        }

    # Strip import statements before safety check
    plot_code = strip_imports(plot_code)
    if not is_code_safe(plot_code):
        return {
            "type": "error",
            "value": "Sorry, your request included unsafe code and could not be executed."
        }

    exec_locals = {}

    # Create a safer attribute getter for DataFrame operations
    def safe_getattr(obj, name, default=None):
        """Safe attribute getter for restricted execution."""
        if hasattr(obj, name):
            return getattr(obj, name)
        return default

    allowed_globals = {
        "__builtins__": safe_builtins,
        "df": df,
        "_getitem_": guarded_getitem,
        "_unpack_sequence_": guarded_unpack_sequence,
        "_getiter_": guarded_getiter,
        "_getattr_": safe_getattr,
        # Always include matplotlib as it's needed for figure creation
        "plt": plt,
        # Add pandas functionality for DataFrame operations
        "pd": pd,
        # Add common Python functions needed for data manipulation
        "len": len,
        "max": max,
        "min": min,
        "sum": sum,
        "sorted": sorted,
        "list": list,
        "dict": dict,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
    }

    # Add library-specific globals
    if plot_lib == "matplotlib":
        # plt already added above
        pass
    elif plot_lib == "seaborn":
        allowed_globals["sns"] = sns
        # plt already added above
    elif plot_lib == "plotly.express":
        allowed_globals["px"] = px
    else:
        return {
            "type": "error",
            "value": f"Unsupported plotting library: {plot_lib}"
        }

    try:
        byte_code = compile_restricted(plot_code, '<string>', 'exec')
        if byte_code is None:
            return {
                "type": "error",
                "value": "Sorry, the code could not be compiled safely."
            }

        exec(byte_code, allowed_globals, exec_locals)
        if "fig" in exec_locals:
            fig = exec_locals["fig"]
            return {
                "type": "plot",
                "value": fig
            }
        else:
            return {
                "type": "error",
                "value": "Sorry, the code didn't create a figure. Please try a different request."  # noqa: E501
            }
    except Exception as e:
        return {
            "type": "error",
            "value": f"Sorry, there was an error executing your plot: {str(e)}"
        }


def plotbot_user_query(session_id: str,
                       df: pd.DataFrame,
                       plot_lib: str,
                       user_input: str,
                       api_key: str,
                       llm_params: dict,
                       code_chunk=None,
                       prompt_position: int = 1,
                       cache_mode: bool = False) -> None:
    """
    Handle user queries for plotbot (iterative plotting assistant).

    Parameters
    ----------
    session_id : str
        The session identifier.
    df : pd.DataFrame
        The dataframe to plot.
    plot_lib : str
        The plotting library to use.
    user_input : str
        User's plotting request.
    api_key : str
        OpenAI API key.
    llm_params : dict
        LLM parameters.
    code_chunk : str, optional
        Existing code to update/modify.
    prompt_position : int
        Position in the conversation for caching.
    cache_mode : bool
        Whether to cache results.
    """
    # Ensure session state keys exist
    if SessionKeys.AI_PLOTBOT_CHAT not in st.session_state[session_id]:
        st.session_state[session_id][SessionKeys.AI_PLOTBOT_CHAT] = []
    if SessionKeys.AI_PLOT_INTENT not in st.session_state[session_id]:
        st.session_state[session_id][SessionKeys.AI_PLOT_INTENT] = False

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

    conditional_async_add_message(enable_firestore=should_store_firestore,
                                  user_id=user_email,
                                  session_id=session_id,
                                  assistant_id=0,
                                  role="user",
                                  message_idx=prompt_position,
                                  message=user_input)

    intent = detect_intent(user_input)

    # Handle schema generation for both pandas and polars DataFrames
    if isinstance(df, pd.DataFrame):
        schema = df.dtypes.to_string()
    elif isinstance(df, pl.DataFrame):
        schema = str(df.dtypes)
    else:
        schema = str(type(df))

    if intent == "none":
        response = (
            ":grey_question: Please enter a request for a plot or chart."
        )
        st.session_state[session_id][SessionKeys.AI_PLOTBOT_CHAT].append(
            {"role": "assistant", "type": "error", "value": response}
        )
        prune_message_thread(session_id, SessionKeys.AI_PLOTBOT_CHAT)
        return

    if intent == "plot":
        st.session_state[session_id][SessionKeys.AI_PLOT_INTENT] = True

        if df is not None:
            # Auto-detect previous code chunk if none provided
            if code_chunk is None:
                chat_history = st.session_state[session_id].get(
                    SessionKeys.AI_PLOTBOT_CHAT, []
                )
                # Look for the most recent code chunk in conversation history
                for message in reversed(chat_history):
                    if (message.get("role") == "assistant" and
                            message.get("type") == "code" and
                            isinstance(message.get("value"), str) and
                            message.get("value").strip()):
                        code_chunk = message.get("value")
                        break

            # Use unified code generation/update function
            cache_dict = st.session_state[session_id].setdefault(
                SessionKeys.AI_PLOTBOT_CACHE, {}
            )

            cache_key = make_plotbot_cache_key(user_input, df, plot_lib, code_chunk)

            # Check for cached code only (never cache figures)
            cached = cache_dict.get(cache_key)
            cached_code = cached.get("code") if cached else None
            if (cached and isinstance(cached_code, str) and
                    cached_code.strip()):
                plot_code = cached_code
            else:
                plot_code = plotbot_code_generate_or_update(
                    df=df,
                    user_request=user_input,
                    plot_lib=plot_lib,
                    schema=schema,
                    api_key=api_key,
                    llm_params=llm_params,
                    code_chunk=code_chunk
                )

                # Standardized error handling for code generation
                if plot_code is None or (isinstance(plot_code, dict) and plot_code.get("type") == "error"):  # noqa: E501
                    error_message = (
                        plot_code.get("value") if isinstance(plot_code, dict) else
                        "Sorry, I couldn't generate your plot. Please try rephrasing your request."  # noqa: E501
                    )

                    # Add specific messaging for enterprise circuit breaker events
                    if (isinstance(plot_code, dict) and
                            "circuit breaker" in error_message.lower()):
                        error_message = (
                            ":warning: **AI Service Temporarily Unavailable**\n\n"
                            "The AI plotting assistant is experiencing high demand. "
                            "Please try again in a few moments, or consider using "
                            "manual plotting tools in the meantime."
                        )
                    elif (isinstance(plot_code, dict) and
                          "rate limit" in error_message.lower()):
                        error_message = (
                            ":hourglass_flowing_sand: **Rate Limit Reached**\n\n"
                            "Please wait a moment before making another plotting request."
                        )

                    st.session_state[session_id][SessionKeys.AI_PLOTBOT_CHAT].append(
                        {"role": "assistant", "type": "error", "value": error_message}
                    )
                    prune_message_thread(session_id, SessionKeys.AI_PLOTBOT_CHAT)
                    return

                # Cache only the code (never cache figures)
                if not (isinstance(plot_code, dict) and plot_code.get("type") == "error"):
                    cache_dict[cache_key] = {"code": plot_code}

            # Final validation: ensure plot_code is a valid string
            if not isinstance(plot_code, str) or not plot_code.strip():
                error_msg = "Sorry, I couldn't generate valid plot code. Please try again."
                st.session_state[session_id][SessionKeys.AI_PLOTBOT_CHAT].append(
                    {"role": "assistant", "type": "error", "value": error_msg}
                )
                prune_message_thread(session_id, SessionKeys.AI_PLOTBOT_CHAT)
                return

            # Always execute the code to generate a fresh figure
            plot_fig = plotbot_code_execute(plot_code=plot_code, plot_lib=plot_lib, df=df)

            if not isinstance(plot_fig, dict):
                plot_fig = {
                    "type": "error",
                    "value": "Sorry, something went wrong while generating your plot."
                }

            if plot_fig.get("type") == "error":
                st.session_state[session_id][SessionKeys.AI_PLOTBOT_CHAT].append(
                    {"role": "assistant", "type": "error", "value": plot_fig.get("value")}
                )
                prune_message_thread(session_id, SessionKeys.AI_PLOTBOT_CHAT)
                return

            # Cache plot if needed
            if plot_fig.get("type") == "plot":
                svg_str = fig_to_svg(figure=plot_fig["value"], plot_lib=plot_lib)
                # Get user email with proper fallback for desktop mode
                try:
                    user_email = (st.user.email if hasattr(st, 'user') and st.user and
                                  hasattr(st.user, 'email') else 'anonymous')
                except Exception:
                    user_email = 'anonymous'

                conditional_async_add_plot(enable_firestore=should_store_firestore,
                                           user_id=user_email,
                                           session_id=session_id,
                                           assistant_id=0,
                                           message_idx=prompt_position,
                                           plot_library=plot_lib,
                                           plot_svg=svg_str)

            # Generate SVG for export capability
            svg_str = ""
            if plot_fig.get("type") == "plot":
                svg_str = fig_to_svg(figure=plot_fig["value"], plot_lib=plot_lib)
                # Store SVG for work preservation export
                st.session_state[session_id]["plotbot_plot_svg"] = svg_str

            # Append code and plot to session state
            st.session_state[session_id][SessionKeys.AI_PLOTBOT_CHAT].append(
                {"role": "assistant", "type": "code", "value": plot_code}
            )
            prune_message_thread(session_id, SessionKeys.AI_PLOTBOT_CHAT)

            if plot_fig.get("type") == "plot":
                st.session_state[session_id]["plotbot"].append(
                    {"role": "assistant", "type": "plot", "value": plot_fig["value"]}
                )
                prune_message_thread(session_id, "plotbot")
            else:
                error_message = (
                    "No plot was generated. As a plotbot, I can only execute specific types of requests. "  # noqa: E501
                    "For more complex tasks, you might want to try AI-assisted analysis."
                )
                st.session_state[session_id]["plotbot"].append(
                    {"role": "assistant", "type": "error", "value": error_message}
                )
                prune_message_thread(session_id, "plotbot")
        else:
            error_message = "No plot was generated. Please check the code."
            st.session_state[session_id]["plotbot"].append(
                {"role": "assistant", "type": "error", "value": error_message}
            )
            prune_message_thread(session_id, "plotbot")
    else:
        response = (
            ":warning: I am unable to assist with that request.\n"
            "I'm a plotbot, not a chat bot.\n"
            "Try asking me to plot something related to the data."
        )
        st.session_state[session_id]["plotbot"].append(
            {"role": "assistant", "type": "error", "value": response}
        )
        prune_message_thread(session_id, "plotbot")


def generate_plotbot_code_and_plot(
    df: pd.DataFrame,
    plot_lib: str,
    user_input: str,
    api_key: str,
    llm_params: dict,
    code_chunk: str = None
) -> tuple[str, dict]:
    """
    Generate plotting code and execute it to create a plot.

    This is a helper function that combines code generation and execution
    in a single call, useful for non-interactive scenarios.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to plot.
    plot_lib : str
        The plotting library to use.
    user_input : str
        User's plotting request.
    api_key : str
        API key for LLM service.
    llm_params : dict
        LLM parameters.
    code_chunk : str, optional
        Existing code to update.

    Returns
    -------
    tuple[str, dict]
        Generated code and plot result.
    """
    schema = str(df.dtypes.to_dict())

    # Generate the code
    plot_code = plotbot_code_generate_or_update(
        df, user_input, plot_lib, schema, api_key, llm_params, code_chunk
    )

    if not plot_code:
        return None, None

    # Execute the code
    plot_result = plotbot_code_execute(plot_code, df, plot_lib)

    if plot_result.get("type") == "plot":
        return plot_code, plot_result.get("value")
    else:
        return plot_code, None
