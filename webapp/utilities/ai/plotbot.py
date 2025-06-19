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
import openai
import streamlit as st
import seaborn as sns
import plotly.express as px
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins, guarded_unpack_sequence
from RestrictedPython.Eval import default_guarded_getitem as guarded_getitem
from RestrictedPython.Eval import default_guarded_getiter as guarded_getiter

from loguru import logger

from webapp.config.session_keys import SessionKeys  # noqa: E402

# Import webapp utilities after ensuring project root is in path
from webapp.utilities.storage import add_message, add_plot  # noqa: E402

# Import shared AI utilities
from webapp.utilities.ai.shared import (  # noqa: E402
    LLM_MODEL,
    detect_intent,
    prune_message_thread,
    fig_to_svg
)
from webapp.utilities.ai.code_execution import is_code_safe, strip_imports  # noqa: E402

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

    if clear_all:
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
    client = openai.OpenAI(api_key=api_key)
    
    if code_chunk:
        # Update existing code
        prompt = f"""
You are a plotting assistant. The user wants to modify existing plotting code.

Current plotting code:
```python
{code_chunk}
```

User's modification request: {user_request}

Data schema:
{schema}

Data sample (first 3 rows):
{df.head(3).to_string()}

Please provide the COMPLETE updated plotting code using {plot_lib}.

Requirements:
- Use ONLY {plot_lib} for plotting
- The dataframe is already available as 'df'
- Create a figure and assign it to variable 'fig'
- Do NOT include import statements
- Do NOT call plt.show(), fig.show(), or display()
- Do NOT save to files
- Return only executable Python code
- Code must work with the provided dataframe schema

Return only the Python code, no explanations or markdown formatting.
"""
    else:
        # Generate new code
        prompt = f"""
You are a plotting assistant. Generate Python plotting code based on the user's request.

User request: {user_request}

Data schema:
{schema}

Data sample (first 3 rows):
{df.head(3).to_string()}

Please generate plotting code using {plot_lib}.

Requirements:
- Use ONLY {plot_lib} for plotting
- The dataframe is already available as 'df'
- Create a figure and assign it to variable 'fig'
- Do NOT include import statements
- Do NOT call plt.show(), fig.show(), or display()
- Do NOT save to files
- Return only executable Python code
- Code must work with the provided dataframe schema

Return only the Python code, no explanations or markdown formatting.
"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            **llm_params
        )
        
        code = response.choices[0].message.content.strip()
        
        # Clean up code (remove markdown formatting if present)
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        
        code = code.strip()
        
        # Validate the code contains required elements
        if not code or "fig" not in code:
            logger.error("Generated code is empty or doesn't create a 'fig' variable")
            return {
                "type": "error",
                "value": "Sorry, I couldn't generate valid plotting code. Please try rephrasing your request."  # noqa: E501
            }
        
        return code
        
    except Exception as e:
        logger.error(f"Error generating plotting code: {e}")
        return {
            "type": "error",
            "value": f"Sorry, I encountered an error: {str(e)}"
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
        logger.error("plot_code is not a valid string.")
        return {
            "type": "error",
            "value": "Sorry, I couldn't generate your plot. Please try rephrasing your request."  # noqa: E501
        }
    
    # Strip import statements before safety check
    plot_code = strip_imports(plot_code)
    if not is_code_safe(plot_code):
        logger.error("Unsafe code detected in plot instructions.")
        return {
            "type": "error",
            "value": "Sorry, your request included unsafe code and could not be executed."
        }

    exec_locals = {}
    allowed_globals = {
        "__builtins__": safe_builtins,
        "df": df,
        "_getitem_": guarded_getitem,
        "_unpack_sequence_": guarded_unpack_sequence,
        "_getiter_": guarded_getiter,
    }

    # Add library-specific globals
    if plot_lib == "matplotlib":
        allowed_globals["plt"] = plt
    elif plot_lib == "seaborn":
        allowed_globals["sns"] = sns
        allowed_globals["plt"] = plt
    elif plot_lib == "plotly.express":
        allowed_globals["px"] = px
    else:
        logger.error(f"Unknown plot_lib: {plot_lib}")
        return {
            "type": "error",
            "value": f"Unsupported plotting library: {plot_lib}"
        }

    try:
        byte_code = compile_restricted(plot_code, '<string>', 'exec')
        exec(byte_code, allowed_globals, exec_locals)
        if "fig" in exec_locals:
            fig = exec_locals["fig"]
            return {
                "type": "plot",
                "value": fig
            }
        else:
            logger.error("No figure object ('fig') was created by the code.")
            return {
                "type": "error",
                "value": "Sorry, the code didn't create a figure. Please try a different request."  # noqa: E501
            }
    except Exception as e:
        logger.error(f"Error executing plot code: {e}")
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

    if cache_mode:
        add_message(user_id=st.user.email,
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
                logger.debug(f"Cache hit for key: {cache_key}")
                plot_code = cached_code
            else:
                if cached:
                    logger.debug(f"Cache hit but invalid cached code for key: {cache_key}")
                else:
                    logger.debug(f"Cache miss for key: {cache_key}")
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
                logger.error("plot_code is not a valid string after generation/retrieval")
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
            if cache_mode and plot_fig.get("type") == "plot":
                svg_str = fig_to_svg(figure=plot_fig["value"], plot_lib=plot_lib)
                add_plot(user_id=st.user.email,
                         session_id=session_id,
                         assistant_id=0,
                         message_idx=prompt_position,
                         plot_library=plot_lib,
                         plot_svg=svg_str)

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
