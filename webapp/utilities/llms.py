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

import base64
import hashlib
import json
import io
import pathlib
import openai
import re
import sys

import docuscospacy as ds
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

from loguru import logger
from PIL import Image
from pandasai.exceptions import MaliciousQueryError, NoResultFoundError
from pandasai_openai import OpenAI
import pandasai as pai

from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins
from RestrictedPython.Guards import guarded_unpack_sequence
from RestrictedPython.Eval import default_guarded_getitem as guarded_getitem
from RestrictedPython.Eval import default_guarded_getiter as guarded_getiter

# Ensure project root is in sys.path for both desktop and online
project_root = pathlib.Path(__file__).parent.parents[1].resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from webapp.utilities import analysis  # noqa: E402
from webapp.utilities.cache import add_message, add_plot  # noqa: E402
from webapp.utilities.handlers import import_options_general  # noqa: E402, E501

# Set up logging
logger.add(
    "webapp/plotbot_error.log",
    level="ERROR",
    rotation="10 MB",
    retention="10 days",
    enqueue=True
    )


# set paths
OPTIONS = str(project_root.joinpath("webapp/options.toml"))

_options = import_options_general(OPTIONS)
LLM_PARAMS = _options['llm']['llm_parameters']
LLM_MODEL = _options['llm']['llm_model']
DESKTOP = _options['global']['desktop_mode']
CACHE = _options['cache']['cache_mode']

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

FORBIDDEN_PATTERNS = [
    r'^\s*import\s',         # import statement at line start
    r'\bexec\s*\(',          # exec(
    r'\beval\s*\(',          # eval(
    r'\bopen\s*\(',          # open(
    r'^\s*os\.',             # os. usage at line start
    r'^\s*sys\.',            # sys. usage at line start
    r'^\s*subprocess\.',     # subprocess. usage at line start
]


def print_settings(dct):
    for key, value in dct.items():
        settings = f"""
        Temperature: {dct["temperature"]}\n
        Frequency Penalty: {dct["frequency_penalty"]}\n
        Presence Penalty: {dct["presence_penalty"]}
        """
    return settings


def is_openai_key_valid(api_key: str) -> bool:
    """
    Checks if the provided OpenAI API key is valid by
    attempting to make a simple API call.
    Returns True if the key is valid, False otherwise.
    """
    openai.api_key = api_key
    try:
        openai.models.list()
        return True
    except openai.AuthenticationError:
        return False
    except Exception:
        return False


def clear_plotbot(session_id: str,
                  clear_all=True):
    if "plotbot" not in st.session_state[session_id]:
        st.session_state[session_id]["plotbot"] = []
    else:
        st.session_state[session_id]["plotbot"] = []

    st.session_state[session_id]["plot_intent"] = False

    if clear_all:
        if "assisted_plotting_PERSIST" not in st.session_state[session_id]:
            st.session_state[session_id]["assisted_plotting_PERSIST"] = {}
        else:
            try:
                st.session_state[
                    session_id
                    ]["assisted_plotting_PERSIST"]["plotbot_query"] = None
                st.session_state[
                    session_id
                    ]["assisted_plotting_PERSIST"]["plotbot_corpus"] = 0
                st.session_state[
                    session_id
                    ]["assisted_plotting_PERSIST"]["pivot_table"] = False
                st.session_state[
                    session_id
                    ]["assisted_plotting_PERSIST"]["make_percent"] = False
            except KeyError:
                pass


def clear_pandasai(session_id):
    if "pandasai" not in st.session_state[session_id]:
        st.session_state[session_id]["pandasai"] = []
    else:
        st.session_state[session_id]["pandasai"] = []


def tables_to_list(session_id: str,
                   corpus: str,
                   categories: list[str] = None) -> list:
    """
    Returns a list of values from all_tables if the keys in
    st.session_state[session_id]["target"] match the keys in all_tables.

    Parameters
    ----------
    session_id : str
        The session identifier used to access the session state.

    Returns
    -------
    list
        A list of matching values from all_tables.
    """
    all_tables = {
        'Tags Table: DocuScope': 'tt_ds',
        'Tags Table: Parts-of-Speech': 'tt_pos',
        'Tags Table: Parts-of-Speech Simplified': 'dtm_pos',
        'Document-Term Matrix: DocuScope': 'dtm_ds',
        'Document-Term Matrix: Parts-of-Speech': 'dtm_pos',
        'Document-Term Matrix: Parts-of-Speech Simplified': 'dtm_pos',
        'Keywords: DocuScope': 'kw_ds',
        'Keywords: Parts-of-Speech': 'kw_pos',
        'Keywords: Parts-of-Speech Simplified': 'kt_pos',
        'Keytags: DocuScope': 'kt_ds',
        'Keytags Table: Parts-of-Speech': 'kt_pos'
    }

    if corpus == "Target":
        target_keys = st.session_state[session_id].get("target", [])
        matching_keys = [
            key for key, value in all_tables.items()
            if value in target_keys
            ]
    elif corpus == "Reference":
        reference_keys = st.session_state[session_id].get("reference", [])
        matching_keys = [
            key for key, value in all_tables.items()
            if value in reference_keys
            ]
    elif corpus == "Grouped":
        target_keys = st.session_state[session_id].get("target", [])
        if categories is not None and len(categories) > 0:
            matching_keys = [
                key for key, value in all_tables.items()
                if value in target_keys
                ]
        else:
            matching_keys = []
    elif corpus == "Keywords":
        keyword_keys = st.session_state[session_id].get("target", [])
        matching_keys = [
            key for key, value in all_tables.items()
            if value in keyword_keys and value.startswith('k')
            ]
    return matching_keys


def table_from_list(session_id: str,
                    corpus: str,
                    table_name: str,
                    categories: list[str] = None) -> pd.DataFrame:
    """
    Returns a table from session memory if the table_name matches a value
    in all_tables, using the keys from the specified corpus.

    Parameters
    ----------
    session_id : str
        The session identifier used to access the session state.
    corpus : str
        The corpus type, either "Target" or "Reference".
    table_name : str
        The name of the table to retrieve.

    Returns
    -------
    object
        The table from session memory if found, otherwise None.
    """
    corpus_tables = {
        'Tags Table: DocuScope': 'tt_ds',
        'Tags Table: Parts-of-Speech': 'tt_pos',
        'Tags Table: Parts-of-Speech Simplified': 'dtm_pos',
        'Document-Term Matrix: DocuScope': 'dtm_ds',
        'Document-Term Matrix: Parts-of-Speech': 'dtm_pos',
        'Document-Term Matrix: Parts-of-Speech Simplified': 'dtm_pos'
    }

    grouped_tables = {
        'Tags Table: DocuScope': 'dtm_ds',
        'Tags Table: Parts-of-Speech': 'dtm_pos',
        'Tags Table: Parts-of-Speech Simplified': 'dtm_pos',
        'Document-Term Matrix: DocuScope': 'dtm_ds',
        'Document-Term Matrix: Parts-of-Speech': 'dtm_pos',
        'Document-Term Matrix: Parts-of-Speech Simplified': 'dtm_pos'
    }

    keyness_tables = {
        'Keywords: DocuScope': 'kw_ds',
        'Keywords: Parts-of-Speech': 'kw_pos',
        'Keywords: Parts-of-Speech Simplified': 'kt_pos',
        'Keytags: DocuScope': 'kt_ds',
        'Keytags Table: Parts-of-Speech': 'kt_pos'
    }

    # Get the appropriate keys based on the corpus type
    if corpus == "Target":
        keys = st.session_state[session_id].get("target", [])
        matching_value = corpus_tables.get(table_name)
    elif corpus == "Reference":
        keys = st.session_state[session_id].get("reference", [])
        matching_value = corpus_tables.get(table_name)
    elif corpus == "Grouped":
        keys = st.session_state[session_id].get("target", [])
        matching_value = grouped_tables.get(table_name)
    elif corpus == "Keywords":
        keys = st.session_state[session_id].get("target", [])
        matching_value = keyness_tables.get(table_name)
    else:
        return None

    # If a matching value is found and it exists in the session keys,
    # return the table
    if matching_value and matching_value in keys:
        df = keys[matching_value]
        if corpus == "Target" or corpus == "Reference":
            if (
                table_name == "Tags Table: Parts-of-Speech Simplified"
            ):
                df = ds.tags_simplify(df)
            elif (
                table_name == "Document-Term Matrix: Parts-of-Speech" or
                table_name == "Document-Term Matrix: DocuScope"
            ):
                df = ds.dtm_weight(df, scheme="prop")
                df = df.unpivot(
                    pl.selectors.numeric(),
                    index="doc_id"
                    ).rename({"value": "RF", "variable": "Tag"})
            elif (
                table_name == "Document-Term Matrix: Parts-of-Speech Simplified"  # noqa: E501
            ):
                df = ds.dtm_simplify(df)
                df = ds.dtm_weight(df, scheme="prop")
                df = df.unpivot(
                    pl.selectors.numeric(),
                    index="doc_id"
                    ).rename({"value": "RF", "variable": "Tag"})

        elif corpus == "Grouped":
            if (
                table_name == "Tags Table: Parts-of-Speech" or
                table_name == "Tags Table: DocuScope"
            ):
                cat = pl.Series("Group", categories)
                if df.get_column("Group", default=None) is None:
                    df = df.with_columns(cat.alias("Group"))
                    df = df.select(
                        [pl.col('doc_id'),
                         pl.col('Group'),
                         pl.selectors.numeric()]
                         )
                df = analysis.tags_table_grouped(df)
            elif (
                table_name == "Tags Table: Parts-of-Speech Simplified"
            ):
                df = ds.dtm_simplify(df)
                cat = pl.Series("Group", categories)
                if df.get_column("Group", default=None) is None:
                    df = df.with_columns(cat.alias("Group"))
                    df = df.select(
                        [pl.col('doc_id'),
                         pl.col('Group'),
                         pl.selectors.numeric()]
                         )
                df = analysis.tags_table_grouped(df)
            elif (
                table_name == "Document-Term Matrix: Parts-of-Speech" or
                table_name == "Document-Term Matrix: DocuScope"
            ):
                cat = pl.Series("Group", categories)
                if df.get_column("Group", default=None) is None:
                    df = df.with_columns(cat.alias("Group"))
                    df = df.select(
                        [pl.col('doc_id'),
                         pl.col('Group'),
                         pl.selectors.numeric()]
                         )
                df = ds.dtm_weight(df, scheme="prop")
                df = df.unpivot(
                    pl.selectors.numeric(),
                    index=["doc_id", "Group"]
                    ).rename({"value": "RF", "variable": "Tag"})
            elif (
                table_name == "Document-Term Matrix: Parts-of-Speech Simplified"  # noqa: E501
            ):
                cat = pl.Series("Group", categories)
                if df.get_column("Group", default=None) is None:
                    df = df.with_columns(cat.alias("Group"))
                    df = df.select(
                        [pl.col('doc_id'),
                         pl.col('Group'),
                         pl.selectors.numeric()]
                         )
                df = analysis.dtm_simplify_grouped(df)
                df = ds.dtm_weight(df, scheme="prop")
                df = df.unpivot(
                    pl.selectors.numeric(),
                    index=["doc_id", "Group"]
                    ).rename({"value": "RF", "variable": "Tag"})

        return df

    # If no match is found, return None
    return None


def detect_intent(user_input: str) -> str:
    """
    Detects if the user's input is a plotting request.

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


def plotbot_code_generate_or_update(
    df: pd.DataFrame,
    user_request: str,
    plot_lib: str,
    schema: str,
    api_key: str,
    llm_params: dict,
    code_chunk: str = None
) -> str:
    valid_columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns.tolist()

    if code_chunk is None:
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
    - If using matplotlib, use the format 'fig, ax = plt.subplots()'.
    - Do not call 'fig.show()' or 'plt.show()'.
    - Use only columns that exist in the DataFrame. If the user mentions a column that does not exist, ignore it and use available columns instead.
    - If the request involves numeric data (like line charts, bar charts, histograms), use only numeric columns.
    - If the request involves categorical or non-numeric data (like pie charts or scatter plots with labels), you can use non-numeric columns.
    - Ensure the code is error-free and matches the DataFrame schema.
    - If you need to set axis labels or titles, use generic names if the user does not specify.
    - Include concise comments in the code to explain non-obvious steps or terminology (e.g., what a spine is or how to remove it).
    - Do not include explanations or markdown outside the code.

    Example for matplotlib:
    fig, ax = plt.subplots()
    ax.plot(df['col1'], df['col2'])
    # (do not include this comment in your output)

    Now, generate the code:
    """  # noqa: E501

    else:
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
    - If using matplotlib, use the format 'fig, ax = plt.subplots()'.
    - Do not call 'fig.show()' or 'plt.show()'.
    - Use only columns that exist in the DataFrame. If the user mentions a column that does not exist, ignore it and use available columns instead.
    - If the request involves numeric data (like line charts, bar charts, histograms), use only numeric columns.
    - If the request involves categorical or non-numeric data (like pie charts or scatter plots with labels), you can use non-numeric columns.
    - Ensure the code is error-free and matches the DataFrame schema.
    - If you need to set axis labels or titles, use generic names if the user does not specify.
    - Include concise comments in the code to explain non-obvious steps or terminology (e.g., what a spine is or how to remove it).
    - Do not include explanations or markdown outside the code.

    Example for matplotlib:
    fig, ax = plt.subplots()
    ax.plot(df['col1'], df['col2'])

    Now, update and output the code:
    """  # noqa: E501

    try:
        openai.api_key = api_key
        response = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            stream=True,
            temperature=llm_params["temperature"],
            max_tokens=llm_params["max_tokens"],
            top_p=llm_params["top_p"],
            frequency_penalty=llm_params["frequency_penalty"],
            presence_penalty=llm_params["presence_penalty"]
        )

        full_response = ""
        for chunk in response:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content:
                full_response += chunk_content

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

    except Exception as e:
        logger.error(f"Error in generating plot code: {e}")  # For developer logs
        return {
            "type": "error",
            "value": "Sorry, I couldn't generate your plot. Please try rephrasing your request."  # noqa: E501
        }


def is_code_safe(plot_code: str) -> bool:
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, plot_code, re.MULTILINE):
            logger.error(f"Unsafe pattern matched: {pattern} in code: {plot_code}")
            return False
    return True


def strip_imports(code: str) -> str:
    """Remove all import statements from the code."""
    return "\n".join(
        line for line in code.splitlines()
        if not re.match(r'^\s*import\s', line)
    )


def plotbot_code_execute(plot_code: str,
                         df: pd.DataFrame,
                         plot_lib: str) -> dict:
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
    if plot_lib == "matplotlib":
        allowed_globals["plt"] = plt
    elif plot_lib == "seaborn":
        allowed_globals["sns"] = sns
        allowed_globals["plt"] = plt
    elif plot_lib == "plotly.express":
        allowed_globals["px"] = px

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
                "value": "Sorry, no plot was generated. Please try a different request."
            }
    except SyntaxError as e:
        logger.error(f"Syntax error in plot code: {e}")
        return {
            "type": "error",
            "value": "Sorry, there was a problem with the plot code. Please try a different request."  # noqa: E501
        }
    except Exception as e:
        logger.error(f"Error in executing plot code: {e}")
        return {
            "type": "error",
            "value": "Sorry, something went wrong while generating your plot."
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
    # Ensure session state keys exist
    if "plotbot" not in st.session_state[session_id]:
        st.session_state[session_id]["plotbot"] = []
    if "plot_intent" not in st.session_state[session_id]:
        st.session_state[session_id]["plot_intent"] = False

    if cache_mode:
        add_message(user_id=st.user.email,
                    session_id=session_id,
                    assistant_id=0,
                    role="user",
                    message_idx=prompt_position,
                    message=user_input)

    intent = detect_intent(user_input)
    schema = df.dtypes.to_string()

    if intent == "none":
        response = (
            ":grey_question: Please enter a request for a plot or chart."
        )
        st.session_state[session_id]["plotbot"].append(
            {"role": "assistant", "type": "error", "value": response}
        )
        prune_message_thread(session_id)
        return

    if intent == "plot":
        st.session_state[session_id]["plot_intent"] = True

        if df is not None:
            # Use unified code generation/update function
            cache_dict = st.session_state[session_id].setdefault("plotbot_cache", {})

            cache_key = make_plotbot_cache_key(user_input, df, plot_lib, code_chunk)

            # Check for cached code and plot
            cached = cache_dict.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for key: {cache_key}")
                plot_code = cached.get("code")
                plot_fig = cached.get("plot")
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

            # Standardized error handling
            if plot_code is None or (isinstance(plot_code, dict) and plot_code.get("type") == "error"):  # noqa: E501
                error_message = plot_code.get("value") if isinstance(plot_code, dict) else (
                    "Sorry, I couldn't generate your plot. Please try rephrasing your request."  # noqa: E501
                )
                st.session_state[session_id]["plotbot"].append(
                    {"role": "assistant", "type": "error", "value": error_message}
                )
                prune_message_thread(session_id)
                return

            plot_fig = plotbot_code_execute(plot_code=plot_code, plot_lib=plot_lib, df=df)
            # Cache the result
            if (
                not (isinstance(plot_code, dict) and plot_code.get("type") == "error") and
                not (isinstance(plot_fig, dict) and plot_fig.get("type") == "error")
            ):
                cache_dict[cache_key] = {"code": plot_code, "plot": plot_fig}

            if not isinstance(plot_fig, dict):
                plot_fig = {
                    "type": "error",
                    "value": "Sorry, something went wrong while generating your plot."
                }

            if plot_fig.get("type") == "error":
                st.session_state[session_id]["plotbot"].append(
                    {"role": "assistant", "type": "error", "value": plot_fig.get("value")}
                )
                prune_message_thread(session_id)
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
            st.session_state[session_id]["plotbot"].append(
                {"role": "assistant", "type": "code", "value": plot_code}
            )
            prune_message_thread(session_id)

            if plot_fig.get("type") == "plot":
                st.session_state[session_id]["plotbot"].append(
                    {"role": "assistant", "type": "plot", "value": plot_fig["value"]}
                )
                prune_message_thread(session_id)
            else:
                error_message = (
                    "No plot was generated. As a plotbot, I can only execute specific types of requests."  # noqa: E501
                    "For more complex tasks, you might want to try AI-assisted analysis."
                )
                st.session_state[session_id]["plotbot"].append(
                    {"role": "assistant", "type": "error", "value": error_message}
                )
                prune_message_thread(session_id)
        else:
            error_message = "No plot was generated. Please check the code."
            st.session_state[session_id]["plotbot"].append(
                {"role": "assistant", "type": "error", "value": error_message}
            )
            prune_message_thread(session_id)
    else:
        response = (
            ":warning: I am unable to assist with that request.\n"
            "I'm a plotbot, not a chat bot.\n"
            "Try asking me to plot something related to the data."
        )
        st.session_state[session_id]["plotbot"].append(
            {"role": "assistant", "type": "error", "value": response}
        )
        prune_message_thread(session_id)


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
    Appends results to session state for Streamlit display.

    - For plot requests, monkeypatches plt.Figure.savefig to capture plot images in-memory,
      preventing cross-user file collisions and ensuring safe, per-session plot handling.
    - For statistical/text/table requests, returns results as string/table.
    - Appends extra instructions to the prompt for plot requests to avoid plt.show() and file paths.
    """  # noqa: E501
    if "pandasai" not in st.session_state[session_id]:
        st.session_state[session_id]["pandasai"] = []
    response = st.session_state[session_id]["pandasai"]

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
        "max_retries": 3,
        "save_charts": False
    })

    # Detect if the prompt is likely a plot request
    intent = detect_intent(prompt)
    plot_prompt_append = (
        "\n- Do not call plt.show(), fig.show(), or display()."
        "\n- Do not save plots to disk or return a file path."
        "\n- Only create the figure and assign it to a variable named 'fig'."
        if intent == "plot" else ""
    )
    full_prompt = prompt + plot_prompt_append

    dfs = pai.DataFrame(df)
    response.append({"role": "user", "type": "string", "value": prompt})
    prune_message_thread(session_id)

    # --- Monkeypatch plt.Figure.savefig to capture image bytes ---
    _original_savefig = plt.Figure.savefig
    _last_img_bytes = {}

    def savefig_to_buffer(self, fname, *args, **kwargs):
        # If saving to a temp_chart path, redirect to buffer
        if isinstance(fname, str) and "temp_chart" in fname:
            buf = io.BytesIO()
            _original_savefig(self, buf, format="png", *args, **kwargs)
            buf.seek(0)
            _last_img_bytes["img"] = buf.getvalue()
            buf.close()
        else:
            _original_savefig(self, fname, *args, **kwargs)

    plt.Figure.savefig = savefig_to_buffer

    try:
        result = dfs.chat(full_prompt)

        # Restore savefig immediately after use
        plt.Figure.savefig = _original_savefig

        # 1. If result is a DataFrame
        if isinstance(result, pd.DataFrame):
            value = result.to_dict()
            response.append({"role": "assistant", "type": "table", "value": value})
            prune_message_thread(session_id)

        # 2. If result is a dict (e.g., pandasai chart)
        elif isinstance(result, dict) and "type" in result and "value" in result:
            response.append({"role": "assistant", "type": result["type"], "value": result["value"]})  # noqa: E501
            prune_message_thread(session_id)

        # 3. If we captured image bytes via monkeypatch, use them
        elif "img" in _last_img_bytes:
            response.append({"role": "assistant", "type": "plot", "value": _last_img_bytes["img"]})  # noqa: E501
            prune_message_thread(session_id)

        # 4. If result is a string that looks like a file path, fallback to memory
        elif isinstance(result, str) and re.match(r"^exports/charts/.*\.png$", result):
            # Try to capture the figure from memory
            if plt.get_fignums():
                fig = plt.gcf()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches='tight')
                buf.seek(0)
                img_bytes = buf.getvalue()
                buf.close()
                plt.close(fig)
                response.append({"role": "assistant", "type": "plot", "value": img_bytes})
                prune_message_thread(session_id)
            else:
                response.append({"role": "assistant", "type": "string", "value": result})
                prune_message_thread(session_id)

        # 5. Fallback: Try to capture the figure from memory
        else:
            if plt.get_fignums():
                fig = plt.gcf()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches='tight')
                buf.seek(0)
                img_bytes = buf.getvalue()
                buf.close()
                plt.close(fig)
                response.append({"role": "assistant", "type": "plot", "value": img_bytes})
                prune_message_thread(session_id)
            else:
                # Log unhandled result type for debugging
                logger.warning(f"Unhandled pandasai result type: {type(result)} - {result}")
                response.append({"role": "assistant", "type": "string", "value": str(result)})  # noqa: E501
                prune_message_thread(session_id)

    except MaliciousQueryError:
        plt.Figure.savefig = _original_savefig
        error = (
            ":confused: Sorry, your request could not be processed. "
            "It may be too complex or reference restricted operations."
        )
        logger.error("MaliciousQueryError in pandasai query.")
        response.append({"role": "assistant", "type": "error", "value": error})
        prune_message_thread(session_id)
    except NoResultFoundError:
        plt.Figure.savefig = _original_savefig
        error = (
            ":confused: Sorry, I couldn't find a result for your request. "
            "Try rephrasing or checking your column names."
        )
        logger.error("NoResultFoundError in pandasai query.")
        response.append({"role": "assistant", "type": "error", "value": error})
        prune_message_thread(session_id)
    except Exception as e:
        plt.Figure.savefig = _original_savefig
        error = (
            ":confused: Sorry, something went wrong. "
            "Try rephrasing your request or using a different data structure."
        )
        logger.error(f"Exception in pandasai query: {e}")
        response.append({"role": "assistant", "type": "error", "value": error})
        prune_message_thread(session_id)
    finally:
        plt.Figure.savefig = _original_savefig


def previous_code_chunk(messages: list[dict]):
    try:
        messages_with_code = list(
            filter(lambda elem: elem.get("type") == "code", messages)
        )
        if len(messages_with_code) > 0:
            last_message = messages_with_code[-1]
            code_chunk = last_message.get("value")
            return code_chunk
    except KeyError:
        return None


def prune_message_thread(session_id: str, thread_key: str, max_length: int = 20):
    """
    Prune the message thread (e.g., 'plotbot', 'pandasai')
    to the most recent max_length messages.
    Keeps the initial user prompt if possible.
    """
    thread = st.session_state[session_id][thread_key]
    if len(thread) > max_length:
        # Optionally, always keep the first user message
        first_user_idx = next((i for i, m in enumerate(thread) if m["role"] == "user"), 0)
        # Keep the first user message and the last (max_length-1) messages
        st.session_state[session_id][thread_key] = (
            [thread[first_user_idx]] + thread[-(max_length-1):]
            if first_user_idx < len(thread) else thread[-max_length:]
        )


def generate_plot_key(session_id, plot_id):
    plot_key = session_id + "-" + str(plot_id)
    return plot_key


def make_plotbot_cache_key(user_input, df, plot_lib, code_chunk=None):
    # Use schema and shape, not full data, for hashing (fast & safe)
    schema = str(df.dtypes.to_dict())
    shape = str(df.shape)
    key_data = {
        "user_input": user_input,
        "schema": schema,
        "shape": shape,
        "plot_lib": plot_lib,
        "code_chunk": code_chunk or "",
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()


def fig_to_svg(figure: plt.Figure,
               plot_lib: str = "matplotlib",
               is_array: bool = False) -> str:
    """
    Converts a Matplotlib figure to an SVG string.

    Parameters
    ----------
    figure : plt.Figure
        The figure to convert.

    Returns
    -------
    str
        The SVG string representation of the figure.
    """
    if is_array:
        try:
            # Create a Matplotlib figure and render the image array
            fig, ax = plt.subplots()
            ax.imshow(figure)  # 'figure' is the image array
            ax.axis('off')  # Turn off axes for a cleaner SVG

            # Save the figure to an SVG string
            buf = io.BytesIO()
            fig.savefig(buf, format='svg', bbox_inches='tight')
            buf.seek(0)
            svg_str = buf.getvalue().decode('utf-8')
            buf.close()

            # Close the Matplotlib figure to free memory
            plt.close(fig)
            return svg_str
        except Exception as e:
            logger.error(f"Error in generating plot code: {e}")
            return None
    if plot_lib == "plotly.express":
        try:
            figure.update_layout(template="plotly_white")
            img_bytes = figure.to_image(format="svg")
            svg_str = img_bytes.decode('utf-8')
            return svg_str
        except Exception as e:
            logger.error(f"Error in generating plot code: {e}")
            return None
    else:
        try:
            buf = io.BytesIO()
            figure.savefig(buf, format='svg', bbox_inches='tight')
            buf.seek(0)
            svg_str = buf.getvalue().decode('utf-8')
            buf.close()
            return svg_str
        except Exception as e:
            logger.error(f"Error in generating plot code: {e}")
            return None


def fig_to_array(fig: plt.Figure,
                 plot_lib: str = "matplotlib",
                 max_width: int = 800,
                 max_height: int = 800,
                 max_bytes: int = 1048487) -> tuple:
    """
    Converts a figure to a downsized NumPy array and
    ensures it fits within Firestore limits.

    Parameters
    ----------
    fig : plt.Figure
        The figure to convert.
    plot_lib : str
        The plotting library used ("matplotlib" or "plotly.express").
    max_width : int
        The maximum width of the downsized image.
    max_height : int
        The maximum height of the downsized image.
    max_bytes : int
        The maximum size of the serialized array in bytes.

    Returns
    -------
    tuple
        A tuple containing:
        - plot_array_serialized: str
        - plot_array_shape: tuple
        - plot_array_dtype: str
    """
    def resize_and_serialize(img: Image.Image, max_bytes: int) -> tuple:
        """
        Resizes and compresses the image until it fits within the size limit.

        Parameters
        ----------
        img : Image.Image
            The PIL Image to resize and compress.
        max_bytes : int
            The maximum size of the serialized image in bytes.

        Returns
        -------
        tuple
            A tuple containing:
            - plot_array_serialized: str
            - plot_array_shape: tuple
            - plot_array_dtype: str
        """
        quality = 95  # Start with high quality
        width, height = img.size

        while True:
            # Convert the image to RGB mode if it has an alpha channel
            if img.mode == "RGBA":
                img = img.convert("RGB")

            # Save the image to a buffer with the current quality
            buf = io.BytesIO()
            img.save(buf, format="PNG", quality=quality)
            buf.seek(0)

            # Convert the buffer to a NumPy array
            img_array = np.asarray(Image.open(buf), dtype=np.uint8)

            # Serialize the array
            plot_array_serialized = base64.b64encode(img_array.tobytes()).decode('utf-8')  # noqa: E501
            plot_array_shape = img_array.shape
            plot_array_dtype = str(img_array.dtype)

            # Check the size of the serialized array
            if (
                len(plot_array_serialized) <= max_bytes or
                (quality <= 10 and width <= 100 and height <= 100)
            ):
                buf.close()
                return plot_array_serialized, plot_array_shape, plot_array_dtype  # noqa: E501

            # Reduce quality to further compress the image
            if len(plot_array_serialized) > max_bytes:
                quality -= 5

            # If quality is too low, reduce dimensions
            if quality <= 10:
                width = int(width * 0.9)
                height = int(height * 0.9)
                img = img.resize((width, height), Image.Resampling.LANCZOS)
                quality = 95  # Reset quality after resizing

    if plot_lib == "plotly.express":
        try:
            # Render the Plotly figure as a static image
            img_bytes = fig.to_image(
                format="png",
                width=max_width,
                height=max_height
                )

            # Open the image as a PIL Image
            img = Image.open(io.BytesIO(img_bytes))

            # Resize and serialize the image
            return resize_and_serialize(img, max_bytes)
        except Exception as e:
            logger.error(f"Error in generating plot code: {e}")
            return None, None, None

    elif plot_lib == "matplotlib":
        try:
            # Render the Matplotlib figure to a buffer
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches='tight')
            buf.seek(0)

            # Open the buffer as a PIL Image
            img = Image.open(buf)

            # Resize and serialize the image
            result = resize_and_serialize(img, max_bytes)

            # Close the buffer to free memory
            buf.close()
            return result
        except Exception as e:
            logger.error(f"Error in generating plot code: {e}")
            return None, None, None
    else:
        logger.error("Error in generating plot code: Unknown plot_lib or unexpected error in fig_to_array.")  # noqa: E501
        return None, None, None
