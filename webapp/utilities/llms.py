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
import io
import pathlib
import openai
import os
import re

import docuscospacy as ds
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

from PIL import Image
from pandasai.exceptions import MaliciousQueryError, NoResultFoundError
from pandasai_openai import OpenAI
import pandasai as pai

from utilities import analysis
from utilities.cache import add_message, add_plot
from utilities.handlers import import_options_general

# set paths
HERE = pathlib.Path(__file__).parents[1].resolve()
OPTIONS = str(HERE.joinpath("options.toml"))

_options = import_options_general(OPTIONS)
LLM_PARAMS = _options['llm']['llm_parameters']
LLM_MODEL = _options['llm']['llm_model']
DESKTOP = _options['global']['desktop_mode']
CACHE = _options['cache']['cache_mode']


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


def detect_intent(input: str) -> str:
    """
    Detects the user's intent based on the input string.

    Parameters
    ----------
    input : str
        The input string from the user.

    Returns
    -------
    str
        The detected intent, which can be "plot", "cancel_plot", or "chat".
    """
    plot_keywords = [
        "plot", "chart", "graph", "draw", "visualize", "sketch", "illustrate",
        "render", "depict", "map", "trace", "diagram",
        "visual representation", "graphical representation",
        "represent data", "graphically show", "graphical illustration",

        "scatter", "bar", "histogram", "pie", "line",
        "area", "heatmap", "box", "boxplot", "violinplot",
        "scatterplot", "bubblechart", "barchart"
        "density", "densityplot", "hexbin", "error",
        "stacked", "polar", "donut", "funnel", "distribution", "point",
        "joint", "pair", "categorical", "swarm", "fit"

        "x-axis", "y-axis", "z-axis", "color", "hue", "size", "shape",
        "xaxis", "yaxis", "bars", "lines", "points", "markers", "labels",
        "label", "legend", "title", "axis", "grid", "background",
        "foreground", "font", "scale", "range", "ticks", "marks",
        "titles", "limits", "range", "grid", "background", "foreground",
        "font", "column", "row", "subplot", "facet", "subplot", "axes",
        "spine", "spines", "border", "tick", "ticks", "ticklabels",
    ]

    if any(keyword in input.lower() for keyword in plot_keywords):
        return "plot"
    if not input or not isinstance(input, str):
        return "chat"
    else:
        return "chat"


def pandabot_user_query(df: pd.DataFrame,
                        api_key: str,
                        prompt: str,
                        session_id: str,
                        prompt_position: int = 1,
                        cache_mode: bool = False) -> None:

    if cache_mode:
        add_message(user_id=st.user.email,
                    session_id=session_id,
                    assistant_id=1,
                    role="user",
                    message_idx=prompt_position,
                    message=prompt)

    model = OpenAI(api_token=api_key)

    pai.config.set({
        "llm": model,
        "save_logs": False,
        "verbose": False,
        "max_retries": 3
    })

    dfs = pai.DataFrame(df)
    # Check if the session state exists
    if "pandasai" not in st.session_state[session_id]:
        st.session_state[session_id]["pandasai"] = []

    response = st.session_state[session_id]["pandasai"]
    response.append(
        {"role": "user", "type": "string", "value": prompt}
        )

    try:
        data = dfs.chat(prompt).to_dict()

        if "type" in data and "value" in data:
            if data["type"] != "chart":
                response.append(
                    {"role": "assistant",
                     "type": data["type"],
                     "value": data["value"]}
                    )
            else:
                pass
        # Extract the plot outputs
        # As plots are not always stored in "type" and "value"
        plot_outputs = re.findall(r'exports\S+png', str(data))
        plot_outputs = list(set(plot_outputs))
        if plot_outputs:
            for path in plot_outputs:
                try:
                    if (
                        os.path.exists(path) and
                        os.path.isfile(path)
                    ):
                        im = plt.imread(path)
                        os.remove(path)
                        response.append(
                            {"role": "assistant", "type": "plot", "value": im}
                            )

                        if cache_mode:
                            # Convert the image to SVG
                            svg_str = fig_to_svg(figure=im,
                                                 plot_lib="matplotlib",
                                                 is_array=True)
                            add_plot(user_id=st.user.email,
                                     session_id=session_id,
                                     assistant_id=1,
                                     message_idx=prompt_position,
                                     plot_library='matplotlib',
                                     plot_svg=svg_str)
                    else:
                        pass
                except Exception:
                    pass

    except MaliciousQueryError:
        error = """:confused: Well, that's embarassing.
        I couldn't respond to your request. It could just be that I'm overwhelmed.
        But it might also be that I don't understand your request.
        You might want to try rephrasing it or using a different data structure.
        """  # noqa: E501
        response.append(
            {"role": "assistant", "type": "error", "value": error}
            )
    except NoResultFoundError:
        error = """:confused: Well, that's embarassing.
        I couldn't respond to your request. It could just be that I'm overwhelmed.
        But it might also be that I don't understand your request.
        You might want to try rephrasing it or using a different data structure.
        """  # noqa: E501
        response.append(
            {"role": "assistant", "type": "error", "value": error}
            )
    except Exception:
        error = """:confused: Well, that's embarassing.
        I couldn't respond to your request. It could just be that I'm overwhelmed.
        But it might also be that I don't understand your request.
        You might want to try rephrasing it or using a different data structure.
        """  # noqa: E501
        response.append(
            {"role": "assistant", "type": "error", "value": error}
            )


def plotbot_code_generate(df: pd.DataFrame,
                          user_request: str,
                          plot_lib: str,
                          schema: str,
                          api_key: str,
                          llm_params: dict) -> str:

    valid_columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns.tolist()

    prompt = f"""
    You are a Python plotting assistant.
    The user has requested to create a plot.
    Here is the data schema:
    {schema}

    The available columns in the DataFrame are:
    {', '.join(valid_columns)}.
    Numeric columns are: {', '.join(numeric_columns)}.
    Non-numeric columns are: {', '.join(non_numeric_columns)}.

    Based on the user request: '{user_request}', generate appropriate code for plotting using the following steps:
    - Use {plot_lib} for plotting.
    - If using matplotlib use the format 'fig, ax = plt.subplots()'.
    - Do not import libraries, only write the plotting code.
    - Do not show the plot using 'fig.show()'.
    - The DataFrame is called 'df'.
    - If the request involves numeric data (like line charts, bar charts, histograms), ensure you only use numeric columns ({', '.join(numeric_columns)}).
    - If the request involves categorical or non-numeric data (like pie charts or scatter plots with labels), you can use non-numeric columns ({', '.join(non_numeric_columns)}).
    - If the user mentions a column that does not exist, ignore it and use available columns instead.
    - Ensure the code is error-free and matches the DataFrame schema.

    Please generate the plot code in Python:
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
        error_message = {
            "type": "error",
            "value": f"Error in generating plot code: {e}"
            }
        return error_message


def plotbot_code_update(df: pd.DataFrame,
                        user_request: str,
                        code_chunk: str,
                        plot_lib: str,
                        schema: str,
                        api_key: str,
                        llm_params: dict) -> str:

    valid_columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns.tolist()

    prompt = f"""
    You are a Python plotting assistant.
    The user has requested to update code that generates plot.
    Here is the data schema:
    {schema}

    The available columns in the DataFrame are:
    {', '.join(valid_columns)}.
    Numeric columns are: {', '.join(numeric_columns)}.
    Non-numeric columns are: {', '.join(non_numeric_columns)}.

    Based on the user request: '{user_request}',
    and the current code {code_chunk},
    update the code to generate the plot using the following steps:
    - Use {plot_lib} for plotting.
    - If using matplotlib use the format 'fig, ax = plt.subplots()'.
    - Do not import libraries, only write the plotting code.
    - Do not show the plot using 'fig.show()'.
    - The DataFrame is called 'df'.
    - If the request involves numeric data (like line charts, bar charts, histograms), ensure you only use numeric columns ({', '.join(numeric_columns)}).
    - If the request involves categorical or non-numeric data (like pie charts or scatter plots with labels), you can use non-numeric columns ({', '.join(non_numeric_columns)}).
    - If the user mentions a column that does not exist, ignore it and use available columns instead.
    - Ensure the code is error-free and matches the DataFrame schema.

    Please generate the plot code in Python:
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
        error_message = {
            "type": "error",
            "value": f"Error in generating plot code: {e}"
            }
        return error_message


def clean_df_plotting(df):
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception as e:
            st.warning(f"Could not convert column {col} to numeric: {e}")
    return df.dropna(axis=1, how="any")


def plotbot_code_execute(plot_code: str,
                         df: pd.DataFrame,
                         plot_lib: str) -> dict:
    exec_locals = {}
    try:
        # df = clean_df_plotting(df)
        if plot_lib == "matplotlib":
            exec(plot_code, {'df': df, 'plt': plt}, exec_locals)
        elif plot_lib == "seaborn":
            exec(plot_code, {'df': df, 'sns': sns, 'plt': plt}, exec_locals)
        elif plot_lib == "plotly.express":
            exec(plot_code, {'df': df, 'px': px}, exec_locals)
        if "fig" in exec_locals:
            fig = exec_locals["fig"]
            return fig
        else:
            return None
    except Exception as e:
        error_message = {
            "type": "error",
            "value": f"Error in executing plot code: {e}"
            }
        return error_message


def plotbot_user_query(session_id: str,
                       df: pd.DataFrame,
                       plot_lib: str,
                       user_input: str,
                       api_key: str,
                       llm_params: dict,
                       code_chunk=None,
                       prompt_position: int = 1,
                       cache_mode: bool = False) -> None:

    if cache_mode:
        add_message(user_id=st.user.email,
                    session_id=session_id,
                    assistant_id=0,
                    role="user",
                    message_idx=prompt_position,
                    message=user_input)

    intent = detect_intent(user_input)
    schema = df.dtypes.to_string()

    if intent == "plot":
        st.session_state[session_id]["plot_intent"] = True

        if df is not None:
            if code_chunk is None:
                plot_code = plotbot_code_generate(df=df,
                                                  user_request=user_input,
                                                  plot_lib=plot_lib,
                                                  schema=schema,
                                                  api_key=api_key,
                                                  llm_params=llm_params)
            elif code_chunk is not None:
                plot_code = plotbot_code_update(df=df,
                                                user_request=user_input,
                                                code_chunk=code_chunk,
                                                plot_lib=plot_lib,
                                                schema=schema,
                                                api_key=api_key,
                                                llm_params=llm_params)

            if plot_code is None:
                error_message = """My apologies...
                I had trouble generating the plot code.
                Try a different request.
                """
                st.session_state[session_id]["plotbot"].append(
                    {"role": "assistant",
                     "type": "string",
                     "value": error_message}
                    )

            elif plot_code is dict and plot_code.get("type") == "error":
                error_message = plot_code.get("value")
                st.session_state[session_id]["plotbot"].append(
                    {"role": "assistant",
                     "type": "string",
                     "value": error_message}
                    )

            plot_fig = plotbot_code_execute(plot_code=plot_code,
                                            plot_lib=plot_lib,
                                            df=df)

            if cache_mode:
                svg_str = fig_to_svg(figure=plot_fig, plot_lib=plot_lib)

                add_plot(user_id=st.user.email,
                         session_id=session_id,
                         assistant_id=0,
                         message_idx=prompt_position,
                         plot_library=plot_lib,
                         plot_svg=svg_str)

            st.session_state[session_id]["plotbot"].append(
                {"role": "assistant", "type": "code", "value": plot_code}
                )

            if plot_fig:
                st.session_state[session_id]["plotbot"].append(
                    {"role": "assistant", "type": "plot", "value": plot_fig}
                    )

            else:
                error_message = """
                No plot was generated.
                As a plotbot, I can only execute specific types of requests.
                For more complex tasks,
                you might want to try AI-assisted analysis.
                """
                st.session_state[session_id]["plotbot"].append(
                    {"role": "assistant",
                     "type": "string",
                     "value": error_message}
                    )
        else:
            error_message = "No plot was generated. Please check the code."
            st.session_state[session_id]["plotbot"].append(
                    {"role": "assistant",
                     "type": "string",
                     "value": error_message}
                    )
    else:
        response = """
        :warning: I am unable to assist with that request.
        I'm a plotbot, not a chat bot.
        Try asking me to plot something related to the data.
        """
        if response:
            st.session_state[session_id]["plotbot"].append(
                {"role": "assistant", "type": "string", "value": response}
                )


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


def generate_plot_key(session_id, plot_id):
    plot_key = session_id + "-" + str(plot_id)
    return plot_key


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
            print(f"Error converting array to SVG: {e}")
            return None
    if plot_lib == "plotly.express":
        try:
            figure.update_layout(template="plotly_white")
            img_bytes = figure.to_image(format="svg")
            svg_str = img_bytes.decode('utf-8')
            return svg_str
        except Exception as e:
            print(f"Error converting Plotly figure to SVG: {e}")
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
            print(f"Error converting Matplotlib figure to SVG: {e}")
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
            print(f"Error converting Plotly figure to array: {e}")
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
            print(f"Error converting Matplotlib figure to array: {e}")
            return None, None, None

    else:
        print(f"Unsupported plotting library: {plot_lib}")
        return None, None, None
