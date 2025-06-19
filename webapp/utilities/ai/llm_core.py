

"""
Core LLM utilities for AI-powered corpus analysis.

This module provides shared functionality for integrating large language models
into the corpus analysis workflow, including API validation, settings management,
and common utilities used by both plotbot and pandabot assistants.
"""
import openai
import docuscospacy as ds
import pandas as pd
import polars as pl
import streamlit as st

from loguru import logger

from webapp.config.session_keys import SessionKeys
from webapp.utilities.state import persist
from webapp.utilities.storage import get_query_count
from webapp.utilities.analysis import tags_table_grouped, dtm_simplify_grouped
from webapp.utilities.configuration import get_ai_configuration
from webapp.utilities.configuration import setup_ai_logging

_options, DESKTOP, CACHE, LLM_MODEL, LLM_PARAMS, QUOTA = get_ai_configuration()

setup_ai_logging()


def print_settings(dct: dict) -> str:
    """
    Print settings dictionary in a formatted way.

    Parameters
    ----------
    dct : dict
        Dictionary to print

    Returns
    -------
    str
        Formatted string representation
    """
    output = []
    for key, value in dct.items():
        output.append(f"**{key}**: {value}")
    return "\\n".join(output)


def is_openai_key_valid(api_key: str) -> bool:
    """
    Test if the provided OpenAI API key is valid.

    Parameters
    ----------
    api_key : str
        The API key to test

    Returns
    -------
    bool
        True if the key is valid, False otherwise
    """
    try:
        client = openai.OpenAI(api_key=api_key)
        # Make a minimal API call to test the key
        client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1
        )
        return True
    except Exception as e:
        logger.error(f"OpenAI API key validation failed: {e}")
        return False


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
    corpus : str
        The corpus type ("Target", "Reference", "Grouped", or "Keywords").
    categories : list[str], optional
        List of categories for grouped corpus analysis.

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
    else:
        matching_keys = []

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
        The corpus type, either "Target", "Reference", "Grouped", or "Keywords".
    table_name : str
        The name of the table to retrieve.
    categories : list[str], optional
        List of categories for grouped corpus analysis.

    Returns
    -------
    pd.DataFrame
        The table from session memory if found, otherwise empty DataFrame.
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
        return pd.DataFrame()

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
                table_name == "Document-Term Matrix: Parts-of-Speech Simplified"
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
                df = tags_table_grouped(df)
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
                df = tags_table_grouped(df)
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
                table_name == "Document-Term Matrix: Parts-of-Speech Simplified"
            ):
                cat = pl.Series("Group", categories)
                if df.get_column("Group", default=None) is None:
                    df = df.with_columns(cat.alias("Group"))
                    df = df.select(
                        [pl.col('doc_id'),
                         pl.col('Group'),
                         pl.selectors.numeric()]
                         )
                df = dtm_simplify_grouped(df)
                df = ds.dtm_weight(df, scheme="prop")
                df = df.unpivot(
                    pl.selectors.numeric(),
                    index=["doc_id", "Group"]
                    ).rename({"value": "RF", "variable": "Tag"})

        return df

    # If no match is found, return empty DataFrame
    return pd.DataFrame()


def previous_code_chunk(messages: list[dict]):
    """
    Extract the previous code chunk from conversation history.

    Parameters
    ----------
    messages : list[dict]
        List of conversation messages

    Returns
    -------
    str | None
        Previous code chunk or None if not found
    """
    try:
        # Find the last code message
        messages_with_code = [msg for msg in messages if msg.get("type") == "code"]
        if messages_with_code:
            last_message = messages_with_code[-1]
            code_chunk = last_message.get("value")
            return code_chunk
    except KeyError:
        return None


def setup_ai_session_state(user_session_id: str, bot_type: str) -> None:
    """
    Initialize session state for AI assistants.

    Parameters
    ----------
    user_session_id : str
        User session identifier
    bot_type : str
        Type of bot ('plotbot' or 'pandasai')
    """
    # Initialize chat history
    if bot_type not in st.session_state[user_session_id]:
        st.session_state[user_session_id][bot_type] = []

    # Initialize user prompt count
    prompt_count_key = f"{bot_type}_user_prompt_count"
    if prompt_count_key not in st.session_state[user_session_id]:
        st.session_state[user_session_id][prompt_count_key] = 0

    # Initialize user key
    if SessionKeys.AI_USER_KEY not in st.session_state[user_session_id]:
        st.session_state[user_session_id][SessionKeys.AI_USER_KEY] = None


def get_api_key(
    user_session_id: str,
    desktop_mode: bool,
    cache_enabled: bool,
    quota: int
) -> str | None:
    """
    Get the appropriate API key based on configuration and quota.

    Parameters
    ----------
    user_session_id : str
        User session identifier
    desktop_mode : bool
        Whether running in desktop mode
    cache_enabled : bool
        Whether caching is enabled
    quota : int
        Daily quota limit

    Returns
    -------
    str | None
        API key to use or None if none available
    """
    try:
        # Try to get community key if not in desktop mode
        community_key = None
        if not desktop_mode:
            try:
                community_key = st.secrets["openai"]["api_key"]
            except Exception:
                logger.debug("Community API key not available")
                community_key = None

        # Check quota if caching is enabled
        if cache_enabled and community_key:
            try:
                daily_tokens = get_query_count(st.user.email)
                if daily_tokens >= quota:
                    logger.debug("Daily quota exceeded, community key disabled")
                    community_key = None
            except Exception as e:
                logger.error(f"Error checking quota: {e}")

        # Return community key if available, otherwise user key
        if community_key is not None:
            return community_key
        else:
            return st.session_state[user_session_id][SessionKeys.AI_USER_KEY]

    except Exception as e:
        logger.error(f"Error getting API key: {e}")
        return None


def render_api_key_input(user_session_id: str) -> None:
    """
    Render API key input interface with validation.

    Parameters
    ----------
    user_session_id : str
        User session identifier
    """
    st.markdown(
        body="### OpenAI API Key",
        help=(
            "To use the AI assistant, you need an OpenAI API key. "
            "You can get one from the "
            "[OpenAI website](https://platform.openai.com/api-keys)."
        )
    )

    user_api_key = st.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        placeholder="sk-...",
        help=(
            "Your API key is used to make requests to OpenAI's servers. "
            "It is not stored permanently and will be cleared when you close the app."
        )
    )

    # Add validation button for explicit user control
    if user_api_key:
        if st.button(
            "Validate API Key",
            type="primary",
            help="Click to test your API key with OpenAI's servers"
        ):
            with st.spinner("Validating API key..."):
                if is_openai_key_valid(user_api_key):
                    st.success(
                        ":material/check_circle: Valid API key! You can now use the AI assistant.",  # noqa: E501
                        icon=":material/check:"
                    )
                    st.session_state[user_session_id][
                        SessionKeys.AI_USER_KEY
                    ] = user_api_key
                    st.rerun()
                else:
                    st.error(
                        ":material/warning: Invalid API key. Please check your key and try again.",  # noqa: E501
                        icon=":material/error:"
                    )
    else:
        st.info(
            "Please enter your OpenAI API key above to get started.",
            icon=":material/key:"
        )


def render_data_selection_interface(
    user_session_id: str,
    session: dict,
    bot_prefix: str,
    page_stem: str,
    clear_function: callable,
    metadata_target: dict = None
) -> tuple[str, str, pl.DataFrame | None]:
    """
    Render data selection interface for AI assistants.

    Parameters
    ----------
    user_session_id : str
        User session identifier
    session : dict
        Session state dictionary
    bot_prefix : str
        Prefix for session keys (e.g., 'plotbot', 'pandasai')
    page_stem : str
        Page stem for state persistence
    clear_function : callable
        Function to clear bot state
    metadata_target : dict, optional
        Target corpus metadata
        
    Returns
    -------
    tuple[str, str, pl.DataFrame | None]
        Selected corpus, query, and dataframe
    """
    try:
        st.markdown(
            body="### Data Selection",
            help=(
                "* To make **Reference** data available, "
                "you must process a reference corpus\n\n"
                "* To make **Keywords** data available, "
                "you must create keywords tables either by "
                "**Comparing Corpora** or **Comparing Corpus Parts**\n\n"
                "* To make **Group** data available you first have "
                "to process metadata (available in Manage Corpus Data)."
            )
        )

        # Corpus selection
        corpus_key = f"{bot_prefix}_corpus"
        selected_corpus = st.radio(
            "Select corpus:",
            ("Target", "Reference", "Keywords", "Grouped"),
            key=persist(corpus_key, page_stem, user_session_id),
            on_change=clear_function,
            args=(user_session_id,),
            index=0,
            horizontal=True
        )

        # Get groups if metadata is available
        groups = []
        if session.get('has_meta', [False])[0] and metadata_target:
            groups = metadata_target.get('doccats', [{}])[0].get('cats', [])

        # Query selection
        query_key = f"{bot_prefix}_query"
        data_label = ("Select data to analyze:" if bot_prefix == 'pandasai'
                      else "Select data to plot:")
        selected_query = st.selectbox(
            data_label,
            tables_to_list(
                session_id=user_session_id,
                corpus=selected_corpus,
                categories=groups
            ),
            key=persist(query_key, page_stem, user_session_id),
            on_change=clear_function,
            args=(user_session_id,),
            index=None,
            placeholder="Select data..."
        )
        if selected_query:
            # Data preview
            st.markdown("### Data Preview")
            df = table_from_list(
                user_session_id,
                selected_corpus,
                selected_query,
                categories=groups
            )

        return selected_corpus, selected_query, df

    except Exception:
        st.warning(
            body="Nothing to preview yet. Select a table from the list.",
            icon=":material/table_eye:")
        return None, None, None


def render_data_preview_controls(
    df: pl.DataFrame,
    query: str,
    page_stem: str,
    user_session_id: str
) -> pl.DataFrame:
    """
    Render data preview controls (pivot table, make percent) for AI assistants.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe
    query : str
        Selected query/table name
    page_stem : str
        Page stem for state persistence
    user_session_id : str
        User session identifier

    Returns
    -------
    pl.DataFrame
        Processed dataframe
    """
    try:
        if df is not None and df.shape[0] > 0:
            # Check if DataFrame has Group column or is Document-Term Matrix
            if (df.get_column("Group", default=None) is not None or
                    "Document-Term Matrix" in str(query)):
                col1, col2 = st.columns(2)

                with col1:
                    pivot_table = st.toggle(
                        "Pivot Table",
                        key=persist("pivot_table", page_stem, user_session_id)
                    )

                    if pivot_table:
                        if ("Document-Term Matrix" in str(query) and
                                df.get_column("Group", default=None) is None):
                            df = df.pivot("Tag", index="doc_id", values="RF")
                        elif ("Document-Term Matrix" in str(query) and
                              df.get_column("Group", default=None) is not None):
                            df = df.pivot("Tag", index=["doc_id", "Group"], values="RF")
                        elif df.get_column("Group", default=None) is not None:
                            df = df.pivot("Tag", index="Group", values="RF")

                with col2:
                    make_percent = st.toggle(
                        "Make Percent",
                        key=persist("make_percent", page_stem, user_session_id),
                        value=False
                    )

                    if make_percent:
                        if "Document-Term Matrix" in str(query):
                            df = df.with_columns(pl.selectors.numeric().mul(100))
                        elif "Tags Table" in str(query):
                            df = df.with_columns(
                                pl.selectors.numeric().exclude(["AF", "Range"]).mul(100)
                            )

            # Display preview
            st.dataframe(df.head(10), use_container_width=True)

        return df

    except Exception as e:
        logger.error(f"Error in data preview controls: {e}")
        return df
