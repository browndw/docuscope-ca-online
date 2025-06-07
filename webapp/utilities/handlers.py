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

import pathlib

import docuscospacy as ds
import streamlit as st
import pandas as pd
import polars as pl
import tomli

from utilities import analysis

project_root = pathlib.Path(__file__).parent.parents[1].resolve()
OPTIONS = str(project_root.joinpath("webapp/options.toml"))


# Functions for handling states and files.
def get_version_from_pyproject() -> str:
    """
    Extract the version string from pyproject.toml.

    Returns
    -------
    str
        The version string, or '0.0.0' if not found.
    """
    pyproject_path = project_root / "pyproject.toml"
    try:
        with open(pyproject_path, "rb") as f:
            data = tomli.load(f)
        return data["project"]["version"]
    except Exception:
        return "0.0.0"


def import_options_general(options_path: str = OPTIONS) -> dict:
    """
    Import general options from a TOML file.

    Parameters
    ----------
    options_path : str, optional
        The path to the options TOML file. Defaults to the global OPTIONS path.

    Returns
    -------
    dict
        A dictionary containing the loaded options.
        If the file cannot be decoded,
        returns a dictionary with default option values.
    """
    try:
        with open(options_path, mode="rb") as fp:
            options = tomli.load(fp)
    except tomli.TOMLDecodeError:
        options = {}
        options['global'] = {}
        options['global']['check_size'] = False
        options['global']['check_language'] = False
        options['global']['enable_save'] = False
        options['global']['desktop_mode'] = False
        options['global']['max_bytes'] = 0
        options['llm']['llm_parameters'] = {}
        options['llm']['llm_model'] = 'gpt-4o-mini'
        options['cache']['cache_mode'] = False
        options['cache']['cache_location'] = None

    return options


# Functions for intializing session states.
def generate_temp(states: dict,
                  session_id: str):
    """
    Initialize session states with the given states for a specific session ID.

    Parameters
    ----------
    states : dict
        A dictionary of key-value pairs representing
        the states to be initialized.
    session_id : str
        The session ID for which the states are to be initialized.

    Returns
    -------
    None
    """
    if session_id not in st.session_state:
        st.session_state[session_id] = {}
    for key, value in states:
        if key not in st.session_state[session_id]:
            st.session_state[session_id][key] = value


def init_session(session_id: str):
    """
    Initialize the session state with default values for a specific session ID.

    Parameters
    ----------
    session_id : str
        The session ID for which the session state is to be initialized.

    Returns
    -------
    None
    """
    session = {}
    session['has_target'] = False
    session['target_db'] = ''
    session['has_meta'] = False
    session['has_reference'] = False
    session['reference_db'] = ''
    session['freq_table'] = False
    session['tags_table'] = False
    session['keyness_table'] = False
    session['ngrams'] = False
    session['kwic'] = False
    session['keyness_parts'] = False
    session['dtm'] = False
    session['pca'] = False
    session['collocations'] = False
    session['doc'] = False

    df = pl.from_dict(session)
    st.session_state[session_id]["session"] = df


def init_ai_assist(session_id: str) -> None:
    """
    Initialize AI assistant-related session state for a specific session ID.

    Parameters
    ----------
    session_id : str
        The session ID for which the AI assistant state is to be initialized.

    Returns
    -------
    None
    """
    if "messages" not in st.session_state[session_id]:
        st.session_state[session_id]["messages"] = [
            {"role": "assistant",
             "content": "Hello, what can I do for you today?"}
        ]

    if "plot_intent" not in st.session_state[session_id]:
        st.session_state[session_id]["plot_intent"] = False


# Functions for managing session values.
def update_session(key: str,
                   value,
                   session_id: str) -> None:
    """
    Update a specific key-value pair in the session state
    for a given session ID.

    Parameters
    ----------
    key : str
        The key in the session state to update.
    value : any
        The value to assign to the specified key.
    session_id : str
        The session ID for which the session state is to be updated.

    Returns
    -------
    None
    """
    session = st.session_state[session_id]["session"]
    session = session.to_dict(as_series=False)
    session[key] = value
    df = pl.from_dict(session)
    st.session_state[session_id]["session"] = df


def get_or_init_user_session() -> tuple[str, dict]:
    """
    Ensure a user session exists and return its ID and session dict.

    Returns
    -------
    tuple[str, dict]
        The user session ID and the session dictionary.
    """
    user_session = st.runtime.scriptrunner_utils.script_run_context.get_script_run_ctx()  # noqa: E501
    user_session_id = user_session.session_id

    if user_session_id not in st.session_state:
        st.session_state[user_session_id] = {}

    try:
        session = pl.DataFrame.to_dict(
            st.session_state[user_session_id]["session"], as_series=False
        )
    except KeyError:
        init_session(user_session_id)
        session = pl.DataFrame.to_dict(
            st.session_state[user_session_id]["session"], as_series=False
        )
    return user_session_id, session


# Functions for storing and managing corpus metadata
def init_metadata_target(session_id: str) -> None:
    """
    Initialize the metadata for the target corpus in the session state.

    Parameters
    ----------
    session_id : str
        The session ID for which the metadata is to be initialized.

    Returns
    -------
    None
    """
    df = st.session_state[session_id]["target"]["ds_tokens"]
    tags_to_check = df.get_column("ds_tag").to_list()
    tags = [
        'Actors',
        'Organization',
        'Planning',
        'Sentiment',
        'Signposting',
        'Stance'
        ]
    if any(tag in item for item in tags_to_check for tag in tags):
        model = 'Common Dictionary'
    else:
        model = 'Large Dictionary'
    ds_tags = df.get_column("ds_tag").unique().to_list()
    tags_pos = df.get_column("pos_tag").unique().to_list()
    if "Untagged" in ds_tags:
        ds_tags.remove("Untagged")
    if "Y" in tags_pos:
        tags_pos.remove("Y")
    temp_metadata_target = {}
    temp_metadata_target['tokens_pos'] = df.group_by(
        ["doc_id", "pos_id", "pos_tag"]
        ).agg(
            pl.col("token").str.concat("")
            ).filter(pl.col("pos_tag") != "Y").height
    temp_metadata_target['tokens_ds'] = df.group_by(
        ["doc_id", "ds_id", "ds_tag"]).agg(
            pl.col("token").str.concat("")
            ).filter(~(pl.col("token").str.contains(
                "^[[[:punct:]] ]+$"
                ) & pl.col("ds_tag").str.contains("Untagged"))).height
    temp_metadata_target['ndocs'] = len(
        df.get_column("doc_id").unique().to_list()
        )
    temp_metadata_target['model'] = model
    temp_metadata_target['docids'] = {'ids': sorted(
        df.get_column("doc_id").unique().to_list()
        )}
    temp_metadata_target['tags_ds'] = {'tags': sorted(ds_tags)}
    temp_metadata_target['tags_pos'] = {'tags': sorted(tags_pos)}
    temp_metadata_target['doccats'] = {'cats': ''}
    temp_metadata_target['collocations'] = {'temp': ''}
    temp_metadata_target['keyness_parts'] = {'temp': ''}
    temp_metadata_target['variance'] = {'temp': ''}

    df = pl.from_dict(temp_metadata_target, strict=False)
    st.session_state[session_id]["metadata_target"] = df


def init_metadata_reference(session_id: str) -> None:
    """
    Initialize the metadata for the reference corpus in the session state.

    Parameters
    ----------
    session_id : str
        The session ID for which the metadata is to be initialized.

    Returns
    -------
    None
    """
    df = st.session_state[session_id]["reference"]["ds_tokens"]
    tags_to_check = df.get_column("ds_tag").to_list()
    tags = [
        'Actors',
        'Organization',
        'Planning',
        'Sentiment',
        'Signposting',
        'Stance'
        ]
    if any(tag in item for item in tags_to_check for tag in tags):
        model = 'Common Dictionary'
    else:
        model = 'Large Dictionary'
    ds_tags = df.get_column("ds_tag").unique().to_list()
    tags_pos = df.get_column("pos_tag").unique().to_list()
    if "Untagged" in ds_tags:
        ds_tags.remove("Untagged")
    if "Y" in tags_pos:
        tags_pos.remove("Y")
    temp_metadata_reference = {}
    temp_metadata_reference['tokens_pos'] = df.group_by(
        ["doc_id", "pos_id", "pos_tag"]
        ).agg(pl.col("token").str.concat("")
              ).filter(pl.col("pos_tag") != "Y").height
    temp_metadata_reference['tokens_ds'] = df.group_by(
        ["doc_id", "ds_id", "ds_tag"]
        ).agg(pl.col("token").str.concat("")).filter(
            ~(pl.col("token").str.contains(
                "^[[[:punct:]] ]+$"
                ) & pl.col("ds_tag").str.contains("Untagged"))).height
    temp_metadata_reference['ndocs'] = len(
        df.get_column("doc_id").unique().to_list()
        )
    temp_metadata_reference['model'] = model
    temp_metadata_reference['doccats'] = False
    temp_metadata_reference['docids'] = {'ids': sorted(
        df.get_column("doc_id").unique().to_list()
        )}
    temp_metadata_reference['tags_ds'] = {'tags': sorted(ds_tags)}
    temp_metadata_reference['tags_pos'] = {'tags': sorted(tags_pos)}

    df = pl.from_dict(temp_metadata_reference, strict=False)
    st.session_state[session_id]["metadata_reference"] = df


def load_metadata(corpus_type,
                  session_id):
    table_name = "metadata_" + corpus_type
    metadata = st.session_state[session_id][table_name]
    metadata = metadata.to_dict(as_series=False)
    return metadata


def update_metadata(corpus_type,
                    key,
                    value,
                    session_id):
    table_name = "metadata_" + corpus_type
    metadata = st.session_state[session_id][table_name]
    metadata = metadata.to_dict(as_series=False)
    if key == "doccats":
        metadata['doccats'] = {'cats': [value]}
    elif key == "collocations":
        metadata['collocations'] = {'temp': [value]}
    elif key == "keyness_parts":
        metadata['keyness_parts'] = {'temp': [value]}
    elif key == "variance":
        metadata['variance'] = {'temp': [value]}
    else:
        metadata[key] = value
    df = pl.from_dict(metadata, strict=False)
    st.session_state[session_id][table_name] = df


def sidebar_action_button(
    button_label: str,
    button_icon: str,
    preconditions: list,  # Now just a list of bools
    action: callable,
    spinner_message: str = "Processing...",
    sidebar=True
) -> None:
    """
    Render a sidebar button that checks preconditions and runs an action.

    Parameters
    ----------
    button_label : str
        The label for the sidebar button.
    preconditions : list
        Lis of conditions.
        If any condition is False, show the error and do not run action.
    action : Callable
        Function to run if all preconditions are met.
    spinner_message : str
        Message to show in spinner.
    sidebar : bool
        If True, use st.sidebar, else use main area.
    error_in_sidebar : bool
        If True, show error messages in the sidebar,
        otherwise show in the main area.
    """
    container = st.sidebar if sidebar else st
    if container.button(button_label, icon=button_icon):
        if not all(preconditions):
            st.error(
                    "It doesn't look like you've loaded a target corpus yet.\n"
                    "You can do this by clicking on the **Manage Corpus Data** button above.",  # noqa: E501
                    icon=":material/sentiment_stressed:"
                )
            return
        with container:
            with st.spinner(spinner_message):
                action()


# Callable actions
def generate_frequency_table(user_session_id: str):
    """
    Load frequency tables for the target corpus.

    Parameters
    ----------
    user_session_id : str
        The session ID for the user.

    Returns
    -------
    None
    """
    # --- Try to get the target tokens table ---
    try:
        tok_pl = st.session_state[user_session_id]["target"]["ds_tokens"]
    except KeyError:
        st.session_state[user_session_id]["frequency_warning"] = (
            "Frequency table cannot be generated: no tokens found in the target corpus.",
            ":material/sentiment_stressed:"
        )
        return

    # --- Check if the tokens table is empty ---
    if tok_pl is None or getattr(tok_pl, "height", 0) == 0:
        st.session_state[user_session_id]["frequency_warning"] = (
            "Frequency table cannot be generated: no tokens found in the target corpus.",
            ":material/sentiment_stressed:"
        )
        return

    # --- Generate load and tags tables ---
    update_session('freq_table', True, user_session_id)
    st.session_state[user_session_id]["frequency_warning"] = None
    st.success('Frequency tables generated!')
    st.rerun()


def generate_tags_table(user_session_id: str):
    """
    Load tags tables for the target corpus.

    Parameters
    ----------
    user_session_id : str
        The session ID for the user.

    Returns
    -------
    None
    """
    # --- Try to get the target tokens table ---
    try:
        tok_pl = st.session_state[user_session_id]["target"]["ds_tokens"]
    except KeyError:
        st.session_state[user_session_id]["tags_warning"] = (
            "Tags table cannot be generated: no tokens found in the target corpus.",
            ":material/info:"
        )
        return

    # --- Check if the tokens table is empty ---
    if tok_pl is None or getattr(tok_pl, "height", 0) == 0:
        st.session_state[user_session_id]["tags_warning"] = (
            "Tags table cannot be generated: no tokens found in the target corpus.",
            ":material/info:"
        )
        return

    # --- Generate load and tags tables ---
    update_session('tags_table', True, user_session_id)
    st.session_state[user_session_id]["tags_warning"] = None
    st.success('Tags tables generated!')
    st.rerun()


def generate_keyness_tables(user_session_id: str, threshold=0.01, swap_target=False):
    # --- Try to get all required frequency/tag tables ---
    try:
        wc_tar_pos = st.session_state[user_session_id]["target"]["ft_pos"]
        wc_tar_ds = st.session_state[user_session_id]["target"]["ft_ds"]
        tc_tar_pos = st.session_state[user_session_id]["target"]["tt_pos"]
        tc_tar_ds = st.session_state[user_session_id]["target"]["tt_ds"]
        wc_ref_pos = st.session_state[user_session_id]["reference"]["ft_pos"]
        wc_ref_ds = st.session_state[user_session_id]["reference"]["ft_ds"]
        tc_ref_pos = st.session_state[user_session_id]["reference"]["tt_pos"]
        tc_ref_ds = st.session_state[user_session_id]["reference"]["tt_ds"]
    except KeyError:
        st.session_state[user_session_id]["keyness_warning"] = (
            """
            Keyness cannot be computed: missing frequency or tag tables.
            Please generate frequency and tag tables for both corpora.
            """,
            ":material/sentiment_stressed:"
        )
        return

    # --- Check for empty or invalid dataframes ---
    freq_tables = [
        wc_tar_pos, wc_tar_ds, tc_tar_pos, tc_tar_ds,
        wc_ref_pos, wc_ref_ds, tc_ref_pos, tc_ref_ds
    ]
    if any(df is None or getattr(df, "height", 0) == 0 for df in freq_tables):
        st.session_state[user_session_id]["keyness_warning"] = (
            "Keyness cannot be computed: one or more required tables are empty.",
            ":material/sentiment_stressed:"
        )
        return

    # --- Compute keyness tables with threshold and swap_target ---
    kw_pos = ds.keyness_table(wc_tar_pos, wc_ref_pos, threshold=threshold, swap_target=swap_target)  # noqa: E501
    kw_ds = ds.keyness_table(wc_tar_ds, wc_ref_ds, threshold=threshold, swap_target=swap_target)  # noqa: E501
    kt_pos = ds.keyness_table(tc_tar_pos, tc_ref_pos, tags_only=True, threshold=threshold, swap_target=swap_target)  # noqa: E501
    kt_ds = ds.keyness_table(tc_tar_ds, tc_ref_ds, tags_only=True, threshold=threshold, swap_target=swap_target)  # noqa: E501

    # --- Check for empty results ---
    keyness_tables = [kw_pos, kw_ds, kt_pos, kt_ds]
    if any(df is None or getattr(df, "height", 0) == 0 for df in keyness_tables):
        st.session_state[user_session_id]["keyness_warning"] = (
            "Keyness computation returned no results. Try different data.",
            ":material/counter_0:"
        )
        return

    # --- Save results and clear warning ---
    st.session_state[user_session_id]["target"]["kw_pos"] = kw_pos
    st.session_state[user_session_id]["target"]["kw_ds"] = kw_ds
    st.session_state[user_session_id]["target"]["kt_pos"] = kt_pos
    st.session_state[user_session_id]["target"]["kt_ds"] = kt_ds

    update_session('keyness_table', True, user_session_id)
    st.session_state[user_session_id]["keyness_warning"] = None
    st.success('Keywords generated!')
    st.rerun()


def generate_keyness_parts(user_session_id: str):
    # --- Check for metadata ---
    session = pl.DataFrame.to_dict(
            st.session_state[user_session_id]["session"],
            as_series=False
            )
    if session.get('has_meta')[0] is False:
        st.session_state[user_session_id]["keyness_parts_warning"] = (
            """
            No metadata found for the target corpus.
            Please load or generate metadata first.
            """,
            ":material/info:"
        )
        return

    tar_list = list(st.session_state[user_session_id].get('tar', []))
    ref_list = list(st.session_state[user_session_id].get('ref', []))

    # --- Check for empty categories ---
    if len(tar_list) == 0 or len(ref_list) == 0:
        st.session_state[user_session_id]["keyness_parts_warning"] = (
            "You must select at least one category for both target and reference parts.",
            ":material/info:"
        )
        return

    # --- Main logic ---
    tok_pl = st.session_state[user_session_id]["target"]["ds_tokens"]

    tar_pl = analysis.subset_pl(tok_pl, tar_list)
    ref_pl = analysis.subset_pl(tok_pl, ref_list)

    wc_tar_pos, wc_tar_ds = ds.frequency_table(tar_pl, count_by="both")
    tc_tar_pos, tc_tar_ds = ds.tags_table(tar_pl, count_by="both")
    wc_ref_pos, wc_ref_ds = ds.frequency_table(ref_pl, count_by="both")
    tc_ref_pos, tc_ref_ds = ds.tags_table(ref_pl, count_by="both")

    kw_pos_cp = ds.keyness_table(wc_tar_pos, wc_ref_pos)
    kw_ds_cp = ds.keyness_table(wc_tar_ds, wc_ref_ds)
    kt_pos_cp = ds.keyness_table(tc_tar_pos, tc_ref_pos, tags_only=True)
    kt_ds_cp = ds.keyness_table(tc_tar_ds, tc_ref_ds, tags_only=True)

    # --- Check for empty results ---
    keyness_tables = [kw_pos_cp, kw_ds_cp, kt_pos_cp, kt_ds_cp]
    if any(df is None or getattr(df, "height", 0) == 0 for df in keyness_tables):
        st.session_state[user_session_id]["keyness_parts_warning"] = (
            """
            Keyness computation for corpus parts returned no results.
            Try different categories.
            """,
            ":material/info:"
        )
        return

    tar_tokens_pos = tar_pl.group_by(
        ["doc_id", "pos_id", "pos_tag"]
    ).agg(pl.col("token").str.concat("")).filter(pl.col("pos_tag") != "Y").height
    ref_tokens_pos = ref_pl.group_by(
        ["doc_id", "pos_id", "pos_tag"]
    ).agg(pl.col("token").str.concat("")).filter(pl.col("pos_tag") != "Y").height

    tar_tokens_ds = tar_pl.group_by(
        ["doc_id", "ds_id", "ds_tag"]
    ).agg(pl.col("token").str.concat("")).filter(
        ~(pl.col("token").str.contains("^[[[:punct:]] ]+$") & pl.col("ds_tag").str.contains("Untagged"))  # noqa: E501
    ).height
    ref_tokens_ds = ref_pl.group_by(
        ["doc_id", "ds_id", "ds_tag"]
    ).agg(pl.col("token").str.concat("")).filter(
        ~(pl.col("token").str.contains("^[[[:punct:]] ]+$") & pl.col("ds_tag").str.contains("Untagged"))  # noqa: E501
    ).height

    tar_ndocs = tar_pl.get_column("doc_id").unique().len()
    ref_ndocs = ref_pl.get_column("doc_id").unique().len()

    # --- Save results and clear warning ---
    st.session_state[user_session_id]["target"]["kw_pos_cp"] = kw_pos_cp
    st.session_state[user_session_id]["target"]["kw_ds_cp"] = kw_ds_cp
    st.session_state[user_session_id]["target"]["kt_pos_cp"] = kt_pos_cp
    st.session_state[user_session_id]["target"]["kt_ds_cp"] = kt_ds_cp

    update_session('keyness_parts', True, user_session_id)

    update_metadata(
        'target',
        key='keyness_parts',
        value=[
            tar_list,
            ref_list,
            str(tar_tokens_pos),
            str(ref_tokens_pos),
            str(tar_tokens_ds),
            str(ref_tokens_ds),
            str(tar_ndocs),
            str(ref_ndocs)
        ],
        session_id=user_session_id
    )

    st.session_state[user_session_id]["keyness_parts_warning"] = None
    st.success('Keywords generated!')
    st.rerun()


def generate_collocations(
    user_session_id: str,
    node_word: str,
    node_tag: str,
    to_left: int,
    to_right: int,
    stat_mode: str,
    count_by: str
):
    # --- User input validation ---
    if not node_word:
        st.session_state[user_session_id]["collocations_warning"] = (
            "Please enter a node word.",
            ":material/info:"
        )
        return
    if " " in node_word:
        st.session_state[user_session_id]["collocations_warning"] = (
            "Node word cannot contain spaces.",
            ":material/info:"
        )
        return
    if len(node_word) > 15:
        st.session_state[user_session_id]["collocations_warning"] = (
            "Node word is too long (max 15 characters).",
            ":material/info:"
        )
        return

    # --- Main logic ---
    tok_pl = st.session_state[user_session_id]["target"].get("ds_tokens")
    if tok_pl is None or getattr(tok_pl, "height", 0) == 0:
        st.session_state[user_session_id]["collocations_warning"] = (
            """
            No tokens found for the target corpus.
            Please load and process a corpus first.
            """,
            ":material/sentiment_stressed:"
        )
        return

    coll_df = ds.coll_table(
        tok_pl,
        node_word=node_word,
        node_tag=node_tag,
        preceding=to_left,
        following=to_right,
        statistic=stat_mode,
        count_by=count_by
    )

    # --- Data-dependent warnings ---
    if coll_df is None or coll_df.is_empty():
        st.session_state[user_session_id]["collocations_warning"] = (
            "Your search didn't return any matches. Try something else.",
            ":material/info:"
        )
        return

    # --- Success ---
    if "collocations" not in st.session_state[user_session_id]["target"]:
        st.session_state[user_session_id]["target"]["collocations"] = {}
    st.session_state[user_session_id]["target"]["collocations"] = coll_df

    update_session(
        'collocations',
        True,
        user_session_id
    )

    update_metadata(
        'target',
        key='collocations',
        value=[node_word, stat_mode, str(to_left), str(to_right)],
        session_id=user_session_id
    )

    st.session_state[user_session_id]["collocations_warning"] = None
    st.success('Collocations generated!')
    st.rerun()


def generate_kwic(
    user_session_id: str,
    node_word: str,
    search_type: str,
    ignore_case: bool
):
    # --- User input validation ---
    if not node_word:
        st.session_state[user_session_id]["kwic_warning"] = (
            "Please enter a node word.",
            ":material/info:"
        )
        return
    if " " in node_word:
        st.session_state[user_session_id]["kwic_warning"] = (
            "Node word cannot contain spaces.",
            ":material/info:"
        )
        return
    if len(node_word) > 15:
        st.session_state[user_session_id]["kwic_warning"] = (
            "Node word is too long (max 15 characters).",
            ":material/info:"
        )
        return

    # --- Main logic ---
    tok_pl = st.session_state[user_session_id]["target"].get("ds_tokens")
    if tok_pl is None or getattr(tok_pl, "height", 0) == 0:
        st.session_state[user_session_id]["kwic_warning"] = (
            """
            No tokens found for the target corpus.
            Please load and process a corpus first.
            """,
            ":material/info:"
        )
        return

    kwic_df = ds.kwic_center_node(
        tok_pl,
        node_word=node_word,
        search_type=search_type,
        ignore_case=ignore_case
    )

    # --- Data-dependent warnings ---
    if kwic_df is None or kwic_df.is_empty():
        st.session_state[user_session_id]["kwic_warning"] = (
            "Your search didn't return any matches. Try something else.",
            ":material/info:"
        )
        return

    # --- Success ---
    if "kwic" not in st.session_state[user_session_id]["target"]:
        st.session_state[user_session_id]["target"]["kwic"] = {}
    st.session_state[user_session_id]["target"]["kwic"] = kwic_df

    update_session(
        'kwic',
        True,
        user_session_id
    )
    st.session_state[user_session_id]["kwic_warning"] = None
    st.rerun()


def generate_ngrams(
    user_session_id: str,
    ngram_span: int,
    ts: str = 'doc_id'  # Default to 'doc_id' for ngram counting
):
    # --- User input validation ---
    if not isinstance(ngram_span, int) or ngram_span < 2 or ngram_span > 10:
        st.session_state[user_session_id]["ngram_warning"] = (
            "Please select a valid n-gram span (2–10).",
            ":material/info:"
        )
        return

    # --- Main logic ---
    tok_pl = st.session_state[user_session_id]["target"].get("ds_tokens")
    if tok_pl is None or getattr(tok_pl, "height", 0) == 0:
        st.session_state[user_session_id]["ngram_warning"] = (
            """
            No tokens found for the target corpus.
            Please load and process a corpus first.
            """,
            ":material/info:"
        )
        return

    ngram_df = ds.ngrams(
        tokens_table=tok_pl,
        span=ngram_span,
        count_by=ts
    )

    # --- Data-dependent warnings ---
    if ngram_df is None or getattr(ngram_df, "height", 0) < 2:
        st.session_state[user_session_id]["ngram_warning"] = (
            "Your search didn't return any results.",
            ":material/info:"
        )
        return

    # --- Success ---
    if "ngrams" not in st.session_state[user_session_id]["target"]:
        st.session_state[user_session_id]["target"]["ngrams"] = {}
    st.session_state[user_session_id]["target"]["ngrams"] = ngram_df
    update_session(
        'ngrams',
        True,
        user_session_id
    )
    st.session_state[user_session_id]["ngram_warning"] = None
    st.rerun()


def generate_clusters(
    user_session_id: str,
    from_anchor: str,
    node_word: str,
    tag: str,
    position: int,
    ngram_span: int,
    search: str,
    ts: str = 'doc_id'
):
    # --- User input validation ---
    if from_anchor == 'Token':
        if not node_word or node_word == 'by_tag':
            st.session_state[user_session_id]["ngram_warning"] = (
                "Please enter a node word.",
                ":material/info:"
            )
            return
        if " " in node_word:
            st.session_state[user_session_id]["ngram_warning"] = (
                "Node word cannot contain spaces.",
                ":material/info:"
            )
            return
        if len(node_word) > 15:
            st.session_state[user_session_id]["ngram_warning"] = (
                "Node word is too long (max 15 characters).",
                ":material/info:"
            )
            return
    elif from_anchor == 'Tag':
        if not tag or tag == 'No tags currently loaded':
            st.session_state[user_session_id]["ngram_warning"] = (
                "Please select a valid tag.",
                ":material/info:"
            )
            return

    # --- Main logic ---
    tok_pl = st.session_state[user_session_id]["target"]["ds_tokens"]
    ngram_df = None
    if from_anchor == 'Token':
        ngram_df = ds.clusters_by_token(
            tokens_table=tok_pl,
            node_word=node_word,
            node_position=position,
            span=ngram_span,
            search_type=search,
            count_by=ts
        )
    elif from_anchor == 'Tag':
        ngram_df = ds.clusters_by_tag(
            tokens_table=tok_pl,
            tag=tag,
            tag_position=position,
            span=ngram_span,
            count_by=ts
        )

    # --- Data-dependent warnings ---
    if ngram_df is None or getattr(ngram_df, "height", 0) == 0:
        st.session_state[user_session_id]["ngram_warning"] = (
            "Your search didn't return any matches. Try something else.",
            ":material/info:"
        )
        return
    elif ngram_df.height > 100000:
        st.session_state[user_session_id]["ngram_warning"] = (
            "Your search returned too many matches! Try something more specific.",
            ":material/info:"
        )
        return

    # --- Success ---
    if "ngrams" not in st.session_state[user_session_id]["target"]:
        st.session_state[user_session_id]["target"]["ngrams"] = {}
    st.session_state[user_session_id]["target"]["ngrams"] = ngram_df
    update_session(
        'ngrams',
        True,
        user_session_id
    )
    st.session_state[user_session_id]["ngram_warning"] = None
    st.rerun()


def generate_pca(
    user_session_id: str,
    df: pd.DataFrame,
    metadata_target: dict,
    session: dict
):
    # --- User input validation ---
    if df is None or df.empty:
        st.session_state[user_session_id]["pca_warning"] = (
            "No data available for PCA. Please process your corpus and select valid tags.",
            ":material/info:"
        )
        return

    # --- Check for metadata grouping ---
    if session.get('has_meta', [False])[0]:
        grouping = metadata_target.get('doccats', [{}])[0].get('cats', [])
    else:
        grouping = []

    # --- Drop unwanted columns ---
    to_drop = ['Other', 'FU', 'Untagged']
    df = df.drop([x for x in to_drop if x in df.columns], axis=1)

    # --- Check if enough columns remain for PCA ---
    if df.shape[1] < 2:
        st.session_state[user_session_id]["pca_warning"] = (
            "Not enough variables for PCA after dropping excluded columns.",
            ":material/info:"
        )
        return

    # --- Compute PCA ---
    try:
        pca_df, contrib_df, ve = analysis.pca_contributions(df, grouping)
    except Exception as e:
        st.session_state[user_session_id]["pca_warning"] = (
            f"PCA computation failed: {e}",
            ":material/sentiment_stressed:"
        )
        return

    # --- Check for empty PCA results ---
    if pca_df is None or pca_df.empty or contrib_df is None or contrib_df.empty:
        st.session_state[user_session_id]["pca_warning"] = (
            "PCA computation returned no results. Try different data.",
            ":material/info:"
        )
        return

    # --- Save results and clear warning ---
    st.session_state[user_session_id]["target"]["pca_df"] = pca_df
    st.session_state[user_session_id]["target"]["contrib_df"] = contrib_df

    update_metadata(
        'target',
        'variance',
        ve,
        user_session_id
    )
    update_session(
        'pca',
        True,
        user_session_id
    )
    st.session_state[user_session_id]["pca_warning"] = None
    st.success('PCA computed!')
    st.rerun()


def generate_scatterplot(
    user_session_id: str,
    df: pl.DataFrame,
    xaxis: str,
    yaxis: str
):
    # --- User input validation ---
    if df is None or df.is_empty():
        st.session_state[user_session_id]["scatter_warning"] = (
            """
            No data available for plotting."
            Please process your corpus and select valid tags.
            """,
            ":material/info:"
        )
        return

    if xaxis not in df.columns or yaxis not in df.columns:
        st.session_state[user_session_id]["scatter_warning"] = (
            "Selected axes are not present in the data.",
            ":material/info:"
        )
        return

    # --- Prepare data ---
    try:
        df_plot = ds.dtm_weight(df).with_columns(
            pl.selectors.numeric().mul(100)
        ).to_pandas()
    except Exception as e:
        st.session_state[user_session_id]["scatter_warning"] = (
            f"Failed to prepare data for plotting: {e}",
            ":material/sentiment_stressed:"
        )
        return

    # --- Check if enough data for plotting ---
    if df_plot.empty:
        st.session_state[user_session_id]["scatter_warning"] = (
            "No data available after weighting for plotting.",
            ":material/info:"
        )
        return

    # --- Compute correlation ---
    try:
        cc_df, cc_r, cc_p = analysis.correlation(
            df_plot,
            xaxis,
            yaxis
        )
    except Exception as e:
        st.session_state[user_session_id]["scatter_warning"] = (
            f"Correlation computation failed: {e}",
            ":material/sentiment_stressed:"
        )
        return

    # --- Save results and clear warning ---
    st.session_state[user_session_id]["scatterplot_df"] = df_plot
    st.session_state[user_session_id]["scatter_correlation"] = (cc_df, cc_r, cc_p)
    st.session_state[user_session_id]["scatter_warning"] = None
    st.success('Scatterplot generated!')
    # Optionally: st.rerun()


def generate_scatterplot_with_groups(
    user_session_id: str,
    df: pl.DataFrame,
    xaxis: str,
    yaxis: str,
    metadata_target: dict
):
    # --- User input validation ---
    if df is None or df.is_empty():
        st.session_state[user_session_id]["scatter_group_warning"] = (
            """
            No data available for plotting.
            Please process your corpus and select valid tags.
            """,
            ":material/info:"
        )
        return

    if xaxis not in df.columns or yaxis not in df.columns:
        st.session_state[user_session_id]["scatter_group_warning"] = (
            "Selected axes are not present in the data.",
            ":material/info:"
        )
        return

    # --- Prepare data ---
    try:
        df_plot = ds.dtm_weight(df).with_columns(
            pl.selectors.numeric().mul(100)
        )
        df_plot = df_plot.with_columns(
            pl.col("doc_id")
            .str.split_exact("_", 0)
            .struct.rename_fields(["Group"])
            .alias("id")
        ).unnest("id").to_pandas()
    except Exception as e:
        st.session_state[user_session_id]["scatter_group_warning"] = (
            f"Failed to prepare data for plotting: {e}",
            ":material/sentiment_stressed:"
        )
        return

    if df_plot.empty:
        st.session_state[user_session_id]["scatter_group_warning"] = (
            "No data available after weighting for plotting.",
            ":material/info:"
        )
        return

    # --- Compute correlation ---
    try:
        cc_df, cc_r, cc_p = analysis.correlation(
            df_plot,
            xaxis,
            yaxis
        )
    except Exception as e:
        st.session_state[user_session_id]["scatter_group_warning"] = (
            f"Correlation computation failed: {e}",
            ":material/sentiment_stressed:"
        )
        return

    # --- Save results and clear warning ---
    st.session_state[user_session_id]["scatterplot_group_df"] = df_plot
    st.session_state[user_session_id]["scatter_group_correlation"] = (cc_df, cc_r, cc_p)
    st.session_state[user_session_id]["scatter_group_warning"] = None
    st.success('Scatterplot with groups generated!')
    # Optionally: st.rerun()


def generate_boxplot(
    user_session_id: str,
    df: pl.DataFrame,
    box_vals: list
):
    # --- User input validation ---
    if df is None or df.is_empty():
        st.session_state[user_session_id]["boxplot_warning"] = (
            """
            No data available for plotting.
            Please process your corpus and select valid tags.
            """,
            ":material/info:"
        )
        return

    if not box_vals or any(val not in df.columns for val in box_vals):
        st.session_state[user_session_id]["boxplot_warning"] = (
            "Please select at least one valid variable for plotting.",
            ":material/info:"
        )
        return

    # --- Prepare data ---
    try:
        df_plot = ds.dtm_weight(df)
        df_plot = analysis.boxplots_pl(
            df_plot,
            box_vals,
            grp_a=None,
            grp_b=None
        )
    except Exception as e:
        st.session_state[user_session_id]["boxplot_warning"] = (
            f"Failed to prepare data for plotting: {e}",
            ":material/sentiment_stressed:"
        )
        return

    # --- Check if enough data for plotting ---
    try:
        df_pandas = df_plot.to_pandas()
    except Exception as e:
        st.session_state[user_session_id]["boxplot_warning"] = (
            f"Failed to convert data for plotting: {e}",
            ":material/sentiment_stressed:"
        )
        return

    if df_pandas.empty:
        st.session_state[user_session_id]["boxplot_warning"] = (
            "No data available after weighting for plotting.",
            ":material/info:"
        )
        return

    # --- Compute descriptive statistics ---
    try:
        stats = (
            df_plot
            .group_by(["Tag"])
            .agg(
                pl.len().alias("count"),
                pl.col("RF").mean().alias("mean"),
                pl.col("RF").median().alias("median"),
                pl.col("RF").std().alias("std"),
                pl.col("RF").min().alias("min"),
                pl.col("RF").quantile(0.25).alias("25%"),
                pl.col("RF").quantile(0.5).alias("50%"),
                pl.col("RF").quantile(0.75).alias("75%"),
                pl.col("RF").max().alias("max")
            )
            .sort("Tag")
        )
    except Exception as e:
        st.session_state[user_session_id]["boxplot_warning"] = (
            f"Failed to compute descriptive statistics: {e}",
            ":material/sentiment_stressed:"
        )
        return

    # --- Save results and clear warning ---
    st.session_state[user_session_id]["boxplot_df"] = df_pandas
    st.session_state[user_session_id]["boxplot_stats"] = stats
    st.session_state[user_session_id]["boxplot_warning"] = None
    st.success('Boxplot generated!')


def generate_boxplot_by_group(
    user_session_id: str,
    df: pl.DataFrame,
    box_vals: list,
    grpa_list: list,
    grpb_list: list
):
    import streamlit as st
    import polars as pl

    # --- User input validation ---
    if df is None or df.is_empty():
        st.session_state[user_session_id]["boxplot_group_warning"] = (
            """
            No data available for plotting.
            Please process your corpus and select valid tags.
            """,
            ":material/info:"
        )
        return

    if not box_vals or any(val not in df.columns for val in box_vals):
        st.session_state[user_session_id]["boxplot_group_warning"] = (
            "Please select at least one valid variable for plotting.",
            ":material/info:"
        )
        return

    if len(grpa_list) == 0 or len(grpb_list) == 0:
        st.session_state[user_session_id]["boxplot_group_warning"] = (
            "You must select at least one category for both Group A and Group B.",
            ":material/info:"
        )
        return

    # --- Prepare data ---
    try:
        df_plot = ds.dtm_weight(df)
        df_plot = analysis.boxplots_pl(
            df_plot,
            box_vals,
            grp_a=grpa_list,
            grp_b=grpb_list
        )
        df_pandas = df_plot.to_pandas()
    except Exception as e:
        st.session_state[user_session_id]["boxplot_group_warning"] = (
            f"Failed to prepare data for plotting: {e}",
            ":material/sentiment_stressed:"
        )
        return

    if df_pandas.empty:
        st.session_state[user_session_id]["boxplot_group_warning"] = (
            "No data available after weighting for plotting.",
            ":material/info:"
        )
        return

    # --- Compute descriptive statistics ---
    try:
        stats = (
            df_plot
            .group_by(["Group", "Tag"])
            .agg(
                pl.len().alias("count"),
                pl.col("RF").mean().alias("mean"),
                pl.col("RF").median().alias("median"),
                pl.col("RF").std().alias("std"),
                pl.col("RF").min().alias("min"),
                pl.col("RF").quantile(0.25).alias("25%"),
                pl.col("RF").quantile(0.5).alias("50%"),
                pl.col("RF").quantile(0.75).alias("75%"),
                pl.col("RF").max().alias("max")
            )
            .sort(["Tag", "Group"])
        )
    except Exception as e:
        st.session_state[user_session_id]["boxplot_group_warning"] = (
            f"Failed to compute descriptive statistics: {e}",
            ":material/sentiment_stressed:"
        )
        return

    # --- Save results and clear warning ---
    st.session_state[user_session_id]["boxplot_group_df"] = df_pandas
    st.session_state[user_session_id]["boxplot_group_stats"] = stats
    st.session_state[user_session_id]["boxplot_group_warning"] = None
    st.success('Boxplot by group generated!')


def generate_document_html(user_session_id: str, doc_key: str):
    """
    Process a single document and generate HTML representations.

    Parameters
    ----------
    user_session_id : str
        The session ID for the user.
    doc_key : str
        The document key or identifier.

    Returns
    -------
    None
    """
    # --- Check if target corpus is loaded ---
    session = pl.DataFrame.to_dict(
        st.session_state[user_session_id]["session"], as_series=False
    )
    if session.get('has_target', [False])[0] is False:
        st.session_state[user_session_id]["doc_warning"] = (
            "No target corpus loaded. Please load a document first.",
            ":material/info:"
        )
        return

    # --- Try to get the target tokens table ---
    try:
        tok_pl = st.session_state[user_session_id]["target"]["ds_tokens"]
    except KeyError:
        st.session_state[user_session_id]["doc_warning"] = (
            "No tokens found in the target corpus.",
            ":material/info:"
        )
        return

    if tok_pl is None or getattr(tok_pl, "height", 0) == 0:
        st.session_state[user_session_id]["doc_warning"] = (
            "No tokens found in the target corpus.",
            ":material/info:"
        )
        return

    # --- Generate HTML representations ---
    try:
        doc_pos, doc_simple, doc_ds = analysis.html_build_pl(tok_pl, doc_key)
    except Exception as e:
        st.session_state[user_session_id]["doc_warning"] = (
            f"Failed to process document: {e}",
            ":material/sentiment_stressed:"
        )
        return

    # --- Save results ---
    st.session_state[user_session_id]["target"]["doc_pos"] = doc_pos
    st.session_state[user_session_id]["target"]["doc_simple"] = doc_simple
    st.session_state[user_session_id]["target"]["doc_ds"] = doc_ds

    update_session('doc', True, user_session_id)
    st.session_state[user_session_id]["doc_warning"] = None
    st.success('Document processed!')
    st.rerun()


# Functions for storing values associated with specific apps
def update_tags(html_state: str,
                session_id: str) -> None:
    """
    Update the HTML style string for tag highlights in the session state.

    Parameters
    ----------
    html_state : str
        The HTML string representing the current tag highlights.
    session_id : str
        The session ID for which the tag highlights are to be updated.

    Returns
    -------
    None
    """
    _TAGS = f"tags_{session_id}"
    html_highlights = [
        ' { background-color:#5fb7ca; }',
        ' { background-color:#e35be5; }',
        ' { background-color:#ffc701; }',
        ' { background-color:#fe5b05; }',
        ' { background-color:#cb7d60; }'
        ]
    if 'html_str' not in st.session_state[session_id]:
        st.session_state[session_id]['html_str'] = ''
    if _TAGS in st.session_state:
        tags = st.session_state[_TAGS]
        if len(tags) > 5:
            tags = tags[:5]
            st.session_state[_TAGS] = tags
    else:
        tags = []
    tags = ['.' + x for x in tags]
    highlights = html_highlights[:len(tags)]
    style_str = [''.join(x) for x in zip(tags, highlights)]
    style_str = ''.join(style_str)
    style_sheet_str = '<style>' + style_str + '</style>'
    st.session_state[session_id]['html_str'] = style_sheet_str + html_state


# Convenience function called by widgets
def is_valid_df(df, required_cols=None):
    if df is None:
        return False
    if hasattr(df, "height"):
        if df.height == 0:
            return False
        cols = df.columns
    elif hasattr(df, "shape"):
        if df.shape[0] == 0:
            return False
        cols = df.columns
    else:
        return False
    if required_cols:
        return all(col in cols for col in required_cols)
    return True


def clear_boxplot_multiselect(user_session_id: str):
    import streamlit as st
    st.session_state[user_session_id]["boxplot_df"] = None
    st.session_state[user_session_id]["boxplot_stats"] = None
    st.session_state[user_session_id]["boxplot_warning"] = None
    st.session_state[user_session_id]["boxplot_group_df"] = None
    st.session_state[user_session_id]["boxplot_group_stats"] = None
    st.session_state[user_session_id]["boxplot_group_warning"] = None


def clear_plots(session_id):
    update_session('pca', False, session_id)
    _GRPA = f"grpa_{session_id}"
    _GRPB = f"grpb_{session_id}"
    _BOXPLOT_VARS = f"boxplot_vars_{session_id}"
    # Clear group selections
    if _GRPA in st.session_state.keys():
        st.session_state[_GRPA] = []
    if _GRPB in st.session_state.keys():
        st.session_state[_GRPB] = []
    # Clear boxplot variable selections
    if _BOXPLOT_VARS in st.session_state.keys():
        st.session_state[_BOXPLOT_VARS] = []
    # Clear highlight multiselects
    highlight_keys = [
        f"highlight_pca_groups_{session_id}",
        f"highlight_scatter_groups_{session_id}",
        # add other highlight keys as needed
    ]
    for key in highlight_keys:
        if key in st.session_state:
            st.session_state[key] = []
    # Clear plot results and warnings
    if session_id in st.session_state:
        # Remove 'Highlight' column from all relevant DataFrames
        for key in [
            "boxplot_df", "boxplot_group_df", "scatterplot_df", "scatterplot_group_df"
        ]:
            df = st.session_state[session_id].get(key)
            if df is not None and hasattr(df, "columns") and "Highlight" in df.columns:
                st.session_state[session_id][key] = df.drop(columns=["Highlight"])
        st.session_state[session_id]["boxplot_df"] = None
        st.session_state[session_id]["boxplot_stats"] = None
        st.session_state[session_id]["boxplot_warning"] = None
        st.session_state[session_id]["boxplot_group_df"] = None
        st.session_state[session_id]["boxplot_group_stats"] = None
        st.session_state[session_id]["boxplot_group_warning"] = None
        st.session_state[session_id]["scatterplot_df"] = None
        st.session_state[session_id]["scatter_correlation"] = None
        st.session_state[session_id]["scatter_warning"] = None
        st.session_state[session_id]["scatterplot_group_df"] = None
        st.session_state[session_id]["scatter_group_correlation"] = None
        st.session_state[session_id]["scatter_group_warning"] = None
        # --- Clear PCA data and warnings ---
        if "target" in st.session_state[session_id]:
            st.session_state[session_id]["target"]["pca_df"] = None
            st.session_state[session_id]["target"]["contrib_df"] = None
        st.session_state[session_id]["pca_warning"] = None
        if "pca_idx" in st.session_state[session_id]:
            st.session_state[session_id]["pca_idx"] = None


def persist(key: str,
            app_name: str,
            session_id: str) -> str:
    _PERSIST_STATE_KEY = f"{app_name}_PERSIST"
    if _PERSIST_STATE_KEY not in st.session_state[session_id].keys():
        st.session_state[session_id][_PERSIST_STATE_KEY] = {}
        st.session_state[session_id][_PERSIST_STATE_KEY][key] = None

    if key in st.session_state:
        st.session_state[session_id][_PERSIST_STATE_KEY][key] = st.session_state[key]  # noqa: E501

    return key


def load_widget_state(app_name: str,
                      session_id):
    _PERSIST_STATE_KEY = f"{app_name}_PERSIST"
    """Load persistent widget state."""
    if _PERSIST_STATE_KEY in st.session_state[session_id]:
        for key in st.session_state[session_id][_PERSIST_STATE_KEY]:
            if st.session_state[session_id][_PERSIST_STATE_KEY][key] is not None:  # noqa: E501
                if key not in st.session_state:
                    st.session_state[key] = st.session_state[session_id][_PERSIST_STATE_KEY][key]  # noqa: E501


# prevent categories from being chosen in both multiselect
def update_grpa(session_id):
    _GRPA = f"grpa_{session_id}"
    _GRPB = f"grpb_{session_id}"
    if _GRPA not in st.session_state.keys():
        st.session_state[_GRPA] = []
    if _GRPB not in st.session_state.keys():
        st.session_state[_GRPB] = []
    if len(
        list(set(st.session_state[_GRPA]) &
             set(st.session_state[_GRPB]))
    ) > 0:
        item = list(
            set(st.session_state[_GRPA]) &
            set(st.session_state[_GRPB])
            )
        st.session_state[_GRPA] = list(
            set(list(st.session_state[_GRPA])) ^ set(item)
            )


def update_grpb(session_id):
    _GRPA = f"grpa_{session_id}"
    _GRPB = f"grpb_{session_id}"
    if _GRPA not in st.session_state.keys():
        st.session_state[_GRPA] = []
    if _GRPB not in st.session_state.keys():
        st.session_state[_GRPB] = []
    if len(
        list(set(st.session_state[_GRPA]) &
             set(st.session_state[_GRPB]))
    ) > 0:
        item = list(
            set(st.session_state[_GRPA]) &
            set(st.session_state[_GRPB])
            )
        st.session_state[_GRPB] = list(
            set(list(st.session_state[_GRPB])) ^ set(item)
            )


# prevent categories from being chosen in both multiselect
def update_tar(session_id):
    _TAR = f"tar_{session_id}"
    _REF = f"ref_{session_id}"
    if _TAR not in st.session_state.keys():
        st.session_state[_TAR] = []
    if _REF not in st.session_state.keys():
        st.session_state[_REF] = []
    if len(
        list(set(st.session_state[_TAR]) &
             set(st.session_state[_REF]))
    ) > 0:
        item = list(
            set(st.session_state[_TAR]) &
            set(st.session_state[_REF])
            )
        st.session_state[_TAR] = list(
            set(list(st.session_state[_TAR])) ^ set(item)
            )


def update_ref(session_id):
    _REF = f"ref_{session_id}"
    _TAR = f"tar_{session_id}"
    if _TAR not in st.session_state.keys():
        st.session_state[_TAR] = []
    if _REF not in st.session_state.keys():
        st.session_state[_REF] = []
    if len(
        list(set(st.session_state[_TAR]) &
             set(st.session_state[_REF]))
    ) > 0:
        item = list(
            set(st.session_state[_TAR]) &
            set(st.session_state[_REF])
            )
        st.session_state[_REF] = list(
            set(list(st.session_state[_REF])) ^ set(item)
            )
