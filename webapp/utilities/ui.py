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

# User interface utilities for Streamlit applications.

import operator
import pathlib
import sys
from typing import Literal
from functools import reduce

import plotly.colors
import polars as pl
import streamlit as st

# Ensure project root is in sys.path for both desktop and online
project_root = pathlib.Path(__file__).parent.parents[1].resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from webapp.utilities.handlers import update_session  # noqa: E402
from webapp.utilities.formatters import get_streamlit_column_config  # noqa: E402

DOCS_BASE_URL = "https://browndw.github.io/docuscope-docs/guide/"


def target_info(
        target_metadata: dict
        ) -> str:
    """
    Generate a string with information about the target corpus.
    This function extracts the number of part-of-speech tokens,
    DocuScope tokens, and documents from the target metadata.

    Parameters
    ----------
    target_metadata : dict
        Metadata dictionary containing information about the target corpus.
        Expected keys: 'tokens_pos', 'tokens_ds', 'ndocs'.

    Returns
    -------
    str
        A formatted string containing the target corpus information.
    """
    tokens_pos = target_metadata.get('tokens_pos')[0]
    tokens_ds = target_metadata.get('tokens_ds')[0]
    ndocs = target_metadata.get('ndocs')[0]
    target_info = f"""##### Target corpus information:

    Number of part-of-speech tokens in corpus: {tokens_pos:,}
    \n    Number of DocuScope tokens in corpus: {tokens_ds:,}
    \n    Number of documents in corpus: {ndocs:,}
    """
    return target_info


def reference_info(
        reference_metadata: dict
        ) -> str:
    """
    Generate a string with information about the reference corpus.
    This function extracts the number of part-of-speech tokens,
    DocuScope tokens, and documents from the reference metadata.

    Parameters
    ----------
    reference_metadata : dict
        Metadata dictionary containing information about the reference corpus.
        Expected keys: 'tokens_pos', 'tokens_ds', 'ndocs'.

    Returns
    -------
    str
        A formatted string containing the reference corpus information.
    """
    tokens_pos = reference_metadata.get('tokens_pos')[0]
    tokens_ds = reference_metadata.get('tokens_ds')[0]
    ndocs = reference_metadata.get('ndocs')[0]
    reference_info = f"""##### Reference corpus information:

    Number of part-of-speech tokens in corpus: {tokens_pos:,}
    \n    Number of DocuScope tokens in corpus: {tokens_ds:,}
    \n    Number of documents in corpus: {ndocs:,}
    """
    return reference_info


def correlation_info(cc_dict):
    """
    Formats correlation info for display in a code block for easy copy-paste.
    r is always shown to 3 decimal places, p to 5 decimal places.
    """
    def fmt_r(val):
        try:
            return f"{float(val):.3f}"
        except Exception:
            return str(val)

    def fmt_p(val):
        try:
            return f"{float(val):.5f}"
        except Exception:
            return str(val)

    lines = []
    if 'all' in cc_dict and cc_dict['all']:
        lines.append(
            f"All points: r({cc_dict['all']['df']}) = {fmt_r(cc_dict['all']['r'])}, p = {fmt_p(cc_dict['all']['p'])}"  # noqa: E501
        )
    if 'highlight' in cc_dict and cc_dict['highlight']:
        lines.append(
            f"Highlighted group: r({cc_dict['highlight']['df']}) = {fmt_r(cc_dict['highlight']['r'])}, p = {fmt_p(cc_dict['highlight']['p'])}"  # noqa: E501
        )
        if 'non_highlight' in cc_dict and cc_dict['non_highlight']:
            lines.append(
                f"Non-highlighted group: r({cc_dict['non_highlight']['df']}) = {fmt_r(cc_dict['non_highlight']['r'])}, p = {fmt_p(cc_dict['non_highlight']['p'])}"  # noqa: E501
            )
    lines_str = "\n    ".join(lines)
    corr_info = f"""##### Pearson's correlation coefficient:

    {lines_str}
    """
    return corr_info


def variance_info(
        pca_x: str,
        pca_y: str,
        ve_1: str,
        ve_2: str
        ) -> str:
    variance_info = f"""##### Variance explained:

    {pca_x}: {ve_1}\n    {pca_y}: {ve_2}
    """
    return variance_info


def contribution_info(
        pca_x: str,
        pca_y: str,
        contrib_x: str,
        contrib_y: str
        ) -> str:
    contrib_info = f"""##### Variables with contribution > mean:

    {pca_x}: {contrib_x}\n    {pca_y}: {contrib_y}
    """
    return contrib_info


def message_stats_info(
        stats: str | None = None
        ) -> str:
    stats_info = f"""##### Descriptive statistics:

    {stats}
    """
    return stats_info


def group_info(
        grp_a: list[str],
        grp_b: list[str]
        ) -> str:
    grp_a = [s.strip('_') for s in grp_a]
    grp_a = ", ".join(str(x) for x in grp_a)
    grp_b = [s.strip('_') for s in grp_b]
    grp_b = ", ".join(str(x) for x in grp_b)
    group_info = f"""##### Grouping variables:

    Group A: {grp_a}\n    Group B: {grp_b}
    """
    return group_info


def target_parts(
        keyness_parts: list[str]
        ) -> str:
    t_cats = keyness_parts[0]
    tokens_pos = keyness_parts[2]
    tokens_ds = keyness_parts[4]
    ndocs = keyness_parts[6]
    target_info = f"""##### Target corpus information:

    Document categories: {t_cats}
    \n    Part-of-speech tokens: {tokens_pos}
    \n    DocuScope tokens: {tokens_ds}
    \n    Documents: {ndocs}
    """
    return target_info


def reference_parts(
        keyness_parts: list[str]
        ) -> str:
    r_cats = keyness_parts[1]
    tokens_pos = keyness_parts[3]
    tokens_ds = keyness_parts[5]
    ndocs = keyness_parts[7]
    reference_info = f"""##### Reference corpus information:

    Document categories: {r_cats}
    \n    Part-of-speech tokens: {tokens_pos}
    \n    DocuScope tokens: {tokens_ds}
    \n    Documents: {ndocs}
    """
    return reference_info


def render_dataframe(
        df: pl.DataFrame | None = None,
        column_config: dict | None = None,
        use_container_width: bool = True,
        num_rows: Literal['fixed', 'dynamic'] = 'dynamic',
        disabled: bool = True
        ) -> None:
    """
    Render a Polars DataFrame in Streamlit using the data editor.

    Parameters
    ----------
    df : pl.DataFrame, optional
        The DataFrame to render. If None, no data will be displayed.
    column_config : dict, optional
        Configuration for the DataFrame columns.
        If None, defaults to a configuration generated from the DataFrame.
    use_container_width : bool
        If True, the DataFrame will use the full width of the container.
    num_rows : Literal['fixed', 'dynamic']
        How many rows to display in the DataFrame.
        'fixed' shows a fixed number of rows, 'dynamic' adjusts based on content.
    disabled : bool
        If True, the DataFrame will be rendered in a read-only mode.
        If False, it will be editable.

    Returns
    -------
    None
        This function does not return anything.
        It renders the DataFrame directly in the Streamlit app.
    """
    if column_config is None and df is not None:
        column_config = get_streamlit_column_config(df)
    if df is not None and getattr(df, "height", 0) > 0:
        st.data_editor(
            df,
            hide_index=True,
            column_config=column_config,
            use_container_width=use_container_width,
            num_rows=num_rows,
            disabled=disabled
        )
    else:
        st.warning("No data to display.")


def keyness_sort_controls(
        sort_options: list[str] = ["Keyness (LL)", "Effect Size (LR)"],
        default: str = "Keyness (LL)",
        reverse_default: bool = True,
        key_prefix: str = ""
        ) -> tuple[str, bool]:
    """
    Render radio buttons for sorting keyness tables and sort order.

    Returns
    -------
    sort_by : str
        The selected column to sort by.
    reverse : bool
        Whether to reverse the sort order (descending).
    """
    col1, col2 = st.columns([2, 1])
    with col1:
        sort_by = st.radio(
            "Sort by:",
            sort_options,
            horizontal=True,
            index=sort_options.index(default) if default in sort_options else 0,
            key=f"{key_prefix}keyness_sort_by"
        )
    with col2:
        order = st.radio(
            "Sort order:",
            options=["Descending", "Ascending"],
            horizontal=True,
            index=0 if reverse_default else 1,
            key=f"{key_prefix}keyness_sort_order"
        )
        reverse = order == "Descending"
    return sort_by, reverse


def sidebar_help_link(
        doc_page: str,
        label: str = "Help",
        icon: str = ":material/help:"
        ) -> None:
    """
    Render a styled help link at the top of the sidebar.

    Parameters
    ----------
    doc_page : str
        The page-specific part of the documentation URL (e.g., "token-frequencies.html").
    label : str
        The label for the help link.
    icon : str
        The icon to display with the help link.

    Returns
    -------
    None
        This function does not return anything.
        It renders a link in the sidebar that navigates to the documentation page.
    """
    st.sidebar.link_button(
        label=label,
        url=f"{DOCS_BASE_URL}{doc_page}",
        icon=icon
    )
    st.sidebar.markdown("<div style='margin-bottom: 0.5em'></div>", unsafe_allow_html=True)


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
    if container.button(button_label, icon=button_icon, type="primary"):
        if not all(preconditions):
            st.error(
                    body=(
                        "It doesn't look like you've loaded the necessary data yet. "
                        "Most apps require a target corpus to be loaded "
                        "before you can run them. "
                        "But other apps may require a reference corpus "
                        "or the processing of metadata.  "
                        "You can review and update your corpus data "
                        " by navigating to **Manage Corpus Data**:"
                        ),
                    icon=":material/sentiment_stressed:"
                )
            # Add a direct page link for user convenience
            st.page_link(
                page="pages/1_load_corpus.py",
                label="Manage Corpus Data",
                icon=":material/database:",
            )
            return
        with container:
            with st.spinner(spinner_message):
                action()


def sidebar_keyness_options(
        user_session_id: str,
        load_metadata_func,
        token_limit: int = 1_500_000,
        sidebar=None,
        target_key: str = 'target',
        reference_key: str = 'reference'
        ) -> tuple[float, bool]:
    """
    Render sidebar widgets for p-value threshold and swap target/reference.
    Returns the selected p-value and swap state.
    """
    if sidebar is None:
        sidebar = st.sidebar

    # Set defaults if not in session state
    if "pval_threshold" not in st.session_state[user_session_id]:
        st.session_state[user_session_id]["pval_threshold"] = 0.01
    if "swap_target" not in st.session_state[user_session_id]:
        st.session_state[user_session_id]["swap_target"] = False

    # Load metadata for size check
    try:
        metadata_target = load_metadata_func(target_key, user_session_id)
        target_tokens = metadata_target.get('tokens_pos', [0])[0] if metadata_target else 0
    except Exception:
        target_tokens = 0

    try:
        metadata_reference = load_metadata_func(reference_key, user_session_id)
        reference_tokens = metadata_reference.get('tokens_pos', [0])[0] if metadata_reference else 0  # noqa: E501
    except Exception:
        reference_tokens = 0

    if target_tokens > token_limit or reference_tokens > token_limit:
        pval_options = [0.05, 0.01]
        sidebar.warning(
            "Corpora are large (>1.5 million tokens). "
            "p < .001 is disabled to prevent memory issues."
        )
    elif target_tokens == 0 or reference_tokens == 0:
        pval_options = []
    else:
        pval_options = [0.05, 0.01, 0.001]

    # Select p-value threshold
    pval_idx = (
        pval_options.index(st.session_state[user_session_id]["pval_threshold"])
        if st.session_state[user_session_id]["pval_threshold"] in pval_options
        else 1
    )

    sidebar.markdown("### Select threshold")
    pval_selected = sidebar.selectbox(
        "p-value threshold",
        options=pval_options,
        format_func=lambda x: f"{x:.3f}" if x < 0.01 else f"{x:.2f}",
        index=pval_idx,
        key=f"pval_threshold_{user_session_id}",
        help=(
            "Select the p-value threshold for keyness analysis. "
            "Lower values are more stringent, but may be useful for larger corpora. "
            "For smaller corpora, a threshold of 0.05 is often sufficient."
        ),
    )
    sidebar.markdown("---")

    st.session_state[user_session_id]["pval_threshold"] = pval_selected

    swap_selected = False
    if target_tokens > 0 and reference_tokens > 0:
        sidebar.markdown("### Swap target/reference corpora")
        swap_selected = sidebar.toggle(
            "Swap Target/Reference",
            value=st.session_state[user_session_id]["swap_target"],
            key=f"swap_target_{user_session_id}",
            help=(
                "If selected, the target corpus will be used as the reference "
                "and the reference corpus will be used as the target for keyness analysis. "
                "This will show what is more frequent in the reference corpus "
                "compared to the target corpus. "
            ),
        )
        sidebar.markdown("---")
        st.session_state[user_session_id]["swap_target"] = swap_selected

    return pval_selected, swap_selected


def tag_filter_multiselect(
        df,
        tag_col="Tag",
        label="Select tags to filter:",
        key=None
        ) -> pl.DataFrame | None:
    """
    Render a segmented control widget (inside an expander) for tag filtering and
    return the filtered DataFrame.
    """
    if df is None or getattr(df, "height", 0) == 0:
        return df
    cats = sorted(df.get_column(tag_col).drop_nulls().unique().to_list())
    if not cats:
        return df
    seg_key = key or f"seg_{tag_col}"
    with st.expander(label):
        if st.button("Deselect All", key=f"{seg_key}_deselect"):
            st.session_state[seg_key] = []
        selected = st.segmented_control(
            f"Select {tag_col}:",
            options=cats,
            selection_mode="multi",
            key=seg_key,
            help="Click to filter by one or more tags. Click again to deselect."
        )
    if selected is None or len(selected) == 0:
        return df
    df = df.filter(pl.col(tag_col).is_in(selected))
    return df


def multi_tag_filter_multiselect(
        df: pl.DataFrame,
        tag_cols: list[str]
        ) -> tuple[pl.DataFrame, dict]:
    """
    Render segmented control widgets (inside expanders) for multiple tag columns and
    return the filtered DataFrame and selections.
    """
    filter_conditions = []
    filter_selections = {}
    for tag_col in tag_cols:
        cats = sorted(df.get_column(tag_col).drop_nulls().unique().to_list())
        seg_key = f"filter_{tag_col}"
        if not cats:
            selected = []
        else:
            with st.expander(f"Filter {tag_col}"):
                if st.button("Deselect All", key=f"{seg_key}_deselect"):
                    st.session_state[seg_key] = []
                selected = st.segmented_control(
                    f"Select {tag_col}:",
                    options=cats,
                    selection_mode="multi",
                    key=seg_key,
                    help="Click to filter by one or more tags. Click again to deselect."
                )
        filter_selections[tag_col] = selected
        if selected:
            filter_conditions.append(pl.col(tag_col).is_in(selected))
    if filter_conditions:
        combined_filter = reduce(operator.and_, filter_conditions)
        df = df.filter(combined_filter)
    return df, filter_selections


def tagset_selection(
        user_session_id: str,
        session_state: dict,
        persist_func: callable,
        page_stem: str,
        tagset_keys: dict = None,
        simplify_funcs: dict = None,
        tag_filters: dict = None,
        tag_radio_key: str = "tag_radio",
        tag_type_key: str = "tag_type_radio",
        on_change=None,
        on_change_args=None
        ) -> tuple:
    """
    Modular sidebar UI for tagset selection,
    supporting custom keys, filters, and simplify functions.

    Returns
    -------
    df : DataFrame or None
        The selected DataFrame.
    tag_options : list
        List of unique tags in the DataFrame (empty if df is None).
    tag_radio : str
        The selected tagset.
    tag_type : str or None
        The selected tag type (if applicable).
    """
    tagset_keys = tagset_keys or {
        "Parts-of-Speech": {"General": "ft_pos", "Specific": "ft_pos"},
        "DocuScope": "ft_ds"
    }
    simplify_funcs = simplify_funcs or {}
    tag_filters = tag_filters or {}

    tag_radio = st.sidebar.radio(
        "Select tags to display:",
        list(tagset_keys.keys()),
        key=persist_func(tag_radio_key, page_stem, user_session_id),
        horizontal=True,
        help=(
            "Select Parts-of-Speech for syntactic analysis, "
            "or DocuScope for rhetorical analysis. "
            "If you select Parts-of-Speech, you can choose between "
            "general (for the full CLAWS7 tagset) "
            "or specific tags (for a simplified, collapsed tagset). "
        ),
        on_change=on_change,
        args=on_change_args
    )

    tag_type = None
    df = None

    # Handle subtypes (e.g., General/Specific)
    if isinstance(tagset_keys[tag_radio], dict):
        tag_type = st.sidebar.radio(
            "Select from general or specific tags",
            list(tagset_keys[tag_radio].keys()),
            key=persist_func(tag_type_key, page_stem, user_session_id),
            horizontal=True,
            on_change=on_change,
            args=on_change_args
        )
        session_key = tagset_keys[tag_radio][tag_type]
        df = session_state[user_session_id]["target"].get(session_key)
        # Apply simplify function if provided
        simplify_func = simplify_funcs.get(tag_radio, {}).get(tag_type)
        if simplify_func and df is not None:
            df = simplify_func(df)
        # Apply filter if provided
        tag_filter = tag_filters.get(tag_radio, {}).get(tag_type)
        if tag_filter and df is not None:
            df = tag_filter(df) if callable(tag_filter) else df.filter(~pl.col("Tag").is_in(tag_filter))  # noqa: E501
    else:
        session_key = tagset_keys[tag_radio]
        df = session_state[user_session_id]["target"].get(session_key)
        # Apply simplify function if provided
        simplify_func = simplify_funcs.get(tag_radio)
        if simplify_func and df is not None:
            df = simplify_func(df)
        # Apply filter if provided
        tag_filter = tag_filters.get(tag_radio)
        if tag_filter and df is not None:
            df = tag_filter(df) if callable(tag_filter) else df.filter(~pl.col("Tag").is_in(tag_filter))  # noqa: E501

    # Get tag options
    tag_options = []
    if df is not None and hasattr(df, "get_column"):
        try:
            tag_options = sorted(df.get_column("Tag").unique().to_list())
        except Exception:
            tag_options = []

    return df, tag_options, tag_radio, tag_type


def toggle_download(
        label: str,
        convert_func,
        convert_args: tuple = (),
        convert_kwargs: dict = None,
        file_name: str = "download.bin",
        mime: str = "application/octet-stream",
        message: str = "Your data is ready!",
        location=st.sidebar
        ) -> None:
    """
    Generalized toggle-based download for Streamlit.

    Parameters
    ----------
    label : str
        The label for the toggle and download button.
    convert_func : callable
        The function to convert data to bytes.
    convert_args : tuple
        Positional arguments for the conversion function.
    convert_kwargs : dict
        Keyword arguments for the conversion function.
    file_name : str
        The name of the file to download.
    mime : str
        The MIME type of the file.
    message : str
        Optional markdown message to display above the button.
    location : Streamlit container
        Where to place the toggle and download button (default: st.sidebar).
    """
    convert_kwargs = convert_kwargs or {}
    toggle_key = f"toggle_{label.replace(' ', '_')}"
    download = location.toggle(f"Download to {label}?", key=toggle_key)
    if download:
        if message:
            location.success(message,
                             icon=":material/celebration:")
        data = convert_func(*convert_args, **convert_kwargs)
        location.download_button(
            label=f"Download to {label}",
            data=data,
            file_name=file_name,
            mime=mime,
            icon=":material/download:"
        )


def persist(
        key: str,
        app_name: str,
        session_id: str
        ) -> str:
    """
    Persist a widget state across sessions.
    This function checks if the key exists in the session state,
    and if not, initializes it with None.
    If the key exists, it updates the session state with the current value.

    Parameters
    ----------
    key : str
        The key to persist in the session state.
    app_name : str
        The name of the application, used to create a unique session state key.
    session_id : str
        The session ID for the current user session.

    Returns
    -------
    str
        The key that was persisted.
    """
    _PERSIST_STATE_KEY = f"{app_name}_PERSIST"
    if _PERSIST_STATE_KEY not in st.session_state[session_id].keys():
        st.session_state[session_id][_PERSIST_STATE_KEY] = {}
        st.session_state[session_id][_PERSIST_STATE_KEY][key] = None

    if key in st.session_state:
        st.session_state[session_id][_PERSIST_STATE_KEY][key] = st.session_state[key]  # noqa: E501

    return key


def load_widget_state(
        app_name: str,
        session_id: str
        ) -> None:
    """
    Load persistent widget state from the session state.
    This function checks if the persistent state key exists in the session state,
    and if it does, it loads the values into the current session state.
    If the key does not exist, it initializes the persistent state with None.

    Parameters
    ----------
    app_name : str
        The name of the application, used to create a unique session state key.
    session_id : str
        The session ID for the current user session.

    Returns
    -------
    None
    """
    _PERSIST_STATE_KEY = f"{app_name}_PERSIST"
    """Load persistent widget state."""
    if _PERSIST_STATE_KEY in st.session_state[session_id]:
        for key in st.session_state[session_id][_PERSIST_STATE_KEY]:
            if st.session_state[session_id][_PERSIST_STATE_KEY][key] is not None:  # noqa: E501
                if key not in st.session_state:
                    st.session_state[key] = st.session_state[session_id][_PERSIST_STATE_KEY][key]  # noqa: E501


def update_pca_idx_tab1(
        session_id: str
        ) -> None:
    """
    Update the PCA index for tab 1.
    This function initializes the selectbox state for PCA index in tab 1
    if it doesn't exist, and updates the shared PCA index in the session state.
    Parameters
    ----------
    session_id : str
        The session ID for the current user session.
    Returns
    -------
    None
    """
    # Initialize the selectbox state if it doesn't exist
    if f"pca_idx_tab1_{session_id}" not in st.session_state:
        st.session_state[f"pca_idx_tab1_{session_id}"] = st.session_state[session_id].get('pca_idx', 1)  # noqa: E501
    # Now update the shared PC index
    st.session_state[session_id]['pca_idx'] = st.session_state[f"pca_idx_tab1_{session_id}"]


def update_pca_idx_tab2(
        session_id: str
        ) -> None:
    # Initialize the selectbox state if it doesn't exist
    if f"pca_idx_tab2_{session_id}" not in st.session_state:
        st.session_state[f"pca_idx_tab2_{session_id}"] = st.session_state[session_id].get('pca_idx', 1)  # noqa: E501
    # Now update the shared PC index
    st.session_state[session_id]['pca_idx'] = st.session_state[f"pca_idx_tab2_{session_id}"]


# prevent categories from being chosen in both multiselect
def update_grpa(
        session_id: str
        ) -> None:
    """
    Prevent categories from being chosen in both multiselects for group A.
    This function checks if the selected items in group A and group B
    overlap, and if so, removes the overlapping items from group A.
    Parameters
    ----------
    session_id : str
        The session ID for the current user session.
    Returns
    -------
    None
    """
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


def update_grpb(
        session_id: str
        ) -> None:
    """
    Prevent categories from being chosen in both multiselects for group B.
    This function checks if the selected items in group A and group B
    overlap, and if so, removes the overlapping items from group B.
    Parameters
    ----------
    session_id : str
        The session ID for the current user session.
    Returns
    -------
    None
    """
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


def update_tar(
        session_id: str
        ) -> None:
    """
    Prevent categories from being chosen in both target and reference multiselects.
    This function checks if the selected items in target and reference
    overlap, and if so, removes the overlapping items from target.
    Parameters
    ----------
    session_id : str
        The session ID for the current user session.
    Returns
    -------
    None
    """
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


def update_ref(
        session_id: str
        ) -> None:
    """
    Prevent categories from being chosen in both target and reference multiselects.
    This function checks if the selected items in target and reference
    overlap, and if so, removes the overlapping items from reference.
    Parameters
    ----------
    session_id : str
        The session ID for the current user session.
    Returns
    -------
    None
    """
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


def clear_boxplot_multiselect(user_session_id: str) -> None:
    """
    Clear the boxplot multiselects and reset related session state.
    This function resets the boxplot variable selections and clears
    any associated DataFrames, statistics, and widget state in the session state.
    """
    if user_session_id not in st.session_state:
        return

    # Clear DataFrames, stats, warnings, and selected variables/groups
    keys = [
        "boxplot_df", "boxplot_stats", "boxplot_warning",
        "boxplot_group_df", "boxplot_group_stats", "boxplot_group_warning"
    ]
    for key in keys:
        st.session_state[user_session_id][key] = None

    # Also clear widget keys related to boxplot UI,
    # including all segmented controls and buttons
    widget_keys = [
        f"grpa_{user_session_id}",
        f"grpb_{user_session_id}",
        f"boxplot_btn_{user_session_id}",
        f"boxplot_group_btn_{user_session_id}",
        f"boxplot_vars_grouped_{user_session_id}",
        f"boxplot_vars_nongrouped_{user_session_id}",
        # Add color picker keys if used
        f"color_picker_boxplot_{user_session_id}_cat_0",
        f"color_picker_boxplot_{user_session_id}_cat_1",
        # If you use more color pickers, add their keys here
    ]
    for wkey in widget_keys:
        if wkey in st.session_state:
            del st.session_state[wkey]


def clear_scatterplot_multiselect(user_session_id: str) -> None:
    """
    Clear the scatterplot multiselects and reset related session state.
    This function resets the scatterplot variable selections and clears
    any associated DataFrames, statistics, and widget state in the session state.
    """
    if user_session_id not in st.session_state:
        return

    # Clear DataFrames, stats, warnings, and selected variables/groups
    keys = [
        "scatterplot_df", "scatter_correlation", "scatter_warning",
        "scatterplot_group_df", "scatter_group_correlation", "scatter_group_warning",
        "scatterplot_group_x", "scatterplot_group_y", "scatterplot_group_selected_groups",
        "scatterplot_nongrouped_x", "scatterplot_nongrouped_y"
    ]
    for key in keys:
        st.session_state[user_session_id][key] = None

    # Also clear widget keys related to scatterplot UI,
    # including all segmented controls and buttons
    widget_keys = [
        f"scatterplot_btn_{user_session_id}",
        f"scatterplot_group_btn_{user_session_id}",
        f"scatter_x_grouped_{user_session_id}",
        f"scatter_y_grouped_{user_session_id}",
        f"scatter_x_nongrouped_{user_session_id}",
        f"scatter_y_nongrouped_{user_session_id}",
        f"highlight_scatter_groups_{user_session_id}",
        # Add color picker keys if used
        f"color_picker_scatter_{user_session_id}_Highlight_0",
        f"color_picker_scatter_{user_session_id}_Non-Highlight_1",
        f"color_picker_scatter_{user_session_id}_All_Points_0",
    ]
    for wkey in widget_keys:
        if wkey in st.session_state:
            del st.session_state[wkey]


def clear_plots(
        session_id: str
        ) -> None:
    """
    Clear all plot-related session state for the given user session.

    This includes group selections, boxplot and scatterplot variables,
    highlight selections, DataFrames, statistics, warnings, PCA data,
    and color picker/segmented control widget states.
    """
    if session_id not in st.session_state:
        return

    update_session('pca', False, session_id)
    _GRPA = f"grpa_{session_id}"
    _GRPB = f"grpb_{session_id}"
    _BOXPLOT_VARS = f"boxplot_vars_{session_id}"

    # Clear group selections
    for key in [_GRPA, _GRPB, _BOXPLOT_VARS]:
        if key in st.session_state:
            st.session_state[key] = []

    # Clear highlight multiselects
    highlight_keys = [
        f"highlight_pca_groups_{session_id}",
        f"highlight_scatter_groups_{session_id}",
        # add other highlight keys as needed
    ]
    for key in highlight_keys:
        if key in st.session_state:
            st.session_state[key] = []

    # Clear plot results and warnings, and remove 'Highlight' column
    if session_id in st.session_state:
        for key in [
            "boxplot_df", "boxplot_group_df", "scatterplot_df", "scatterplot_group_df"
        ]:
            df = st.session_state[session_id].get(key)
            if df is not None and hasattr(df, "columns") and "Highlight" in df.columns:
                st.session_state[session_id][key] = df.drop(columns=["Highlight"])
        keys_to_clear = [
            "boxplot_df", "boxplot_stats", "boxplot_warning",
            "boxplot_group_df", "boxplot_group_stats", "boxplot_group_warning",
            "scatterplot_df", "scatter_correlation", "scatter_warning",
            "scatterplot_group_df", "scatter_group_correlation", "scatter_group_warning"
        ]
        for key in keys_to_clear:
            st.session_state[session_id][key] = None
        # --- Clear PCA data and warnings ---
        if "target" in st.session_state[session_id]:
            st.session_state[session_id]["target"]["pca_df"] = None
            st.session_state[session_id]["target"]["contrib_df"] = None
        st.session_state[session_id]["pca_warning"] = None
        if "pca_idx" in st.session_state[session_id]:
            st.session_state[session_id]["pca_idx"] = 1

    # --- Clear color picker and segmented control widget states ---
    widget_prefixes = [
        "color_picker_form_", "seg_", "filter_", "highlight_", "toggle_", "download_", "boxplot_vars_"  # noqa: E501
    ]
    keys_to_remove = [k for k in st.session_state.keys()
                      if any(k.startswith(prefix) for prefix in widget_prefixes)]
    for k in keys_to_remove:
        del st.session_state[k]
    # --- Clear boxplot and scatterplot multiselects ---
    clear_boxplot_multiselect(session_id)
    clear_scatterplot_multiselect(session_id)


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


def rgb_to_hex(rgb_str):
    if rgb_str.startswith("rgb"):
        nums = rgb_str[rgb_str.find("(")+1:rgb_str.find(")")].split(",")
        return "#{:02x}{:02x}{:02x}".format(*(int(float(n)) for n in nums))
    return rgb_str


def color_picker_controls(
        cats: list[str] = None,
        default_hex: str = "#133955",
        default_palette: str = "Plotly",
        expander_label: str = "Plot Colors",
        key_prefix: str = "color_picker_form",
        non_highlight_default: str = "#d3d3d3",
        reference_corpus_default: str = "#e67e22"
        ) -> dict:
    """
    Modular color picker controls for per-category coloring.
    Returns a dict: {category: hex_color}
    key_prefix: a string to ensure unique Streamlit widget keys.
    """

    # Get qualitative palettes, omitting any that end with '_r' except 'Alphabet'
    qualitative_palettes = [
        p for p in dir(plotly.colors.qualitative)
        if not p.startswith("_")
        and isinstance(getattr(plotly.colors.qualitative, p), list)
        and (not p.endswith("_r") or p == "Alphabet")
    ]

    # Add sequential palettes (flat list, not dicts), omitting any that end with '_r'
    sequential_palettes = [
        p for p in dir(plotly.colors.sequential)
        if not p.startswith("_")
        and isinstance(getattr(plotly.colors.sequential, p), list)
        and (not p.endswith("_r"))
    ]

    # Combine and sort palettes alphabetically
    plotly_palettes = sorted(qualitative_palettes + sequential_palettes)

    if not cats:
        cats = ["All"]

    color_mode_key = f"{key_prefix}_mode"
    palette_key = f"{key_prefix}_palette"

    color_dict = {}

    with st.expander(expander_label):
        color_mode = st.radio(
            "Color mode",
            ["Custom (pick colors)", "Plotly palette"],
            horizontal=True,
            key=color_mode_key
        )

        if color_mode == "Custom (pick colors)":
            prev_color = default_hex
            seen_keys = set()
            for idx, cat in enumerate(cats):
                # Set special defaults for certain categories
                if cat.lower() == "non-highlight":
                    color_default = non_highlight_default
                elif cat.lower() == "reference corpus":
                    color_default = reference_corpus_default
                else:
                    color_default = prev_color
                safe_cat = str(cat).replace(" ", "_").replace(",", "_").replace("/", "_")
                if not safe_cat:
                    safe_cat = f"cat_{idx}"
                # Ensure uniqueness even if cats has duplicates or empty strings
                color_key = f"{key_prefix}_{safe_cat}_{idx}"
                while color_key in seen_keys:
                    color_key = f"{key_prefix}_{safe_cat}_{idx}_{len(seen_keys)}"
                seen_keys.add(color_key)
                color = st.color_picker(
                    f"Color for {cat}",
                    value=st.session_state.get(color_key, color_default),
                    key=color_key
                )
                color_dict[cat] = color
                prev_color = color  # Default next color to previous
        else:
            palette = st.selectbox(
                "Plotly palette",
                plotly_palettes,
                index=plotly_palettes.index(default_palette) if default_palette in plotly_palettes else 0,  # noqa: E501
                key=palette_key
            )
            palette_colors_raw = getattr(plotly.colors.qualitative, palette, None) or getattr(plotly.colors.sequential, palette, None)  # noqa: E501
            palette_colors = [rgb_to_hex(c) for c in palette_colors_raw] if palette_colors_raw else [default_hex]  # noqa: E501
            prev_color = palette_colors[0] if palette_colors else default_hex
            seen_keys = set()
            for idx, cat in enumerate(cats):
                safe_cat = str(cat).replace(" ", "_").replace(",", "_").replace("/", "_")
                if not safe_cat:
                    safe_cat = f"cat_{idx}"
                color_key = f"{key_prefix}_{safe_cat}_{idx}"
                while color_key in seen_keys:
                    color_key = f"{key_prefix}_{safe_cat}_{idx}_{len(seen_keys)}"
                seen_keys.add(color_key)
                default_idx = (
                    palette_colors.index(st.session_state.get(color_key, prev_color))
                    if st.session_state.get(color_key, prev_color) in palette_colors
                    else idx % len(palette_colors)
                )
                color = st.segmented_control(
                    f"Color for {cat}",
                    options=palette_colors,
                    default=palette_colors[default_idx],
                    selection_mode="single",
                    key=color_key
                )
                color_dict[cat] = color
                prev_color = color  # Default next color to previous

    return color_dict
