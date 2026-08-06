"""
Form control utilities for user interface components.

This module provides complex form controls like tagset selection,
toggle downloads, and other interactive UI components.
"""

import streamlit as st
import polars as pl
import operator
import plotly.colors
from collections import OrderedDict
from functools import reduce
from threading import Lock
from time import perf_counter
from typing import Dict, Optional, Callable, Tuple

# Import widget key manager for centralized widget management
from webapp.utilities.state.widget_key_manager import (
    register_persistent_widgets,
    get_widget_state,
)

# Import corpus data manager for unified data access
from webapp.utilities.corpus import get_corpus_data
from webapp.utilities.configuration.logging_config import get_logger

SLOW_FORM_CONTROL_MS = 100
SIMPLIFIED_FRAME_CACHE_MAX_ITEMS = 16
TAGSET_FRAME_CACHE_MAX_ITEMS = 64
SimplifiedFrameCacheKey = tuple[int, str, str, int]
_simplified_frame_cache: OrderedDict[SimplifiedFrameCacheKey, pl.DataFrame] = OrderedDict()
_simplified_frame_cache_locks: dict[SimplifiedFrameCacheKey, Lock] = {}
_simplified_frame_cache_locks_guard = Lock()
_tagset_frame_cache: OrderedDict[tuple, pl.DataFrame] = OrderedDict()
logger = get_logger()

# Register persistent widgets used across form controls
# These widgets should persist across page loads
PERSISTENT_FORM_WIDGETS = [
    "tag_radio",           # Main tagset selection radio
    "tag_type_radio",      # Sub-tagset selection (General/Specific)
    "pval_threshold",      # p-value threshold setting
    "swap_target",         # Target/reference swap setting
]

# Register the persistent widgets
register_persistent_widgets(PERSISTENT_FORM_WIDGETS)


def _frame_shape(df) -> tuple[int, int]:
    if df is None:
        return 0, 0

    return getattr(df, "height", 0), getattr(df, "width", 0)


def _log_slow_form_control(
    operation: str,
    start_time: float,
    user_session_id: str | None,
    df=None,
) -> None:
    elapsed_ms = (perf_counter() - start_time) * 1000
    if elapsed_ms < SLOW_FORM_CONTROL_MS:
        return

    height, width = _frame_shape(df)
    logger.warning(
        "Slow form control step op={} session={} rows={} cols={} duration_ms={:.2f}",
        operation,
        user_session_id or "unknown",
        height,
        width,
        elapsed_ms,
    )


def _get_cached_simplified_frame(
    cache_key: SimplifiedFrameCacheKey,
) -> pl.DataFrame | None:
    cached = _simplified_frame_cache.get(cache_key)
    if cached is None:
        return None

    _simplified_frame_cache.move_to_end(cache_key)
    return cached


def _set_cached_simplified_frame(
    cache_key: SimplifiedFrameCacheKey,
    df: pl.DataFrame,
) -> None:
    _simplified_frame_cache[cache_key] = df
    _simplified_frame_cache.move_to_end(cache_key)

    while len(_simplified_frame_cache) > SIMPLIFIED_FRAME_CACHE_MAX_ITEMS:
        _simplified_frame_cache.popitem(last=False)


def _get_simplified_frame_lock(cache_key: SimplifiedFrameCacheKey) -> Lock:
    with _simplified_frame_cache_locks_guard:
        lock = _simplified_frame_cache_locks.get(cache_key)
        if lock is None:
            lock = Lock()
            _simplified_frame_cache_locks[cache_key] = lock
        return lock


def _get_tagset_frame_signature(user_session_id: str, session_key: str) -> tuple:
    session_container = st.session_state.get(user_session_id, {})
    target_data = session_container.get("target", {})
    if not isinstance(target_data, dict):
        return (user_session_id, session_key, "missing")

    refs = target_data.get("_artifact_refs")
    ref_key = "ft_pos" if session_key == "ft_pos_general" else session_key
    if isinstance(refs, dict) and ref_key in refs:
        ref = refs[ref_key]
        if isinstance(ref, dict):
            return (
                user_session_id,
                session_key,
                "artifact",
                ref_key,
                ref.get("artifact_type"),
                ref.get("artifact_id"),
                ref.get("storage_type"),
                ref.get("path"),
            )

    data = target_data.get(session_key)
    if data is None and ref_key != session_key:
        data = target_data.get(ref_key)
    return (user_session_id, session_key, "session", id(data))


def _get_cached_tagset_frame(cache_key: tuple) -> pl.DataFrame | None:
    cached = _tagset_frame_cache.get(cache_key)
    if cached is None:
        return None

    _tagset_frame_cache.move_to_end(cache_key)
    return cached


def _set_cached_tagset_frame(cache_key: tuple, df: pl.DataFrame) -> None:
    _tagset_frame_cache[cache_key] = df
    _tagset_frame_cache.move_to_end(cache_key)

    while len(_tagset_frame_cache) > TAGSET_FRAME_CACHE_MAX_ITEMS:
        _tagset_frame_cache.popitem(last=False)


def _get_tagset_frame(user_session_id: str, session_key: str) -> pl.DataFrame | None:
    cache_key = _get_tagset_frame_signature(user_session_id, session_key)
    cached = _get_cached_tagset_frame(cache_key)
    if cached is not None:
        return cached

    df = get_corpus_data(user_session_id, "target", session_key)
    if df is not None:
        _set_cached_tagset_frame(cache_key, df)
    return df


def _apply_cached_simplify(
    df: pl.DataFrame,
    source_key: str,
    cache_label: str,
    simplify_func: Callable,
) -> pl.DataFrame:
    cache_key = (id(df), source_key, cache_label, id(simplify_func))
    cached = _get_cached_simplified_frame(cache_key)
    if cached is not None:
        return cached

    simplify_lock = _get_simplified_frame_lock(cache_key)
    with simplify_lock:
        cached = _get_cached_simplified_frame(cache_key)
        if cached is not None:
            return cached

        simplified_df = simplify_func(df)
        _set_cached_simplified_frame(cache_key, simplified_df)
        return simplified_df


def tagset_selection(
    user_session_id: str,
    session_state: dict,
    persist_func: Callable,
    tagset_keys: Optional[Dict] = None,
    simplify_funcs: Optional[Dict] = None,
    tag_filters: Optional[Dict] = None,
    tag_radio_key: str = "tag_radio",
    tag_type_key: str = "tag_type_radio",
    include_tag_options: bool = True,
    on_change: Optional[Callable] = None,
    on_change_args: Optional[Tuple] = None
) -> Tuple[Optional[pl.DataFrame], list, str, Optional[str]]:
    """
    Modular sidebar UI for tagset selection,
    supporting custom keys, filters, and simplify functions.

    Parameters
    ----------
    user_session_id : str
        The user session ID.
    session_state : dict
        The session state dictionary.
    persist_func : Callable
        Function to persist widget state. The function should auto-detect page stem.
    tagset_keys : Optional[Dict], default=None
        Dictionary mapping tagset names to session keys.
    simplify_funcs : Optional[Dict], default=None
        Dictionary mapping tagsets to simplification functions.
    tag_filters : Optional[Dict], default=None
        Dictionary mapping tagsets to filter functions or lists.
    tag_radio_key : str, default="tag_radio"
        Key for the tagset radio button.
    tag_type_key : str, default="tag_type_radio"
        Key for the tag type radio button.
    include_tag_options : bool, default=True
        Whether to derive and return unique tag options from the selected frame.
    on_change : Optional[Callable], default=None
        Callback function for radio button changes.
    on_change_args : Optional[Tuple], default=None
        Arguments for the callback function.

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
    total_start = perf_counter()

    tag_radio = st.sidebar.radio(
        "Select tags to display:",
        list(tagset_keys.keys()),
        key=persist_func(tag_radio_key, user_session_id),
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
            key=persist_func(tag_type_key, user_session_id),
            horizontal=True,
            on_change=on_change,
            args=on_change_args
        )
        session_key = tagset_keys[tag_radio][tag_type]

        # Use corpus data manager for unified data access
        get_data_start = perf_counter()
        df = _get_tagset_frame(user_session_id, session_key)
        _log_slow_form_control(
            f"tagset_selection_get_data key={session_key}",
            get_data_start,
            user_session_id,
            df,
        )

        # Apply simplify function if provided
        simplify_func = simplify_funcs.get(tag_radio, {}).get(tag_type)
        if simplify_func and df is not None:
            simplify_start = perf_counter()
            df = _apply_cached_simplify(
                df,
                session_key,
                f"{tag_radio}:{tag_type}",
                simplify_func,
            )
            _log_slow_form_control(
                f"tagset_selection_simplify tagset={tag_radio} subtype={tag_type}",
                simplify_start,
                user_session_id,
                df,
            )
        # Apply filter if provided
        tag_filter = tag_filters.get(tag_radio, {}).get(tag_type)
        if tag_filter and df is not None:
            filter_start = perf_counter()
            if callable(tag_filter):
                df = tag_filter(df)
            else:
                df = df.filter(~pl.col("Tag").is_in(tag_filter))
            _log_slow_form_control(
                f"tagset_selection_filter tagset={tag_radio} subtype={tag_type}",
                filter_start,
                user_session_id,
                df,
            )
    else:
        session_key = tagset_keys[tag_radio]

        # Use corpus data manager for unified data access
        get_data_start = perf_counter()
        df = _get_tagset_frame(user_session_id, session_key)
        _log_slow_form_control(
            f"tagset_selection_get_data key={session_key}",
            get_data_start,
            user_session_id,
            df,
        )

        # Apply simplify function if provided
        simplify_func = simplify_funcs.get(tag_radio)
        if simplify_func and df is not None:
            simplify_start = perf_counter()
            df = _apply_cached_simplify(
                df,
                session_key,
                tag_radio,
                simplify_func,
            )
            _log_slow_form_control(
                f"tagset_selection_simplify tagset={tag_radio}",
                simplify_start,
                user_session_id,
                df,
            )
        # Apply filter if provided
        tag_filter = tag_filters.get(tag_radio)
        if tag_filter and df is not None:
            filter_start = perf_counter()
            if callable(tag_filter):
                df = tag_filter(df)
            else:
                df = df.filter(~pl.col("Tag").is_in(tag_filter))
            _log_slow_form_control(
                f"tagset_selection_filter tagset={tag_radio}",
                filter_start,
                user_session_id,
                df,
            )

    # Get tag options
    tag_options = []
    if include_tag_options and df is not None and hasattr(df, "get_column"):
        try:
            tag_options_start = perf_counter()
            tag_options = sorted(df.get_column("Tag").unique().to_list())
            _log_slow_form_control(
                "tagset_selection_tag_options",
                tag_options_start,
                user_session_id,
                df,
            )
        except Exception:
            tag_options = []

    _log_slow_form_control("tagset_selection_total", total_start, user_session_id, df)
    return df, tag_options, tag_radio, tag_type


# Tag filtering functions
def tag_filter_multiselect(
    df,
    tag_col="Tag",
    label="Select tags to filter:",
    key=None,
    tag_options: list | None = None,
    user_session_id=None,
) -> pl.DataFrame | None:
    """
    Render a segmented control widget (inside an expander) for tag filtering and
    return the filtered DataFrame.
    """
    if df is None or getattr(df, "height", 0) == 0:
        return df

    total_start = perf_counter()
    cats = tag_options
    if cats is None:
        options_start = perf_counter()
        cats = sorted(df.get_column(tag_col).drop_nulls().unique().to_list())
        _log_slow_form_control(
            f"tag_filter_options tag_col={tag_col}",
            options_start,
            user_session_id,
            df,
        )
    if not cats:
        return df
    seg_key = key or f"seg_{tag_col}"
    widget_start = perf_counter()
    with st.expander(
        label=label,
        icon=":material/filter_alt:"
    ):
        if st.button(
            label="Deselect All",
            key=f"{seg_key}_deselect",
            type="tertiary"
        ):
            # Use session-scoped state management
            if user_session_id:
                # Set the value directly in the user's session state
                if user_session_id not in st.session_state:
                    st.session_state[user_session_id] = {}
                st.session_state[user_session_id][seg_key] = []
                # Also set the global key for immediate effect
                st.session_state[seg_key] = []
            else:
                st.session_state[seg_key] = []
        selected = st.segmented_control(
            f"Select {tag_col}:",
            options=cats,
            selection_mode="multi",
            key=seg_key,
            help="Click to filter by one or more tags. Click again to deselect."
        )
    _log_slow_form_control(
        f"tag_filter_widget tag_col={tag_col} options={len(cats)}",
        widget_start,
        user_session_id,
        df,
    )
    if selected is None or len(selected) == 0:
        _log_slow_form_control(
            f"tag_filter_total tag_col={tag_col} selected=0",
            total_start,
            user_session_id,
            df,
        )
        return df
    filter_start = perf_counter()
    df = df.filter(pl.col(tag_col).is_in(selected))
    _log_slow_form_control(
        f"tag_filter_apply tag_col={tag_col} selected={len(selected)}",
        filter_start,
        user_session_id,
        df,
    )
    _log_slow_form_control(
        f"tag_filter_total tag_col={tag_col} selected={len(selected)}",
        total_start,
        user_session_id,
        df,
    )
    return df


def multi_tag_filter_multiselect(
        df: pl.DataFrame,
        tag_cols: list[str],
        user_session_id: str = None
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
            with st.expander(
                label=f"Filter {tag_col}",
                icon=":material/filter_alt:"
            ):
                if st.button("Deselect All",
                             key=f"{seg_key}_deselect",
                             type="tertiary"
                             ):
                    # Use session-scoped state management
                    if user_session_id:
                        # Set the value directly in the user's session state
                        if user_session_id not in st.session_state:
                            st.session_state[user_session_id] = {}
                        st.session_state[user_session_id][seg_key] = []
                        # Also set the global key for immediate effect
                        st.session_state[seg_key] = []
                    else:
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


def keyness_settings_info(
    user_session_id: str,
    pval_threshold: float | None = None,
    swap_target: bool | None = None,
) -> str:
    """
    Generate keyness settings information string.

    Parameters
    ----------
    user_session_id : str
        The user session identifier

    Returns
    -------
    str
        Formatted string with p-value threshold and swap settings
    """
    if pval_threshold is None:
        pval_threshold = st.session_state[user_session_id].get('pval_threshold', 0.01)
    if swap_target is None:
        swap_target = st.session_state[user_session_id].get('swap_target', False)

    return (
        f"**p-value threshold:** {pval_threshold} &nbsp;&nbsp; "
        f"**Swapped:** {'Yes' if swap_target else 'No'}"
    )


def rgb_to_hex(rgb_str):
    """Convert RGB string to hex color code."""
    if rgb_str.startswith("rgb"):
        nums = rgb_str[rgb_str.find("(")+1:rgb_str.find(")")].split(",")
        return "#{:02x}{:02x}{:02x}".format(*(int(float(n)) for n in nums))
    return rgb_str


def color_picker_controls(
        cats: list[str] = None,
        default_hex: str = "#133955",
        default_palette: str = "Plotly",
        expander_label: str = "Select Plot Colors",
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

    with st.expander(
        label=expander_label,
        icon=":material/palette:"
    ):
        color_mode = st.radio(
            "Color mode",
            ["Default colors", "Plotly palette", "Custom (pick colors)"],
            horizontal=True,
            key=color_mode_key
        )

        if color_mode == "Default colors":
            for cat in cats:
                if cat.lower() == "non-highlight":
                    color = non_highlight_default
                elif cat.lower() == "reference corpus":
                    color = reference_corpus_default
                else:
                    color = default_hex
                color_dict[cat] = color
        elif color_mode == "Custom (pick colors)":
            for idx, cat in enumerate(cats):
                if cat.lower() == "non-highlight":
                    color_default = non_highlight_default
                elif cat.lower() == "reference corpus":
                    color_default = reference_corpus_default
                else:
                    color_default = default_hex
                safe_cat = (str(cat)
                            .replace(" ", "_")
                            .replace(",", "_")
                            .replace("/", "_")
                            .replace("(", "")
                            .replace(")", ""))
                if not safe_cat:
                    safe_cat = f"cat_{idx}"
                color_key = f"{key_prefix}_{safe_cat}"
                color = st.color_picker(
                    f"Color for {cat}",
                    value=get_widget_state(color_key, color_default),
                    key=color_key
                )
                color_dict[cat] = color
        else:  # Plotly palette
            palette = st.selectbox(
                "Plotly palette",
                plotly_palettes,
                index=(plotly_palettes.index(default_palette)
                       if default_palette in plotly_palettes else 0),
                key=palette_key
            )
            palette_colors_raw = (getattr(plotly.colors.qualitative, palette, None) or
                                  getattr(plotly.colors.sequential, palette, None))
            palette_colors = ([rgb_to_hex(c) for c in palette_colors_raw]
                              if palette_colors_raw else [default_hex])
            for idx, cat in enumerate(cats):
                safe_cat = (str(cat)
                            .replace(" ", "_")
                            .replace(",", "_")
                            .replace("/", "_")
                            .replace("(", "")
                            .replace(")", ""))
                if not safe_cat:
                    safe_cat = f"cat_{idx}"
                color_key = f"{key_prefix}_{safe_cat}"
                # Always use the last selected value for this category, or palette default
                last_value = get_widget_state(
                    color_key, palette_colors[idx % len(palette_colors)]
                    )
                default_idx = (
                    palette_colors.index(last_value)
                    if last_value in palette_colors
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
    return color_dict
