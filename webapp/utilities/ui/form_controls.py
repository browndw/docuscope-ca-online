"""
Form control utilities for user interface components.

This module provides complex form controls like tagset selection,
toggle downloads, and other interactive UI components.
"""

import streamlit as st
import polars as pl
import operator
import plotly.colors
from functools import reduce
from typing import Dict, Optional, Callable, Tuple

# Import widget key manager for centralized widget management
from webapp.utilities.state.widget_key_manager import (
    register_persistent_widgets,
    get_widget_state,
)

# Import corpus data manager for unified data access
from webapp.utilities.corpus import get_corpus_data

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


def tagset_selection(
    user_session_id: str,
    session_state: dict,
    persist_func: Callable,
    tagset_keys: Optional[Dict] = None,
    simplify_funcs: Optional[Dict] = None,
    tag_filters: Optional[Dict] = None,
    tag_radio_key: str = "tag_radio",
    tag_type_key: str = "tag_type_radio",
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
        df = get_corpus_data(user_session_id, "target", session_key)

        # Apply simplify function if provided
        simplify_func = simplify_funcs.get(tag_radio, {}).get(tag_type)
        if simplify_func and df is not None:
            df = simplify_func(df)
        # Apply filter if provided
        tag_filter = tag_filters.get(tag_radio, {}).get(tag_type)
        if tag_filter and df is not None:
            if callable(tag_filter):
                df = tag_filter(df)
            else:
                df = df.filter(~pl.col("Tag").is_in(tag_filter))
    else:
        session_key = tagset_keys[tag_radio]

        # Use corpus data manager for unified data access
        df = get_corpus_data(user_session_id, "target", session_key)

        # Apply simplify function if provided
        simplify_func = simplify_funcs.get(tag_radio)
        if simplify_func and df is not None:
            df = simplify_func(df)
        # Apply filter if provided
        tag_filter = tag_filters.get(tag_radio)
        if tag_filter and df is not None:
            if callable(tag_filter):
                df = tag_filter(df)
            else:
                df = df.filter(~pl.col("Tag").is_in(tag_filter))

    # Get tag options
    tag_options = []
    if df is not None and hasattr(df, "get_column"):
        try:
            tag_options = sorted(df.get_column("Tag").unique().to_list())
        except Exception:
            tag_options = []

    return df, tag_options, tag_radio, tag_type


# Tag filtering functions
def tag_filter_multiselect(
        df,
        tag_col="Tag",
        label="Select tags to filter:",
        key=None,
        user_session_id=None
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
    if selected is None or len(selected) == 0:
        return df
    df = df.filter(pl.col(tag_col).is_in(selected))
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


def keyness_settings_info(user_session_id: str) -> str:
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
    pval_threshold = st.session_state[user_session_id].get('pval_threshold', 0.01)
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
