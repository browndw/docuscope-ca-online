import polars as pl
import streamlit as st

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


def tagset_selection(
    user_session_id: str,
    session_state: dict,
    persist_func: Callable,
    page_stem: str,
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
        Function to persist widget state.
    page_stem : str
        The page identifier for persistence.
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
            if callable(tag_filter):
                df = tag_filter(df)
            else:
                df = df.filter(~pl.col("Tag").is_in(tag_filter))
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
        key=None
        ) -> pl.DataFrame | None:
    """
    Render a segmented control widget (inside an expander) for tag filtering and
    return the filtered DataFrame.
    """
    # import polars as pl  # Moved to module level
    # import streamlit as st  # Moved to module level

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
            with st.expander(
                label=f"Filter {tag_col}",
                icon=":material/filter_alt:"
            ):
                if st.button("Deselect All",
                             key=f"{seg_key}_deselect",
                             type="tertiary"
                             ):
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
            # Use default hex for all, with special cases
            prev_color = default_hex
            for idx, cat in enumerate(cats):
                if cat.lower() == "non-highlight":
                    color = non_highlight_default
                elif cat.lower() == "reference corpus":
                    color = reference_corpus_default
                else:
                    color = prev_color
                color_dict[cat] = color
                prev_color = color
        elif color_mode == "Custom (pick colors)":
            prev_color = default_hex
            seen_keys = set()
            for idx, cat in enumerate(cats):
                if cat.lower() == "non-highlight":
                    color_default = non_highlight_default
                elif cat.lower() == "reference corpus":
                    color_default = reference_corpus_default
                else:
                    color_default = prev_color
                safe_cat = str(cat).replace(" ", "_").replace(",", "_").replace("/", "_")
                if not safe_cat:
                    safe_cat = f"cat_{idx}"
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
                prev_color = color
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
                prev_color = color

    return color_dict
