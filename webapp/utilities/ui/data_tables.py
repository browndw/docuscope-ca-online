"""
Data table utilities for Streamlit interface.

This module provides functions for formatting and displaying Streamlit
dataframes, including column configurations and data transformations.
"""

import streamlit as st
import polars as pl
from typing import Literal
from time import perf_counter

from webapp.utilities.configuration.logging_config import get_logger
from webapp.utilities.ui.corpus_display import target_info
from webapp.utilities.ui.form_controls import tag_filter_multiselect
from webapp.utilities.exports import convert_to_excel


SLOW_TABLE_OPERATION_MS = 100
DEFAULT_TABLE_BATCH_SIZE = 500
logger = get_logger()


def _frame_shape(df) -> tuple[int, int]:
    if df is None:
        return 0, 0

    height = getattr(df, "height", 0)
    width = getattr(df, "width", 0)
    return height, width


def _log_table_operation(
    operation: str,
    start_time: float,
    user_session_id: str | None,
    df=None,
) -> None:
    elapsed_ms = (perf_counter() - start_time) * 1000
    if elapsed_ms < SLOW_TABLE_OPERATION_MS:
        return

    height, width = _frame_shape(df)
    logger.warning(
        "Slow data table step op={} session={} rows={} cols={} duration_ms={:.2f}",
        operation,
        user_session_id or "unknown",
        height,
        width,
        elapsed_ms,
    )


def _get_table_window_keys(
    base_filename: str,
    user_session_id: str | None,
) -> tuple[str, str]:
    """Return session-state keys for incremental table rendering."""

    session_prefix = user_session_id or "global"
    state_prefix = f"{session_prefix}_{base_filename}"
    return (
        f"{state_prefix}_visible_rows",
        f"{state_prefix}_dataset_signature",
    )


def _table_dataset_signature(df: pl.DataFrame | None) -> tuple:
    """Build a cheap signature to reset the visible window when data changes."""

    if df is None or getattr(df, "height", 0) == 0:
        return (0, 0, (), ())

    first_row = ()
    try:
        first_row = tuple(df.head(1).row(0)) if df.height > 0 else ()
    except Exception:
        first_row = ()

    return (df.height, df.width, tuple(df.columns), first_row)


def _get_visible_table_slice(
    df: pl.DataFrame,
    base_filename: str,
    user_session_id: str | None,
    batch_size: int,
) -> tuple[pl.DataFrame, int, bool]:
    """Return the current visible slice and whether more rows remain."""

    slice_start = perf_counter()

    visible_key, signature_key = _get_table_window_keys(
        base_filename,
        user_session_id,
    )

    signature_start = perf_counter()
    signature = _table_dataset_signature(df)
    _log_table_operation(
        "table_dataset_signature",
        signature_start,
        user_session_id,
        df,
    )
    previous_signature = st.session_state.get(signature_key)

    if previous_signature != signature:
        st.session_state[signature_key] = signature
        st.session_state[visible_key] = min(batch_size, df.height)

    visible_rows = st.session_state.get(visible_key, min(batch_size, df.height))
    visible_rows = min(max(int(visible_rows), batch_size), df.height)
    st.session_state[visible_key] = visible_rows

    visible_df = df.head(visible_rows)
    has_more_rows = visible_rows < df.height
    _log_table_operation(
        "get_visible_table_slice",
        slice_start,
        user_session_id,
        visible_df,
    )
    return visible_df, visible_rows, has_more_rows


def get_streamlit_column_config(df) -> dict:
    """
    Returns a column_config dictionary for st.dataframe based on column name patterns,
    including helpful tooltips for each column.
    Adjusts RF tooltips based on whether the table is token-based or tag-based.
    """
    # Detect if this is a tags-only table (no 'Token' or 'Token_*' column)
    tags_only = not any(col.startswith("Token") for col in df.columns)

    # Define tooltips for common columns
    tooltips = {
        "AF": "Absolute frequency (raw count)",
        "RF": (
            "Relative frequency (percent of tokens)"
            if tags_only else
            "Relative frequency (per million tokens)"
        ),
        "LL": "Log-likelihood (keyness statistic)",
        "LR": "Log ratio (effect size)",
        "Range": "Document range (proportion of docs containing item)",
        "PV": "p-value (statistical significance)",
        "MI": "Mutual information (association strength)",
        "AF_Ref": "Absolute frequency in reference corpus",
        "RF_Ref": (
            "Relative frequency in reference corpus (percent of tokens)"
            if tags_only else
            "Relative frequency in reference corpus (per million tokens)"
        ),
        "Range_Ref": "Document range in reference corpus",
    }

    config = {}
    for col in df.columns:
        # Find base name for tooltip matching (handles e.g. "RF_Ref")
        base = col
        if col.endswith("_Ref"):
            base = col
        elif "_" in col:
            base = col.split("_")[0]
        # Set format and help
        if col.startswith("AF"):
            config[col] = st.column_config.NumberColumn(
                format="%.0f",
                help=tooltips.get(col, tooltips.get(base, "Absolute frequency"))
            )
        elif col.startswith("RF"):
            config[col] = st.column_config.NumberColumn(
                format="%.2f",
                help=tooltips.get(col, tooltips.get(base, "Relative frequency"))
            )
        elif col.startswith("LL"):
            config[col] = st.column_config.NumberColumn(
                format="%.2f",
                help=tooltips.get(col, tooltips.get(base, "Log-likelihood"))
            )
        elif col.startswith("LR"):
            config[col] = st.column_config.NumberColumn(
                format="%.2f",
                help=tooltips.get(col, tooltips.get(base, "Log ratio"))
            )
        elif col.startswith("Range"):
            config[col] = st.column_config.NumberColumn(
                format="%.2f %%",
                help=tooltips.get(col, tooltips.get(base, "Document range"))
            )
        elif col.startswith("PV"):
            config[col] = st.column_config.NumberColumn(
                format="%.3f",
                help=tooltips.get(col, tooltips.get(base, "p-value"))
            )
        elif col.startswith("MI"):
            config[col] = st.column_config.NumberColumn(
                format="%.3f",
                help=tooltips.get(col, tooltips.get(base, "Mutual information"))
            )
        elif col == "Pre-Node":
            config[col] = st.column_config.TextColumn(alignment="right")
        elif col == "Node":
            config[col] = st.column_config.TextColumn(alignment="center")
        elif col == "Post-Node":
            config[col] = st.column_config.TextColumn(alignment="left")
    return config


def render_data_table_interface(
    df,
    metadata_target: dict,
    base_filename: str,
    no_data_message: str = "No data available to display.",
    apply_tag_filter: bool = True,
    tag_options: list[str] | None = None,
    user_session_id: str = None,
    batch_size: int = DEFAULT_TABLE_BATCH_SIZE,
) -> None:
    """
    Render data table interface with target info and download options.

    Args:
        df: DataFrame to display
        metadata_target: Target corpus metadata
        base_filename: Base filename for downloads
        no_data_message: Message to show when no data is available
        apply_tag_filter: Whether to apply tag filtering (default: True)
        user_session_id: The user session identifier for scoped state management

    Example usage:
        # For pages with tag filtering (most common):
        render_data_table_interface(df, metadata, "token_frequencies")

        # For pages without tag filtering:
        render_data_table_interface(df, metadata, "corpus_stats", apply_tag_filter=False)
    """

    total_start = perf_counter()
    input_height, input_width = _frame_shape(df)

    # Display the target information first
    target_info_start = perf_counter()
    st.info(target_info(metadata_target))
    _log_table_operation(
        "target_info_render",
        target_info_start,
        user_session_id,
    )

    # Apply tag filtering if requested (this shows the filter expander)
    if apply_tag_filter:
        filter_start = perf_counter()
        df = tag_filter_multiselect(
            df,
            tag_options=tag_options,
            user_session_id=user_session_id,
        )
        _log_table_operation(
            "tag_filter_multiselect",
            filter_start,
            user_session_id,
            df,
        )

    # Display the data table or warning
    if df is not None and hasattr(df, "height") and df.height > 0:
        visible_slice_start = perf_counter()
        visible_df, visible_rows, has_more_rows = _get_visible_table_slice(
            df,
            base_filename,
            user_session_id,
            batch_size,
        )
        _log_table_operation(
            "visible_table_slice_total",
            visible_slice_start,
            user_session_id,
            visible_df,
        )

        st.caption(
            f"Showing {visible_rows:,} of {df.height:,} rows. "
            "Downloads include the full filtered table."
        )

        render_start = perf_counter()
        render_dataframe(visible_df, user_session_id=user_session_id)
        _log_table_operation(
            "render_dataframe",
            render_start,
            user_session_id,
            visible_df,
        )

        if has_more_rows:
            visible_key, _ = _get_table_window_keys(base_filename, user_session_id)
            if st.button(
                f"Show next {batch_size}",
                key=f"{visible_key}_next",
                type="secondary",
            ):
                st.session_state[visible_key] = min(
                    visible_rows + batch_size,
                    df.height,
                )
                st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.markdown(
            "### Download Options",
            help=(
                "Generate and download the current data as an Excel file"
                " for offline analysis or reporting."
                )
        )
        to_download = st.sidebar.toggle("Download as Excel")
        if to_download:
            with st.sidebar.status("Generating Excel file..."):
                try:
                    render_excel_download_option(df, base_filename)
                except Exception as e:
                    st.sidebar.error(
                        f"Error generating Excel file: {e}",
                        icon=":material/error:"
                        )

    else:
        st.warning(no_data_message, icon=":material/info:")

    _log_table_operation(
        (
            "render_data_table_interface_total "
            f"input_rows={input_height} input_cols={input_width}"
        ),
        total_start,
        user_session_id,
        df,
    )


def render_dataframe(
    df: pl.DataFrame | None = None,
    column_config: dict | None = None,
    width: Literal['stretch', 'content'] = 'stretch',
    num_rows: Literal['fixed', 'dynamic'] = 'dynamic',
    disabled: bool = True,
    user_session_id: str | None = None,
) -> None:
    """
    Render a DataFrame in Streamlit.

    Parameters
    ----------
    df : pl.DataFrame, optional
        The DataFrame to render. If None, no data will be displayed.
    column_config : dict, optional
        Configuration for the DataFrame columns.
        If None, defaults to a configuration generated from the DataFrame.
    width : Literal['stretch', 'content']
        If 'stretch', the DataFrame will use the full width of the container.
        If 'content', the DataFrame will adjust based on content.
    num_rows : Literal['fixed', 'dynamic']
        How many rows to display in the DataFrame.
        'fixed' shows a fixed number of rows, 'dynamic' adjusts based on content.
    disabled : bool
        If True, use the lighter read-only dataframe renderer.
        If False, use the editable data editor.

    Returns
    -------
    None
        This function does not return anything.
        It renders the DataFrame directly in the Streamlit app.
    """
    if column_config is None and df is not None:
        column_config_start = perf_counter()
        column_config = get_streamlit_column_config(df)
        _log_table_operation(
            "get_streamlit_column_config",
            column_config_start,
            user_session_id,
            df,
        )
    if df is not None and getattr(df, "height", 0) > 0:
        if disabled:
            dataframe_start = perf_counter()
            st.dataframe(
                df,
                hide_index=True,
                column_config=column_config,
                width=width,
            )
            _log_table_operation(
                "st_dataframe",
                dataframe_start,
                user_session_id,
                df,
            )
        else:
            data_editor_start = perf_counter()
            st.dataframe(
                df,
                hide_index=True,
                column_config=column_config,
                width=width
            )
            _log_table_operation(
                "st_data_editor",
                data_editor_start,
                user_session_id,
                df,
            )
    else:
        st.warning("No data to display.")


def render_excel_download_option(df, base_filename: str, location=None) -> None:
    """
    Create and display an Excel download button for a DataFrame.

    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        The DataFrame to convert to Excel.
    base_filename : str
        The base filename for the download.
    location : optional
        The Streamlit component where to place the download button.
        If None, uses st.sidebar.
    """
    try:
        # Use optimized conversion function which handles both DataFrame types
        excel_buffer = convert_to_excel(df)
        filename = f"{base_filename}.xlsx"  # noqa: E501

        download_component = location if location else st.sidebar
        download_component.download_button(
            label="Download Excel file",
            data=excel_buffer,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            icon=":material/download:"
        )
    except Exception as e:
        st.error(f"Error generating Excel file: {e}")
