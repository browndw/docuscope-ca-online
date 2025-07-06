"""
Data table utilities for Streamlit interface.

This module provides functions for formatting and displaying Streamlit
dataframes, including column configurations and data transformations.
"""

import streamlit as st
import polars as pl
import pandas as pd
from typing import Literal

from webapp.utilities.ui.corpus_display import target_info
from webapp.utilities.ui.form_controls import tag_filter_multiselect
from webapp.utilities.exports import convert_to_excel


def get_streamlit_column_config(
        df: pl.DataFrame | pd.DataFrame
        ) -> dict:
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
    return config


def render_data_table_interface(
    df,
    metadata_target: dict,
    base_filename: str,
    no_data_message: str = "No data available to display.",
    apply_tag_filter: bool = True,
    user_session_id: str = None
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

    # Display the target information first
    st.info(target_info(metadata_target))

    # Apply tag filtering if requested (this shows the filter expander)
    if apply_tag_filter:
        df = tag_filter_multiselect(df, user_session_id=user_session_id)

    # Display the data table or warning
    if df is not None and hasattr(df, "height") and df.height > 0:
        render_dataframe(df)
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
