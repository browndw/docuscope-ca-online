"""
Memory management utilities for efficient dataframe handling.

This module provides utilities for managing memory usage, cleaning up temporary data,
and implementing lazy loading patterns to improve application performance.
"""

import gc
from typing import Any, Callable, Dict, Optional, Union
import polars as pl
import pandas as pd
import streamlit as st

from webapp.utilities.state import SessionKeys


class DataFrameCache:
    """
    Session-scoped cache for expensive DataFrame computations with automatic cleanup.
    """

    def __init__(self, user_session_id: str, max_size: int = 10, session_manager=None):
        """
        Initialize cache for a specific user session.

        Parameters
        ----------
        user_session_id : str
            The user session identifier
        max_size : int
            Maximum number of cached items before cleanup
        session_manager : optional
            Session manager instance for dependency injection
        """
        self.user_session_id = user_session_id
        self.max_size = max_size
        self._cache_key = f"_dataframe_cache_{user_session_id}"
        self._session_manager = session_manager

    def _get_cache(self) -> Dict[str, Any]:
        """Get or initialize the cache dictionary."""
        if self._session_manager:
            return self._session_manager.get_session_value(
                self.user_session_id,
                self._cache_key,
                default={}
            )
        else:
            # Fallback to direct session state access
            if self.user_session_id in st.session_state:
                return st.session_state[self.user_session_id].get(self._cache_key, {})
            return {}

    def _set_cache(self, cache: Dict[str, Any]) -> None:
        """Set the cache dictionary."""
        if self._session_manager:
            self._session_manager.update_session_state(
                self.user_session_id,
                self._cache_key,
                cache
            )
        else:
            # Fallback to direct session state access
            if self.user_session_id in st.session_state:
                st.session_state[self.user_session_id][self._cache_key] = cache

    def get(self, key: str) -> Optional[Any]:
        """
        Get a cached value by key.

        Parameters
        ----------
        key : str
            Cache key

        Returns
        -------
        Optional[Any]
            Cached value or None if not found
        """
        cache = self._get_cache()
        return cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """
        Set a cached value with automatic cleanup if needed.

        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache
        """
        cache = self._get_cache()
        cache[key] = value

        # Cleanup if cache is too large
        if len(cache) > self.max_size:
            # Remove oldest items (simple FIFO)
            keys_to_remove = list(cache.keys())[:-self.max_size]
            for old_key in keys_to_remove:
                del cache[old_key]

        self._set_cache(cache)

    def clear(self) -> None:
        """Clear all cached items."""
        self._set_cache({})
        gc.collect()


def cleanup_dataframe_memory(df: Union[pl.DataFrame, pd.DataFrame]) -> None:
    """
    Clean up memory used by a DataFrame.

    Parameters
    ----------
    df : Union[pl.DataFrame, pd.DataFrame]
        DataFrame to clean up
    """
    if hasattr(df, 'clear'):
        df.clear()
    elif hasattr(df, '_clear_cache'):
        df._clear_cache()

    # Force garbage collection
    del df
    gc.collect()


def optimize_dataframe_memory(
        df: Union[pl.DataFrame, pd.DataFrame]
) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Optimize DataFrame memory usage by downcasting numeric types where possible.

    Parameters
    ----------
    df : Union[pl.DataFrame, pd.DataFrame]
        DataFrame to optimize

    Returns
    -------
    Union[pl.DataFrame, pd.DataFrame]
        Memory-optimized DataFrame
    """
    if isinstance(df, pd.DataFrame):
        # Optimize pandas DataFrame
        for col in df.select_dtypes(include=['int64']).columns:
            if df[col].min() >= 0:
                if df[col].max() < 255:
                    df[col] = df[col].astype('uint8')
                elif df[col].max() < 65535:
                    df[col] = df[col].astype('uint16')
                elif df[col].max() < 4294967295:
                    df[col] = df[col].astype('uint32')
            else:
                if df[col].min() >= -128 and df[col].max() <= 127:
                    df[col] = df[col].astype('int8')
                elif df[col].min() >= -32768 and df[col].max() <= 32767:
                    df[col] = df[col].astype('int16')
                elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                    df[col] = df[col].astype('int32')

        # Optimize float columns
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')

    elif isinstance(df, pl.DataFrame):
        # Polars handles memory optimization better by default
        # Just ensure we're using appropriate types
        try:
            # Shrink string columns if possible
            string_cols = [col for col in df.columns if df[col].dtype == pl.Utf8]
            if string_cols:
                df = df.with_columns([
                    pl.col(col).cast(pl.Categorical)
                    for col in string_cols
                    if df[col].n_unique() < len(df) * 0.5  # Only if many duplicates
                ])
        except Exception:
            # If optimization fails, return original
            pass

    return df


def lazy_computation(cache_key: str, computation_func: Callable,
                     user_session_id: str, force_refresh: bool = False,
                     session_manager=None, **kwargs) -> Any:
    """
    Lazy computation with session-scoped caching.

    Parameters
    ----------
    cache_key : str
        Unique key for caching the computation result
    computation_func : Callable
        Function to compute the result
    user_session_id : str
        User session identifier
    force_refresh : bool
        Whether to force recomputation even if cached
    session_manager : optional
        Session manager instance for dependency injection
    **kwargs
        Arguments to pass to computation_func

    Returns
    -------
    Any
        Computation result (cached or newly computed)
    """
    cache = DataFrameCache(user_session_id, session_manager=session_manager)

    if not force_refresh:
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result

    # Compute and cache result
    result = computation_func(**kwargs)
    cache.set(cache_key, result)

    return result


def session_cleanup_on_exit(user_session_id: str) -> None:
    """
    Clean up session data when user exits or session expires.

    Parameters
    ----------
    user_session_id : str
        User session identifier
    """
    if user_session_id in st.session_state:
        session = st.session_state[user_session_id]

        # Clear large DataFrames
        df_keys = [
            'token_frequency', SessionKeys.TAGS_TABLE, 'dtm_target', 'dtm_reference',
            'ngrams_target', 'ngrams_reference', 'collocations_target',
            'collocations_reference', 'kwic_target', 'kwic_reference'
        ]

        for key in df_keys:
            if key in session:
                cleanup_dataframe_memory(session[key])
                del session[key]

        # Clear cached computations
        cache_keys = [k for k in session.keys() if k.startswith('_dataframe_cache_')]
        for key in cache_keys:
            del session[key]

        # Force garbage collection
        gc.collect()


def convert_once_and_cache(df: Union[pl.DataFrame, pd.DataFrame],
                           user_session_id: str,
                           cache_suffix: str = "",
                           session_manager=None) -> pd.DataFrame:
    """
    Convert Polars DataFrame to Pandas once and cache for reuse.

    Parameters
    ----------
    df : Union[pl.DataFrame, pd.DataFrame]
        DataFrame to convert
    user_session_id : str
        User session identifier
    cache_suffix : str
        Suffix for cache key to avoid conflicts
    session_manager : optional
        Session manager instance for dependency injection

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame (original or converted)
    """
    if isinstance(df, pd.DataFrame):
        return df

    # Create cache key based on DataFrame hash and suffix
    try:
        # Use a simple hash of the shape and column names as cache key
        cache_key = (f"pandas_conversion_"
                     f"{hash((tuple(df.columns), df.shape))}_{cache_suffix}")
    except Exception:
        # Fallback if hashing fails
        cache_key = f"pandas_conversion_{id(df)}_{cache_suffix}"

    cache = DataFrameCache(user_session_id, max_size=5, session_manager=session_manager)

    cached_df = cache.get(cache_key)
    if cached_df is not None:
        return cached_df

    # Convert and cache
    pandas_df = df.to_pandas()
    cache.set(cache_key, pandas_df)

    return pandas_df
