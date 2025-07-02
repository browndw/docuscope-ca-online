"""
Corpus Data Manager for memory-efficient lazy loading and data access.

This module provides a centralized system for managing corpus data with:
- Lazy loading of derived data structures
- Session-scoped caching with automatic cleanup
- Unified data access across all corpus types (Internal, External, New)
- Backward compatibility with existing session state patterns
"""

import gc
from typing import Dict, Optional, Any
import polars as pl
import streamlit as st
import docuscospacy as ds

from webapp.utilities.memory import DataFrameCache, lazy_computation
from webapp.utilities.configuration.logging_config import get_logger
from webapp.utilities.session.session_persistence import auto_persist_session

logger = get_logger()


class CorpusDataManager:
    """
    Session-scoped corpus data manager with smart caching and lazy loading.

    This manager provides a unified interface for accessing corpus data while
    implementing memory-efficient patterns like lazy loading and intelligent caching.
    """

    def __init__(
        self,
        user_session_id: str,
        corpus_type: str = "target",
        session_manager=None
    ):
        """
        Initialize corpus data manager for a specific user session and corpus type.

        Parameters
        ----------
        user_session_id : str
            The user session identifier
        corpus_type : str
            Type of corpus ('target' or 'reference')
        session_manager : optional
            Session manager instance for dependency injection
        """
        self.user_session_id = user_session_id
        self.corpus_type = corpus_type
        self.cache = DataFrameCache(
            user_session_id,
            max_size=15,
            session_manager=session_manager
        )

        # Core data keys (always loaded immediately)
        self.core_keys = ["ds_tokens"]

        # Derived data keys (computed on-demand)
        self.derived_keys = [
            "dtm_ds", "dtm_pos", "ft_ds", "ft_pos", "tt_ds", "tt_pos"
        ]

        # Additional data keys (generated/stored independently)
        self.additional_keys = [
            "collocations"
        ]

        # All expected data keys
        self.all_keys = self.core_keys + self.derived_keys + self.additional_keys

    @property
    def session_corpus_data(self) -> Dict:
        """Get the corpus data dictionary from session state."""
        if self.user_session_id not in st.session_state:
            st.session_state[self.user_session_id] = {}

        if self.corpus_type not in st.session_state[self.user_session_id]:
            st.session_state[self.user_session_id][self.corpus_type] = {}

        return st.session_state[self.user_session_id][self.corpus_type]

    def has_core_data(self) -> bool:
        """Check if core data (ds_tokens) is available."""
        return "ds_tokens" in self.session_corpus_data

    def has_data_key(self, key: str) -> bool:
        """Check if a specific data key exists in session or can be generated."""
        if key in self.session_corpus_data:
            return True
        if key in self.derived_keys and self.has_core_data():
            return True
        # Additional keys are only available if explicitly stored
        if key in self.additional_keys:
            return key in self.session_corpus_data
        return False

    def get_core_data(self) -> Optional[pl.DataFrame]:
        """Get the core ds_tokens DataFrame."""
        return self.session_corpus_data.get("ds_tokens")

    def set_core_data(self, ds_tokens: pl.DataFrame) -> None:
        """Set the core ds_tokens DataFrame."""
        self.session_corpus_data["ds_tokens"] = ds_tokens
        # Clear any cached derived data since core data changed
        self._invalidate_derived_cache()
        # Persist the session with new core data
        auto_persist_session(self.user_session_id)

    def _invalidate_derived_cache(self) -> None:
        """Clear cached derived data when core data changes."""
        cache_keys_to_clear = [
            f"{self.corpus_type}_{key}" for key in self.derived_keys
        ]
        for cache_key in cache_keys_to_clear:
            # Remove from both session state and cache
            if cache_key in self.session_corpus_data:
                del self.session_corpus_data[cache_key]

        # Clear cache entries
        cache = self.cache._get_cache()
        for key in list(cache.keys()):
            if any(derived_key in key for derived_key in self.derived_keys):
                del cache[key]
        self.cache._set_cache(cache)

    def _generate_frequency_tables(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Generate frequency tables from core data."""
        ds_tokens = self.get_core_data()
        if ds_tokens is None:
            raise ValueError(
                "Core data (ds_tokens) not available for frequency table generation"
            )

        # Generate frequency tables - no logging needed for normal operations
        return ds.frequency_table(ds_tokens, count_by="both")

    def _generate_tags_tables(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Generate tags tables from core data."""
        ds_tokens = self.get_core_data()
        if ds_tokens is None:
            raise ValueError(
                "Core data (ds_tokens) not available for tags table generation"
            )

        # Generate tags tables - no logging needed for normal operations
        return ds.tags_table(ds_tokens, count_by="both")

    def _generate_dtm_tables(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Generate document-term matrices from core data."""
        ds_tokens = self.get_core_data()
        if ds_tokens is None:
            raise ValueError("Core data (ds_tokens) not available for DTM generation")

        # Generate DTM tables - no logging needed for normal operations
        return ds.tags_dtm(ds_tokens, count_by="both")

    def get_data(self, key: str, force_refresh: bool = False) -> Optional[pl.DataFrame]:
        """
        Get data by key with lazy loading for derived data.

        Parameters
        ----------
        key : str
            Data key to retrieve
        force_refresh : bool
            Whether to force regeneration of derived data

        Returns
        -------
        Optional[pl.DataFrame]
            The requested data or None if not available
        """
        # Check if data exists in session state first
        if not force_refresh and key in self.session_corpus_data:
            return self.session_corpus_data[key]

        # If it's core data and not found, return None
        if key in self.core_keys:
            return self.session_corpus_data.get(key)

        # Handle derived data with lazy loading
        if key in self.derived_keys and self.has_core_data():
            return self._get_derived_data(key, force_refresh)

        # Handle additional keys (stored independently)
        if key in self.additional_keys:
            return self.session_corpus_data.get(key)

        return None

    def _get_derived_data(
        self, key: str, force_refresh: bool = False
    ) -> Optional[pl.DataFrame]:
        """Get derived data with caching."""
        cache_key = f"{self.corpus_type}_{key}_{self.user_session_id}"

        def compute_derived_data():
            if key in ["ft_pos", "ft_ds"]:
                ft_pos, ft_ds = self._generate_frequency_tables()
                # Cache both tables
                self.session_corpus_data["ft_pos"] = ft_pos
                self.session_corpus_data["ft_ds"] = ft_ds
                # Persist session with new cached data
                auto_persist_session(self.user_session_id)
                return ft_ds if key == "ft_ds" else ft_pos

            elif key in ["tt_pos", "tt_ds"]:
                tt_pos, tt_ds = self._generate_tags_tables()
                # Cache both tables
                self.session_corpus_data["tt_pos"] = tt_pos
                self.session_corpus_data["tt_ds"] = tt_ds
                # Persist session with new cached data
                auto_persist_session(self.user_session_id)
                return tt_ds if key == "tt_ds" else tt_pos

            elif key in ["dtm_pos", "dtm_ds"]:
                dtm_pos, dtm_ds = self._generate_dtm_tables()
                # Cache both tables
                self.session_corpus_data["dtm_pos"] = dtm_pos
                self.session_corpus_data["dtm_ds"] = dtm_ds
                # Persist session with new cached data
                auto_persist_session(self.user_session_id)
                return dtm_ds if key == "dtm_ds" else dtm_pos

            return None

        return lazy_computation(
            cache_key=cache_key,
            computation_func=compute_derived_data,
            user_session_id=self.user_session_id,
            force_refresh=force_refresh
        )

    def set_data(self, key: str, data: pl.DataFrame) -> None:
        """
        Set data by key in session state.

        Parameters
        ----------
        key : str
            Data key to set
        data : pl.DataFrame
            Data to store
        """
        self.session_corpus_data[key] = data

        # If setting core data, invalidate derived cache
        if key in self.core_keys:
            self._invalidate_derived_cache()

        # Persist the session with new data
        auto_persist_session(self.user_session_id)

    def load_all_data(self, data_dict: Dict[str, pl.DataFrame]) -> None:
        """
        Load all data at once (for legacy compatibility).

        Parameters
        ----------
        data_dict : Dict[str, pl.DataFrame]
            Dictionary of all data to load
        """
        for key, data in data_dict.items():
            self.session_corpus_data[key] = data

        # Persist the session with all loaded data
        auto_persist_session(self.user_session_id)

    def get_available_keys(self) -> list[str]:
        """Get list of available data keys."""
        available = list(self.session_corpus_data.keys())

        # Add derived keys that can be generated
        if self.has_core_data():
            for key in self.derived_keys:
                if key not in available:
                    available.append(key)

        return available

    def is_ready(self) -> bool:
        """Check if corpus has minimum required data."""
        return self.has_core_data()

    def clear_data(self) -> None:
        """Clear all corpus data and cache."""
        # Clear session state data
        self.session_corpus_data.clear()

        # Clear cache
        self.cache.clear()

        # Force garbage collection
        gc.collect()

    def get_memory_usage_info(self) -> Dict[str, Any]:
        """Get information about memory usage."""
        info = {
            "corpus_type": self.corpus_type,
            "user_session_id": self.user_session_id,
            "available_keys": self.get_available_keys(),
            "core_data_loaded": self.has_core_data(),
            "session_data_keys": list(self.session_corpus_data.keys()),
            "cache_size": len(self.cache._get_cache())
        }

        # Add size information if possible
        if self.has_core_data():
            core_data = self.get_core_data()
            if core_data is not None:
                info["core_data_shape"] = core_data.shape
                info["core_data_memory_mb"] = core_data.estimated_size() / (1024 * 1024)

        return info


def get_corpus_manager(
    user_session_id: str, corpus_type: str = "target"
) -> CorpusDataManager:
    """
    Get or create a corpus data manager instance.

    This function creates a new manager instance each time to avoid serialization
    issues with storing complex objects in session state. The managers access the
    same underlying data through session state, ensuring consistency.

    Parameters
    ----------
    user_session_id : str
        The user session identifier
    corpus_type : str
        Type of corpus ('target' or 'reference')

    Returns
    -------
    CorpusDataManager
        A new corpus data manager instance
    """
    # Always create a new manager instance to avoid session state serialization issues
    # The manager will access the same underlying data through session state
    return CorpusDataManager(user_session_id, corpus_type)
