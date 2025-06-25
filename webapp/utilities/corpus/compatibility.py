"""
Compatibility layer for gradual migration to corpus data manager.

This module provides backward-compatible functions that mirror the existing
session state access patterns while gradually migrating to the new corpus
data manager system.
"""

from typing import Optional
import polars as pl
import streamlit as st

from webapp.utilities.corpus.data_manager import get_corpus_manager
from webapp.utilities.configuration.logging_config import get_logger

logger = get_logger()

# Alias for backward compatibility
get_corpus_data_manager = get_corpus_manager


def get_corpus_data(
    user_session_id: str,
    corpus_type: str,
    data_key: str
) -> Optional[pl.DataFrame]:
    """
    Get corpus data using the new manager with fallback to legacy access.

    This function provides backward compatibility while migrating to the new
    corpus data manager system.

    Parameters
    ----------
    user_session_id : str
        The user session identifier
    corpus_type : str
        Type of corpus ('target' or 'reference')
    data_key : str
        The data key to retrieve

    Returns
    -------
    Optional[pl.DataFrame]
        The requested data or None if not available
    """
    try:
        # Try new manager first
        manager = get_corpus_manager(user_session_id, corpus_type)
        data = manager.get_data(data_key)

        if data is not None:
            return data

        # Fallback to legacy session state access
        if (user_session_id in st.session_state and
                corpus_type in st.session_state[user_session_id] and
                data_key in st.session_state[user_session_id][corpus_type]):

            logger.debug(f"Fallback to legacy access for {corpus_type}.{data_key}")
            return st.session_state[user_session_id][corpus_type][data_key]

    except Exception as e:
        logger.error(f"Error getting corpus data {corpus_type}.{data_key}: {e}")

        # Final fallback to direct session state
        try:
            return st.session_state[user_session_id][corpus_type][data_key]
        except (KeyError, TypeError):
            pass

    return None


def set_corpus_data(
    user_session_id: str,
    corpus_type: str,
    data_key: str,
    data: pl.DataFrame
) -> None:
    """
    Set corpus data using the new manager while maintaining legacy compatibility.

    Parameters
    ----------
    user_session_id : str
        The user session identifier
    corpus_type : str
        Type of corpus ('target' or 'reference')
    data_key : str
        The data key to set
    data : pl.DataFrame
        The data to store
    """
    try:
        # Use new manager
        manager = get_corpus_manager(user_session_id, corpus_type)
        manager.set_data(data_key, data)

        # Also maintain legacy session state for backward compatibility
        if user_session_id not in st.session_state:
            st.session_state[user_session_id] = {}
        if corpus_type not in st.session_state[user_session_id]:
            st.session_state[user_session_id][corpus_type] = {}

        st.session_state[user_session_id][corpus_type][data_key] = data

    except Exception as e:
        logger.error(f"Error setting corpus data {corpus_type}.{data_key}: {e}")

        # Fallback to legacy session state only
        if user_session_id not in st.session_state:
            st.session_state[user_session_id] = {}
        if corpus_type not in st.session_state[user_session_id]:
            st.session_state[user_session_id][corpus_type] = {}

        st.session_state[user_session_id][corpus_type][data_key] = data


def has_corpus_data(
    user_session_id: str,
    corpus_type: str,
    data_key: str
) -> bool:
    """
    Check if corpus data exists using the new manager with legacy fallback.

    Parameters
    ----------
    user_session_id : str
        The user session identifier
    corpus_type : str
        Type of corpus ('target' or 'reference')
    data_key : str
        The data key to check

    Returns
    -------
    bool
        True if data exists, False otherwise
    """
    try:
        # Check with new manager first
        manager = get_corpus_manager(user_session_id, corpus_type)
        if manager.has_data_key(data_key):
            return True

        # Fallback to legacy session state check
        return (user_session_id in st.session_state and
                corpus_type in st.session_state[user_session_id] and
                data_key in st.session_state[user_session_id][corpus_type])

    except Exception as e:
        logger.error(f"Error checking corpus data {corpus_type}.{data_key}: {e}")

        # Final fallback to direct session state check
        try:
            return (user_session_id in st.session_state and
                    corpus_type in st.session_state[user_session_id] and
                    data_key in st.session_state[user_session_id][corpus_type])
        except (KeyError, TypeError):
            return False


def corpus_is_ready(user_session_id: str, corpus_type: str) -> bool:
    """
    Check if corpus has minimum required data to be considered ready.

    Parameters
    ----------
    user_session_id : str
        The user session identifier
    corpus_type : str
        Type of corpus ('target' or 'reference')

    Returns
    -------
    bool
        True if corpus is ready, False otherwise
    """
    try:
        manager = get_corpus_manager(user_session_id, corpus_type)
        return manager.is_ready()
    except Exception as e:
        logger.error(f"Error checking corpus readiness {corpus_type}: {e}")

        # Fallback to legacy check
        return has_corpus_data(user_session_id, corpus_type, "ds_tokens")


def get_available_corpus_keys(user_session_id: str, corpus_type: str) -> list[str]:
    """
    Get list of available data keys for a corpus.

    Parameters
    ----------
    user_session_id : str
        The user session identifier
    corpus_type : str
        Type of corpus ('target' or 'reference')

    Returns
    -------
    list[str]
        List of available data keys
    """
    try:
        manager = get_corpus_manager(user_session_id, corpus_type)
        return manager.get_available_keys()
    except Exception as e:
        logger.error(f"Error getting available keys {corpus_type}: {e}")

        # Fallback to legacy session state keys
        try:
            if (
                user_session_id in st.session_state and
                corpus_type in st.session_state[user_session_id]
            ):
                return list(st.session_state[user_session_id][corpus_type].keys())
        except (KeyError, TypeError):
            pass

        return []


def migrate_legacy_session_data(user_session_id: str, corpus_type: str) -> None:
    """
    Migrate existing legacy session data to the new corpus manager.

    This function should be called when we detect legacy data exists
    to ensure it's properly managed by the new system.

    Parameters
    ----------
    user_session_id : str
        The user session identifier
    corpus_type : str
        Type of corpus ('target' or 'reference')
    """
    try:
        if (
            user_session_id not in st.session_state or
            corpus_type not in st.session_state[user_session_id]
        ):
            return

        legacy_data = st.session_state[user_session_id][corpus_type]
        if not legacy_data:
            return

        manager = get_corpus_manager(user_session_id, corpus_type)

        # Load all legacy data into the manager
        manager.load_all_data(legacy_data)

    except Exception as e:
        logger.error(f"Error migrating legacy session data {corpus_type}: {e}")


def clear_corpus_data(
    user_session_id: str, corpus_type: str, keys: Optional[list] = None
) -> None:
    """
    Clear corpus data from both new manager and legacy session state.

    Parameters
    ----------
    user_session_id : str
        The user session identifier
    corpus_type : str
        Type of corpus ('target' or 'reference')
    keys : Optional[list]
        Specific keys to clear. If None, clears all data.
    """
    try:
        manager = get_corpus_manager(user_session_id, corpus_type)

        if keys is None:
            # Clear all data
            manager.clear_data()
            # Clear from legacy session state
            if (user_session_id in st.session_state and
                    corpus_type in st.session_state[user_session_id]):
                st.session_state[user_session_id][corpus_type].clear()
        else:
            # Clear specific keys
            for key in keys:
                if key in manager.session_corpus_data:
                    del manager.session_corpus_data[key]
                # Also clear from legacy session state
                if (user_session_id in st.session_state and
                        corpus_type in st.session_state[user_session_id] and
                        key in st.session_state[user_session_id][corpus_type]):
                    del st.session_state[user_session_id][corpus_type][key]

            # Trigger cache invalidation if needed
            manager._invalidate_derived_cache()

    except Exception as e:
        logger.error(f"Error clearing corpus data {corpus_type}: {e}")


# Convenience functions that match existing usage patterns
def get_target_data(user_session_id: str, data_key: str) -> Optional[pl.DataFrame]:
    """Get target corpus data."""
    return get_corpus_data(user_session_id, "target", data_key)


def get_reference_data(user_session_id: str, data_key: str) -> Optional[pl.DataFrame]:
    """Get reference corpus data."""
    return get_corpus_data(user_session_id, "reference", data_key)


def set_target_data(user_session_id: str, data_key: str, data: pl.DataFrame) -> None:
    """Set target corpus data."""
    set_corpus_data(user_session_id, "target", data_key, data)


def set_reference_data(
    user_session_id: str, data_key: str, data: pl.DataFrame
) -> None:
    """Set reference corpus data."""
    set_corpus_data(user_session_id, "reference", data_key, data)
