"""
Unified Session Manager for comprehensive session state management.

This module provides a centralized SessionManager class that consolidates all
session-related functionality including initialization, validation, health checks,
metadata handling, and lifecycle management.

Key Features:
- Unified session initialization and management
- Comprehensive validation and health checking
- Automatic repair capabilities
- Memory and performance monitoring
- Widget state integration
- AI assistant state management
- Metadata processing
"""

from typing import Dict, Any, List, Optional, Tuple
import streamlit as st
import polars as pl
from datetime import datetime, timezone

# Core utilities
from webapp.utilities.configuration.logging_config import get_logger

# Session components to unify
from webapp.utilities.session.session_core import (
    init_session as core_init_session,
    update_session,
    init_metadata_target,
    init_metadata_reference,
    get_corpus_categories
)
from webapp.utilities.session.session_management import (
    init_ai_assist,
    ensure_session_key,
    get_session_value,
    generate_temp
)
from webapp.utilities.session.validation_enhanced import (
    SessionHealthChecker,
    SessionStateValidator,
    safe_clear_session_state
)
from webapp.utilities.session.metadata_handlers import (
    load_metadata,
    update_metadata,
    handle_target_metadata_processing
)

# Widget state integration
from webapp.utilities.state.widget_key_manager import widget_key_manager

logger = get_logger()


class SessionManager:
    """
    Unified session manager providing comprehensive session state management.

    This class consolidates all session-related functionality into a single,
    consistent interface with built-in validation, health checking, and
    automatic repair capabilities.
    """

    def __init__(self):
        """Initialize the session manager."""
        self.health_checker = SessionHealthChecker()
        self.validator = SessionStateValidator()
        self._initialized_sessions = set()
        self._session_timestamps = {}

    def create_session(self, user_session_id: str) -> bool:
        """
        Create and initialize a new session with all required components.

        Parameters
        ----------
        user_session_id : str
            The unique session identifier

        Returns
        -------
        bool
            True if session created successfully, False otherwise
        """
        try:
            if user_session_id in st.session_state:
                return True

            # Initialize base session structure
            st.session_state[user_session_id] = {}

            # Initialize core session state
            core_init_session(user_session_id)

            # Initialize AI assistant state
            init_ai_assist(user_session_id)

            # Mark session as initialized
            self._initialized_sessions.add(user_session_id)
            self._session_timestamps[user_session_id] = datetime.now(timezone.utc)
            return True

        except Exception as e:
            logger.error(f"Failed to create session {user_session_id}: {e}")
            return False

    def get_or_create_session(
        self, user_session_id: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get existing session or create new one with auto-detection.

        Parameters
        ----------
        user_session_id : str, optional
            Session ID. If None, will auto-detect from Streamlit context

        Returns
        -------
        Tuple[str, Dict[str, Any]]
            Session ID and session data dictionary
        """
        # Auto-detect session ID if not provided
        if user_session_id is None:
            try:
                ctx = st.runtime.scriptrunner_utils.script_run_context
                user_session = ctx.get_script_run_ctx()
                user_session_id = user_session.session_id
            except Exception as e:
                logger.error(f"Failed to auto-detect session ID: {e}")
                raise RuntimeError("Cannot determine session ID")

        # Create session if it doesn't exist
        if user_session_id not in st.session_state:
            self.create_session(user_session_id)

        # Get session data
        try:
            session_data_raw = st.session_state[user_session_id]["session"]
            # Handle both DataFrame and dict cases
            if (hasattr(session_data_raw, 'to_dict') and
                    hasattr(session_data_raw, 'columns')):
                # It's a Polars DataFrame (has both to_dict and columns attributes)
                session_data = session_data_raw.to_dict(as_series=False)
            else:
                # It's already a dictionary or other object
                session_data = (session_data_raw if isinstance(session_data_raw, dict)
                                else {})
        except KeyError:
            # Session exists but missing core structure - recreate
            logger.warning(f"Session {user_session_id} missing core structure, recreating")
            self.create_session(user_session_id)
            session_data_raw = st.session_state[user_session_id]["session"]
            if (hasattr(session_data_raw, 'to_dict') and
                    hasattr(session_data_raw, 'columns')):
                session_data = session_data_raw.to_dict(as_series=False)
            else:
                session_data = (session_data_raw if isinstance(session_data_raw, dict)
                                else {})

        return user_session_id, session_data

    def validate_session(self, user_session_id: str, with_repair: bool = True) -> bool:
        """
        Validate session state with optional auto-repair.

        Parameters
        ----------
        user_session_id : str
            Session ID to validate
        with_repair : bool
            Whether to attempt automatic repairs

        Returns
        -------
        bool
            True if session is valid (after any repairs)
        """
        if with_repair:
            return self.validator.validate_and_repair_session(user_session_id)
        else:
            is_valid, _ = self.validator.validate_session_with_report(user_session_id)
            return is_valid

    def get_health_report(self, user_session_id: str) -> Dict[str, Any]:
        """
        Get comprehensive session health report.

        Parameters
        ----------
        user_session_id : str
            Session ID to check

        Returns
        -------
        Dict[str, Any]
            Detailed health report
        """
        return self.health_checker.perform_health_check(user_session_id)

    def get_session_diagnostics(self, user_session_id: str) -> Dict[str, Any]:
        """
        Get detailed session diagnostics for debugging.

        Parameters
        ----------
        user_session_id : str
            Session ID to diagnose

        Returns
        -------
        Dict[str, Any]
            Comprehensive diagnostics
        """
        return self.validator.get_session_diagnostics(user_session_id)

    def update_session_state(self, user_session_id: str, key: str, value: Any) -> bool:
        """
        Safely update session state value.

        Parameters
        ----------
        user_session_id : str
            Session ID
        key : str
            Key to update
        value : Any
            New value

        Returns
        -------
        bool
            True if updated successfully
        """
        try:
            update_session(key, value, user_session_id)
            return True
        except Exception as e:
            logger.error(f"Failed to update session {user_session_id} key {key}: {e}")
            return False

    def get_session_value(self, user_session_id: str, key: str, default: Any = None) -> Any:
        """
        Safely get session state value.

        Parameters
        ----------
        user_session_id : str
            Session ID
        key : str
            Key to retrieve
        default : Any
            Default value if key not found

        Returns
        -------
        Any
            Session value or default
        """
        return get_session_value(user_session_id, key, default)

    def ensure_session_key(
        self, user_session_id: str, key: str, default_value: Any = None
    ) -> None:
        """
        Ensure session key exists with default value.

        Parameters
        ----------
        user_session_id : str
            Session ID
        key : str
            Key to ensure exists
        default_value : Any
            Default value if key doesn't exist
        """
        ensure_session_key(user_session_id, key, default_value)

    def initialize_metadata(
        self, user_session_id: str, corpus_type: str = "target"
    ) -> bool:
        """
        Initialize metadata for specified corpus type.

        Parameters
        ----------
        user_session_id : str
            Session ID
        corpus_type : str
            Type of corpus ('target' or 'reference')

        Returns
        -------
        bool
            True if initialized successfully
        """
        try:
            if corpus_type == "target":
                init_metadata_target(user_session_id)
            elif corpus_type == "reference":
                init_metadata_reference(user_session_id)
            else:
                raise ValueError(f"Unknown corpus type: {corpus_type}")
            return True

        except Exception as e:
            msg = (f"Failed to initialize {corpus_type} metadata for session "
                   f"{user_session_id}: {e}")
            logger.error(msg)
            return False

    def load_metadata(
        self, user_session_id: str, corpus_type: str = "target"
    ) -> Optional[pl.DataFrame]:
        """
        Load metadata for specified corpus type.

        Parameters
        ----------
        user_session_id : str
            Session ID
        corpus_type : str
            Type of corpus ('target' or 'reference')

        Returns
        -------
        Optional[pl.DataFrame]
            Metadata DataFrame or None if not found
        """
        return load_metadata(user_session_id, corpus_type)

    def update_metadata(
        self, user_session_id: str, corpus_type: str, updates: Dict[str, Any]
    ) -> bool:
        """
        Update metadata for specified corpus type.

        Parameters
        ----------
        user_session_id : str
            Session ID
        corpus_type : str
            Type of corpus ('target' or 'reference')
        updates : Dict[str, Any]
            Metadata updates to apply

        Returns
        -------
        bool
            True if updated successfully
        """
        try:
            # Use the imported update_metadata function for each key-value pair
            for key, value in updates.items():
                update_metadata(corpus_type, key, value, user_session_id)
            return True
        except Exception as e:
            msg = (f"Failed to update {corpus_type} metadata for session "
                   f"{user_session_id}: {e}")
            logger.error(msg)
            return False

    def handle_target_metadata_processing(
        self, user_session_id: str, df: pl.DataFrame
    ) -> bool:
        """
        Handle target metadata processing with category validation.

        Parameters
        ----------
        user_session_id : str
            Session ID
        df : pl.DataFrame
            Target corpus data

        Returns
        -------
        bool
            True if processed successfully
        """
        try:
            handle_target_metadata_processing(user_session_id, df)
            return True
        except Exception as e:
            msg = f"Failed to process target metadata for session {user_session_id}: {e}"
            logger.error(msg)
            return False

    def get_corpus_categories(
        self, user_session_id: str, doc_ids: List[str]
    ) -> Tuple[List[str], int]:
        """
        Get document categories with session-scoped caching.

        Parameters
        ----------
        user_session_id : str
            Session ID
        doc_ids : List[str]
            Document IDs to get categories for

        Returns
        -------
        Tuple[List[str], int]
            Categories list and count
        """
        return get_corpus_categories(doc_ids, user_session_id)

    def initialize_temp_states(self, user_session_id: str, states: Dict[str, Any]) -> bool:
        """
        Initialize temporary session states.

        Parameters
        ----------
        user_session_id : str
            Session ID
        states : Dict[str, Any]
            State key-value pairs to initialize

        Returns
        -------
        bool
            True if initialized successfully
        """
        try:
            generate_temp(list(states.items()), user_session_id)
            return True
        except Exception as e:
            msg = f"Failed to initialize temp states for session {user_session_id}: {e}"
            logger.error(msg)
            return False

    def clear_session(self, user_session_id: str) -> bool:
        """
        Safely clear session state.

        Parameters
        ----------
        user_session_id : str
            Session ID to clear

        Returns
        -------
        bool
            True if cleared successfully
        """
        success = safe_clear_session_state(user_session_id)
        if success:
            self._initialized_sessions.discard(user_session_id)
            self._session_timestamps.pop(user_session_id, None)
        return success

    def clear_session_with_widgets(self, user_session_id: str) -> bool:
        """
        Clear widget states and UI components associated with a session.

        This method performs cleanup of global widget keys that are not
        automatically cleared when session data is reset. It should be called
        in addition to normal session clearing to ensure complete cleanup.

        Parameters
        ----------
        user_session_id : str
            Session ID to clear widget states for

        Returns
        -------
        bool
            True if widget clearing completed successfully
        """
        try:
            # Import here to avoid circular imports
            from webapp.utilities.state.widget_state import safe_clear_widget_states

            # Clear document selection widgets (main issue)
            document_widget_keys = [
                f"sd_random_{user_session_id}",
                f"sd_random_doc_{user_session_id}",
                f"sd_random_changed_{user_session_id}",
                f"sd_reroll_{user_session_id}",
            ]

            # Clear plotting and analysis widgets following clear_plots patterns
            plotting_widget_keys = [
                f"grpa_{user_session_id}",
                f"grpb_{user_session_id}",
                f"boxplot_vars_{user_session_id}",
                f"tar_{user_session_id}",
                f"ref_{user_session_id}",
                f"highlight_pca_groups_{user_session_id}",
                f"highlight_scatter_groups_{user_session_id}",
                f"trend_scatter_groups_{user_session_id}",
                f"trend_scatter_{user_session_id}",
                f"by_group_boxplot_{user_session_id}",
                f"by_group_scatter_{user_session_id}",
                f"pca_idx_tab1_{user_session_id}",
                f"pca_idx_tab2_{user_session_id}",
                f"boxplot_btn_{user_session_id}",
                f"boxplot_group_btn_{user_session_id}",
                f"scatterplot_btn_{user_session_id}",
                f"scatterplot_group_btn_{user_session_id}",
                f"scatter_x_grouped_{user_session_id}",
                f"scatter_y_grouped_{user_session_id}",
                f"scatter_x_nongrouped_{user_session_id}",
                f"scatter_y_nongrouped_{user_session_id}",
                f"highlight_boxplot_groups_{user_session_id}",
                f"highlight_scatter_groups_{user_session_id}",
                f"swap_target_{user_session_id}",
                f"pval_threshold_{user_session_id}",
            ]

            # Clear widget keys using prefix patterns (following clear_plots approach)
            widget_prefixes = [
                f"color_picker_form_{user_session_id}",
                f"seg_{user_session_id}",
                f"filter_{user_session_id}",
                f"highlight_{user_session_id}",
                f"toggle_{user_session_id}",
                f"download_{user_session_id}",
                f"boxplot_vars_{user_session_id}",
                f"color_picker_boxplot_{user_session_id}",
                f"color_picker_boxplot_general_{user_session_id}",
                f"color_picker_scatter_{user_session_id}",
                f"tags_{user_session_id}",
            ]

            # Find all keys matching the prefixes
            prefix_matched_keys = [
                key for key in st.session_state.keys()
                if any(key.startswith(prefix) for prefix in widget_prefixes)
            ]

            # Combine all widget keys to clear
            all_widget_keys = (
                document_widget_keys +
                plotting_widget_keys +
                prefix_matched_keys
            )

            # Remove duplicates while preserving order
            unique_widget_keys = list(dict.fromkeys(all_widget_keys))

            # Clear widget keys safely
            if unique_widget_keys:
                cleared_results = safe_clear_widget_states(unique_widget_keys)  # noqa: F841

            # Clear widget manager session-specific states
            try:
                widget_manager_cleared = widget_key_manager.cleanup_all_session_widgets()  # noqa: F841, E501

            except Exception as e:
                logger.warning(f"Widget manager cleanup failed: {e}")
                # Continue even if widget manager cleanup fails

            # Clear any persistence stores that might contain session data
            persistence_patterns = [
                "*_PERSIST",  # App-specific persistence stores
                "preserved_state",  # Widget state preservation
            ]

            for pattern in persistence_patterns:
                persistence_keys = [
                    key for key in st.session_state.keys()
                    if (pattern.replace('*', '') in key and
                        user_session_id in str(st.session_state.get(key, {})))
                ]
                if persistence_keys:
                    safe_clear_widget_states(persistence_keys)

            return True

        except Exception as e:
            logger.error(f"Failed to clear widget states for {user_session_id}: {e}")
            return False

    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session manager statistics.

        Returns
        -------
        Dict[str, Any]
            Session manager statistics
        """
        active_sessions = len([
            sid for sid in self._initialized_sessions
            if sid in st.session_state
        ])

        return {
            'total_sessions_created': len(self._initialized_sessions),
            'active_sessions': active_sessions,
            'session_timestamps': dict(self._session_timestamps),
            'widget_manager_stats': widget_key_manager.get_widget_statistics()
        }

    def cleanup_expired_sessions(self, max_age_hours: float = 24.0) -> int:
        """
        Clean up expired sessions based on age.

        Parameters
        ----------
        max_age_hours : float
            Maximum session age in hours

        Returns
        -------
        int
            Number of sessions cleaned up
        """
        cleanup_count = 0
        current_time = datetime.now(timezone.utc)
        expired_sessions = []

        for session_id, timestamp in self._session_timestamps.items():
            age_hours = (current_time - timestamp).total_seconds() / 3600
            if age_hours > max_age_hours:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            if self.clear_session(session_id):
                cleanup_count += 1

        return cleanup_count

    def validate_all_sessions(self) -> Dict[str, Any]:
        """
        Validate all active sessions and return summary.

        Returns
        -------
        Dict[str, Any]
            Validation summary for all sessions
        """
        results = {
            'total_sessions': 0,
            'valid_sessions': 0,
            'invalid_sessions': 0,
            'repaired_sessions': 0,
            'failed_sessions': []
        }

        for session_id in list(st.session_state.keys()):
            if isinstance(st.session_state.get(session_id), dict):
                results['total_sessions'] += 1

                try:
                    is_valid = self.validate_session(session_id, with_repair=True)
                    if is_valid:
                        results['valid_sessions'] += 1
                    else:
                        results['invalid_sessions'] += 1
                        results['failed_sessions'].append(session_id)
                except Exception as e:
                    results['failed_sessions'].append(session_id)
                    logger.error(f"Failed to validate session {session_id}: {e}")

        return results


# Global session manager instance
session_manager = SessionManager()
