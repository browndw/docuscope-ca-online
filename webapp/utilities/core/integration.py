"""
Application Core Integration Module.

This module provides a unified interface to all the core utilities and patterns
used throughout the corpus tagger application, including standardized
configuration access, error handling, UI boundaries, and session management.
"""

# Configuration Access Patterns
from webapp.config.unified import (
    get_config,
    get_ai_config
)

# Error Handling and Boundaries
from webapp.utilities.error_handling import (
    error_handler
)

# Core Error Handling (UI-independent)
from webapp.utilities.core.error_handling import (
    core_error_handler
)

# Memory Management
from webapp.utilities.memory import (
    DataFrameCache,
    cleanup_dataframe_memory,
    optimize_dataframe_memory
)

# Enhanced Session Validation and Unified Session Manager
from webapp.utilities.session.validation_enhanced import (
    enhanced_validator
)
from webapp.utilities.session.session_manager import (
    session_manager
)

# Corpus Data Management
from webapp.utilities.corpus import (
    get_corpus_data_manager
)

# Widget State Management
from webapp.utilities.state.widget_key_manager import (
    widget_key_manager,
    register_persistent_widgets
)


class ApplicationCore:
    """
    Core application interface providing unified access to utilities.

    This class serves as the central hub for accessing all standardized
    patterns, utilities, and improvements throughout the application.
    It provides a consistent interface for configuration, error handling,
    session management, and UI components.
    """

    def __init__(self):
        """Initialize the application core interface."""
        self.error_handler = error_handler
        self.core_error_handler = core_error_handler
        self.session_validator = enhanced_validator
        self.session_manager = session_manager  # New unified session manager
        self.memory_cache = None  # Will be initialized per session
        self.widget_manager = widget_key_manager

    def get_safe_config(self, key: str, default=None, config_type: str = 'ai'):
        """
        Safely get configuration value with error handling.

        Parameters
        ----------
        key : str
            Configuration key
        default : Any
            Default value if key not found
        config_type : str
            Type of configuration ('ai' or 'general')

        Returns
        -------
        Any
            Configuration value or default
        """
        try:
            if config_type == 'ai':
                return get_ai_config(key, default)
            else:
                return get_config(key, config_type, default)
        except Exception:
            return default

    def validate_and_repair_session(self, user_session_id: str) -> bool:
        """
        Validate session with enhanced checking and auto-repair.

        Parameters
        ----------
        user_session_id : str
            The user session ID to validate

        Returns
        -------
        bool
            True if session is valid after any repairs
        """
        return self.session_validator.validate_and_repair_session(user_session_id)

    def get_session_health_report(self, user_session_id: str) -> dict:
        """
        Get comprehensive session health report.

        Parameters
        ----------
        user_session_id : str
            The user session ID to check

        Returns
        -------
        dict
            Detailed health report
        """
        is_valid, health_report = self.session_validator.validate_session_with_report(
            user_session_id
        )
        return {
            'is_valid': is_valid,
            'health_report': health_report
        }

    def safe_execute_function(self, component_func, *args, **kwargs):
        """
        Safely execute a function with error handling (UI-independent).

        Parameters
        ----------
        component_func : callable
            The function to execute
        *args : tuple
            Arguments for the function
        **kwargs : dict
            Keyword arguments for the function

        Returns
        -------
        Any
            Function result or None if error occurred
        """
        return self.core_error_handler.safe_execute(component_func, *args, **kwargs)

    def optimize_dataframe_for_session(
        self, df, cache_key: str = None, user_session_id: str = None
    ):
        """
        Optimize DataFrame for session storage with caching.

        Parameters
        ----------
        df : DataFrame
            DataFrame to optimize
        cache_key : str, optional
            Cache key for reuse
        user_session_id : str, optional
            User session ID for session-scoped caching

        Returns
        -------
        DataFrame
            Optimized DataFrame
        """
        # Initialize session-specific cache if needed
        if user_session_id and self.memory_cache is None:
            self.memory_cache = DataFrameCache(
                user_session_id,
                session_manager=self.session_manager
            )

        if cache_key and self.memory_cache and cache_key in self.memory_cache.cache:
            return self.memory_cache.get(cache_key)

        optimized_df = optimize_dataframe_memory(df)

        if cache_key and self.memory_cache:
            self.memory_cache.put(cache_key, optimized_df)

        return optimized_df

    def register_page_widgets(self, widget_keys: list):
        """
        Register persistent widgets for a page.

        Parameters
        ----------
        widget_keys : list
            List of widget keys to register
        """
        register_persistent_widgets(widget_keys)

    def cleanup_session_memory(self, user_session_id: str):
        """
        Clean up memory for a session.

        Parameters
        ----------
        user_session_id : str
            The user session ID to clean up
        """
        cleanup_dataframe_memory(user_session_id)
        if self.memory_cache:
            self.memory_cache.clear()

    def get_corpus_data_manager(self, user_session_id: str, corpus_type: str):
        """
        Get corpus data manager for a session and corpus type.

        Parameters
        ----------
        user_session_id : str
            The user session ID
        corpus_type : str
            Type of corpus ('target' or 'reference')

        Returns
        -------
        CorpusDataManager
            Corpus data manager instance
        """
        return get_corpus_data_manager(user_session_id, corpus_type)


# Global application core instance
app_core = ApplicationCore()


# Convenience functions for common patterns
def safe_config_value(key: str, default=None, config_type: str = 'ai'):
    """Get config value with error handling."""
    return app_core.get_safe_config(key, default, config_type)


def safe_function_execute(func):
    """Decorator for safe function execution with error handling."""
    def wrapper(*args, **kwargs):
        return app_core.safe_execute_function(func, *args, **kwargs)
    return wrapper


def validate_session_health(user_session_id: str) -> bool:
    """Validate session health with auto-repair."""
    return app_core.validate_and_repair_session(user_session_id)


def get_session_diagnostics(user_session_id: str) -> dict:
    """Get session diagnostics report."""
    return app_core.get_session_health_report(user_session_id)


__all__ = [
    # Main interface
    'app_core',
    'ApplicationCore',

    # Convenience functions
    'safe_config_value',
    'safe_function_execute',
    'validate_session_health',
    'get_session_diagnostics',

    # Direct imports for flexibility
    'error_handler',
    'enhanced_validator',
    'widget_key_manager',
    'get_corpus_data_manager'
]
