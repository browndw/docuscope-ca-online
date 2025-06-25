"""
Enhanced error handling and logging system for Phase 3 optimization.

This module provides consistent, deployment-ready error handling patterns
with selective logging that focuses on systemic issues while providing
appropriate user feedback for data-related problems.
"""

import functools
from typing import Any, Callable
import streamlit as st

from webapp.utilities.configuration.logging_config import get_logger


class SelectiveErrorHandler:
    """
    Enhanced error handler with selective logging for deployment scenarios.

    This handler distinguishes between:
    - User data issues (show to user, don't log)
    - System issues (log for diagnosis, show generic message to user)
    - Development issues (full logging and user feedback)
    """

    def __init__(self, deployment_mode: str = "production"):
        """
        Initialize error handler with deployment-specific behavior.

        Parameters
        ----------
        deployment_mode : str
            'production', 'staging', or 'development'
        """
        self.logger = get_logger()
        self.deployment_mode = deployment_mode

    def handle_user_data_error(self, error: Exception, user_message: str) -> None:
        """
        Handle user data errors (improperly encoded data, format issues, etc.).
        Shows informative message to user but doesn't log as these aren't systemic.

        Parameters
        ----------
        error : Exception
            The error that occurred
        user_message : str
            User-friendly message explaining the issue and how to resolve it
        """
        st.error(
            user_message,
            icon=":material/error:"
        )

        # Only log in development mode for debugging
        if self.deployment_mode == "development":
            self.logger.debug(f"User data error: {str(error)}", exc_info=True)

    def handle_system_error(
            self, error: Exception, context: str = "",
            fallback_message: str = None
    ) -> None:
        """
        Handle system errors that need logging for diagnosis.

        Parameters
        ----------
        error : Exception
            The error that occurred
        context : str
            Context information about where error occurred
        fallback_message : str
            Custom message to show user (otherwise uses generic message)
        """
        # Always log system errors for diagnosis
        error_msg = f"{context}: {str(error)}" if context else str(error)
        self.logger.error(error_msg, exc_info=True)

        # Show generic message to user
        user_msg = fallback_message or (
            "A system error occurred. Please try again or contact support "
            "if the issue persists."
        )
        st.error(user_msg, icon=":material/error:")

    def handle_operation_failure(self, operation: str, fallback_result: Any = None) -> Any:
        """
        Handle operation failures with graceful degradation.

        Parameters
        ----------
        operation : str
            Name of the operation that failed
        fallback_result : Any
            Result to return if operation fails

        Returns
        -------
        Any
            Fallback result
        """
        st.warning(
            f"{operation} is temporarily unavailable. "
            "Please try again in a moment.",
            icon=":material/warning:"
        )
        return fallback_result


# Global enhanced error handler
enhanced_error_handler = SelectiveErrorHandler()


def user_data_safe(
        user_message: str
) -> Callable:
    """
    Decorator for handling user data errors without logging.
    Use for file format issues, encoding problems, etc.

    Parameters
    ----------
    user_message : str
        Message to show users about the data issue
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                enhanced_error_handler.handle_user_data_error(e, user_message)
                return None
        return wrapper
    return decorator


def system_safe(
        context: str = "",
        fallback: Any = None,
        custom_message: str = None
):
    """
    Decorator for handling system errors with logging.
    Use for database connections, API calls, internal logic errors, etc.

    Parameters
    ----------
    context : str
        Context description for logging
    fallback : Any
        Fallback value to return on error
    custom_message : str
        Custom user message (otherwise uses generic)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = context or f"{func.__name__}"
                enhanced_error_handler.handle_system_error(
                    e, error_context, custom_message
                )
                return fallback
        return wrapper
    return decorator


def operation_safe(operation_name: str, fallback: Any = None):
    """
    Decorator for handling operation failures with graceful degradation.
    Use for optional features, non-critical operations, etc.

    Parameters
    ----------
    operation_name : str
        User-friendly operation name
    fallback : Any
        Fallback value to return
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                return enhanced_error_handler.handle_operation_failure(
                    operation_name, fallback
                )
        return wrapper
    return decorator


# Legacy compatibility - gradually migrate to new decorators
def with_error_handling(
        context: str = "",
        severity: str = "error",
        show_user: bool = True,
        fallback: Any = None
):
    """
    Legacy error handling decorator - use new decorators for new code.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if show_user:
                    enhanced_error_handler.handle_system_error(
                        e, context or f"{func.__name__}"
                    )
                return fallback
        return wrapper
    return decorator
