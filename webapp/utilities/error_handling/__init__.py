"""
Standardized error handling patterns for the application.

This module provides consistent error handling and logging patterns
across different components of the application.
"""

import functools
from typing import Any, Callable, Dict, Optional
import streamlit as st

from webapp.utilities.configuration.logging_config import get_logger


class StandardErrorHandler:
    """
    Standardized error handler with different severity levels.
    """

    def __init__(self, logger_name: str = __name__):
        """Initialize with a specific logger."""
        self.logger = get_logger()

    def handle_error(self, error: Exception, context: str = "",
                     severity: str = "error", show_user: bool = True) -> None:
        """
        Handle error with appropriate logging and user feedback.

        Parameters
        ----------
        error : Exception
            The error that occurred
        context : str
            Context information about where error occurred
        severity : str
            Error severity level ('debug', 'info', 'warning', 'error', 'critical')
        show_user : bool
            Whether to show error to user
        """
        error_msg = f"{context}: {str(error)}" if context else str(error)

        # Only log systemic issues - avoid logging user data problems
        should_log = self._should_log_error(error, context)
        
        if should_log:
            # Log based on severity
            if severity == 'debug':
                self.logger.debug(error_msg, exc_info=True)
            elif severity == 'info':
                self.logger.info(error_msg)
            elif severity == 'warning':
                self.logger.warning(error_msg)
            elif severity == 'error':
                self.logger.error(error_msg, exc_info=True)
            elif severity == 'critical':
                self.logger.critical(error_msg, exc_info=True)

        # Show to user if requested
        if show_user:
            if severity in ['error', 'critical']:
                st.error(
                    f"An error occurred: {error_msg}",
                    icon=":material/error:"
                )
            elif severity == 'warning':
                st.warning(
                    f"Warning: {error_msg}",
                    icon=":material/warning:"
                )
            elif severity == 'info':
                st.info(error_msg, icon=":material/info:")

    def _should_log_error(self, error: Exception, context: str) -> bool:
        """
        Determine if error should be logged based on type and context.
        
        User data issues (encoding, format) are not logged.
        System issues (network, database, internal logic) are logged.
        """
        # Convert to string for pattern matching
        error_str = str(error).lower()
        context_str = context.lower()
        
        # Don't log common user data issues
        user_data_indicators = [
            'encoding', 'decode', 'utf-8', 'ascii',
            'file format', 'invalid format', 'malformed',
            'empty file', 'no data found',
            'invalid column', 'missing column'
        ]
        
        if any(indicator in error_str or indicator in context_str 
               for indicator in user_data_indicators):
            return False
            
        # Log system and internal errors
        return True


# Global error handler instance
error_handler = StandardErrorHandler()


def with_error_handling(
        context: str = "",
        severity: str = "error",
        show_user: bool = True,
        fallback: Any = None
):
    """
    Decorator for standardized error handling.

    Parameters
    ----------
    context : str
        Context description for error messages
    severity : str
        Error severity level
    show_user : bool
        Whether to show error to user
    fallback : Any
        Fallback value to return on error

    Returns
    -------
    callable
        Decorated function with error handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = context or f"{func.__name__}"
                error_handler.handle_error(e, error_context, severity, show_user)
                return fallback
        return wrapper
    return decorator


def safe_operation(operation_name: str, fallback: Any = None):
    """
    Decorator for safe operations with user-friendly error messages.

    Parameters
    ----------
    operation_name : str
        Name of the operation for error messages
    fallback : Any
        Fallback value on error

    Returns
    -------
    callable
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                st.warning(
                    f"{operation_name} temporarily unavailable. "
                    f"Please try again or contact support if the issue persists.",
                    icon=":material/warning:"
                )
                error_handler.logger.error(
                    f"Error in {operation_name}: {str(e)}",
                    exc_info=True
                )
                return fallback
        return wrapper
    return decorator


def validate_input(validator_func: Callable, error_message: str = "Invalid input"):
    """
    Decorator for input validation with error handling.

    Parameters
    ----------
    validator_func : Callable
        Function to validate inputs
    error_message : str
        Error message for invalid inputs

    Returns
    -------
    callable
        Decorated function with validation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                if not validator_func(*args, **kwargs):
                    st.error(error_message, icon=":material/error:")
                    return None
                return func(*args, **kwargs)
            except Exception as e:
                st.error(f"Validation error: {str(e)}", icon=":material/error:")
                return None
        return wrapper
    return decorator


class ComponentHealthChecker:
    """
    Health checker for application components.
    """

    def __init__(self):
        self.component_status: Dict[str, bool] = {}

    def check_component(self, component_name: str, check_func: Callable) -> bool:
        """
        Check if a component is healthy.

        Parameters
        ----------
        component_name : str
            Name of the component
        check_func : Callable
            Function to check component health

        Returns
        -------
        bool
            True if component is healthy
        """
        try:
            result = check_func()
            self.component_status[component_name] = bool(result)
            return self.component_status[component_name]
        except Exception as e:
            self.component_status[component_name] = False
            error_handler.handle_error(
                e,
                f"Health check failed for {component_name}",
                severity="warning",
                show_user=False
            )
            return False

    def get_status(self, component_name: str) -> Optional[bool]:
        """Get component status."""
        return self.component_status.get(component_name)

    def get_all_status(self) -> Dict[str, bool]:
        """Get all component statuses."""
        return self.component_status.copy()


# Global health checker instance
health_checker = ComponentHealthChecker()


def with_health_check(component_name: str, fallback_message: str = None):
    """
    Decorator to add health checking to components.

    Parameters
    ----------
    component_name : str
        Name of the component
    fallback_message : str
        Message to show if component is unhealthy

    Returns
    -------
    callable
        Decorated function with health checking
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if component is marked as unhealthy
            if health_checker.get_status(component_name) is False:
                message = fallback_message or f"{component_name} is currently unavailable"
                st.warning(message, icon=":material/warning:")
                return None

            try:
                result = func(*args, **kwargs)
                # Mark as healthy if function succeeds
                health_checker.component_status[component_name] = True
                return result
            except Exception as e:
                # Mark as unhealthy on error
                health_checker.component_status[component_name] = False
                error_handler.handle_error(
                    e,
                    f"Component {component_name} failed",
                    severity="error",
                    show_user=True
                )
                return None
        return wrapper
    return decorator
