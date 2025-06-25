"""
Core Error Handling Module.

This module provides error handling utilities that are independent of UI components
to avoid circular imports. UI-specific error boundaries should be imported
separately when needed.
"""

from webapp.utilities.error_handling import (
    error_handler
)


class CoreErrorHandler:
    """
    Core error handling interface providing basic error handling
    without UI dependencies.
    """

    def __init__(self):
        """Initialize the core error handler."""
        self.error_handler = error_handler

    def safe_execute(self, func, *args, fallback=None, **kwargs):
        """
        Safely execute a function with error handling.

        Parameters
        ----------
        func : callable
            Function to execute
        *args : tuple
            Arguments for the function
        fallback : Any
            Fallback value if error occurs
        **kwargs : dict
            Keyword arguments for the function

        Returns
        -------
        Any
            Function result or fallback value
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.error_handler.handle_error(e)
            return fallback

    def with_error_context(self, context: str):
        """
        Create an error handler with specific context.

        Parameters
        ----------
        context : str
            Context description for errors

        Returns
        -------
        callable
            Context-specific error handler
        """
        def error_context_wrapper(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.error_handler.handle_error(e, context=context)
                    return None
            return wrapper
        return error_context_wrapper


# Global core error handler instance
core_error_handler = CoreErrorHandler()


__all__ = [
    'CoreErrorHandler',
    'core_error_handler'
]
