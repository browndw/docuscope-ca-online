"""
Standardized logging patterns for Phase 3 optimization.

This module provides simplified logging setup patterns to reduce repetitive
code across modules and ensure consistent logging configuration.
"""

from typing import Optional
from webapp.utilities.configuration.logging_config import get_logger, setup_utility_logging


def setup_module_logger(module_type: str, module_name: Optional[str] = None):
    """
    Simplified logging setup for any module with automatic type detection.
    
    Parameters
    ----------
    module_type : str
        Type of module ('ai', 'page', 'utility', 'state', 'session', 'core')
    module_name : str, optional
        Specific module name. If None, will be inferred from caller.
        
    Returns
    -------
    Logger
        Configured logger instance
    """
    if module_name is None:
        # Auto-detect module name from caller
        import inspect
        import pathlib
        
        frame = inspect.currentframe().f_back
        if frame and frame.f_globals.get('__file__'):
            module_name = pathlib.Path(frame.f_globals['__file__']).stem
        else:
            module_name = "unknown"
    
    # Set up logging based on module type
    utility_types = [
        'state', 'session', 'core', 'corpus', 'analysis', 'storage', 'configuration'
    ]
    if module_type in utility_types:
        setup_utility_logging(module_type)
    elif module_type == 'ai':
        setup_utility_logging("ai")
    elif module_type == 'page':
        setup_utility_logging(f"page_{module_name}")
    else:
        setup_utility_logging(module_name)
    
    return get_logger()


# Removed redundant get_standard_logger() - use get_logger() directly


class LoggerMixin:
    """
    Mixin class to add standardized logging to any class.
    
    Usage:
        class MyClass(LoggerMixin):
            def __init__(self):
                super().__init__()
                self.setup_logging('utility')  # or 'ai', 'page', etc.
    """
    
    def setup_logging(self, module_type: str, module_name: Optional[str] = None):
        """Set up logging for this class."""
        self.logger = setup_module_logger(module_type, module_name)
    
    def get_logger(self):
        """Get the logger for this class."""
        if not hasattr(self, 'logger'):
            self.logger = get_logger()
        return self.logger


# Convenience decorators for common logging patterns
def with_logging(module_type: str):
    """
    Decorator to add logging to a class.
    
    Usage:
        @with_logging('utility')
        class MyClass:
            def some_method(self):
                self.logger.info("This will work")
    """
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            self.logger = setup_module_logger(module_type)
            if original_init:
                original_init(self, *args, **kwargs)
        
        cls.__init__ = new_init
        return cls
    
    return decorator


def auto_logger(func):
    """
    Decorator to automatically add logging to a function.
    
    Usage:
        @auto_logger
        def my_function():
            logger = get_logger()  # Now available directly
            logger.info("Function called")
    """
    def wrapper(*args, **kwargs):
        # Make logger available in function scope
        import builtins
        builtins.logger = get_logger()
        try:
            return func(*args, **kwargs)
        finally:
            if hasattr(builtins, 'logger'):
                delattr(builtins, 'logger')
    
    return wrapper
