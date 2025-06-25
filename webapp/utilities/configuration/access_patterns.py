"""
Standardized configuration access patterns.

This module provides standardized ways to access configuration values
across the application, with consistent error handling and fallbacks.
"""

from typing import Any, Dict
from functools import lru_cache
import streamlit as st

from webapp.utilities.configuration import (
    config_manager, get_ai_configuration,
    get_desktop_mode, get_cache_mode
)


class ConfigAccessor:
    """
    Centralized configuration accessor with caching and error handling.
    """

    def __init__(self):
        """Initialize the configuration accessor."""
        self._ai_config_cache = None
        self._general_config_cache = None

    @lru_cache(maxsize=1)
    def get_ai_config(self) -> Dict[str, Any]:
        """
        Get AI configuration with caching.

        Returns
        -------
        Dict[str, Any]
            AI configuration dictionary
        """
        try:
            if self._ai_config_cache is None:
                options, desktop, cache, model, params, quota = get_ai_configuration()
                self._ai_config_cache = {
                    'options': options,
                    'desktop': desktop,
                    'cache': cache,
                    'model': model,
                    'params': params,
                    'quota': quota
                }
            return self._ai_config_cache
        except Exception as e:
            # Fallback to safe defaults
            st.warning(f"Failed to load AI configuration: {e}")
            return {
                'options': {},
                'desktop': True,
                'cache': False,
                'model': 'gpt-3.5-turbo',
                'params': {},
                'quota': 100
            }

    def get_desktop_mode(self) -> bool:
        """Get desktop mode setting with fallback."""
        try:
            return get_desktop_mode()
        except Exception:
            return True  # Safe default

    def get_cache_mode(self) -> bool:
        """Get cache mode setting with fallback."""
        try:
            return get_cache_mode()
        except Exception:
            return False  # Safe default

    def get_ai_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a specific AI configuration setting.

        Parameters
        ----------
        key : str
            Configuration key
        default : Any
            Default value if key not found

        Returns
        -------
        Any
            Configuration value or default
        """
        config = self.get_ai_config()
        return config.get(key, default)

    def get_llm_model(self) -> str:
        """Get LLM model with fallback."""
        return self.get_ai_setting('model', 'gpt-3.5-turbo')

    def get_llm_params(self) -> Dict[str, Any]:
        """Get LLM parameters with fallback."""
        return self.get_ai_setting('params', {})

    def is_ai_enabled(self) -> bool:
        """Check if AI features are enabled."""
        try:
            config = self.get_ai_config()
            return config.get('desktop', True) or config.get('quota', 0) > 0
        except Exception:
            return False

    def clear_cache(self) -> None:
        """Clear cached configuration."""
        self._ai_config_cache = None
        self._general_config_cache = None
        self.get_ai_config.cache_clear()


# Global configuration accessor instance
config_accessor = ConfigAccessor()


def get_standardized_ai_config() -> Dict[str, Any]:
    """
    Get AI configuration using standardized access pattern.

    Returns
    -------
    Dict[str, Any]
        AI configuration dictionary
    """
    return config_accessor.get_ai_config()


def get_config_value(key: str, default: Any = None,
                     config_type: str = 'ai') -> Any:
    """
    Get a configuration value with standardized error handling.

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
            return config_accessor.get_ai_setting(key, default)
        else:
            # Add general config access when needed
            return getattr(config_manager, key, default)
    except Exception:
        return default


def safe_config_access(func, *args, fallback=None, **kwargs):
    """
    Safely access configuration with fallback.

    Parameters
    ----------
    func : callable
        Configuration function to call
    *args
        Arguments for the function
    fallback : Any
        Fallback value if function fails
    **kwargs
        Keyword arguments for the function

    Returns
    -------
    Any
        Function result or fallback
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if fallback is not None:
            return fallback
        # Re-raise if no fallback provided
        raise e


def with_config_fallback(config_keys: Dict[str, Any]):
    """
    Decorator to provide configuration fallbacks for functions.

    Parameters
    ----------
    config_keys : Dict[str, Any]
        Mapping of configuration keys to fallback values

    Returns
    -------
    callable
        Decorated function with config fallbacks
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Inject configuration values with fallbacks
            for key, fallback in config_keys.items():
                if key not in kwargs:
                    kwargs[key] = get_config_value(key, fallback)
            return func(*args, **kwargs)
        return wrapper
    return decorator
