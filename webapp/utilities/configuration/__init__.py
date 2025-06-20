"""
Configuration management utilities.

This module provides functions for loading and managing configuration
from TOML files and other configuration sources, including centralized
logging configuration for the entire application.
"""

from datetime import datetime
from webapp.utilities.configuration.config_manager import (
    get_version_from_pyproject,
    import_options_general,
    get_ai_configuration,
    ConfigurationManager,
    config_manager,
    # Convenient module-level accessors
    get_desktop_mode,
    get_cache_mode,
    get_llm_model,
    get_llm_parameters,
    get_max_bytes_text,
    get_max_bytes_polars,
    is_online_mode,
    should_check_size,
    should_check_language,
    get_enable_language_detection,
    get_max_text_size,
    get_max_polars_size
)
from webapp.utilities.configuration.logging_config import (
    LoggingConfig,
    get_logging_config,
    setup_ai_logging,
    setup_page_logging,
    setup_utility_logging,
    setup_debug_logging,
    get_log_directory
)

__all__ = [
    'get_version_from_pyproject',
    'import_options_general',
    'get_ai_configuration',
    'ConfigurationManager',
    'config_manager',
    'load_configuration',
    'get_timestamp',
    'get_options_doc_tags',
    # Convenient accessors
    'get_desktop_mode',
    'get_cache_mode',
    'get_llm_model',
    'get_llm_parameters',
    'get_max_bytes_text',
    'get_max_bytes_polars',
    'is_online_mode',
    'should_check_size',
    'should_check_language',
    'get_enable_language_detection',
    'get_max_text_size',
    'get_max_polars_size',
    # Logging utilities
    'LoggingConfig',
    'get_logging_config',
    'setup_ai_logging',
    'setup_page_logging',
    'setup_utility_logging',
    'setup_debug_logging',
    'get_log_directory'
]


def load_configuration(config_path: str = None) -> dict:
    """
    Load configuration from a TOML file.
    
    For new code, prefer using config_manager.get_config() directly.
    This function is kept for backward compatibility.
    
    Parameters
    ----------
    config_path : str, optional
        Path to the configuration file. Ignored - uses centralized config.
        
    Returns
    -------
    dict
        Configuration dictionary.
    """
    return config_manager.get_config()


def get_timestamp() -> str:
    """
    Generate a formatted timestamp string.

    Returns
    -------
    str
        A string containing the current timestamp in the format 'YYYY-MM-DD_HH-MM-SS'.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_options_doc_tags() -> dict:
    """
    Get options for document tags.
    """
    # TODO: Implement this function
    return {}
