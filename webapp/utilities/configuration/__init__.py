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
    get_ai_configuration
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
    'load_configuration',
    'get_timestamp',
    'get_options_doc_tags',
    # Logging utilities
    'LoggingConfig',
    'get_logging_config',
    'setup_ai_logging',
    'setup_page_logging',
    'setup_utility_logging',
    'setup_debug_logging',
    'get_log_directory'
]


def load_configuration(config_path: str) -> dict:
    """
    Load configuration from a TOML file.
    
    Alias for import_options_general for clarity.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file.
        
    Returns
    -------
    dict
        Configuration dictionary.
    """
    return import_options_general(config_path)


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
