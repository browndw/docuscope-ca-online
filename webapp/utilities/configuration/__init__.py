"""
Configuration management utilities.

This module provides functions for loading and managing configuration
from TOML files and other configuration sources, including centralized
logging configuration for the entire application.
"""

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
    is_online_mode,
    should_check_size,
    should_check_language,
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
    get_log_directory,
    get_logger
)


__all__ = [
    # Configuration manager
    'get_version_from_pyproject',
    'import_options_general',
    'get_ai_configuration',
    'ConfigurationManager',
    'config_manager',
    # Convenient accessors
    'get_desktop_mode',
    'get_cache_mode',
    'get_llm_model',
    'get_llm_parameters',
    'is_online_mode',
    'should_check_size',
    'should_check_language',
    'get_max_text_size',
    'get_max_polars_size',
    # Logging utilities
    'LoggingConfig',
    'get_logging_config',
    'setup_ai_logging',
    'setup_page_logging',
    'setup_utility_logging',
    'setup_debug_logging',
    'get_log_directory',
    'get_logger'

]
