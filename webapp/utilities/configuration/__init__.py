"""
Configuration management utilities.

This module provides centralized logging configuration for the entire
application. General app configuration access lives in
webapp/config/unified.py (the ConfigManager `config` singleton).
"""

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
