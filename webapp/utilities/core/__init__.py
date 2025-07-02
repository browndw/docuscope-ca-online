"""
Core utilities for application integration and orchestration.

This module provides core functionality that integrates and orchestrates
various utilities across the application, including configuration management,
error handling, session management, and UI components.

Note: For UI error boundaries, import SafeComponentRenderer directly from:
from webapp.utilities.ui.error_boundaries import SafeComponentRenderer
"""

from .integration import (
    # Main interface
    app_core,
    ApplicationCore,
    # Convenience functions
    safe_config_value,
    validate_session_health,
    get_session_diagnostics,
    # Direct imports for flexibility
    error_handler,
    enhanced_validator,
    widget_key_manager,
    get_corpus_data_manager
)

__all__ = [
    # Main interface
    'app_core',
    'ApplicationCore',
    # Convenience functions
    'safe_config_value',
    'validate_session_health',
    'get_session_diagnostics',
    # Direct imports for flexibility
    'error_handler',
    'enhanced_validator',
    'widget_key_manager',
    'get_corpus_data_manager'
]
