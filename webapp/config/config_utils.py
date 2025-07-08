"""
Centralized configuration utilities for runtime configuration management.

This module provides helper functions to access runtime configuration that can be
overridden at runtime through the admin interface without requiring application restart.
"""

from webapp.config.runtime_config import RuntimeConfigManager

# Module-level instance to avoid session state serialization issues
_runtime_config_instance = None


def get_runtime_config():
    """
    Get runtime config manager instance.
    Centralizes access to runtime configuration throughout the application.
    Uses module-level instance to avoid Streamlit session state serialization issues.
    """
    global _runtime_config_instance
    if _runtime_config_instance is None:
        _runtime_config_instance = RuntimeConfigManager()
    return _runtime_config_instance


def get_runtime_setting(key: str, default_value=None, section: str = None):
    """
    Convenience function to get a runtime setting.
    Checks runtime overrides first, then falls back to TOML config.

    Args:
        key: Configuration key
        default_value: Default value if not found
        section: TOML section (optional)

    Returns:
        The configuration value, checking runtime overrides first
    """
    runtime_config = get_runtime_config()
    return runtime_config.get_config_value(key, default_value, section)
