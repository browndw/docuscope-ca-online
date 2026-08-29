"""
Static configuration foundation - TOML-only, no dependencies.

This module provides the foundational configuration layer that reads only
from TOML files with no external dependencies. This is the bottom layer
of the configuration hierarchy and should never import from other webapp modules.
"""

import toml
from typing import Any, Dict, Optional
from pathlib import Path


class StaticConfigManager:
    """
    Static configuration manager that reads only from TOML files.

    This class has zero external dependencies to avoid circular imports.
    It serves as the foundation for all configuration access.
    """

    def __init__(self):
        """Initialize static configuration manager."""
        self._config_cache: Optional[Dict[str, Any]] = None
        self._config_path = self._find_config_path()

    def _find_config_path(self) -> Path:
        """Find the configuration file path."""
        # Start from current directory and work up to find config
        current_dir = Path.cwd()

        # Common config paths to check
        config_paths = [
            current_dir / "webapp" / "config" / "options.toml",
            current_dir / "config" / "options.toml",
            current_dir / "options.toml"
        ]

        for path in config_paths:
            if path.exists():
                return path

        # Default fallback
        return current_dir / "webapp" / "config" / "options.toml"

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from TOML file."""
        if self._config_cache is not None:
            return self._config_cache

        try:
            if self._config_path.exists():
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    self._config_cache = toml.load(f)
            else:
                # Return empty config if file doesn't exist
                self._config_cache = {}
        except Exception:
            # Fail-safe: return empty config on any error
            self._config_cache = {}

        return self._config_cache

    def get_value(self, key: str, section: str = 'global', default: Any = None) -> Any:
        """
        Get a configuration value from the specified section.

        Parameters
        ----------
        key : str
            Configuration key
        section : str
            Configuration section (default: 'global')
        default : Any
            Default value if key not found

        Returns
        -------
        Any
            Configuration value or default
        """
        try:
            config = self._load_config()
            section_config = config.get(section, {})
            return section_config.get(key, default)
        except Exception:
            return default

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.

        Parameters
        ----------
        section : str
            Configuration section name

        Returns
        -------
        Dict[str, Any]
            Section configuration or empty dict
        """
        try:
            config = self._load_config()
            return config.get(section, {})
        except Exception:
            return {}

    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration data.

        Returns
        -------
        Dict[str, Any]
            Complete configuration or empty dict
        """
        return self._load_config()

    def has_section(self, section: str) -> bool:
        """
        Check if configuration section exists.

        Parameters
        ----------
        section : str
            Section name to check

        Returns
        -------
        bool
            True if section exists
        """
        try:
            config = self._load_config()
            return section in config
        except Exception:
            return False

    def has_key(self, key: str, section: str = 'global') -> bool:
        """
        Check if configuration key exists in section.

        Parameters
        ----------
        key : str
            Key name to check
        section : str
            Section to check in

        Returns
        -------
        bool
            True if key exists in section
        """
        try:
            config = self._load_config()
            return key in config.get(section, {})
        except Exception:
            return False

    def clear_cache(self) -> None:
        """Clear cached configuration."""
        self._config_cache = None


# Global static configuration instance
static_config = StaticConfigManager()


# Convenience functions for common access patterns
def get_static_value(key: str, section: str = 'global', default: Any = None) -> Any:
    """Get static configuration value."""
    return static_config.get_value(key, section, default)


def get_static_section(section: str) -> Dict[str, Any]:
    """Get static configuration section."""
    return static_config.get_section(section)


def is_desktop_mode() -> bool:
    """Check if application is in desktop mode."""
    return get_static_value('desktop_mode', 'global', True)


def get_cache_mode() -> bool:
    """Check if cache mode is enabled."""
    return get_static_value('cache_mode', 'cache', False)


def get_llm_model() -> str:
    """Get configured LLM model."""
    return get_static_value('llm_model', 'llm', 'gpt-4o-mini')


def get_llm_params() -> Dict[str, Any]:
    """Get LLM parameters."""
    return get_static_value('llm_parameters', 'llm', {})
