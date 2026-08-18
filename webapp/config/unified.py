"""
Unified configuration interface.

This module provides the single, standardized way to access configuration
across the entire application. It combines static TOML configuration with
runtime overrides while maintaining clean dependency management.
"""

import os
import tomllib
from pathlib import Path
from typing import Any, Dict, Optional
from webapp.config.static_config import static_config


class ConfigManager:
    """
    Unified configuration manager that provides a single interface
    for accessing both static and runtime configuration.
    """

    def __init__(self):
        """Initialize configuration manager."""
        self._runtime_overrides: Dict[str, Any] = {}
        self._runtime_config_available = False
        self._desktop_fallback_reason: str | None = None

    def activate_desktop_fallback(self, reason: str) -> None:
        """Use desktop behavior for this process after local service failure."""
        self._desktop_fallback_reason = reason

    def clear_desktop_fallback(self) -> None:
        """Clear the process-local desktop fallback state."""
        self._desktop_fallback_reason = None

    _project_root: Optional[Path] = None

    @classmethod
    def get_project_root(cls) -> Path:
        """
        Get the project root directory with caching.

        Handles both regular execution and PyInstaller/Tauri bundles.

        Returns
        -------
        Path
            The project root directory path.
        """
        if cls._project_root is None:
            if hasattr(os.sys, '_MEIPASS'):
                # In PyInstaller bundle - use the extracted directory
                cls._project_root = Path(os.sys._MEIPASS)
            else:
                cls._project_root = Path(__file__).resolve().parents[2]
        return cls._project_root

    # Convenient static path/property accessors (migrated from the legacy
    # ConfigurationManager in webapp/utilities/configuration/config_manager.py)
    @property
    def desktop_mode(self) -> bool:
        """Get desktop mode setting (with intelligent fallback)."""
        return self.is_desktop_mode()

    @property
    def test_mode(self) -> bool:
        """Get enterprise test mode setting."""
        return self.is_test_mode()

    @property
    def check_size(self) -> bool:
        """Get size checking setting."""
        return self.get('check_size', 'global', False)

    @property
    def check_language(self) -> bool:
        """Get language checking setting."""
        return self.get('check_language', 'global', False)

    @property
    def max_text_size(self) -> int:
        """Get maximum text bytes setting."""
        return self.get('max_text_size', 'global', 20000000)

    @property
    def max_polars_size(self) -> int:
        """Get maximum polars bytes setting."""
        return self.get('max_polars_size', 'global', 150000000)

    @property
    def model_large_path(self) -> str:
        """Get path to large DocuScope spaCy model."""
        return str(self.get_project_root() / "webapp" / "_models" / "en_docusco_spacy")

    @property
    def model_small_path(self) -> str:
        """Get path to small DocuScope spaCy model."""
        return str(self.get_project_root() / "webapp" / "_models" / "en_docusco_spacy_cd")

    @property
    def corpus_dir_path(self) -> str:
        """Get path to corpora directory."""
        return str(self.get_project_root() / "webapp" / "_corpora")

    @property
    def docuscope_logo_path(self) -> str:
        """Get path to DocuScope logo PNG file."""
        return str(self.get_project_root() / "webapp" / "_static" / "docuscope-logo.png")

    @property
    def porpoise_badge_path(self) -> str:
        """Get path to Porpoise badge SVG file."""
        return str(self.get_project_root() / "webapp" / "_static" / "porpoise_badge.svg")

    @property
    def user_guide_badge_path(self) -> str:
        """Get path to User Guide badge SVG file."""
        return str(self.get_project_root() / "webapp" / "_static" / "user_guide.svg")

    @property
    def spacy_model_meta_path(self) -> str:
        """Get path to spaCy model meta.json file."""
        base_path = self.get_project_root() / "webapp" / "_models"
        return str(base_path / "en_docusco_spacy" / "meta.json")

    @property
    def version(self) -> str:
        """Get application version from pyproject.toml."""
        return get_version_from_pyproject()

    def _try_get_runtime_override(self, key: str, section: str) -> tuple[bool, Any]:
        """
        Try to get runtime override value.

        Returns (found, value) tuple. Uses lazy import to avoid circular deps.
        """
        if os.getenv("DOCUSCOPE_DISABLE_RUNTIME_CONFIG", "").strip() == "1":
            self._runtime_config_available = False
            return False, None

        if not static_config.has_key(key, section):
            self._runtime_config_available = False
            return False, None

        if static_config.get_value('test_mode', 'global', False):
            self._runtime_config_available = False
            return False, None

        try:
            # Lazy import to avoid circular dependency
            from webapp.config.runtime_config import runtime_config
            self._runtime_config_available = True

            # Check for runtime override
            override_key = f"{section}.{key}"
            overrides = runtime_config.get_all_overrides()
            if override_key in overrides:
                return True, overrides[override_key]['value']

        except ImportError:
            # Runtime config not available (expected during initialization)
            self._runtime_config_available = False
        except Exception:
            # Runtime config failed - continue with static config
            pass

        return False, None

    def get(self, key: str, section: str = 'global', default: Any = None) -> Any:
        """
        Get configuration value, checking runtime overrides first.

        Special handling for desktop_mode to ensure intelligent fallback works
        transparently through all access methods.

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
            Configuration value (runtime override > static config > default)
        """
        # Special handling for desktop_mode to ensure fallback logic works
        # through all access methods (get_config, AI_CONFIG, etc.)
        if key == 'desktop_mode' and section == 'global':
            return self.is_desktop_mode()

        # Check for runtime override first
        found, value = self._try_get_runtime_override(key, section)
        if found:
            return value

        # Fall back to static configuration
        return static_config.get_value(key, section, default)

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section with runtime overrides applied.

        Parameters
        ----------
        section : str
            Configuration section name

        Returns
        -------
        Dict[str, Any]
            Section configuration with overrides applied
        """
        # Start with static configuration
        config = static_config.get_section(section).copy()

        if (
            not config or
            os.getenv("DOCUSCOPE_DISABLE_RUNTIME_CONFIG", "").strip() == "1" or
            static_config.get_value('test_mode', 'global', False)
        ):
            return config

        # Apply any runtime overrides for this section
        try:
            from webapp.config.runtime_config import runtime_config
            overrides = runtime_config.get_all_overrides()

            section_prefix = f"{section}."
            for override_key, override_data in overrides.items():
                if override_key.startswith(section_prefix):
                    key = override_key[len(section_prefix):]
                    config[key] = override_data['value']

        except (ImportError, Exception):
            # Runtime config not available or failed - use static only
            pass

        return config

    def get_static(self, key: str, section: str = 'global', default: Any = None) -> Any:
        """
        Get static configuration value, ignoring runtime overrides.

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
            Static configuration value or default
        """
        return static_config.get_value(key, section, default)

    def is_test_mode(self) -> bool:
        """Check if enterprise test mode is enabled."""
        return bool(self.get_static('test_mode', 'global', False))

    def is_desktop_mode(self) -> bool:
        """
        Check if application is in configured or fallback desktop mode.

        Returns True if:
        1. desktop_mode is explicitly set to True in config, OR
        2. local Postgres initialization activated the process fallback.

        Returns
        -------
        bool
            True if running in desktop mode (including fallback scenarios)
        """
        if self._desktop_fallback_reason is not None:
            return True

        # Get the configured desktop mode value
        configured_desktop_mode = self.get_static('desktop_mode', 'global', True)

        # If already in desktop mode, return True
        if configured_desktop_mode:
            return True

        return False

    def is_cache_enabled(self) -> bool:
        """
        Check if cache mode is enabled (respects overrides and desktop fallback).

        Cache is automatically disabled in desktop mode for safety.

        Returns
        -------
        bool
            True if cache is enabled and not in desktop mode
        """
        # Cache is disabled in desktop mode for safety
        if self.is_desktop_mode() or self.is_test_mode():
            return False

        return self.get('cache_mode', 'cache', False)

    def get_llm_model(self) -> str:
        """Get configured LLM model."""
        return self.get('llm_model', 'llm', 'gpt-4o-mini')

    def get_llm_params(self) -> Dict[str, Any]:
        """Get LLM parameters."""
        return self.get('llm_parameters', 'llm', {})

    def get_ai_config(self) -> Dict[str, Any]:
        """
        Get standardized AI configuration.

        Returns
        -------
        Dict[str, Any]
            Complete AI configuration with runtime overrides applied
        """
        return {
            'desktop_mode': self.is_desktop_mode(),
            'cache_enabled': self.is_cache_enabled(),
            'model': self.get_llm_model(),
            'parameters': self.get_llm_params(),
            'quota': self.get('quota', 'llm', 10),
            'enabled': self.get('enabled', 'llm', True)
        }

    def get_secret(self, key: str, section: str = "openai", default: Any = None) -> Any:
        """
        Safely get a secret value independently of deployment mode.

        This is the recommended way to access secrets throughout the application.

        Parameters
        ----------
        key : str
            Secret key to retrieve
        section : str
            Secret section (default: "openai")
        default : Any
            Default value if secret not available or in desktop mode

        Returns
        -------
        Any
            Secret value or default
        """
        try:
            import streamlit as st
            if section in st.secrets and key in st.secrets[section]:
                secret_value = st.secrets[section][key]
                # Return default for empty/whitespace-only secrets
                if not secret_value or not str(secret_value).strip():
                    return default
                return secret_value
        except Exception:
            pass

        return default


def get_version_from_pyproject() -> str:
    """
    Extract the version string from pyproject.toml.

    Returns
    -------
    str
        The version string, or '0.0.0' if not found.
    """
    pyproject_path = ConfigManager.get_project_root() / "pyproject.toml"
    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        return data["project"]["version"]
    except Exception:
        return "0.0.0"


# Global configuration manager instance
config = ConfigManager()


# Standardized configuration access functions
def get_config(key: str, section: str = 'global', default: Any = None) -> Any:
    """
    Get configuration value with runtime override support.

    This is the primary function for configuration access across the application.

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
        Configuration value (runtime override > static config > default)
    """
    return config.get(key, section, default)


def get_static_config(key: str, section: str = 'global', default: Any = None) -> Any:
    """
    Get static configuration value, ignoring runtime overrides.

    Use this when you specifically need the TOML-defined value.

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
        Static configuration value or default
    """
    return config.get_static(key, section, default)


def get_config_section(section: str) -> Dict[str, Any]:
    """
    Get entire configuration section with runtime overrides.

    Parameters
    ----------
    section : str
        Configuration section name

    Returns
    -------
    Dict[str, Any]
        Section configuration with overrides applied
    """
    return config.get_section(section)


def get_ai_config() -> Dict[str, Any]:
    """
    Get standardized AI configuration.

    Returns
    -------
    Dict[str, Any]
        Complete AI configuration with runtime overrides applied
    """
    return config.get_ai_config()


def get_secret(key: str, section: str = "openai", default: Any = None) -> Any:
    """
    Safely get a secret value, respecting desktop mode fallback.

    This is the recommended way to access secrets throughout the application.
    In desktop mode (including fallback scenarios), always returns the default
    value to prevent attempts to access missing secrets.

    Parameters
    ----------
    key : str
        Secret key to retrieve
    section : str
        Secret section (default: "openai")
    default : Any
        Default value if secret not available or in desktop mode

    Returns
    -------
    Any
        Secret value or default
    """
    return config.get_secret(key, section, default)
