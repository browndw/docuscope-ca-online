"""
Centralized configuration management for the application.

This module provides a robust, cached configuration management system with
runtime sanity checks and automatic fallbacks for both desktop (Tauri) and
online (Streamlit Cloud) deployments.
"""

import os
import tomli
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Import centralized logging configuration and logger
from webapp.utilities.configuration.logging_config import get_logger

logger = get_logger()


class ConfigurationManager:
    """
    Centralized configuration manager with caching and runtime sanity checks.

    This class ensures that configuration is loaded once, cached for the app lifetime,
    and includes automatic adjustments based on environment capabilities (e.g., secrets
    availability for online mode).
    """

    _instance: Optional['ConfigurationManager'] = None
    _config_cache: Optional[Dict[str, Any]] = None
    _project_root: Optional[Path] = None

    def __new__(cls) -> 'ConfigurationManager':
        """Singleton pattern to ensure only one configuration manager exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_project_root(cls) -> Path:
        """
        Get the project root directory with caching.

        Returns
        -------
        Path
            The project root directory path.
        """
        if cls._project_root is None:
            cls._project_root = Path(__file__).resolve().parents[3]
        return cls._project_root

    @classmethod
    def get_config_path(cls) -> Path:
        """
        Get the path to the options.toml configuration file.

        Returns
        -------
        Path
            The path to the configuration file.
        """
        return cls.get_project_root() / "webapp" / "config" / "options.toml"

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration values as fallback.

        Returns
        -------
        Dict[str, Any]
            Default configuration dictionary.
        """
        return {
            'global': {
                'check_size': False,
                'check_language': False,
                'desktop_mode': True,  # Safe default
                'max_text_size': 20000000,
                'max_polars_size': 150000000
            },
            'llm': {
                'llm_parameters': {
                    'temperature': 0.7,
                    'top_p': 0.7,
                    'max_tokens': 500,
                    'frequency_penalty': 0,
                    'presence_penalty': 0
                },
                'llm_model': 'gpt-4o-mini',
                'quota': 100  # Default quota for LLM usage
            },
            'cache': {
                'cache_mode': False,  # Safe default
                'cache_location': 'firestore'
            },
            'session': {
                'inactivity_timeout_minutes': 90,
                'inactivity_warning_minutes': 85,
                'absolute_timeout_hours': 24,
                'absolute_warning_hours': 23.5
            }
        }

    def _check_secrets_availability(self) -> bool:
        """
        Check if required secrets are available for online mode.

        Returns
        -------
        bool
            True if secrets are available, False otherwise.
        """
        try:
            import streamlit as st
            # Check for OpenAI API key
            if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
                openai_key = st.secrets["openai"]["api_key"]
                if openai_key and len(str(openai_key).strip()) > 0:
                    return True
        except Exception:
            pass

        # Check for environment variables as fallback
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key and len(openai_key.strip()) > 0:
            return True

        return False

    def _apply_runtime_sanity_checks(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply runtime sanity checks and automatic adjustments to configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            The loaded configuration dictionary.

        Returns
        -------
        Dict[str, Any]
            The configuration with runtime adjustments applied.
        """
        original_desktop_mode = config['global']['desktop_mode']
        secrets_available = self._check_secrets_availability()

        # If desktop_mode = false but secrets are missing, force desktop mode
        if not original_desktop_mode and not secrets_available:
            logger.warning(
                "Configuration specifies desktop_mode=false, but required secrets "
                "(OpenAI API key) are not available. Automatically switching to "
                "desktop_mode=true and cache_mode=false for safety."
            )
            config['global']['desktop_mode'] = True
            config['cache']['cache_mode'] = False

        # If desktop mode is enabled, disable caching for safety
        if config['global']['desktop_mode']:
            config['cache']['cache_mode'] = False

        return config

    def load_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load and cache the application configuration.

        Parameters
        ----------
        force_reload : bool, optional
            If True, force reload the configuration ignoring cache.

        Returns
        -------
        Dict[str, Any]
            The loaded and processed configuration dictionary.
        """
        if self._config_cache is not None and not force_reload:
            return self._config_cache

        config_path = self.get_config_path()

        try:
            with open(config_path, "rb") as f:
                config = tomli.load(f)
        except (FileNotFoundError, tomli.TOMLDecodeError) as e:
            logger.warning(
                f"Could not load configuration from {config_path}: {e}. "
                "Using default configuration."
            )
            config = self._get_default_config()
        except Exception:
            config = self._get_default_config()

        # Apply runtime sanity checks
        config = self._apply_runtime_sanity_checks(config)

        # Cache the configuration
        self._config_cache = config

        return config

    def get_config(self) -> Dict[str, Any]:
        """
        Get the cached configuration, loading it if necessary.

        Returns
        -------
        Dict[str, Any]
            The application configuration dictionary.
        """
        return self.load_config()

    def get_ai_config(self) -> Tuple[Dict[str, Any], bool, bool, str, Dict[str, Any], int]:
        """
        Get AI-specific configuration settings.

        Returns
        -------
        Tuple[Dict[str, Any], bool, bool, str, Dict[str, Any], int]
            A tuple containing (config, desktop_mode, cache_mode,
            llm_model, llm_params, quota)
        """
        config = self.get_config()

        desktop_mode = config['global']['desktop_mode']
        llm_params = config['llm']['llm_parameters']
        llm_model = config['llm']['llm_model']
        quota = config['llm']['quota']

        # Cache mode logic
        if desktop_mode:
            cache_mode = False
        else:
            cache_mode = config['cache']['cache_mode']

        return config, desktop_mode, cache_mode, llm_model, llm_params, quota

    # Convenient property accessors for common configuration values
    @property
    def desktop_mode(self) -> bool:
        """Get desktop mode setting."""
        return self.get_config()['global']['desktop_mode']

    @property
    def cache_mode(self) -> bool:
        """Get cache mode setting (considers desktop mode logic)."""
        config = self.get_config()
        if config['global']['desktop_mode']:
            return False
        return config['cache']['cache_mode']

    @property
    def check_size(self) -> bool:
        """Get size checking setting."""
        return self.get_config()['global']['check_size']

    @property
    def check_language(self) -> bool:
        """Get language checking setting."""
        return self.get_config()['global']['check_language']

    @property
    def max_text_size(self) -> int:
        """Get maximum text bytes setting."""
        return self.get_config()['global']['max_text_size']

    @property
    def max_polars_size(self) -> int:
        """Get maximum polars bytes setting."""
        return self.get_config()['global']['max_polars_size']

    @property
    def llm_model(self) -> str:
        """Get LLM model setting."""
        return self.get_config()['llm']['llm_model']

    @property
    def llm_parameters(self) -> Dict[str, Any]:
        """Get LLM parameters setting."""
        return self.get_config()['llm']['llm_parameters']

    @property
    def llm_quota(self) -> int:
        """Get LLM quota setting."""
        return self.get_config()['llm']['quota']

    @property
    def cache_location(self) -> str:
        """Get cache location setting."""
        return self.get_config()['cache']['cache_location']

    # Session timeout settings
    @property
    def inactivity_timeout_minutes(self) -> int:
        """Get inactivity timeout in minutes."""
        return self.get_config()['session']['inactivity_timeout_minutes']

    @property
    def inactivity_warning_minutes(self) -> int:
        """Get inactivity warning threshold in minutes."""
        return self.get_config()['session']['inactivity_warning_minutes']

    @property
    def absolute_timeout_hours(self) -> float:
        """Get absolute session timeout in hours."""
        return self.get_config()['session']['absolute_timeout_hours']

    @property
    def absolute_warning_hours(self) -> float:
        """Get absolute session warning threshold in hours."""
        return self.get_config()['session']['absolute_warning_hours']

    # Static resource paths
    @property
    def google_logo_path(self) -> str:
        """Get path to Google logo SVG file."""
        return str(self.get_project_root() / "webapp" / "_static" / "web_light_rd_na.svg")

    @property
    def model_large_path(self) -> str:
        """Get path to large DocuScope spaCy model."""
        return str(self.get_project_root() / "webapp" / "_models" / "en_docusco_spacy")

    @property
    def model_small_path(self) -> str:
        """Get path to small DocuScope spaCy model."""
        return str(self.get_project_root() / "webapp" / "_models" / "en_docusco_spacy_cd")

    @property
    def static_dir_path(self) -> str:
        """Get path to static files directory."""
        return str(self.get_project_root() / "webapp" / "_static")

    @property
    def models_dir_path(self) -> str:
        """Get path to models directory."""
        return str(self.get_project_root() / "webapp" / "_models")

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
        """Get application version from pyproject.toml (cached)."""
        return get_version_from_pyproject()

    # Convenience methods for common access patterns
    def get_global_settings(self) -> Dict[str, Any]:
        """Get all global settings."""
        return self.get_config()['global']

    def get_llm_settings(self) -> Dict[str, Any]:
        """Get all LLM settings."""
        return self.get_config()['llm']

    def get_cache_settings(self) -> Dict[str, Any]:
        """Get all cache settings."""
        return self.get_config()['cache']

    def get_session_settings(self) -> Dict[str, Any]:
        """Get all session timeout settings."""
        return self.get_config()['session']

    def is_online_mode(self) -> bool:
        """Check if running in online mode (not desktop mode)."""
        return not self.desktop_mode

    def should_check_size(self) -> bool:
        """Alias for check_size property for better readability."""
        return self.check_size

    def should_check_language(self) -> bool:
        """Alias for check_language property for better readability."""
        return self.check_language


# Global configuration manager instance
config_manager = ConfigurationManager()


def get_version_from_pyproject() -> str:
    """
    Extract the version string from pyproject.toml.

    Returns
    -------
    str
        The version string, or '0.0.0' if not found.
    """
    pyproject_path = ConfigurationManager.get_project_root() / "pyproject.toml"
    try:
        with open(pyproject_path, "rb") as f:
            data = tomli.load(f)
        return data["project"]["version"]
    except Exception:
        return "0.0.0"


# Backward compatibility functions
def import_options_general(options_path: str = None) -> Dict[str, Any]:
    """
    Import general options from a TOML file.

    This is a backward compatibility function. For new code, use
    config_manager.get_config() directly.

    Parameters
    ----------
    options_path : str, optional
        The path to the options TOML file. Ignored - uses centralized config.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the loaded options.
    """
    return config_manager.get_config()


def get_ai_configuration() -> Tuple[Dict[str, Any], bool, bool, str, Dict[str, Any], int]:
    """
    Get AI-specific configuration settings.

    This is a backward compatibility function. For new code, use
    config_manager.get_ai_config() directly.

    Returns
    -------
    Tuple[Dict[str, Any], bool, bool, str, Dict[str, Any], int]
        A tuple containing (config, desktop_mode, cache_mode, llm_model, llm_params, quota)
    """
    return config_manager.get_ai_config()


# Convenient module-level accessors for common patterns
def get_desktop_mode() -> bool:
    """Get desktop mode setting - convenient module-level function."""
    return config_manager.desktop_mode


def get_cache_mode() -> bool:
    """Get cache mode setting - convenient module-level function."""
    return config_manager.cache_mode


def get_llm_model() -> str:
    """Get LLM model setting - convenient module-level function."""
    return config_manager.llm_model


def get_llm_parameters() -> Dict[str, Any]:
    """Get LLM parameters setting - convenient module-level function."""
    return config_manager.llm_parameters


def is_online_mode() -> bool:
    """Check if running in online mode - convenient module-level function."""
    return config_manager.is_online_mode()


def should_check_size() -> bool:
    """Check if size validation should be performed - convenient module-level function."""
    return config_manager.should_check_size()


def should_check_language() -> bool:
    """Check if language validation should be performed - convenient function."""
    return config_manager.should_check_language()


def get_max_text_size() -> int:
    """Get maximum text size setting - convenient module-level function."""
    return config_manager.max_text_size


def get_max_polars_size() -> int:
    """Get maximum polars size setting - convenient module-level function."""
    return config_manager.max_polars_size


# Set up centralized logging for configuration management
try:
    from webapp.utilities.configuration.logging_config import setup_utility_logging
    setup_utility_logging("configuration")
except ImportError:
    # Fallback if logging_config is not available (shouldn't happen in normal operation)
    pass
