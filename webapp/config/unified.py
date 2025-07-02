"""
Unified configuration interface.

This module provides the single, standardized way to access configuration
across the entire application. It combines static TOML configuration with
runtime overrides while maintaining clean dependency management.
"""

from typing import Any, Dict
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
    
    def _try_get_runtime_override(self, key: str, section: str) -> tuple[bool, Any]:
        """
        Try to get runtime override value.
        
        Returns (found, value) tuple. Uses lazy import to avoid circular deps.
        """
        if not self._runtime_config_available:
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
                pass
            except Exception:
                # Runtime config failed - continue with static config
                pass
                
        return False, None
    
    def get(self, key: str, section: str = 'global', default: Any = None) -> Any:
        """
        Get configuration value, checking runtime overrides first.
        
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
    
    def is_desktop_mode(self) -> bool:
        """Check if application is in desktop mode."""
        return self.get('desktop_mode', 'global', True)
    
    def is_cache_enabled(self) -> bool:
        """Check if cache mode is enabled (respects runtime overrides)."""
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
