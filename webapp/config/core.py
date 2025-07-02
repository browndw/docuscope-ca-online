"""
Configuration kernel - the foundation layer with no dependencies.
This module contains only the most basic configuration access patterns.
"""

import os
from typing import Any, Dict
from functools import lru_cache
import tomllib  # or tomli for older Python


@lru_cache(maxsize=1)
def load_raw_config() -> Dict[str, Any]:
    """Load raw configuration from TOML file with no dependencies."""
    config_path = os.path.join(os.path.dirname(__file__), 'options.toml')
    try:
        with open(config_path, 'rb') as f:
            return tomllib.load(f)
    except Exception:
        # For production, you might want to log this error
        return {}


def get_raw_config_value(key: str, section: str = 'global', default: Any = None) -> Any:
    """Get a raw configuration value with no dependencies."""
    config = load_raw_config()
    return config.get(section, {}).get(key, default)
