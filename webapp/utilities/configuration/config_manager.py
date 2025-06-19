"""
Configuration management utilities for the application.

This module provides functions for loading configuration files,
version information, and application settings.
"""

import tomli
from pathlib import Path
from typing import Dict, Any, Tuple

# Get the project root directory
project_root = Path(__file__).resolve().parents[3]


def get_version_from_pyproject() -> str:
    """
    Extract the version string from pyproject.toml.

    Returns
    -------
    str
        The version string, or '0.0.0' if not found.
    """
    pyproject_path = project_root / "pyproject.toml"
    try:
        with open(pyproject_path, "rb") as f:
            data = tomli.load(f)
        return data["project"]["version"]
    except Exception:
        return "0.0.0"


def import_options_general(options_path: str = None) -> Dict[str, Any]:
    """
    Import general options from a TOML file.

    Parameters
    ----------
    options_path : str, optional
        The path to the options TOML file. If None, defaults to webapp/config/options.toml.

    Returns
    -------
    dict
        A dictionary containing the loaded options.
        If the file cannot be decoded,
        returns a dictionary with default option values.
    """
    if options_path is None:
        options_path = str(project_root / "webapp"  "config" / "options.toml")

    try:
        with open(options_path, mode="rb") as fp:
            options = tomli.load(fp)
    except tomli.TOMLDecodeError:
        options = {}
        options['global'] = {}
        options['global']['check_size'] = False
        options['global']['check_language'] = False
        options['global']['enable_save'] = False
        options['global']['desktop_mode'] = False
        options['global']['max_bytes'] = 0
        options['llm'] = {}
        options['llm']['llm_parameters'] = {}
        options['llm']['llm_model'] = 'gpt-4o-mini'
        options['cache'] = {}
        options['cache']['cache_mode'] = False
        options['cache']['cache_location'] = None

    return options


def get_ai_configuration() -> Tuple[Dict[str, Any], bool, bool, str, str, int]:
    """
    Get AI-specific configuration settings.

    Returns
    -------
    tuple
        A tuple containing (_options, DESKTOP, CACHE, LLM_MODEL, LLM_PARAMS, QUOTA)
    """
    options_path = str(project_root / "webapp" / "config" / "options.toml")
    _options = import_options_general(options_path)

    DESKTOP = _options['global']['desktop_mode']
    LLM_PARAMS = _options['llm']['llm_parameters']
    LLM_MODEL = _options['llm']['llm_model']
    QUOTA = _options['llm']['quota']

    if DESKTOP:
        CACHE = False
    else:
        CACHE = _options['cache']['cache_mode']

    return _options, DESKTOP, CACHE, LLM_MODEL, LLM_PARAMS, QUOTA
