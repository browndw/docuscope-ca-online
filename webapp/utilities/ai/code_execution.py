"""
Safe code execution utilities for AI-generated plotting code.

This module provides secure code execution for AI-generated plotting code
using RestrictedPython to prevent malicious code execution.
"""

import re

# Import centralized logging configuration and logger
from webapp.utilities.configuration.logging_config import get_logger

logger = get_logger()


# Security: Define forbidden patterns for code safety.
# These are a defense-in-depth text scan in addition to RestrictedPython's
# compiled-bytecode sandbox and `safer_getattr` attribute guard, since some
# escapes (e.g. pandas `.eval()`/`.query()`, dunder attribute chains) are not
# stopped by the bytecode restrictions alone.
FORBIDDEN_PATTERNS = [
    r'^\s*(import|from)\s',  # import / from-import statements anywhere in the code
    r'\bexec\s*\(',           # exec(
    r'\beval\s*\(',           # eval(
    r'\.eval\s*\(',           # DataFrame/pd.eval( can run arbitrary Python internally
    r'\.query\s*\(',          # DataFrame.query( can evaluate arbitrary expressions
    r'\bopen\s*\(',           # open(
    r'\bos\.',                # os. usage anywhere, not just at line start
    r'\bsys\.',                # sys. usage anywhere
    r'\bsubprocess\.',        # subprocess. usage anywhere
    r'__\w+__',                # dunder access (e.g. __class__, __globals__, __import__)
]


def is_code_safe(plot_code: str) -> bool:
    """
    Check if the provided code is safe to execute.

    Parameters
    ----------
    plot_code : str
        The code to check for safety.

    Returns
    -------
    bool
        True if the code is safe, False otherwise.
    """
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, plot_code, re.MULTILINE):
            logger.warning(f"Unsafe pattern matched: {pattern} in code: {plot_code}")
            return False
    return True


def strip_imports(code: str) -> str:
    """
    Remove all import statements from the code.

    Parameters
    ----------
    code : str
        The code from which to remove imports.

    Returns
    -------
    str
        Code with import statements removed.
    """
    return "\n".join(
        line for line in code.splitlines()
        if not re.match(r'^\s*(import|from)\s', line)
    )
