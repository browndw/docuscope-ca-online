"""
Safe code execution utilities for AI-generated plotting code.

This module provides secure code execution for AI-generated plotting code
using RestrictedPython to prevent malicious code execution.
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins
from RestrictedPython.Guards import guarded_unpack_sequence
from RestrictedPython.Eval import default_guarded_getitem as guarded_getitem
from RestrictedPython.Eval import default_guarded_getiter as guarded_getiter

# Import centralized logging configuration and logger
from webapp.utilities.configuration.logging_config import get_logger

logger = get_logger()


# Security: Define forbidden patterns for code safety (legacy working version)
FORBIDDEN_PATTERNS = [
    r'^\s*import\s',         # import statement at line start
    r'\bexec\s*\(',          # exec(
    r'\beval\s*\(',          # eval(
    r'\bopen\s*\(',          # open(
    r'^\s*os\.',             # os. usage at line start
    r'^\s*sys\.',            # sys. usage at line start
    r'^\s*subprocess\.',     # subprocess. usage at line start
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
            logger(f"Unsafe pattern matched: {pattern} in code: {plot_code}")
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
        if not re.match(r'^\s*import\s', line)
    )


def plotbot_code_execute(
    plot_code: str,
    df: pd.DataFrame,
    plot_lib: str
) -> dict:
    """
    Safely execute AI-generated plotting code.

    Parameters
    ----------
    plot_code : str
        The plotting code to execute.
    df : pd.DataFrame
        The dataframe to plot.
    plot_lib : str
        The plotting library to use ('matplotlib', 'seaborn', 'plotly.express').

    Returns
    -------
    dict
        Result dictionary with 'type' and 'value' keys.
    """
    if not isinstance(plot_code, str) or not plot_code.strip():
        return {
            "type": "error",
            "value": ("Sorry, I couldn't generate your plot. "
                      "Please try rephrasing your request.")
        }

    # Strip import statements before safety check
    plot_code = strip_imports(plot_code)
    if not is_code_safe(plot_code):
        return {
            "type": "error",
            "value": "Sorry, your request included unsafe code and could not be executed."
        }

    exec_locals = {}
    allowed_globals = {
        "__builtins__": safe_builtins,
        "df": df,
        "_getitem_": guarded_getitem,
        "_unpack_sequence_": guarded_unpack_sequence,
        "_getiter_": guarded_getiter,
    }

    # Add plotting library to allowed globals
    if plot_lib == "matplotlib":
        allowed_globals["plt"] = plt
    elif plot_lib == "seaborn":
        allowed_globals["sns"] = sns
        allowed_globals["plt"] = plt
    elif plot_lib == "plotly.express":
        allowed_globals["px"] = px

    try:
        byte_code = compile_restricted(plot_code, '<string>', 'exec')
        exec(byte_code, allowed_globals, exec_locals)

        if "fig" in exec_locals:
            fig = exec_locals["fig"]
            return {
                "type": "plot",
                "value": fig
            }
        else:
            return {
                "type": "error",
                "value": "Sorry, no plot was generated. Please try a different request."
            }

    except SyntaxError:
        return {
            "type": "error",
            "value": ("Sorry, there was a problem with the plot code. "
                      "Please try a different request.")
        }
    except Exception:
        return {
            "type": "error",
            "value": "Sorry, something went wrong while generating your plot."
        }
