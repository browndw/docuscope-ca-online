"""
Consolidated imports for common utilities - Phase 3 optimization.

This module provides grouped imports to reduce repetitive import statements
across page files and utilities, making the codebase more maintainable
for both online and desktop deployments.
"""

# Core application patterns
from webapp.utilities.core import app_core
from webapp.utilities.error_handling.enhanced import (
    enhanced_error_handler,
    user_data_safe,
    system_safe,
    operation_safe
)

# Session and state management
from webapp.utilities.session import (
    get_or_init_user_session,
    load_metadata
)
from webapp.utilities.state import (
    persist,
    CorpusKeys,
    SessionKeys,
    TargetKeys,
    WarningKeys
)

# Corpus and data management
from webapp.utilities.corpus import (
    get_corpus_data_manager,
    clear_corpus_data
)

# UI components (common patterns)
from webapp.utilities.ui import (
    render_table_generation_interface,
    sidebar_help_link,
    tagset_selection
)

# Analysis utilities
from webapp.utilities.analysis import (
    generate_frequency_table,
    generate_tags_table,
    has_target_corpus
)

# Menu and authentication
from webapp.menu import (
    menu,
    require_login
)


class CommonImports:
    """
    Container for commonly used imports to reduce import statements.

    This pattern allows pages to import everything they typically need
    in a single line while maintaining clear dependency management.
    """

    # Core patterns
    app_core = app_core
    error_handler = enhanced_error_handler

    # Decorators
    user_data_safe = user_data_safe
    system_safe = system_safe
    operation_safe = operation_safe

    # Session management
    get_or_init_user_session = get_or_init_user_session
    load_metadata = load_metadata
    session_manager = app_core.session_manager

    # State management
    persist = persist
    widget_manager = app_core.widget_manager

    # State keys
    CorpusKeys = CorpusKeys
    SessionKeys = SessionKeys
    TargetKeys = TargetKeys
    WarningKeys = WarningKeys

    # Corpus management
    get_corpus_data_manager = get_corpus_data_manager
    clear_corpus_data = clear_corpus_data

    # UI patterns
    sidebar_help_link = sidebar_help_link
    render_table_generation_interface = render_table_generation_interface
    tagset_selection = tagset_selection

    # Analysis
    generate_frequency_table = generate_frequency_table
    generate_tags_table = generate_tags_table
    has_target_corpus = has_target_corpus

    # Menu
    menu = menu
    require_login = require_login


# For pages that want everything in one import
common = CommonImports()


# Specialized import groups for specific use cases

class FrequencyPageImports:
    """Imports specifically for frequency analysis pages."""

    from webapp.utilities.ui import (
        render_data_table_interface,
        tagset_selection
    )

    from webapp.utilities.analysis import (
        generate_frequency_table,
        generate_tags_table
    )


class ComparisonPageImports:
    """Imports specifically for corpus comparison pages."""

    from webapp.utilities.ui import (
        keyness_settings_info,
        keyness_sort_controls,
        reference_parts,
        target_parts
    )

    from webapp.utilities.analysis import (
        generate_keyness_table,
        generate_keyness_parts
    )

    from webapp.utilities.plotting import (
        plot_compare_corpus_bar,
        plot_download_link
    )


class ExportPageImports:
    """Imports specifically for export/download pages."""

    from webapp.utilities.exports import (
        convert_to_excel,
        export_corpus_data
    )

    from webapp.utilities.ui import (
        toggle_download,
        render_dataframe
    )


# Convenience functions for the most common patterns
def get_page_basics():
    """
    Get the most common imports needed by any page.

    Returns
    -------
    tuple
        (user_session_id, session, error_handler)
    """
    user_session_id, session = get_or_init_user_session()
    return user_session_id, session, enhanced_error_handler


def register_page_widgets(widget_list: list, page_name: str = None):
    """
    Convenience function for widget registration with consistent pattern.

    Parameters
    ----------
    widget_list : list
        List of widget keys to register
    page_name : str, optional
        Page name for error context
    """
    try:
        app_core.register_page_widgets(widget_list)
    except Exception as e:
        context = f"Widget registration for {page_name}" if page_name else "Widget registration"  # noqa: E501
        enhanced_error_handler.handle_system_error(e, context)


def ensure_corpus_ready(
        user_session_id: str,
        corpus_type: str = "target"
):
    """
    Ensure corpus is ready with proper error handling.

    Parameters
    ----------
    user_session_id : str
        User session ID
    corpus_type : str
        Corpus type to check

    Returns
    -------
    bool
        True if corpus is ready, False otherwise
    """
    try:
        manager = get_corpus_data_manager(user_session_id, corpus_type)
        return manager.is_ready()
    except Exception as e:
        enhanced_error_handler.handle_system_error(
            e, f"Corpus readiness check for {corpus_type}"
        )
        return False
