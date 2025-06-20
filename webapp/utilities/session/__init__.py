"""
Session management utilities for corpus analysis application.

This package provides functions for managing session state, metadata,
and user session persistence.
"""

# Import core session functions 
from webapp.utilities.session.session_core import (
    get_corpus_categories,
    init_metadata_target,
    init_metadata_reference
)
# Import session management functions from session_management.py (matches legacy)
from webapp.utilities.session.session_management import (
    init_session,
    get_or_init_user_session,
    validate_session_state,
    update_session,
    init_ai_assist,
    generate_temp
)
from webapp.utilities.session.metadata_handlers import (
    load_metadata,
    update_metadata,
    handle_target_metadata_processing,
    MIN_CATEGORIES,
    MAX_CATEGORIES
)

__all__ = [
    'init_session',
    'init_ai_assist',
    'update_session',
    'get_or_init_user_session',
    'get_corpus_categories',
    'validate_session_state',
    'generate_temp',
    'init_metadata_target',
    'init_metadata_reference',
    'load_metadata',
    'update_metadata',
    'handle_target_metadata_processing',
    'MIN_CATEGORIES',
    'MAX_CATEGORIES'
]
