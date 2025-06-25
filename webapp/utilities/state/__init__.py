"""
State management utilities for widget and application state.

This package provides utilities for managing Streamlit widget states,
form states, application state persistence, and session state keys.
"""

from webapp.utilities.state.widget_state import (
    get_widget_state,
    set_widget_state,
    clear_widget_state,
    reset_form_state,
    preserve_widget_state,
    get_form_state,
    validate_required_fields,
    WidgetStateManager,
    safe_clear_widget_state,
    safe_clear_widget_states
)

from webapp.utilities.state.widget_key_manager import (
    widget_key_manager,
)

from webapp.utilities.state.session_keys import (
    SessionKeys,
    MetadataKeys,
    CorpusKeys,
    TargetKeys,
    ReferenceKeys,
    WarningKeys,
    LoadCorpusKeys,
    BoxplotKeys,
    ScatterplotKeys,
    PCAKeys
)

__all__ = [
    # Widget state utilities
    'get_widget_state',
    'set_widget_state',
    'clear_widget_state',
    'reset_form_state',
    'preserve_widget_state',
    'get_form_state',
    'validate_required_fields',
    'WidgetStateManager',
    'safe_clear_widget_state',
    'safe_clear_widget_states',
    # Widget key manager
    'widget_key_manager',
    # Session state keys
    'SessionKeys',
    'MetadataKeys',
    'CorpusKeys',
    'TargetKeys',
    'ReferenceKeys',
    'WarningKeys',
    'LoadCorpusKeys',
    'BoxplotKeys',
    'ScatterplotKeys',
    'PCAKeys'
]
