"""
AI and LLM integration utilities for corpus analysis.

This package provides AI-powered features including language model integration,
safe code execution, and AI-assisted plotting functionality.

The module is organized as follows:
- llm_core: Shared utilities for all AI assistants
- plotbot: Iterative plotting assistant
- pandabot: PandasAI-based analysis and plotting assistant
- code_execution: Safe code execution utilities
"""

# Import shared utilities
from webapp.utilities.ai.shared import (
    LLM_MODEL,
    LLM_PARAMS,
    detect_intent,
    prune_message_thread,
    fig_to_svg,
    validate_api_key,
    get_quota_info,
    increment_session_quota,
    render_quota_tracker,
    should_show_api_key_input,
    export_conversation_history,
    get_current_plot_as_svg,
    get_current_plot_as_png,
    render_work_preservation_interface,
    should_show_work_preservation_interface
)

# Import enterprise circuit breaker and routing
from webapp.utilities.ai.enterprise_circuit_breaker import (
    get_circuit_breaker,
    get_circuit_breaker_manager,
    circuit_breaker_protected_call,
    CircuitBreakerError,
    CircuitBreakerState
)
from webapp.utilities.ai.enterprise_router import (
    get_ai_router,
    route_ai_request
)
from webapp.utilities.ai.enterprise_integration import (
    determine_api_key_type,
    protected_openai_call,
    with_circuit_breaker_protection,
    get_circuit_breaker_status,
    reset_circuit_breakers
)

# Import core LLM utilities (non-circular)
from webapp.utilities.ai.llm_core import (
    print_settings,
    is_openai_key_valid,
    tables_to_list,
    table_from_list,
    previous_code_chunk,
    setup_ai_session_state,
    get_api_key,
    render_api_key_input,
    render_data_selection_interface,
    render_data_preview_controls
)

# Import plotbot-specific functions
from webapp.utilities.ai.plotbot import (
    clear_plotbot,
    clear_plotbot_table,
    make_plotbot_cache_key,
    plotbot_code_generate_or_update,
    plotbot_code_execute,
    generate_plotbot_code_and_plot,
    plotbot_user_query
)

# Import pandabot-specific functions
from webapp.utilities.ai.pandabot import (
    clear_pandasai,
    clear_pandasai_table,
    pandabot_user_query
)

# Import code execution utilities
from webapp.utilities.ai.code_execution import (
    is_code_safe,
    strip_imports
)

__all__ = [
    # Shared utilities
    'print_settings',
    'is_openai_key_valid',
    'detect_intent',
    'tables_to_list',
    'table_from_list',
    'prune_message_thread',
    'previous_code_chunk',
    'fig_to_svg',
    'validate_api_key',
    'LLM_MODEL',
    'LLM_PARAMS',

    # AI session management utilities
    'setup_ai_session_state',
    'get_api_key',
    'render_api_key_input',
    'render_data_selection_interface',
    'render_data_preview_controls',

    # Quota tracking utilities
    'get_quota_info',
    'increment_session_quota',
    'render_quota_tracker',
    'should_show_api_key_input',
    'export_conversation_history',
    'get_current_plot_as_svg',
    'get_current_plot_as_png',
    'render_work_preservation_interface',
    'should_show_work_preservation_interface',

    # Enterprise circuit breaker and routing
    'get_circuit_breaker',
    'get_circuit_breaker_manager',
    'circuit_breaker_protected_call',
    'CircuitBreakerError',
    'CircuitBreakerState',
    'get_ai_router',
    'route_ai_request',

    # Enterprise integration helpers
    'determine_api_key_type',
    'protected_openai_call',
    'with_circuit_breaker_protection',
    'get_circuit_breaker_status',
    'reset_circuit_breakers',

    # Plotbot functions
    'clear_plotbot',
    'clear_plotbot_table',
    'plotbot_user_query',
    'make_plotbot_cache_key',
    'plotbot_code_generate_or_update',
    'plotbot_code_execute',
    'generate_plotbot_code_and_plot',

    # Pandabot functions
    'clear_pandasai',
    'clear_pandasai_table',
    'pandabot_user_query',

    # Code execution utilities
    'is_code_safe',
    'strip_imports'
]
