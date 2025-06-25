"""
User interface utilities for Streamlit components.

This package provides UI components, form controls, data display utilities,
and text visualization functions.
"""

from webapp.utilities.ui.corpus_display import (
    target_info,
    reference_info,
    collocation_info,
    correlation_info,
    variance_info,
    contribution_info,
    group_info,
    target_parts,
    reference_parts,
    load_and_display_target_corpus,
    render_corpus_info_expanders
)
from webapp.utilities.ui.data_tables import (
    get_streamlit_column_config,
    render_data_table_interface,
    render_dataframe,
    render_excel_download_option
)
from webapp.utilities.ui.error_boundaries import (
    UIErrorBoundary,
    with_fallback,
    safe_asset_loader,
    SafeComponentRenderer,
    graceful_component,
    safe_session_operation,
    AssetManager,
    asset_manager
)
from webapp.utilities.ui.shared_utils import (
    add_category_description
)
from webapp.utilities.ui.text_visualization import (
    generate_tag_html_legend
)
from webapp.utilities.ui.form_controls import (
    tagset_selection,
    tag_filter_multiselect,
    multi_tag_filter_multiselect,
    keyness_sort_controls,
    keyness_settings_info,
    color_picker_controls
)
from webapp.utilities.ui.helpers import (
    get_page_base_filename,
    sidebar_help_link,
    sidebar_action_button,
    render_table_generation_interface,
    toggle_download
)
from webapp.utilities.ui.sidebar import (
    sidebar_keyness_options,
    plot_action_button,
    show_plot_warning
)
from webapp.utilities.ui.text_display import (
    update_tags,
    render_document_selection_interface,
    render_document_interface
)
from webapp.utilities.ui.download_interface import (
    render_download_page_header,
    render_data_loading_interface,
    render_corpus_selection,
    render_data_type_selection,
    render_format_selection,
    render_tagset_selection,
    render_download_button,
    check_reference_corpus_availability,
    get_corpus_data
)

__all__ = [
    'target_info',
    'reference_info',
    'collocation_info',
    'correlation_info',
    'variance_info',
    'contribution_info',
    'group_info',
    'target_parts',
    'reference_parts',
    'load_and_display_target_corpus',
    'render_corpus_info_expanders',
    'get_streamlit_column_config',
    'add_category_description',
    'render_data_table_interface',
    'render_dataframe',
    'render_excel_download_option',
    'generate_tag_html_legend',
    'tagset_selection',
    'tag_filter_multiselect',
    'multi_tag_filter_multiselect',
    'keyness_sort_controls',
    'keyness_settings_info',
    'color_picker_controls',
    'get_page_base_filename',
    'sidebar_help_link',
    'sidebar_action_button',
    'render_table_generation_interface',
    'toggle_download',
    'sidebar_keyness_options',
    'plot_action_button',
    'show_plot_warning',
    'update_tags',
    'render_document_selection_interface',
    'render_document_interface',
    'render_download_page_header',
    'render_data_loading_interface',
    'render_corpus_selection',
    'render_data_type_selection',
    'render_format_selection',
    'render_tagset_selection',
    'render_download_button',
    'check_reference_corpus_availability',
    'get_corpus_data',
    'UIErrorBoundary',
    'with_fallback',
    'safe_asset_loader',
    'SafeComponentRenderer',
    'graceful_component',
    'safe_session_operation',
    'AssetManager',
    'asset_manager'
]
