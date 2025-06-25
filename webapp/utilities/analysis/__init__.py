"""
Analysis utilities for corpus analysis and statistical computations.

This package provides functions for statistical analysis, corpus generation,
comparative analysis between corpora, data loading, and validation.
"""

from webapp.utilities.analysis.statistical_analysis import (
    generate_frequency_table,
    generate_tags_table,
    generate_keyness_tables,
    generate_keyness_parts,
    freq_simplify_pl,
    correlation_update,
    update_pca_plot
)
from webapp.utilities.analysis.corpus_generators import (
    generate_ngrams,
    generate_clusters,
    generate_kwic,
    generate_collocations
)
from webapp.utilities.analysis.corpus_loading import (
    load_detector
)
# Import metadata functions from their centralized location
from webapp.utilities.session import (
    load_metadata
)
# Import these from the processing module
from webapp.utilities.processing import (
    find_saved,
    find_saved_reference,
    load_corpus_internal,
    load_corpus_new
)
# Import advanced plotting functions from plotting module
from webapp.utilities.plotting import (
    generate_pca,
    generate_scatterplot,
    generate_scatterplot_with_groups
)
from webapp.utilities.analysis.data_validation import (
    check_language,
    check_schema,
    check_corpus_new,
    validate_dataframe_content,
    normalize_text,
    has_target_corpus,
    has_reference_corpus,
    has_metadata,
    safe_get_categories,
    render_corpus_not_loaded_error,
    render_metadata_not_processed_error,
    is_valid_df
)
from webapp.utilities.analysis.grouped_analysis import (
    dtm_simplify_grouped,
    tags_table_grouped
)

__all__ = [
    'generate_frequency_table',
    'generate_tags_table',
    'generate_keyness_tables',
    'generate_keyness_parts',
    'freq_simplify_pl',
    'correlation_update',
    'update_pca_plot',
    'generate_ngrams',
    'generate_clusters',
    'generate_kwic',
    'generate_collocations',
    'load_detector',
    'load_corpus_internal',
    'load_corpus_new',
    'find_saved',
    'find_saved_reference',
    'check_language',
    'check_schema',
    'check_corpus_new',
    'validate_dataframe_content',
    'normalize_text',
    'has_target_corpus',
    'has_reference_corpus',
    'has_metadata',
    'safe_get_categories',
    'render_corpus_not_loaded_error',
    'render_metadata_not_processed_error',
    'load_metadata',
    'is_valid_df',
    'dtm_simplify_grouped',
    'tags_table_grouped',
    'generate_pca',
    'generate_scatterplot',
    'generate_scatterplot_with_groups'
]
