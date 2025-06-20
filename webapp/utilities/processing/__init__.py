"""
Corpus processing utilities.

This module provides functions for processing different types of corpora,
including internal, external, and newly uploaded text or parquet files.
"""

# Import corpus discovery functions
from webapp.utilities.processing.corpus_discovery import (
    find_saved,
    find_saved_reference
)
from webapp.utilities.processing.document_processing import (
    generate_document_html
)
from webapp.utilities.processing.corpus_loading import (
    load_corpus_internal,
    load_corpus_new
)
from webapp.utilities.processing.corpus_processing import (
    process_external,
    process_internal,
    process_new,
    handle_uploaded_parquet,
    handle_uploaded_text,
    sidebar_process_section,
    finalize_corpus_load
)

# Import original functions from process.py and delegate to them
# Legacy functions that need to be migrated or removed
# TODO: Migrate these functions from the legacy process module


# Legacy functions temporarily disabled until migration is complete
# These functions were previously imported from the legacy process module

__all__ = [
    'find_saved',
    'find_saved_reference',
    'generate_document_html',
    'load_corpus_internal',
    'load_corpus_new',
    'process_external',
    'process_internal',
    'process_new',
    'handle_uploaded_parquet',
    'handle_uploaded_text',
    'sidebar_process_section',
    'finalize_corpus_load'
]
