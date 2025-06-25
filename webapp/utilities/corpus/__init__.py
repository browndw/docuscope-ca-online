"""
Corpus management utilities for memory-efficient data handling.

This package provides centralized corpus data management with lazy loading,
intelligent caching, and backward compatibility with existing session patterns.
"""

from webapp.utilities.corpus.data_manager import (
    CorpusDataManager,
    get_corpus_manager
)
from webapp.utilities.corpus.compatibility import (
    get_corpus_data,
    set_corpus_data,
    has_corpus_data,
    corpus_is_ready,
    get_available_corpus_keys,
    migrate_legacy_session_data,
    clear_corpus_data,
    get_target_data,
    get_reference_data,
    set_target_data,
    set_reference_data,
    get_corpus_data_manager
)

__all__ = [
    'CorpusDataManager',
    'get_corpus_manager',
    'get_corpus_data_manager',
    'get_corpus_data',
    'set_corpus_data',
    'has_corpus_data',
    'corpus_is_ready',
    'get_available_corpus_keys',
    'migrate_legacy_session_data',
    'clear_corpus_data',
    'get_target_data',
    'get_reference_data',
    'set_target_data',
    'set_reference_data'
]
