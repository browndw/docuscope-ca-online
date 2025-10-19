"""
Data utilities for document processing and basic data manipulation.

This package provides utility functions for working with documents, IDs,
and basic data structures without complex dependencies.
"""

from webapp.utilities.common.document_utils import get_doc_cats, safe_metadata_get

__all__ = [
    'get_doc_cats',
    'safe_metadata_get'
]
