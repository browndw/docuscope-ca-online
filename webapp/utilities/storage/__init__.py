"""
Storage and caching utilities for persistent data management.

This package provides caching, persistence, and cloud storage functionality
for the corpus analysis application.
"""

from webapp.utilities.storage.cache_management import (
    persistent_hash,
    get_query_count,
    add_message,
    add_plot,
    add_login
)

# Import async storage functions for non-blocking operations
from webapp.utilities.storage.async_storage import (
    conditional_async_add_message,
    conditional_async_add_plot,
    get_storage_manager
)

__all__ = [
    'persistent_hash',
    'get_query_count',
    'add_message',
    'add_plot',
    'add_login',
    # Async storage functions
    'conditional_async_add_message',
    'conditional_async_add_plot',
    'get_storage_manager'
]
