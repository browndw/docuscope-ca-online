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

__all__ = [
    'persistent_hash',
    'get_query_count',
    'add_message',
    'add_plot',
    'add_login'
]
