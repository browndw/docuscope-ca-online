"""
Document utility functions for basic document ID processing.

This module provides simple utility functions for working with document IDs
and categories without dependencies on session or complex analysis modules.
"""


# Utility function to safely access metadata values in both formats
def safe_metadata_get(metadata: dict, key: str, default=None, nested_key: str = None):
    """
    Safely get a value from metadata dict, handling both list and scalar formats.

    Metadata can be in different formats:
    - DataFrame converted: {'docids': [{'ids': [...]}], 'doccats': [{'cats': [...]}]}
    - Direct dict: {'docids': {'ids': [...]}, 'doccats': {'cats': [...]}}

    Parameters
    ----------
    metadata : dict
        The metadata dictionary
    key : str
        The primary key to access
    default : any
        Default value if key not found
    nested_key : str, optional
        Optional nested key (e.g., 'ids' for docids, 'cats' for doccats)

    Returns
    -------
    any
        The value from the metadata
    """
    value = metadata.get(key, default)

    if value is None:
        return default

    # If it's a list (from DataFrame conversion) and we want nested access
    if isinstance(value, list) and len(value) > 0 and nested_key:
        if isinstance(value[0], dict):
            return value[0].get(nested_key, default)
        else:
            return value[0] if not nested_key else default

    # If it's a dict and we want nested access
    if isinstance(value, dict) and nested_key:
        return value.get(nested_key, default)

    # If it's a list but no nested key needed, return first element
    if isinstance(value, list) and len(value) > 0 and not nested_key:
        return value[0]

    # Return as-is
    return value


def get_doc_cats(doc_ids: list) -> list:
    """
    Extract document categories from document IDs.

    Parameters
    ----------
    doc_ids : list
        List of document IDs to extract categories from.

    Returns
    -------
    list
        List of document categories.
    """
    doc_cats = []

    for doc_id in doc_ids:
        if isinstance(doc_id, str) and '_' in doc_id:
            # Extract category (everything before first underscore)
            category = doc_id.split('_')[0]
            doc_cats.append(category)
        else:
            # If no underscore or not a string, use the whole ID
            doc_cats.append(str(doc_id))

    return doc_cats
