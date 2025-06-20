"""
Document utility functions for basic document ID processing.

This module provides simple utility functions for working with document IDs
and categories without dependencies on session or complex analysis modules.
"""


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
