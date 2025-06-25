"""
Text visualization and tagging utilities.

This module provides functions for generating HTML legends and tag density plots
for text analysis visualization.
"""


def generate_tag_html_legend(tag_list: list, tag_colors: list) -> str:
    """
    Generate HTML legend string for tag highlighting.

    Args:
        tag_list: List of tag names to display in legend
        tag_colors: List of hex color codes corresponding to tags

    Returns:
        HTML string with colored spans for tag legend
    """
    if not tag_list or not tag_colors:
        return ""

    # Ensure we don't have more tags than colors
    tag_colors = tag_colors[:len(tag_list)]

    # Create HTML spans for each tag
    tag_html = []
    for color, tag in zip(tag_colors, tag_list):
        tag_html.append(f'<span style="background-color: {color}">{tag}</span>')

    return '; '.join(tag_html)
