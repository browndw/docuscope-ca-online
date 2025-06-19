"""
Text and document processing utilities for the corpus analysis application.
"""

import streamlit as st


def update_tags(html_state: str, session_id: str) -> None:
    """
    Update the HTML style string for tag highlights in the session state.

    Parameters
    ----------
    html_state : str
        The HTML string representing the current tag highlights.
    session_id : str
        The session ID for which the tag highlights are to be updated.

    Returns
    -------
    None
    """
    _TAGS = f"tags_{session_id}"
    html_highlights = [
        ' { background-color:#5fb7ca; }',
        ' { background-color:#e35be5; }',
        ' { background-color:#ffc701; }',
        ' { background-color:#fe5b05; }',
        ' { background-color:#cb7d60; }'
        ]
    if 'html_str' not in st.session_state[session_id]:
        st.session_state[session_id]['html_str'] = ''
    if _TAGS in st.session_state:
        tags = st.session_state[_TAGS]
        if len(tags) > 5:
            tags = tags[:5]
            st.session_state[_TAGS] = tags
    else:
        tags = []
    tags = ['.' + x for x in tags]
    highlights = html_highlights[:len(tags)]
    style_str = [''.join(x) for x in zip(tags, highlights)]
    style_str = ''.join(style_str)
    style_sheet_str = '<style>' + style_str + '</style>'
    st.session_state[session_id]['html_str'] = style_sheet_str + html_state
