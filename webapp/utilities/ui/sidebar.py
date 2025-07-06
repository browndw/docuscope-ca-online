"""
Sidebar UI components for the corpus analysis application.
"""

import streamlit as st
from typing import Tuple, Optional, Callable

# Import widget key manager for centralized widget management
from webapp.utilities.state.widget_key_manager import (
    register_persistent_widgets,
)

# Register persistent sidebar widgets
PERSISTENT_SIDEBAR_WIDGETS = [
    "pval_threshold",  # p-value threshold selection
    "swap_target",     # Target/reference swap toggle
]

# Register the persistent widgets
register_persistent_widgets(PERSISTENT_SIDEBAR_WIDGETS)


def sidebar_keyness_options(
    user_session_id: str,
    load_metadata_func: Callable,
    token_limit: int = 1_500_000,
    sidebar=None,
    target_key: str = 'target',
    reference_key: str = 'reference',
    require_reference: bool = True
) -> Tuple[float, bool]:
    """
    Render sidebar widgets for p-value threshold and swap target/reference.

    Args:
        user_session_id: Session ID for the current user
        load_metadata_func: Function to load metadata for corpus
        token_limit: Maximum tokens before restricting p-value options
        sidebar: Streamlit sidebar object (defaults to st.sidebar)
        target_key: Key for target corpus
        reference_key: Key for reference corpus
        require_reference: Whether reference corpus is required

    Returns:
        Tuple of (selected p-value, swap state)
    """
    if sidebar is None:
        sidebar = st.sidebar

    # Set defaults if not in session state
    if "pval_threshold" not in st.session_state[user_session_id]:
        st.session_state[user_session_id]["pval_threshold"] = 0.01
    if "swap_target" not in st.session_state[user_session_id]:
        st.session_state[user_session_id]["swap_target"] = False

    # Load metadata for size check
    try:
        metadata_target = load_metadata_func(target_key, user_session_id)
        target_tokens = metadata_target.get('tokens_pos', [0])[0] if metadata_target else 0
    except Exception:
        target_tokens = 0

    reference_tokens = 0
    if require_reference:
        try:
            metadata_reference = load_metadata_func(reference_key, user_session_id)
            reference_tokens = (
                metadata_reference.get('tokens_pos', [0])[0]
                if metadata_reference else 0
            )
        except Exception:
            reference_tokens = 0

    # Logic for p-value options
    if require_reference:
        if target_tokens > token_limit or reference_tokens > token_limit:
            pval_options = [0.05, 0.01]
            sidebar.warning(
                "Corpora are large (>1.5 million tokens). "
                "p < .001 is disabled to prevent memory issues."
            )
        elif target_tokens == 0 or reference_tokens == 0:
            pval_options = []
        else:
            pval_options = [0.05, 0.01, 0.001]
    else:
        if target_tokens > token_limit:
            pval_options = [0.01, 0.001]
            sidebar.warning(
                "Corpus is large (>1.5 million tokens). "
                "p < .05 is disabled to prevent memory issues."
            )
        elif target_tokens == 0:
            pval_options = []
        else:
            pval_options = [0.05, 0.01, 0.001]

    # Select p-value threshold
    pval_idx = (
        pval_options.index(st.session_state[user_session_id]["pval_threshold"])
        if st.session_state[user_session_id]["pval_threshold"] in pval_options
        else 1
    )

    sidebar.markdown("### Select threshold")
    pval_selected = sidebar.selectbox(
        "p-value threshold",
        options=pval_options,
        format_func=lambda x: f"{x:.3f}" if x < 0.01 else f"{x:.2f}",
        index=pval_idx,
        key=f"pval_threshold_{user_session_id}",
        help=(
            "Select the p-value threshold for keyness analysis. "
            "Lower values are more stringent, but may be useful for larger corpora. "
            "For smaller corpora, a threshold of 0.05 is often sufficient."
        ),
    )
    sidebar.markdown("---")

    st.session_state[user_session_id]["pval_threshold"] = pval_selected

    swap_selected = False
    if require_reference and target_tokens > 0 and reference_tokens > 0:
        sidebar.markdown("### Swap target/reference corpora")
        swap_selected = sidebar.toggle(
            "Swap Target/Reference",
            value=st.session_state[user_session_id]["swap_target"],
            key=f"swap_target_{user_session_id}",
            help=(
                "If selected, the target corpus will be used as the reference "
                "and the reference corpus will be used as the target for keyness analysis. "
                "This will show what is more frequent in the reference corpus "
                "compared to the target corpus. "
            ),
        )
        sidebar.markdown("---")
        st.session_state[user_session_id]["swap_target"] = swap_selected

    return pval_selected, swap_selected


def plot_action_button(
    label: str,
    key: str,
    help_text: str,
    user_session_id: Optional[str] = None,
    attempted_flag: Optional[str] = None,
    clear_func: Optional[Callable] = None
) -> bool:
    """
    Create a sidebar action button for plotting operations.

    Args:
        label: Button label text
        key: Unique key for the button
        help_text: Help text for the button
        user_session_id: Session ID for the current user
        attempted_flag: Flag to set when button is pressed
        clear_func: Function to call to clear previous results

    Returns:
        True if button was pressed, False otherwise
    """
    pressed = st.sidebar.button(
        label=label,
        key=key,
        help=help_text,
        type="secondary",
        use_container_width=False,
        icon=":material/manufacturing:"
    )
    if pressed:
        if clear_func and user_session_id:
            clear_func(user_session_id)
        if user_session_id and attempted_flag:
            st.session_state[user_session_id][attempted_flag] = True
    return pressed


def show_plot_warning(
    session: dict,
    user_session_id: str,
    warning_key: str,
    attempted_flag: str,
    df_keys: Optional[list] = None
) -> bool:
    """
    Display a plot warning and clear associated data if warning exists.

    Args:
        session: Session state dictionary
        user_session_id: Session ID for the current user
        warning_key: Key for the warning message in session state
        attempted_flag: Flag indicating if plot was attempted
        df_keys: List of DataFrame keys to clear if warning exists

    Returns:
        True if warning was shown, False otherwise
    """
    warning_exists = (
        session[user_session_id].get(warning_key) and
        session[user_session_id].get(attempted_flag)
    )
    if warning_exists:
        if df_keys:
            for k in df_keys:
                if isinstance(k, tuple):
                    # Nested key, e.g. ("target", "pca_df")
                    d = session[user_session_id]
                    for subkey in k[:-1]:
                        d = d.get(subkey, {})
                    d[k[-1]] = None
                else:
                    session[user_session_id][k] = None
        msg, icon = session[user_session_id][warning_key]
        st.warning(msg, icon=icon)
        return True
    return False
