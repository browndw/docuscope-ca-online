"""
This app allows users to compare two corpora by generating keyness tables
for tokens and tags.

It provides functionality to:
- Generate keyness tables for tokens and tags
- Filter tokens and tags by tagset and tag type
- Sort keyness tables by keyness or effect size
- Download keyness tables in Excel format
- Visualize keyness tables with bar plots
- Reset keyness tables and start over
"""
import polars as pl
import streamlit as st

# Core application utilities with standardized patterns
from webapp.utilities.core import app_core

from webapp.utilities.session import (
    get_or_init_user_session, load_metadata,
    safe_session_get
    )
from webapp.utilities.corpus import (
    get_corpus_data_manager, clear_corpus_data
    )
from webapp.utilities.ui import (
    keyness_sort_controls, keyness_settings_info,
    reference_info, render_dataframe,
    sidebar_help_link, tag_filter_multiselect,
    tagset_selection, target_info,
    toggle_download, sidebar_keyness_options,
    color_picker_controls
)
from webapp.utilities.state import (
    CorpusKeys, SessionKeys,
    TargetKeys, WarningKeys
)
from webapp.utilities.state.widget_key_manager import create_persist_function
from webapp.utilities.analysis import (
    has_target_corpus, has_reference_corpus,
    render_corpus_not_loaded_error, generate_keyness_tables,
    freq_simplify_pl, attach_keyness_artifact
)
from webapp.persistence import registry_service
from webapp.queue import enqueue_keyness_preparation, get_queue, get_redis_queue_config
from webapp.utilities.plotting import (
    plot_compare_corpus_bar, plot_download_link
)
from webapp.utilities.ui.error_boundaries import SafeComponentRenderer
from webapp.menu import (
    menu, require_login
    )


TITLE = "Compare Corpora"
ICON = ":material/compare_arrows:"

# Register persistent widgets for this page
COMPARE_CORPORA_PERSISTENT_WIDGETS = [
    "kt_radio1",  # Radio button for keyness table tagset selection
]
app_core.register_page_widgets(COMPARE_CORPORA_PERSISTENT_WIDGETS)

TOKEN_LIMIT = 1_500_000
KEYNESS_QUEUE_STATE_KEY = "keyness_queue_state"

# Configuration constants
KEYNESS_TOKEN_TAGSET_CONFIG = {
    "Parts-of-Speech": {"General": TargetKeys.KW_POS, "Specific": TargetKeys.KW_POS},
    "DocuScope": TargetKeys.KW_DS
}
KEYNESS_TAG_TAGSET_CONFIG = {
    "Parts-of-Speech": TargetKeys.KT_POS,
    "DocuScope": TargetKeys.KT_DS
}
KEYNESS_SIMPLIFY_CONFIG = {
    "Parts-of-Speech": {"General": freq_simplify_pl, "Specific": None}
}
KEYNESS_TAG_FILTERS_CONFIG = {
    "Parts-of-Speech": lambda df: df.filter(pl.col("Tag") != "FU"),
    "DocuScope": lambda df: df.filter(pl.col("Tag") != "Untagged")
}

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )


def _get_keyness_queue_state(user_session_id: str) -> dict | None:
    """Return queued keyness state for the current session, if any."""

    queue_state = st.session_state.get(KEYNESS_QUEUE_STATE_KEY)
    if not isinstance(queue_state, dict):
        return None
    if queue_state.get("session_id") != user_session_id:
        return None
    return queue_state


def _clear_keyness_queue_state(user_session_id: str) -> None:
    """Clear queued keyness state for the current session."""

    if _get_keyness_queue_state(user_session_id) is not None:
        st.session_state.pop(KEYNESS_QUEUE_STATE_KEY, None)


def _can_queue_keyness(session: dict) -> tuple[str, str] | None:
    """Return built-in target/reference paths when keyness can be queued."""

    if not get_redis_queue_config().enabled:
        return None

    target_source = safe_session_get(session, SessionKeys.TARGET_DB, "")
    reference_source = safe_session_get(session, SessionKeys.REFERENCE_DB, "")
    if not target_source or not reference_source:
        return None
    return str(target_source), str(reference_source)


def _submit_keyness_job(
    user_session_id: str,
    target_source: str,
    reference_source: str,
    threshold: float,
    swap_target: bool,
) -> None:
    """Submit built-in keyness generation through Redis/RQ."""

    result = enqueue_keyness_preparation(
        target_source=target_source,
        reference_source=reference_source,
        threshold=threshold,
        swap_target=swap_target,
    )
    if result.state == "ready" and result.artifact_id is not None:
        _clear_keyness_queue_state(user_session_id)
        if attach_keyness_artifact(user_session_id, result.artifact_id):
            st.rerun()
        st.error("Ready keyness artifact could not be attached. Please retry.")
        return

    st.session_state[KEYNESS_QUEUE_STATE_KEY] = {
        "session_id": user_session_id,
        "target_source": target_source,
        "reference_source": reference_source,
        "threshold": threshold,
        "swap_target": swap_target,
        "control_plane_job_id": result.control_plane_job_id,
        "rq_job_id": result.rq_job_id,
        "artifact_id": result.artifact_id,
    }
    st.rerun()


@st.fragment(run_every="2s")
def _render_keyness_queue_status(user_session_id: str) -> None:
    """Poll queued keyness status and attach it once the job completes."""

    queue_state = _get_keyness_queue_state(user_session_id)
    if queue_state is None:
        return

    control_plane_job_id = queue_state.get("control_plane_job_id")
    rq_job_id = queue_state.get("rq_job_id")
    if not isinstance(rq_job_id, str) or not rq_job_id:
        if isinstance(control_plane_job_id, int):
            rq_job_id = f"keyness-{control_plane_job_id}"
            queue_state["rq_job_id"] = rq_job_id
            st.session_state[KEYNESS_QUEUE_STATE_KEY] = queue_state
        else:
            rq_job_id = None

    if control_plane_job_id is None:
        _clear_keyness_queue_state(user_session_id)
        st.warning("Queued keyness job state was incomplete. Please try again.")
        return

    job_row = registry_service.get_job_by_id(control_plane_job_id)
    if job_row is None:
        _clear_keyness_queue_state(user_session_id)
        st.warning("Queued keyness job could not be found. Please try again.")
        return

    if job_row.status == "pending":
        if not isinstance(rq_job_id, str) or not rq_job_id:
            registry_service.mark_job_failed(
                control_plane_job_id,
                "Queued keyness job was pending without a queue job id.",
            )
            _clear_keyness_queue_state(user_session_id)
            st.error("Queued keyness generation failed before execution. Please retry.")
            return

        try:
            rq_job = get_queue().fetch_job(rq_job_id)
        except Exception:
            rq_job = None

        if rq_job is None:
            registry_service.mark_job_failed(
                control_plane_job_id,
                "Queue job record was missing while control-plane status was pending.",
            )
            _clear_keyness_queue_state(user_session_id)
            st.error("Queued keyness generation failed before execution. Please retry.")
            return

        rq_status_raw = rq_job.get_status(refresh=True)
        rq_status = str(getattr(rq_status_raw, "value", rq_status_raw)).strip().lower()
        if rq_status == "failed":
            failure_reason = "Queued keyness job failed before completion."
            if isinstance(rq_job.exc_info, str) and rq_job.exc_info.strip():
                failure_reason = rq_job.exc_info.strip().splitlines()[-1]
            registry_service.mark_job_failed(control_plane_job_id, failure_reason)
            _clear_keyness_queue_state(user_session_id)
            st.error(f"Queued keyness generation failed: {failure_reason}")
            return

    if job_row.status == "completed":
        _clear_keyness_queue_state(user_session_id)
        if job_row.artifact_id is not None and attach_keyness_artifact(
            user_session_id,
            job_row.artifact_id,
        ):
            st.rerun()
        st.error("Queued keyness artifact could not be attached. Please retry.")
        return

    if job_row.status == "failed":
        _clear_keyness_queue_state(user_session_id)
        st.error(f"Queued keyness generation failed: {job_row.failure_reason}")
        return

    st.info("Generating keyness tables in the background...")


def render_corpus_info_headers(
        user_session_id: str
) -> None:
    """Render target and reference corpus information headers."""
    # Load target and reference metadata
    metadata_target = load_metadata(CorpusKeys.TARGET, user_session_id)
    metadata_reference = load_metadata(CorpusKeys.REFERENCE, user_session_id)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.info(target_info(metadata_target))
    with col2:
        st.info(reference_info(metadata_reference))


def render_keyness_settings_info(user_session_id: str) -> None:
    """Display current keyness table settings using utility function."""
    st.info(keyness_settings_info(user_session_id))


def render_tokens_keyness_interface(
        user_session_id: str
) -> None:
    """Render the tokens keyness table interface."""
    df, tag_options, tag_radio, tag_type = tagset_selection(
        user_session_id=user_session_id,
        session_state=st.session_state,
        persist_func=create_persist_function(user_session_id),
        tagset_keys=KEYNESS_TOKEN_TAGSET_CONFIG,
        simplify_funcs=KEYNESS_SIMPLIFY_CONFIG,
        tag_filters={
            # Add filters here if needed
        },
        tag_radio_key="kt_radio2",
        tag_type_key="kt_type_radio2"
    )

    sort_by, reverse = keyness_sort_controls(
        sort_options=["Keyness (LL)", "Effect Size (LR)"],
        default="Keyness (LL)",
        reverse_default=True,
        key_prefix="kt_"  # or something unique per page/tab
    )

    df = tag_filter_multiselect(df, user_session_id=user_session_id)

    # Map UI label to actual DataFrame column
    sort_col_map = {
        "Keyness (LL)": "LL",
        "Effect Size (LR)": "LR"
    }
    sort_col = sort_col_map[sort_by]

    if df is not None and getattr(df, "height", 0) > 0:
        df = df.sort(sort_col, descending=reverse)
    render_dataframe(df)

    st.sidebar.markdown("---")
    # Add download button for the DataFrame
    toggle_download(
        label="Excel",
        convert_args=(df.to_pandas(),) if (df is not None and getattr(df, "height", 0) > 0) else (None,),  # noqa: E501
        file_name="keywords_tokens.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        location=st.sidebar
    )


def render_tags_keyness_interface(
        user_session_id: str
) -> None:
    """Render the tags keyness table interface with tabs."""
    df, tag_options, tag_radio, tag_type = tagset_selection(
        user_session_id=user_session_id,
        session_state=st.session_state,
        persist_func=create_persist_function(user_session_id),
        tagset_keys=KEYNESS_TAG_TAGSET_CONFIG,
        tag_filters=KEYNESS_TAG_FILTERS_CONFIG,
        tag_radio_key="kt_radio3"
    )

    # Tabs for displaying keyness table and plot
    tab1, tab2 = st.tabs(["Keyness Table", "Keyness Plot"])

    with tab1:
        # Sorting and filtering controls
        sort_by, reverse = keyness_sort_controls(
            sort_options=["Keyness (LL)", "Effect Size (LR)"],
            default="Keyness (LL)",
            reverse_default=True,
            key_prefix="kt_"
        )

        df = tag_filter_multiselect(df, user_session_id=user_session_id)

        # Map UI label to actual DataFrame column
        sort_col_map = {
            "Keyness (LL)": "LL",
            "Effect Size (LR)": "LR"
        }
        sort_col = sort_col_map[sort_by]

        if df is not None and getattr(df, "height", 0) > 0:
            df = df.sort(sort_col, descending=reverse)

        render_dataframe(df)

    with tab2:
        if df is not None and getattr(df, "height", 0) > 0:
            # Color picker for bar colors
            color_dict = color_picker_controls(
                ["Target Color", "Reference Corpus"],
                key_prefix="compare_corpus_bar_",
                default_hex="#133955"
            )

            # Plot with color customization
            fig = plot_compare_corpus_bar(df, color_dict=color_dict)
            SafeComponentRenderer.safe_plotly_chart(fig, width="stretch")
            plot_download_link(fig, filename="compare_corpus_bar.png")

    st.sidebar.markdown("---")
    # Add download button for the DataFrame
    toggle_download(
        label="Excel",
        convert_args=(df.to_pandas(),) if (df is not None and getattr(df, "height", 0) > 0) else (None,),  # noqa: E501
        file_name="keyness_corpora",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        location=st.sidebar
    )


def render_keyness_reset_controls(
        user_session_id: str
) -> None:
    """Render controls for resetting keyness table."""
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        body=(
            "### Generate new table\n\n"
            "Use the button to reset the keyness table and start over."
            )
        )
    if st.sidebar.button("Generate New Keyness Table", icon=":material/refresh:"):
        # Clear keyness tables using the corpus data manager
        clear_corpus_data(user_session_id, CorpusKeys.TARGET, [
            TargetKeys.KW_POS, TargetKeys.KW_DS,
            TargetKeys.KT_POS, TargetKeys.KT_DS
        ])

        # Reset keyness_table state using session key
        app_core.session_manager.update_session_state(
            user_session_id,
            SessionKeys.KEYNESS_TABLE,
            False
            )
        # Clear warnings using session key
        st.session_state[user_session_id][WarningKeys.KEYNESS] = None
        st.rerun()


def render_keyness_interface(
        user_session_id: str, session: dict
) -> None:
    """Render the main keyness interface with validation."""
    try:
        # Validate session state using the new manager
        target_manager = get_corpus_data_manager(user_session_id, CorpusKeys.TARGET)
        reference_manager = get_corpus_data_manager(user_session_id, CorpusKeys.REFERENCE)

        if not target_manager.is_ready() or not reference_manager.is_ready():
            st.error("Invalid session state. Please reload the page or reset your data.")
            return

        # Initialize widget state management
        app_core.widget_manager.register_persistent_keys([
            'compare_corp_sort', 'compare_corp_ascending', 'compare_corp_display_limit',
            'compare_corp_metric', 'compare_corp_filter_zero'
        ])

        # Render corpus info headers
        render_corpus_info_headers(user_session_id)

        # Show user selections
        render_keyness_settings_info(user_session_id)

        st.sidebar.markdown("### Comparison")
        table_radio = st.sidebar.radio(
            "Select the keyness table to display:",
            ("Tokens", "Tags Only"),
            key="kt_radio1",
            horizontal=True
        )

        st.sidebar.markdown("---")

        # Route to appropriate interface based on table type
        if table_radio == 'Tokens':
            render_tokens_keyness_interface(user_session_id)
        else:
            render_tags_keyness_interface(user_session_id)

        # Add reset controls
        render_keyness_reset_controls(user_session_id)
        st.sidebar.markdown("---")

    except Exception as e:
        st.error(f"Error loading keyness interface: {str(e)}", icon=":material/error:")
        st.info("Try regenerating the keyness table if this error persists.")


def main():
    """
    Main function to run the Streamlit app for comparing corpora.
    It initializes the user session, loads the necessary data,
    and provides the UI for generating and displaying keyness tables.
    """
    # Set login requirements for navigation
    require_login()
    menu()
    st.markdown(
        body=f"## {TITLE}",
        help=(
            "This app allows you to generate and view keyness tables "
            "for the loaded target and reference corpora. You can filter by tags, "
            "select p-value thresholds, and download the results."
            )
        )

    # Get or initialize user session
    user_session_id, session = get_or_init_user_session()

    sidebar_help_link("compare-corpora.html")

    # Route to appropriate interface based on whether keyness table exists
    if safe_session_get(session, SessionKeys.KEYNESS_TABLE, False):
        render_keyness_interface(user_session_id, session)
    else:
        # Display instructions and options for keyness generation
        st.markdown(
            body=(
                ":material/manufacturing: Use the button in the sidebar to **generate keywords**.\n\n"  # noqa: E501
                ":material/priority: A **target corpus** and a **reference corpus** must be loaded first.\n\n"  # noqa: E501
                ":material/priority: After the table has been generated, "
                "you will be able to **toggle between the tagsets**."
                )
        )

        # Get the sidebar options that will be used for generation
        pval_selected, swap_selected = sidebar_keyness_options(
            user_session_id,
            load_metadata_func=load_metadata
        )

        # Store the options in a lambda for the generation function
        def keyness_generation_action():
            # Check preconditions specific to keyness generation
            if not has_target_corpus(session):
                render_corpus_not_loaded_error("target")
                return

            if not has_reference_corpus(session):
                render_corpus_not_loaded_error("reference")
                return

            queue_sources = _can_queue_keyness(session)
            if queue_sources is not None:
                target_source, reference_source = queue_sources
                _submit_keyness_job(
                    user_session_id,
                    target_source,
                    reference_source,
                    pval_selected,
                    swap_selected,
                )
                return

            # If all validations pass, generate the keyness tables
            generate_keyness_tables(
                user_session_id,
                threshold=pval_selected,
                swap_target=swap_selected
            )

        # Use a custom table generation interface that handles keyness specifics
        st.sidebar.markdown(
            body=(
                "### Generate keyness table\n\n"
                "Use the button to process a table."
            ),
            help=(
                "Keyness tables are generated based on the loaded target and "
                "reference corpora. You can filter the table by tags after it has "
                "been generated. The table will include frequencies and hypothesis "
                "testing for the selected tagsets.\n\n"
                "Click on the **Help** button for more information on how to use this app."
            )
        )

        # Custom action button that handles keyness-specific validation
        if st.sidebar.button(
            "Keyness Table",
            icon=":material/manufacturing:",
            type="secondary"
        ):
            with st.sidebar.status("Generating keywords...", expanded=True):
                keyness_generation_action()

        # Display warnings if there are any
        if st.session_state[user_session_id].get(WarningKeys.KEYNESS):
            msg, icon = st.session_state[user_session_id][WarningKeys.KEYNESS]
            st.error(msg, icon=icon)

        _render_keyness_queue_status(user_session_id)

        st.sidebar.markdown("---")


if __name__ == "__main__":
    main()
