"""
App for loading and managing corpora in the Corpus Tagger web application.

This module provides functionality for:
- Loading existing corpora from internal databases or user uploads.
- Processing new corpora from text files.
- Uploading external corpora in Parquet format.
- Managing corpus metadata and categories.
- Resetting corpus data and clearing session state.
"""

import os

import spacy
import streamlit as st
from time import perf_counter

# Core application utilities with standardized patterns
from webapp.utilities.core import app_core
from webapp.utilities.configuration.config_manager import config_manager
from webapp.utilities.configuration.logging_config import get_logger

# Module-specific imports
from webapp.utilities.session import (
    get_or_init_user_session, generate_temp, safe_session_get
    )
from webapp.utilities.session.metadata_handlers import (
    handle_target_metadata_processing, load_metadata
    )
from webapp.utilities.analysis import (
    find_saved, find_saved_reference
    )
from webapp.utilities.processing import (
    process_external, process_internal,
    process_new, handle_uploaded_parquet,
    attach_queued_internal_target,
    handle_uploaded_text, sidebar_process_section
    )
from webapp.persistence import registry_service
from webapp.queue import (
    enqueue_internal_target_preparation,
    get_queue,
    get_redis_queue_config,
)
from webapp.utilities.ui import (
    load_and_display_target_corpus, render_corpus_info_expanders
)
from webapp.menu import (
    menu, require_login
    )
from webapp.utilities.state import (
    CorpusPersistencePolicy,
    CorpusKeys, LoadCorpusKeys,
    MetadataKeys, SessionKeys,
    WarningKeys
    )

# Register persistent widgets for this page
LOAD_CORPUS_PERSISTENT_WIDGETS = [
    "reffiles",           # File uploader for reference files
    "corpora_to_load",    # Selection for corpora to load
    "target_persistence_policy",
    "reference_persistence_policy",
]
app_core.register_page_widgets(LOAD_CORPUS_PERSISTENT_WIDGETS)

# Configuration values from centralized config manager
MODEL_LARGE = config_manager.model_large_path
MODEL_SMALL = config_manager.model_small_path

# Global flags and limits from configuration
DESKTOP = config_manager.desktop_mode
CHECK_SIZE = config_manager.check_size
ENABLE_DETECT = config_manager.check_language
MAX_TEXT = config_manager.max_text_size
MAX_POLARS = config_manager.max_polars_size


TITLE = "Manage Corpus Data"
ICON = ":material/database:"
SLOW_LOAD_CORPUS_PAGE_MS = 100
logger = get_logger()
TEST_MODE = config_manager.test_mode
PROCESS_TARGET_PROBE_STAGE_KEY = "load_test_process_target_probe_stage"
PROCESS_TARGET_PROBE_TIMINGS_KEY = "load_test_process_target_probe_stage_timings_ms"
PROCESS_TARGET_PROBE_FLAT_TIMINGS_KEY = "load_test_process_target_probe_stage_timings_flat"
PROCESS_TARGET_PROBE_SUMMARY_KEY = "load_test_process_target_probe_stage_summary"
INTERNAL_TARGET_QUEUE_STATE_KEY = "internal_target_queue_state"

# Define labels and options for the app
CORPUS_SOURCES = ["Internal", "External", "New"]

# Button and form labels
LABEL_PROCESS_TARGET = "Process Target"
LABEL_PROCESS_REFERENCE = "Process Reference"
LABEL_UPLOAD_TARGET = "UPLOAD TARGET"
LABEL_UPLOAD_REFERENCE = "UPLOAD REFERENCE"
LABEL_RESET_CORPUS = "Reset Corpus"

# Model configuration
MODEL_LARGE_LABEL = "Large Dictionary"
MODEL_SMALL_LABEL = "Common Dictionary"
MODEL_OPTIONS = [MODEL_LARGE_LABEL, MODEL_SMALL_LABEL]
PERSISTENCE_POLICY_OPTIONS = {
    "Save on this server for future analysis": (
        CorpusPersistencePolicy.SERVER_SAVED
    ),
    "Temporary session only": (
        CorpusPersistencePolicy.TEMPORARY_SESSION_ONLY
    ),
    "Keep locally only after processing": (
        CorpusPersistencePolicy.LOCAL_EXPORT_ONLY
    ),
}


def _load_test_minimal_load_corpus_enabled() -> bool:
    """Return whether load-test-only minimal manage-corpus rendering is enabled."""

    return os.environ.get("DOCUSCOPE_LOAD_TEST_MINIMAL_LOAD_CORPUS", "").strip() == "1"


def _get_internal_target_queue_state(user_session_id: str) -> dict | None:
    """Return queued built-in target state for the current session, if any."""

    queue_state = st.session_state.get(INTERNAL_TARGET_QUEUE_STATE_KEY)
    if not isinstance(queue_state, dict):
        return None
    if queue_state.get("session_id") != user_session_id:
        return None
    return queue_state


def _clear_internal_target_queue_state(user_session_id: str) -> None:
    """Clear any queued built-in target state for the current session."""

    queue_state = _get_internal_target_queue_state(user_session_id)
    if queue_state is not None:
        st.session_state.pop(INTERNAL_TARGET_QUEUE_STATE_KEY, None)


def _submit_internal_target_job(corp_path: str, user_session_id: str) -> None:
    """Submit the built-in target path to Redis/RQ when that feature is enabled."""

    result = enqueue_internal_target_preparation(corp_path)
    if result.state == "ready":
        _clear_internal_target_queue_state(user_session_id)
        attach_queued_internal_target(
            corp_path,
            user_session_id,
            queue_artifact_id=result.artifact_id,
        )
        return

    st.session_state[INTERNAL_TARGET_QUEUE_STATE_KEY] = {
        "session_id": user_session_id,
        "corp_path": str(corp_path),
        "control_plane_job_id": result.control_plane_job_id,
        "rq_job_id": result.rq_job_id,
        "artifact_id": result.artifact_id,
    }
    st.rerun()


@st.fragment(run_every="2s")
def _render_internal_target_queue_status(user_session_id: str) -> None:
    """Poll queued built-in target status and attach it once the job completes."""

    queue_state = _get_internal_target_queue_state(user_session_id)
    if queue_state is None:
        return

    control_plane_job_id = queue_state.get("control_plane_job_id")
    corp_path = queue_state.get("corp_path")
    rq_job_id = queue_state.get("rq_job_id")
    if not isinstance(rq_job_id, str) or not rq_job_id:
        if isinstance(control_plane_job_id, int):
            rq_job_id = f"internal-target-{control_plane_job_id}"
            queue_state["rq_job_id"] = rq_job_id
            st.session_state[INTERNAL_TARGET_QUEUE_STATE_KEY] = queue_state
        else:
            rq_job_id = None
    if control_plane_job_id is None or not corp_path:
        _clear_internal_target_queue_state(user_session_id)
        st.warning("Queued target job state was incomplete. Please try again.")
        return

    job_row = registry_service.get_job_by_id(control_plane_job_id)
    if job_row is None:
        _clear_internal_target_queue_state(user_session_id)
        st.warning("Queued target job could not be found. Please try again.")
        return

    if job_row.status == "pending":
        if not isinstance(rq_job_id, str) or not rq_job_id:
            registry_service.mark_job_failed(
                control_plane_job_id,
                "Queued target job was pending without a queue job id.",
            )
            _clear_internal_target_queue_state(user_session_id)
            st.error("Queued target preparation failed before execution. Please retry.")
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
            _clear_internal_target_queue_state(user_session_id)
            st.error("Queued target preparation failed before execution. Please retry.")
            return

        rq_status_raw = rq_job.get_status(refresh=True)
        rq_status = str(getattr(rq_status_raw, "value", rq_status_raw)).strip().lower()
        if rq_status == "failed":
            failure_reason = "Queued target job failed before completion."
            if isinstance(rq_job.exc_info, str) and rq_job.exc_info.strip():
                failure_reason = rq_job.exc_info.strip().splitlines()[-1]
            registry_service.mark_job_failed(control_plane_job_id, failure_reason)
            _clear_internal_target_queue_state(user_session_id)
            st.error(f"Queued target preparation failed: {failure_reason}")
            return

    if job_row.status == "completed":
        _clear_internal_target_queue_state(user_session_id)
        attach_queued_internal_target(
            corp_path,
            user_session_id,
            queue_artifact_id=job_row.artifact_id,
        )
        return

    if job_row.status == "failed":
        _clear_internal_target_queue_state(user_session_id)
        st.error(f"Queued target preparation failed: {job_row.failure_reason}")
        return

    st.info("Processing target corpus in the background...")


# Session state initialization template
STATES = {
    SessionKeys.METADATA_TARGET: {},
    SessionKeys.METADATA_REFERENCE: {},
    SessionKeys.SESSION_DATAFRAME: {},  # Container for SessionKeys DataFrame
    WarningKeys.LOAD_CORPUS: 0,
    LoadCorpusKeys.READY_TO_PROCESS: False,
    LoadCorpusKeys.CORPUS_DF: None,
    LoadCorpusKeys.EXCEPTIONS: None,
    LoadCorpusKeys.MODEL: None,
    LoadCorpusKeys.REF_READY_TO_PROCESS: False,
    LoadCorpusKeys.REF_CORPUS_DF: None,
    LoadCorpusKeys.REF_EXCEPTIONS: None,
}

st.set_page_config(
    page_title=TITLE, page_icon=ICON,
    layout="wide"
    )


# Cache spaCy models for efficiency
@st.cache_resource(show_spinner=False)
def load_models():
    """Load and cache spaCy models for efficient reuse."""
    large_model = spacy.load(MODEL_LARGE)
    small_model = spacy.load(MODEL_SMALL)
    models = {MODEL_LARGE_LABEL: large_model,
              MODEL_SMALL_LABEL: small_model}
    return models


def render_persistence_policy_selector(scope: str, key: str) -> str:
    """Render a persistence selector for user-supplied corpora."""

    choice = st.radio(
        f"How should this {scope} corpus be retained?",
        options=list(PERSISTENCE_POLICY_OPTIONS.keys()),
        index=0,
        key=key,
        help=(
            "Choose whether this corpus should remain available on the server, "
            "stay only in the current session, or be kept locally through the "
            "download flow after processing."
        ),
    )

    policy = PERSISTENCE_POLICY_OPTIONS[choice]
    if policy == CorpusPersistencePolicy.TEMPORARY_SESSION_ONLY:
        st.caption(
            "This corpus will be available during the current session only and "
            "will not be durably persisted by the session backend."
        )
    elif policy == CorpusPersistencePolicy.LOCAL_EXPORT_ONLY:
        st.caption(
            "Process now, then use the existing download flow to keep the corpus "
            "under your own local control."
        )
    else:
        st.caption(
            "This corpus is marked for future server-side reuse once the "
            "user-owned save workflow is wired."
        )

    return policy


def main() -> None:
    """
    Main function for the Load Corpus page.

    Handles corpus loading, processing, and management including:
    - Loading existing target and reference corpora
    - Processing new corpora from text files
    - Uploading external corpora (parquet files)
    - Managing corpus metadata and categories
    - Resetting corpus data
    """
    # Set login requirements for navigaton
    require_login()
    menu()
    st.markdown(f"## {TITLE}")

    # Get or initialize user session
    user_session_id, session = get_or_init_user_session()
    has_target = bool(safe_session_get(session, SessionKeys.HAS_TARGET, False))
    has_reference = bool(safe_session_get(session, SessionKeys.HAS_REFERENCE, False))
    has_meta = bool(safe_session_get(session, SessionKeys.METADATA_TARGET, {}))

    # Initialize processing state if not exists
    if LoadCorpusKeys.READY_TO_PROCESS not in st.session_state[user_session_id]:
        st.session_state[user_session_id][LoadCorpusKeys.READY_TO_PROCESS] = False

    minimal_load_corpus = _load_test_minimal_load_corpus_enabled()

    if minimal_load_corpus and has_target:
        st.markdown("---")
        st.markdown('##### Reference corpus:')
        st.radio(
            "Would you like to load a reference corpus?",
            ("No", "Yes"),
            horizontal=True,
            help=(
                "A reference corpus is a pre-processed corpus "
                "or set of documents that you can use "
                "to compare against your target corpus "
                "with the **Compare Corpora** app."
            )
        )
        st.markdown("---")

    if minimal_load_corpus and not has_target:
        st.markdown("###  :dart: Load or process a target corpus")
        corpus_source = st.radio(
            "What kind of corpus would you like to prepare?",
            CORPUS_SOURCES,
            horizontal=False,
            index=None,
            help="Click Internal to load the built-in target corpus used by load tests."
        )

        if corpus_source == 'Internal':
            st.sidebar.markdown("### Corpora")
            from_model = st.sidebar.radio(
                "Select data tagged with:",
                MODEL_OPTIONS,
                key='corpora_to_load'
            )
            if from_model == 'Large Dictionary':
                saved_corpora = find_saved('ld')
                to_load = st.sidebar.selectbox(
                    'Select a saved corpus to load:',
                    sorted(saved_corpora)
                )
            if from_model == 'Common Dictionary':
                saved_corpora = find_saved('cd')
                to_load = st.sidebar.selectbox(
                    'Select a saved corpus to load:',
                    sorted(saved_corpora)
                )
            process_fn = lambda: process_internal(
                saved_corpora.get(to_load),
                user_session_id,
                CorpusKeys.TARGET
            )
            if get_redis_queue_config().enabled:
                process_fn = lambda: _submit_internal_target_job(
                    saved_corpora.get(to_load),
                    user_session_id,
                )
            sidebar_process_section(
                section_title=LABEL_PROCESS_TARGET,
                button_label=LABEL_PROCESS_TARGET,
                process_fn=process_fn
            )

    st.sidebar.link_button(
        label="Help",
        url="https://browndw.github.io/docuscope-docs/guide/load-corpus.html",
        icon=":material/help:"
        )

    has_target = safe_session_get(session, 'has_target', False) is True
    has_reference = safe_session_get(session, SessionKeys.HAS_REFERENCE, False) is True
    has_meta = safe_session_get(session, SessionKeys.HAS_META, False) is True
    if not has_target and _get_internal_target_queue_state(user_session_id) is not None:
        _render_internal_target_queue_status(user_session_id)

    # If a target corpus is already loaded
    if has_target:
        target_branch_start = perf_counter()
        probe_state = st.session_state.get(PROCESS_TARGET_PROBE_STAGE_KEY)
        if (
            isinstance(probe_state, dict) and
            probe_state.get("session_id") == user_session_id and
            probe_state.get("probe_mode") == "split_ready" and
            probe_state.get("corpus_type") == CorpusKeys.TARGET and
            probe_state.get("stage") == "callback_finished"
        ):
            stage_timings = probe_state.get("stage_timings_ms")
            st.session_state[PROCESS_TARGET_PROBE_STAGE_KEY] = {
                "corpus_type": CorpusKeys.TARGET,
                "stage": "ready_branch",
                "probe_mode": "split_ready",
                "session_id": user_session_id,
                "stage_timings_ms": (
                    stage_timings.copy()
                    if isinstance(stage_timings, dict)
                    else {}
                ),
            }
        st.success(
            "Target corpus loaded and ready.",
            icon=":material/check_circle:"
        )

        metadata_target = None

        # Load and display corpus information
        load_and_display_target_corpus(session, user_session_id)


        if not has_meta:
            if metadata_target is None:
                metadata_target = load_metadata("target", user_session_id)

            target_metadata_start = perf_counter()
            handle_target_metadata_processing(metadata_target, user_session_id)

        # If reference corpus is loaded, show info and warnings
        if not has_reference:
            # Reference corpus not loaded: offer options to load one
            reference_prompt_start = perf_counter()
            st.markdown("---")
            st.markdown('##### Reference corpus:')
            load_ref = st.radio(
                "Would you like to load a reference corpus?",
                ("No", "Yes"),
                horizontal=True,
                help=(
                    "A reference corpus is a pre-processed corpus "
                    "or set of documents that you can use "
                    "to compare against your target corpus "
                    "with the **Compare Corpora** app. "
                    "If you choose to load a reference corpus, "
                    "be considered about the data that you choose. "
                    "What are trying to learn from the comparison?"
                    )
                )

            st.markdown("---")

            if load_ref == 'Yes':
                if metadata_target is None:
                    metadata_start = perf_counter()
                    metadata_target = load_metadata("target", user_session_id)

                # Choose reference corpus source
                ref_corpus_source = st.radio(
                    "What kind of reference corpus would you like to prepare?",
                    CORPUS_SOURCES,
                    captions=[
                        """:material/database:
                        Load a pre-processed corpus from the interface.
                        (Note that only MICUSP and ELSEVIER can be compared.)
                        """,
                        """:material/upload:
                        Upload a pre-processed corpus from your computer.
                        """,
                        """:material/library_books:
                        Process a new corpus from plain text files.
                        """
                        ],
                    horizontal=False,
                    index=None)

                st.markdown("---")

                # Option 1: Load internal reference corpus
                if ref_corpus_source == 'Internal':
                    st.markdown(
                        """
                        :material/priority:
                        Select a saved corpus from the lists in the sidebar.

                        :material/priority: Only corpora tagged with the same model
                        as your target corpus will be available as a reference.
                        """
                        )
                    st.sidebar.markdown("### Reference corpora")
                    st.sidebar.markdown(
                        """Use the button to load
                        a previously processed corpus.
                        """
                        )
                    saved_corpora, saved_ref = find_saved_reference(  # noqa: E501
                        metadata_target.get(LoadCorpusKeys.MODEL)[0],
                        safe_session_get(session, SessionKeys.TARGET_DB, '')
                        )
                    to_load = st.sidebar.selectbox(
                        'Select a saved corpus to load:',
                        (sorted(saved_ref))
                        )
                    sidebar_process_section(
                        section_title=LABEL_PROCESS_REFERENCE,
                        button_label=LABEL_PROCESS_REFERENCE,
                        process_fn=lambda: process_internal(
                                saved_corpora.get(to_load),
                                user_session_id,
                                CorpusKeys.REFERENCE
                                ))

                # Option 2: Upload external reference corpus (parquet)
                if ref_corpus_source == 'External':
                    st.markdown(
                        """
                        :material/priority:
                        Use the widget to select the corpus you'd like to load,
                        either by browsing for them or dragging-and-dropping..

                        :material/priority:
                        Once you've selected your file,
                        click the **UPLOAD REFERENCE** button
                        and a processing button will appear in the sidebar.
                        """
                        )
                    ref_persistence_policy = render_persistence_policy_selector(
                        "reference",
                        "reference_persistence_policy",
                    )

                    with st.form("ref-file-form", clear_on_submit=True):
                        ref_file = st.file_uploader(
                            "Upload your reference corpus",
                            type=["parquet"],
                            accept_multiple_files=False
                            )
                        submitted = st.form_submit_button(
                            LABEL_UPLOAD_REFERENCE
                            )

                    if submitted:
                        st.session_state[user_session_id][WarningKeys.LOAD_CORPUS] = 0

                    # Use the helper function for upload and validation
                    tok_pl, ready = handle_uploaded_parquet(
                        ref_file, CHECK_SIZE, MAX_POLARS,
                        target_docs=metadata_target.get(MetadataKeys.DOCIDS)[0]['ids']
                    )

                    if ready:
                        st.session_state[user_session_id][LoadCorpusKeys.READY_TO_PROCESS] = True  # noqa: E501

                    # Sidebar UI for processing reference corpus
                    if st.session_state[user_session_id][LoadCorpusKeys.READY_TO_PROCESS]:
                        sidebar_process_section(
                            section_title=LABEL_PROCESS_REFERENCE,
                            button_label=LABEL_UPLOAD_REFERENCE,
                            process_fn=lambda: process_external(
                                tok_pl,
                                user_session_id,
                                CorpusKeys.REFERENCE,
                                ref_persistence_policy,
                            ))

                # Option 3: Process new reference corpus from text files
                if ref_corpus_source == 'New':
                    st.markdown(
                        """
                        :material/priority:
                        Use the widget to **select the files**
                        you'd like process, either by browsing for them
                        or dragging-and-dropping.

                        :material/priority:
                        Once you've selected your files, click the **UPLOAD REFERENCE**
                        button and a processing button will appear in the sidebar.

                        :material/priority: Your reference will be tagged with
                        **the same model** as your target corpus.

                        :material/priority: Be sure that all file names are unique
                        and that they don't share names with your target corpus.

                        :material/timer: Processing times may vary, but you can expect
                        the initial corpus processing to take roughly
                        1 minute for every 1 million words.
                        """
                        )
                    ref_persistence_policy = render_persistence_policy_selector(
                        "reference",
                        "reference_persistence_policy",
                    )

                    # Initialize variables with default values
                    corp_df, ready, exceptions = None, False, []

                    with st.form("ref-form", clear_on_submit=True):
                        ref_files = st.file_uploader(
                            "Upload your reference corpus",
                            type=["txt"],
                            accept_multiple_files=True,
                            key='reffiles'
                        )
                        submitted = st.form_submit_button(
                            LABEL_UPLOAD_REFERENCE
                            )

                        if submitted:
                            st.session_state[user_session_id][WarningKeys.LOAD_CORPUS] = 0

                        # Check text files to ensure they are valid
                        # and ready for processing
                        if submitted:  # Only process if form was submitted
                            corp_df, ready, exceptions = handle_uploaded_text(  # noqa: E501
                                ref_files,
                                CHECK_SIZE,
                                MAX_TEXT,
                                check_language_flag=ENABLE_DETECT,
                                check_ref=True,
                                target_docs=metadata_target.get(MetadataKeys.DOCIDS)[0]['ids']  # noqa: E501
                            )

                            # Store the reference corpus dataframe and exceptions
                            if ready and corp_df is not None:
                                st.session_state[user_session_id][LoadCorpusKeys.REF_CORPUS_DF] = corp_df  # noqa: E501
                                st.session_state[user_session_id][LoadCorpusKeys.REF_EXCEPTIONS] = exceptions  # noqa: E501

                    if ready:
                        st.session_state[user_session_id][LoadCorpusKeys.REF_READY_TO_PROCESS] = True  # noqa: E501

                    # Sidebar UI for processing reference corpus
                    if st.session_state[user_session_id][LoadCorpusKeys.REF_READY_TO_PROCESS]:  # noqa: E501
                        # Retrieve stored reference corpus data from session state
                        stored_ref_corp_df = st.session_state[user_session_id].get(LoadCorpusKeys.REF_CORPUS_DF)  # noqa: E501
                        stored_ref_exceptions = st.session_state[user_session_id].get(LoadCorpusKeys.REF_EXCEPTIONS)  # noqa: E501

                        models = load_models()
                        selected_dict = metadata_target.get('model')[0]
                        nlp = models[selected_dict]

                        sidebar_process_section(
                            section_title=LABEL_PROCESS_REFERENCE,
                            button_label=LABEL_PROCESS_REFERENCE,
                            process_fn=lambda: process_new(
                                stored_ref_corp_df,
                                nlp,
                                user_session_id,
                                CorpusKeys.REFERENCE,
                                stored_ref_exceptions,
                                ref_persistence_policy,
                            ))

        # Sidebar: Reset all tools and files
        st.sidebar.markdown(
            body='### Reset all tools and files:'
            )
        st.sidebar.markdown(
            body=(
                ":warning: Using the **reset** button will cause "
                "all files, tables, and plots to be cleared."
            ),
            help=(
                "If you have any unsaved plots or tables "
                "that you'd like to retain, "
                "go back and save them before resetting."
            ))
        if st.sidebar.button(label=LABEL_RESET_CORPUS,
                             icon=":material/refresh:"):
            # Clear session data (original functionality)
            st.session_state[user_session_id] = {}
            # Clear associated widget states
            app_core.session_manager.clear_session_with_widgets(user_session_id)
            generate_temp(
                STATES.items(),
                user_session_id
                )
            app_core.session_manager.create_session(user_session_id)
            st.rerun()
        st.sidebar.markdown("""---""")

    else:
        # No target corpus loaded: show options and info
        st.markdown("###  :dart: Load or process a target corpus")
        st.markdown(
            """
            * From this page you can **load a saved corpus** or **process a new one**
            by selecting the desired (**.txt**) files. You can also reset
            your target corpus or manage any corpora you have saved.

            * Once you have loaded a target corpus, you can add a **reference corpus**
            for comparison. Also note that you can encode metadata into your filenames,
            which can used for further analysis.
            (See the **About new corpora** expander.)
            """
            )

        render_corpus_info_expanders()

        st.markdown("---")
        st.markdown("### Process a corpus:")

        # Choose corpus source
        corpus_source = st.radio(
            "What kind of corpus would you like to prepare?",
            CORPUS_SOURCES,
            captions=[
                """:material/database:
                Load a pre-processed corpus from the interface.
                """,
                """:material/upload:
                Upload a pre-processed corpus from your computer.
                """,
                """:material/library_books:
                Process a new corpus from plain text files.
                """
                ],
            horizontal=False,
            index=None,
            help="Click on the expanders above to learn more about each option."
            )

        # Option 1: Load internal target corpus
        if corpus_source == 'Internal':
            st.markdown("---")
            st.markdown(
                """
                :material/priority:
                Select a saved corpus from the lists in the sidebar.

                :material/priority:  Note that corpora are organized by model
                with which they were tagged.
                """
                )
            st.sidebar.markdown("### Corpora")
            st.sidebar.markdown(
                """
                Use the button to load a previously processed corpus.
                """
                )
            from_model = st.sidebar.radio(
                "Select data tagged with:",
                MODEL_OPTIONS,
                key='corpora_to_load'
                )
            if from_model == 'Large Dictionary':
                saved_corpora = find_saved('ld')
                to_load = st.sidebar.selectbox(
                    'Select a saved corpus to load:',
                    (sorted(saved_corpora))
                    )
            if from_model == 'Common Dictionary':
                saved_corpora = find_saved('cd')
                to_load = st.sidebar.selectbox(
                    'Select a saved corpus to load:',
                    (sorted(saved_corpora))
                    )
            process_fn = lambda: process_internal(
                saved_corpora.get(to_load),
                user_session_id,
                CorpusKeys.TARGET
            )
            if get_redis_queue_config().enabled:
                process_fn = lambda: _submit_internal_target_job(
                    saved_corpora.get(to_load),
                    user_session_id,
                )
            sidebar_process_section(
                section_title=LABEL_PROCESS_TARGET,
                button_label=LABEL_PROCESS_TARGET,
                process_fn=process_fn)

        # Option 2: Upload external target corpus (parquet)
        if corpus_source == 'External':
            st.markdown("---")
            st.markdown(
                """
                :material/priority:
                Use the widget to select the corpus you'd like to load,
                either by browsing for them or dragging-and-dropping..

                :material/priority:
                Once you've selected your file,
                click the **UPLOAD TARGET** button
                and a processing button will appear in the sidebar.
                """
                )
            target_persistence_policy = render_persistence_policy_selector(
                "target",
                "target_persistence_policy",
            )

            with st.form("corpus-file-form", clear_on_submit=True):
                corp_file = st.file_uploader(
                    "Upload your target corpus",
                    type=["parquet"],
                    accept_multiple_files=False
                )
                # Submit button for file upload
                submitted = st.form_submit_button(LABEL_UPLOAD_TARGET)

                if submitted:
                    st.session_state[user_session_id][WarningKeys.LOAD_CORPUS] = 0

                # Use the helper function for upload and validation
                tok_pl, ready = handle_uploaded_parquet(
                    corp_file, CHECK_SIZE, MAX_POLARS
                )

            if ready:
                st.session_state[user_session_id][LoadCorpusKeys.READY_TO_PROCESS] = True

            # Sidebar UI for processing target corpus
            if st.session_state[user_session_id][LoadCorpusKeys.READY_TO_PROCESS]:
                sidebar_process_section(
                    section_title=LABEL_PROCESS_TARGET,
                    button_label=LABEL_PROCESS_TARGET,
                    process_fn=lambda: process_external(
                        tok_pl,
                        user_session_id,
                        CorpusKeys.TARGET,
                        target_persistence_policy,
                    ))

        # Option 3: Process new target corpus from text files
        if corpus_source == 'New':
            st.markdown("---")
            st.markdown(
                """
                :material/priority:
                Use the widget to **select the files**
                you'd like process, either by browsing for them
                or dragging-and-dropping.

                :material/priority:
                Once you've selected your files, click the **UPLOAD** button
                and a processing button will appear in the sidebar.

                :material/priority:
                Select **a model** from the sidebar.

                :material/priority:
                After processing, you will have the option
                to save your corpus to use for future analysis.

                :material/priority:
                Be sure that all file names are unique.

                :material/timer:
                Processing times may vary, but you can expect
                the initial corpus processing to take roughly
                1 minute for every 1 million words.
                """
                )
            target_persistence_policy = render_persistence_policy_selector(
                "target",
                "target_persistence_policy",
            )

            # Initialize variables with default values
            corp_df, ready, exceptions = None, False, []

            with st.form("corpus-form", clear_on_submit=True):
                corp_files = st.file_uploader(
                    "Upload your target corpus",
                    type=["txt"],
                    accept_multiple_files=True
                )
                submitted = st.form_submit_button(LABEL_UPLOAD_TARGET)

                if submitted:
                    st.session_state[user_session_id][WarningKeys.LOAD_CORPUS] = 0
                if submitted and not corp_files:
                    st.warning(
                        "Please select at least one file to upload.",
                        icon=":material/warning:")

                # Check text files to ensure they are valid
                # and ready for processing
                if submitted:  # Only process if form was submitted
                    corp_df, ready, exceptions = handle_uploaded_text(  # noqa: E501
                        corp_files,
                        CHECK_SIZE,
                        MAX_TEXT,
                        check_language_flag=ENABLE_DETECT
                    )

                    # Store the corpus dataframe and exceptions in session state
                    if ready and corp_df is not None:
                        st.session_state[user_session_id][LoadCorpusKeys.CORPUS_DF] = corp_df  # noqa: E501
                        st.session_state[user_session_id][LoadCorpusKeys.EXCEPTIONS] = exceptions  # noqa: E501

            if ready:
                st.session_state[user_session_id][LoadCorpusKeys.READY_TO_PROCESS] = True

            # Sidebar UI for model selection and processing
            st.sidebar.markdown("### Models")
            models = load_models()
            selected_dict = st.sidebar.selectbox(
                "Select a DocuScope model:",
                options=MODEL_OPTIONS,
                help="The Large Dictionary model has a more eleaborated tagset than the Common Dictionary model. Click 'About the models' (on the right) to learn more.",  # noqa: E501
                )
            nlp = models[selected_dict]
            st.session_state[user_session_id][LoadCorpusKeys.MODEL] = selected_dict

            st.sidebar.markdown("---")

            if st.session_state[user_session_id][LoadCorpusKeys.READY_TO_PROCESS]:
                # Retrieve stored corpus data from session state
                stored_corp_df = st.session_state[user_session_id].get(LoadCorpusKeys.CORPUS_DF)  # noqa: E501
                stored_exceptions = st.session_state[user_session_id].get(LoadCorpusKeys.EXCEPTIONS)  # noqa: E501

                sidebar_process_section(
                    section_title=LABEL_PROCESS_TARGET,
                    button_label=LABEL_PROCESS_TARGET,
                    process_fn=lambda: process_new(
                        stored_corp_df,
                        nlp,
                        user_session_id,
                        CorpusKeys.TARGET,
                        stored_exceptions,
                        target_persistence_policy,
                    ))


if __name__ == "__main__":
    main()
